"""OTE Entry Refinement — wait for optimal pullback before entering.

Instead of entering at the stale 1H candle close, signals are queued and
monitored on 5-minute candles for a better entry price.

Sweep Strategy (OTE Zone):
  After sweep + displacement on 1H, price thrusts in the intended direction.
  We calculate the OTE (Optimal Trade Entry) zone — the 50-79% Fibonacci
  retracement of the thrust move from sweep_level → thrust_extreme.
  Enter only when price pulls back INTO that zone and shows rejection
  (wick, engulfing, or volume confirmation).

Breakout Strategy (Level Retest):
  After a breakout is confirmed (price broke + held beyond a level), we
  wait for price to come back and RETEST the breakout level.  Old resistance
  becomes new support (or vice versa).  Enter on the bounce.

If price never pulls back within the expiry window, we SKIP the trade.
The best setups always give you a pullback — V-shaped moves that never
retrace are typically traps or liquidation cascades.

Flow:
  1H scan detects signal (score >= threshold) → queue in EntryRefiner
  Every 60s: check 5m candles for OTE zone entry / breakout retest
  Pullback confirmed → return refined signal with better entry price
  Expired (30 min) → SKIP trade (no chase) unless ote_skip_on_expiry=False
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.config import Settings
from src.data.candles import CandleManager
from src.exchange.models import SignalCandidate
from src.strategy.volume import VolumeAnalyzer
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────
# Minimum RVOL on the rejection candle to confirm institutional interest
OTE_REJECTION_RVOL = 1.2

# Minimum wick ratio (wick / body) for rejection wick confirmation
REJECTION_WICK_RATIO = 0.5

# Minimum thrust size (% from sweep level) before OTE zone is valid
# Prevents calculating an OTE zone from noise
MIN_THRUST_PCT = 0.004  # 0.4%
MIN_THRUST_PCT_1H = 0.008  # 0.8% — higher threshold for 1H scale

# Breakout retest tolerance — how close price must get to the breakout level
BREAKOUT_RETEST_TOLERANCE = 0.003  # 0.3%


@dataclass
class RefinerEntry:
    """A signal queued for 5m entry refinement."""

    symbol: str
    signal: SignalCandidate
    added_at: datetime
    expires_at: datetime
    original_1h_price: float
    check_count: int = 0

    # Entry type
    entry_type: str = "sweep"  # "sweep" or "breakout"

    # Sweep-specific
    sweep_level: float = 0.0
    sweep_direction: str = ""  # "bullish" (swept lows) or "bearish" (swept highs)

    # Breakout-specific
    breakout_level: float = 0.0
    breakout_direction: str = ""  # "bullish" or "bearish"

    # OTE tracking (dynamically updated from 5m candles)
    thrust_high: float = 0.0
    thrust_low: float = 1e18  # Start very high so any real low replaces it
    ote_zone_top: float = 0.0
    ote_zone_bottom: float = 0.0
    ote_zone_valid: bool = False
    zone_source: str = ""  # "agent", "1h_thrust", "5m_thrust" — for logging/dashboard


class EntryRefiner:
    """Monitors queued signals on 5m candles for better entry timing.

    Called from the engine's monitor loop on each 60-second tick.
    Handles both sweep and breakout signals with strategy-specific
    pullback detection.
    """

    def __init__(
        self,
        candle_manager: CandleManager,
        config: Settings,
    ) -> None:
        self.candles = candle_manager
        self.config = config
        self.queue: dict[str, RefinerEntry] = {}
        self.vol_analyzer = VolumeAnalyzer()

        # OTE zone parameters
        self.ote_min_retrace = getattr(config, "ote_min_retracement", 0.50)
        self.ote_max_retrace = getattr(config, "ote_max_retracement", 0.79)
        self.skip_on_expiry = getattr(config, "ote_skip_on_expiry", True)

        # Heartbeat tracking — lets the dashboard confirm the loop is alive
        self.last_check_at: datetime | None = None
        self.total_checks: int = 0
        self.total_confirmed: int = 0
        self.total_expired: int = 0

    # ── OTE Zone Computation ─────────────────────────────────

    def _compute_ote_from_thrust(
        self,
        thrust_extreme: float,
        origin: float,
        direction: str,
    ) -> tuple[float, float]:
        """Compute OTE zone boundaries from a thrust move.

        Returns (ote_top, ote_bottom) based on Fibonacci retracement
        of the thrust from origin → thrust_extreme.
        """
        if direction == "bullish":
            move_size = thrust_extreme - origin
            ote_top = thrust_extreme - move_size * self.ote_min_retrace
            ote_bottom = thrust_extreme - move_size * self.ote_max_retrace
        else:
            move_size = origin - thrust_extreme
            ote_bottom = thrust_extreme + move_size * self.ote_min_retrace
            ote_top = thrust_extreme + move_size * self.ote_max_retrace
        return ote_top, ote_bottom

    # ── Public API ──────────────────────────────────────────────

    def add(self, signal: SignalCandidate) -> bool:
        """Queue a sweep signal for OTE zone entry refinement.

        Returns True if added, False if duplicate.
        """
        if signal.symbol in self.queue:
            return False

        sweep_result = signal.sweep_result
        if not sweep_result or not sweep_result.sweep_detected:
            return False

        now = datetime.now(timezone.utc)
        entry = RefinerEntry(
            symbol=signal.symbol,
            signal=signal,
            added_at=now,
            expires_at=now + timedelta(minutes=self.config.entry_refiner_expiry_minutes),
            original_1h_price=signal.entry_price,
            entry_type="sweep",
            sweep_level=sweep_result.sweep_level,
            sweep_direction=sweep_result.sweep_direction,
        )
        self.queue[signal.symbol] = entry

        logger.info(
            "refiner_queued_sweep",
            symbol=signal.symbol,
            score=signal.score,
            sweep_level=round(sweep_result.sweep_level, 6),
            sweep_direction=sweep_result.sweep_direction,
            sweep_type=sweep_result.sweep_type,
            original_1h_price=round(signal.entry_price, 6),
            ote_zone=f"{self.ote_min_retrace:.0%}-{self.ote_max_retrace:.0%}",
            expires_at=entry.expires_at.isoformat(),
            queue_size=len(self.queue),
        )
        return True

    def add_breakout(self, signal: SignalCandidate) -> bool:
        """Queue a breakout signal for level-retest entry refinement.

        Returns True if added, False if duplicate.
        """
        if signal.symbol in self.queue:
            return False

        breakout_result = signal.breakout_result
        if not breakout_result or not breakout_result.breakout_detected:
            return False

        now = datetime.now(timezone.utc)
        entry = RefinerEntry(
            symbol=signal.symbol,
            signal=signal,
            added_at=now,
            expires_at=now + timedelta(minutes=self.config.entry_refiner_expiry_minutes),
            original_1h_price=signal.entry_price,
            entry_type="breakout",
            breakout_level=breakout_result.breakout_level,
            breakout_direction=breakout_result.breakout_direction or "",
        )
        self.queue[signal.symbol] = entry

        logger.info(
            "refiner_queued_breakout",
            symbol=signal.symbol,
            score=signal.score,
            breakout_level=round(breakout_result.breakout_level, 6),
            breakout_direction=breakout_result.breakout_direction,
            breakout_type=breakout_result.breakout_type,
            original_1h_price=round(signal.entry_price, 6),
            expires_at=entry.expires_at.isoformat(),
            queue_size=len(self.queue),
        )
        return True

    async def check_all(
        self, open_position_symbols: set[str] | None = None,
    ) -> list[SignalCandidate]:
        """Check all queued entries on 5m data.

        Returns a list of signals ready to enter (refined with better price).
        Removes completed and expired entries from the queue.
        Evicts entries whose symbol already has an open position.
        """
        # Always update heartbeat — even if queue is empty
        self.last_check_at = datetime.now(timezone.utc)
        self.total_checks += 1

        if not self.queue:
            return []

        # Evict entries for symbols that now have open positions
        if open_position_symbols:
            for sym in list(self.queue.keys()):
                if sym in open_position_symbols:
                    logger.info(
                        "refiner_evicted_open_position",
                        symbol=sym,
                        check_count=self.queue[sym].check_count,
                    )
                    del self.queue[sym]

        ready: list[SignalCandidate] = []
        now = self.last_check_at

        for symbol, entry in list(self.queue.items()):
            try:
                # Check expiry first
                if now >= entry.expires_at:
                    expired_signal = self._handle_expiry(entry)
                    if expired_signal:
                        ready.append(expired_signal)
                    self.total_expired += 1
                    del self.queue[symbol]
                    continue

                # Fetch 5m candles and check for pullback
                candles_5m = await self.candles.get_candles(symbol, "5m", limit=30)
                if candles_5m is None or candles_5m.empty or len(candles_5m) < 10:
                    continue

                entry.check_count += 1

                # Dispatch to strategy-specific check
                if entry.entry_type == "sweep":
                    result = self._check_sweep_ote(entry, candles_5m)
                else:
                    result = self._check_breakout_retest(entry, candles_5m)

                if result is not None:
                    ready.append(result)
                    self.total_confirmed += 1
                    del self.queue[symbol]

            except Exception as e:
                logger.warning(
                    "refiner_check_failed",
                    symbol=symbol,
                    error=str(e)[:100],
                )

        return ready

    def get_queued_symbols(self) -> set[str]:
        """Return symbols currently being refined."""
        return set(self.queue.keys())

    # ── Sweep OTE Zone Check ──────────────────────────────────────

    def _check_sweep_ote(
        self, entry: RefinerEntry, candles_5m: pd.DataFrame
    ) -> SignalCandidate | None:
        """Check if price has pulled back into the OTE zone with rejection.

        Three-tier zone resolution (priority order):
          Tier 1 — Agent zone: AI-provided entry_zone_high/low (1H/4H/1D analysis)
          Tier 2 — 1H thrust: OTE from scanner's 1H thrust_extreme + displacement_open
          Tier 3 — 5m thrust: Original logic, OTE from 5m candle thrust (fallback)

        5m candles are always used for ENTRY TIMING (rejection candle detection)
        regardless of which tier sets the zone boundaries.
        """
        direction = entry.sweep_direction
        sweep_level = entry.sweep_level
        signal = entry.signal

        # ── Resolve zone boundaries (three-tier priority) ──

        agent_high = getattr(signal, "agent_entry_zone_high", None)
        agent_low = getattr(signal, "agent_entry_zone_low", None)
        thrust_1h = getattr(signal, "thrust_extreme_1h", None)
        disp_open_1h = getattr(signal, "displacement_open_1h", None)

        ote_top = 0.0
        ote_bottom = 0.0
        zone_resolved = False

        # ── Tier 1: Agent zone (highest priority) ──
        if agent_high is not None and agent_low is not None and agent_high > agent_low:
            ote_top = agent_high
            ote_bottom = agent_low
            entry.zone_source = "agent"
            zone_resolved = True
            if entry.check_count == 0:
                logger.info(
                    "refiner_zone_from_agent",
                    symbol=entry.symbol,
                    direction=direction,
                    agent_zone=f"{round(agent_low, 6)}-{round(agent_high, 6)}",
                )

        # ── Tier 2: 1H thrust ──
        if not zone_resolved and thrust_1h is not None and disp_open_1h is not None:
            if direction == "bullish":
                move = thrust_1h - disp_open_1h
                if move > 0 and disp_open_1h > 0 and (move / disp_open_1h) >= MIN_THRUST_PCT_1H:
                    ote_top, ote_bottom = self._compute_ote_from_thrust(
                        thrust_1h, disp_open_1h, "bullish"
                    )
                    entry.zone_source = "1h_thrust"
                    zone_resolved = True
                    if entry.check_count == 0:
                        logger.info(
                            "refiner_zone_from_1h_thrust",
                            symbol=entry.symbol,
                            direction=direction,
                            thrust_extreme=round(thrust_1h, 6),
                            displacement_open=round(disp_open_1h, 6),
                            ote_zone=f"{round(ote_bottom, 6)}-{round(ote_top, 6)}",
                        )
            else:  # bearish
                move = disp_open_1h - thrust_1h
                if move > 0 and disp_open_1h > 0 and (move / disp_open_1h) >= MIN_THRUST_PCT_1H:
                    ote_top, ote_bottom = self._compute_ote_from_thrust(
                        thrust_1h, disp_open_1h, "bearish"
                    )
                    entry.zone_source = "1h_thrust"
                    zone_resolved = True
                    if entry.check_count == 0:
                        logger.info(
                            "refiner_zone_from_1h_thrust",
                            symbol=entry.symbol,
                            direction=direction,
                            thrust_extreme=round(thrust_1h, 6),
                            displacement_open=round(disp_open_1h, 6),
                            ote_zone=f"{round(ote_bottom, 6)}-{round(ote_top, 6)}",
                        )

        # ── Tier 3: 5m thrust (fallback) ──
        if not zone_resolved:
            entry.zone_source = "5m_thrust"
            highs = candles_5m["high"].astype(float)
            lows = candles_5m["low"].astype(float)

            # Update 5m thrust tracking
            current_max_high = float(highs.max())
            current_min_low = float(lows.min())
            if current_max_high > entry.thrust_high:
                entry.thrust_high = current_max_high
            if current_min_low < entry.thrust_low:
                entry.thrust_low = current_min_low

            if direction == "bullish":
                thrust = entry.thrust_high
                move_size = thrust - sweep_level
                if move_size <= 0 or (sweep_level > 0 and (move_size / sweep_level) < MIN_THRUST_PCT):
                    if entry.check_count % 5 == 0:
                        logger.debug(
                            "refiner_waiting_thrust",
                            symbol=entry.symbol,
                            direction=direction,
                            thrust_high=round(thrust, 6),
                            sweep_level=round(sweep_level, 6),
                            move_pct=f"{(move_size / sweep_level * 100) if sweep_level > 0 else 0:.2f}%",
                        )
                    return None
                ote_top = thrust - move_size * self.ote_min_retrace
                ote_bottom = thrust - move_size * self.ote_max_retrace
            else:  # bearish
                thrust = entry.thrust_low
                move_size = sweep_level - thrust
                if move_size <= 0 or (sweep_level > 0 and (move_size / sweep_level) < MIN_THRUST_PCT):
                    if entry.check_count % 5 == 0:
                        logger.debug(
                            "refiner_waiting_thrust",
                            symbol=entry.symbol,
                            direction=direction,
                            thrust_low=round(thrust, 6),
                            sweep_level=round(sweep_level, 6),
                            move_pct=f"{(move_size / sweep_level * 100) if sweep_level > 0 else 0:.2f}%",
                        )
                    return None
                ote_bottom = thrust + move_size * self.ote_min_retrace
                ote_top = thrust + move_size * self.ote_max_retrace

            if entry.check_count == 0:
                logger.info(
                    "refiner_zone_from_5m_thrust",
                    symbol=entry.symbol,
                    direction=direction,
                    ote_zone=f"{round(ote_bottom, 6)}-{round(ote_top, 6)}",
                )

        # ── Store zone and check for rejection ──
        entry.ote_zone_valid = True
        entry.ote_zone_top = ote_top
        entry.ote_zone_bottom = ote_bottom

        return self._find_rejection_in_zone(
            entry=entry,
            candles_5m=candles_5m,
            zone_top=ote_top,
            zone_bottom=ote_bottom,
            direction=direction,
        )

    # ── Breakout Retest Check ─────────────────────────────────────

    def _check_breakout_retest(
        self, entry: RefinerEntry, candles_5m: pd.DataFrame
    ) -> SignalCandidate | None:
        """Check if price has retested the breakout level and bounced.

        For bullish breakout (broke above resistance):
          - Old resistance = new support
          - Wait for price to pull back TO the breakout level
          - Enter when price touches the level and closes above it (bounce)

        For bearish breakout (broke below support):
          - Old support = new resistance
          - Wait for price to pull back UP to the breakout level
          - Enter when price touches the level and closes below it (rejection)
        """
        direction = entry.breakout_direction
        level = entry.breakout_level
        tolerance = level * BREAKOUT_RETEST_TOLERANCE

        closes = candles_5m["close"].astype(float)
        highs = candles_5m["high"].astype(float)
        lows = candles_5m["low"].astype(float)
        opens = candles_5m["open"].astype(float)

        # Retest zone around the breakout level
        zone_top = level + tolerance
        zone_bottom = level - tolerance

        # Check last 5 candles for retest + bounce
        recent = candles_5m.tail(5)

        for i in range(len(recent)):
            candle = recent.iloc[i]
            c_close = float(candle["close"])
            c_open = float(candle["open"])
            c_high = float(candle["high"])
            c_low = float(candle["low"])
            body = abs(c_close - c_open)

            if direction == "bullish":
                # Broke above → wait for pullback to level → bounce
                # Candle low touched the retest zone AND closed above it
                if c_low <= zone_top and c_close > level:
                    # Rejection wick: long lower wick showing buyers at the level
                    lower_wick = min(c_open, c_close) - c_low
                    has_wick = body > 0 and lower_wick / body >= REJECTION_WICK_RATIO

                    # Volume check
                    vol_profile = self.vol_analyzer.analyze(candles_5m)
                    rvol = vol_profile.relative_volume
                    has_volume = rvol >= OTE_REJECTION_RVOL

                    if has_wick or has_volume:
                        # Target = the breakout level (where support is)
                        target_entry = min(c_close, zone_top)
                        return self._create_refined_signal(
                            entry=entry,
                            entry_price=target_entry,
                            rejection_type="breakout_retest",
                            has_wick=has_wick,
                            rvol=rvol,
                        )

            elif direction == "bearish":
                # Broke below → wait for pullback up to level → rejection
                # Candle high reached the retest zone AND closed below it
                if c_high >= zone_bottom and c_close < level:
                    upper_wick = c_high - max(c_open, c_close)
                    has_wick = body > 0 and upper_wick / body >= REJECTION_WICK_RATIO

                    vol_profile = self.vol_analyzer.analyze(candles_5m)
                    rvol = vol_profile.relative_volume
                    has_volume = rvol >= OTE_REJECTION_RVOL

                    if has_wick or has_volume:
                        # Target = the breakout level (where resistance is)
                        target_entry = max(c_close, zone_bottom)
                        return self._create_refined_signal(
                            entry=entry,
                            entry_price=target_entry,
                            rejection_type="breakout_retest",
                            has_wick=has_wick,
                            rvol=rvol,
                        )

        # No retest detected yet
        if entry.check_count % 5 == 0:
            current_price = float(closes.iloc[-1])
            logger.debug(
                "refiner_waiting_retest",
                symbol=entry.symbol,
                direction=direction,
                breakout_level=round(level, 6),
                current_price=round(current_price, 6),
                distance_pct=f"{abs(current_price - level) / level * 100:.2f}%",
                check_count=entry.check_count,
            )
        return None

    # ── OTE Zone Rejection Detection ──────────────────────────────

    def _find_rejection_in_zone(
        self,
        entry: RefinerEntry,
        candles_5m: pd.DataFrame,
        zone_top: float,
        zone_bottom: float,
        direction: str,
    ) -> SignalCandidate | None:
        """Look for a rejection candle within the OTE zone.

        For bullish: price pulled back down into zone, look for buying rejection
          (lower wick, bullish close, volume)
        For bearish: price pulled back up into zone, look for selling rejection
          (upper wick, bearish close, volume)

        Only checks the last 2 candles (most recent 10 minutes) to avoid entering
        based on stale data where price has already bounced away from the zone.
        """
        recent = candles_5m.tail(2)

        for i in range(len(recent)):
            candle = recent.iloc[i]
            c_close = float(candle["close"])
            c_open = float(candle["open"])
            c_high = float(candle["high"])
            c_low = float(candle["low"])
            body = abs(c_close - c_open)

            if direction == "bullish":
                # Price pulled back DOWN into zone
                # Candle low entered the OTE zone (below zone_top)
                if c_low <= zone_top and c_low >= zone_bottom * 0.995:
                    # Rejection: closed above zone midpoint (buyers stepped in)
                    zone_mid = (zone_top + zone_bottom) / 2
                    closed_above_mid = c_close >= zone_mid

                    # Rejection wick: long lower wick
                    lower_wick = min(c_open, c_close) - c_low
                    has_wick = body > 0 and lower_wick / body >= REJECTION_WICK_RATIO

                    # Bullish candle (close > open)
                    is_bullish_candle = c_close > c_open

                    # Volume
                    vol_profile = self.vol_analyzer.analyze(candles_5m)
                    rvol = vol_profile.relative_volume
                    has_volume = rvol >= OTE_REJECTION_RVOL

                    # Need at least 2 of: wick rejection, bullish close above mid, volume
                    confirmations = sum([
                        has_wick,
                        closed_above_mid and is_bullish_candle,
                        has_volume,
                    ])

                    if confirmations >= 2:
                        # Target entry = zone top (best we can get on the pullback).
                        # If close is inside the zone, use the close (even better).
                        # The drift check in core.py will only execute when live
                        # price is near this target, preventing entries far from zone.
                        target_entry = min(c_close, zone_top)
                        return self._create_refined_signal(
                            entry=entry,
                            entry_price=target_entry,
                            rejection_type="ote_zone",
                            has_wick=has_wick,
                            rvol=rvol,
                            zone_top=zone_top,
                            zone_bottom=zone_bottom,
                        )

            elif direction == "bearish":
                # Price pulled back UP into zone
                # Candle high entered the OTE zone (above zone_bottom)
                if c_high >= zone_bottom and c_high <= zone_top * 1.005:
                    zone_mid = (zone_top + zone_bottom) / 2
                    closed_below_mid = c_close <= zone_mid

                    upper_wick = c_high - max(c_open, c_close)
                    has_wick = body > 0 and upper_wick / body >= REJECTION_WICK_RATIO

                    is_bearish_candle = c_close < c_open

                    vol_profile = self.vol_analyzer.analyze(candles_5m)
                    rvol = vol_profile.relative_volume
                    has_volume = rvol >= OTE_REJECTION_RVOL

                    confirmations = sum([
                        has_wick,
                        closed_below_mid and is_bearish_candle,
                        has_volume,
                    ])

                    if confirmations >= 2:
                        # Target entry = zone bottom (best short entry on pullback).
                        # If close is inside the zone, use the close (even better).
                        target_entry = max(c_close, zone_bottom)
                        return self._create_refined_signal(
                            entry=entry,
                            entry_price=target_entry,
                            rejection_type="ote_zone",
                            has_wick=has_wick,
                            rvol=rvol,
                            zone_top=zone_top,
                            zone_bottom=zone_bottom,
                        )

        # No rejection in zone yet — log periodic status
        if entry.check_count % 5 == 0:
            current_price = float(candles_5m["close"].iloc[-1])
            logger.debug(
                "refiner_waiting_ote",
                symbol=entry.symbol,
                direction=direction,
                ote_zone=f"{round(zone_bottom, 4)}-{round(zone_top, 4)}",
                current_price=round(current_price, 6),
                check_count=entry.check_count,
            )
        return None

    # ── Signal Creation ───────────────────────────────────────────

    def _create_refined_signal(
        self,
        entry: RefinerEntry,
        entry_price: float,
        rejection_type: str,
        has_wick: bool,
        rvol: float,
        zone_top: float = 0.0,
        zone_bottom: float = 0.0,
    ) -> SignalCandidate:
        """Create a refined signal with improved entry price."""
        now = datetime.now(timezone.utc)
        duration = (now - entry.added_at).total_seconds()

        # Calculate improvement
        improvement_pct = 0.0
        if entry.original_1h_price > 0:
            improvement_pct = abs(
                entry_price - entry.original_1h_price
            ) / entry.original_1h_price * 100

        signal = entry.signal
        signal.entry_price = entry_price
        signal.refined_entry = True
        signal.refinement_duration_seconds = duration
        signal.original_1h_price = entry.original_1h_price

        logger.info(
            "refiner_confirmed",
            symbol=entry.symbol,
            entry_type=entry.entry_type,
            rejection_type=rejection_type,
            entry_price=round(entry_price, 6),
            original_1h_price=round(entry.original_1h_price, 6),
            improvement_pct=round(improvement_pct, 2),
            rvol=round(rvol, 2),
            rejection_wick=has_wick,
            ote_zone=f"{round(zone_bottom, 4)}-{round(zone_top, 4)}" if zone_top > 0 else "n/a",
            zone_source=entry.zone_source or "n/a",
            check_count=entry.check_count,
            duration_seconds=round(duration, 0),
            direction=entry.sweep_direction or entry.breakout_direction,
        )

        return signal

    def _handle_expiry(self, entry: RefinerEntry) -> SignalCandidate | None:
        """Handle expired entries — return for agent re-evaluation.

        Instead of silently dropping the signal, return it with a flag so
        the engine can re-feed it to the agent with fresh market context.
        The agent gets one more look and can ENTER_NOW, WAIT_PULLBACK
        (re-queue with new zone), or SKIP (done for real).
        """
        now = datetime.now(timezone.utc)
        duration = (now - entry.added_at).total_seconds()

        signal = entry.signal
        signal.refined_entry = False
        signal.refinement_duration_seconds = duration
        signal.original_1h_price = entry.original_1h_price

        # Mark for re-evaluation (only once — prevent infinite loops)
        already_reviewed = getattr(signal, "_expiry_reviewed", False)
        if already_reviewed:
            logger.info(
                "refiner_expired_final_skip",
                symbol=entry.symbol,
                entry_type=entry.entry_type,
                check_count=entry.check_count,
                original_1h_price=round(entry.original_1h_price, 6),
                duration_seconds=round(duration, 0),
                reason="already_reviewed_on_expiry",
            )
            return None

        signal._expiry_reviewed = True
        signal._expired_from_refiner = True

        logger.info(
            "refiner_expired_for_review",
            symbol=entry.symbol,
            entry_type=entry.entry_type,
            check_count=entry.check_count,
            original_1h_price=round(entry.original_1h_price, 6),
            duration_seconds=round(duration, 0),
            ote_zone_valid=entry.ote_zone_valid,
            ote_zone=f"{round(entry.ote_zone_bottom, 4)}-{round(entry.ote_zone_top, 4)}"
            if entry.ote_zone_valid else "n/a",
        )
        return signal

    # ── State Persistence ──────────────────────────────────────────

    def get_state(self) -> dict:
        """Serialize refiner state for DB persistence."""
        entries_data = {}
        for sym, entry in self.queue.items():
            entries_data[sym] = {
                "symbol": entry.symbol,
                "added_at": entry.added_at.isoformat(),
                "expires_at": entry.expires_at.isoformat(),
                "original_1h_price": entry.original_1h_price,
                "check_count": entry.check_count,
                "score": entry.signal.score,
                "direction": entry.signal.direction,
                "components": entry.signal.components,
                "entry_type": entry.entry_type,
                "sweep_level": entry.sweep_level,
                "sweep_direction": entry.sweep_direction,
                "breakout_level": entry.breakout_level,
                "breakout_direction": entry.breakout_direction,
                "thrust_high": entry.thrust_high,
                "thrust_low": entry.thrust_low if entry.thrust_low < 1e17 else 0,
                "agent_zone_high": entry.signal.agent_entry_zone_high,
                "agent_zone_low": entry.signal.agent_entry_zone_low,
                "zone_source": entry.zone_source,
            }
        return {
            "entries": entries_data,
            "total_queued": len(entries_data),
            "last_check_at": self.last_check_at.isoformat() if self.last_check_at else None,
            "total_checks": self.total_checks,
            "total_confirmed": self.total_confirmed,
            "total_expired": self.total_expired,
        }

    def restore_state(self, data: dict) -> None:
        """Restore refiner state from DB.

        Restored entries will expire naturally (30 min window is short enough
        that stale entries are unlikely to survive a restart).
        """
        if not data or "entries" not in data:
            return

        now = datetime.now(timezone.utc)
        restored = 0
        expired = 0

        for sym, entry_data in data.get("entries", {}).items():
            try:
                expires_at = datetime.fromisoformat(entry_data["expires_at"])
                if expires_at <= now:
                    expired += 1
                    continue

                placeholder = SignalCandidate(
                    score=entry_data.get("score", 0),
                    direction=entry_data.get("direction"),
                    symbol=sym,
                    entry_price=entry_data.get("original_1h_price", 0),
                    components=entry_data.get("components", {}),
                )

                entry = RefinerEntry(
                    symbol=sym,
                    signal=placeholder,
                    added_at=datetime.fromisoformat(entry_data["added_at"]),
                    expires_at=expires_at,
                    original_1h_price=entry_data.get("original_1h_price", 0),
                    check_count=entry_data.get("check_count", 0),
                    entry_type=entry_data.get("entry_type", "sweep"),
                    sweep_level=entry_data.get("sweep_level", 0),
                    sweep_direction=entry_data.get("sweep_direction", "bullish"),
                    breakout_level=entry_data.get("breakout_level", 0),
                    breakout_direction=entry_data.get("breakout_direction", ""),
                    thrust_high=entry_data.get("thrust_high", 0),
                    thrust_low=entry_data.get("thrust_low", 1e18),
                )
                self.queue[sym] = entry
                restored += 1
            except Exception as e:
                logger.warning(
                    "refiner_restore_failed",
                    symbol=sym,
                    error=str(e),
                )

        if restored > 0 or expired > 0:
            logger.info(
                "refiner_restored",
                restored=restored,
                expired_on_restore=expired,
            )

    def stop(self) -> None:
        """Clear the queue (for reset)."""
        self.queue.clear()
