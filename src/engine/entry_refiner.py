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

import time as _time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.config import Settings
from src.data.candles import CandleManager
from src.exchange.models import PullbackPlan, SignalCandidate
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
    zone_source: str = ""  # "agent", "1h_thrust", "5m_thrust", "agent2_adjusted"

    # PullbackPlan — formal pending order contract for WAIT_PULLBACK entries
    pullback_plan: PullbackPlan | None = None

    # Agent 2 (Refiner Monitor) reasoning continuity
    last_agent2_action: str = ""       # Previous Agent 2 decision (WAIT, ADJUST_ZONE, etc.)
    last_agent2_reasoning: str = ""    # Previous Agent 2 reasoning (fed back into next prompt)
    last_agent2_urgency: str = ""      # Previous urgency level
    agent2_check_count: int = 0        # How many times Agent 2 has evaluated this signal

    # ENTER_NOW routing: Agent 1 said enter immediately, but route through
    # Agent 2 for quick confirmation. Agent 2 called on first tick (no 5m wait).
    enter_now: bool = False


class EntryRefiner:
    """Monitors queued signals on 5m candles for better entry timing.

    Called from the engine's monitor loop on each 60-second tick.
    Handles both sweep and breakout signals with strategy-specific
    pullback detection.

    When refiner_agent (Agent 2) is provided, it replaces the algorithmic
    rejection detection every 5 minutes. Between Agent 2 checks, the
    algorithmic fallback still runs on each 60-second tick.
    """

    def __init__(
        self,
        candle_manager: CandleManager,
        config: Settings,
        refiner_agent=None,
        market_filter=None,
        exchange=None,
    ) -> None:
        self.candles = candle_manager
        self.config = config
        self.queue: dict[str, RefinerEntry] = {}
        self.vol_analyzer = VolumeAnalyzer()

        # Agent 2 (Refiner Monitor) — AI-powered tactical entry timing
        self.refiner_agent = refiner_agent
        self.market_filter = market_filter
        self._exchange = exchange  # For order book data (Feature 2)
        self._last_agent_check: dict[str, float] = {}  # symbol → timestamp
        self._agent_check_interval = getattr(
            config, "refiner_agent_check_interval_minutes", 5.0
        ) * 60  # Convert to seconds

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
        plan = getattr(signal, "pullback_plan", None)
        # Use plan expiry if present (30min default), else refiner default (4h)
        expiry = plan.expires_at if plan else now + timedelta(minutes=self.config.entry_refiner_expiry_minutes)
        entry = RefinerEntry(
            symbol=signal.symbol,
            signal=signal,
            added_at=now,
            expires_at=expiry,
            original_1h_price=signal.entry_price,
            entry_type="sweep",
            sweep_level=sweep_result.sweep_level,
            sweep_direction=sweep_result.sweep_direction,
            pullback_plan=plan,
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
        plan = getattr(signal, "pullback_plan", None)
        expiry = plan.expires_at if plan else now + timedelta(minutes=self.config.entry_refiner_expiry_minutes)
        entry = RefinerEntry(
            symbol=signal.symbol,
            signal=signal,
            added_at=now,
            expires_at=expiry,
            original_1h_price=signal.entry_price,
            entry_type="breakout",
            breakout_level=breakout_result.breakout_level,
            breakout_direction=breakout_result.breakout_direction or "",
            pullback_plan=plan,
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

    def add_enter_now(self, signal: SignalCandidate) -> bool:
        """Queue an ENTER_NOW signal for Agent 2 quick confirmation.

        Agent 1 said enter immediately, but we route through Agent 2 for
        a fast confirmation check. Agent 2 is called on the very first tick
        (no 5-minute wait). Short 10-minute expiry.

        Returns True if added, False if duplicate.
        """
        if signal.symbol in self.queue:
            return False

        now = datetime.now(timezone.utc)
        # Determine entry type from signal
        entry_type = "sweep"
        sweep_level = 0.0
        sweep_direction = ""
        breakout_level = 0.0
        breakout_direction = ""

        if signal.sweep_result and signal.sweep_result.sweep_detected:
            sweep_level = signal.sweep_result.sweep_level
            sweep_direction = signal.sweep_result.sweep_direction
        elif signal.breakout_result and signal.breakout_result.breakout_detected:
            entry_type = "breakout"
            breakout_level = signal.breakout_result.breakout_level
            breakout_direction = signal.breakout_result.breakout_direction or ""

        entry = RefinerEntry(
            symbol=signal.symbol,
            signal=signal,
            added_at=now,
            expires_at=now + timedelta(minutes=30),  # Shorter than pullback but enough for Agent 2 confirmation
            original_1h_price=signal.entry_price,
            entry_type=entry_type,
            sweep_level=sweep_level,
            sweep_direction=sweep_direction,
            breakout_level=breakout_level,
            breakout_direction=breakout_direction,
            pullback_plan=getattr(signal, "pullback_plan", None),
            enter_now=True,
        )
        self.queue[signal.symbol] = entry

        logger.info(
            "refiner_queued_enter_now",
            symbol=signal.symbol,
            score=signal.score,
            entry_type=entry_type,
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

                # PullbackPlan invalidation check — if price broke invalidation level, expire
                if entry.pullback_plan is not None:
                    try:
                        ticker = await self.candles.get_candles(symbol, "5m", limit=1)
                        if ticker is not None and not ticker.empty:
                            last_close = float(ticker["close"].iloc[-1])
                            if entry.pullback_plan.invalidation_hit(last_close):
                                logger.info(
                                    "pullback_plan_invalidated",
                                    symbol=symbol,
                                    last_close=round(last_close, 6),
                                    invalidation=round(entry.pullback_plan.invalidation_level, 6),
                                    zone=entry.pullback_plan.zone_str(),
                                    age_seconds=round(entry.pullback_plan.age_seconds, 1),
                                    path_taken="refiner_invalidation",
                                )
                                self.total_expired += 1
                                del self.queue[symbol]
                                self._last_agent_check.pop(symbol, None)
                                continue
                    except Exception:
                        pass  # Non-critical — the pre-dispatch gate will catch it

                # Fetch 5m candles and check for pullback
                candles_5m = await self.candles.get_candles(symbol, "5m", limit=30)
                if candles_5m is None or candles_5m.empty or len(candles_5m) < 10:
                    continue

                # Phase gate: expected_high must be reached before zone entry is allowed
                if entry.pullback_plan and entry.pullback_plan.expected_high > 0:
                    last_close = float(candles_5m["close"].iloc[-1])
                    candle_high = float(candles_5m["high"].iloc[-1])
                    candle_low = float(candles_5m["low"].iloc[-1])
                    # Check both close and wick extremes
                    check_price = candle_high if entry.pullback_plan.direction in ("bullish", "long") else candle_low
                    if not entry.pullback_plan.pullback_allowed(check_price):
                        if entry.check_count % 5 == 0:
                            logger.debug(
                                "refiner_waiting_expected_high",
                                symbol=symbol,
                                direction=entry.pullback_plan.direction,
                                expected_high=round(entry.pullback_plan.expected_high, 6),
                                current_price=round(last_close, 6),
                                check_count=entry.check_count,
                            )
                        entry.check_count += 1
                        continue

                entry.check_count += 1

                # Dispatch to strategy-specific check
                if entry.entry_type == "sweep":
                    result = await self._check_sweep_ote(entry, candles_5m)
                else:
                    result = await self._check_breakout_retest(entry, candles_5m)

                # Agent 2 ABANDON → remove from queue
                if getattr(entry, "_agent_abandoned", False):
                    logger.info(
                        "refiner_agent2_abandoned",
                        symbol=symbol,
                        reason=getattr(entry, "_agent_abandon_reason", ""),
                        check_count=entry.check_count,
                        agent2_checks=entry.agent2_check_count,
                    )
                    self.total_expired += 1
                    del self.queue[symbol]
                    # Clean up agent check timing
                    self._last_agent_check.pop(symbol, None)
                    continue

                if result is not None:
                    ready.append(result)
                    self.total_confirmed += 1
                    del self.queue[symbol]
                    self._last_agent_check.pop(symbol, None)

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

    async def _check_sweep_ote(
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

        # Guard: if Agent 2 already adjusted the zone, preserve it — don't recompute
        if entry.zone_source == "agent2_adjusted" and entry.ote_zone_valid:
            ote_top = entry.ote_zone_top
            ote_bottom = entry.ote_zone_bottom
            zone_resolved = True
        else:
            zone_resolved = False

        agent_high = getattr(signal, "agent_entry_zone_high", None)
        agent_low = getattr(signal, "agent_entry_zone_low", None)
        thrust_1h = getattr(signal, "thrust_extreme_1h", None)
        disp_open_1h = getattr(signal, "displacement_open_1h", None)

        ote_top = ote_top if zone_resolved else 0.0
        ote_bottom = ote_bottom if zone_resolved else 0.0

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

        # ── Agent 2 check (every 5 minutes, or immediately for ENTER_NOW) ──
        now_ts = _time.time()
        last_check = self._last_agent_check.get(entry.symbol, 0)
        agent_due = entry.enter_now or (now_ts - last_check) >= self._agent_check_interval

        if self.refiner_agent and agent_due:
            try:
                result = await self._agent_evaluate(
                    entry, candles_5m, ote_top, ote_bottom, direction,
                )
                self._last_agent_check[entry.symbol] = now_ts
                if result is not None:
                    return result
                # Agent said WAIT or ADJUST — don't fall through to algorithm
                # (algorithm runs on non-agent ticks as safety net)
                return None
            except Exception as e:
                logger.warning(
                    "refiner_agent2_exception",
                    symbol=entry.symbol,
                    error=str(e)[:100],
                )
                # Fall through to algorithmic check

        return self._find_rejection_in_zone(
            entry=entry,
            candles_5m=candles_5m,
            zone_top=ote_top,
            zone_bottom=ote_bottom,
            direction=direction,
        )

    # ── Breakout Retest Check ─────────────────────────────────────

    async def _check_breakout_retest(
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

        When Agent 2 is enabled, it handles ALL entry decisions.
        The algorithmic retest detection is the fallback.
        """
        direction = entry.breakout_direction
        level = entry.breakout_level
        tolerance = level * BREAKOUT_RETEST_TOLERANCE

        closes = candles_5m["close"].astype(float)

        # Guard: if Agent 2 already adjusted the zone, preserve it — don't recompute.
        if entry.zone_source == "agent2_adjusted" and entry.ote_zone_valid:
            zone_top = entry.ote_zone_top
            zone_bottom = entry.ote_zone_bottom
        else:
            # Retest zone around the breakout level
            zone_top = level + tolerance
            zone_bottom = level - tolerance

            # Store zone for dashboard visibility
            entry.ote_zone_valid = True
            entry.ote_zone_top = zone_top
            entry.ote_zone_bottom = zone_bottom
            if not entry.zone_source:
                entry.zone_source = "breakout_level"

        # ── Agent 2 check (every 5 minutes, or immediately for ENTER_NOW) ──
        now_ts = _time.time()
        last_check = self._last_agent_check.get(entry.symbol, 0)
        agent_due = entry.enter_now or (now_ts - last_check) >= self._agent_check_interval

        if self.refiner_agent and agent_due:
            try:
                result = await self._agent_evaluate(
                    entry, candles_5m, zone_top, zone_bottom, direction,
                )
                self._last_agent_check[entry.symbol] = now_ts
                if result is not None:
                    return result
                # Agent said WAIT or ADJUST — don't fall through to algorithm
                return None
            except Exception as e:
                logger.warning(
                    "refiner_agent2_exception",
                    symbol=entry.symbol,
                    error=str(e)[:100],
                )
                # Fall through to algorithmic check

        # ── Algorithmic fallback: retest + bounce detection ──
        highs = candles_5m["high"].astype(float)
        lows = candles_5m["low"].astype(float)
        opens = candles_5m["open"].astype(float)

        recent = candles_5m.tail(5)

        for i in range(len(recent)):
            candle = recent.iloc[i]
            c_close = float(candle["close"])
            c_open = float(candle["open"])
            c_high = float(candle["high"])
            c_low = float(candle["low"])
            body = abs(c_close - c_open)

            if direction == "bullish":
                if c_low <= zone_top and c_close > level:
                    lower_wick = min(c_open, c_close) - c_low
                    has_wick = body > 0 and lower_wick / body >= REJECTION_WICK_RATIO

                    vol_profile = self.vol_analyzer.analyze(candles_5m)
                    rvol = vol_profile.relative_volume
                    has_volume = rvol >= OTE_REJECTION_RVOL

                    if has_wick or has_volume:
                        target_entry = min(c_close, zone_top)
                        return self._create_refined_signal(
                            entry=entry,
                            entry_price=target_entry,
                            rejection_type="breakout_retest",
                            has_wick=has_wick,
                            rvol=rvol,
                        )

            elif direction == "bearish":
                if c_high >= zone_bottom and c_close < level:
                    upper_wick = c_high - max(c_open, c_close)
                    has_wick = body > 0 and upper_wick / body >= REJECTION_WICK_RATIO

                    vol_profile = self.vol_analyzer.analyze(candles_5m)
                    rvol = vol_profile.relative_volume
                    has_volume = rvol >= OTE_REJECTION_RVOL

                    if has_wick or has_volume:
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

    # ── Agent 2 Integration ─────────────────────────────────────────

    async def _agent_evaluate(
        self,
        entry: RefinerEntry,
        candles_5m: pd.DataFrame,
        zone_top: float,
        zone_bottom: float,
        direction: str,
    ) -> SignalCandidate | None:
        """Run Agent 2 evaluation and handle the decision.

        Returns:
            SignalCandidate if ENTER (ready to trade),
            None if WAIT/ADJUST_ZONE/ABANDON (stays in queue or gets removed).
        """
        # Fetch order book for Agent 2 context (non-critical, best-effort)
        # Req 6: robustness — toggle, staleness check, rate-limit safety, sanity checks
        order_book_data = {"status": "unavailable"}
        ob_enabled = getattr(self.config, "order_book_enabled", True) if self.config else True
        if self._exchange and ob_enabled:
            try:
                import time as _time
                ob_fetch_start = _time.time()
                ob = await self._exchange.fetch_order_book(entry.symbol, limit=20)
                ob_fetch_ms = (_time.time() - ob_fetch_start) * 1000

                if ob and ob.get("bids") and ob.get("asks"):
                    bids = ob["bids"][:10]  # Top 10 bid levels
                    asks = ob["asks"][:10]  # Top 10 ask levels

                    # Sanity check: bids/asks must have prices > 0
                    if not bids or not asks or bids[0][0] <= 0 or asks[0][0] <= 0:
                        logger.debug("order_book_invalid_prices", symbol=entry.symbol)
                    else:
                        bid_vol = sum(b[1] for b in bids)
                        ask_vol = sum(a[1] for a in asks)
                        total_vol = bid_vol + ask_vol
                        spread = asks[0][0] - bids[0][0]
                        spread_pct = (spread / bids[0][0] * 100) if bids[0][0] > 0 else 0

                        # Staleness guard: if spread is negative or fetch was very slow, skip
                        if spread < 0:
                            logger.debug("order_book_crossed_spread", symbol=entry.symbol)
                        elif ob_fetch_ms > 10000:
                            # Fetch took >10s — data may be stale
                            logger.debug("order_book_stale", symbol=entry.symbol, fetch_ms=round(ob_fetch_ms))
                        else:
                            imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0
                            # Wall detection: any level with volume > 3x the average
                            all_levels = bids + asks
                            avg_vol = total_vol / len(all_levels) if all_levels else 0
                            walls = []
                            for price, vol in all_levels:
                                if avg_vol > 0 and vol > avg_vol * 3:
                                    side = "bid" if [price, vol] in bids else "ask"
                                    walls.append({"price": price, "volume": vol, "side": side})
                            order_book_data = {
                                "status": "available",
                                "spread_pct": round(spread_pct, 4),
                                "bid_volume_top10": round(bid_vol, 2),
                                "ask_volume_top10": round(ask_vol, 2),
                                "imbalance_ratio": round(imbalance, 3),
                                "walls": walls[:3],
                                "best_bid": bids[0][0] if bids else 0,
                                "best_ask": asks[0][0] if asks else 0,
                                "fetch_ms": round(ob_fetch_ms, 1),
                            }
            except Exception as e:
                logger.debug("order_book_fetch_error", symbol=entry.symbol, error=str(e))

        context = self._build_agent_context(
            entry, candles_5m, zone_top, zone_bottom, direction, order_book_data,
        )

        decision = await self.refiner_agent.evaluate_entry(context)

        # Always store reasoning for continuity on next check
        entry.last_agent2_action = decision.action
        entry.last_agent2_reasoning = decision.reasoning
        entry.last_agent2_urgency = decision.urgency
        entry.agent2_check_count += 1

        # Req 8: shadow mode — log all decisions but don't act on them
        shadow = getattr(self.config, "agent2_shadow_mode", False) if self.config else False

        if decision.action == "ENTER":
            logger.info(
                "refiner_agent2_enter",
                symbol=entry.symbol,
                entry_price=round(decision.entry_price, 6),
                sl=round(decision.stop_loss, 6),
                tp=round(decision.take_profit, 6),
                size_modifier=decision.position_size_modifier,
                confidence=decision.confidence,
                agent2_check=entry.agent2_check_count,
                shadow=shadow,
            )
            if shadow:
                return None  # Shadow mode: log only, don't create signal
            return self._create_refined_signal_from_agent(entry, decision)

        elif decision.action == "ADJUST_ZONE":
            logger.info(
                "refiner_agent2_adjust_zone",
                symbol=entry.symbol,
                new_zone=f"{round(decision.adjusted_zone_low, 6)}-{round(decision.adjusted_zone_high, 6)}",
                confidence=decision.confidence,
                agent2_check=entry.agent2_check_count,
                shadow=shadow,
            )
            if not shadow:
                # Only adjust zone in live mode, not shadow
                entry.ote_zone_top = decision.adjusted_zone_high
                entry.ote_zone_bottom = decision.adjusted_zone_low
                entry.zone_source = "agent2_adjusted"
                # Sync PullbackPlan — update zone but do NOT reset expiry
                if entry.pullback_plan is not None:
                    entry.pullback_plan.zone_high = decision.adjusted_zone_high
                    entry.pullback_plan.zone_low = decision.adjusted_zone_low
                    entry.pullback_plan.limit_price = entry.pullback_plan.compute_limit_price()
                    entry.pullback_plan.zone_updates += 1
                    logger.info(
                        "pullback_plan_zone_adjusted",
                        symbol=entry.symbol,
                        new_zone=entry.pullback_plan.zone_str(),
                        new_limit=round(entry.pullback_plan.limit_price, 6),
                        zone_updates=entry.pullback_plan.zone_updates,
                        age_seconds=round(entry.pullback_plan.age_seconds, 1),
                    )
            return None

        elif decision.action == "ABANDON":
            entry._agent_abandoned = True
            entry._agent_abandon_reason = decision.invalidation_reason
            logger.info(
                "refiner_agent2_abandon",
                symbol=entry.symbol,
                reason=decision.invalidation_reason[:200],
                confidence=decision.confidence,
                agent2_check=entry.agent2_check_count,
            )
            return None

        else:  # WAIT
            if entry.agent2_check_count % 3 == 0:
                logger.debug(
                    "refiner_agent2_wait",
                    symbol=entry.symbol,
                    urgency=decision.urgency,
                    reasoning=decision.reasoning[:150],
                    agent2_check=entry.agent2_check_count,
                )
            return None

    def _build_agent_context(
        self,
        entry: RefinerEntry,
        candles_5m: pd.DataFrame,
        zone_top: float,
        zone_bottom: float,
        direction: str,
        order_book: dict | None = None,
    ) -> dict:
        """Build the context dict for Agent 2's prompt."""
        signal = entry.signal
        current_price = float(candles_5m["close"].iloc[-1])

        # Agent 1's analysis from signal components
        agent_analysis = {}
        components = signal.components or {}
        if "agent_analysis" in components:
            aa = components["agent_analysis"]
            if isinstance(aa, dict):
                agent_analysis = aa
            elif hasattr(aa, "__dict__"):
                agent_analysis = {
                    "action": getattr(aa, "action", ""),
                    "confidence": getattr(aa, "confidence", 0),
                    "reasoning": getattr(aa, "reasoning", ""),
                    "market_regime": getattr(aa, "market_regime", ""),
                    "risk_assessment": getattr(aa, "risk_assessment", ""),
                }

        # Build compact candle table (last 15 candles, newest first)
        candle_rows = []
        tail = candles_5m.tail(15).iloc[::-1]  # Reverse for newest first
        for _, row in tail.iterrows():
            c_open = float(row["open"])
            c_high = float(row["high"])
            c_low = float(row["low"])
            c_close = float(row["close"])
            c_vol = float(row.get("volume", 0))
            body = abs(c_close - c_open)
            direction_char = "▲" if c_close >= c_open else "▼"
            ts = row.get("timestamp", row.name)
            if hasattr(ts, "strftime"):
                ts_str = ts.strftime("%H:%M")
            else:
                ts_str = str(ts)[-5:]
            candle_rows.append(
                f"  {ts_str} | O:{c_open:.6g} H:{c_high:.6g} L:{c_low:.6g} "
                f"C:{c_close:.6g} | Vol:{c_vol:.0f} | Body:{body:.6g} {direction_char}"
            )
        candles_table = "\n".join(candle_rows)

        # Volume profile
        vol_profile = {}
        try:
            vp = self.vol_analyzer.analyze(candles_5m)
            vol_profile = {
                "relative_volume": vp.relative_volume,
                "displacement_detected": vp.displacement_detected,
                "volume_trend": getattr(vp, "volume_trend", ""),
            }
        except Exception:
            pass

        # Price change since queued
        price_change_pct = 0.0
        if entry.original_1h_price > 0:
            price_change_pct = (current_price - entry.original_1h_price) / entry.original_1h_price * 100

        # BTC context (from market filter if available)
        btc_trend = "unknown"
        btc_price = 0.0
        btc_1h_change = 0.0
        if self.market_filter:
            btc_trend = getattr(self.market_filter, "_cached_btc_trend", "unknown") or "unknown"
            btc_price = getattr(self.market_filter, "_cached_btc_price", 0) or 0
            btc_1h_change = getattr(self.market_filter, "_cached_btc_1h_change", 0) or 0

        # Structural levels from agent_context (flat dict, NOT organized by timeframe)
        ac = signal.agent_context or {}
        structural = {
            "order_blocks": ac.get("order_blocks", []),
            "fair_value_gaps": ac.get("fair_value_gaps", []),
            "liquidity": ac.get("liquidity", {}),
            "market_structure": ac.get("market_structure", {}),
            "price_in_ob": ac.get("price_in_ob", False),
            "price_in_fvg": ac.get("price_in_fvg", False),
        }

        # Pullback metrics (thrust magnitude, optimal entry, displacement origin)
        pullback = ac.get("pullback", {})

        # Leverage/funding profile (OI, funding rate, crowding, liquidation levels)
        leverage = ac.get("leverage", {})

        # Also grab the simple bonus if leverage profile is missing
        leverage_bonus = getattr(signal, "leverage_bonus", None)

        # Sweep details (type, depth, level)
        sweep_info = {}
        if signal.sweep_result:
            sr = signal.sweep_result
            sweep_info = {
                "sweep_type": sr.sweep_type,
                "sweep_direction": sr.sweep_direction,
                "sweep_depth": round(sr.sweep_depth, 6),
                "sweep_level": sr.sweep_level,
            }

        # Pre-computed SL/TP from algorithmic calculation (Agent 1 context)
        # These give Agent 2 concrete price targets to use
        agent1_sl = None
        agent1_tp = None
        agent1_rr = None
        if "agent_analysis" in components:
            aa = components.get("agent_analysis", {})
            if isinstance(aa, dict):
                # The SL/TP were stored in the signal's agent context during core.py processing
                agent1_sl = components.get("sl_price")
                agent1_tp = components.get("tp_price")
                agent1_rr = components.get("rr_ratio")

        # Also check signal-level attributes (set during pre-calculation)
        if agent1_sl is None:
            agent1_sl = getattr(signal, "_pre_sl", None)
        if agent1_tp is None:
            agent1_tp = getattr(signal, "_pre_tp", None)

        return {
            "symbol": entry.symbol,
            "direction": direction,
            "current_price": current_price,
            "agent1_analysis": agent_analysis,
            "previous_action": entry.last_agent2_action,
            "previous_reasoning": entry.last_agent2_reasoning,
            "previous_urgency": entry.last_agent2_urgency,
            "check_count": entry.agent2_check_count,
            "zone_top": zone_top,
            "zone_bottom": zone_bottom,
            "zone_source": entry.zone_source,
            "candles_5m_table": candles_table,
            "price_change_since_queue_pct": price_change_pct,
            "btc_trend": btc_trend,
            "btc_price": btc_price,
            "btc_1h_change": btc_1h_change,
            "structural_levels": structural,
            "volume_profile": vol_profile,
            "fibonacci_levels": signal.fibonacci_levels or {},
            "pullback": pullback,
            "leverage": leverage,
            "leverage_bonus": leverage_bonus,
            "sweep_info": sweep_info,
            "agent1_sl": agent1_sl,
            "agent1_tp": agent1_tp,
            "agent1_rr": agent1_rr,
            "order_book": order_book or {"status": "unavailable"},
            # PullbackPlan context — helps Agent 2 make time-aware decisions
            "pullback_plan": {
                "zone_low": entry.pullback_plan.zone_low,
                "zone_high": entry.pullback_plan.zone_high,
                "invalidation_level": entry.pullback_plan.invalidation_level,
                "limit_price": entry.pullback_plan.limit_price,
                "time_remaining_seconds": max(0, (entry.pullback_plan.expires_at - datetime.now(timezone.utc)).total_seconds()),
                "age_seconds": entry.pullback_plan.age_seconds,
                "zone_updates": entry.pullback_plan.zone_updates,
                "max_chase_bps": entry.pullback_plan.max_chase_bps,
                "expected_high": entry.pullback_plan.expected_high,
                "expected_high_reached": entry.pullback_plan.expected_high_reached,
            } if entry.pullback_plan else None,
            # ENTER_NOW flag: Agent 1 said enter immediately, Agent 2 is doing
            # a quick confirmation. Bias toward ENTER unless clearly bad setup.
            "enter_now_confirmation": entry.enter_now,
        }

    def _create_refined_signal_from_agent(
        self, entry: RefinerEntry, decision,
    ) -> SignalCandidate:
        """Create a refined signal using Agent 2's concrete parameters."""
        now = datetime.now(timezone.utc)
        duration = (now - entry.added_at).total_seconds()

        # Calculate improvement
        improvement_pct = 0.0
        if entry.original_1h_price > 0:
            improvement_pct = abs(
                decision.entry_price - entry.original_1h_price
            ) / entry.original_1h_price * 100

        signal = entry.signal
        signal.entry_price = decision.entry_price
        signal.refined_entry = True
        signal.refinement_duration_seconds = duration
        signal.original_1h_price = entry.original_1h_price

        # Attach Agent 2's overrides for core.py to pass to execute_entry
        signal._agent2_sl = decision.stop_loss
        signal._agent2_tp = decision.take_profit
        signal._agent2_size_modifier = decision.position_size_modifier

        logger.info(
            "refiner_agent2_confirmed",
            symbol=entry.symbol,
            entry_price=round(decision.entry_price, 6),
            original_1h_price=round(entry.original_1h_price, 6),
            improvement_pct=round(improvement_pct, 2),
            sl=round(decision.stop_loss, 6),
            tp=round(decision.take_profit, 6),
            size_modifier=decision.position_size_modifier,
            confidence=decision.confidence,
            zone_source=entry.zone_source,
            agent2_checks=entry.agent2_check_count,
            duration_seconds=round(duration, 0),
            direction=entry.sweep_direction or entry.breakout_direction,
        )

        return signal

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
                "zone_top": entry.ote_zone_top,
                "zone_bottom": entry.ote_zone_bottom,
                "last_agent2_action": entry.last_agent2_action,
                "last_agent2_reasoning": entry.last_agent2_reasoning,
                "last_agent2_urgency": entry.last_agent2_urgency,
                "agent2_check_count": entry.agent2_check_count,
                "enter_now": entry.enter_now,
                # PullbackPlan persistence
                "pullback_plan": {
                    "zone_low": entry.pullback_plan.zone_low,
                    "zone_high": entry.pullback_plan.zone_high,
                    "created_at": entry.pullback_plan.created_at.isoformat(),
                    "expires_at": entry.pullback_plan.expires_at.isoformat(),
                    "invalidation_level": entry.pullback_plan.invalidation_level,
                    "max_chase_bps": entry.pullback_plan.max_chase_bps,
                    "zone_tolerance_bps": entry.pullback_plan.zone_tolerance_bps,
                    "valid_for_candles": entry.pullback_plan.valid_for_candles,
                    "direction": entry.pullback_plan.direction,
                    "limit_price": entry.pullback_plan.limit_price,
                    "original_suggested_entry": entry.pullback_plan.original_suggested_entry,
                    "zone_updates": entry.pullback_plan.zone_updates,
                    "expected_high": entry.pullback_plan.expected_high,
                    "expected_high_reached": entry.pullback_plan.expected_high_reached,
                } if entry.pullback_plan else None,
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
                # Restore zone state so Agent 2 adjustments survive restarts
                entry.zone_source = entry_data.get("zone_source", "")
                if entry.zone_source:
                    entry.ote_zone_top = entry_data.get("zone_top", 0.0)
                    entry.ote_zone_bottom = entry_data.get("zone_bottom", 0.0)
                    entry.ote_zone_valid = entry.ote_zone_top > 0 and entry.ote_zone_bottom > 0
                # Restore enter_now flag and Agent 2 continuity fields
                entry.enter_now = entry_data.get("enter_now", False)
                entry.last_agent2_action = entry_data.get("last_agent2_action", "")
                entry.last_agent2_reasoning = entry_data.get("last_agent2_reasoning", "")
                entry.last_agent2_urgency = entry_data.get("last_agent2_urgency", "")
                # Restore PullbackPlan if present
                plan_data = entry_data.get("pullback_plan")
                if plan_data and isinstance(plan_data, dict):
                    try:
                        entry.pullback_plan = PullbackPlan(
                            zone_low=plan_data["zone_low"],
                            zone_high=plan_data["zone_high"],
                            created_at=datetime.fromisoformat(plan_data["created_at"]),
                            expires_at=datetime.fromisoformat(plan_data["expires_at"]),
                            invalidation_level=plan_data.get("invalidation_level", 0.0),
                            max_chase_bps=plan_data.get("max_chase_bps", 3.0),
                            zone_tolerance_bps=plan_data.get("zone_tolerance_bps", 2.0),
                            valid_for_candles=plan_data.get("valid_for_candles", 6),
                            direction=plan_data.get("direction", ""),
                            limit_price=plan_data.get("limit_price", 0.0),
                            original_suggested_entry=plan_data.get("original_suggested_entry", 0.0),
                            zone_updates=plan_data.get("zone_updates", 0),
                            expected_high=plan_data.get("expected_high", 0.0),
                            expected_high_reached=plan_data.get("expected_high_reached", False),
                        )
                        # Also restore plan on the signal
                        entry.signal.pullback_plan = entry.pullback_plan
                    except Exception as pe:
                        logger.warning("refiner_restore_plan_failed", symbol=sym, error=str(pe)[:80])
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
