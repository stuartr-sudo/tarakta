"""Post-Sweep Entry Refinement — find better entries on 5m after 1H sweep detection.

When the custom bot's 1H scanner detects a qualifying sweep signal (score >= 60),
instead of entering at the stale 1H candle close, the signal is queued here.
The refiner monitors 5-minute candles every 60 seconds looking for the moment
price *reclaims* the swept level with volume — the actual reversal point.

This gets entries much closer to the real bottom/top of the sweep.

Flow:
  1H scan detects sweep signal (score >= 60) → queue in EntryRefiner
  Every 60s: check 5m candles for sweep reclaim + volume confirmation
  Reclaim confirmed → return refined signal with better entry price
  Expired (30 min) → return signal with current price (fallback)
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

# Minimum RVOL on the reclaim candle(s) to confirm institutional interest
RECLAIM_RVOL_THRESHOLD = 1.3

# Minimum wick ratio (wick / body) for rejection wick bonus confirmation
REJECTION_WICK_RATIO = 0.5


@dataclass
class RefinerEntry:
    """A sweep signal queued for 5m entry refinement."""

    symbol: str
    signal: SignalCandidate        # Full signal from 1H scan
    added_at: datetime
    expires_at: datetime
    sweep_level: float             # Level that was swept (from SweepResult)
    sweep_direction: str           # "bullish" (swept lows) or "bearish" (swept highs)
    original_1h_price: float       # 1H close we're trying to improve on
    check_count: int = 0
    best_5m_price: float | None = None


class EntryRefiner:
    """Monitors queued sweep signals on 5m candles for better entry timing.

    Not an independent async loop — called from FlippedTrader's _monitor_loop()
    on each 60-second tick. Lightweight: just fetches 5m candles and checks for
    sweep reclaim confirmation.
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

    # ── Public API ──────────────────────────────────────────────

    def add(self, signal: SignalCandidate) -> bool:
        """Queue a sweep signal for 5m entry refinement.

        Returns True if added, False if full or duplicate.
        """
        if signal.symbol in self.queue:
            return False
        if len(self.queue) >= self.config.entry_refiner_max_queue:
            logger.info(
                "refiner_full",
                rejected=signal.symbol,
                score=signal.score,
                current_size=len(self.queue),
            )
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
            sweep_level=sweep_result.sweep_level,
            sweep_direction=sweep_result.sweep_direction,
            original_1h_price=signal.entry_price,
        )
        self.queue[signal.symbol] = entry

        logger.info(
            "refiner_queued",
            symbol=signal.symbol,
            score=signal.score,
            sweep_level=round(sweep_result.sweep_level, 6),
            sweep_direction=sweep_result.sweep_direction,
            sweep_type=sweep_result.sweep_type,
            original_1h_price=round(signal.entry_price, 6),
            expires_at=entry.expires_at.isoformat(),
            queue_size=len(self.queue),
        )
        return True

    async def check_all(self) -> list[SignalCandidate]:
        """Check all queued entries on 5m data.

        Returns a list of signals ready to enter (either refined or expired-fallback).
        Removes completed entries from the queue.
        """
        if not self.queue:
            return []

        ready: list[SignalCandidate] = []
        now = datetime.now(timezone.utc)

        for symbol, entry in list(self.queue.items()):
            try:
                # Check expiry first
                if now >= entry.expires_at:
                    # Expired — return signal with current price as fallback
                    expired_signal = self._create_expired_signal(entry)
                    if expired_signal:
                        ready.append(expired_signal)
                    del self.queue[symbol]
                    logger.info(
                        "refiner_expired",
                        symbol=symbol,
                        check_count=entry.check_count,
                        original_1h_price=round(entry.original_1h_price, 6),
                        duration_seconds=round(
                            (now - entry.added_at).total_seconds(), 0
                        ),
                    )
                    continue

                # Fetch 5m candles and check for reclaim
                candles_5m = await self.candles.get_candles(symbol, "5m", limit=30)
                if candles_5m is None or candles_5m.empty or len(candles_5m) < 10:
                    continue

                entry.check_count += 1
                result = self._check_entry(entry, candles_5m)

                if result is not None:
                    ready.append(result)
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

    # ── Reclaim Detection ──────────────────────────────────────

    def _check_entry(
        self, entry: RefinerEntry, candles_5m: pd.DataFrame
    ) -> SignalCandidate | None:
        """Check if the swept level has been reclaimed on 5m with volume.

        For bullish sweep (swept lows, expecting reversal up):
          - Reclaim = 5m candle close ABOVE the swept level
          - Volume = RVOL >= 1.3x on recent candles
          - Rejection wick = lower wick >= 50% of body (buyers stepping in)

        For bearish sweep (swept highs, expecting reversal down):
          - Mirror: 5m close BELOW swept level, upper wick rejection
        """
        sweep_level = entry.sweep_level
        direction = entry.sweep_direction  # "bullish" or "bearish"

        # Look at the last 5 candles for reclaim evidence
        recent = candles_5m.tail(5)

        reclaim_detected = False
        reclaim_price = None
        has_rejection_wick = False

        for i in range(len(recent)):
            candle = recent.iloc[i]
            close = float(candle["close"])
            open_ = float(candle["open"])
            high = float(candle["high"])
            low = float(candle["low"])
            body = abs(close - open_)

            if direction == "bullish":
                # Swept lows → expecting price to reverse UP
                # Reclaim = close above the swept low level
                if close > sweep_level:
                    reclaim_detected = True
                    reclaim_price = close
                    # Rejection wick: long lower wick (buyers stepping in)
                    lower_wick = min(open_, close) - low
                    if body > 0 and lower_wick / body >= REJECTION_WICK_RATIO:
                        has_rejection_wick = True
                    break

            elif direction == "bearish":
                # Swept highs → expecting price to reverse DOWN
                # Reclaim = close below the swept high level
                if close < sweep_level:
                    reclaim_detected = True
                    reclaim_price = close
                    # Rejection wick: long upper wick (sellers stepping in)
                    upper_wick = high - max(open_, close)
                    if body > 0 and upper_wick / body >= REJECTION_WICK_RATIO:
                        has_rejection_wick = True
                    break

        if not reclaim_detected or reclaim_price is None:
            # Log periodic check status
            if entry.check_count % 5 == 0:  # Every 5 checks (~5 min)
                current_price = float(candles_5m["close"].iloc[-1])
                logger.debug(
                    "refiner_check",
                    symbol=entry.symbol,
                    check_count=entry.check_count,
                    reclaim_detected=False,
                    current_price=round(current_price, 6),
                    sweep_level=round(sweep_level, 6),
                    direction=direction,
                )
            return None

        # Volume confirmation — check RVOL on the reclaim candle's neighborhood
        vol_profile = self.vol_analyzer.analyze(candles_5m)
        rvol = vol_profile.relative_volume

        if rvol < RECLAIM_RVOL_THRESHOLD:
            logger.debug(
                "refiner_check",
                symbol=entry.symbol,
                check_count=entry.check_count,
                reclaim_detected=True,
                rvol=round(rvol, 2),
                rvol_threshold=RECLAIM_RVOL_THRESHOLD,
                rejected_reason="low_volume",
            )
            return None

        # ── Reclaim confirmed with volume ──
        now = datetime.now(timezone.utc)
        duration = (now - entry.added_at).total_seconds()

        # Calculate improvement
        improvement_pct = 0.0
        if entry.original_1h_price > 0:
            improvement_pct = abs(
                reclaim_price - entry.original_1h_price
            ) / entry.original_1h_price * 100

        # Update signal with refined entry
        signal = entry.signal
        signal.entry_price = reclaim_price
        signal.refined_entry = True
        signal.refinement_duration_seconds = duration
        signal.original_1h_price = entry.original_1h_price

        logger.info(
            "refiner_confirmed",
            symbol=entry.symbol,
            entry_price=round(reclaim_price, 6),
            original_1h_price=round(entry.original_1h_price, 6),
            improvement_pct=round(improvement_pct, 2),
            rvol=round(rvol, 2),
            rejection_wick=has_rejection_wick,
            check_count=entry.check_count,
            duration_seconds=round(duration, 0),
            direction=direction,
        )

        return signal

    def _create_expired_signal(self, entry: RefinerEntry) -> SignalCandidate | None:
        """Create a fallback signal when refinement window expires.

        The 1H signal was already validated (score >= 60), so we still enter
        but at the current 5m close price instead of the stale 1H close.
        """
        signal = entry.signal
        now = datetime.now(timezone.utc)
        duration = (now - entry.added_at).total_seconds()

        # Mark as non-refined (entered on expiry, not on reclaim)
        signal.refined_entry = False
        signal.refinement_duration_seconds = duration
        signal.original_1h_price = entry.original_1h_price
        # entry_price stays as whatever it was (will be updated by _try_enter's
        # live ticker fetch anyway)

        return signal

    # ── State Persistence ──────────────────────────────────────

    def get_state(self) -> dict:
        """Serialize refiner state for DB persistence."""
        entries_data = {}
        for sym, entry in self.queue.items():
            entries_data[sym] = {
                "symbol": entry.symbol,
                "added_at": entry.added_at.isoformat(),
                "expires_at": entry.expires_at.isoformat(),
                "sweep_level": entry.sweep_level,
                "sweep_direction": entry.sweep_direction,
                "original_1h_price": entry.original_1h_price,
                "check_count": entry.check_count,
                "score": entry.signal.score,
                "direction": entry.signal.direction,
                "components": entry.signal.components,
            }
        return {
            "entries": entries_data,
            "total_queued": len(entries_data),
        }

    def restore_state(self, data: dict) -> None:
        """Restore refiner state from DB.

        Like WatchlistMonitor, we can't fully reconstruct SignalCandidate objects
        from serialized state. Restored entries will expire naturally (30 min window
        is short enough that stale entries are unlikely to survive a restart).
        We only restore enough to let them either enter on expiry or drop.
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

                # Create minimal placeholder signal
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
                    sweep_level=entry_data.get("sweep_level", 0),
                    sweep_direction=entry_data.get("sweep_direction", "bullish"),
                    original_1h_price=entry_data.get("original_1h_price", 0),
                    check_count=entry_data.get("check_count", 0),
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
