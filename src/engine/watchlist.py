"""Hyper-Watchlist Monitor — fast 5m monitoring for near-miss signals.

When the 15-minute scanner detects "something's happening" (sweep found but
missing displacement/pullback/timing), the symbol is promoted to this watchlist.
The monitor checks every 2.5 minutes on 5-minute candles for the missing
confirmation to appear.

Flow:
  Regular scan (15 min, 1H) → near-miss detected → add to watchlist
  Watchlist monitor (2.5 min, 5m) → check for displacement/pullback on 5m
  Conditions align → graduate signal → feed to main engine execution path
  Expired (3 hours) → release back to regular pool
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.config import Settings
from src.data.candles import CandleManager
from src.exchange.models import SignalCandidate
from src.strategy.confluence import BREAKOUT_THRESHOLD, WEIGHTS
from src.strategy.pullback import PullbackAnalyzer
from src.strategy.volume import VolumeAnalyzer
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WatchlistEntry:
    """A near-miss signal being monitored on 5m candles."""

    symbol: str
    added_at: datetime
    expires_at: datetime
    initial_score: float
    initial_signal: SignalCandidate  # Full signal with components, sweep_result, etc.
    signal_type: str  # "sweep" or "breakout"
    direction: str  # From sweep/breakout detection
    htf_direction: str | None
    check_count: int = 0
    last_checked: datetime | None = None
    best_5m_score: float = 0.0


class WatchlistMonitor:
    """Monitors near-miss symbols on 5m candles for missing confirmations.

    Runs as an independent async loop (like FlippedTrader). Checks every 2.5
    minutes for displacement, pullback, or volume confirmation appearing on 5m
    data. When conditions align, graduates the signal and pushes it to the main
    engine's execution queue.
    """

    def __init__(
        self,
        candle_manager: CandleManager,
        config: Settings,
        signal_queue: asyncio.Queue,
    ) -> None:
        self.candles = candle_manager
        self.config = config
        self.signal_queue = signal_queue
        self.entries: dict[str, WatchlistEntry] = {}
        self.vol_analyzer = VolumeAnalyzer()
        self.pullback_analyzer = PullbackAnalyzer(
            min_retracement=config.pullback_min_retracement,
            max_retracement=config.pullback_max_retracement,
        )
        self._running = True

    # ── Public API ──────────────────────────────────────────────

    def add_entry(self, signal: SignalCandidate, signal_type: str) -> bool:
        """Add a near-miss signal to the watchlist.

        Returns True if added, False if full or duplicate.
        """
        if signal.symbol in self.entries:
            return False
        if self.config.watchlist_max_size > 0 and len(self.entries) >= self.config.watchlist_max_size:
            logger.info(
                "watchlist_full",
                rejected=signal.symbol,
                score=signal.score,
                current_size=len(self.entries),
            )
            return False
        if signal.direction is None:
            return False

        now = datetime.now(timezone.utc)
        entry = WatchlistEntry(
            symbol=signal.symbol,
            added_at=now,
            expires_at=now + timedelta(hours=self.config.watchlist_expiry_hours),
            initial_score=signal.score,
            initial_signal=signal,
            signal_type=signal_type,
            direction=signal.direction,
            htf_direction=signal.htf_direction_cache,
        )
        self.entries[signal.symbol] = entry

        logger.info(
            "watchlist_added",
            symbol=signal.symbol,
            initial_score=signal.score,
            signal_type=signal_type,
            direction=signal.direction,
            components=signal.components,
            expires_at=entry.expires_at.isoformat(),
            watchlist_size=len(self.entries),
        )
        return True

    def remove_entry(self, symbol: str) -> None:
        """Remove a symbol from the watchlist."""
        self.entries.pop(symbol, None)

    def get_excluded_symbols(self) -> set[str]:
        """Return symbols currently being monitored (excluded from regular scan)."""
        return set(self.entries.keys())

    # ── Async Loop ──────────────────────────────────────────────

    async def run_loop(self) -> None:
        """Main monitoring loop — checks every 2.5 min."""
        logger.info(
            "watchlist_monitor_started",
            interval_seconds=self.config.watchlist_monitor_interval_seconds,
            expiry_hours=self.config.watchlist_expiry_hours,
            max_size=self.config.watchlist_max_size,
        )

        while self._running:
            try:
                await asyncio.sleep(self.config.watchlist_monitor_interval_seconds)
                if not self.entries:
                    continue
                await self._monitor_cycle()
            except asyncio.CancelledError:
                logger.info("watchlist_monitor_cancelled")
                break
            except Exception as e:
                logger.error("watchlist_monitor_error", error=str(e))
                await asyncio.sleep(30)

    async def _monitor_cycle(self) -> None:
        """Check all watchlist entries on 5m data."""
        # 1. Expire stale entries
        self._expire_entries()

        if not self.entries:
            return

        logger.debug(
            "watchlist_cycle",
            entries=len(self.entries),
            symbols=[e.symbol for e in self.entries.values()],
        )

        # 2. Check each entry (stagger API calls to avoid rate limits)
        for idx, (symbol, entry) in enumerate(list(self.entries.items())):
            # Small delay between symbols to avoid Binance rate limit flooding
            if idx > 0:
                await asyncio.sleep(1.0)
            try:
                graduated = await self._check_entry(entry)
                if graduated is not None:
                    # Push to main engine's signal queue
                    await self.signal_queue.put(graduated)
                    self.remove_entry(symbol)
                    logger.info(
                        "watchlist_graduated",
                        symbol=symbol,
                        final_score=graduated.score,
                        direction=graduated.direction,
                        duration_seconds=graduated.watchlist_duration_seconds,
                        checks_taken=entry.check_count,
                        reasons=graduated.reasons,
                    )
            except Exception as e:
                logger.warning(
                    "watchlist_check_failed",
                    symbol=symbol,
                    error=str(e),
                )

    async def _check_entry(self, entry: WatchlistEntry) -> SignalCandidate | None:
        """Check if a watchlist entry now meets threshold on 5m data."""
        candles_5m = await self.candles.get_candles(entry.symbol, "5m", limit=60)

        if candles_5m is None or candles_5m.empty or len(candles_5m) < 25:
            return None

        entry.check_count += 1
        entry.last_checked = datetime.now(timezone.utc)

        if entry.signal_type == "sweep":
            return self._check_sweep_entry(entry, candles_5m)
        else:
            return self._check_breakout_entry(entry, candles_5m)

    def _check_sweep_entry(self, entry, candles_5m) -> SignalCandidate | None:
        """For sweep near-misses: look for displacement + pullback on 5m."""
        # Run volume analysis on 5m (displacement detection)
        vol_profile = self.vol_analyzer.analyze(candles_5m)

        displacement_confirmed = vol_profile.displacement_detected
        displacement_direction = vol_profile.displacement_direction

        # Volume sustainability check (same as 1H pipeline)
        if displacement_confirmed and vol_profile.volume_trend == "decreasing":
            displacement_confirmed = False
            displacement_direction = None

        # Direction must match the original sweep
        if displacement_confirmed and displacement_direction != entry.direction:
            displacement_confirmed = False

        # Pullback on 5m (if displacement found)
        pullback_result = None
        if displacement_confirmed and vol_profile.displacement_candle_idx is not None:
            pullback_result = self.pullback_analyzer.analyze(
                candles_1h=candles_5m,  # Works on any TF DataFrame
                displacement_candle_idx=vol_profile.displacement_candle_idx,
                direction=displacement_direction,
            )

        # Re-score using original components + new 5m data
        orig = entry.initial_signal
        score = entry.initial_score
        reasons = list(orig.reasons)
        components = dict(orig.components)

        # Add displacement points if detected on 5m (and wasn't already scored)
        disp_weight = WEIGHTS.get("displacement_confirmed", 25)
        if displacement_confirmed and components.get("displacement_confirmed", 0) == 0:
            score += disp_weight
            components["displacement_confirmed"] = disp_weight
            reasons.append(f"5m displacement confirmed: {displacement_direction}")

        # Add pullback points if detected on 5m (and wasn't already scored)
        pb_weight = WEIGHTS.get("pullback_confirmed", 10)
        if (
            pullback_result
            and pullback_result.pullback_detected
            and components.get("pullback_confirmed", 0) == 0
        ):
            score += pb_weight
            components["pullback_confirmed"] = pb_weight
            reasons.append(
                f"5m pullback detected: {pullback_result.retracement_pct:.0%} "
                f"retracement"
            )

        entry.best_5m_score = max(entry.best_5m_score, score)

        # Check threshold
        threshold = self.config.entry_threshold  # 60 for sweep signals
        if score >= threshold:
            now = datetime.now(timezone.utc)
            current_price = float(candles_5m["close"].iloc[-1])

            # Use pullback price for better entry if available
            if pullback_result and pullback_result.pullback_detected:
                current_price = pullback_result.current_price

            graduated = SignalCandidate(
                score=score,
                direction=entry.direction,
                reasons=reasons + [
                    f"Watchlist-promoted after {entry.check_count} checks "
                    f"({(now - entry.added_at).total_seconds():.0f}s)"
                ],
                symbol=entry.symbol,
                entry_price=current_price,
                components=components,
                atr_1h=orig.atr_1h,
                sweep_result=orig.sweep_result,
                session_result=orig.session_result,
                breakout_result=orig.breakout_result,
                key_levels=orig.key_levels,
                watchlist_promoted=True,
                watchlist_duration_seconds=(now - entry.added_at).total_seconds(),
                htf_direction_cache=entry.htf_direction,
            )
            return graduated

        return None

    def _check_breakout_entry(self, entry, candles_5m) -> SignalCandidate | None:
        """For breakout near-misses: check volume confirmation + hold on 5m."""
        orig = entry.initial_signal
        score = entry.initial_score
        reasons = list(orig.reasons)
        components = dict(orig.components)

        # Check RVOL on 5m for volume confirmation
        vol_profile = self.vol_analyzer.analyze(candles_5m)

        if vol_profile.relative_volume >= 1.5 and components.get("volume_confirmed", 0) == 0:
            from src.strategy.confluence import BREAKOUT_WEIGHTS

            vol_weight = BREAKOUT_WEIGHTS.get("volume_confirmed", 20)
            score += vol_weight
            components["volume_confirmed"] = vol_weight
            reasons.append(
                f"5m volume confirmed: RVOL {vol_profile.relative_volume:.1f}x"
            )

        # Check if price is still holding beyond breakout level
        if orig.breakout_result is not None:
            current_price = float(candles_5m["close"].iloc[-1])
            bo = orig.breakout_result
            if bo.breakout_direction == "bullish" and current_price < bo.breakout_level:
                # Price fell back below breakout level — invalidated
                return None
            if bo.breakout_direction == "bearish" and current_price > bo.breakout_level:
                return None

        entry.best_5m_score = max(entry.best_5m_score, score)

        # Check threshold
        if score >= BREAKOUT_THRESHOLD:
            now = datetime.now(timezone.utc)
            current_price = float(candles_5m["close"].iloc[-1])

            graduated = SignalCandidate(
                score=score,
                direction=entry.direction,
                reasons=reasons + [
                    f"Watchlist-promoted (breakout) after {entry.check_count} checks"
                ],
                symbol=entry.symbol,
                entry_price=current_price,
                components=components,
                atr_1h=orig.atr_1h,
                sweep_result=orig.sweep_result,
                session_result=orig.session_result,
                breakout_result=orig.breakout_result,
                key_levels=orig.key_levels,
                watchlist_promoted=True,
                watchlist_duration_seconds=(now - entry.added_at).total_seconds(),
                htf_direction_cache=entry.htf_direction,
            )
            return graduated

        return None

    # ── Expiry ──────────────────────────────────────────────────

    def _expire_entries(self) -> None:
        """Remove entries past their expiry time."""
        now = datetime.now(timezone.utc)
        expired = [sym for sym, e in self.entries.items() if now >= e.expires_at]
        for sym in expired:
            entry = self.entries[sym]
            logger.info(
                "watchlist_expired",
                symbol=sym,
                initial_score=entry.initial_score,
                best_5m_score=entry.best_5m_score,
                checks_taken=entry.check_count,
                duration_hours=round(
                    (now - entry.added_at).total_seconds() / 3600, 1
                ),
            )
            del self.entries[sym]

    # ── State Persistence ───────────────────────────────────────

    def get_state(self) -> dict:
        """Serialize watchlist state for DB persistence."""
        entries_data = {}
        for sym, entry in self.entries.items():
            entries_data[sym] = {
                "symbol": entry.symbol,
                "added_at": entry.added_at.isoformat(),
                "expires_at": entry.expires_at.isoformat(),
                "initial_score": entry.initial_score,
                "signal_type": entry.signal_type,
                "direction": entry.direction,
                "htf_direction": entry.htf_direction,
                "check_count": entry.check_count,
                "best_5m_score": entry.best_5m_score,
            }
        return {
            "entries": entries_data,
            "total_added": len(entries_data),
        }

    def restore_state(self, data: dict) -> None:
        """Restore watchlist state from DB.

        Note: We only restore metadata (not full SignalCandidate objects) because
        those contain complex nested objects (SweepResult, etc.). Restored entries
        will be checked on 5m data independently — if the conditions that made them
        near-misses have changed, they'll either graduate or expire naturally.

        Rather than trying to reconstruct full signals from serialized state,
        we let expired entries drop and fresh scans repopulate the watchlist.
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

                # We can't fully restore the initial_signal (complex object),
                # so we create a minimal placeholder. The entry will either
                # graduate from fresh 5m analysis or expire.
                placeholder_signal = SignalCandidate(
                    score=entry_data.get("initial_score", 0),
                    direction=entry_data.get("direction"),
                    symbol=sym,
                    components={},
                )

                entry = WatchlistEntry(
                    symbol=sym,
                    added_at=datetime.fromisoformat(entry_data["added_at"]),
                    expires_at=expires_at,
                    initial_score=entry_data.get("initial_score", 0),
                    initial_signal=placeholder_signal,
                    signal_type=entry_data.get("signal_type", "sweep"),
                    direction=entry_data.get("direction", ""),
                    htf_direction=entry_data.get("htf_direction"),
                    check_count=entry_data.get("check_count", 0),
                    best_5m_score=entry_data.get("best_5m_score", 0),
                )
                self.entries[sym] = entry
                restored += 1
            except Exception as e:
                logger.warning(
                    "watchlist_restore_entry_failed",
                    symbol=sym,
                    error=str(e),
                )

        # Trim to max_size: keep the highest-scored entries
        trimmed = 0
        if self.config.watchlist_max_size > 0 and len(self.entries) > self.config.watchlist_max_size:
            sorted_entries = sorted(
                self.entries.items(),
                key=lambda kv: kv[1].initial_score,
                reverse=True,
            )
            keep = {sym for sym, _ in sorted_entries[: self.config.watchlist_max_size]}
            to_remove = [sym for sym in self.entries if sym not in keep]
            for sym in to_remove:
                del self.entries[sym]
                trimmed += 1

        if restored > 0 or expired > 0 or trimmed > 0:
            logger.info(
                "watchlist_restored",
                restored=restored,
                expired_on_restore=expired,
                trimmed_to_max=trimmed,
                final_size=len(self.entries),
            )

    def stop(self) -> None:
        """Stop the monitor loop."""
        self._running = False
