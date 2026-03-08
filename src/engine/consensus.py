"""Market Consensus Check — portfolio direction + BTC trend alignment before entry.

Before entering a new trade, this module checks:
1. Portfolio Direction Bias — are most *profitable* open positions LONG or SHORT?
2. BTC/USDT Trend — is the market-dominant pair aligned with the signal?

If the new signal disagrees with the consensus, a weighted score penalty is applied:
  - Portfolio bias alone disagrees → -10 points
  - BTC trend also disagrees      → -15 points total
  - Only BTC disagrees            → -10 points

If the penalized score drops below the entry threshold, the signal is queued in a
ConsensusMonitor for periodic re-evaluation. When consensus changes (e.g. portfolio
flips, BTC reverses), the signal graduates — potentially with a flipped direction.

Flow:
  Signal scores >= 60 → compute_consensus() → penalty applied
  Score still >= 60   → proceed to entry (penalty logged as informational)
  Score drops < 60    → queue in ConsensusMonitor
  Every 60s: re-check → consensus changed? → graduate (possibly direction-flipped)
  Expired (30 min)    → drop from queue
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.config import Settings
from src.data.candles import CandleManager
from src.exchange.models import Position, SignalCandidate
from src.strategy.market_structure import MarketStructureAnalyzer
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConsensusResult:
    """Result of a consensus check against portfolio + BTC."""

    penalty: float                         # Points to subtract (0, 10, or 15)
    portfolio_bias: str                    # "long", "short", "neutral", "insufficient"
    profitable_longs: int = 0
    profitable_shorts: int = 0
    losing_longs: int = 0
    losing_shorts: int = 0
    btc_trend: str = "ranging"             # "bullish", "bearish", "ranging"
    signal_direction: str = ""             # "bullish" or "bearish"
    disagreement_reasons: list[str] = field(default_factory=list)
    applied: bool = False                  # False when < min_positions or fetch failed


@dataclass
class ConsensusEntry:
    """A signal parked in the consensus monitor queue."""

    symbol: str
    signal: SignalCandidate
    original_score: float                  # Score before penalty
    adjusted_score: float                  # Score after penalty
    consensus_result: ConsensusResult      # Snapshot of why it was penalized
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    check_count: int = 0
    last_checked: datetime | None = None


class ConsensusMonitor:
    """Checks portfolio + BTC consensus before entry and monitors deferred signals.

    Used by both the main bot and custom bot. Not an independent async loop —
    called from the existing monitor loops (like EntryRefiner).
    """

    def __init__(
        self,
        candle_manager: CandleManager,
        config: Settings,
    ) -> None:
        self.candles = candle_manager
        self.config = config
        self.queue: dict[str, ConsensusEntry] = {}
        self._ms_analyzer = MarketStructureAnalyzer()
        # Cache BTC trend to avoid re-fetching on every signal in the same scan cycle
        self._btc_trend_cache: str | None = None
        self._btc_trend_cache_time: datetime | None = None
        self._btc_cache_ttl_seconds = 120  # 2 min cache

    # ── Public API ──────────────────────────────────────────────

    async def compute_consensus(
        self,
        signal: SignalCandidate,
        open_positions: dict[str, Position],
        exchange,
        candle_manager: CandleManager | None = None,
    ) -> ConsensusResult:
        """Check if a new signal aligns with portfolio consensus + BTC trend.

        Returns a ConsensusResult with the penalty to apply (0, 10, or 15).
        """
        cm = candle_manager or self.candles
        signal_dir = signal.direction  # "bullish" or "bearish"

        # Early exit: not enough positions to form consensus
        if len(open_positions) < self.config.consensus_min_positions:
            return ConsensusResult(
                penalty=0.0,
                portfolio_bias="insufficient",
                signal_direction=signal_dir or "",
                applied=False,
                disagreement_reasons=["Not enough open positions for consensus"],
            )

        # ── 1. Portfolio Direction Bias ────────────────────────────
        portfolio_bias, p_longs, p_shorts, l_longs, l_shorts, portfolio_reasons = (
            await self._compute_portfolio_bias(open_positions, exchange)
        )

        # ── 2. BTC/USDT Trend ─────────────────────────────────────
        btc_trend = await self._get_btc_trend(cm)

        # ── 3. Weighted Penalty ───────────────────────────────────
        penalty = 0.0
        reasons: list[str] = list(portfolio_reasons)
        portfolio_disagrees = False
        btc_disagrees = False

        if signal_dir and portfolio_bias not in ("neutral", "insufficient"):
            # Portfolio is biased — check if signal goes against it
            # bullish signal vs short bias, or bearish signal vs long bias
            if (signal_dir == "bullish" and portfolio_bias == "short") or (
                signal_dir == "bearish" and portfolio_bias == "long"
            ):
                portfolio_disagrees = True
                reasons.append(
                    f"Signal {signal_dir} vs portfolio bias {portfolio_bias} "
                    f"(profitable: {p_longs}L/{p_shorts}S)"
                )

        if signal_dir and btc_trend != "ranging":
            # BTC has a clear trend — check if signal goes against it
            if (signal_dir == "bullish" and btc_trend == "bearish") or (
                signal_dir == "bearish" and btc_trend == "bullish"
            ):
                btc_disagrees = True
                reasons.append(f"Signal {signal_dir} vs BTC trend {btc_trend}")

        # Weighted penalty:
        #   Portfolio alone disagrees → -10
        #   BTC also disagrees       → -15 total
        #   Only BTC disagrees       → -10
        if portfolio_disagrees and btc_disagrees:
            penalty = self.config.consensus_btc_penalty  # 15.0
        elif portfolio_disagrees:
            penalty = self.config.consensus_portfolio_penalty  # 10.0
        elif btc_disagrees:
            penalty = self.config.consensus_portfolio_penalty  # 10.0

        applied = penalty > 0

        if applied:
            logger.info(
                "consensus_penalty",
                symbol=signal.symbol,
                signal_direction=signal_dir,
                portfolio_bias=portfolio_bias,
                btc_trend=btc_trend,
                penalty=penalty,
                profitable_longs=p_longs,
                profitable_shorts=p_shorts,
            )

        return ConsensusResult(
            penalty=penalty,
            portfolio_bias=portfolio_bias,
            profitable_longs=p_longs,
            profitable_shorts=p_shorts,
            losing_longs=l_longs,
            losing_shorts=l_shorts,
            btc_trend=btc_trend,
            signal_direction=signal_dir or "",
            disagreement_reasons=reasons,
            applied=applied,
        )

    def add(self, signal: SignalCandidate, consensus_result: ConsensusResult) -> bool:
        """Queue a penalized signal for consensus monitoring.

        Returns True if added, False if full or duplicate.
        """
        if signal.symbol in self.queue:
            return False
        if len(self.queue) >= self.config.consensus_max_queue:
            logger.info(
                "consensus_queue_full",
                rejected=signal.symbol,
                current_size=len(self.queue),
            )
            return False

        now = datetime.now(timezone.utc)
        entry = ConsensusEntry(
            symbol=signal.symbol,
            signal=signal,
            original_score=signal.score,
            adjusted_score=signal.score - consensus_result.penalty,
            consensus_result=consensus_result,
            added_at=now,
            expires_at=now + timedelta(minutes=self.config.consensus_monitor_expiry_minutes),
        )
        self.queue[signal.symbol] = entry

        logger.info(
            "consensus_queued",
            symbol=signal.symbol,
            original_score=signal.score,
            adjusted_score=entry.adjusted_score,
            penalty=consensus_result.penalty,
            portfolio_bias=consensus_result.portfolio_bias,
            btc_trend=consensus_result.btc_trend,
            direction=signal.direction,
            expires_at=entry.expires_at.isoformat(),
            queue_size=len(self.queue),
        )
        return True

    async def check_all(
        self,
        open_positions: dict[str, Position],
        exchange,
        candle_manager: CandleManager | None = None,
    ) -> list[SignalCandidate]:
        """Re-check all queued entries against current consensus.

        Returns signals ready to graduate (consensus changed or direction flipped).
        Removes completed/expired entries from the queue.
        """
        if not self.queue:
            return []

        cm = candle_manager or self.candles
        ready: list[SignalCandidate] = []
        now = datetime.now(timezone.utc)

        for symbol, entry in list(self.queue.items()):
            try:
                # Check expiry
                if now >= entry.expires_at:
                    logger.info(
                        "consensus_expired",
                        symbol=symbol,
                        check_count=entry.check_count,
                        original_score=entry.original_score,
                        duration_seconds=round((now - entry.added_at).total_seconds(), 0),
                    )
                    del self.queue[symbol]
                    continue

                entry.check_count += 1
                entry.last_checked = now

                # Re-compute consensus with current portfolio state
                new_result = await self.compute_consensus(
                    signal=entry.signal,
                    open_positions=open_positions,
                    exchange=exchange,
                    candle_manager=cm,
                )

                if not new_result.applied or new_result.penalty == 0:
                    # Consensus no longer disagrees — graduate with original score
                    graduated = entry.signal
                    graduated.score = entry.original_score
                    graduated.reasons.append(
                        f"Consensus-graduated after {entry.check_count} checks "
                        f"({(now - entry.added_at).total_seconds():.0f}s) — "
                        f"consensus cleared"
                    )
                    ready.append(graduated)
                    del self.queue[symbol]
                    logger.info(
                        "consensus_graduated",
                        symbol=symbol,
                        direction=graduated.direction,
                        score=graduated.score,
                        check_count=entry.check_count,
                        reason="consensus_cleared",
                    )
                    continue

                # Check if consensus now agrees with the OPPOSITE direction
                # e.g. signal was "bullish" but portfolio is now "short" and BTC is "bearish"
                # → flip to "bearish" and graduate
                original_dir = entry.signal.direction
                if original_dir and new_result.portfolio_bias not in ("neutral", "insufficient"):
                    opposite_dir = "bearish" if original_dir == "bullish" else "bullish"
                    # Check if the opposite direction would NOT be penalized
                    opp_portfolio_agrees = (
                        (opposite_dir == "bullish" and new_result.portfolio_bias == "long")
                        or (opposite_dir == "bearish" and new_result.portfolio_bias == "short")
                    )
                    opp_btc_agrees = (
                        new_result.btc_trend == "ranging"
                        or (opposite_dir == "bullish" and new_result.btc_trend == "bullish")
                        or (opposite_dir == "bearish" and new_result.btc_trend == "bearish")
                    )

                    if opp_portfolio_agrees and opp_btc_agrees:
                        # Flip direction and graduate
                        graduated = entry.signal
                        graduated.direction = opposite_dir
                        graduated.score = entry.original_score
                        graduated.reasons.append(
                            f"Consensus-flipped: {original_dir}→{opposite_dir} after "
                            f"{entry.check_count} checks — portfolio={new_result.portfolio_bias}, "
                            f"btc={new_result.btc_trend}"
                        )
                        ready.append(graduated)
                        del self.queue[symbol]
                        logger.info(
                            "consensus_graduated_flipped",
                            symbol=symbol,
                            original_direction=original_dir,
                            new_direction=opposite_dir,
                            score=graduated.score,
                            check_count=entry.check_count,
                        )
                        continue

            except Exception as e:
                logger.warning(
                    "consensus_check_failed",
                    symbol=symbol,
                    error=str(e)[:100],
                )

        return ready

    def get_queued_symbols(self) -> set[str]:
        """Return symbols currently in the consensus monitor."""
        return set(self.queue.keys())

    # ── Internal Helpers ───────────────────────────────────────

    async def _compute_portfolio_bias(
        self,
        open_positions: dict[str, Position],
        exchange,
    ) -> tuple[str, int, int, int, int, list[str]]:
        """Compute portfolio direction bias from profitable positions.

        Returns (bias, profitable_longs, profitable_shorts, losing_longs, losing_shorts, reasons).
        """
        reasons: list[str] = []

        # Batch fetch live prices
        symbols = list(open_positions.keys())
        try:
            tickers = await exchange.fetch_tickers(symbols)
        except Exception as e:
            logger.warning("consensus_ticker_fetch_failed", error=str(e)[:100])
            return "insufficient", 0, 0, 0, 0, ["Price fetch failed"]

        profitable_longs = 0
        profitable_shorts = 0
        losing_longs = 0
        losing_shorts = 0

        threshold = self.config.consensus_profitable_threshold

        for symbol, pos in open_positions.items():
            ticker = tickers.get(symbol) if tickers else None
            if not ticker or not ticker.get("last"):
                continue

            live_price = float(ticker["last"])
            if pos.direction == "long":
                unrealized = (live_price - pos.entry_price) * pos.quantity
            else:
                unrealized = (pos.entry_price - live_price) * pos.quantity

            if unrealized > threshold:
                if pos.direction == "long":
                    profitable_longs += 1
                else:
                    profitable_shorts += 1
            else:
                if pos.direction == "long":
                    losing_longs += 1
                else:
                    losing_shorts += 1

        # Determine bias from profitable positions only
        if profitable_longs > profitable_shorts:
            bias = "long"
        elif profitable_shorts > profitable_longs:
            bias = "short"
        else:
            bias = "neutral"

        reasons.append(
            f"Portfolio: {profitable_longs} profitable longs, {profitable_shorts} profitable shorts, "
            f"{losing_longs} losing longs, {losing_shorts} losing shorts → bias={bias}"
        )

        return bias, profitable_longs, profitable_shorts, losing_longs, losing_shorts, reasons

    async def _get_btc_trend(self, candle_manager: CandleManager) -> str:
        """Get BTC/USDT 4H trend direction (cached for 2 minutes)."""
        now = datetime.now(timezone.utc)

        # Return cached value if fresh
        if (
            self._btc_trend_cache is not None
            and self._btc_trend_cache_time is not None
            and (now - self._btc_trend_cache_time).total_seconds() < self._btc_cache_ttl_seconds
        ):
            return self._btc_trend_cache

        try:
            btc_candles = await candle_manager.get_candles("BTC/USDT", "4h", limit=200)
            if btc_candles is None or btc_candles.empty or len(btc_candles) < 50:
                self._btc_trend_cache = "ranging"
                self._btc_trend_cache_time = now
                return "ranging"

            ms_result = self._ms_analyzer.analyze(btc_candles, timeframe="4h")
            self._btc_trend_cache = ms_result.trend
            self._btc_trend_cache_time = now
            return ms_result.trend
        except Exception as e:
            logger.warning("consensus_btc_trend_failed", error=str(e)[:100])
            self._btc_trend_cache = "ranging"
            self._btc_trend_cache_time = now
            return "ranging"

    # ── State Persistence ──────────────────────────────────────

    def get_state(self) -> dict:
        """Serialize consensus monitor state for DB persistence."""
        entries_data = {}
        for sym, entry in self.queue.items():
            entries_data[sym] = {
                "symbol": entry.symbol,
                "added_at": entry.added_at.isoformat(),
                "expires_at": entry.expires_at.isoformat(),
                "original_score": entry.original_score,
                "adjusted_score": entry.adjusted_score,
                "penalty": entry.consensus_result.penalty,
                "portfolio_bias": entry.consensus_result.portfolio_bias,
                "btc_trend": entry.consensus_result.btc_trend,
                "disagreement_reasons": entry.consensus_result.disagreement_reasons,
                "check_count": entry.check_count,
                "direction": entry.signal.direction,
                "score": entry.signal.score,
                "components": entry.signal.components,
            }
        return {
            "entries": entries_data,
            "total_queued": len(entries_data),
        }

    def restore_state(self, data: dict) -> None:
        """Restore consensus monitor state from DB.

        Like EntryRefiner, we create minimal placeholder signals. Restored entries
        will either graduate from fresh consensus checks or expire naturally.
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

                # Minimal placeholder signal
                placeholder = SignalCandidate(
                    score=entry_data.get("score", 0),
                    direction=entry_data.get("direction"),
                    symbol=sym,
                    entry_price=0,
                    components=entry_data.get("components", {}),
                )

                # Minimal placeholder consensus result
                cr = ConsensusResult(
                    penalty=entry_data.get("penalty", 0),
                    portfolio_bias=entry_data.get("portfolio_bias", "neutral"),
                    btc_trend=entry_data.get("btc_trend", "ranging"),
                    signal_direction=entry_data.get("direction", ""),
                    disagreement_reasons=entry_data.get("disagreement_reasons", []),
                    applied=True,
                )

                entry = ConsensusEntry(
                    symbol=sym,
                    signal=placeholder,
                    original_score=entry_data.get("original_score", 0),
                    adjusted_score=entry_data.get("adjusted_score", 0),
                    consensus_result=cr,
                    added_at=datetime.fromisoformat(entry_data["added_at"]),
                    expires_at=expires_at,
                    check_count=entry_data.get("check_count", 0),
                )
                self.queue[sym] = entry
                restored += 1
            except Exception as e:
                logger.warning(
                    "consensus_restore_failed",
                    symbol=sym,
                    error=str(e),
                )

        if restored > 0 or expired > 0:
            logger.info(
                "consensus_restored",
                restored=restored,
                expired_on_restore=expired,
            )

    def stop(self) -> None:
        """Clear the queue (for reset)."""
        self.queue.clear()
