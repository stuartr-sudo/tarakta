"""Market-level cross-reference filters.

Provides three layers of filtering that operate on the *list* of signals
after the scanner has scored them, but before individual entry decisions:

1. **BTC Macro Gate** — hard block: only allow trades that align with
   BTC's macro trend (4H structure + EMA50).  Bearish BTC → no longs.

2. **Market Breadth Filter** — if >N% of qualifying signals point in one
   direction, block the minority direction (the contrarian is likely a trap).

3. **Funding Rate Gate** — block signals where the crowd is already heavily
   positioned in the same direction (extreme funding = crowded trade).

4. **Correlation Clustering** — limit concurrent positions in correlated
   assets (e.g., max 2 L1 longs at once).

5. **Signal Persistence** — signals must appear in 2+ consecutive scan
   cycles before being allowed to enter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from src.exchange.models import SignalCandidate
from src.strategy.market_structure import MarketStructureAnalyzer
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Correlation Clusters ────────────────────────────────────────────────
# Coins that move together.  Max 2 positions per cluster.
CORRELATION_CLUSTERS: dict[str, str] = {
    # BTC-correlated majors
    "BTC": "btc_major", "LTC": "btc_major", "BCH": "btc_major", "ETC": "btc_major",
    # ETH ecosystem
    "ETH": "eth_eco", "OP": "eth_eco", "ARB": "eth_eco", "STRK": "eth_eco",
    "ZK": "eth_eco", "IMX": "eth_eco", "MANTA": "eth_eco", "BLAST": "eth_eco",
    "MATIC": "eth_eco", "POL": "eth_eco", "MNT": "eth_eco", "METIS": "eth_eco",
    # SOL ecosystem
    "SOL": "sol_eco", "JUP": "sol_eco", "JTO": "sol_eco", "RAY": "sol_eco",
    "PYTH": "sol_eco", "BONK": "sol_eco", "WIF": "sol_eco",
    # Alt L1s
    "ADA": "alt_l1", "AVAX": "alt_l1", "DOT": "alt_l1", "ATOM": "alt_l1",
    "NEAR": "alt_l1", "SUI": "alt_l1", "APT": "alt_l1", "SEI": "alt_l1",
    "TIA": "alt_l1", "INJ": "alt_l1", "FTM": "alt_l1", "ALGO": "alt_l1",
    "EGLD": "alt_l1", "HBAR": "alt_l1",
    # DeFi blue chips
    "UNI": "defi", "AAVE": "defi", "MKR": "defi", "SNX": "defi",
    "COMP": "defi", "CRV": "defi", "DYDX": "defi", "GMX": "defi",
    "PENDLE": "defi", "LDO": "defi", "1INCH": "defi",
    # Meme coins
    "DOGE": "meme", "SHIB": "meme", "PEPE": "meme", "FLOKI": "meme",
    # AI / Compute
    "RENDER": "ai_compute", "FET": "ai_compute", "AGIX": "ai_compute",
    "TAO": "ai_compute", "AKT": "ai_compute", "AR": "ai_compute",
    # Gaming
    "AXS": "gaming", "SAND": "gaming", "MANA": "gaming", "GALA": "gaming",
    "ENJ": "gaming", "RONIN": "gaming", "PIXEL": "gaming",
    # Oracles / Infra
    "LINK": "oracle_infra", "GRT": "oracle_infra", "API3": "oracle_infra",
    # Exchange tokens
    "BNB": "exchange_token", "OKB": "exchange_token", "CRO": "exchange_token",
}


@dataclass
class MarketFilterResult:
    """Summary of all market-level filters applied."""

    btc_trend: str = "ranging"  # "bullish", "bearish", "ranging"
    breadth_direction: str | None = None  # majority direction
    breadth_ratio: float = 0.5  # 0-1, fraction in majority direction
    signals_before: int = 0
    signals_after: int = 0
    blocked_by_btc_gate: list[str] = field(default_factory=list)
    blocked_by_breadth: list[str] = field(default_factory=list)
    blocked_by_funding: list[str] = field(default_factory=list)
    blocked_by_persistence: list[str] = field(default_factory=list)


class MarketFilter:
    """Cross-reference filter layer applied after scanner scoring."""

    def __init__(
        self,
        btc_macro_gate_enabled: bool = True,
        market_breadth_enabled: bool = True,
        market_breadth_threshold: float = 0.70,
        funding_gate_enabled: bool = True,
        funding_gate_threshold: float = 0.0005,
        signal_persistence_scans: int = 2,
        max_per_correlation_cluster: int = 2,
    ) -> None:
        self.btc_macro_gate_enabled = btc_macro_gate_enabled
        self.market_breadth_enabled = market_breadth_enabled
        self.breadth_threshold = market_breadth_threshold
        self.funding_gate_enabled = funding_gate_enabled
        self.funding_gate_threshold = funding_gate_threshold
        self.persistence_scans = signal_persistence_scans
        self.max_per_cluster = max_per_correlation_cluster

        self._ms_analyzer = MarketStructureAnalyzer()
        self._btc_trend_cache: tuple[str, datetime] | None = None
        self._btc_cache_ttl = 120  # 2 minutes

        # Signal persistence: symbol → (direction, consecutive_count, last_seen_cycle)
        self._signal_history: dict[str, tuple[str, int, int]] = {}

    # ──────────────────────────────────────────────────────────────────
    # 1. BTC Macro Gate
    # ──────────────────────────────────────────────────────────────────

    async def get_btc_trend(self, candle_manager) -> str:
        """Determine BTC macro trend from 4H candles.

        Uses structure analysis + EMA50 cross-check.
        Cached for 2 minutes to avoid redundant API calls.
        """
        now = datetime.now(timezone.utc)
        if self._btc_trend_cache:
            cached_trend, cached_at = self._btc_trend_cache
            if (now - cached_at).total_seconds() < self._btc_cache_ttl:
                return cached_trend

        try:
            btc_candles = await candle_manager.get_candles(
                "BTC/USDT:USDT", "4h", limit=200,
            )
            if btc_candles is None or btc_candles.empty:
                # Fallback: try without :USDT suffix
                btc_candles = await candle_manager.get_candles(
                    "BTC/USDT", "4h", limit=200,
                )

            if btc_candles is None or len(btc_candles) < 50:
                return "ranging"

            ms_result = self._ms_analyzer.analyze(btc_candles, timeframe="4h")
            trend = ms_result.trend

            # Guard 1: structure must be confident
            if trend != "ranging" and ms_result.structure_strength < 0.6:
                trend = "ranging"

            # Guard 2: price must sit on correct side of 50 EMA
            if trend != "ranging":
                ema_50 = btc_candles["close"].astype(float).ewm(
                    span=50, adjust=False,
                ).mean()
                current_price = float(btc_candles["close"].iloc[-1])
                current_ema = float(ema_50.iloc[-1])

                if trend == "bullish" and current_price <= current_ema:
                    trend = "ranging"
                elif trend == "bearish" and current_price > current_ema:
                    trend = "ranging"

            self._btc_trend_cache = (trend, now)
            logger.info("btc_macro_trend", trend=trend)
            return trend

        except Exception as e:
            logger.warning("btc_trend_fetch_failed", error=str(e))
            return "ranging"

    def apply_btc_gate(
        self, signals: list[SignalCandidate], btc_trend: str,
    ) -> tuple[list[SignalCandidate], list[str]]:
        """Hard gate: block signals that oppose BTC macro trend.

        - BTC bearish → block all bullish signals
        - BTC bullish → block all bearish signals
        - BTC ranging → allow everything
        """
        if not self.btc_macro_gate_enabled or btc_trend == "ranging":
            return signals, []

        blocked: list[str] = []
        passed: list[SignalCandidate] = []

        for sig in signals:
            if btc_trend == "bearish" and sig.direction == "bullish":
                blocked.append(sig.symbol)
                sig.reasons.append(
                    f"BLOCKED:btc_macro_gate (BTC bearish, signal bullish)"
                )
                logger.info(
                    "btc_gate_blocked",
                    symbol=sig.symbol,
                    signal_dir=sig.direction,
                    btc_trend=btc_trend,
                )
            elif btc_trend == "bullish" and sig.direction == "bearish":
                blocked.append(sig.symbol)
                sig.reasons.append(
                    f"BLOCKED:btc_macro_gate (BTC bullish, signal bearish)"
                )
                logger.info(
                    "btc_gate_blocked",
                    symbol=sig.symbol,
                    signal_dir=sig.direction,
                    btc_trend=btc_trend,
                )
            else:
                passed.append(sig)

        return passed, blocked

    # ──────────────────────────────────────────────────────────────────
    # 2. Market Breadth Filter
    # ──────────────────────────────────────────────────────────────────

    def apply_breadth_filter(
        self, signals: list[SignalCandidate],
    ) -> tuple[list[SignalCandidate], list[str], str | None, float]:
        """Block minority-direction signals when breadth is strongly directional.

        If >=70% of qualifying signals are bearish, block all bullish signals
        (and vice versa).  The minority direction is likely a trap.
        """
        if not self.market_breadth_enabled or len(signals) < 3:
            return signals, [], None, 0.5

        bullish = sum(1 for s in signals if s.direction == "bullish")
        bearish = sum(1 for s in signals if s.direction == "bearish")
        total = bullish + bearish
        if total == 0:
            return signals, [], None, 0.5

        bull_ratio = bullish / total
        bear_ratio = bearish / total

        breadth_dir: str | None = None
        breadth_ratio = 0.5

        if bear_ratio >= self.breadth_threshold:
            breadth_dir = "bearish"
            breadth_ratio = bear_ratio
        elif bull_ratio >= self.breadth_threshold:
            breadth_dir = "bullish"
            breadth_ratio = bull_ratio

        if breadth_dir is None:
            return signals, [], None, max(bull_ratio, bear_ratio)

        blocked: list[str] = []
        passed: list[SignalCandidate] = []
        minority = "bullish" if breadth_dir == "bearish" else "bearish"

        for sig in signals:
            if sig.direction == minority:
                blocked.append(sig.symbol)
                sig.reasons.append(
                    f"BLOCKED:breadth_filter ({breadth_dir} "
                    f"{breadth_ratio:.0%}, {sig.direction} is minority)"
                )
                logger.info(
                    "breadth_filter_blocked",
                    symbol=sig.symbol,
                    signal_dir=sig.direction,
                    breadth=breadth_dir,
                    ratio=f"{breadth_ratio:.0%}",
                )
            else:
                passed.append(sig)

        return passed, blocked, breadth_dir, breadth_ratio

    # ──────────────────────────────────────────────────────────────────
    # 3. Funding Rate Gate
    # ──────────────────────────────────────────────────────────────────

    def check_funding_gate(
        self, signal: SignalCandidate,
    ) -> bool:
        """Block if signal direction matches extreme crowd positioning.

        Returns True if signal should be BLOCKED.

        Uses leverage_profile attached during scanner enrichment.
        """
        if not self.funding_gate_enabled:
            return False

        profile = signal.leverage_profile
        if profile is None:
            return False

        funding = abs(profile.funding_rate) if hasattr(profile, "funding_rate") else 0
        if funding < self.funding_gate_threshold:
            return False

        # Extreme funding: check if signal goes with the crowd
        crowded = profile.crowded_side
        if crowded is None:
            return False

        # If longs are crowded (paying) and signal is bullish → block
        # If shorts are crowded (paying) and signal is bearish → block
        if (crowded == "long" and signal.direction == "bullish") or \
           (crowded == "short" and signal.direction == "bearish"):
            signal.reasons.append(
                f"BLOCKED:funding_gate ({crowded}s crowded, "
                f"funding={funding:.4%}, signal {signal.direction})"
            )
            logger.info(
                "funding_gate_blocked",
                symbol=signal.symbol,
                crowded=crowded,
                funding=f"{funding:.4%}",
                signal_dir=signal.direction,
            )
            return True

        return False

    # ──────────────────────────────────────────────────────────────────
    # 4. Signal Persistence
    # ──────────────────────────────────────────────────────────────────

    def update_signal_history(
        self, signals: list[SignalCandidate], scan_cycle: int,
    ) -> list[SignalCandidate]:
        """Track signals across scan cycles.  Only allow signals that
        have appeared in N consecutive scans (same symbol + direction).

        Returns only the signals that have persisted long enough.
        """
        if self.persistence_scans <= 1:
            return signals  # Disabled

        confirmed: list[SignalCandidate] = []
        blocked: list[str] = []

        for sig in signals:
            key = sig.symbol
            direction = sig.direction or "none"

            if key in self._signal_history:
                prev_dir, count, last_cycle = self._signal_history[key]
                if prev_dir == direction and last_cycle == scan_cycle - 1:
                    # Same direction, consecutive scan → increment
                    new_count = count + 1
                    self._signal_history[key] = (direction, new_count, scan_cycle)
                    if new_count >= self.persistence_scans:
                        confirmed.append(sig)
                        sig.reasons.append(
                            f"signal_confirmed: persisted {new_count} scans"
                        )
                        logger.info(
                            "signal_persistence_confirmed",
                            symbol=sig.symbol,
                            direction=direction,
                            scans=new_count,
                        )
                    else:
                        blocked.append(sig.symbol)
                        logger.info(
                            "signal_persistence_pending",
                            symbol=sig.symbol,
                            direction=direction,
                            scans=new_count,
                            required=self.persistence_scans,
                        )
                else:
                    # Direction changed or non-consecutive → reset
                    self._signal_history[key] = (direction, 1, scan_cycle)
                    blocked.append(sig.symbol)
            else:
                # First sighting
                self._signal_history[key] = (direction, 1, scan_cycle)
                blocked.append(sig.symbol)

        # Clean stale entries (not seen in last 5 cycles)
        stale = [
            k for k, (_, _, last) in self._signal_history.items()
            if scan_cycle - last > 5
        ]
        for k in stale:
            del self._signal_history[k]

        if blocked:
            logger.info(
                "signal_persistence_blocked",
                blocked=blocked,
                confirmed=[s.symbol for s in confirmed],
            )

        return confirmed

    # ──────────────────────────────────────────────────────────────────
    # 5. Correlation Clustering
    # ──────────────────────────────────────────────────────────────────

    def filter_by_correlation(
        self,
        signals: list[SignalCandidate],
        open_positions: dict,
    ) -> list[SignalCandidate]:
        """Limit positions per correlation cluster.

        Counts existing open positions per cluster and blocks new signals
        that would exceed the cluster limit.
        """
        # Count existing positions per cluster
        cluster_counts: dict[str, int] = {}
        for symbol in open_positions:
            base = symbol.split("/")[0] if "/" in symbol else symbol
            cluster = CORRELATION_CLUSTERS.get(base, f"other_{base}")
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

        passed: list[SignalCandidate] = []
        for sig in signals:
            base = sig.symbol.split("/")[0] if "/" in sig.symbol else sig.symbol
            cluster = CORRELATION_CLUSTERS.get(base, f"other_{base}")
            current = cluster_counts.get(cluster, 0)

            if self.max_per_cluster > 0 and current >= self.max_per_cluster:
                sig.reasons.append(
                    f"BLOCKED:correlation_cluster ({cluster} has "
                    f"{current}/{self.max_per_cluster} positions)"
                )
                logger.info(
                    "correlation_cluster_blocked",
                    symbol=sig.symbol,
                    cluster=cluster,
                    current=current,
                    max=self.max_per_cluster,
                )
                continue

            # Reserve the slot for this signal
            cluster_counts[cluster] = current + 1
            passed.append(sig)

        return passed

    # ──────────────────────────────────────────────────────────────────
    # Full pipeline
    # ──────────────────────────────────────────────────────────────────

    async def apply_all(
        self,
        signals: list[SignalCandidate],
        candle_manager,
        scan_cycle: int,
        open_positions: dict | None = None,
    ) -> tuple[list[SignalCandidate], MarketFilterResult]:
        """Apply all market-level filters in sequence.

        Order: BTC gate → breadth → persistence → correlation.
        Funding gate is applied per-signal later (needs leverage data).
        """
        result = MarketFilterResult(signals_before=len(signals))

        # 1. BTC Macro Gate
        btc_trend = await self.get_btc_trend(candle_manager)
        result.btc_trend = btc_trend
        signals, blocked_btc = self.apply_btc_gate(signals, btc_trend)
        result.blocked_by_btc_gate = blocked_btc

        # 2. Market Breadth Filter
        signals, blocked_breadth, breadth_dir, breadth_ratio = (
            self.apply_breadth_filter(signals)
        )
        result.blocked_by_breadth = blocked_breadth
        result.breadth_direction = breadth_dir
        result.breadth_ratio = breadth_ratio

        # 3. Signal Persistence
        signals = self.update_signal_history(signals, scan_cycle)
        result.blocked_by_persistence = [
            s for s in result.blocked_by_btc_gate  # placeholder
        ]

        # 4. Correlation Clustering (needs open positions)
        if open_positions is not None:
            signals = self.filter_by_correlation(signals, open_positions)

        result.signals_after = len(signals)

        logger.info(
            "market_filter_summary",
            btc_trend=btc_trend,
            breadth=breadth_dir,
            breadth_ratio=f"{breadth_ratio:.0%}" if breadth_dir else "mixed",
            before=result.signals_before,
            after=result.signals_after,
            blocked_btc=len(blocked_btc),
            blocked_breadth=len(blocked_breadth),
        )

        return signals, result
