"""Footprint (order flow) analysis for sweep confirmation.

Fetches recent aggTrades from Binance and checks whether actual buying/selling
pressure supports the expected reversal after a sweep.  This filters out
"trap" sweeps where the sweep is the start of a bigger move, not a reversal.
"""
from __future__ import annotations

from src.exchange.models import FootprintResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FootprintAnalyzer:
    """Analyse recent trade flow to confirm or reject a sweep reversal."""

    def __init__(
        self,
        min_delta_pct: float = 0.10,
        absorption_threshold: float = 0.30,
        min_confidence: float = 0.40,
    ) -> None:
        self.min_delta_pct = min_delta_pct
        self.absorption_threshold = absorption_threshold
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        exchange,
        symbol: str,
        sweep_direction: str,
        sweep_level: float,
        current_price: float,
        trade_limit: int = 1000,
        sweep_oi_usd: float = 0.0,
    ) -> FootprintResult:
        """Run full order flow analysis for a signal.

        Parameters
        ----------
        exchange : BinanceFuturesClient (or any client with ``fetch_trades`` / ``fetch_open_interest``)
        symbol : Trading pair, e.g. ``"ETH/USDT:USDT"``
        sweep_direction : The sweep type — contains ``"low"`` for bullish or ``"high"`` for bearish.
        sweep_level : The price level that was swept.
        current_price : Current market price of the symbol.
        trade_limit : Number of recent trades to fetch (default 1000).
        sweep_oi_usd : Open interest in USD at sweep detection time (0 = unavailable).
        """
        is_bullish = "low" in sweep_direction  # swing_low, asian_low, ny_low etc.

        # 1. Fetch recent trades
        trades = await exchange.fetch_trades(symbol, limit=trade_limit)
        if not trades:
            return self._neutral("no_trades_returned")

        # 2. Build footprint
        buy_vol, sell_vol = self._aggregate_volume(trades)
        total_vol = buy_vol + sell_vol
        if total_vol == 0:
            return self._neutral("zero_volume")

        delta = buy_vol - sell_vol
        delta_pct = delta / total_vol

        # 3. Absorption at sweep level
        absorption = self._calculate_absorption(
            trades, sweep_level, current_price, is_bullish,
        )

        # 4. Cumulative delta direction (accelerating in reversal direction?)
        cum_delta_confirms = self._evaluate_cumulative_delta(trades, is_bullish)

        # 5. OI change comparison (sweep time vs now)
        oi_change_pct = 0.0
        oi_confirms = False
        try:
            oi_data = await exchange.fetch_open_interest(symbol)
            current_oi = oi_data.get("open_interest_usd", 0)

            if sweep_oi_usd > 0 and current_oi > 0:
                oi_change_pct = (current_oi - sweep_oi_usd) / sweep_oi_usd

                # OI rising during sweep = new positions opening = continuation likely (bad for reversal)
                # OI dropping during sweep = positions closing = exhaustion (good for reversal)
                # OI rising + reversal direction delta = fresh conviction (good)
                delta_direction_ok_here = (delta_pct > 0 and is_bullish) or (delta_pct < 0 and not is_bullish)

                if oi_change_pct < -0.005:
                    # OI dropped >0.5% — positions are closing, sweep is exhausting
                    oi_confirms = True
                    logger.info("footprint_oi_exhaustion", symbol=symbol,
                                oi_change=f"{oi_change_pct:+.2%}",
                                sweep_oi=f"${sweep_oi_usd:,.0f}",
                                current_oi=f"${current_oi:,.0f}")
                elif oi_change_pct > 0.005 and delta_direction_ok_here:
                    # OI rising BUT delta supports reversal = fresh conviction entering reversal side
                    oi_confirms = True
                    logger.info("footprint_oi_fresh_conviction", symbol=symbol,
                                oi_change=f"{oi_change_pct:+.2%}",
                                delta_pct=f"{delta_pct:+.2%}")
                elif oi_change_pct > 0.005 and not delta_direction_ok_here:
                    # OI rising AND delta opposes reversal = new positions driving continuation (TRAP)
                    oi_confirms = False
                    logger.info("footprint_oi_trap_detected", symbol=symbol,
                                oi_change=f"{oi_change_pct:+.2%}",
                                delta_pct=f"{delta_pct:+.2%}")
                else:
                    # OI flat — neutral
                    oi_confirms = True
            else:
                # No sweep OI to compare — fail open
                oi_confirms = True
        except Exception:
            oi_confirms = True  # fail open

        # 6. Delta direction check
        delta_direction_ok = (delta_pct > 0 and is_bullish) or (
            delta_pct < 0 and not is_bullish
        )
        delta_strong = abs(delta_pct) >= self.min_delta_pct

        # 7. Confidence scoring
        reasons: list[str] = []

        # Delta component (35%)
        if delta_direction_ok and delta_strong:
            delta_score = 1.0
            reasons.append(f"Delta confirms: {delta_pct:+.1%}")
        elif delta_direction_ok:
            delta_score = abs(delta_pct) / self.min_delta_pct  # partial credit
            reasons.append(f"Delta weak but directional: {delta_pct:+.1%}")
        else:
            delta_score = 0.0
            reasons.append(f"Delta opposes reversal: {delta_pct:+.1%}")

        # Absorption component (25%)
        if absorption >= self.absorption_threshold:
            absorption_score_weight = min(1.0, absorption / 0.8)
            reasons.append(f"Absorption detected: {absorption:.2f}")
        else:
            absorption_score_weight = absorption / self.absorption_threshold * 0.5
            reasons.append(f"Weak absorption: {absorption:.2f}")

        # Cumulative delta component (25%)
        cum_delta_score = 1.0 if cum_delta_confirms else 0.0
        if cum_delta_confirms:
            reasons.append("Cumulative delta accelerating in reversal direction")
        else:
            reasons.append("Cumulative delta not confirming reversal")

        # OI component (15%)
        oi_score = 1.0 if oi_confirms else 0.0
        if oi_change_pct != 0:
            if oi_confirms:
                reasons.append(f"OI change {oi_change_pct:+.2%} supports reversal")
            else:
                reasons.append(f"OI change {oi_change_pct:+.2%} suggests trap (continuation)")
        elif sweep_oi_usd == 0:
            reasons.append("OI comparison unavailable (no sweep baseline)")
        else:
            reasons.append("OI flat since sweep")

        confidence = (
            0.35 * delta_score
            + 0.25 * absorption_score_weight
            + 0.25 * cum_delta_score
            + 0.15 * oi_score
        )

        passed = confidence >= self.min_confidence and delta_direction_ok

        if passed:
            reasons.insert(0, f"PASS (confidence={confidence:.2f})")
        else:
            reasons.insert(0, f"FAIL (confidence={confidence:.2f})")

        return FootprintResult(
            passed=passed,
            confidence=round(confidence, 4),
            delta=round(delta, 4),
            delta_pct=round(delta_pct, 6),
            absorption_score=round(absorption, 4),
            cumulative_delta_confirms=cum_delta_confirms,
            oi_change_pct=round(oi_change_pct, 4),
            oi_confirms=oi_confirms,
            total_volume=round(total_vol, 4),
            buy_volume=round(buy_vol, 4),
            sell_volume=round(sell_vol, 4),
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_volume(trades: list[dict]) -> tuple[float, float]:
        """Sum aggressive buy and sell volume."""
        buy_vol = 0.0
        sell_vol = 0.0
        for t in trades:
            cost = t.get("cost") or (t["price"] * t["amount"])
            if t["side"] == "buy":
                buy_vol += cost
            else:
                sell_vol += cost
        return buy_vol, sell_vol

    def _calculate_absorption(
        self,
        trades: list[dict],
        sweep_level: float,
        current_price: float,
        is_bullish: bool,
    ) -> float:
        """Detect passive absorption near the sweep level.

        Absorption = high volume at the sweep level without price continuing
        through it. This means passive limit orders are absorbing the aggressive
        flow — a sign of genuine demand (for bullish) or supply (for bearish).

        Returns a score between 0.0 and 1.0.
        """
        if sweep_level <= 0:
            return 0.0

        # Define "near sweep" as within 0.3% of the sweep level
        tolerance = sweep_level * 0.003
        low_bound = sweep_level - tolerance
        high_bound = sweep_level + tolerance

        sweep_zone_vol = 0.0
        total_vol = 0.0
        for t in trades:
            cost = t.get("cost") or (t["price"] * t["amount"])
            total_vol += cost
            if low_bound <= t["price"] <= high_bound:
                sweep_zone_vol += cost

        if total_vol == 0:
            return 0.0

        # Volume concentration ratio at sweep level
        vol_ratio = sweep_zone_vol / (total_vol / max(1, len(set(
            round(t["price"], 8) for t in trades
        ))))

        # Did price actually reverse away from the sweep?
        if is_bullish:
            reversed_away = current_price > sweep_level
        else:
            reversed_away = current_price < sweep_level

        if not reversed_away:
            # Price is still at or beyond the sweep — absorption failed
            return min(0.2, vol_ratio * 0.1)

        # Score: how much volume was absorbed at sweep without continuation
        return min(1.0, vol_ratio / 3.0)

    @staticmethod
    def _evaluate_cumulative_delta(trades: list[dict], is_bullish: bool) -> bool:
        """Check if cumulative delta is accelerating in the reversal direction.

        Split trades into two time halves. If the second half's delta is more
        favourable to the reversal than the first half, the flow is confirming.
        """
        if len(trades) < 10:
            return False

        mid = len(trades) // 2
        first_half = trades[:mid]
        second_half = trades[mid:]

        def half_delta(half: list[dict]) -> float:
            buy = sum(
                (t.get("cost") or t["price"] * t["amount"])
                for t in half if t["side"] == "buy"
            )
            sell = sum(
                (t.get("cost") or t["price"] * t["amount"])
                for t in half if t["side"] == "sell"
            )
            return buy - sell

        d1 = half_delta(first_half)
        d2 = half_delta(second_half)

        if is_bullish:
            return d2 > d1  # buying accelerating
        else:
            return d2 < d1  # selling accelerating

    @staticmethod
    def _neutral(reason: str) -> FootprintResult:
        """Return a neutral pass result when analysis can't be performed."""
        return FootprintResult(
            passed=True,
            confidence=0.5,
            delta=0.0,
            delta_pct=0.0,
            absorption_score=0.0,
            cumulative_delta_confirms=False,
            oi_change_pct=0.0,
            oi_confirms=True,
            total_volume=0.0,
            buy_volume=0.0,
            sell_volume=0.0,
            reasons=[f"NEUTRAL ({reason}) — fail open"],
        )
