from __future__ import annotations

from src.exchange.models import (
    FVGResult,
    LiquidityResult,
    MarketStructureResult,
    OrderBlockResult,
    SignalCandidate,
)
from src.strategy.premium_discount import PremiumDiscountScore
from src.strategy.regime import MarketRegime
from src.strategy.volume import VolumeResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Weight allocation (total = 100)
# Advanced SMC scoring: volume, displacement, and premium/discount zones
# are first-class citizens alongside structure and POI.
WEIGHTS = {
    "htf_trend": 20,           # 4H/1D alignment
    "market_structure": 15,    # 1H BOS/CHoCH
    "order_block": 15,         # 15m/1H OB proximity
    "fvg": 10,                 # 15m/1H FVG proximity
    "liquidity_sweep": 10,     # 15m/1H sweep confirmation
    "volume": 15,              # Displacement + RVOL + volume trend
    "premium_discount": 10,    # Correct zone + OTE alignment
    "rr_bonus": 5,             # Risk/Reward (added later)
}


class ConfluenceEngine:
    """
    Multi-timeframe scoring engine that combines all strategy module outputs.

    Advanced SMC Scoring System (0-100):
      HTF Trend Alignment (4H/1D):         20 points
      Market Structure (1H BOS/CHoCH):     15 points
      Order Block Proximity (15m/1H):      15 points
      Volume/Displacement Confirmation:    15 points
      Fair Value Gap (15m/1H):             10 points
      Liquidity Sweep (15m/1H):            10 points
      Premium/Discount Zone:               10 points
      Risk/Reward Ratio Bonus:              5 points (added later)

    This integrates ICT/SMC advanced concepts:
    - Volume confirms institutional participation (displacement candles,
      elevated RVOL, increasing volume trend on HTF).
    - Premium/discount zones ensure entries align with smart money —
      buying in discount for longs, selling in premium for shorts.
    - OB and FVG scoring is boosted when accompanied by volume.
    """

    def __init__(self, entry_threshold: float = 65.0) -> None:
        self.entry_threshold = entry_threshold
        self._weights: dict[str, int] = dict(WEIGHTS)

    def update_weights(self, weights: dict[str, int]) -> None:
        """Update scoring weights (called by DynamicWeightOptimizer)."""
        self._weights = dict(weights)

    def score_signal(
        self,
        symbol: str,
        current_price: float,
        ms_results: dict[str, MarketStructureResult],
        liq_results: dict[str, LiquidityResult],
        ob_results: dict[str, OrderBlockResult],
        fvg_results: dict[str, FVGResult],
        volume_result: VolumeResult | None = None,
        pd_result: PremiumDiscountScore | None = None,
        regime: MarketRegime | None = None,
    ) -> SignalCandidate:
        score = 0.0
        reasons: list[str] = []
        components: dict[str, float] = {}

        # --- HTF Trend (20 pts) ---
        htf_score, direction, htf_reasons = self._score_htf_trend(ms_results)
        score += htf_score
        components["htf_trend"] = htf_score
        reasons.extend(htf_reasons)

        if direction is None:
            return SignalCandidate(
                score=0,
                direction=None,
                reasons=["No HTF directional bias"],
                symbol=symbol,
                entry_price=current_price,
                components=components,
            )

        # --- Market Structure on 1H (15 pts) ---
        ms_score, ms_reasons = self._score_market_structure(ms_results, direction)
        score += ms_score
        components["market_structure"] = ms_score
        reasons.extend(ms_reasons)

        # --- Order Block (15 pts) ---
        ob_score, ob_reasons, ob_context = self._score_order_blocks(
            ob_results, current_price, direction
        )
        score += ob_score
        components["order_block"] = ob_score
        reasons.extend(ob_reasons)

        # --- Fair Value Gap (10 pts) ---
        fvg_score, fvg_reasons, fvg_context = self._score_fvg(
            fvg_results, current_price, direction
        )
        score += fvg_score
        components["fvg"] = fvg_score
        reasons.extend(fvg_reasons)

        # --- Liquidity Sweep (10 pts) ---
        liq_score, liq_reasons = self._score_liquidity(liq_results, direction)
        score += liq_score
        components["liquidity_sweep"] = liq_score
        reasons.extend(liq_reasons)

        # --- Volume / Displacement (15 pts) ---
        if volume_result is not None:
            vol_score = min(volume_result.overall_volume_score, self._weights["volume"])
            score += vol_score
            components["volume"] = vol_score
            reasons.extend(volume_result.reasons)
        else:
            components["volume"] = 0

        # --- Premium/Discount Zone (10 pts) ---
        if pd_result is not None:
            pd_score = min(pd_result.score, self._weights["premium_discount"])
            score += pd_score
            components["premium_discount"] = pd_score
            reasons.extend(pd_result.reasons)
        else:
            components["premium_discount"] = 0

        # R:R bonus (5 pts) is added later in execution when SL/TP are calculated

        # Collect key levels for SL/TP calculation
        key_levels = {}
        for tf, ms in ms_results.items():
            if ms.key_levels.get("swing_high"):
                key_levels[f"{tf}_swing_high"] = ms.key_levels["swing_high"]
            if ms.key_levels.get("swing_low"):
                key_levels[f"{tf}_swing_low"] = ms.key_levels["swing_low"]

        # Add premium/discount equilibrium as a key level
        if pd_result is not None:
            for tf, r in pd_result.results.items():
                if r.equilibrium > 0:
                    key_levels[f"{tf}_equilibrium"] = r.equilibrium

        return SignalCandidate(
            score=score,
            direction=direction,
            reasons=reasons,
            symbol=symbol,
            entry_price=current_price,
            ob_context=ob_context,
            fvg_context=fvg_context,
            key_levels=key_levels,
            components=components,
        )

    def _score_htf_trend(
        self, ms_results: dict[str, MarketStructureResult]
    ) -> tuple[float, str | None, list[str]]:
        """Score higher timeframe trend alignment (4H + Daily)."""
        htf_4h = ms_results.get("4h")
        htf_1d = ms_results.get("1d")

        trend_4h = htf_4h.trend if htf_4h else "ranging"
        trend_1d = htf_1d.trend if htf_1d else "ranging"

        reasons: list[str] = []

        if trend_4h == trend_1d and trend_4h != "ranging":
            reasons.append(f"HTF aligned: {trend_4h} (4H + Daily)")
            return self._weights["htf_trend"], trend_4h, reasons

        if trend_4h != "ranging":
            reasons.append(f"4H trend: {trend_4h}, Daily: {trend_1d}")
            return 12, trend_4h, reasons

        if trend_1d != "ranging":
            reasons.append(f"Only Daily trend: {trend_1d}")
            return 8, trend_1d, reasons

        return 0, None, ["No clear HTF trend"]

    def _score_market_structure(
        self, ms_results: dict[str, MarketStructureResult], direction: str
    ) -> tuple[float, list[str]]:
        """Score 1H market structure alignment."""
        ms_1h = ms_results.get("1h")
        if not ms_1h:
            return 0, []

        score = 0.0
        reasons: list[str] = []

        if ms_1h.trend == direction:
            score += 10
            reasons.append(f"1H structure: {ms_1h.trend}")

            # Bonus for CHoCH confirming direction
            if ms_1h.last_choch_direction is not None:
                if (direction == "bullish" and ms_1h.last_choch_direction > 0) or (
                    direction == "bearish" and ms_1h.last_choch_direction < 0
                ):
                    score += 5
                    reasons.append("1H CHoCH confirms direction")
        else:
            # Partial credit: 1H CHoCH in our direction (structure is shifting)
            if ms_1h.last_choch_direction is not None:
                if (direction == "bullish" and ms_1h.last_choch_direction > 0) or (
                    direction == "bearish" and ms_1h.last_choch_direction < 0
                ):
                    score += 8
                    reasons.append(f"1H CHoCH shifting {direction} (structure: {ms_1h.trend})")

        return min(score, self._weights["market_structure"]), reasons

    def _score_order_blocks(
        self,
        ob_results: dict[str, OrderBlockResult],
        current_price: float,
        direction: str,
    ) -> tuple[float, list[str], object]:
        """Score order block proximity."""
        reasons: list[str] = []
        ob_context = None

        for tf in ["15m", "1h"]:
            ob = ob_results.get(tf)
            if not ob:
                continue

            # Price is inside an order block
            if ob.price_in_order_block:
                if (direction == "bullish" and ob.price_in_order_block.direction == "bullish") or (
                    direction == "bearish" and ob.price_in_order_block.direction == "bearish"
                ):
                    reasons.append(f"Price in {direction} OB on {tf}")
                    ob_context = ob.price_in_order_block
                    return self._weights["order_block"], reasons, ob_context

            # Price is near a bullish OB (for long entries)
            if direction == "bullish" and ob.nearest_bullish_ob:
                dist = abs(current_price - ob.nearest_bullish_ob.top) / current_price
                if dist < 0.01:  # within 1%
                    reasons.append(f"Price near bullish OB on {tf} ({dist:.1%})")
                    ob_context = ob.nearest_bullish_ob
                    return 12, reasons, ob_context
                elif dist < 0.03:  # within 3%
                    reasons.append(f"Price approaching bullish OB on {tf} ({dist:.1%})")
                    ob_context = ob.nearest_bullish_ob
                    return 8, reasons, ob_context

            # Price near bearish OB (for short entries)
            if direction == "bearish" and ob.nearest_bearish_ob:
                dist = abs(current_price - ob.nearest_bearish_ob.bottom) / current_price
                if dist < 0.01:
                    reasons.append(f"Price near bearish OB on {tf} ({dist:.1%})")
                    ob_context = ob.nearest_bearish_ob
                    return 12, reasons, ob_context
                elif dist < 0.03:
                    reasons.append(f"Price approaching bearish OB on {tf} ({dist:.1%})")
                    ob_context = ob.nearest_bearish_ob
                    return 8, reasons, ob_context

        return 0, reasons, ob_context

    def _score_fvg(
        self,
        fvg_results: dict[str, FVGResult],
        current_price: float,
        direction: str,
    ) -> tuple[float, list[str], object]:
        """Score fair value gap proximity."""
        reasons: list[str] = []
        fvg_context = None

        for tf in ["15m", "1h"]:
            fvg = fvg_results.get(tf)
            if not fvg:
                continue

            if fvg.price_in_fvg:
                if (direction == "bullish" and fvg.price_in_fvg.direction == "bullish") or (
                    direction == "bearish" and fvg.price_in_fvg.direction == "bearish"
                ):
                    reasons.append(f"Price in {direction} FVG on {tf}")
                    fvg_context = fvg.price_in_fvg
                    return self._weights["fvg"], reasons, fvg_context

            # Near an FVG
            target_fvg = fvg.nearest_bullish_fvg if direction == "bullish" else fvg.nearest_bearish_fvg
            if target_fvg:
                dist = abs(current_price - target_fvg.midpoint) / current_price
                if dist < 0.01:  # within 1%
                    reasons.append(f"Price near {direction} FVG on {tf} ({dist:.1%})")
                    fvg_context = target_fvg
                    return 8, reasons, fvg_context
                elif dist < 0.03:  # within 3%
                    reasons.append(f"Price approaching {direction} FVG on {tf} ({dist:.1%})")
                    fvg_context = target_fvg
                    return 5, reasons, fvg_context

        return 0, reasons, fvg_context

    def _score_liquidity(
        self,
        liq_results: dict[str, LiquidityResult],
        direction: str,
    ) -> tuple[float, list[str]]:
        """Score liquidity sweep events."""
        reasons: list[str] = []

        for tf in ["15m", "1h"]:
            liq = liq_results.get(tf)
            if not liq or not liq.sweep_detected_recently:
                continue

            for sweep in liq.recent_sweeps:
                if (direction == "bullish" and sweep.direction == "bullish_sweep") or (
                    direction == "bearish" and sweep.direction == "bearish_sweep"
                ):
                    reasons.append(f"Liquidity sweep on {tf}: {sweep.direction}")
                    return self._weights["liquidity_sweep"], reasons

        return 0, reasons
