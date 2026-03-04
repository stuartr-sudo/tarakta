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
# Market Maker Method / CRT scoring: session timing and CRT patterns
# are the primary edge, alongside HTF trend and 1H structure.
WEIGHTS = {
    "htf_trend": 15,           # 4H/1D alignment
    "crt_session": 20,         # 4H CRT pattern + kill zone timing
    "market_structure": 15,    # 1H BOS/CHoCH
    "order_block": 12,         # 1H OB proximity
    "fvg": 8,                  # 1H FVG proximity
    "liquidity_sweep": 10,     # 1H sweep confirmation
    "volume": 10,              # Displacement + RVOL + volume trend
    "premium_discount": 10,    # Correct zone + OTE alignment
}


class ConfluenceEngine:
    """
    Multi-timeframe scoring engine (Market Maker Method / CRT).

    Scoring System (0-100):
      HTF Trend Alignment (4H/1D):         15 points
      CRT + Session/Kill Zone:             20 points
      Market Structure (1H BOS/CHoCH):     15 points
      Order Block Proximity (1H):          12 points
      Volume/Displacement Confirmation:    10 points
      Fair Value Gap (1H):                  8 points
      Liquidity Sweep (1H):               10 points
      Premium/Discount Zone:              10 points

    The CRT + Session component is the primary edge:
    - 4H Candle Range Theory pattern confirms direction
    - Kill zone timing ensures institutional volume
    - Asian range sweep confirms manipulation phase
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
        crt_result: object | None = None,
        session_result: object | None = None,
    ) -> SignalCandidate:
        score = 0.0
        reasons: list[str] = []
        components: dict[str, float] = {}

        # --- HTF Trend (15 pts) ---
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

        # --- CRT + Session (20 pts) ---
        crt_score, crt_reasons = self._score_crt_session(crt_result, session_result, direction)
        score += crt_score
        components["crt_session"] = crt_score
        reasons.extend(crt_reasons)

        # --- Market Structure on 1H (15 pts) ---
        ms_score, ms_reasons = self._score_market_structure(ms_results, direction)
        score += ms_score
        components["market_structure"] = ms_score
        reasons.extend(ms_reasons)

        # --- Order Block (12 pts) ---
        ob_score, ob_reasons, ob_context = self._score_order_blocks(
            ob_results, current_price, direction
        )
        score += ob_score
        components["order_block"] = ob_score
        reasons.extend(ob_reasons)

        # --- Fair Value Gap (8 pts) ---
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

        # --- Volume / Displacement (10 pts) ---
        if volume_result is not None:
            vol_score = min(volume_result.overall_volume_score, self._weights["volume"])
            score += vol_score
            components["volume"] = vol_score
            reasons.extend(volume_result.reasons)
        else:
            components["volume"] = 0

        # --- Premium/Discount Zone (10 pts) ---
        if pd_result is not None:
            pd_score_val = min(pd_result.score, self._weights["premium_discount"])
            score += pd_score_val
            components["premium_discount"] = pd_score_val
            reasons.extend(pd_result.reasons)
        else:
            components["premium_discount"] = 0

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
            return 10, trend_4h, reasons

        if trend_1d != "ranging":
            reasons.append(f"Only Daily trend: {trend_1d}")
            return 7, trend_1d, reasons

        return 0, None, ["No clear HTF trend"]

    def _score_crt_session(
        self,
        crt_result: object | None,
        session_result: object | None,
        direction: str,
    ) -> tuple[float, list[str]]:
        """Score CRT pattern + session/kill zone alignment (up to 20 pts).

        This is the primary edge in the Market Maker Method:
        - CRT pattern on 4H confirms direction (8-12 pts)
        - Kill zone timing ensures institutional participation (+4 pts)
        - Asian range sweep confirms manipulation phase (+4 pts)
        - Outside kill zone: score capped at 8 (discourages off-session entries)
        """
        score = 0.0
        reasons: list[str] = []

        # --- CRT Pattern (0-12 pts) ---
        if crt_result is not None and getattr(crt_result, "crt_detected", False):
            crt_dir = getattr(crt_result, "direction", None)
            crt_strength = getattr(crt_result, "strength", 0.0)
            if crt_dir == direction:
                # Aligned CRT: 8-12 pts based on strength
                pts = 8 + crt_strength * 4  # strength 0-1 maps to 8-12
                score += pts
                reasons.append(f"4H CRT {crt_dir} (strength {crt_strength:.0%})")
            elif crt_dir is not None:
                # CRT exists but against our direction
                reasons.append(f"4H CRT against direction ({crt_dir})")

        # --- Kill Zone (0-4 pts) ---
        if session_result is not None:
            in_kz = getattr(session_result, "in_kill_zone", False)
            kz_name = getattr(session_result, "kill_zone_name", None)
            if in_kz:
                score += 4
                kz_display = "London" if kz_name == "london_kz" else "NY"
                reasons.append(f"In {kz_display} kill zone")
            else:
                # Outside kill zone: cap total CRT+session score at 8
                score = min(score, 8)
                session_name = getattr(session_result, "current_session", "unknown")
                if session_name != "off_hours":
                    reasons.append(f"Session: {session_name} (no kill zone)")

        # --- Asian Range Sweep (0-4 pts) ---
        if session_result is not None:
            sweep_dir = getattr(session_result, "asian_range_swept", None)
            if sweep_dir is not None:
                # "above" sweep = bearish signal, "below" sweep = bullish signal
                sweep_bullish = sweep_dir == "below"
                if (direction == "bullish" and sweep_bullish) or \
                   (direction == "bearish" and not sweep_bullish):
                    score += 4
                    reasons.append(f"Asian range swept {sweep_dir} (confirms {direction})")

        score = min(score, self._weights.get("crt_session", 20))
        return score, reasons

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
        """Score order block proximity (1H only)."""
        reasons: list[str] = []
        ob_context = None

        for tf in ["1h"]:
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
                    return 10, reasons, ob_context
                elif dist < 0.03:  # within 3%
                    reasons.append(f"Price approaching bullish OB on {tf} ({dist:.1%})")
                    ob_context = ob.nearest_bullish_ob
                    return 6, reasons, ob_context

            # Price near bearish OB (for short entries)
            if direction == "bearish" and ob.nearest_bearish_ob:
                dist = abs(current_price - ob.nearest_bearish_ob.bottom) / current_price
                if dist < 0.01:
                    reasons.append(f"Price near bearish OB on {tf} ({dist:.1%})")
                    ob_context = ob.nearest_bearish_ob
                    return 10, reasons, ob_context
                elif dist < 0.03:
                    reasons.append(f"Price approaching bearish OB on {tf} ({dist:.1%})")
                    ob_context = ob.nearest_bearish_ob
                    return 6, reasons, ob_context

        return 0, reasons, ob_context

    def _score_fvg(
        self,
        fvg_results: dict[str, FVGResult],
        current_price: float,
        direction: str,
    ) -> tuple[float, list[str], object]:
        """Score fair value gap proximity (1H only)."""
        reasons: list[str] = []
        fvg_context = None

        for tf in ["1h"]:
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
                    return 6, reasons, fvg_context
                elif dist < 0.03:  # within 3%
                    reasons.append(f"Price approaching {direction} FVG on {tf} ({dist:.1%})")
                    fvg_context = target_fvg
                    return 4, reasons, fvg_context

        return 0, reasons, fvg_context

    def _score_liquidity(
        self,
        liq_results: dict[str, LiquidityResult],
        direction: str,
    ) -> tuple[float, list[str]]:
        """Score liquidity sweep events (1H only)."""
        reasons: list[str] = []

        for tf in ["1h"]:
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
