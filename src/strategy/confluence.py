"""Post-Sweep Displacement scoring engine (Trade Travel Chill strategy).

Replaces the old 8-component weighted SMC/ICT confluence system with a
5-component binary checklist. The core insight: retail SMC signals
(CRT, OBs, FVGs) are too well-known and get hunted by market makers. Instead
of entering at those levels, we wait for the sweep to COMPLETE, confirm with
displacement, then enter on the PULLBACK (preferred but not required).

Scoring System (0-100):
  Completed Sweep Detection:      35 points (REQUIRED)
  Post-Sweep Displacement:        25 points (REQUIRED)
  Pullback Confirmed:             10 points (bonus, improves entry)
  HTF Trend Alignment (4H/1D):    15 points (bonus)
  Post-Kill Zone Timing:          15 points (bonus)

Minimum threshold: 60 (requires sweep + displacement at minimum).
"""
from __future__ import annotations

from src.exchange.models import (
    BreakoutResult,
    MarketStructureResult,
    PullbackResult,
    SignalCandidate,
    SweepResult,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Weight allocation (base = 100, max with leverage bonus = 110)
# Trade Travel Chill: sweep + displacement + pullback are required components.
# Leverage alignment is a bonus scored in the scanner after the base score.
WEIGHTS = {
    "sweep_detected": 35,
    "displacement_confirmed": 25,
    "pullback_confirmed": 10,
    "htf_aligned": 15,
    "timing_optimal": 15,
    "leverage_aligned": 10,  # Bonus: scored in scanner._score_leverage, not here
}

# Backward compat alias
POST_SWEEP_WEIGHTS = WEIGHTS

# Breakout scoring weights (separate path, max 90 before leverage bonus)
BREAKOUT_WEIGHTS = {
    "breakout_confirmed": 25,   # Price broke AND held beyond level (REQUIRED)
    "volume_confirmed": 20,     # Elevated volume on breakout (REQUIRED)
    "htf_aligned": 15,          # 4H/1D trend agrees with breakout direction
    "timing_optimal": 10,       # Post-kill zone timing
    "candles_held_bonus": 10,   # Extra candles held = stronger conviction
    "atr_distance_bonus": 10,   # Distance from level in ATR = genuine move
}

BREAKOUT_THRESHOLD = 45  # breakout (25) + volume (20) = 45 minimum
MIN_HOLD = 2  # Minimum candles held to qualify as breakout


class PostSweepEngine:
    """Post-sweep displacement scoring engine.

    Binary checklist approach:
    - sweep_detected (35) + displacement_confirmed (25) = 60 minimum
    - pullback_confirmed (10) = bonus (better entry, not required)
    - htf_aligned (15) + timing_optimal (15) = bonus probability

    Threshold = 60 (requires sweep AND displacement at minimum).
    """

    def __init__(self, entry_threshold: float = 60.0) -> None:
        self.entry_threshold = entry_threshold
        self._weights: dict[str, int] = dict(WEIGHTS)

    def update_weights(self, weights: dict[str, int]) -> None:
        """Update scoring weights (called by DynamicWeightOptimizer)."""
        self._weights = dict(weights)

    def score_signal(
        self,
        symbol: str,
        current_price: float,
        sweep_result: SweepResult,
        displacement_confirmed: bool,
        displacement_direction: str | None,
        htf_direction: str | None,
        in_post_kill_zone: bool,
        ms_results: dict[str, MarketStructureResult],
        pullback_result: PullbackResult | None = None,
    ) -> SignalCandidate:
        """Score a signal using the post-sweep displacement + pullback checklist.

        Args:
            symbol: Trading pair.
            current_price: Current market price.
            sweep_result: Output from SweepDetector.detect().
            displacement_confirmed: Whether displacement candle was detected.
            displacement_direction: Direction of displacement ("bullish"/"bearish").
            htf_direction: HTF trend direction from 4H/1D structure.
            in_post_kill_zone: Whether we're in a post-kill-zone window.
            ms_results: Market structure results for key levels.
            pullback_result: Output from PullbackAnalyzer.analyze().

        Returns:
            SignalCandidate with score, direction, and reasons.
        """
        score = 0.0
        direction: str | None = None
        reasons: list[str] = []
        components: dict[str, float] = {}

        # --- Sweep Detection (35 pts) --- REQUIRED
        if sweep_result.sweep_detected:
            score += self._weights.get("sweep_detected", 35)
            direction = sweep_result.sweep_direction
            reasons.append(
                f"Sweep completed: {sweep_result.sweep_type} "
                f"(depth={sweep_result.sweep_depth:.4f})"
            )
            components["sweep_detected"] = self._weights.get("sweep_detected", 35)
        else:
            # No sweep = no trade
            return SignalCandidate(
                score=0,
                direction=None,
                reasons=["No completed sweep detected"],
                symbol=symbol,
                entry_price=current_price,
                components={"sweep_detected": 0},
            )

        # --- Post-Sweep Displacement (25 pts) --- REQUIRED
        if displacement_confirmed and displacement_direction == direction:
            score += self._weights.get("displacement_confirmed", 25)
            reasons.append(f"Post-sweep displacement confirmed: {direction}")
            components["displacement_confirmed"] = self._weights.get("displacement_confirmed", 25)
        else:
            if displacement_confirmed and displacement_direction != direction:
                reasons.append(
                    f"Displacement direction mismatch: "
                    f"sweep={direction}, displacement={displacement_direction}"
                )
            else:
                reasons.append("No displacement after sweep")
            components["displacement_confirmed"] = 0

            key_levels = self._collect_key_levels(ms_results)
            return SignalCandidate(
                score=score,
                direction=direction,
                reasons=reasons,
                symbol=symbol,
                entry_price=current_price,
                components=components,
                key_levels=key_levels,
            )

        # --- Pullback Confirmation (10 pts) --- BONUS (improves entry, not required)
        if pullback_result is not None and pullback_result.pullback_detected:
            score += self._weights.get("pullback_confirmed", 10)
            reasons.append(
                f"Pullback confirmed: {pullback_result.retracement_pct:.0%} "
                f"retracement ({pullback_result.pullback_status})"
            )
            components["pullback_confirmed"] = self._weights.get("pullback_confirmed", 10)
            # Use the pullback price as entry (better than displacement close)
            current_price = pullback_result.current_price
        else:
            # No pullback — still tradeable, just a less optimal entry
            if pullback_result and pullback_result.pullback_status == "waiting":
                reasons.append("Pullback pending (entering at displacement level)")
            elif pullback_result and pullback_result.pullback_status == "failed":
                reasons.append(
                    f"Pullback too deep ({pullback_result.retracement_pct:.0%}) — entering at displacement level"
                )
            else:
                reasons.append("No pullback detected — entering at displacement level")
            components["pullback_confirmed"] = 0

        # --- HTF Alignment (15 pts) --- PREFERRED
        htf_score, htf_reasons = self._score_htf(htf_direction, direction, ms_results)
        score += htf_score
        components["htf_aligned"] = htf_score
        reasons.extend(htf_reasons)

        # --- Post-Kill Zone Timing (15 pts) --- PREFERRED
        if in_post_kill_zone:
            score += self._weights.get("timing_optimal", 15)
            reasons.append("In post-kill-zone window (manipulation phase complete)")
            components["timing_optimal"] = self._weights.get("timing_optimal", 15)
        else:
            components["timing_optimal"] = 0

        # Collect key levels for SL/TP calculation
        key_levels = self._collect_key_levels(ms_results)

        return SignalCandidate(
            score=score,
            direction=direction,
            reasons=reasons,
            symbol=symbol,
            entry_price=current_price,
            key_levels=key_levels,
            components=components,
        )

    def score_breakout(
        self,
        symbol: str,
        current_price: float,
        breakout_result: BreakoutResult,
        htf_direction: str | None,
        in_post_kill_zone: bool,
        ms_results: dict[str, MarketStructureResult],
    ) -> SignalCandidate:
        """Score a breakout signal (complementary to sweep scoring).

        Breakout Scoring (0-90, threshold 45):
          Breakout confirmed:       25 pts (REQUIRED — price broke + held level)
          Volume confirmed:         20 pts (REQUIRED — institutional participation)
          HTF trend aligned:        15 pts (bonus — 4H/1D agree)
          Post-kill zone timing:    10 pts (bonus — manipulation done)
          Candles held bonus:       10 pts (bonus — more hold = higher conviction)
          ATR distance bonus:       10 pts (bonus — genuine move, not noise)

        Minimum threshold: 45 (breakout + volume).
        """
        score = 0.0
        direction = breakout_result.breakout_direction
        reasons: list[str] = []
        components: dict[str, float] = {}

        if not breakout_result.breakout_detected:
            return SignalCandidate(
                score=0,
                direction=None,
                reasons=["No breakout detected"],
                symbol=symbol,
                entry_price=current_price,
                components={"breakout_confirmed": 0},
            )

        # --- Breakout Confirmed (25 pts) --- REQUIRED
        score += BREAKOUT_WEIGHTS["breakout_confirmed"]
        reasons.append(
            f"Breakout confirmed: {breakout_result.breakout_type} "
            f"({breakout_result.candles_held} candles held, "
            f"{breakout_result.atr_distance:.1f} ATR distance)"
        )
        components["breakout_confirmed"] = BREAKOUT_WEIGHTS["breakout_confirmed"]

        # --- Volume Confirmed (20 pts) --- REQUIRED for threshold
        if breakout_result.volume_confirmed:
            score += BREAKOUT_WEIGHTS["volume_confirmed"]
            reasons.append("Volume confirmed on breakout (>1.5x average)")
            components["volume_confirmed"] = BREAKOUT_WEIGHTS["volume_confirmed"]
        else:
            reasons.append("Low volume breakout (caution)")
            components["volume_confirmed"] = 0

        # --- HTF Alignment (15 pts) --- PREFERRED
        htf_score, htf_reasons = self._score_htf(htf_direction, direction, ms_results)
        score += htf_score
        components["htf_aligned"] = htf_score
        reasons.extend(htf_reasons)

        # --- Post-Kill Zone Timing (10 pts) --- PREFERRED
        if in_post_kill_zone:
            score += BREAKOUT_WEIGHTS["timing_optimal"]
            reasons.append("In post-kill-zone window")
            components["timing_optimal"] = BREAKOUT_WEIGHTS["timing_optimal"]
        else:
            components["timing_optimal"] = 0

        # --- Candles Held Bonus (up to 10 pts) ---
        # 2 candles = 0 bonus, 3 = +3, 4 = +6, 5+ = +10
        hold_bonus = min(
            BREAKOUT_WEIGHTS["candles_held_bonus"],
            max(0, (breakout_result.candles_held - MIN_HOLD) * 3),
        )
        if hold_bonus > 0:
            score += hold_bonus
            reasons.append(
                f"Strong hold: {breakout_result.candles_held} candles beyond level (+{hold_bonus})"
            )
        components["candles_held_bonus"] = hold_bonus

        # --- ATR Distance Bonus (up to 10 pts) ---
        # 0.3 ATR = 0 bonus, 0.5 = +3, 1.0 = +7, 1.5+ = +10
        atr_bonus = min(
            BREAKOUT_WEIGHTS["atr_distance_bonus"],
            max(0, int((breakout_result.atr_distance - 0.3) * 14)),
        )
        if atr_bonus > 0:
            score += atr_bonus
            reasons.append(
                f"Clear breakout: {breakout_result.atr_distance:.1f} ATR from level (+{atr_bonus})"
            )
        components["atr_distance_bonus"] = atr_bonus

        # Collect key levels for SL/TP
        key_levels = self._collect_key_levels(ms_results)

        return SignalCandidate(
            score=score,
            direction=direction,
            reasons=reasons,
            symbol=symbol,
            entry_price=current_price,
            key_levels=key_levels,
            components=components,
        )

    def _score_htf(
        self,
        htf_direction: str | None,
        signal_direction: str,
        ms_results: dict[str, MarketStructureResult],
    ) -> tuple[float, list[str]]:
        """Score higher timeframe trend alignment (4H + Daily)."""
        max_pts = self._weights.get("htf_aligned", 15)
        reasons: list[str] = []

        if htf_direction is None:
            return 0, []

        htf_4h = ms_results.get("4h")
        htf_1d = ms_results.get("1d")
        trend_4h = htf_4h.trend if htf_4h else "ranging"
        trend_1d = htf_1d.trend if htf_1d else "ranging"

        if trend_4h == signal_direction and trend_1d == signal_direction:
            reasons.append(f"HTF aligned: {signal_direction} (4H + Daily)")
            return max_pts, reasons

        if trend_4h == signal_direction:
            reasons.append(f"4H trend: {trend_4h}, Daily: {trend_1d}")
            return round(max_pts * 0.7), reasons

        if trend_1d == signal_direction:
            reasons.append(f"Only Daily trend: {trend_1d}")
            return round(max_pts * 0.5), reasons

        return 0, []

    def _collect_key_levels(
        self, ms_results: dict[str, MarketStructureResult],
    ) -> dict:
        """Collect swing levels from market structure for SL/TP calculation."""
        key_levels: dict = {}
        for tf, ms in ms_results.items():
            if ms.key_levels.get("swing_high"):
                key_levels[f"{tf}_swing_high"] = ms.key_levels["swing_high"]
            if ms.key_levels.get("swing_low"):
                key_levels[f"{tf}_swing_low"] = ms.key_levels["swing_low"]
        return key_levels


# Backward compat alias for imports that use the old class name
ConfluenceEngine = PostSweepEngine
