"""Pullback detection for refined entry timing.

After a sweep + displacement, price typically thrusts in the intended
direction, then pulls back before continuing. This module detects
that pullback and validates it as an entry zone.

Key insight: If price NEVER pulls back (V-shape), the setup is likely
failing. The pullback creates the entry — no pullback = no trade.

Retracement thresholds (Fibonacci-inspired):
  < 20%  → "waiting"  — still in thrust, skip, re-check next scan
  20-78% → "optimal"  — valid pullback entry zone
  > 78%  → "failed"   — setup invalidated, reject
"""
from __future__ import annotations

import pandas as pd

from src.exchange.models import PullbackResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

MIN_RETRACEMENT = 0.20
MAX_RETRACEMENT = 0.78


class PullbackAnalyzer:
    """Detects pullback entries after sweep + displacement."""

    def __init__(
        self,
        min_retracement: float = MIN_RETRACEMENT,
        max_retracement: float = MAX_RETRACEMENT,
    ) -> None:
        self.min_retracement = min_retracement
        self.max_retracement = max_retracement

    def analyze(
        self,
        candles_1h: pd.DataFrame,
        displacement_candle_idx: int,
        direction: str,
    ) -> PullbackResult:
        """Check if a valid pullback has occurred after displacement.

        Args:
            candles_1h: Full 1H OHLCV DataFrame.
            displacement_candle_idx: Absolute index of the displacement candle.
            direction: Direction of the displacement ("bullish" or "bearish").

        Returns:
            PullbackResult with detection status and entry info.
        """
        if candles_1h is None or candles_1h.empty:
            return self._no_pullback(0.0, 0.0, 0.0)

        if displacement_candle_idx < 0 or displacement_candle_idx >= len(candles_1h):
            return self._no_pullback(0.0, 0.0, 0.0)

        open_ = candles_1h["open"].astype(float)
        high = candles_1h["high"].astype(float)
        low = candles_1h["low"].astype(float)
        close = candles_1h["close"].astype(float)

        disp_open = float(open_.iloc[displacement_candle_idx])
        current_price = float(close.iloc[-1])

        # Post-displacement candles (from displacement+1 to latest)
        post_start = displacement_candle_idx + 1
        if post_start >= len(candles_1h):
            # Displacement is the most recent candle — no pullback possible yet
            return self._no_pullback(disp_open, current_price, current_price, status="waiting")

        post_disp_high = high.iloc[post_start:]
        post_disp_low = low.iloc[post_start:]

        if direction == "bullish":
            thrust_extreme = float(post_disp_high.max())
            move_size = thrust_extreme - disp_open
            if move_size <= 0:
                return self._no_pullback(disp_open, thrust_extreme, current_price, status="failed")

            retrace_distance = thrust_extreme - current_price
            retracement_pct = retrace_distance / move_size

        else:  # bearish
            thrust_extreme = float(post_disp_low.min())
            move_size = disp_open - thrust_extreme
            if move_size <= 0:
                return self._no_pullback(disp_open, thrust_extreme, current_price, status="failed")

            retrace_distance = current_price - thrust_extreme
            retracement_pct = retrace_distance / move_size

        retracement_pct = max(0.0, min(retracement_pct, 1.5))

        if retracement_pct < self.min_retracement:
            status = "waiting"
            detected = False
        elif retracement_pct > self.max_retracement:
            status = "failed"
            detected = False
        else:
            status = "optimal"
            detected = True

        if detected:
            logger.info(
                "pullback_detected",
                direction=direction,
                retracement=f"{retracement_pct:.1%}",
                displacement_open=disp_open,
                thrust_extreme=thrust_extreme,
                current_price=current_price,
            )

        return PullbackResult(
            pullback_detected=detected,
            retracement_pct=round(retracement_pct, 4),
            displacement_open=disp_open,
            thrust_extreme=thrust_extreme,
            current_price=current_price,
            optimal_entry=current_price,
            pullback_status=status,
        )

    def _no_pullback(
        self,
        disp_open: float,
        thrust_extreme: float,
        current_price: float,
        status: str = "waiting",
    ) -> PullbackResult:
        return PullbackResult(
            pullback_detected=False,
            retracement_pct=0.0,
            displacement_open=disp_open,
            thrust_extreme=thrust_extreme,
            current_price=current_price,
            optimal_entry=current_price,
            pullback_status=status,
        )
