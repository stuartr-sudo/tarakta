"""Completed liquidity sweep detector for the Trade Travel Chill strategy.

Detects when price has swept through a significant level (Asian range,
swing high/low) and closed back on the other side within the last few
candles. This indicates market makers have grabbed the liquidity and
the real move is about to begin.

A completed sweep is the ENTRY SIGNAL — not the level itself.
"""
from __future__ import annotations

import pandas as pd

from src.exchange.models import SweepResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _empty_result() -> SweepResult:
    return SweepResult(
        sweep_detected=False,
        sweep_direction=None,
        sweep_level=0.0,
        sweep_type=None,
        target_level=0.0,
        sweep_depth=0.0,
    )


class SweepDetector:
    """Detects completed liquidity sweeps on 1H candles.

    A completed sweep = price wicked through a level and closed back
    on the other side. This means market makers have grabbed the
    liquidity and are now likely to drive price in the opposite direction.
    """

    def detect(
        self,
        candles_1h: pd.DataFrame,
        asian_high: float,
        asian_low: float,
        swing_high: float | None = None,
        swing_low: float | None = None,
        lookback: int = 8,
        prefer_direction: str | None = None,
        london_high: float = 0.0,
        london_low: float = 0.0,
        ny_high: float = 0.0,
        ny_low: float = 0.0,
    ) -> SweepResult:
        """Check last `lookback` completed 1H candles for sweep completion.

        Priority order:
        1. Asian range sweep (most reliable — overnight consolidation)
        2. London range sweep (London session extremes)
        3. NY range sweep (NY session extremes)
        4. Swing high/low sweep (structural levels)

        Args:
            candles_1h: 1H OHLCV DataFrame.
            asian_high: Today's Asian session high.
            asian_low: Today's Asian session low.
            swing_high: Recent 1H swing high level.
            swing_low: Recent 1H swing low level.
            lookback: Number of completed candles to check.
            prefer_direction: If set ("bullish"/"bearish"), prefer sweeps
                matching this direction.  Falls back to any sweep if no
                match is found in the window.
            london_high: London session high (07:00-12:00 UTC).
            london_low: London session low (07:00-12:00 UTC).
            ny_high: NY session high (12:00-17:00 UTC).
            ny_low: NY session low (12:00-17:00 UTC).

        Returns:
            SweepResult with detection details. sweep_level is the wick
            extreme (for SL placement), target_level is the opposite side
            liquidity (for TP placement).
        """
        if candles_1h is None or len(candles_1h) < lookback + 2:
            return _empty_result()

        # Build list of levels to check in priority order
        # Priority: Asian > London > NY > Swing (overnight consolidation is most reliable)
        levels: list[tuple[float, str, float]] = []  # (level, type, opposite_target)

        if asian_low > 0 and asian_high > 0 and asian_high > asian_low:
            levels.append((asian_low, "asian_low", asian_high))
            levels.append((asian_high, "asian_high", asian_low))

        if london_low > 0 and london_high > 0 and london_high > london_low:
            levels.append((london_low, "london_low", london_high))
            levels.append((london_high, "london_high", london_low))

        if ny_low > 0 and ny_high > 0 and ny_high > ny_low:
            levels.append((ny_low, "ny_low", ny_high))
            levels.append((ny_high, "ny_high", ny_low))

        if swing_low and swing_low > 0:
            target = swing_high if (swing_high and swing_high > 0) else 0.0
            levels.append((swing_low, "swing_low", target))

        if swing_high and swing_high > 0:
            target = swing_low if (swing_low and swing_low > 0) else 0.0
            levels.append((swing_high, "swing_high", target))

        if not levels:
            return _empty_result()

        high = candles_1h["high"].astype(float)
        low = candles_1h["low"].astype(float)
        close = candles_1h["close"].astype(float)

        # Check last `lookback` completed candles (skip -1 which may be incomplete)
        first_any: SweepResult | None = None

        for offset in range(-2, -2 - lookback, -1):
            if abs(offset) >= len(candles_1h):
                continue

            h = float(high.iloc[offset])
            l = float(low.iloc[offset])  # noqa: E741
            c = float(close.iloc[offset])

            for level, level_type, target in levels:
                result = self._check_sweep(h, l, c, level, level_type, target)
                if result is not None:
                    # No preference → return first found (original behavior)
                    if prefer_direction is None:
                        logger.info(
                            "sweep_detected",
                            type=result.sweep_type,
                            direction=result.sweep_direction,
                            level=level,
                            sweep_level=result.sweep_level,
                            target=result.target_level,
                            depth=f"{result.sweep_depth:.4f}",
                        )
                        return result

                    # Preferred direction match → return immediately
                    if result.sweep_direction == prefer_direction:
                        logger.info(
                            "sweep_detected",
                            type=result.sweep_type,
                            direction=result.sweep_direction,
                            level=level,
                            sweep_level=result.sweep_level,
                            target=result.target_level,
                            depth=f"{result.sweep_depth:.4f}",
                        )
                        return result

                    # Non-matching direction → stash as fallback
                    if first_any is None:
                        first_any = result

        # No preferred-direction match; return any sweep found as fallback
        if first_any is not None:
            logger.info(
                "sweep_detected",
                type=first_any.sweep_type,
                direction=first_any.sweep_direction,
                level=first_any.sweep_level,
                sweep_level=first_any.sweep_level,
                target=first_any.target_level,
                depth=f"{first_any.sweep_depth:.4f}",
                note="fallback (preferred direction not found)",
            )
            return first_any

        return _empty_result()

    def _check_sweep(
        self,
        h: float,
        l: float,  # noqa: E741
        c: float,
        level: float,
        level_type: str,
        target: float,
    ) -> SweepResult | None:
        """Check if a single candle swept a level and closed back.

        Bullish sweep (swept below a low): low < level AND close > level
        → MMs grabbed sell-side liquidity, expect bullish move
        → sweep_level = candle_low (SL goes below this)
        → target_level = opposite high (TP targets this)

        Bearish sweep (swept above a high): high > level AND close < level
        → MMs grabbed buy-side liquidity, expect bearish move
        → sweep_level = candle_high (SL goes above this)
        → target_level = opposite low (TP targets this)
        """
        # Bullish sweep: wick below level, close back above
        if level_type in ("asian_low", "london_low", "ny_low", "swing_low"):
            if l < level and c > level:
                depth = level - l
                return SweepResult(
                    sweep_detected=True,
                    sweep_direction="bullish",
                    sweep_level=l,       # wick extreme for SL
                    sweep_type=level_type,
                    target_level=target,  # opposite side for TP
                    sweep_depth=depth,
                )

        # Bearish sweep: wick above level, close back below
        if level_type in ("asian_high", "london_high", "ny_high", "swing_high"):
            if h > level and c < level:
                depth = h - level
                return SweepResult(
                    sweep_detected=True,
                    sweep_direction="bearish",
                    sweep_level=h,       # wick extreme for SL
                    sweep_type=level_type,
                    target_level=target,  # opposite side for TP
                    sweep_depth=depth,
                )

        return None
