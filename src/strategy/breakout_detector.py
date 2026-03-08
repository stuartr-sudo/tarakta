"""Breakout detection for the Trade Travel Chill strategy.

Catches genuine breakouts — price breaking through a key level with volume
and HOLDING above/below it. This is the OPPOSITE of a sweep: a sweep wicks
through and closes back, while a breakout breaks through and keeps going.

This provides a complementary signal path for trending markets where the
sweep-and-reverse pattern doesn't occur (e.g., NVDA ATH breakout, strong
momentum moves).

Breakout criteria:
1. Price closes beyond a key level (session range or swing high/low)
2. Price HOLDS beyond that level for 3+ completed candles
3. Volume is elevated (> 1.5x average) on majority of hold candles — sustained institutional participation
4. Distance from level is meaningful (> 0.5 ATR to filter noise)
"""
from __future__ import annotations

import math

import pandas as pd

from src.exchange.models import BreakoutResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Minimum candles price must hold beyond the level
MIN_HOLD_CANDLES = 3

# Minimum volume ratio vs 20-period average
MIN_VOLUME_RATIO = 1.5

# Minimum distance from level in ATR units (filters noise touches)
MIN_ATR_DISTANCE = 0.5


def _empty_result() -> BreakoutResult:
    return BreakoutResult(
        breakout_detected=False,
        breakout_direction=None,
        breakout_level=0.0,
        breakout_type=None,
        target_level=0.0,
        volume_confirmed=False,
        candles_held=0,
        atr_distance=0.0,
    )


class BreakoutDetector:
    """Detects genuine breakouts on 1H candles.

    A breakout = price closes beyond a key level AND stays beyond it
    for multiple candles with elevated volume. This is the opposite of
    a sweep (which wicks through and closes back).
    """

    def detect(
        self,
        candles_1h: pd.DataFrame,
        asian_high: float = 0.0,
        asian_low: float = 0.0,
        london_high: float = 0.0,
        london_low: float = 0.0,
        ny_high: float = 0.0,
        ny_low: float = 0.0,
        swing_high: float | None = None,
        swing_low: float | None = None,
        lookback: int = 6,
    ) -> BreakoutResult:
        """Check recent candles for a breakout that's holding beyond a level.

        Priority order (same as sweep, but reversed priority since
        breakouts through session ranges during their own session are
        most meaningful):
        1. London/NY range breakout (session extremes being violated)
        2. Asian range breakout
        3. Swing high/low breakout (structural levels)

        Args:
            candles_1h: 1H OHLCV DataFrame.
            asian_high/low: Asian session range.
            london_high/low: London session range.
            ny_high/low: NY session range.
            swing_high/low: Recent structural swing levels.
            lookback: How many candles back to check for the initial break.

        Returns:
            BreakoutResult with detection details.
        """
        if candles_1h is None or len(candles_1h) < lookback + 5:
            return _empty_result()

        high = candles_1h["high"].astype(float)
        low = candles_1h["low"].astype(float)
        close = candles_1h["close"].astype(float)
        volume = candles_1h["volume"].astype(float)

        # Calculate ATR for distance filtering
        atr = self._calc_atr(high, low, close, period=14)
        if atr is None or atr <= 0:
            return _empty_result()

        # Calculate average volume for volume confirmation
        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        if avg_vol <= 0:
            avg_vol = 1.0

        # Build levels to check — session ranges first, then structural
        levels: list[tuple[float, str, float]] = []

        if london_high > 0 and london_low > 0 and london_high > london_low:
            levels.append((london_high, "london_high", london_low))
            levels.append((london_low, "london_low", london_high))

        if ny_high > 0 and ny_low > 0 and ny_high > ny_low:
            levels.append((ny_high, "ny_high", ny_low))
            levels.append((ny_low, "ny_low", ny_high))

        if asian_high > 0 and asian_low > 0 and asian_high > asian_low:
            levels.append((asian_high, "asian_high", asian_low))
            levels.append((asian_low, "asian_low", asian_high))

        if swing_high and swing_high > 0:
            target = swing_low if (swing_low and swing_low > 0) else 0.0
            levels.append((swing_high, "swing_high", target))

        if swing_low and swing_low > 0:
            target = swing_high if (swing_high and swing_high > 0) else 0.0
            levels.append((swing_low, "swing_low", target))

        if not levels:
            return _empty_result()

        # Check each level for a breakout
        best_result: BreakoutResult | None = None

        for level, level_type, target in levels:
            result = self._check_breakout(
                close, high, low, volume,
                level, level_type, target,
                atr, avg_vol, lookback,
            )
            if result is not None:
                # Prefer breakouts with more candles held and volume confirmation
                if best_result is None:
                    best_result = result
                elif (result.volume_confirmed and not best_result.volume_confirmed):
                    best_result = result
                elif (result.candles_held > best_result.candles_held
                      and result.volume_confirmed == best_result.volume_confirmed):
                    best_result = result

        if best_result is not None:
            logger.info(
                "breakout_detected",
                type=best_result.breakout_type,
                direction=best_result.breakout_direction,
                level=best_result.breakout_level,
                target=best_result.target_level,
                volume_confirmed=best_result.volume_confirmed,
                candles_held=best_result.candles_held,
                atr_distance=f"{best_result.atr_distance:.2f}",
            )
            return best_result

        return _empty_result()

    def _check_breakout(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        level: float,
        level_type: str,
        target: float,
        atr: float,
        avg_vol: float,
        lookback: int,
    ) -> BreakoutResult | None:
        """Check if price has broken AND held beyond a level.

        Bullish breakout (above a high): closes above level for 2+ candles
        Bearish breakout (below a low): closes below level for 2+ candles
        """
        n = len(close)

        # Determine direction based on level type
        if level_type in ("asian_high", "london_high", "ny_high", "swing_high"):
            # Bullish breakout: price breaks above a high level
            # Check the most recent candles (skip -1 which may be incomplete)
            candles_above = 0
            vol_elevated_count = 0
            max_distance = 0.0

            for i in range(-2, -2 - lookback, -1):
                if abs(i) >= n:
                    break
                c = float(close.iloc[i])
                if c > level:
                    candles_above += 1
                    distance = c - level
                    max_distance = max(max_distance, distance)
                    # Count candles with elevated volume (sustained participation)
                    v = float(volume.iloc[i])
                    if v > avg_vol * MIN_VOLUME_RATIO:
                        vol_elevated_count += 1
                else:
                    # Price closed back below — not a clean breakout hold
                    break

            atr_dist = max_distance / atr if atr > 0 else 0
            # Require majority of hold candles to have elevated volume
            vol_elevated = vol_elevated_count >= math.ceil(candles_above / 2) if candles_above > 0 else False

            if candles_above >= MIN_HOLD_CANDLES and atr_dist >= MIN_ATR_DISTANCE:
                return BreakoutResult(
                    breakout_detected=True,
                    breakout_direction="bullish",
                    breakout_level=level,
                    breakout_type=level_type,
                    target_level=target,
                    volume_confirmed=vol_elevated,
                    candles_held=candles_above,
                    atr_distance=round(atr_dist, 3),
                )

        elif level_type in ("asian_low", "london_low", "ny_low", "swing_low"):
            # Bearish breakout: price breaks below a low level
            candles_below = 0
            vol_elevated_count = 0
            max_distance = 0.0

            for i in range(-2, -2 - lookback, -1):
                if abs(i) >= n:
                    break
                c = float(close.iloc[i])
                if c < level:
                    candles_below += 1
                    distance = level - c
                    max_distance = max(max_distance, distance)
                    v = float(volume.iloc[i])
                    if v > avg_vol * MIN_VOLUME_RATIO:
                        vol_elevated_count += 1
                else:
                    break

            atr_dist = max_distance / atr if atr > 0 else 0
            # Require majority of hold candles to have elevated volume
            vol_elevated = vol_elevated_count >= math.ceil(candles_below / 2) if candles_below > 0 else False

            if candles_below >= MIN_HOLD_CANDLES and atr_dist >= MIN_ATR_DISTANCE:
                return BreakoutResult(
                    breakout_detected=True,
                    breakout_direction="bearish",
                    breakout_level=level,
                    breakout_type=level_type,
                    target_level=target,
                    volume_confirmed=vol_elevated,
                    candles_held=candles_below,
                    atr_distance=round(atr_dist, 3),
                )

        return None

    def _calc_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> float | None:
        """Calculate latest ATR value."""
        if len(close) < period + 1:
            return None
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr_series = tr.rolling(period).mean()
        val = atr_series.iloc[-1]
        return float(val) if pd.notna(val) else None
