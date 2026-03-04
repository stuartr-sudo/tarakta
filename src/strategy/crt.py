"""Candle Range Theory (CRT) analysis on 4H timeframe.

Detects CRT patterns — a key Market Maker Method concept where a
candle opens near one extreme of a dealing range, sweeps (wicks beyond)
the other extreme to grab liquidity, then closes back inside the range,
signaling the real directional move.

Bullish CRT: wick sweeps below dealing range low, closes back above it.
Bearish CRT: wick sweeps above dealing range high, closes back below it.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CRTResult:
    """Result of Candle Range Theory analysis."""
    crt_detected: bool
    direction: str | None       # "bullish" or "bearish" or None
    strength: float             # 0.0 to 1.0
    dealing_range_high: float
    dealing_range_low: float
    sweep_level: float          # The level that was swept
    reversal_strength: float    # Wick-to-body ratio of the reversal candle


class CRTAnalyzer:
    """Detects CRT patterns on 4H candles."""

    def analyze(
        self,
        candles_4h: pd.DataFrame,
        swing_hl: object | None = None,
    ) -> CRTResult:
        """Analyze 4H candles for CRT pattern.

        Args:
            candles_4h: 4H OHLCV DataFrame.
            swing_hl: Swing highs/lows DataFrame from smc library.

        Returns:
            CRTResult with pattern detection details.
        """
        empty = CRTResult(
            crt_detected=False,
            direction=None,
            strength=0.0,
            dealing_range_high=0.0,
            dealing_range_low=0.0,
            sweep_level=0.0,
            reversal_strength=0.0,
        )

        if candles_4h is None or len(candles_4h) < 20:
            return empty

        # --- Step 1: Find the dealing range from 4H swing highs/lows ---
        high = candles_4h["high"].astype(float)
        low = candles_4h["low"].astype(float)

        range_high, range_low = self._find_dealing_range(high, low, swing_hl)

        if range_high <= range_low or range_high <= 0:
            return empty

        dealing_range = range_high - range_low
        if dealing_range / range_high < 0.001:
            # Range too tight to be meaningful
            return empty

        # --- Step 2: Check last 3 completed candles for CRT pattern ---
        # iloc[-1] may be the current (incomplete) candle, so check [-2], [-3], [-4]
        for offset in [-2, -3, -4]:
            if abs(offset) >= len(candles_4h):
                continue

            o = float(candles_4h["open"].iloc[offset])
            h = float(candles_4h["high"].iloc[offset])
            l = float(candles_4h["low"].iloc[offset])  # noqa: E741
            c = float(candles_4h["close"].iloc[offset])

            candle_range = h - l
            if candle_range <= 0:
                continue

            result = self._check_crt_candle(
                o, h, l, c, range_high, range_low, dealing_range, candle_range,
            )
            if result is not None:
                return result

        return empty

    def _check_crt_candle(
        self,
        o: float,
        h: float,
        l: float,  # noqa: E741
        c: float,
        range_high: float,
        range_low: float,
        dealing_range: float,
        candle_range: float,
    ) -> CRTResult | None:
        """Check a single 4H candle for CRT characteristics.

        Returns CRTResult if CRT detected, None otherwise.
        """
        body_top = max(o, c)
        body_bottom = min(o, c)
        body_size = body_top - body_bottom

        # --- Bullish CRT ---
        # Wick sweeps below dealing range low, close is back above it
        if l < range_low and c > range_low:
            sweep_depth = range_low - l
            reversal = (c - l) / candle_range if candle_range > 0 else 0

            # Opening position: for bullish CRT, open should be in the
            # upper portion of the range (near high, then sweeps low, closes back up)
            # OR open near the low and it's a hammer pattern
            open_position = (o - range_low) / dealing_range if dealing_range > 0 else 0.5

            strength = self._calculate_strength(
                sweep_depth, dealing_range, reversal, open_position, body_size, candle_range,
            )

            if strength >= 0.2:  # Minimum strength threshold
                return CRTResult(
                    crt_detected=True,
                    direction="bullish",
                    strength=min(strength, 1.0),
                    dealing_range_high=range_high,
                    dealing_range_low=range_low,
                    sweep_level=range_low,
                    reversal_strength=reversal,
                )

        # --- Bearish CRT ---
        # Wick sweeps above dealing range high, close is back below it
        if h > range_high and c < range_high:
            sweep_depth = h - range_high
            reversal = (h - c) / candle_range if candle_range > 0 else 0

            # For bearish CRT, open should be in the lower portion
            # OR open near the high and it's a shooting star
            open_position = (range_high - o) / dealing_range if dealing_range > 0 else 0.5

            strength = self._calculate_strength(
                sweep_depth, dealing_range, reversal, open_position, body_size, candle_range,
            )

            if strength >= 0.2:
                return CRTResult(
                    crt_detected=True,
                    direction="bearish",
                    strength=min(strength, 1.0),
                    dealing_range_high=range_high,
                    dealing_range_low=range_low,
                    sweep_level=range_high,
                    reversal_strength=reversal,
                )

        return None

    def _calculate_strength(
        self,
        sweep_depth: float,
        dealing_range: float,
        reversal: float,
        open_position: float,
        body_size: float,
        candle_range: float,
    ) -> float:
        """Calculate CRT pattern strength (0-1).

        Components:
        - Reversal strength: how far the close is from the sweep end (0-0.4)
        - Sweep depth: how far price swept beyond the level (0-0.3)
        - Body-to-range ratio: stronger body = more conviction (0-0.3)
        """
        score = 0.0

        # Reversal strength (0-0.4): close far from sweep end = strong reversal
        score += min(reversal * 0.5, 0.4)

        # Sweep depth relative to dealing range (0-0.3)
        # Deeper sweep = more liquidity grabbed = stronger signal
        if dealing_range > 0:
            sweep_pct = sweep_depth / dealing_range
            score += min(sweep_pct * 2.0, 0.3)

        # Body-to-range ratio (0-0.3): big body relative to candle range = conviction
        if candle_range > 0:
            body_ratio = body_size / candle_range
            score += min(body_ratio * 0.4, 0.3)

        return score

    def _find_dealing_range(
        self,
        high: pd.Series,
        low: pd.Series,
        swing_hl: object | None = None,
    ) -> tuple[float, float]:
        """Find the dealing range from 4H swing highs/lows.

        Reuses the same pattern as PremiumDiscountAnalyzer._find_dealing_range().
        """
        # Try using swing highs/lows from smc library
        if swing_hl is not None and hasattr(swing_hl, "columns"):
            try:
                df = swing_hl
                if "HighLow" in df.columns:
                    recent_highs: list[float] = []
                    recent_lows: list[float] = []

                    for i in range(len(df) - 1, max(len(df) - 50, -1), -1):
                        val = df["HighLow"].iloc[i]
                        if val == 1 and len(recent_highs) < 2:
                            recent_highs.append(float(high.iloc[i]))
                        elif val == -1 and len(recent_lows) < 2:
                            recent_lows.append(float(low.iloc[i]))

                        if recent_highs and recent_lows:
                            break

                    if recent_highs and recent_lows:
                        return max(recent_highs), min(recent_lows)
            except Exception:
                pass

        # Fallback: rolling 50-bar window
        lookback = min(50, len(high))
        range_high = float(high.iloc[-lookback:].max())
        range_low = float(low.iloc[-lookback:].min())

        return range_high, range_low
