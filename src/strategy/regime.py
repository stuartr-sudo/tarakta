"""Market regime detection using ADX and Choppiness Index.

Classifies the current market environment as strong_trend, trending,
choppy, or ranging. Feeds a score multiplier into confluence scoring
and a threshold adjustment into the adaptive entry threshold.

Computes ADX and Choppiness Index directly with pandas/numpy
(no pandas_ta dependency required).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ADX thresholds
ADX_STRONG_TREND = 30
ADX_TRENDING = 22
ADX_WEAK = 18

# Choppiness Index thresholds (higher = choppier)
CHOP_TRENDING = 45
CHOP_NEUTRAL = 55
CHOP_CHOPPY = 62

# Minimum candles required for reliable indicator values
MIN_CANDLES = 30


@dataclass
class MarketRegime:
    """Market regime classification result."""

    regime: str  # "strong_trend", "trending", "choppy", "ranging"
    adx: float  # ADX value (0-100)
    plus_di: float  # +DI value
    minus_di: float  # -DI value
    choppiness: float  # Choppiness Index value (0-100)
    score_multiplier: float  # 0.7-1.10 modifier for confluence scoring
    threshold_adjustment: float  # points to add/subtract from entry threshold
    di_trend: str  # "bullish" (+DI > -DI) or "bearish" (-DI > +DI)


class MarketRegimeAnalyzer:
    """Analyzes market conditions to classify the current regime.

    Uses two complementary indicators:
    - ADX (Average Directional Index): Measures trend STRENGTH regardless
      of direction. ADX > 25 = trending, > 30 = strong trend, < 20 = weak/no trend.
    - Choppiness Index: Measures how "choppy" vs "trending" the market is.
      Lower = more trending, higher = more choppy/consolidating.

    These together give a robust view of whether the market environment
    is favorable for directional SMC entries.
    """

    def __init__(self, adx_length: int = 14, chop_length: int = 14) -> None:
        self._adx_length = adx_length
        self._chop_length = chop_length

    def analyze(
        self,
        ohlc_4h: pd.DataFrame,
        ohlc_1d: pd.DataFrame | None = None,
    ) -> MarketRegime:
        """Classify the current market regime.

        Args:
            ohlc_4h: 4H OHLCV DataFrame (primary timeframe for regime).
            ohlc_1d: Optional 1D OHLCV DataFrame for confirmation.

        Returns:
            MarketRegime with classification, indicator values, and modifiers.
        """
        # Calculate on 4h (primary)
        adx_val, plus_di, minus_di = self._calc_adx(ohlc_4h)
        chop_val = self._calc_choppiness(ohlc_4h)

        # If 1D is available, use it to moderate extreme classifications
        if ohlc_1d is not None and len(ohlc_1d) >= MIN_CANDLES:
            adx_1d, _, _ = self._calc_adx(ohlc_1d)
            chop_1d = self._calc_choppiness(ohlc_1d)

            # Average with 1D to smooth out noise (4h weighted 70%, 1d weighted 30%)
            adx_val = adx_val * 0.7 + adx_1d * 0.3
            chop_val = chop_val * 0.7 + chop_1d * 0.3

        # Classify regime
        regime, multiplier, threshold_adj = self._classify(adx_val, chop_val)

        di_trend = "bullish" if plus_di >= minus_di else "bearish"

        result = MarketRegime(
            regime=regime,
            adx=round(adx_val, 1),
            plus_di=round(plus_di, 1),
            minus_di=round(minus_di, 1),
            choppiness=round(chop_val, 1),
            score_multiplier=multiplier,
            threshold_adjustment=threshold_adj,
            di_trend=di_trend,
        )

        logger.debug(
            "regime_detected",
            regime=regime,
            adx=result.adx,
            choppiness=result.choppiness,
            multiplier=multiplier,
            di_trend=di_trend,
        )

        return result

    def _calc_adx(self, df: pd.DataFrame) -> tuple[float, float, float]:
        """Calculate ADX, +DI, and -DI from OHLCV data using Wilder's smoothing."""
        if len(df) < MIN_CANDLES:
            return 0.0, 0.0, 0.0

        try:
            high = df["high"].astype(float).values
            low = df["low"].astype(float).values
            close = df["close"].astype(float).values
            n = self._adx_length

            # True Range
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

            # Directional Movement
            up_move = high[1:] - high[:-1]
            down_move = low[:-1] - low[1:]
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

            # Wilder's smoothing (EMA with alpha = 1/n)
            alpha = 1.0 / n
            tr_s = tr[1:]  # align with dm arrays

            def wilder_smooth(arr: np.ndarray) -> np.ndarray:
                out = np.empty_like(arr)
                out[0] = arr[0]
                for i in range(1, len(arr)):
                    out[i] = out[i - 1] * (1 - alpha) + arr[i] * alpha
                return out

            sm_tr = wilder_smooth(tr_s)
            sm_plus_dm = wilder_smooth(plus_dm)
            sm_minus_dm = wilder_smooth(minus_dm)

            # +DI / -DI
            plus_di_arr = np.where(sm_tr > 0, 100.0 * sm_plus_dm / sm_tr, 0.0)
            minus_di_arr = np.where(sm_tr > 0, 100.0 * sm_minus_dm / sm_tr, 0.0)

            # DX and ADX
            di_sum = plus_di_arr + minus_di_arr
            dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di_arr - minus_di_arr) / di_sum, 0.0)
            adx_arr = wilder_smooth(dx)

            adx = float(adx_arr[-1])
            plus_di = float(plus_di_arr[-1])
            minus_di = float(minus_di_arr[-1])
            return adx, plus_di, minus_di

        except Exception as e:
            logger.warning("adx_calc_failed", error=str(e))
            return 0.0, 0.0, 0.0

    def _calc_choppiness(self, df: pd.DataFrame) -> float:
        """Calculate the Choppiness Index from OHLCV data.

        CHOP = 100 * LOG10(SUM(ATR1, n) / (HH - LL)) / LOG10(n)
        where ATR1 = True Range (period 1), HH/LL = highest high / lowest low over n bars.
        """
        if len(df) < MIN_CANDLES:
            return 50.0  # neutral default

        try:
            high = df["high"].astype(float).values
            low = df["low"].astype(float).values
            close = df["close"].astype(float).values
            n = self._chop_length

            # ATR(1) = True Range per bar
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

            # Use last n bars
            tr_sum = np.sum(tr[-n:])
            hh = np.max(high[-n:])
            ll = np.min(low[-n:])
            hl_range = hh - ll

            if hl_range <= 0:
                return 50.0

            chop = 100.0 * np.log10(tr_sum / hl_range) / np.log10(n)
            return float(np.clip(chop, 0.0, 100.0))

        except Exception as e:
            logger.warning("choppiness_calc_failed", error=str(e))
            return 50.0

    @staticmethod
    def _classify(adx: float, choppiness: float) -> tuple[str, float, float]:
        """Classify market regime from ADX and Choppiness values.

        Returns:
            (regime_name, score_multiplier, threshold_adjustment)
        """
        # Strong trend: high ADX + low choppiness
        if adx >= ADX_STRONG_TREND and choppiness < CHOP_TRENDING:
            return "strong_trend", 1.10, -3.0

        # Trending: decent ADX + moderate choppiness
        if adx >= ADX_TRENDING and choppiness < CHOP_NEUTRAL:
            return "trending", 1.00, 0.0

        # Ranging: very low ADX + very high choppiness
        if adx < ADX_WEAK and choppiness >= CHOP_CHOPPY:
            return "ranging", 0.70, 8.0

        # Choppy: low ADX or high choppiness
        if adx < ADX_TRENDING or choppiness >= CHOP_NEUTRAL:
            return "choppy", 0.80, 5.0

        # Default: trending (shouldn't reach here but be safe)
        return "trending", 1.00, 0.0
