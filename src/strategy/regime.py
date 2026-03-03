"""Market regime detection using ADX and Choppiness Index.

Classifies the current market environment as strong_trend, trending,
choppy, or ranging. Feeds a score multiplier into confluence scoring
and a threshold adjustment into the adaptive entry threshold.

Uses pandas_ta (already installed but previously unused) for indicator
calculations on 4H candles with 1D confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta

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
        """Calculate ADX, +DI, and -DI from OHLCV data."""
        if len(df) < MIN_CANDLES:
            return 0.0, 0.0, 0.0

        try:
            result = ta.adx(
                high=df["high"].astype(float),
                low=df["low"].astype(float),
                close=df["close"].astype(float),
                length=self._adx_length,
            )

            if result is None or result.empty:
                return 0.0, 0.0, 0.0

            # pandas_ta returns columns like ADX_14, DMP_14, DMN_14
            adx_col = f"ADX_{self._adx_length}"
            dmp_col = f"DMP_{self._adx_length}"
            dmn_col = f"DMN_{self._adx_length}"

            adx = float(result[adx_col].iloc[-1]) if pd.notna(result[adx_col].iloc[-1]) else 0.0
            plus_di = float(result[dmp_col].iloc[-1]) if pd.notna(result[dmp_col].iloc[-1]) else 0.0
            minus_di = float(result[dmn_col].iloc[-1]) if pd.notna(result[dmn_col].iloc[-1]) else 0.0

            return adx, plus_di, minus_di

        except Exception as e:
            logger.warning("adx_calc_failed", error=str(e))
            return 0.0, 0.0, 0.0

    def _calc_choppiness(self, df: pd.DataFrame) -> float:
        """Calculate the Choppiness Index from OHLCV data."""
        if len(df) < MIN_CANDLES:
            return 50.0  # neutral default

        try:
            result = ta.chop(
                high=df["high"].astype(float),
                low=df["low"].astype(float),
                close=df["close"].astype(float),
                length=self._chop_length,
            )

            if result is None or result.empty:
                return 50.0

            val = float(result.iloc[-1]) if pd.notna(result.iloc[-1]) else 50.0
            return val

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
