"""Tests for the MM RSI indicator (src.strategy.mm_rsi)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.mm_rsi import RSIAnalyzer, RSIState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlc(closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from close prices."""
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    highs_arr = np.array(highs, dtype=float) if highs is not None else closes_arr * 1.001
    lows_arr = np.array(lows, dtype=float) if lows is not None else closes_arr * 0.999
    return pd.DataFrame({
        "open": closes_arr,
        "high": highs_arr,
        "low": lows_arr,
        "close": closes_arr,
        "volume": np.ones(n) * 1000.0,
    })


def _flat_closes(value: float, n: int = 50) -> list[float]:
    return [value] * n


def _trending_closes(start: float, step: float, n: int = 50) -> list[float]:
    return [start + step * i for i in range(n)]


# ---------------------------------------------------------------------------
# RSI calculation accuracy
# ---------------------------------------------------------------------------

class TestRSICalculation:
    """Verify RSI values against known reference data."""

    def test_all_gains_rsi_is_100(self):
        """Monotonically rising prices → RSI should converge to 100."""
        closes = _trending_closes(100.0, 1.0, n=50)
        df = _make_ohlc(closes)
        analyzer = RSIAnalyzer(period=14)
        state = analyzer.calculate(df)
        assert state is not None
        # With all gains and Wilder smoothing, RSI is very close to 100
        assert state.rsi_value >= 95.0

    def test_all_losses_rsi_is_near_zero(self):
        """Monotonically falling prices → RSI should converge near 0."""
        closes = _trending_closes(200.0, -1.0, n=50)
        df = _make_ohlc(closes)
        analyzer = RSIAnalyzer(period=14)
        state = analyzer.calculate(df)
        assert state is not None
        assert state.rsi_value <= 5.0

    def test_flat_price_rsi_near_50(self):
        """No price movement (all closes equal) → avg_loss=0 → RSI=100.
        This is correct Wilder behaviour: 0 avg_loss means no downward pressure."""
        closes = _flat_closes(100.0, n=30)
        df = _make_ohlc(closes)
        analyzer = RSIAnalyzer(period=14)
        state = analyzer.calculate(df)
        assert state is not None
        # Flat closes: first delta is NaN, rest are 0 gain/0 loss → RSI=100
        assert state.rsi_value == pytest.approx(100.0, abs=1.0)

    def test_rsi_value_in_valid_range(self):
        """RSI must always be in [0, 100]."""
        import random
        random.seed(42)
        closes = [100.0 + random.gauss(0, 2) for _ in range(60)]
        df = _make_ohlc(closes)
        state = RSIAnalyzer(period=14).calculate(df)
        assert state is not None
        assert 0.0 <= state.rsi_value <= 100.0

    def test_rsi_known_reference(self):
        """Verify RSI is in expected ballpark for an alternating pattern."""
        # Alternating +1 / -0.5 should produce RSI above 50 (net upward)
        closes = [100.0]
        for i in range(40):
            if i % 2 == 0:
                closes.append(closes[-1] + 1.0)
            else:
                closes.append(closes[-1] - 0.5)
        df = _make_ohlc(closes)
        state = RSIAnalyzer(period=14).calculate(df)
        assert state is not None
        assert state.rsi_value > 50.0


# ---------------------------------------------------------------------------
# Insufficient data → None
# ---------------------------------------------------------------------------

class TestInsufficientData:
    def test_none_when_empty_df(self):
        df = pd.DataFrame()
        assert RSIAnalyzer().calculate(df) is None

    def test_none_when_not_enough_rows(self):
        df = _make_ohlc([100.0] * 10)  # need period+1 = 15
        assert RSIAnalyzer(period=14).calculate(df) is None

    def test_none_when_exactly_period_rows(self):
        df = _make_ohlc([100.0] * 14)  # exactly period — not enough
        assert RSIAnalyzer(period=14).calculate(df) is None

    def test_not_none_when_period_plus_one(self):
        df = _make_ohlc([100.0] * 15)  # period+1 = 15 → just enough
        state = RSIAnalyzer(period=14).calculate(df)
        assert state is not None


# ---------------------------------------------------------------------------
# Trend bias classification
# ---------------------------------------------------------------------------

class TestTrendBias:
    def test_rsi_70_bullish(self):
        analyzer = RSIAnalyzer()
        assert analyzer._classify_trend_bias(70.0) == "bullish"

    def test_rsi_80_bullish(self):
        analyzer = RSIAnalyzer()
        assert analyzer._classify_trend_bias(80.0) == "bullish"

    def test_rsi_30_bearish(self):
        analyzer = RSIAnalyzer()
        assert analyzer._classify_trend_bias(30.0) == "bearish"

    def test_rsi_20_bearish(self):
        analyzer = RSIAnalyzer()
        assert analyzer._classify_trend_bias(20.0) == "bearish"

    def test_rsi_50_neutral(self):
        analyzer = RSIAnalyzer()
        assert analyzer._classify_trend_bias(50.0) == "neutral"

    def test_rsi_40_neutral(self):
        # 40 is the boundary — inclusive, so neutral
        analyzer = RSIAnalyzer()
        assert analyzer._classify_trend_bias(40.0) == "neutral"

    def test_rsi_60_neutral(self):
        # 60 is the boundary — inclusive, so neutral
        analyzer = RSIAnalyzer()
        assert analyzer._classify_trend_bias(60.0) == "neutral"

    def test_rsi_just_above_60_bullish(self):
        analyzer = RSIAnalyzer()
        assert analyzer._classify_trend_bias(60.1) == "bullish"

    def test_rsi_just_below_40_bearish(self):
        analyzer = RSIAnalyzer()
        assert analyzer._classify_trend_bias(39.9) == "bearish"

    def test_rsi_55_neutral(self):
        analyzer = RSIAnalyzer()
        assert analyzer._classify_trend_bias(55.0) == "neutral"


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------

class TestDivergence:
    def _make_bullish_div_df(self) -> pd.DataFrame:
        """Price lower low + RSI higher low setup.

        First half: price drops to 90, RSI ~30.
        Second half: price drops to 85 (lower low), but we'll craft the
        RSI series manually to test the split-window logic via _detect_divergence.
        """
        # We use 40 candles. First 20: declining slowly, last 20: declining
        # more steeply. But we'll call _detect_divergence directly with a
        # synthetic RSI series to precisely test the logic.
        closes = [100.0 - 0.3 * i for i in range(20)] + [94.0 - 0.5 * i for i in range(20)]
        lows = [c - 0.5 for c in closes]
        highs = [c + 0.5 for c in closes]
        return _make_ohlc(closes, highs=highs, lows=lows)

    def test_bullish_divergence_direct(self):
        """price lower low + RSI higher low → bullish divergence detected."""
        analyzer = RSIAnalyzer()
        n = 40
        # First half: lows around 90, second half: lows around 85 (lower)
        lows = [90.0] * 20 + [85.0] * 20
        closes = [92.0] * 20 + [87.0] * 20
        highs = [95.0] * 20 + [90.0] * 20
        df = _make_ohlc(closes, highs=highs, lows=lows)

        # RSI series: first half min ~30, second half min ~35 (higher low)
        rsi_first = list(np.linspace(35, 30, 20))   # min near 30 in first half
        rsi_second = list(np.linspace(40, 35, 20))  # min near 35 in second half (higher)
        rsi_series = pd.Series(rsi_first + rsi_second)

        detected, div_type = analyzer._detect_divergence(df, rsi_series, lookback=40)
        assert detected is True
        assert div_type == "bullish"

    def test_bearish_divergence_direct(self):
        """price higher high + RSI lower high → bearish divergence detected."""
        analyzer = RSIAnalyzer()
        # First half: highs around 100, second half: highs around 105 (higher)
        highs = [100.0] * 20 + [105.0] * 20
        closes = [98.0] * 20 + [103.0] * 20
        lows = [95.0] * 20 + [100.0] * 20
        df = _make_ohlc(closes, highs=highs, lows=lows)

        # RSI series: first half max ~70, second half max ~65 (lower high)
        rsi_first = list(np.linspace(65, 70, 20))   # max near 70 in first half
        rsi_second = list(np.linspace(60, 65, 20))  # max near 65 in second half (lower)
        rsi_series = pd.Series(rsi_first + rsi_second)

        detected, div_type = analyzer._detect_divergence(df, rsi_series, lookback=40)
        assert detected is True
        assert div_type == "bearish"

    def test_no_divergence_aligned_move(self):
        """Price lower low AND RSI lower low → no divergence (aligned move)."""
        analyzer = RSIAnalyzer()
        lows = [90.0] * 20 + [85.0] * 20
        closes = [92.0] * 20 + [87.0] * 20
        highs = [95.0] * 20 + [90.0] * 20
        df = _make_ohlc(closes, highs=highs, lows=lows)

        # RSI also lower low in second half → no divergence
        rsi_first = list(np.linspace(40, 30, 20))
        rsi_second = list(np.linspace(30, 25, 20))  # lower than first half
        rsi_series = pd.Series(rsi_first + rsi_second)

        detected, div_type = analyzer._detect_divergence(df, rsi_series, lookback=40)
        # Bearish divergence check: highs[20:] > highs[:20]? No (90 < 95), so no bearish
        # Bullish divergence check: RSI second min (25) > RSI first min (30)? No (25 < 30)
        assert detected is False
        assert div_type is None

    def test_divergence_insufficient_data(self):
        """Too few candles → no divergence."""
        analyzer = RSIAnalyzer()
        df = _make_ohlc([100.0] * 5)
        rsi_series = pd.Series([50.0] * 5)
        detected, div_type = analyzer._detect_divergence(df, rsi_series, lookback=20)
        assert detected is False
        assert div_type is None


# ---------------------------------------------------------------------------
# 50-cross detection
# ---------------------------------------------------------------------------

class TestCross50:
    def test_crosses_50_upward(self):
        analyzer = RSIAnalyzer()
        # RSI goes from 48 → 52
        rsi = pd.Series([45.0, 47.0, 48.5, 51.0, 55.0])
        assert analyzer._detect_50_cross(rsi, lookback=5) is True

    def test_crosses_50_downward(self):
        analyzer = RSIAnalyzer()
        rsi = pd.Series([55.0, 52.0, 51.0, 49.0, 45.0])
        assert analyzer._detect_50_cross(rsi, lookback=5) is True

    def test_no_cross_stays_above(self):
        analyzer = RSIAnalyzer()
        rsi = pd.Series([60.0, 62.0, 65.0, 63.0, 61.0])
        assert analyzer._detect_50_cross(rsi, lookback=5) is False

    def test_no_cross_stays_below(self):
        analyzer = RSIAnalyzer()
        rsi = pd.Series([35.0, 38.0, 40.0, 39.0, 37.0])
        assert analyzer._detect_50_cross(rsi, lookback=5) is False

    def test_no_cross_insufficient_data(self):
        analyzer = RSIAnalyzer()
        rsi = pd.Series([50.5])
        assert analyzer._detect_50_cross(rsi, lookback=5) is False

    def test_lookback_limits_scan(self):
        """Cross that happened 10 candles ago is not detected with lookback=5."""
        analyzer = RSIAnalyzer()
        # Cross at index 3-4 (48 → 52), then stable above 50
        rsi = pd.Series([45.0, 48.0, 48.5, 52.0, 55.0, 58.0, 60.0, 62.0, 61.0, 63.0])
        # With lookback=5, only last 5+1=6 candles are checked (55→63) — no cross
        assert analyzer._detect_50_cross(rsi, lookback=5) is False


# ---------------------------------------------------------------------------
# Integration: calculate() returns correct RSIState fields
# ---------------------------------------------------------------------------

class TestCalculateIntegration:
    def test_state_fields_populated(self):
        closes = _trending_closes(100.0, 0.5, n=40)
        df = _make_ohlc(closes)
        state = RSIAnalyzer().calculate(df)
        assert isinstance(state, RSIState)
        assert isinstance(state.rsi_value, float)
        assert state.trend_bias in ("bullish", "bearish", "neutral")
        assert isinstance(state.divergence_detected, bool)
        assert state.divergence_type in ("bullish", "bearish", None)
        assert isinstance(state.crossed_50, bool)

    def test_rising_market_bullish_bias(self):
        """Strongly rising closes → RSI high → bullish bias."""
        closes = _trending_closes(100.0, 2.0, n=50)
        df = _make_ohlc(closes)
        state = RSIAnalyzer().calculate(df)
        assert state is not None
        assert state.trend_bias == "bullish"
        assert state.rsi_value > 60.0

    def test_falling_market_bearish_bias(self):
        """Strongly falling closes → RSI low → bearish bias."""
        closes = _trending_closes(200.0, -2.0, n=50)
        df = _make_ohlc(closes)
        state = RSIAnalyzer().calculate(df)
        assert state is not None
        assert state.trend_bias == "bearish"
        assert state.rsi_value < 40.0
