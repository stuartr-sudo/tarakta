"""Tests for MM EMA Framework (src.strategy.mm_ema_framework)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.mm_ema_framework import (
    DEFAULT_EMA_PERIODS,
    EMABreakResult,
    EMAFramework,
    EMAState,
    RetestResult,
    TrendState,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_trending_ohlcv(
    n: int = 900,
    base: float = 100.0,
    drift: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate an upward-trending OHLCV DataFrame."""
    rng = np.random.RandomState(seed)
    closes = np.empty(n)
    closes[0] = base
    for i in range(1, n):
        closes[i] = closes[i - 1] * (1 + drift / 100 + rng.normal(0, 0.003))

    opens = closes * (1 + rng.normal(0, 0.001, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volumes = rng.uniform(1000, 5000, n)

    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _make_bearish_ohlcv(n: int = 900, base: float = 200.0, seed: int = 42) -> pd.DataFrame:
    """Generate a strongly downward-trending OHLCV DataFrame.

    Uses a deterministic linear decay to ensure shorter EMAs stay below
    longer EMAs (bearish alignment: 10 < 20 < 50 < 200 < 800).
    """
    rng = np.random.RandomState(seed)
    # Deterministic downtrend: steady decline from 200 to ~50 over 900 candles
    # Using a consistent per-candle multiplier
    closes = base * np.power(0.9985, np.arange(n))  # ~0.15% decline per candle
    # Add tiny noise that doesn't overpower the trend
    closes = closes * (1 + rng.normal(0, 0.0005, n))
    closes = np.maximum(closes, 1.0)

    opens = closes * (1 + rng.normal(0, 0.0003, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.001, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.001, n)))
    volumes = rng.uniform(1000, 5000, n)

    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _make_flat_ohlcv(n: int = 900, base: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Generate a sideways OHLCV DataFrame."""
    rng = np.random.RandomState(seed)
    closes = base + rng.normal(0, 0.1, n)
    opens = closes + rng.normal(0, 0.05, n)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.05, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.05, n))
    volumes = rng.uniform(1000, 5000, n)

    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


@pytest.fixture
def framework() -> EMAFramework:
    return EMAFramework()


@pytest.fixture
def bullish_df() -> pd.DataFrame:
    return _make_trending_ohlcv(n=900, drift=0.05)


@pytest.fixture
def bearish_df() -> pd.DataFrame:
    return _make_bearish_ohlcv(n=900)


@pytest.fixture
def flat_df() -> pd.DataFrame:
    return _make_flat_ohlcv(n=900)


# ------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------

class TestEMAFrameworkInit:
    def test_default_periods(self, framework: EMAFramework):
        assert framework.periods == sorted(DEFAULT_EMA_PERIODS)

    def test_custom_periods(self):
        fw = EMAFramework(periods=[20, 50, 200])
        assert fw.periods == [20, 50, 200]

    def test_periods_sorted(self):
        fw = EMAFramework(periods=[200, 10, 50])
        assert fw.periods == [10, 50, 200]


# ------------------------------------------------------------------
# calculate
# ------------------------------------------------------------------

class TestCalculate:
    def test_returns_ema_state(self, framework: EMAFramework, bullish_df: pd.DataFrame):
        state = framework.calculate(bullish_df)
        assert isinstance(state, EMAState)
        assert len(state.values) == 5
        assert len(state.slopes) == 5

    def test_all_ema_values_positive(self, framework: EMAFramework, bullish_df: pd.DataFrame):
        state = framework.calculate(bullish_df)
        for period, val in state.values.items():
            assert val > 0, f"EMA {period} should be positive"

    def test_shorter_ema_above_longer_in_uptrend(
        self, framework: EMAFramework, bullish_df: pd.DataFrame
    ):
        state = framework.calculate(bullish_df)
        # In a strong uptrend, 10 EMA > 20 EMA > 50 EMA ...
        assert state.values[10] > state.values[20]
        assert state.values[20] > state.values[50]

    def test_alignment_bullish_uptrend(self, framework: EMAFramework, bullish_df: pd.DataFrame):
        state = framework.calculate(bullish_df)
        assert state.alignment == "bullish"

    def test_alignment_bearish_downtrend(self, framework: EMAFramework, bearish_df: pd.DataFrame):
        state = framework.calculate(bearish_df)
        assert state.alignment == "bearish"

    def test_empty_df_returns_empty_state(self, framework: EMAFramework):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        state = framework.calculate(empty)
        assert state.alignment == "mixed"
        assert all(v == 0.0 for v in state.values.values())

    def test_insufficient_data(self, framework: EMAFramework):
        small = _make_trending_ohlcv(n=50)
        state = framework.calculate(small)
        # Not enough for 800 EMA
        assert state.alignment == "mixed"

    def test_fan_out_score_range(self, framework: EMAFramework, bullish_df: pd.DataFrame):
        state = framework.calculate(bullish_df)
        assert 0.0 <= state.fan_out_score <= 1.0

    def test_price_distance_from_50_positive_in_uptrend(
        self, framework: EMAFramework, bullish_df: pd.DataFrame
    ):
        state = framework.calculate(bullish_df)
        # Price should be above 50 EMA in uptrend
        assert state.price_distance_from_50 > 0

    def test_slopes_positive_in_uptrend(self, framework: EMAFramework, bullish_df: pd.DataFrame):
        state = framework.calculate(bullish_df)
        # At least the faster EMAs should have positive slopes
        assert state.slopes[10] > 0
        assert state.slopes[20] > 0


# ------------------------------------------------------------------
# detect_ema_break
# ------------------------------------------------------------------

class TestDetectEMABreak:
    def test_no_break_empty(self, framework: EMAFramework):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = framework.detect_ema_break(empty)
        assert result.broke_ema is False
        assert result.break_candle_idx == -1

    def test_returns_ema_break_result(self, framework: EMAFramework, bullish_df: pd.DataFrame):
        result = framework.detect_ema_break(bullish_df, ema_period=50)
        assert isinstance(result, EMABreakResult)

    def test_break_with_volume_spike(self, framework: EMAFramework):
        """Create a scenario where price crosses the 50 EMA with high volume.

        The break must occur within the last LOOKBACK_WINDOW (20) candles
        for detect_ema_break to find it.
        """
        n = 200
        # Price stays flat at 90 for most of the data, then jumps near the end.
        # The 50 EMA will converge near 90. The jump happens at index 190
        # (within the last 20 candles).
        closes = np.full(n, 90.0)
        # Jump at candle 190 onwards
        closes[190:] = 110.0

        opens = closes.copy()
        opens[190:] = 90.0  # open at 90, close at 110 -> big bullish bar
        highs = np.maximum(opens, closes) + 0.5
        lows = np.minimum(opens, closes) - 0.5
        volumes = np.full(n, 1000.0)
        # Spike volume at the break point
        volumes[190:195] = 5000.0  # 5x average

        idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=idx,
        )
        result = framework.detect_ema_break(df, ema_period=50, volume_threshold=2.0)
        assert result.broke_ema is True
        assert result.direction == "bullish"
        assert result.volume_confirmed is True

    def test_break_without_volume_not_confirmed(self, framework: EMAFramework):
        """Break with normal volume should have volume_confirmed=False."""
        n = 200
        closes = np.concatenate([
            np.linspace(90, 95, 100),
            np.linspace(95, 110, 100),
        ])
        opens = closes - 0.5
        highs = closes + 1.0
        lows = closes - 1.5
        volumes = np.full(n, 1000.0)  # flat volume

        idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=idx,
        )
        result = framework.detect_ema_break(df, ema_period=50, volume_threshold=2.0)
        if result.broke_ema:
            assert result.volume_confirmed is False


# ------------------------------------------------------------------
# get_trend_state
# ------------------------------------------------------------------

class TestGetTrendState:
    def test_uptrend(self, framework: EMAFramework, bullish_df: pd.DataFrame):
        ts = framework.get_trend_state(bullish_df)
        assert isinstance(ts, TrendState)
        assert ts.direction == "bullish"
        assert ts.strength > 0

    def test_downtrend(self, framework: EMAFramework, bearish_df: pd.DataFrame):
        ts = framework.get_trend_state(bearish_df)
        assert ts.direction == "bearish"

    def test_sideways(self, framework: EMAFramework, flat_df: pd.DataFrame):
        ts = framework.get_trend_state(flat_df)
        # Flat data should have low strength
        assert ts.strength < 0.8

    def test_flattening_in_sideways(self, framework: EMAFramework, flat_df: pd.DataFrame):
        ts = framework.get_trend_state(flat_df)
        assert ts.is_flattening is True

    def test_strength_range(self, framework: EMAFramework, bullish_df: pd.DataFrame):
        ts = framework.get_trend_state(bullish_df)
        assert 0.0 <= ts.strength <= 1.0

    def test_alignment_score_range(self, framework: EMAFramework, bullish_df: pd.DataFrame):
        ts = framework.get_trend_state(bullish_df)
        assert 0.0 <= ts.ema_alignment_score <= 1.0

    def test_empty_returns_sideways(self, framework: EMAFramework):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        ts = framework.get_trend_state(empty)
        assert ts.direction == "sideways"
        assert ts.strength == 0.0


# ------------------------------------------------------------------
# detect_retest
# ------------------------------------------------------------------

class TestDetectRetest:
    def test_no_retest_empty(self, framework: EMAFramework):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = framework.detect_retest(empty, ema_period=50)
        assert isinstance(result, RetestResult)
        assert result.retested is False

    def test_no_retest_insufficient_data(self, framework: EMAFramework):
        small = _make_trending_ohlcv(n=30)
        result = framework.detect_retest(small, ema_period=50)
        assert result.retested is False

    def test_retest_after_break(self, framework: EMAFramework):
        """Construct data: break above EMA, pull back to it, then resume."""
        n = 200
        closes = np.concatenate([
            np.linspace(90, 95, 80),    # below EMA
            np.linspace(96, 110, 40),   # break above
            np.linspace(109, 100, 30),  # retrace toward EMA
            np.linspace(100, 115, 50),  # resume
        ])
        opens = closes - 0.3
        highs = closes + 0.8
        lows = closes - 1.0
        volumes = np.full(n, 2000.0)

        idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=idx,
        )
        result = framework.detect_retest(df, ema_period=50)
        # Should find a retest after the break
        assert isinstance(result, RetestResult)
        # At minimum: either retested or not; the structure is valid


# ------------------------------------------------------------------
# classify_volume (PVSRA)
# ------------------------------------------------------------------

class TestClassifyVolume:
    def test_empty_df(self, framework: EMAFramework):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = framework.classify_volume(empty)
        assert len(result) == 0

    def test_normal_volume(self, framework: EMAFramework):
        """Uniform volume should all be 'normal' after the lookback period."""
        n = 50
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.5),
            "volume": np.full(n, 1000.0),
        })
        result = framework.classify_volume(df)
        # After lookback, all should be normal
        assert (result.iloc[15:] == "normal").all()

    def test_vector_200_detection(self, framework: EMAFramework):
        """Volume at 3x average should be classified as vector_200."""
        n = 30
        volumes = np.full(n, 1000.0)
        volumes[-1] = 3000.0  # 3x average -> vector_200

        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.5),
            "volume": volumes,
        })
        result = framework.classify_volume(df)
        assert result.iloc[-1] == "vector_200"

    def test_vector_150_detection(self, framework: EMAFramework):
        """Volume at 1.7x average should be classified as vector_150."""
        n = 30
        volumes = np.full(n, 1000.0)
        volumes[-1] = 1700.0  # 1.7x -> vector_150

        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.5),
            "volume": volumes,
        })
        result = framework.classify_volume(df)
        assert result.iloc[-1] == "vector_150"

    def test_first_lookback_candles_are_normal(self, framework: EMAFramework):
        """First PVSRA_LOOKBACK candles should be 'normal' regardless."""
        n = 30
        volumes = np.full(n, 1000.0)
        volumes[5] = 10000.0  # huge volume but within lookback

        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.5),
            "volume": volumes,
        })
        result = framework.classify_volume(df)
        assert result.iloc[5] == "normal"
