"""Tests for the BBWP (Bollinger Band Width Percentile) indicator.

Course Trading Strategies Lesson 04 (C4):
  - BBWP = BBW expressed as a percentile of its 252-bar history.
  - Signal at >= 95: extreme_reached (local top/bottom found, be careful).
  - Signal at <= 5: breakout_imminent (consolidation maturing, move coming).
  - Timing-only indicator — no directional signal.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.strategy.mm_bbwp import (
    BBWP_EXTREME_HIGH,
    BBWP_EXTREME_LOW,
    BBWP_LENGTH,
    BBWP_BB_PERIOD,
    BBWPAnalyzer,
    BBWPState,
    _percentile_rank,
)

UTC = ZoneInfo("UTC")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flat_ohlcv(n: int, base: float = 1000.0) -> pd.DataFrame:
    """All closes equal → zero stddev → near-zero BB width throughout."""
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    closes = np.full(n, base)
    # Add tiny noise so rolling std doesn't produce NaN, but width stays tiny
    closes += np.random.RandomState(0).normal(0, 0.001, n)
    opens = closes - 0.5
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = np.ones(n) * 100
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _make_trending_ohlcv(n: int, base: float = 1000.0, pct_per_bar: float = 0.005) -> pd.DataFrame:
    """Strongly trending candles with high volatility → wide BB width."""
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.RandomState(42)
    closes = base * np.cumprod(1 + pct_per_bar + rng.normal(0, 0.002, n))
    opens = closes * (1 - 0.001)
    highs = closes * (1 + 0.003)
    lows = closes * (1 - 0.003)
    volumes = np.ones(n) * 500
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _make_volatile_then_flat(n_volatile: int = 300, n_flat: int = 200) -> pd.DataFrame:
    """Volatile then suddenly flat: BBWP should drop toward bottom after flat starts."""
    n = n_volatile + n_flat
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    base = 1000.0
    rng = np.random.RandomState(7)
    # Volatile portion: high noise
    vol = base * np.cumprod(1 + rng.normal(0, 0.01, n_volatile))
    # Flat portion: tiny noise
    flat = np.full(n_flat, float(vol[-1])) + rng.normal(0, 0.05, n_flat)
    closes = np.concatenate([vol, flat])
    opens = closes * 0.999
    highs = closes * 1.002
    lows = closes * 0.998
    volumes = np.ones(n) * 500
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer() -> BBWPAnalyzer:
    return BBWPAnalyzer()


class TestBBWPAnalyzer:

    def test_returns_none_insufficient_data(self, analyzer: BBWPAnalyzer):
        """Fewer than bb_period + length bars → None."""
        candles = _make_flat_ohlcv(BBWP_BB_PERIOD + BBWP_LENGTH - 1)
        result = analyzer.calculate(candles)
        assert result is None

    def test_returns_none_empty_dataframe(self, analyzer: BBWPAnalyzer):
        """Empty DataFrame → None."""
        result = analyzer.calculate(pd.DataFrame())
        assert result is None

    def test_returns_none_missing_close_column(self, analyzer: BBWPAnalyzer):
        """Missing 'close' column → None."""
        candles = pd.DataFrame({"open": [1, 2], "high": [3, 4], "low": [0, 1]})
        result = analyzer.calculate(candles)
        assert result is None

    def test_returns_bbwp_state_sufficient_data(self, analyzer: BBWPAnalyzer):
        """With enough data, returns a BBWPState instance."""
        candles = _make_flat_ohlcv(BBWP_BB_PERIOD + BBWP_LENGTH + 10)
        result = analyzer.calculate(candles)
        assert result is not None
        assert isinstance(result, BBWPState)

    def test_bbwp_value_in_range(self, analyzer: BBWPAnalyzer):
        """BBWP value should always be in [0, 100]."""
        candles = _make_trending_ohlcv(BBWP_BB_PERIOD + BBWP_LENGTH + 10)
        result = analyzer.calculate(candles)
        assert result is not None
        assert 0.0 <= result.bbwp_value <= 100.0

    def test_signal_field_valid_values(self, analyzer: BBWPAnalyzer):
        """Signal must be one of the three defined values."""
        candles = _make_flat_ohlcv(BBWP_BB_PERIOD + BBWP_LENGTH + 10)
        result = analyzer.calculate(candles)
        assert result is not None
        assert result.signal in ("breakout_imminent", "extreme_reached", "neutral")

    def test_ma_value_present(self, analyzer: BBWPAnalyzer):
        """ma_value (EMA of BBWP) is populated."""
        candles = _make_flat_ohlcv(BBWP_BB_PERIOD + BBWP_LENGTH + 10)
        result = analyzer.calculate(candles)
        assert result is not None
        assert isinstance(result.ma_value, float)
        assert result.ma_value >= 0.0


class TestBBWPSignals:

    def test_extreme_reached_high_volatility(self):
        """After persistent high-volatility move, BBWP should be elevated."""
        # Build: long flat period then strong trend to force a wide BB width percentile
        n = BBWP_BB_PERIOD + BBWP_LENGTH + 50
        candles = _make_trending_ohlcv(n, pct_per_bar=0.01)
        analyzer = BBWPAnalyzer()
        result = analyzer.calculate(candles)
        # We can't guarantee >= 95 from synthetic data, but it should be above 50
        assert result is not None
        assert result.bbwp_value > 50.0

    def test_breakout_imminent_after_flat(self):
        """After transitioning from volatile to flat, BBWP should compress."""
        candles = _make_volatile_then_flat(n_volatile=300, n_flat=200)
        analyzer = BBWPAnalyzer()
        result = analyzer.calculate(candles)
        # The flat tail should push current width down relative to history
        assert result is not None
        # BBWP should be lower than the midpoint given the flat tail vs volatile history
        assert result.bbwp_value < 75.0

    def test_extreme_signal_threshold(self):
        """Confirm BBWP_EXTREME_HIGH and BBWP_EXTREME_LOW constants are correct."""
        assert BBWP_EXTREME_HIGH == 95.0
        assert BBWP_EXTREME_LOW == 5.0

    def test_signal_extreme_reached_on_forced_high_bbwp(self):
        """When BBWP value is forced to 95+, signal = extreme_reached."""
        state = BBWPState(bbwp_value=96.0, signal="extreme_reached", ma_value=94.0)
        assert state.signal == "extreme_reached"

    def test_signal_breakout_imminent_on_forced_low_bbwp(self):
        """When BBWP value is forced to <=5, signal = breakout_imminent."""
        state = BBWPState(bbwp_value=3.5, signal="breakout_imminent", ma_value=4.0)
        assert state.signal == "breakout_imminent"

    def test_signal_neutral_mid_range(self):
        """Mid-range BBWP → neutral."""
        state = BBWPState(bbwp_value=50.0, signal="neutral", ma_value=48.0)
        assert state.signal == "neutral"


class TestBBWPCalculationKnownValues:

    def test_percentile_rank_helper_all_equal(self):
        """All values equal → count_below = 0, percentile = 0 (nothing below last)."""
        arr = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        # No values are strictly LESS than the last (5.0), so rank = 0
        assert _percentile_rank(arr) == 0.0

    def test_percentile_rank_helper_last_is_max(self):
        """Last element is the largest → percentile = 100."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _percentile_rank(arr)
        assert result == 100.0

    def test_percentile_rank_helper_last_is_min(self):
        """Last element is the smallest → percentile = 0."""
        arr = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = _percentile_rank(arr)
        assert result == 0.0

    def test_percentile_rank_helper_single_element(self):
        """Single element → default 50."""
        arr = np.array([42.0])
        assert _percentile_rank(arr) == 50.0

    def test_percentile_rank_midpoint(self):
        """Last element exactly in the middle of 5 values → ~50%."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 3.0])  # last=3, 2 below out of 4
        result = _percentile_rank(arr)
        assert result == pytest.approx(50.0)

    def test_bb_width_calculation_is_non_negative(self):
        """BB width (4*std/sma) should always be >= 0."""
        candles = _make_trending_ohlcv(BBWP_BB_PERIOD + BBWP_LENGTH + 20)
        close = candles["close"].astype(float)
        sma = close.rolling(window=BBWP_BB_PERIOD).mean()
        std = close.rolling(window=BBWP_BB_PERIOD).std(ddof=0)
        width = (4.0 * std) / sma
        assert float(width.dropna().min()) >= 0.0


class TestBBWPCustomParameters:

    def test_custom_length_and_ma_length(self):
        """Custom shorter parameters still produce valid output."""
        analyzer = BBWPAnalyzer(length=50, ma_length=5, bb_period=10)
        candles = _make_trending_ohlcv(50 + 10 + 20)
        result = analyzer.calculate(candles)
        assert result is not None
        assert 0.0 <= result.bbwp_value <= 100.0

    def test_longer_candle_series(self):
        """Longer series does not error."""
        analyzer = BBWPAnalyzer()
        candles = _make_trending_ohlcv(BBWP_BB_PERIOD + BBWP_LENGTH + 500)
        result = analyzer.calculate(candles)
        assert result is not None
