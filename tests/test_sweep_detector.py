"""Tests for the SweepDetector (Trade Travel Chill core signal)."""
import numpy as np
import pandas as pd
import pytest

from src.strategy.sweep_detector import SweepDetector


@pytest.fixture
def detector():
    return SweepDetector()


def _make_candles(n=20, base=100.0, seed=42):
    """Create a basic 1H candle DataFrame."""
    np.random.seed(seed)
    data = {"open": [], "high": [], "low": [], "close": [], "volume": []}
    for i in range(n):
        o = base + np.random.normal(0, 0.5)
        c = base + np.random.normal(0, 0.5)
        h = max(o, c) + abs(np.random.normal(0, 0.3))
        l = min(o, c) - abs(np.random.normal(0, 0.3))
        data["open"].append(o)
        data["high"].append(h)
        data["low"].append(max(l, 0.01))
        data["close"].append(c)
        data["volume"].append(np.random.uniform(1000, 5000))

    timestamps = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(data, index=timestamps)


class TestSweepDetector:
    def test_asian_low_sweep_detected(self, detector):
        """Candle wicks below Asian low, closes above -> bullish sweep."""
        candles = _make_candles(20, base=100.0)

        # Inject a sweep candle at index -2 (last completed candle)
        # Wick below 95 (Asian low), close back above 95
        idx = len(candles) - 2
        candles.iloc[idx, candles.columns.get_loc("low")] = 94.0    # Below Asian low
        candles.iloc[idx, candles.columns.get_loc("close")] = 96.0  # Back above
        candles.iloc[idx, candles.columns.get_loc("high")] = 97.0

        result = detector.detect(
            candles_1h=candles,
            asian_high=105.0,
            asian_low=95.0,
        )

        assert result.sweep_detected is True
        assert result.sweep_direction == "bullish"
        assert result.sweep_type == "asian_low"
        assert result.sweep_level == 94.0  # Wick extreme
        assert result.target_level == 105.0  # Opposite side (Asian high)
        assert result.sweep_depth == 1.0  # 95 - 94 = 1

    def test_asian_high_sweep_detected(self, detector):
        """Candle wicks above Asian high, closes below -> bearish sweep."""
        candles = _make_candles(20, base=100.0)

        idx = len(candles) - 2
        candles.iloc[idx, candles.columns.get_loc("high")] = 106.0   # Above Asian high
        candles.iloc[idx, candles.columns.get_loc("close")] = 104.0  # Back below
        candles.iloc[idx, candles.columns.get_loc("low")] = 103.0

        result = detector.detect(
            candles_1h=candles,
            asian_high=105.0,
            asian_low=95.0,
        )

        assert result.sweep_detected is True
        assert result.sweep_direction == "bearish"
        assert result.sweep_type == "asian_high"
        assert result.sweep_level == 106.0  # Wick extreme
        assert result.target_level == 95.0  # Opposite side (Asian low)

    def test_swing_low_sweep_detected(self, detector):
        """Candle wicks below swing low, closes above -> bullish sweep."""
        candles = _make_candles(20, base=100.0)

        idx = len(candles) - 2
        candles.iloc[idx, candles.columns.get_loc("low")] = 89.0    # Below swing low
        candles.iloc[idx, candles.columns.get_loc("close")] = 91.0  # Back above
        candles.iloc[idx, candles.columns.get_loc("high")] = 92.0

        result = detector.detect(
            candles_1h=candles,
            asian_high=0.0,  # No Asian range
            asian_low=0.0,
            swing_high=110.0,
            swing_low=90.0,
        )

        assert result.sweep_detected is True
        assert result.sweep_direction == "bullish"
        assert result.sweep_type == "swing_low"
        assert result.sweep_level == 89.0
        assert result.target_level == 110.0

    def test_swing_high_sweep_detected(self, detector):
        """Candle wicks above swing high, closes below -> bearish sweep."""
        candles = _make_candles(20, base=100.0)

        idx = len(candles) - 2
        candles.iloc[idx, candles.columns.get_loc("high")] = 111.0  # Above swing high
        candles.iloc[idx, candles.columns.get_loc("close")] = 109.0  # Back below
        candles.iloc[idx, candles.columns.get_loc("low")] = 108.0

        result = detector.detect(
            candles_1h=candles,
            asian_high=0.0,
            asian_low=0.0,
            swing_high=110.0,
            swing_low=90.0,
        )

        assert result.sweep_detected is True
        assert result.sweep_direction == "bearish"
        assert result.sweep_type == "swing_high"
        assert result.sweep_level == 111.0

    def test_no_sweep_when_close_stays_below(self, detector):
        """Candle wicks below AND closes below -> NOT a completed sweep."""
        candles = _make_candles(20, base=100.0)

        idx = len(candles) - 2
        candles.iloc[idx, candles.columns.get_loc("low")] = 94.0    # Below Asian low
        candles.iloc[idx, candles.columns.get_loc("close")] = 94.5  # Still below!
        candles.iloc[idx, candles.columns.get_loc("high")] = 95.5

        result = detector.detect(
            candles_1h=candles,
            asian_high=105.0,
            asian_low=95.0,
        )

        assert result.sweep_detected is False

    def test_no_sweep_when_no_wick_through(self, detector):
        """Price stays within range -> no sweep."""
        candles = _make_candles(20, base=100.0)

        # Ensure last candles don't penetrate Asian range
        for offset in [-2, -3, -4]:
            idx = len(candles) + offset
            candles.iloc[idx, candles.columns.get_loc("low")] = 96.0
            candles.iloc[idx, candles.columns.get_loc("high")] = 104.0
            candles.iloc[idx, candles.columns.get_loc("close")] = 100.0

        result = detector.detect(
            candles_1h=candles,
            asian_high=105.0,
            asian_low=95.0,
        )

        assert result.sweep_detected is False

    def test_lookback_window(self, detector):
        """Only checks last 3 completed candles, not further back."""
        candles = _make_candles(20, base=100.0)

        # Put sweep at index -5 (outside lookback of 3)
        idx = len(candles) - 5
        candles.iloc[idx, candles.columns.get_loc("low")] = 94.0
        candles.iloc[idx, candles.columns.get_loc("close")] = 96.0
        candles.iloc[idx, candles.columns.get_loc("high")] = 97.0

        # Ensure indices -2, -3, -4 are clean
        for offset in [-2, -3, -4]:
            i = len(candles) + offset
            candles.iloc[i, candles.columns.get_loc("low")] = 96.0
            candles.iloc[i, candles.columns.get_loc("high")] = 104.0
            candles.iloc[i, candles.columns.get_loc("close")] = 100.0

        result = detector.detect(
            candles_1h=candles,
            asian_high=105.0,
            asian_low=95.0,
            lookback=3,
        )

        assert result.sweep_detected is False

    def test_asian_sweep_priority_over_swing(self, detector):
        """Asian range sweep takes priority over swing level sweep."""
        candles = _make_candles(20, base=100.0)

        # Both Asian low and swing low swept in the same candle
        idx = len(candles) - 2
        candles.iloc[idx, candles.columns.get_loc("low")] = 84.0    # Below both
        candles.iloc[idx, candles.columns.get_loc("close")] = 96.0  # Back above both
        candles.iloc[idx, candles.columns.get_loc("high")] = 97.0

        result = detector.detect(
            candles_1h=candles,
            asian_high=105.0,
            asian_low=95.0,
            swing_high=110.0,
            swing_low=90.0,
        )

        assert result.sweep_detected is True
        assert result.sweep_type == "asian_low"  # Asian takes priority

    def test_empty_candles(self, detector):
        """Empty DataFrame returns no sweep."""
        result = detector.detect(
            candles_1h=pd.DataFrame(),
            asian_high=105.0,
            asian_low=95.0,
        )
        assert result.sweep_detected is False

    def test_no_levels_provided(self, detector):
        """No levels at all -> no sweep."""
        candles = _make_candles(20, base=100.0)
        result = detector.detect(
            candles_1h=candles,
            asian_high=0.0,
            asian_low=0.0,
        )
        assert result.sweep_detected is False
