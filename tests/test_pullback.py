"""Tests for PullbackAnalyzer — validates pullback detection after displacement."""
import pandas as pd
import pytest

from src.strategy.pullback import PullbackAnalyzer


def _make_candles(data: list[dict]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of dicts."""
    df = pd.DataFrame(data)
    df.index = range(len(df))
    return df


@pytest.fixture
def analyzer():
    return PullbackAnalyzer(min_retracement=0.20, max_retracement=0.78)


class TestPullbackAnalyzer:
    def test_valid_bullish_pullback(self, analyzer):
        """Price thrust up from displacement, then pulled back ~45%."""
        candles = _make_candles([
            # Pre-displacement candles
            {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 100},
            {"open": 101, "high": 103, "low": 100, "close": 102, "volume": 100},
            # Displacement candle (idx=2): big bullish move
            {"open": 100, "high": 110, "low": 99, "close": 109, "volume": 500},
            # Post-displacement: thrust continues
            {"open": 109, "high": 115, "low": 108, "close": 114, "volume": 300},
            # Pullback candle: retraces
            {"open": 114, "high": 114, "low": 107, "close": 108, "volume": 150},
        ])
        result = analyzer.analyze(candles, displacement_candle_idx=2, direction="bullish")
        # Move: disp_open=100, thrust_extreme=115, current=108
        # Retracement = (115-108)/(115-100) = 7/15 = 0.467
        assert result.pullback_detected is True
        assert result.pullback_status == "optimal"
        assert 0.40 <= result.retracement_pct <= 0.50

    def test_valid_bearish_pullback(self, analyzer):
        """Price thrust down from displacement, then pulled back ~40%."""
        candles = _make_candles([
            {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 100},
            {"open": 101, "high": 103, "low": 100, "close": 102, "volume": 100},
            # Displacement candle (idx=2): big bearish move
            {"open": 110, "high": 111, "low": 101, "close": 102, "volume": 500},
            # Post-displacement: thrust continues down
            {"open": 102, "high": 103, "low": 95, "close": 96, "volume": 300},
            # Pullback candle: retraces up
            {"open": 96, "high": 103, "low": 96, "close": 102, "volume": 150},
        ])
        result = analyzer.analyze(candles, displacement_candle_idx=2, direction="bearish")
        # Move: disp_open=110, thrust_extreme=95, current=102
        # Retracement = (102-95)/(110-95) = 7/15 = 0.467
        assert result.pullback_detected is True
        assert result.pullback_status == "optimal"
        assert 0.40 <= result.retracement_pct <= 0.50

    def test_no_pullback_yet_waiting(self, analyzer):
        """Displacement just happened, price still at thrust peak."""
        candles = _make_candles([
            {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 100},
            # Displacement candle (idx=1)
            {"open": 100, "high": 110, "low": 99, "close": 109, "volume": 500},
            # Price still near the peak — no pullback
            {"open": 109, "high": 115, "low": 108, "close": 114, "volume": 300},
        ])
        result = analyzer.analyze(candles, displacement_candle_idx=1, direction="bullish")
        # Retracement = (115-114)/(115-100) = 1/15 = 0.067 < 0.20
        assert result.pullback_detected is False
        assert result.pullback_status == "waiting"

    def test_shallow_pullback_waiting(self, analyzer):
        """Price pulled back only 15% — below 20% threshold."""
        candles = _make_candles([
            {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 100},
            # Displacement (idx=1)
            {"open": 100, "high": 108, "low": 99, "close": 107, "volume": 500},
            # Thrust
            {"open": 107, "high": 120, "low": 106, "close": 119, "volume": 300},
            # Tiny pullback: 15% of move
            {"open": 119, "high": 119, "low": 116, "close": 117, "volume": 100},
        ])
        result = analyzer.analyze(candles, displacement_candle_idx=1, direction="bullish")
        # Move: 100→120, current=117. Retracement = 3/20 = 0.15
        assert result.pullback_detected is False
        assert result.pullback_status == "waiting"

    def test_too_deep_pullback_failed(self, analyzer):
        """Price pulled back 85% — setup is failing."""
        candles = _make_candles([
            {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 100},
            # Displacement (idx=1)
            {"open": 100, "high": 110, "low": 99, "close": 109, "volume": 500},
            # Thrust
            {"open": 109, "high": 120, "low": 108, "close": 119, "volume": 300},
            # Deep pullback: 85% of move
            {"open": 119, "high": 119, "low": 101, "close": 103, "volume": 200},
        ])
        result = analyzer.analyze(candles, displacement_candle_idx=1, direction="bullish")
        # Move: 100→120, current=103. Retracement = 17/20 = 0.85
        assert result.pullback_detected is False
        assert result.pullback_status == "failed"

    def test_displacement_is_last_candle_waiting(self, analyzer):
        """Displacement is the most recent candle — no pullback possible."""
        candles = _make_candles([
            {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 100},
            # Displacement is the last candle (idx=1)
            {"open": 100, "high": 110, "low": 99, "close": 109, "volume": 500},
        ])
        result = analyzer.analyze(candles, displacement_candle_idx=1, direction="bullish")
        assert result.pullback_detected is False
        assert result.pullback_status == "waiting"

    def test_empty_candles(self, analyzer):
        """Empty DataFrame returns safe default."""
        result = analyzer.analyze(pd.DataFrame(), displacement_candle_idx=0, direction="bullish")
        assert result.pullback_detected is False

    def test_invalid_displacement_idx(self, analyzer):
        """Out-of-bounds displacement index returns safe default."""
        candles = _make_candles([
            {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 100},
        ])
        result = analyzer.analyze(candles, displacement_candle_idx=5, direction="bullish")
        assert result.pullback_detected is False

    def test_deep_but_valid_pullback(self, analyzer):
        """Price pulled back 75% — still within valid zone (under 78%)."""
        candles = _make_candles([
            {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 100},
            # Displacement (idx=1)
            {"open": 100, "high": 110, "low": 99, "close": 109, "volume": 500},
            # Thrust
            {"open": 109, "high": 120, "low": 108, "close": 119, "volume": 300},
            # Deep but valid: 75% of move
            {"open": 119, "high": 119, "low": 104, "close": 105, "volume": 200},
        ])
        result = analyzer.analyze(candles, displacement_candle_idx=1, direction="bullish")
        # Move: 100→120, current=105. Retracement = 15/20 = 0.75
        assert result.pullback_detected is True
        assert result.pullback_status == "optimal"
        assert 0.70 <= result.retracement_pct <= 0.80

    def test_retracement_values_returned(self, analyzer):
        """Check that displacement_open, thrust_extreme, current_price are correct."""
        candles = _make_candles([
            {"open": 50, "high": 55, "low": 49, "close": 52, "volume": 100},
            # Displacement (idx=1), open=50
            {"open": 50, "high": 60, "low": 49, "close": 59, "volume": 500},
            # Thrust peak at 70
            {"open": 59, "high": 70, "low": 58, "close": 68, "volume": 300},
            # Pullback to 60
            {"open": 68, "high": 68, "low": 58, "close": 60, "volume": 150},
        ])
        result = analyzer.analyze(candles, displacement_candle_idx=1, direction="bullish")
        assert result.displacement_open == 50.0
        assert result.thrust_extreme == 70.0
        assert result.current_price == 60.0
        assert result.optimal_entry == 60.0
        # Retracement = (70-60)/(70-50) = 10/20 = 0.50
        assert result.retracement_pct == 0.50
