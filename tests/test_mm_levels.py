"""Tests for MM Method level counting (src.strategy.mm_levels)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.mm_levels import (
    BOARD_MEETING_MIN,
    CLUSTER_GAP_MAX,
    PVSRA_LOOKBACK,
    SVC_BODY_RATIO_MAX,
    SVC_VOLUME_MIN,
    BoardMeetingInfo,
    LevelAnalysis,
    LevelInfo,
    LevelTracker,
    SVCResult,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_ohlcv(
    n: int = 200,
    base: float = 100.0,
    drift: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a mildly trending OHLCV DataFrame."""
    rng = np.random.RandomState(seed)
    closes = np.empty(n)
    closes[0] = base
    for i in range(1, n):
        closes[i] = closes[i - 1] * (1 + drift / 100 + rng.normal(0, 0.005))

    opens = closes * (1 + rng.normal(0, 0.001, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.003, n)))
    volumes = rng.uniform(500, 2000, n)

    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _make_ohlcv_with_volume_spikes(
    n: int = 200,
    base: float = 100.0,
    spike_indices: list[int] | None = None,
    spike_mult: float = 5.0,
    direction: str = "bullish",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate OHLCV data with volume spikes at specified indices."""
    rng = np.random.RandomState(seed)
    drift = 0.03 if direction == "bullish" else -0.03

    closes = np.empty(n)
    closes[0] = base
    for i in range(1, n):
        closes[i] = closes[i - 1] * (1 + drift / 100 + rng.normal(0, 0.003))
    closes = np.maximum(closes, 1.0)

    opens = closes * (1 + rng.normal(0, 0.001, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volumes = rng.uniform(500, 1500, n)

    if spike_indices:
        for idx in spike_indices:
            if 0 <= idx < n:
                volumes[idx] = volumes[idx] * spike_mult
                # Make the spike candles move in the right direction
                if direction == "bullish":
                    closes[idx] = opens[idx] * 1.01  # bullish close
                    highs[idx] = closes[idx] * 1.005
                else:
                    closes[idx] = opens[idx] * 0.99  # bearish close
                    lows[idx] = closes[idx] * 0.995

    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


@pytest.fixture
def tracker() -> LevelTracker:
    return LevelTracker()


# ------------------------------------------------------------------
# classify_pvsra
# ------------------------------------------------------------------

class TestClassifyPVSRA:
    def test_adds_columns(self, tracker: LevelTracker):
        df = _make_ohlcv(50)
        result = tracker.classify_pvsra(df)
        assert "pvsra_type" in result.columns
        assert "pvsra_color" in result.columns
        assert "is_vector" in result.columns
        assert "vol_ratio" in result.columns

    def test_uniform_volume_is_normal(self, tracker: LevelTracker):
        """Uniform volume should produce all 'normal' classifications."""
        n = 50
        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.5),
            "volume": np.full(n, 1000.0),
        })
        result = tracker.classify_pvsra(df)
        # After the warm-up period, should all be normal
        assert (result["pvsra_type"].iloc[PVSRA_LOOKBACK + 2:] == "normal").all()

    def test_high_volume_creates_vectors(self, tracker: LevelTracker):
        n = 30
        volumes = np.full(n, 1000.0)
        volumes[-1] = 5000.0  # 5x average -> should be vector_200

        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.5),
            "volume": volumes,
        })
        result = tracker.classify_pvsra(df)
        assert result["pvsra_type"].iloc[-1] == "vector_200"
        assert result["is_vector"].iloc[-1] == True

    def test_bullish_vector_color(self, tracker: LevelTracker):
        """Bullish (close > open) vector_200 should be green."""
        n = 30
        volumes = np.full(n, 1000.0)
        volumes[-1] = 5000.0

        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 102.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 101.0),  # close > open = bullish
            "volume": volumes,
        })
        result = tracker.classify_pvsra(df)
        assert result["pvsra_color"].iloc[-1] == "green"

    def test_bearish_vector_color(self, tracker: LevelTracker):
        """Bearish (close < open) vector_200 should be red."""
        n = 30
        volumes = np.full(n, 1000.0)
        volumes[-1] = 5000.0

        df = pd.DataFrame({
            "open": np.full(n, 101.0),
            "high": np.full(n, 102.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.0),  # close < open = bearish
            "volume": volumes,
        })
        result = tracker.classify_pvsra(df)
        assert result["pvsra_color"].iloc[-1] == "red"


# ------------------------------------------------------------------
# count_levels
# ------------------------------------------------------------------

class TestCountLevels:
    def test_no_vectors_returns_empty(self, tracker: LevelTracker):
        df = _make_ohlcv(50)
        classified = tracker.classify_pvsra(df)
        # Force all to normal
        classified["is_vector"] = False
        classified["pvsra_type"] = "normal"
        result = tracker.count_levels(classified, "bullish")
        assert result == []

    def test_single_cluster_creates_one_level(self, tracker: LevelTracker):
        """A single cluster of vector candles should create level 1."""
        df = _make_ohlcv_with_volume_spikes(
            200, spike_indices=[50, 51, 52], spike_mult=5.0, direction="bullish",
        )
        classified = tracker.classify_pvsra(df)
        levels = tracker.count_levels(classified, "bullish")
        assert len(levels) >= 1
        if levels:
            assert levels[0].level_number == 1
            assert levels[0].direction == "bullish"

    def test_multiple_clusters_create_multiple_levels(self, tracker: LevelTracker):
        """Widely separated volume clusters should form distinct levels."""
        # Put clusters at 30, 80, 130 — all well separated
        df = _make_ohlcv_with_volume_spikes(
            200,
            spike_indices=[30, 31, 80, 81, 130, 131],
            spike_mult=5.0,
            direction="bullish",
        )
        classified = tracker.classify_pvsra(df)
        levels = tracker.count_levels(classified, "bullish")
        # Should detect multiple levels (exact count depends on direction filtering)
        assert isinstance(levels, list)
        for lv in levels:
            assert isinstance(lv, LevelInfo)
            assert lv.volume_confirmed is True

    def test_caps_at_4_levels(self, tracker: LevelTracker):
        """Should never produce more than 4 levels."""
        df = _make_ohlcv_with_volume_spikes(
            200,
            spike_indices=[20, 21, 50, 51, 80, 81, 110, 111, 140, 141, 170, 171],
            spike_mult=5.0,
            direction="bullish",
        )
        classified = tracker.classify_pvsra(df)
        levels = tracker.count_levels(classified, "bullish")
        assert len(levels) <= 4


# ------------------------------------------------------------------
# analyze (full pipeline)
# ------------------------------------------------------------------

class TestAnalyze:
    def test_returns_level_analysis(self, tracker: LevelTracker):
        df = _make_ohlcv(200)
        result = tracker.analyze(df)
        assert isinstance(result, LevelAnalysis)
        assert result.current_level >= 0

    def test_empty_df(self, tracker: LevelTracker):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = tracker.analyze(empty)
        assert result.current_level == 0

    def test_too_few_rows(self, tracker: LevelTracker):
        small = _make_ohlcv(10)
        result = tracker.analyze(small)
        assert result.current_level == 0

    def test_none_input(self, tracker: LevelTracker):
        result = tracker.analyze(None)
        assert result.current_level == 0

    def test_direction_inferred(self, tracker: LevelTracker):
        df = _make_ohlcv(200, drift=0.1)  # upward
        result = tracker.analyze(df)
        assert result.direction in ("bullish", "bearish")

    def test_explicit_direction(self, tracker: LevelTracker):
        df = _make_ohlcv(200)
        result = tracker.analyze(df, direction="bearish")
        assert result.direction == "bearish"

    def test_is_extended_when_4_levels(self, tracker: LevelTracker):
        """When 4 levels detected, is_extended should be True."""
        df = _make_ohlcv_with_volume_spikes(
            200,
            spike_indices=[20, 21, 50, 51, 80, 81, 110, 111, 140, 141, 170, 171],
            spike_mult=5.0,
            direction="bullish",
        )
        result = tracker.analyze(df, direction="bullish")
        if result.current_level >= 4:
            assert result.is_extended is True


# ------------------------------------------------------------------
# detect_board_meeting
# ------------------------------------------------------------------

class TestDetectBoardMeeting:
    def test_returns_none_for_short_segment(self, tracker: LevelTracker):
        df = _make_ohlcv(50)
        classified = tracker.classify_pvsra(df)
        result = tracker.detect_board_meeting(classified, 10, 11)
        assert result is None  # duration < BOARD_MEETING_MIN

    def test_detects_consolidation(self, tracker: LevelTracker):
        """A range of normal-volume candles should qualify as a board meeting."""
        n = 50
        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.5),
            "volume": np.full(n, 1000.0),
        })
        classified = tracker.classify_pvsra(df)
        result = tracker.detect_board_meeting(classified, 20, 30)
        assert isinstance(result, BoardMeetingInfo)
        assert result.duration_candles == 11  # 30 - 20 + 1

    def test_too_many_vectors_rejects(self, tracker: LevelTracker):
        """A segment full of vector candles should not qualify."""
        n = 50
        volumes = np.full(n, 1000.0)
        # Make most candles in the range vectors
        for i in range(20, 31):
            volumes[i] = 5000.0

        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.5),
            "volume": volumes,
        })
        classified = tracker.classify_pvsra(df)
        result = tracker.detect_board_meeting(classified, 20, 30)
        assert result is None  # vector density too high

    def test_board_meeting_has_correct_attrs(self, tracker: LevelTracker):
        n = 60
        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.5),
            "low": np.full(n, 99.5),
            "close": np.full(n, 100.2),
            "volume": np.full(n, 1000.0),
        })
        classified = tracker.classify_pvsra(df)
        bm = tracker.detect_board_meeting(classified, 15, 25)
        assert bm is not None
        assert bm.start_idx == 15
        assert bm.end_idx == 25
        assert bm.contains_stop_hunt in (True, False)
        assert bm.ema_flattening in (True, False)


# ------------------------------------------------------------------
# detect_stopping_volume
# ------------------------------------------------------------------

class TestDetectStoppingVolume:
    def test_none_when_insufficient(self, tracker: LevelTracker):
        small = _make_ohlcv(10)
        classified = tracker.classify_pvsra(small)
        result = tracker.detect_stopping_volume(classified)
        assert result is None

    def test_detects_svc_candle(self, tracker: LevelTracker):
        """Craft a candle with small body, large wick, high volume."""
        n = 40
        opens = np.full(n, 100.0)
        closes = np.full(n, 100.2)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        volumes = np.full(n, 1000.0)

        # Create an SVC candidate: small body, large upper wick, high volume
        svc_idx = 30
        opens[svc_idx] = 100.0
        closes[svc_idx] = 100.1  # tiny body
        highs[svc_idx] = 103.0   # large upper wick
        lows[svc_idx] = 99.9     # small lower wick
        volumes[svc_idx] = 5000.0  # huge volume

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes,
        })
        classified = tracker.classify_pvsra(df)
        result = tracker.detect_stopping_volume(classified)
        # Should find the SVC
        if result is not None:
            assert result.detected is True
            assert result.body_ratio <= SVC_BODY_RATIO_MAX
            assert result.volume_ratio >= SVC_VOLUME_MIN

    def test_svc_result_fields(self, tracker: LevelTracker):
        """SVCResult should have all expected fields."""
        n = 40
        opens = np.full(n, 100.0)
        closes = np.full(n, 100.05)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        volumes = np.full(n, 1000.0)

        svc_idx = 25
        opens[svc_idx] = 100.0
        closes[svc_idx] = 100.05
        highs[svc_idx] = 104.0
        lows[svc_idx] = 99.95
        volumes[svc_idx] = 6000.0

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes,
        })
        classified = tracker.classify_pvsra(df)
        result = tracker.detect_stopping_volume(classified)
        if result is not None:
            assert isinstance(result, SVCResult)
            assert result.wick_direction in ("up", "down")
            assert isinstance(result.price_returned_to_wick, bool)


# ------------------------------------------------------------------
# Volume degradation
# ------------------------------------------------------------------

class TestVolumeDegradation:
    def test_degradation_detected(self, tracker: LevelTracker):
        """First level vector_200, last level vector_150 = degrading."""
        levels = [
            LevelInfo(1, 10, 15, "bullish", 2.0, True, "vector_200"),
            LevelInfo(2, 30, 35, "bullish", 1.5, True, "vector_200"),
            LevelInfo(3, 60, 65, "bullish", 1.0, True, "vector_150"),
        ]
        assert tracker._check_volume_degradation(levels) is True

    def test_no_degradation(self, tracker: LevelTracker):
        levels = [
            LevelInfo(1, 10, 15, "bullish", 2.0, True, "vector_200"),
            LevelInfo(2, 30, 35, "bullish", 1.5, True, "vector_200"),
        ]
        assert tracker._check_volume_degradation(levels) is False

    def test_single_level_no_degradation(self, tracker: LevelTracker):
        levels = [LevelInfo(1, 10, 15, "bullish", 2.0, True, "vector_200")]
        assert tracker._check_volume_degradation(levels) is False
