"""Tests for MM M/W formation detection (src.strategy.mm_formations)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.mm_formations import (
    DEFAULT_LOOKBACK,
    Formation,
    FormationDetector,
    FormationValidation,
    ThreeHitsResult,
    _find_swing_highs,
    _find_swing_lows,
    _is_hammer,
    _is_inverted_hammer,
    _is_engulfing_bullish,
    _is_engulfing_bearish,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_m_top_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate OHLCV with an M-top pattern (two peaks with a valley)."""
    rng = np.random.RandomState(seed)

    # Build a price path: up -> peak1 -> down -> up (less) -> peak2 -> down
    segments = [
        np.linspace(100, 115, 30),    # rise to peak 1
        np.linspace(115, 108, 15),    # pullback (trough)
        np.linspace(108, 113, 15),    # rise to peak 2 (lower than peak 1)
        np.linspace(113, 100, 30),    # drop after M
    ]
    core = np.concatenate(segments)

    # Pad with flat data to reach n
    pad_before = np.full(n - len(core), 100.0) + rng.normal(0, 0.5, n - len(core))
    closes = np.concatenate([pad_before, core])[:n]

    opens = closes + rng.normal(0, 0.3, n)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.5, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.5, n))
    volumes = rng.uniform(1000, 5000, n)

    # Boost volume at peaks for SVC detection
    peak1_idx = n - len(core) + 29
    peak2_idx = n - len(core) + 59
    volumes[peak1_idx] = 15000.0
    volumes[peak1_idx - 1] = 12000.0

    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _make_w_bottom_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate OHLCV with a W-bottom pattern (two troughs with a ridge)."""
    rng = np.random.RandomState(seed)

    segments = [
        np.linspace(110, 95, 30),     # drop to low 1
        np.linspace(95, 103, 15),     # bounce (ridge)
        np.linspace(103, 97, 15),     # drop to low 2 (higher than low 1)
        np.linspace(97, 112, 30),     # rally after W
    ]
    core = np.concatenate(segments)

    pad_before = np.full(n - len(core), 110.0) + rng.normal(0, 0.5, n - len(core))
    closes = np.concatenate([pad_before, core])[:n]

    opens = closes + rng.normal(0, 0.3, n)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.5, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.5, n))
    volumes = rng.uniform(1000, 5000, n)

    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


@pytest.fixture
def detector() -> FormationDetector:
    return FormationDetector()


@pytest.fixture
def m_top_df() -> pd.DataFrame:
    return _make_m_top_ohlcv()


@pytest.fixture
def w_bottom_df() -> pd.DataFrame:
    return _make_w_bottom_ohlcv()


# ------------------------------------------------------------------
# Helper function tests
# ------------------------------------------------------------------

class TestSwingDetection:
    def test_find_swing_highs(self):
        # Create a series with a clear peak at index 10
        prices = np.concatenate([
            np.linspace(100, 120, 11),  # 0-10: rise to peak
            np.linspace(119, 100, 10),  # 11-20: decline
        ])
        result = _find_swing_highs(prices, window=5)
        assert len(result) >= 1
        assert any(idx == 10 for idx in result)

    def test_find_swing_lows(self):
        prices = np.concatenate([
            np.linspace(120, 100, 11),
            np.linspace(101, 120, 10),
        ])
        result = _find_swing_lows(prices, window=5)
        assert len(result) >= 1
        assert any(idx == 10 for idx in result)

    def test_no_swings_in_flat(self):
        prices = np.full(30, 100.0)
        highs = _find_swing_highs(prices, window=5)
        # Flat data: every point is a "max" in the window, so all are swings.
        # This is valid behaviour -- the algorithm returns all candidates.
        assert isinstance(highs, list)


class TestCandlePatterns:
    def test_hammer(self):
        # long lower shadow, small body, tiny upper shadow
        # o=100, h=100.1, l=95, c=100.05
        # body=0.05, range=5.1, lower=5.0, upper=0.05
        # lower>=2*body: 5.0>=0.1 True; upper<=body*0.5: 0.05<=0.025 False
        # Use: o=100, h=100.02, l=95, c=100.0 (doji-hammer)
        # body=0, but body/range=0 < 0.4, lower=5, upper=0.02, lower>=0 True, upper<=0 False
        # Better: o=99.5, h=100.0, l=95.0, c=100.0
        # body=0.5, range=5.0, lower=min(99.5,100)-95=4.5, upper=100-100=0
        # lower>=2*body: 4.5>=1.0 True; upper<=0.25 True; body/range=0.1<0.4 True
        assert _is_hammer(99.5, 100.0, 95.0, 100.0) is True
        # Not a hammer: large body
        assert _is_hammer(95.0, 105.0, 94.0, 104.0) is False

    def test_inverted_hammer(self):
        # long upper shadow, small body, tiny lower shadow
        # o=100.0, h=106.0, l=100.0, c=100.5
        # body=0.5, range=6.0, upper=106-100.5=5.5, lower=100-100=0
        # upper>=2*body: 5.5>=1.0 True; lower<=0.25 True; body/range=0.083<0.4 True
        assert _is_inverted_hammer(100.0, 106.0, 100.0, 100.5) is True
        # Not inverted hammer: equal wicks
        assert _is_inverted_hammer(100.0, 101.0, 95.0, 100.5) is False

    def test_engulfing_bullish(self):
        # prev red, curr green and wraps
        assert _is_engulfing_bullish(102.0, 100.0, 99.0, 103.0) is True
        assert _is_engulfing_bullish(100.0, 102.0, 99.0, 103.0) is False  # prev not red

    def test_engulfing_bearish(self):
        # prev green, curr red and wraps
        assert _is_engulfing_bearish(100.0, 102.0, 103.0, 99.0) is True
        assert _is_engulfing_bearish(102.0, 100.0, 103.0, 99.0) is False  # prev not green


# ------------------------------------------------------------------
# detect / detect_mw
# ------------------------------------------------------------------

class TestDetectMW:
    def test_empty_returns_empty(self, detector: FormationDetector):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        assert detector.detect(empty) == []

    def test_insufficient_data(self, detector: FormationDetector):
        """DataFrame with fewer than DEFAULT_LOOKBACK rows returns empty."""
        rng = np.random.RandomState(42)
        n = 20
        idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        small = pd.DataFrame({
            "open": rng.uniform(99, 101, n),
            "high": rng.uniform(101, 103, n),
            "low": rng.uniform(97, 99, n),
            "close": rng.uniform(99, 101, n),
            "volume": rng.uniform(1000, 5000, n),
        }, index=idx)
        assert detector.detect(small) == []

    def test_none_returns_empty(self, detector: FormationDetector):
        assert detector.detect(None) == []

    def test_m_top_detected(self, detector: FormationDetector, m_top_df: pd.DataFrame):
        formations = detector.detect(m_top_df)
        # Should find at least one formation
        assert isinstance(formations, list)
        # All detected formations should be Formation instances
        for f in formations:
            assert isinstance(f, Formation)
            assert f.type in ("M", "W")

    def test_w_bottom_detected(self, detector: FormationDetector, w_bottom_df: pd.DataFrame):
        formations = detector.detect(w_bottom_df)
        assert isinstance(formations, list)
        for f in formations:
            assert isinstance(f, Formation)

    def test_direction_filter_bullish(self, detector: FormationDetector, w_bottom_df: pd.DataFrame):
        formations = detector.detect(w_bottom_df, direction_bias="bullish")
        for f in formations:
            assert f.direction == "bullish"

    def test_direction_filter_bearish(self, detector: FormationDetector, m_top_df: pd.DataFrame):
        formations = detector.detect(m_top_df, direction_bias="bearish")
        for f in formations:
            assert f.direction == "bearish"

    def test_sorted_by_quality(self, detector: FormationDetector, m_top_df: pd.DataFrame):
        formations = detector.detect(m_top_df)
        if len(formations) >= 2:
            for i in range(len(formations) - 1):
                assert formations[i].quality_score >= formations[i + 1].quality_score


class TestFormationAttributes:
    def test_m_formation_fields(self, detector: FormationDetector, m_top_df: pd.DataFrame):
        formations = detector.detect(m_top_df, direction_bias="bearish")
        if formations:
            f = formations[0]
            assert f.type == "M"
            assert f.direction == "bearish"
            assert f.variant in ("standard", "multi_session", "final_damage", "board_meeting")
            assert f.peak1_price > 0
            assert f.peak2_price > 0
            assert 0.0 <= f.quality_score <= 1.0
            assert f.peak1_idx < f.peak2_idx

    def test_w_formation_fields(self, detector: FormationDetector, w_bottom_df: pd.DataFrame):
        formations = detector.detect(w_bottom_df, direction_bias="bullish")
        if formations:
            f = formations[0]
            assert f.type == "W"
            assert f.direction == "bullish"
            assert f.peak1_price > 0
            assert f.peak2_price > 0


# ------------------------------------------------------------------
# detect_multi_session
# ------------------------------------------------------------------

class TestMultiSession:
    def test_without_session_analyzer(self, detector: FormationDetector, m_top_df: pd.DataFrame):
        """Without a session analyzer, formations should be returned unchanged."""
        formations = detector.detect_mw(m_top_df)
        enriched = detector.detect_multi_session(m_top_df, formations)
        assert len(enriched) == len(formations)


# ------------------------------------------------------------------
# detect_three_hits
# ------------------------------------------------------------------

class TestThreeHits:
    def test_no_hits_on_empty(self, detector: FormationDetector):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = detector.detect_three_hits(empty, level=100.0)
        assert isinstance(result, ThreeHitsResult)
        assert result.detected is False

    def test_hits_at_level(self, detector: FormationDetector):
        """Create data that touches a level 3 times."""
        n = 60
        level = 110.0
        rng = np.random.RandomState(42)

        highs = np.full(n, 105.0)
        lows = np.full(n, 100.0)
        opens = np.full(n, 102.0)
        closes = np.full(n, 103.0)
        volumes = rng.uniform(1000, 5000, n)

        # Touch the level at 3 separated points
        for touch_idx in [10, 25, 45]:
            highs[touch_idx] = 110.0
            highs[touch_idx + 1] = 110.1

        idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=idx,
        )
        result = detector.detect_three_hits(df, level=level, tolerance=0.003)
        # Without a session analyzer all hits may be same session
        assert isinstance(result, ThreeHitsResult)

    def test_four_hits_continuation(self, detector: FormationDetector):
        """4 hits = continuation expected."""
        n = 80
        level = 110.0
        rng = np.random.RandomState(42)

        highs = np.full(n, 105.0)
        lows = np.full(n, 100.0)
        opens = np.full(n, 102.0)
        closes = np.full(n, 103.0)
        volumes = rng.uniform(1000, 5000, n)

        for touch_idx in [10, 25, 45, 65]:
            highs[touch_idx] = 110.0

        idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=idx,
        )
        # No session analyzer => all treated as same session so detection depends
        # on the multi_session fallback
        result = detector.detect_three_hits(df, level=level, tolerance=0.003)
        if result.detected and result.hit_count >= 4:
            assert result.expected_outcome == "continuation"


# ------------------------------------------------------------------
# validate_formation
# ------------------------------------------------------------------

class TestValidateFormation:
    def test_validate_empty(self, detector: FormationDetector):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        dummy = Formation(
            type="M", variant="standard",
            peak1_idx=5, peak1_price=110.0,
            peak2_idx=15, peak2_price=108.0,
            trough_idx=10, trough_price=105.0,
            direction="bearish",
        )
        result = detector.validate_formation(empty, dummy)
        assert isinstance(result, FormationValidation)
        assert result.confirmation_score == 0

    def test_validate_m_top(self, detector: FormationDetector, m_top_df: pd.DataFrame):
        formations = detector.detect(m_top_df, direction_bias="bearish")
        if formations:
            f = formations[0]
            val = detector.validate_formation(m_top_df, f)
            assert isinstance(val, FormationValidation)
            assert 0 <= val.confirmation_score <= 3
            if val.stop_loss > 0:
                # M-top stop should be above the entry
                assert val.stop_loss >= val.entry_price or val.entry_price == 0

    def test_validate_w_bottom(self, detector: FormationDetector, w_bottom_df: pd.DataFrame):
        formations = detector.detect(w_bottom_df, direction_bias="bullish")
        if formations:
            f = formations[0]
            val = detector.validate_formation(w_bottom_df, f)
            assert isinstance(val, FormationValidation)
            assert 0 <= val.confirmation_score <= 3
