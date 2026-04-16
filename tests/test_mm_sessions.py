"""Tests for MM session timing (src.strategy.mm_sessions)."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

import numpy as np
import pandas as pd

from src.strategy.mm_sessions import (
    AsiaSpike,
    MMSessionAnalyzer,
    SessionInfo,
    _in_range,
    _to_ny,
)

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


@pytest.fixture
def analyzer() -> MMSessionAnalyzer:
    return MMSessionAnalyzer()


# ------------------------------------------------------------------
# _to_ny helper
# ------------------------------------------------------------------

class TestToNy:
    def test_naive_assumed_utc(self):
        naive = datetime(2026, 3, 10, 12, 0, 0)
        result = _to_ny(naive)
        assert result.tzinfo is not None
        # March 10 is after spring-forward: UTC-4 (EDT)
        assert result.hour == 8  # 12 UTC -> 8 EDT

    def test_aware_utc(self):
        aware = datetime(2026, 1, 15, 20, 0, 0, tzinfo=UTC)
        result = _to_ny(aware)
        # January: UTC-5 (EST)
        assert result.hour == 15

    def test_already_ny(self):
        ny_dt = datetime(2026, 6, 1, 10, 0, 0, tzinfo=NY)
        result = _to_ny(ny_dt)
        assert result.hour == 10


# ------------------------------------------------------------------
# _in_range helper
# ------------------------------------------------------------------

class TestInRange:
    def test_normal_range(self):
        assert _in_range(time(10, 0), time(9, 30), time(17, 0)) is True
        assert _in_range(time(9, 0), time(9, 30), time(17, 0)) is False

    def test_midnight_wrap(self):
        # Asia session 20:30 -> 03:00 wraps midnight
        assert _in_range(time(22, 0), time(20, 30), time(3, 0), wraps_midnight=True) is True
        assert _in_range(time(1, 0), time(20, 30), time(3, 0), wraps_midnight=True) is True
        assert _in_range(time(4, 0), time(20, 30), time(3, 0), wraps_midnight=True) is False

    def test_boundary_inclusive_start(self):
        assert _in_range(time(9, 30), time(9, 30), time(17, 0)) is True

    def test_boundary_exclusive_end(self):
        assert _in_range(time(17, 0), time(9, 30), time(17, 0)) is False


# ------------------------------------------------------------------
# get_current_session
# ------------------------------------------------------------------

class TestGetCurrentSession:
    def test_us_session(self, analyzer: MMSessionAnalyzer):
        # Tuesday 2pm NY = US session
        dt = datetime(2026, 3, 10, 14, 0, 0, tzinfo=NY)
        info = analyzer.get_current_session(dt)
        assert info.session_name == "us"
        assert info.is_gap is False
        assert info.day_of_week == 1  # Tuesday

    def test_asia_session_evening(self, analyzer: MMSessionAnalyzer):
        # Wednesday 10pm NY = Asia session (evening side)
        dt = datetime(2026, 3, 11, 22, 0, 0, tzinfo=NY)
        info = analyzer.get_current_session(dt)
        assert info.session_name == "asia"
        assert info.is_gap is False

    def test_asia_session_morning(self, analyzer: MMSessionAnalyzer):
        # Thursday 1am NY = Asia session (morning side, after midnight)
        dt = datetime(2026, 3, 12, 1, 0, 0, tzinfo=NY)
        info = analyzer.get_current_session(dt)
        assert info.session_name == "asia"

    def test_uk_session(self, analyzer: MMSessionAnalyzer):
        # Wednesday 5am NY = UK session
        dt = datetime(2026, 3, 11, 5, 0, 0, tzinfo=NY)
        info = analyzer.get_current_session(dt)
        assert info.session_name == "uk"
        assert info.is_gap is False

    def test_dead_zone(self, analyzer: MMSessionAnalyzer):
        # Monday 6pm NY = Dead Zone
        dt = datetime(2026, 3, 9, 18, 0, 0, tzinfo=NY)
        info = analyzer.get_current_session(dt)
        assert info.session_name == "dead_zone"
        assert info.is_gap is False

    def test_asia_gap(self, analyzer: MMSessionAnalyzer):
        # Monday 8:15pm NY = Asia gap
        dt = datetime(2026, 3, 9, 20, 15, 0, tzinfo=NY)
        info = analyzer.get_current_session(dt)
        assert info.session_name == "asia_gap"
        assert info.is_gap is True

    def test_uk_gap(self, analyzer: MMSessionAnalyzer):
        # Wednesday 3:10am NY = UK gap
        dt = datetime(2026, 3, 11, 3, 10, 0, tzinfo=NY)
        info = analyzer.get_current_session(dt)
        assert info.session_name == "uk_gap"
        assert info.is_gap is True

    def test_us_gap(self, analyzer: MMSessionAnalyzer):
        # Thursday 9:15am NY = US gap
        dt = datetime(2026, 3, 12, 9, 15, 0, tzinfo=NY)
        info = analyzer.get_current_session(dt)
        assert info.session_name == "us_gap"
        assert info.is_gap is True

    def test_minutes_remaining_positive(self, analyzer: MMSessionAnalyzer):
        # US session starts 9:30, ends 17:00 -> at 16:00 should have 60 min remaining
        dt = datetime(2026, 3, 10, 16, 0, 0, tzinfo=NY)
        info = analyzer.get_current_session(dt)
        assert info.session_name == "us"
        assert info.minutes_remaining == pytest.approx(60.0, abs=0.5)

    def test_session_info_has_correct_types(self, analyzer: MMSessionAnalyzer):
        dt = datetime(2026, 3, 10, 10, 0, 0, tzinfo=NY)
        info = analyzer.get_current_session(dt)
        assert isinstance(info, SessionInfo)
        assert isinstance(info.session_start, datetime)
        assert isinstance(info.session_end, datetime)
        assert isinstance(info.minutes_remaining, float)

    def test_utc_input(self, analyzer: MMSessionAnalyzer):
        # 2pm UTC on a Tuesday in January -> 9am EST -> us_gap
        dt = datetime(2026, 1, 13, 14, 0, 0, tzinfo=UTC)
        info = analyzer.get_current_session(dt)
        assert info.session_name == "us_gap"


# ------------------------------------------------------------------
# is_session_changeover
# ------------------------------------------------------------------

class TestIsSessionChangeover:
    def test_during_gap(self, analyzer: MMSessionAnalyzer):
        dt = datetime(2026, 3, 11, 3, 10, 0, tzinfo=NY)
        assert analyzer.is_session_changeover(dt) is True

    def test_during_session(self, analyzer: MMSessionAnalyzer):
        dt = datetime(2026, 3, 11, 10, 0, 0, tzinfo=NY)
        assert analyzer.is_session_changeover(dt) is False


# ------------------------------------------------------------------
# is_weekend
# ------------------------------------------------------------------

class TestIsWeekend:
    def test_friday_after_5pm(self, analyzer: MMSessionAnalyzer):
        dt = datetime(2026, 3, 13, 18, 0, 0, tzinfo=NY)  # Friday 6pm
        assert analyzer.is_weekend(dt) is True

    def test_saturday(self, analyzer: MMSessionAnalyzer):
        dt = datetime(2026, 3, 14, 12, 0, 0, tzinfo=NY)
        assert analyzer.is_weekend(dt) is True

    def test_sunday_before_5pm(self, analyzer: MMSessionAnalyzer):
        dt = datetime(2026, 3, 15, 10, 0, 0, tzinfo=NY)
        assert analyzer.is_weekend(dt) is True

    def test_sunday_at_5pm(self, analyzer: MMSessionAnalyzer):
        dt = datetime(2026, 3, 15, 17, 0, 0, tzinfo=NY)
        assert analyzer.is_weekend(dt) is False

    def test_weekday(self, analyzer: MMSessionAnalyzer):
        dt = datetime(2026, 3, 10, 12, 0, 0, tzinfo=NY)
        assert analyzer.is_weekend(dt) is False


# ------------------------------------------------------------------
# is_dead_zone
# ------------------------------------------------------------------

class TestIsDeadZone:
    def test_in_dead_zone(self, analyzer: MMSessionAnalyzer):
        dt = datetime(2026, 3, 10, 18, 0, 0, tzinfo=NY)
        assert analyzer.is_dead_zone(dt) is True

    def test_outside_dead_zone(self, analyzer: MMSessionAnalyzer):
        dt = datetime(2026, 3, 10, 10, 0, 0, tzinfo=NY)
        assert analyzer.is_dead_zone(dt) is False


# ------------------------------------------------------------------
# get_week_boundaries
# ------------------------------------------------------------------

class TestGetWeekBoundaries:
    def test_midweek(self, analyzer: MMSessionAnalyzer):
        # Wednesday March 11 2026 noon NY
        dt = datetime(2026, 3, 11, 12, 0, 0, tzinfo=NY)
        start, end = analyzer.get_week_boundaries(dt)
        assert start.weekday() == 6  # Sunday
        assert start.hour == 17
        assert end.weekday() == 4  # Friday
        assert end.hour == 17
        assert start < dt < end

    def test_weekend_returns_next_week(self, analyzer: MMSessionAnalyzer):
        # Saturday March 14 2026
        dt = datetime(2026, 3, 14, 12, 0, 0, tzinfo=NY)
        start, end = analyzer.get_week_boundaries(dt)
        # Should point to next week
        assert start > dt


# ------------------------------------------------------------------
# get_day_boundaries
# ------------------------------------------------------------------

class TestGetDayBoundaries:
    def test_before_5pm(self, analyzer: MMSessionAnalyzer):
        dt = datetime(2026, 3, 10, 10, 0, 0, tzinfo=NY)
        day_start, day_end = analyzer.get_day_boundaries(dt)
        # Before 5pm -> day started at 5pm yesterday
        assert day_start.day == 9
        assert day_start.hour == 17
        assert day_end.day == 10
        assert day_end.hour == 17

    def test_after_5pm(self, analyzer: MMSessionAnalyzer):
        dt = datetime(2026, 3, 10, 18, 0, 0, tzinfo=NY)
        day_start, day_end = analyzer.get_day_boundaries(dt)
        assert day_start.day == 10
        assert day_start.hour == 17
        assert day_end.day == 11
        assert day_end.hour == 17


# ------------------------------------------------------------------
# get_session_for_candle (alias)
# ------------------------------------------------------------------

class TestGetSessionForCandle:
    def test_returns_session_info(self, analyzer: MMSessionAnalyzer):
        dt = datetime(2026, 3, 11, 5, 0, 0, tzinfo=NY)
        info = analyzer.get_session_for_candle(dt)
        assert info.session_name == "uk"


# ------------------------------------------------------------------
# Full 24-hour coverage
# ------------------------------------------------------------------

class TestFullDayCoverage:
    """Every hour of the day should resolve to a session without errors."""

    def test_all_hours_resolve(self, analyzer: MMSessionAnalyzer):
        base = datetime(2026, 3, 10, 0, 0, 0, tzinfo=NY)  # Monday
        for h in range(24):
            dt = base + timedelta(hours=h)
            info = analyzer.get_current_session(dt)
            assert info.session_name in {
                "dead_zone", "asia_gap", "asia", "uk_gap", "uk", "us_gap", "us",
            }


# ------------------------------------------------------------------
# D2: AsiaSpike detection
# ------------------------------------------------------------------

def _make_candles_with_spike(
    spike_open: float,
    spike_close: float,
    spike_ny_hour: int = 2,
    spike_ny_minute: int = 0,
    n_candles: int = 10,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with one spike candle at the given NY time.

    The spike candle is placed at 2:00am NY (UTC = 7:00am for EDT, UTC-4).
    """
    # 2026-03-10 is after DST spring-forward (EDT = UTC-4)
    # 2:00am NY EDT = 6:00am UTC
    spike_utc_hour = spike_ny_hour + 4  # EDT offset
    spike_utc = datetime(2026, 3, 10, spike_utc_hour, spike_ny_minute, 0, tzinfo=UTC)

    # Create n_candles with the spike at position n-2 (second-to-last)
    base_utc = spike_utc - timedelta(hours=n_candles - 2)
    times = [base_utc + timedelta(hours=i) for i in range(n_candles)]
    idx = pd.DatetimeIndex(times)

    opens = np.full(n_candles, 100.0)
    closes = np.full(n_candles, 100.0)
    highs = np.full(n_candles, 101.0)
    lows = np.full(n_candles, 99.0)
    volumes = np.full(n_candles, 1000.0)

    # Insert the spike candle at the 2am NY position (index n-2)
    spike_idx = n_candles - 2
    opens[spike_idx] = spike_open
    closes[spike_idx] = spike_close
    highs[spike_idx] = max(spike_open, spike_close) + 0.1
    lows[spike_idx] = min(spike_open, spike_close) - 0.1

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


class TestAsiaClosingSpike:
    """Tests for MMSessionAnalyzer.detect_asia_closing_spike (D2)."""

    def test_spike_down_gives_bullish_bias(self, analyzer: MMSessionAnalyzer):
        """Asia spike down at 2am → bullish bias (expect W/higher low in London)."""
        # 100 open, 96.5 close = -3.5% move → well above 0.3% threshold
        candles = _make_candles_with_spike(spike_open=100.0, spike_close=96.5)
        now = datetime(2026, 3, 10, 10, 0, 0, tzinfo=UTC)
        result = analyzer.detect_asia_closing_spike(candles, now)
        assert result.detected is True
        assert result.direction == "down"
        assert result.bias == "bullish"
        assert result.magnitude_pct > 0.3

    def test_spike_up_gives_bearish_bias(self, analyzer: MMSessionAnalyzer):
        """Asia spike up at 2am → bearish bias (expect M/lower high in London)."""
        # 100 open, 103.5 close = +3.5% move
        candles = _make_candles_with_spike(spike_open=100.0, spike_close=103.5)
        now = datetime(2026, 3, 10, 10, 0, 0, tzinfo=UTC)
        result = analyzer.detect_asia_closing_spike(candles, now)
        assert result.detected is True
        assert result.direction == "up"
        assert result.bias == "bearish"
        assert result.magnitude_pct > 0.3

    def test_no_spike_small_move(self, analyzer: MMSessionAnalyzer):
        """Candle at 2am with move < 0.3% → no detection."""
        # 100 open, 100.1 close = 0.1% — below threshold
        candles = _make_candles_with_spike(spike_open=100.0, spike_close=100.1)
        now = datetime(2026, 3, 10, 10, 0, 0, tzinfo=UTC)
        result = analyzer.detect_asia_closing_spike(candles, now)
        assert result.detected is False
        assert result.direction == "none"
        assert result.bias == "none"

    def test_no_spike_candle_not_in_window(self, analyzer: MMSessionAnalyzer):
        """Candle outside 2:00-2:30am window → no detection."""
        # Build candles where only the 4am candle is large — not the 2am window
        spike_utc = datetime(2026, 3, 10, 8, 0, 0, tzinfo=UTC)  # 4am NY EDT
        n = 10
        base_utc = spike_utc - timedelta(hours=n - 2)
        times = [base_utc + timedelta(hours=i) for i in range(n)]
        idx = pd.DatetimeIndex(times)
        opens = np.full(n, 100.0)
        closes = np.full(n, 100.0)
        # Make the 4am candle large (index n-2) — but that maps to 4am NY not 2am
        opens[-2] = 100.0
        closes[-2] = 95.0  # big spike but wrong window
        highs = np.maximum(opens, closes) + 0.1
        lows = np.minimum(opens, closes) - 0.1
        volumes = np.full(n, 1000.0)
        candles = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=idx,
        )
        now = datetime(2026, 3, 10, 10, 0, 0, tzinfo=UTC)
        result = analyzer.detect_asia_closing_spike(candles, now)
        assert result.detected is False

    def test_insufficient_data_returns_no_spike(self, analyzer: MMSessionAnalyzer):
        """Empty DataFrame → no detection, no error."""
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        now = datetime(2026, 3, 10, 10, 0, 0, tzinfo=UTC)
        result = analyzer.detect_asia_closing_spike(empty, now)
        assert result.detected is False
        assert result.bias == "none"

    def test_small_dataframe_returns_no_spike(self, analyzer: MMSessionAnalyzer):
        """DataFrame with fewer than 5 rows → no detection."""
        idx = pd.DatetimeIndex([datetime(2026, 3, 10, 6, 0, 0, tzinfo=UTC)])
        candles = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [97.0], "volume": [1000.0]},
            index=idx,
        )
        now = datetime(2026, 3, 10, 10, 0, 0, tzinfo=UTC)
        result = analyzer.detect_asia_closing_spike(candles, now)
        assert result.detected is False
