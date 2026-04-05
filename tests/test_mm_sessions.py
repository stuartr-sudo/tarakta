"""Tests for MM session timing (src.strategy.mm_sessions)."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.strategy.mm_sessions import (
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
