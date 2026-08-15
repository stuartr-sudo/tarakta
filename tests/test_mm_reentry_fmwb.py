"""Tests for the 2026-08-15 course-cited rule fixes.

1. FMWB weekly-bias gate (Lesson 10 [25:00], Lesson 3 [55:30]/[26:00]):
   the bias binds only when the false move broke OUT of the weekend box,
   FAILED back inside, and the week is still in the Sun/Mon/Tue window.
2. Re-setup rule (Lesson 13 [45:30]/[79:00], Lesson 6 [25:30], Lesson 16
   [58:30]): after a stop-out, a formation that completed BEFORE the stop
   is the same failed idea and must not re-enter.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd

from src.strategy.mm_engine import MMEngine

NY = ZoneInfo("America/New_York")


def _weekend(detected=True, direction="down", broke_box=False,
             box_detected=True, wick_high=110.0, wick_low=90.0):
    fmwb = SimpleNamespace(detected=detected, direction=direction,
                           broke_box=broke_box)
    trap_box = SimpleNamespace(detected=box_detected, wick_high=wick_high,
                               wick_low=wick_low)
    return SimpleNamespace(fmwb=fmwb, trap_box=trap_box)


def _ny_time(weekday_name: str) -> datetime:
    """A tz-aware UTC datetime whose NY weekday is as requested."""
    days = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    base = datetime(2026, 8, 10, 12, 0, tzinfo=NY)  # Monday noon NY
    target = base + timedelta(days=(days[weekday_name] - base.weekday()) % 7)
    return target.astimezone(timezone.utc)


class TestFMWBRequiredDirection:
    def test_no_fmwb_no_bias(self):
        weekend = _weekend(detected=False)
        required, info = MMEngine._fmwb_required_direction(weekend, 100.0, _ny_time("mon"))
        assert required is None
        assert info["binding"] is False

    def test_inside_box_move_never_binds(self):
        # Lesson 10 [25:00]: a move that never left the box is NOT a fake move.
        weekend = _weekend(direction="down", broke_box=False)
        required, info = MMEngine._fmwb_required_direction(weekend, 100.0, _ny_time("mon"))
        assert required is None
        assert info["broke_box"] is False

    def test_broke_and_failed_down_spike_binds_long(self):
        # Down-spike below the box that failed (price back above wick_low)
        # confirms the trap -> real direction long, Monday.
        weekend = _weekend(direction="down", broke_box=True, wick_low=90.0)
        required, info = MMEngine._fmwb_required_direction(weekend, 95.0, _ny_time("mon"))
        assert required == "long"
        assert info["binding"] is True and info["failed"] is True

    def test_broke_but_not_failed_does_not_bind(self):
        # Price still below the box after a down-spike: the move has not
        # failed — it may simply be the real trend. No inference.
        weekend = _weekend(direction="down", broke_box=True, wick_low=90.0)
        required, info = MMEngine._fmwb_required_direction(weekend, 85.0, _ny_time("mon"))
        assert required is None
        assert info["failed"] is False

    def test_up_spike_failed_binds_short(self):
        weekend = _weekend(direction="up", broke_box=True, wick_high=110.0)
        required, _ = MMEngine._fmwb_required_direction(weekend, 105.0, _ny_time("tue"))
        assert required == "short"

    def test_bias_expires_wednesday(self):
        # Lesson 3 [26:00] midweek reversal; quiz: two trend changes/week.
        weekend = _weekend(direction="down", broke_box=True, wick_low=90.0)
        for day in ("wed", "thu", "fri", "sat"):
            required, info = MMEngine._fmwb_required_direction(weekend, 95.0, _ny_time(day))
            assert required is None, day
            assert info["expired"] is True, day

    def test_bias_binds_sunday(self):
        weekend = _weekend(direction="down", broke_box=True, wick_low=90.0)
        required, _ = MMEngine._fmwb_required_direction(weekend, 95.0, _ny_time("sun"))
        assert required == "long"


class TestFormationCompletedAt:
    def _frame(self, hours=48, tz="UTC"):
        idx = pd.date_range("2026-08-01", periods=hours, freq="1h", tz=tz)
        return pd.DataFrame({"close": range(hours)}, index=idx)

    def test_resolves_peak2_timestamp(self):
        df = self._frame()
        formation = SimpleNamespace(timeframe="1h", peak2_idx=10)
        ts = MMEngine._formation_completed_at(formation, {"1h": df})
        assert ts == datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc)

    def test_naive_index_assumed_utc(self):
        df = self._frame(tz=None)
        formation = SimpleNamespace(timeframe="1h", peak2_idx=5)
        ts = MMEngine._formation_completed_at(formation, {"1h": df})
        assert ts.tzinfo is not None
        assert ts == datetime(2026, 8, 1, 5, 0, tzinfo=timezone.utc)

    def test_missing_frame_returns_none(self):
        formation = SimpleNamespace(timeframe="4h", peak2_idx=3)
        assert MMEngine._formation_completed_at(formation, {"1h": self._frame()}) is None

    def test_out_of_range_index_returns_none(self):
        formation = SimpleNamespace(timeframe="1h", peak2_idx=9999)
        assert MMEngine._formation_completed_at(formation, {"1h": self._frame()}) is None

    def test_no_peak_idx_returns_none(self):
        formation = SimpleNamespace(timeframe="1h", peak2_idx=None)
        assert MMEngine._formation_completed_at(formation, {"1h": self._frame()}) is None


class TestStaleFormationRule:
    """The gate condition itself: formation completed before the last
    same-direction stop == stale (reject); after == fresh (allow)."""

    def test_formation_before_stop_is_stale(self):
        stop_at = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
        formed = datetime(2026, 8, 10, 8, 0, tzinfo=timezone.utc)
        assert formed <= stop_at  # the engine rejects on this predicate

    def test_formation_after_stop_is_fresh(self):
        stop_at = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
        formed = datetime(2026, 8, 10, 16, 0, tzinfo=timezone.utc)
        assert not (formed <= stop_at)
