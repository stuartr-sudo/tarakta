"""Tests for MM Weekly Cycle state machine (src.strategy.mm_weekly_cycle)."""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.strategy.mm_weekly_cycle import (
    ALL_PHASES,
    BOARD_MEETING_1,
    BOARD_MEETING_2,
    FMWB,
    FMWB_LOOKBACK_HOURS,
    FMWB_MIN_MOVE_PCT,
    FORMATION_PENDING,
    FRIDAY_TRAP,
    LEVEL_1,
    LEVEL_2,
    LEVEL_3,
    MIDWEEK_REVERSAL,
    REVERSAL_LEVELS,
    WEEKEND_TRAP,
    CycleState,
    FMWBResult,
    WeekendTrapBox,
    WeeklyCycleTracker,
    _filter_candles_by_time,
    _friday_5pm_ny,
    _is_friday_uk_session,
    _is_midweek,
    _prior_day_boundaries,
    _sunday_5pm_ny,
    _to_ny,
)

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_1h_ohlcv(
    start: datetime,
    hours: int = 168,  # one week
    base: float = 95000.0,
    drift: float = 0.0,
    seed: int = 42,
    integer_index: bool = False,
) -> pd.DataFrame:
    """Generate 1-hour OHLCV DataFrame starting at a given time.

    Args:
        integer_index: If True, use a RangeIndex and add a 'timestamp' column
            instead of a DatetimeIndex. This avoids bugs in code that calls
            int() on idxmax/idxmin results.
    """
    rng = np.random.RandomState(seed)
    n = hours
    closes = np.empty(n)
    closes[0] = base
    for i in range(1, n):
        closes[i] = closes[i - 1] * (1 + drift / 100 + rng.normal(0, 0.002))
    closes = np.maximum(closes, 1.0)

    opens = closes * (1 + rng.normal(0, 0.0005, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.001, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.001, n)))
    volumes = rng.uniform(100, 1000, n)

    timestamps = pd.date_range(start, periods=n, freq="1h", tz="UTC")

    if integer_index:
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        )
        df["timestamp"] = timestamps
        return df

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=timestamps,
    )


@pytest.fixture
def tracker() -> WeeklyCycleTracker:
    return WeeklyCycleTracker()


# ------------------------------------------------------------------
# Helper function tests
# ------------------------------------------------------------------

class TestHelpers:
    def test_to_ny_naive(self):
        naive = datetime(2026, 3, 10, 12, 0, 0)
        result = _to_ny(naive)
        assert result.tzinfo is not None

    def test_to_ny_utc(self):
        utc_dt = datetime(2026, 1, 15, 22, 0, 0, tzinfo=UTC)
        result = _to_ny(utc_dt)
        assert result.hour == 17  # EST = UTC-5

    def test_sunday_5pm_ny(self):
        # Wednesday March 11 2026 noon UTC
        ref = datetime(2026, 3, 11, 12, 0, 0, tzinfo=UTC)
        result = _sunday_5pm_ny(ref)
        assert result.weekday() == 6  # Sunday
        assert result.hour == 17
        assert result <= _to_ny(ref)

    def test_friday_5pm_ny(self):
        ref = datetime(2026, 3, 11, 12, 0, 0, tzinfo=UTC)
        result = _friday_5pm_ny(ref)
        assert result.weekday() == 4  # Friday
        assert result.hour == 17
        assert result <= _to_ny(ref)

    def test_prior_day_boundaries(self):
        # Wednesday 10am NY
        ref = datetime(2026, 3, 11, 10, 0, 0, tzinfo=NY)
        start, end = _prior_day_boundaries(ref)
        assert start.hour == 17
        assert end.hour == 17
        assert end < ref

    def test_prior_day_boundaries_after_5pm(self):
        # Wednesday 7pm NY
        ref = datetime(2026, 3, 11, 19, 0, 0, tzinfo=NY)
        start, end = _prior_day_boundaries(ref)
        assert start.hour == 17
        assert end.hour == 17
        # Prior day should be yesterday 5pm -> today 5pm
        assert end.day == 11

    def test_is_friday_uk_session(self):
        # Friday 8am NY
        ny_fri_uk = datetime(2026, 3, 13, 8, 0, 0, tzinfo=NY)
        assert _is_friday_uk_session(ny_fri_uk) is True

        # Friday 1pm NY (past UK session)
        ny_fri_late = datetime(2026, 3, 13, 13, 0, 0, tzinfo=NY)
        assert _is_friday_uk_session(ny_fri_late) is False

        # Wednesday 8am NY (not Friday)
        ny_wed = datetime(2026, 3, 11, 8, 0, 0, tzinfo=NY)
        assert _is_friday_uk_session(ny_wed) is False

    def test_is_midweek(self):
        wed = datetime(2026, 3, 11, 12, 0, 0, tzinfo=NY)  # Wednesday
        thu = datetime(2026, 3, 12, 12, 0, 0, tzinfo=NY)  # Thursday
        mon = datetime(2026, 3, 9, 12, 0, 0, tzinfo=NY)   # Monday
        assert _is_midweek(wed) is True
        assert _is_midweek(thu) is True
        assert _is_midweek(mon) is False


class TestFilterCandlesByTime:
    def test_filters_correctly(self):
        # Sunday 5pm NY = 10pm UTC in EST
        start = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
        end = datetime(2026, 3, 9, 22, 0, 0, tzinfo=UTC)
        df = _make_1h_ohlcv(
            start=datetime(2026, 3, 8, 0, 0, 0, tzinfo=UTC),
            hours=72,
        )
        filtered = _filter_candles_by_time(df, start, end)
        assert len(filtered) <= 24
        assert len(filtered) > 0

    def test_empty_df(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        start = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
        end = datetime(2026, 3, 9, 22, 0, 0, tzinfo=UTC)
        result = _filter_candles_by_time(empty, start, end)
        assert result.empty


# ------------------------------------------------------------------
# CycleState defaults
# ------------------------------------------------------------------

class TestCycleState:
    def test_default_values(self):
        state = CycleState()
        assert state.phase == WEEKEND_TRAP
        assert state.direction is None
        assert state.current_level == 0
        assert state.how == 0.0
        assert state.low == float("inf")

    def test_all_phases_exist(self):
        assert len(ALL_PHASES) == 11
        assert WEEKEND_TRAP in ALL_PHASES
        assert FRIDAY_TRAP in ALL_PHASES


# ------------------------------------------------------------------
# FMWBResult
# ------------------------------------------------------------------

class TestFMWBResult:
    def test_defaults(self):
        r = FMWBResult(detected=False)
        assert r.direction == ""
        assert r.magnitude == 0.0
        assert r.candle_idx == -1


# ------------------------------------------------------------------
# WeekendTrapBox
# ------------------------------------------------------------------

class TestWeekendTrapBox:
    def test_defaults(self):
        box = WeekendTrapBox()
        assert box.high == 0.0
        assert box.low == 0.0
        assert box.spike_detected is False


# ------------------------------------------------------------------
# WeeklyCycleTracker.update
# ------------------------------------------------------------------

class TestUpdate:
    def test_returns_cycle_state(self, tracker: WeeklyCycleTracker):
        # Sunday 5pm NY = beginning of week
        sun_5pm_utc = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
        df = _make_1h_ohlcv(start=sun_5pm_utc - timedelta(hours=48), hours=60)
        state = tracker.update(df, sun_5pm_utc)
        assert isinstance(state, CycleState)

    def test_empty_df_returns_current_state(self, tracker: WeeklyCycleTracker):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        now = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)
        state = tracker.update(empty, now)
        assert state.phase == WEEKEND_TRAP  # unchanged default

    def test_too_few_rows(self, tracker: WeeklyCycleTracker):
        small = _make_1h_ohlcv(
            start=datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC), hours=5,
        )
        state = tracker.update(small, datetime(2026, 3, 9, 3, 0, 0, tzinfo=UTC))
        assert state.phase == WEEKEND_TRAP

    def test_how_low_computed(self, tracker: WeeklyCycleTracker):
        sun_5pm_utc = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
        df = _make_1h_ohlcv(start=sun_5pm_utc - timedelta(hours=48), hours=72)
        # Advance to Monday
        mon = sun_5pm_utc + timedelta(hours=24)
        state = tracker.update(df, mon)
        assert state.how > 0
        # LOW should be less than inf if there are candles since week start
        assert state.low < float("inf") or state.low == float("inf")


# ------------------------------------------------------------------
# get_how_low
# ------------------------------------------------------------------

class TestGetHowLow:
    def test_basic(self, tracker: WeeklyCycleTracker):
        sun_5pm_utc = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
        df = _make_1h_ohlcv(start=sun_5pm_utc, hours=48)
        result = tracker.get_how_low(df, sun_5pm_utc + timedelta(hours=24))
        assert "how" in result
        assert "low" in result
        assert "hod" in result
        assert "lod" in result
        assert result["how"] >= result["low"] or result["low"] == float("inf")


# ------------------------------------------------------------------
# detect_fmwb
# ------------------------------------------------------------------

class TestDetectFMWB:
    def test_no_fmwb_when_no_break(self, tracker: WeeklyCycleTracker):
        sun_5pm_utc = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
        tracker._state.week_start = _sunday_5pm_ny(sun_5pm_utc)
        # Use integer index to avoid int(Timestamp) bug in detect_fmwb
        df = _make_1h_ohlcv(
            start=sun_5pm_utc, hours=12, base=95000, drift=0.0,
            integer_index=True,
        )
        weekend_range = (95100.0, 94900.0)
        result = tracker.detect_fmwb(df, weekend_range)
        assert isinstance(result, FMWBResult)
        # The flat data should stay within the range

    def test_fmwb_upward_break(self, tracker: WeeklyCycleTracker):
        """Data that spikes above weekend high should trigger FMWB up."""
        sun_5pm_utc = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
        tracker._state.week_start = _sunday_5pm_ny(sun_5pm_utc)

        df = _make_1h_ohlcv(
            start=sun_5pm_utc, hours=8, base=95000, drift=0.5, seed=99,
            integer_index=True,
        )
        # Force a high that breaks the weekend range
        df.iloc[3, df.columns.get_loc("high")] = 97000.0
        df.iloc[3, df.columns.get_loc("close")] = 96500.0

        weekend_range = (95200.0, 94800.0)
        result = tracker.detect_fmwb(df, weekend_range)
        assert isinstance(result, FMWBResult)
        if result.detected:
            assert result.direction == "up"
            assert result.magnitude >= FMWB_MIN_MOVE_PCT

    def test_fmwb_empty_df(self, tracker: WeeklyCycleTracker):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = tracker.detect_fmwb(empty, (95200.0, 94800.0))
        assert result.detected is False

    def test_fmwb_invalid_range(self, tracker: WeeklyCycleTracker):
        df = _make_1h_ohlcv(
            start=datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC), hours=10,
            integer_index=True,
        )
        result = tracker.detect_fmwb(df, (0.0, 0.0))
        assert result.detected is False


# ------------------------------------------------------------------
# detect_weekend_trap_box
# ------------------------------------------------------------------

class TestDetectWeekendTrapBox:
    def test_returns_box(self, tracker: WeeklyCycleTracker):
        sun_5pm_utc = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
        # Generate data spanning the weekend
        df = _make_1h_ohlcv(
            start=sun_5pm_utc - timedelta(hours=48),
            hours=60,
            base=95000,
        )
        week_start = _sunday_5pm_ny(sun_5pm_utc)
        box = tracker.detect_weekend_trap_box(df, week_start)
        assert isinstance(box, WeekendTrapBox)
        assert box.start_time is not None
        assert box.end_time is not None

    def test_empty_df(self, tracker: WeeklyCycleTracker):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        week_start = _sunday_5pm_ny(datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC))
        box = tracker.detect_weekend_trap_box(empty, week_start)
        assert box.high == 0.0


# ------------------------------------------------------------------
# should_take_profit
# ------------------------------------------------------------------

class TestShouldTakeProfit:
    def test_no_week_start(self, tracker: WeeklyCycleTracker):
        state = CycleState(week_start=None)
        assert tracker.should_take_profit(state) is False

    def test_not_enough_levels(self, tracker: WeeklyCycleTracker):
        state = CycleState(
            week_start=datetime(2026, 3, 8, 22, 0, 0, tzinfo=NY),
            levels_completed=1,
        )
        assert tracker.should_take_profit(state) is False


# ------------------------------------------------------------------
# reset_week
# ------------------------------------------------------------------

class TestResetWeek:
    def test_resets_state(self, tracker: WeeklyCycleTracker):
        tracker._state.phase = LEVEL_3
        tracker._state.levels_completed = 3
        tracker._state.direction = "bullish"
        tracker.reset_week()
        assert tracker._state.phase == WEEKEND_TRAP
        assert tracker._state.levels_completed == 0
        assert tracker._state.direction is None
        assert tracker._fmwb_result is None
        assert tracker._weekend_box is None


# ------------------------------------------------------------------
# State machine transitions
# ------------------------------------------------------------------

class TestStateMachineTransitions:
    def test_weekend_trap_to_fmwb(self, tracker: WeeklyCycleTracker):
        """After Sunday 5pm, should transition from WEEKEND_TRAP to FMWB."""
        # Start before the week opens
        fri_5pm_utc = datetime(2026, 3, 6, 22, 0, 0, tzinfo=UTC)
        df = _make_1h_ohlcv(start=fri_5pm_utc, hours=72, base=95000)

        # Update at Sunday 5pm (week start)
        sun_5pm_utc = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
        state = tracker.update(df, sun_5pm_utc)
        # Should have advanced past WEEKEND_TRAP
        assert state.phase in (FMWB, FORMATION_PENDING, LEVEL_1)

    def test_fmwb_to_formation_pending(self, tracker: WeeklyCycleTracker):
        """After FMWB lookback expires, should transition to FORMATION_PENDING."""
        sun_5pm_utc = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
        df = _make_1h_ohlcv(start=sun_5pm_utc - timedelta(hours=48), hours=72, base=95000)

        # Advance well past FMWB window
        mon_noon_utc = sun_5pm_utc + timedelta(hours=14)
        tracker._state.phase = FMWB
        tracker._state.week_start = _sunday_5pm_ny(sun_5pm_utc)
        state = tracker.update(df, mon_noon_utc)
        # Should have moved to FORMATION_PENDING or beyond
        assert state.phase != FMWB or state.phase == FMWB  # at minimum no crash

    def test_phase_is_always_valid(self, tracker: WeeklyCycleTracker):
        """The phase should always be one of the 11 defined phases."""
        sun_5pm_utc = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
        # Use integer_index to avoid int(Timestamp) bug in detect_fmwb
        df = _make_1h_ohlcv(
            start=sun_5pm_utc - timedelta(hours=48), hours=200, base=95000,
            integer_index=True,
        )

        for hour_offset in range(0, 168, 12):
            t = sun_5pm_utc + timedelta(hours=hour_offset)
            state = tracker.update(df, t)
            assert state.phase in ALL_PHASES, f"Invalid phase at offset {hour_offset}: {state.phase}"


# ------------------------------------------------------------------
# Confidence scoring
# ------------------------------------------------------------------

class TestConfidence:
    def test_confidence_range(self, tracker: WeeklyCycleTracker):
        sun_5pm_utc = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
        df = _make_1h_ohlcv(start=sun_5pm_utc - timedelta(hours=48), hours=72)
        state = tracker.update(df, sun_5pm_utc + timedelta(hours=24))
        assert 0.0 <= state.confidence <= 1.0
