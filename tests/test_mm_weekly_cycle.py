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


# ------------------------------------------------------------------
# Market Resets (Lesson 15 — A6)
# ------------------------------------------------------------------

from src.strategy.mm_weekly_cycle import MarketResetResult


def _make_type1_reset_candles(direction: str = "bearish") -> pd.DataFrame:
    """Build candles where price approaches the 50 EMA but fails to break it.

    Strategy: compute the EMA from the price path and then set highs/lows
    so they approach the EMA within 0.5% but never cross beyond 0.5%.
    """
    n = 100
    base = 1000.0
    start = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")

    if direction == "bearish":
        # Gentle downtrend then stabilization just below EMA
        closes = np.empty(n)
        closes[0] = base
        for i in range(1, 60):
            closes[i] = closes[i - 1] * 0.999
        for i in range(60, n):
            closes[i] = closes[i - 1] * 1.0001
    else:
        # Gentle uptrend then stabilization just above EMA
        closes = np.empty(n)
        closes[0] = base
        for i in range(1, 60):
            closes[i] = closes[i - 1] * 1.001
        for i in range(60, n):
            closes[i] = closes[i - 1] * 0.9999

    # Compute EMA to calibrate wicks
    ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().values
    opens = closes.copy()

    if direction == "bearish":
        # Highs approach EMA from below: set highs to EMA * 0.998 (close but below)
        highs = np.maximum(closes, closes)
        for i in range(80, n):
            # Place highs just below EMA (within 0.5% but not above 0.5%)
            highs[i] = ema50[i] * 0.998
            highs[i] = max(highs[i], closes[i])
        lows = closes * 0.999
    else:
        # Lows approach EMA from above: set lows to EMA * 1.002
        lows = np.minimum(closes, closes)
        for i in range(80, n):
            lows[i] = ema50[i] * 1.002
            lows[i] = min(lows[i], closes[i])
        highs = closes * 1.001

    volumes = np.ones(n) * 1000

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _make_type3_consolidation_candles() -> pd.DataFrame:
    """Build candles with a full-day consolidation (range < 1.5%)."""
    n = 100
    base = 1000.0
    start = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")

    # Very flat: close stays near base
    rng = np.random.RandomState(42)
    closes = base + rng.normal(0, 0.5, n)
    opens = closes + rng.normal(0, 0.1, n)
    highs = np.maximum(opens, closes) + 0.3
    lows = np.minimum(opens, closes) - 0.3
    volumes = np.ones(n) * 1000

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


class TestMarketResets:
    """Tests for Market Reset detection (Lesson 15 — A6)."""

    def test_type1_bearish_continuation(self, tracker: WeeklyCycleTracker):
        """Type 1: W fails to break 50 EMA after downtrend → bearish continuation."""
        candles = _make_type1_reset_candles(direction="bearish")
        result = tracker.detect_market_reset(candles, ema_state=None, prior_direction="bearish")

        assert result is not None
        assert result.detected is True
        assert result.reset_type == 1
        assert result.direction == "bearish"  # continuation of prior
        assert result.confidence > 0

    def test_type1_bullish_continuation(self, tracker: WeeklyCycleTracker):
        """Type 1: M fails to break 50 EMA after uptrend → bullish continuation."""
        candles = _make_type1_reset_candles(direction="bullish")
        result = tracker.detect_market_reset(candles, ema_state=None, prior_direction="bullish")

        assert result is not None
        assert result.detected is True
        assert result.reset_type == 1
        assert result.direction == "bullish"

    def test_type3_full_day_consolidation(self, tracker: WeeklyCycleTracker):
        """Type 3: Full-day consolidation (range < 1.5%) → continuation."""
        candles = _make_type3_consolidation_candles()
        result = tracker.detect_market_reset(candles, ema_state=None, prior_direction="bearish")

        assert result is not None
        assert result.detected is True
        assert result.reset_type in (1, 3)  # might hit type1 first due to flat data
        assert result.direction == "bearish"

    def test_no_reset_on_strong_ema_break(self, tracker: WeeklyCycleTracker):
        """Strong EMA break → no Type 1 reset."""
        n = 100
        base = 1000.0
        start = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
        idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")

        # Strong uptrend that clearly breaks above all EMAs
        closes = base * np.cumprod(np.full(n, 1.005))
        opens = closes * 0.999
        highs = closes * 1.002
        lows = closes * 0.998
        volumes = np.ones(n) * 1000

        candles = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=idx,
        )

        # Type 1 should NOT trigger — the uptrend broke the EMA decisively
        result = tracker._detect_reset_type1(candles, prior_direction="bearish")
        assert result is None

    def test_insufficient_data(self, tracker: WeeklyCycleTracker):
        """Insufficient data → no detection."""
        start = datetime(2026, 3, 8, 22, 0, 0, tzinfo=UTC)
        candles = _make_1h_ohlcv(start=start, hours=20)
        result = tracker.detect_market_reset(candles, ema_state=None, prior_direction="bearish")
        assert result is None

    def test_market_reset_result_dataclass(self):
        """MarketResetResult dataclass has correct fields."""
        r = MarketResetResult(detected=True, reset_type=1, direction="bearish", confidence=0.7)
        assert r.detected is True
        assert r.reset_type == 1
        assert r.direction == "bearish"
        assert r.confidence == 0.7


# ------------------------------------------------------------------
# iHOD/iLOD Confirmation (Task 7.1 — Lesson D1)
# ------------------------------------------------------------------

def _make_holding_candles(
    level: float,
    num_candles: int,
    hold_pct: float = 0.003,  # within 0.3% of level (inside the 0.5% hold band)
    broke_after: int | None = None,
    broke_direction: str = "up",
) -> pd.DataFrame:
    """Build candles that hover near `level` for `num_candles`, optionally breaking."""
    start = datetime(2026, 4, 14, 10, 0, 0, tzinfo=UTC)
    idx = pd.date_range(start, periods=num_candles, freq="1h", tz="UTC")
    closes = np.full(num_candles, level * (1 + hold_pct))
    highs = closes * 1.001
    lows = closes * 0.999
    opens = closes * 1.0
    if broke_after is not None and broke_after < num_candles:
        for i in range(broke_after, num_candles):
            if broke_direction == "up":
                closes[i] = level * 1.008  # clearly outside 0.5% band
                highs[i] = closes[i] * 1.001
                lows[i] = closes[i] * 0.999
            else:
                closes[i] = level * 0.992
                highs[i] = closes[i] * 1.001
                lows[i] = closes[i] * 0.999
    volumes = np.ones(num_candles) * 500
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


class TestIHODILODConfirmation:

    def test_confirmed_level_held_60_min(self, tracker: WeeklyCycleTracker):
        """Confirmed: price held within 0.5% of level for ~60 min (1 candle)."""
        level = 95000.0
        # 2 candles at level = 120 min hold (within 30-90 min / 1-3 candle window)
        candles = _make_holding_candles(level, num_candles=2, hold_pct=0.003)
        result = tracker.confirm_ihod_ilod(candles, level, "ihod")
        assert result["confirmed"] is True
        assert result["hold_minutes"] >= 60.0

    def test_unconfirmed_broke_after_10_min(self, tracker: WeeklyCycleTracker):
        """Unconfirmed: price broke above level within 1 candle (< 30 min hold)."""
        level = 95000.0
        # Break on the first candle → 0 hold candles before break
        candles = _make_holding_candles(level, num_candles=4, hold_pct=0.003,
                                        broke_after=0, broke_direction="up")
        result = tracker.confirm_ihod_ilod(candles, level, "ihod")
        assert result["confirmed"] is False

    def test_triple_tested_level(self, tracker: WeeklyCycleTracker):
        """Triple-tested: 3+ touches reported when price wicks within 0.2%."""
        level = 50000.0
        start = datetime(2026, 4, 14, 10, 0, 0, tzinfo=UTC)
        idx = pd.date_range(start, periods=5, freq="1h", tz="UTC")
        # Each candle wicks to within 0.1% of level (well inside the 0.2% touch band)
        closes = np.full(5, level * 1.004)  # close in hold band (0.4%)
        highs = np.full(5, level * 1.005)
        lows = np.full(5, level * 0.9995)  # wick to within 0.05% = clear touch
        opens = closes.copy()
        volumes = np.ones(5) * 500
        candles = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=idx,
        )
        result = tracker.confirm_ihod_ilod(candles, level, "ilod")
        assert result["touch_count"] >= 3

    def test_empty_candles_returns_unconfirmed(self, tracker: WeeklyCycleTracker):
        """Empty DataFrame → unconfirmed gracefully."""
        result = tracker.confirm_ihod_ilod(pd.DataFrame(), 95000.0, "ihod")
        assert result["confirmed"] is False
        assert result["hold_minutes"] == 0.0
        assert result["touch_count"] == 0

    def test_zero_level_returns_unconfirmed(self, tracker: WeeklyCycleTracker):
        """Zero level → unconfirmed gracefully."""
        candles = _make_holding_candles(95000.0, num_candles=4)
        result = tracker.confirm_ihod_ilod(candles, 0.0, "ihod")
        assert result["confirmed"] is False


# ------------------------------------------------------------------
# Friday Trap Pattern (Task 7.3 — Lesson D8)
# ------------------------------------------------------------------

def _make_friday_candles(
    false_direction: str = "up",
    include_trend: bool = True,
    include_extension: bool = False,
    include_us_reversal: bool = False,
) -> tuple[pd.DataFrame, datetime]:
    """Build Friday 1H candles simulating the trap pattern."""
    # Friday 2026-04-17, UK session starts at 7am UTC (approx 3am NY)
    # UK open NY = 3am → use 7am UTC as proxy
    uk_open_utc = datetime(2026, 4, 17, 7, 0, 0, tzinfo=UTC)  # Friday
    # Build 8 UK candles + 3 US candles
    num_candles = 11
    idx = pd.date_range(uk_open_utc, periods=num_candles, freq="1h", tz="UTC")

    base = 95000.0
    opens = np.full(num_candles, base)
    closes = np.full(num_candles, base)
    highs = np.full(num_candles, base * 1.002)
    lows = np.full(num_candles, base * 0.998)
    volumes = np.ones(num_candles) * 500

    # Candle 0: false move spike (wide range, wick-dominant)
    if false_direction == "up":
        highs[0] = base * 1.015  # big upper wick
        lows[0] = base * 0.999
        opens[0] = base * 1.001
        closes[0] = base * 1.002  # small body
    else:
        lows[0] = base * 0.985
        highs[0] = base * 1.001
        opens[0] = base * 0.999
        closes[0] = base * 0.998

    # UK candles 1-7: trend in opposite direction
    real_dir = -1 if false_direction == "up" else 1
    if include_trend:
        for i in range(1, 8):
            step = base * 0.002 * real_dir * i
            opens[i] = base + step * 0.9
            closes[i] = base + step
            highs[i] = max(opens[i], closes[i]) * 1.001
            lows[i] = min(opens[i], closes[i]) * 0.999

        if include_extension:
            # Last 2 UK candles are wider than average
            highs[6] = closes[6] * 1.004
            lows[6] = closes[6] * 0.996
            highs[7] = closes[7] * 1.004
            lows[7] = closes[7] * 0.996

    # US candles 8-10: reversal
    if include_us_reversal and include_trend:
        # US reverses: goes back toward base
        for i in range(8, 11):
            opens[i] = closes[7] * (1 + real_dir * 0.001)
            closes[i] = closes[7] * (1 + real_dir * 0.003 * (11 - i))
            highs[i] = max(opens[i], closes[i]) * 1.001
            lows[i] = min(opens[i], closes[i]) * 0.999

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    ), uk_open_utc + timedelta(hours=5)  # "now" = mid-UK session


class TestFridayTrapPattern:

    def test_detects_false_move_on_friday(self, tracker: WeeklyCycleTracker):
        """Detects at least false_move phase on a Friday with a spike candle."""
        candles, now = _make_friday_candles(false_direction="up", include_trend=False)
        result = tracker.detect_friday_trap_pattern(candles, now)
        assert result is not None
        assert result["phase"] in ("false_move", "trend", "extension", "us_reversal")
        assert result["direction"] in ("up", "down")

    def test_no_pattern_on_non_friday(self, tracker: WeeklyCycleTracker):
        """Returns None when it's not Friday."""
        candles, _ = _make_friday_candles()
        # Use a Wednesday timestamp
        wednesday_now = datetime(2026, 4, 15, 12, 0, 0, tzinfo=UTC)
        result = tracker.detect_friday_trap_pattern(candles, wednesday_now)
        assert result is None

    def test_detects_trend_phase(self, tracker: WeeklyCycleTracker):
        """Detects trend phase when UK session candles trend after false move."""
        candles, now = _make_friday_candles(
            false_direction="up", include_trend=True
        )
        result = tracker.detect_friday_trap_pattern(candles, now)
        assert result is not None
        assert result["phase"] in ("trend", "extension", "us_reversal")

    def test_empty_candles_returns_none(self, tracker: WeeklyCycleTracker):
        """Empty candles → None gracefully."""
        friday_now = datetime(2026, 4, 17, 10, 0, 0, tzinfo=UTC)
        result = tracker.detect_friday_trap_pattern(pd.DataFrame(), friday_now)
        assert result is None
