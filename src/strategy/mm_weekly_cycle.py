"""Market Maker Weekly Cycle State Machine.

Tracks where we are in the Market Maker's weekly business cycle.
This is the core orchestrator that ties sessions, formations, and levels together.

The Weekly Cycle (11 phases):
  1. WEEKEND_TRAP    — Fri 5pm -> Sun 5pm NY: sideways consolidation, stop hunt spike near end
  2. FMWB            — False Move Week Beginning: aggressive trap at/near Sun 5pm NY
  3. FORMATION_PENDING — After FMWB, waiting for M or W formation to complete
  4. LEVEL_1         — First level running (must break 50 EMA with volume)
  5. BOARD_MEETING_1 — Consolidation after Level 1 (retraces toward 50 EMA)
  6. LEVEL_2         — Second level running (targets 200 EMA)
  7. BOARD_MEETING_2 — Consolidation after Level 2
  8. LEVEL_3         — Third level running (trend acceleration, EMAs fan out)
  9. MIDWEEK_REVERSAL — M or W forms at Level 3 (Wed/Thu for crypto). Direction flips
 10. REVERSAL_LEVELS — New 3-level swing in opposite direction (states 4-8 repeat)
 11. FRIDAY_TRAP     — Friday UK session trap move: take profit time

Key rules:
  - HOW = High of Week: max high since Sun 5pm NY (includes wicks)
  - LOW = Low of Week: min low since Sun 5pm NY (includes wicks)
  - HOD = High of Day: max high in the prior day (5pm-5pm NY window)
  - LOD = Low of Day: min low in the prior day (5pm-5pm NY window)
  - MM will attempt either HOW or LOW every week
  - Midweek reversal expected Wed/Thu for crypto, Wed for forex
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------
NY_TZ = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------
WEEKEND_TRAP = "WEEKEND_TRAP"
FMWB = "FMWB"
FORMATION_PENDING = "FORMATION_PENDING"
LEVEL_1 = "LEVEL_1"
BOARD_MEETING_1 = "BOARD_MEETING_1"
LEVEL_2 = "LEVEL_2"
BOARD_MEETING_2 = "BOARD_MEETING_2"
LEVEL_3 = "LEVEL_3"
MIDWEEK_REVERSAL = "MIDWEEK_REVERSAL"
REVERSAL_LEVELS = "REVERSAL_LEVELS"
FRIDAY_TRAP = "FRIDAY_TRAP"

ALL_PHASES = [
    WEEKEND_TRAP, FMWB, FORMATION_PENDING,
    LEVEL_1, BOARD_MEETING_1,
    LEVEL_2, BOARD_MEETING_2,
    LEVEL_3,
    MIDWEEK_REVERSAL, REVERSAL_LEVELS,
    FRIDAY_TRAP,
]

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
FMWB_LOOKBACK_HOURS = 6       # hours after Sun 5pm to look for the false move
FMWB_MIN_MOVE_PCT = 1.0       # minimum % move to qualify as FMWB (BTC)
FMWB_MIN_MOVE_ALT_PCT = 1.5   # for altcoins (higher volatility)

EMA_50_PERIOD = 50
EMA_200_PERIOD = 200

LEVEL_CONFIRM_VOLUME_RVOL = 1.3  # relative volume threshold for level confirmation
BOARD_MEETING_RETRACE_PCT = 0.38  # minimum retrace toward 50 EMA for board meeting

WEEKEND_SPIKE_MIN_PCT = 0.5   # minimum % spike at end of weekend trap
FRIDAY_UK_START_HOUR = 8      # 8am London = 3am NY (approx)
FRIDAY_UK_END_HOUR = 16       # 4pm London = 11am NY (approx)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FMWBResult:
    """Result of False Move Week Beginning detection."""
    detected: bool
    direction: str = ""  # "up" or "down" -- the FALSE move direction
    magnitude: float = 0.0  # % move
    candle_idx: int = -1  # index in the DataFrame where FMWB peaked
    weekend_range: tuple[float, float] = (0.0, 0.0)  # (high, low) of weekend


@dataclass
class WeekendTrapBox:
    """Weekend consolidation range and any spike at its end."""
    high: float = 0.0
    low: float = 0.0
    spike_detected: bool = False
    spike_direction: str | None = None  # "up" or "down"
    spike_magnitude: float = 0.0  # % move of the spike
    start_time: datetime | None = None
    end_time: datetime | None = None


@dataclass
class CycleState:
    """Full state of the weekly cycle at a given moment."""
    phase: str = WEEKEND_TRAP  # one of the 11 phase constants
    direction: str | None = None  # "bullish", "bearish", or None
    current_level: int = 0  # 0 = no level active, 1-3 = which level, 4 = reversal levels
    levels_completed: int = 0

    # Weekly extremes
    how: float = 0.0  # High of Week
    low: float = float("inf")  # Low of Week
    hod: float = 0.0  # High of Day (prior 5pm-5pm NY window)
    lod: float = float("inf")  # Low of Day

    # Timing
    week_start: datetime | None = None  # Sunday 5pm NY
    day_start: datetime | None = None  # prior day's 5pm NY boundary

    # FMWB
    fmwb_detected: bool = False
    fmwb_direction: str | None = None  # "up"/"down" -- the FALSE direction

    # Formation
    formation_detected: bool = False
    formation_type: str | None = None  # "M" or "W"

    # Reversal
    midweek_reversal_expected: bool = False  # True if Wed/Thu and 3 levels done
    take_profit_signal: bool = False  # True if Friday UK + 3 levels

    confidence: float = 0.0  # 0-1


@dataclass
class MarketResetResult:
    """Result of market reset detection (Lesson 15 — A6).

    Three reset patterns that indicate CONTINUATION (not reversal):

    Type 1: W/M fails to break 50 EMA → continuation in prior direction.
    Type 2: Two consecutive Asia sessions at same price level.
    Type 3: Full-day consolidation (Asia to Asia).

    Attributes:
        detected: True if a reset pattern was found.
        reset_type: 1, 2, or 3.
        direction: The continuation direction (same as prior trend).
        confidence: 0-1 confidence score.
    """

    detected: bool
    reset_type: int  # 1, 2, or 3
    direction: str  # The continuation direction (same as prior trend)
    confidence: float


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _to_ny(dt: datetime) -> datetime:
    """Convert any datetime to New York timezone."""
    if dt.tzinfo is None:
        # Assume UTC if naive
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(NY_TZ)


def _sunday_5pm_ny(ref_time: datetime) -> datetime:
    """Find the most recent Sunday 5:00 PM New York time at or before ref_time.

    This is the week start boundary for the MM weekly cycle.
    """
    ny_time = _to_ny(ref_time)
    # Walk backward to find Sunday
    candidate = ny_time.replace(hour=17, minute=0, second=0, microsecond=0)
    while candidate.weekday() != 6:  # 6 = Sunday
        candidate -= timedelta(days=1)
    # If the candidate is in the future relative to ref_time, go back a week
    if candidate > ny_time:
        candidate -= timedelta(days=7)
    return candidate


def _friday_5pm_ny(ref_time: datetime) -> datetime:
    """Find the most recent Friday 5:00 PM New York time at or before ref_time.

    This is the weekend trap start boundary.
    """
    ny_time = _to_ny(ref_time)
    candidate = ny_time.replace(hour=17, minute=0, second=0, microsecond=0)
    while candidate.weekday() != 4:  # 4 = Friday
        candidate -= timedelta(days=1)
    if candidate > ny_time:
        candidate -= timedelta(days=7)
    return candidate


def _prior_day_boundaries(ref_time: datetime) -> tuple[datetime, datetime]:
    """Get the prior trading day boundaries (5pm-5pm NY).

    A 'trading day' runs from 5pm NY of the prior calendar day to 5pm NY of
    the current calendar day. The 'prior day' is the completed window before
    the current one.

    Returns:
        (start, end) of the prior 5pm-5pm window, both in NY timezone.
    """
    ny_time = _to_ny(ref_time)
    today_5pm = ny_time.replace(hour=17, minute=0, second=0, microsecond=0)

    if ny_time >= today_5pm:
        # We are past today's 5pm: current day = today 5pm -> tomorrow 5pm
        # Prior day = yesterday 5pm -> today 5pm
        prior_end = today_5pm
        prior_start = today_5pm - timedelta(days=1)
    else:
        # We are before today's 5pm: current day = yesterday 5pm -> today 5pm
        # Prior day = day-before-yesterday 5pm -> yesterday 5pm
        prior_end = today_5pm - timedelta(days=1)
        prior_start = today_5pm - timedelta(days=2)

    return prior_start, prior_end


def _is_friday_uk_session(ny_time: datetime) -> bool:
    """Check if current time is within Friday UK session hours.

    UK session is roughly 8am-4pm London, which maps to approximately
    3am-11am NY (EST) or 2am-10am NY (EDT). We use a generous window
    in NY time to account for DST shifts.
    """
    if ny_time.weekday() != 4:  # Not Friday
        return False
    # Friday between 2am and 12pm NY covers UK session with margin
    return 2 <= ny_time.hour <= 12


def _is_midweek(ny_time: datetime) -> bool:
    """Check if we are in the midweek reversal window (Wed/Thu for crypto)."""
    return ny_time.weekday() in (2, 3)  # Wednesday=2, Thursday=3


def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Compute exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def _filter_candles_by_time(
    ohlc: pd.DataFrame,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Filter candle DataFrame to a time window.

    Handles both timezone-aware and naive index/columns. The DataFrame is
    expected to have a DatetimeIndex or a 'timestamp' column.
    """
    if ohlc.empty:
        return ohlc

    idx = ohlc.index
    if not isinstance(idx, pd.DatetimeIndex):
        if "timestamp" in ohlc.columns:
            idx = pd.DatetimeIndex(ohlc["timestamp"])
        else:
            return ohlc

    # Ensure start/end are UTC for comparison
    start_utc = start.astimezone(ZoneInfo("UTC"))
    end_utc = end.astimezone(ZoneInfo("UTC"))

    if idx.tz is None:
        # Assume UTC
        idx = idx.tz_localize("UTC")

    mask = (idx >= start_utc) & (idx < end_utc)
    return ohlc.loc[mask]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WeeklyCycleTracker:
    """State machine tracking the Market Maker weekly business cycle.

    Usage:
        tracker = WeeklyCycleTracker()
        state = tracker.update(ohlc_1h, current_time)
        # state.phase tells you where we are
        # state.direction tells you the real bias
        # state.take_profit_signal tells you if it's time to exit
    """

    def __init__(self) -> None:
        self._state = CycleState()
        self._fmwb_result: FMWBResult | None = None
        self._weekend_box: WeekendTrapBox | None = None
        self._level_swing_high: float = 0.0  # running high for current level
        self._level_swing_low: float = float("inf")  # running low for current level
        self._reversal_direction: str | None = None  # direction after midweek reversal
        self._reversal_levels_completed: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        ohlc_1h: pd.DataFrame,
        current_time: datetime,
    ) -> CycleState:
        """Main method: update the state machine with latest candles.

        Call this on each scan cycle with the latest 1H candles. The method
        determines where we are in the weekly cycle and updates all state
        fields accordingly.

        Args:
            ohlc_1h: 1-hour OHLCV DataFrame with columns: open, high, low, close, volume.
                     Must have a DatetimeIndex (UTC).
            current_time: Current wall-clock time (timezone-aware or naive=UTC).

        Returns:
            Updated CycleState snapshot.
        """
        if ohlc_1h is None or ohlc_1h.empty or len(ohlc_1h) < 10:
            logger.warning("mm_weekly_cycle_insufficient_data", candle_count=0 if ohlc_1h is None else len(ohlc_1h))
            return self._state

        ny_now = _to_ny(current_time)
        week_start = _sunday_5pm_ny(current_time)
        self._state.week_start = week_start

        # Calculate HOW, LOW, HOD, LOD
        extremes = self.get_how_low(ohlc_1h, current_time)
        self._state.how = extremes["how"]
        self._state.low = extremes["low"]
        self._state.hod = extremes["hod"]
        self._state.lod = extremes["lod"]
        self._state.day_start = extremes.get("day_start")

        # Determine which phase we are in based on time and market structure
        self._advance_state_machine(ohlc_1h, ny_now, week_start)

        # Check for midweek reversal expectation
        self._state.midweek_reversal_expected = (
            _is_midweek(ny_now) and self._state.levels_completed >= 3
        )

        # Check for take profit signal
        self._state.take_profit_signal = self.should_take_profit(self._state)

        # Compute confidence
        self._state.confidence = self._compute_confidence(ohlc_1h, ny_now)

        logger.debug(
            "mm_weekly_cycle_update",
            phase=self._state.phase,
            direction=self._state.direction,
            level=self._state.current_level,
            levels_completed=self._state.levels_completed,
            how=round(self._state.how, 2),
            low=round(self._state.low, 2),
            confidence=round(self._state.confidence, 3),
        )

        return self._state

    def get_how_low(
        self,
        ohlc_1h: pd.DataFrame,
        current_time: datetime,
    ) -> dict:
        """Calculate High of Week, Low of Week, High of Day, Low of Day.

        Args:
            ohlc_1h: 1-hour OHLCV DataFrame.
            current_time: Current time (timezone-aware or naive=UTC).

        Returns:
            dict with keys: how, low, hod, lod, day_start.
        """
        week_start = _sunday_5pm_ny(current_time)
        prior_start, prior_end = _prior_day_boundaries(current_time)

        # HOW / LOW: all candles since Sunday 5pm NY
        week_candles = _filter_candles_by_time(
            ohlc_1h, week_start, _to_ny(current_time) + timedelta(hours=1),
        )
        if not week_candles.empty:
            how = float(week_candles["high"].max())
            low_val = float(week_candles["low"].min())
        else:
            how = 0.0
            low_val = float("inf")

        # HOD / LOD: prior 5pm-5pm NY trading day
        day_candles = _filter_candles_by_time(ohlc_1h, prior_start, prior_end)
        if not day_candles.empty:
            hod = float(day_candles["high"].max())
            lod = float(day_candles["low"].min())
        else:
            hod = 0.0
            lod = float("inf")

        return {
            "how": how,
            "low": low_val,
            "hod": hod,
            "lod": lod,
            "day_start": prior_start,
        }

    def detect_fmwb(
        self,
        ohlc_1h: pd.DataFrame,
        weekend_range: tuple[float, float],
    ) -> FMWBResult:
        """Detect the False Move Week Beginning (FMWB).

        After Sunday 5pm NY, look for a sharp move that breaks outside the
        weekend consolidation range. This is the FALSE direction -- the real
        trend will go the opposite way.

        Args:
            ohlc_1h: 1-hour OHLCV candles (must include post-Sunday 5pm data).
            weekend_range: (high, low) of the weekend consolidation box.

        Returns:
            FMWBResult indicating whether FMWB was detected.
        """
        if ohlc_1h is None or ohlc_1h.empty:
            return FMWBResult(detected=False, weekend_range=weekend_range)

        wk_high, wk_low = weekend_range
        if wk_high <= 0 or wk_low <= 0 or wk_high <= wk_low:
            return FMWBResult(detected=False, weekend_range=weekend_range)

        weekend_range_size = wk_high - wk_low
        midpoint = (wk_high + wk_low) / 2.0

        week_start = self._state.week_start
        if week_start is None:
            return FMWBResult(detected=False, weekend_range=weekend_range)

        # Look at candles in the first FMWB_LOOKBACK_HOURS after week start
        fmwb_end = week_start + timedelta(hours=FMWB_LOOKBACK_HOURS)
        fmwb_candles = _filter_candles_by_time(ohlc_1h, week_start, fmwb_end)

        if fmwb_candles.empty:
            return FMWBResult(detected=False, weekend_range=weekend_range)

        # Check for a break above weekend high or below weekend low
        max_high = float(fmwb_candles["high"].max())
        min_low = float(fmwb_candles["low"].min())

        break_above = max_high > wk_high
        break_below = min_low < wk_low

        if not break_above and not break_below:
            return FMWBResult(detected=False, weekend_range=weekend_range)

        # Determine which break is more significant
        above_magnitude_pct = ((max_high - wk_high) / midpoint) * 100 if break_above else 0.0
        below_magnitude_pct = ((wk_low - min_low) / midpoint) * 100 if break_below else 0.0

        # Take the larger break
        if above_magnitude_pct >= below_magnitude_pct:
            direction = "up"  # false move is UP -> real direction is bearish
            magnitude = above_magnitude_pct
            peak_idx_raw = fmwb_candles["high"].idxmax()
            peak_idx = int(peak_idx_raw) if isinstance(peak_idx_raw, (int, np.integer)) else -1
        else:
            direction = "down"  # false move is DOWN -> real direction is bullish
            magnitude = below_magnitude_pct
            peak_idx_raw = fmwb_candles["low"].idxmin()
            peak_idx = int(peak_idx_raw) if isinstance(peak_idx_raw, (int, np.integer)) else -1

        # Must exceed minimum threshold
        if magnitude < FMWB_MIN_MOVE_PCT:
            logger.debug(
                "fmwb_below_threshold",
                magnitude=round(magnitude, 3),
                threshold=FMWB_MIN_MOVE_PCT,
            )
            return FMWBResult(detected=False, weekend_range=weekend_range)

        logger.info(
            "fmwb_detected",
            direction=direction,
            magnitude=round(magnitude, 3),
            weekend_high=round(wk_high, 2),
            weekend_low=round(wk_low, 2),
        )

        return FMWBResult(
            detected=True,
            direction=direction,
            magnitude=magnitude,
            candle_idx=peak_idx,
            weekend_range=weekend_range,
        )

    def detect_weekend_trap_box(
        self,
        ohlc_1h: pd.DataFrame,
        week_start: datetime,
    ) -> WeekendTrapBox:
        """Identify the weekend consolidation range and any exit spike.

        The weekend trap runs from Friday 5pm NY to Sunday 5pm NY. During
        this window, price consolidates in a range. Near the end (typically
        Sat night / Sun), there may be a stop-hunt spike that pokes outside
        the consolidation range before returning.

        Args:
            ohlc_1h: 1-hour OHLCV candles.
            week_start: The Sunday 5pm NY boundary for the current week.

        Returns:
            WeekendTrapBox describing the range and any spike.
        """
        # Weekend: Friday 5pm -> Sunday 5pm NY
        friday_5pm = week_start - timedelta(days=2)  # Sunday - 2 = Friday
        weekend_candles = _filter_candles_by_time(ohlc_1h, friday_5pm, week_start)

        if weekend_candles.empty:
            return WeekendTrapBox(start_time=friday_5pm, end_time=week_start)

        box_high = float(weekend_candles["high"].max())
        box_low = float(weekend_candles["low"].min())

        # Detect spike: look at the last few candles of the weekend
        # A spike is a wick that extends significantly beyond the consolidation body
        num_candles = len(weekend_candles)
        late_candles = weekend_candles.iloc[max(0, num_candles - 6):]  # last 6 hours

        if late_candles.empty:
            return WeekendTrapBox(
                high=box_high,
                low=box_low,
                start_time=friday_5pm,
                end_time=week_start,
            )

        # The "body" of consolidation (exclude the last 6 hours to isolate spike)
        core_candles = weekend_candles.iloc[:max(1, num_candles - 6)]
        core_high = float(core_candles["high"].max())
        core_low = float(core_candles["low"].min())
        core_range = core_high - core_low if core_high > core_low else 1.0

        # Check if late candles spike beyond the core range
        late_max = float(late_candles["high"].max())
        late_min = float(late_candles["low"].min())

        spike_up = late_max > core_high
        spike_down = late_min < core_low

        spike_detected = False
        spike_direction = None
        spike_magnitude = 0.0

        if spike_up or spike_down:
            up_pct = ((late_max - core_high) / core_high) * 100 if spike_up else 0.0
            down_pct = ((core_low - late_min) / core_low) * 100 if spike_down else 0.0

            if up_pct >= WEEKEND_SPIKE_MIN_PCT or down_pct >= WEEKEND_SPIKE_MIN_PCT:
                spike_detected = True
                if up_pct >= down_pct:
                    spike_direction = "up"
                    spike_magnitude = up_pct
                else:
                    spike_direction = "down"
                    spike_magnitude = down_pct

                logger.info(
                    "weekend_spike_detected",
                    direction=spike_direction,
                    magnitude=round(spike_magnitude, 3),
                    core_high=round(core_high, 2),
                    core_low=round(core_low, 2),
                )

        return WeekendTrapBox(
            high=box_high,
            low=box_low,
            spike_detected=spike_detected,
            spike_direction=spike_direction,
            spike_magnitude=spike_magnitude,
            start_time=friday_5pm,
            end_time=week_start,
        )

    def should_take_profit(self, state: CycleState) -> bool:
        """Check if conditions are met for Friday take-profit.

        Returns True if:
          - We are in Friday UK session
          - At least 3 levels have completed (original + reversal counts)
        """
        if state.week_start is None:
            return False

        ny_now = _to_ny(datetime.now(ZoneInfo("UTC")))
        if _is_friday_uk_session(ny_now) and state.levels_completed >= 3:
            return True
        return False

    def reset_week(self) -> None:
        """Reset all state for a new week. Call at Sunday 5pm NY."""
        logger.info("mm_weekly_cycle_reset")
        self._state = CycleState()
        self._fmwb_result = None
        self._weekend_box = None
        self._level_swing_high = 0.0
        self._level_swing_low = float("inf")
        self._reversal_direction = None
        self._reversal_levels_completed = 0

    # ------------------------------------------------------------------
    # State machine transitions
    # ------------------------------------------------------------------

    def _advance_state_machine(
        self,
        ohlc_1h: pd.DataFrame,
        ny_now: datetime,
        week_start: datetime,
    ) -> None:
        """Run the state machine forward based on current conditions.

        This is the heart of the module. It evaluates market conditions and
        advances through the 11 phases in order.
        """
        current_phase = self._state.phase
        hours_since_open = (ny_now - week_start).total_seconds() / 3600.0

        # ----- Phase 1: WEEKEND_TRAP -----
        # Active from Friday 5pm to Sunday 5pm NY
        if current_phase == WEEKEND_TRAP:
            self._handle_weekend_trap(ohlc_1h, ny_now, week_start, hours_since_open)
            return

        # ----- Phase 2: FMWB -----
        if current_phase == FMWB:
            self._handle_fmwb(ohlc_1h, ny_now, week_start, hours_since_open)
            return

        # ----- Phase 3: FORMATION_PENDING -----
        if current_phase == FORMATION_PENDING:
            self._handle_formation_pending(ohlc_1h, ny_now, hours_since_open)
            return

        # ----- Phases 4-8: Levels and Board Meetings -----
        if current_phase in (LEVEL_1, BOARD_MEETING_1, LEVEL_2, BOARD_MEETING_2, LEVEL_3):
            self._handle_levels(ohlc_1h, ny_now, hours_since_open)
            return

        # ----- Phase 9: MIDWEEK_REVERSAL -----
        if current_phase == MIDWEEK_REVERSAL:
            self._handle_midweek_reversal(ohlc_1h, ny_now)
            return

        # ----- Phase 10: REVERSAL_LEVELS -----
        if current_phase == REVERSAL_LEVELS:
            self._handle_reversal_levels(ohlc_1h, ny_now)
            return

        # ----- Phase 11: FRIDAY_TRAP -----
        if current_phase == FRIDAY_TRAP:
            # Terminal state for the week -- stay here until reset
            return

    def _handle_weekend_trap(
        self,
        ohlc_1h: pd.DataFrame,
        ny_now: datetime,
        week_start: datetime,
        hours_since_open: float,
    ) -> None:
        """Handle WEEKEND_TRAP -> FMWB transition.

        We are in the weekend consolidation zone. Detect the weekend box
        and wait for the week to officially open (Sunday 5pm NY) or for
        a spike signaling the end of the trap.
        """
        # Build weekend box if not done yet
        if self._weekend_box is None:
            self._weekend_box = self.detect_weekend_trap_box(ohlc_1h, week_start)

        # Transition: once we are past Sunday 5pm NY, move to FMWB detection
        if hours_since_open >= 0:
            # We are at or past Sunday 5pm -- the week has started
            logger.info(
                "transition_weekend_to_fmwb",
                weekend_high=round(self._weekend_box.high, 2) if self._weekend_box else 0,
                weekend_low=round(self._weekend_box.low, 2) if self._weekend_box else 0,
                spike=self._weekend_box.spike_detected if self._weekend_box else False,
            )
            self._state.phase = FMWB

    def _handle_fmwb(
        self,
        ohlc_1h: pd.DataFrame,
        ny_now: datetime,
        week_start: datetime,
        hours_since_open: float,
    ) -> None:
        """Handle FMWB -> FORMATION_PENDING transition.

        Detect the False Move Week Beginning. Once detected (or the lookback
        window expires), transition to FORMATION_PENDING.
        """
        if self._fmwb_result is None:
            weekend_range = (
                (self._weekend_box.high, self._weekend_box.low)
                if self._weekend_box
                else (self._state.how, self._state.low)
            )
            self._fmwb_result = self.detect_fmwb(ohlc_1h, weekend_range)

        if self._fmwb_result.detected:
            self._state.fmwb_detected = True
            self._state.fmwb_direction = self._fmwb_result.direction
            # Real direction is opposite of false move
            if self._fmwb_result.direction == "up":
                self._state.direction = "bearish"
            else:
                self._state.direction = "bullish"

            logger.info(
                "transition_fmwb_to_formation",
                fmwb_direction=self._fmwb_result.direction,
                real_direction=self._state.direction,
                magnitude=round(self._fmwb_result.magnitude, 3),
            )
            self._state.phase = FORMATION_PENDING
            return

        # If the lookback window has expired without FMWB, still move forward
        # (sometimes the false move is subtle or doesn't happen)
        if hours_since_open > FMWB_LOOKBACK_HOURS:
            logger.info("fmwb_lookback_expired_no_detection", hours=round(hours_since_open, 1))
            self._state.fmwb_detected = False
            self._state.phase = FORMATION_PENDING

    def _handle_formation_pending(
        self,
        ohlc_1h: pd.DataFrame,
        ny_now: datetime,
        hours_since_open: float,
    ) -> None:
        """Handle FORMATION_PENDING -> LEVEL_1 transition.

        After FMWB, we wait for an M or W formation to confirm the real
        direction. For now, we use a simplified detection:
        - If direction is bullish, look for a W bottom (double bottom)
        - If direction is bearish, look for an M top (double top)

        Also transitions to LEVEL_1 if price breaks the 50 EMA with volume
        confirmation, even without a classic formation.
        """
        direction = self._state.direction

        # Simplified formation detection: look for price structure in recent candles
        if direction and len(ohlc_1h) >= 20:
            formation = self._detect_simple_formation(ohlc_1h, direction)
            if formation:
                self._state.formation_detected = True
                self._state.formation_type = formation
                logger.info(
                    "formation_detected",
                    formation_type=formation,
                    direction=direction,
                )

        # Check for 50 EMA break with volume as level confirmation
        ema_break = self._check_ema_break(ohlc_1h, EMA_50_PERIOD, direction)

        if self._state.formation_detected or ema_break:
            self._state.phase = LEVEL_1
            self._state.current_level = 1
            self._reset_level_tracking(ohlc_1h)
            logger.info(
                "transition_to_level_1",
                formation=self._state.formation_detected,
                ema_break=ema_break,
                direction=direction,
            )
            return

        # Fallback: if we have been waiting too long (>12h after week start),
        # infer direction from price action relative to FMWB
        if hours_since_open > 12 and direction:
            logger.info("formation_timeout_advancing_to_level_1", hours=round(hours_since_open, 1))
            self._state.phase = LEVEL_1
            self._state.current_level = 1
            self._reset_level_tracking(ohlc_1h)

    def _handle_levels(
        self,
        ohlc_1h: pd.DataFrame,
        ny_now: datetime,
        hours_since_open: float,
    ) -> None:
        """Handle level progression: LEVEL_1 -> BM1 -> LEVEL_2 -> BM2 -> LEVEL_3.

        Each level runs until price retraces (board meeting). Each board
        meeting waits for the next push.
        """
        phase = self._state.phase
        direction = self._state.direction

        if phase == LEVEL_1:
            if self._check_level_retrace(ohlc_1h, direction):
                self._state.phase = BOARD_MEETING_1
                self._state.levels_completed = 1
                logger.info("transition_to_board_meeting_1")
            elif _is_friday_uk_session(ny_now) and self._state.levels_completed >= 0:
                self._transition_to_friday_trap()

        elif phase == BOARD_MEETING_1:
            if self._check_ema_break(ohlc_1h, EMA_50_PERIOD, direction):
                self._state.phase = LEVEL_2
                self._state.current_level = 2
                self._reset_level_tracking(ohlc_1h)
                logger.info("transition_to_level_2")
            elif _is_midweek(ny_now):
                # If it's midweek and we're stuck in board meeting, check for reversal
                pass

        elif phase == LEVEL_2:
            if self._check_level_retrace(ohlc_1h, direction):
                self._state.phase = BOARD_MEETING_2
                self._state.levels_completed = 2
                logger.info("transition_to_board_meeting_2")
            elif _is_friday_uk_session(ny_now):
                self._transition_to_friday_trap()

        elif phase == BOARD_MEETING_2:
            # Check for 200 EMA break for level 3
            if self._check_ema_break(ohlc_1h, EMA_200_PERIOD, direction):
                self._state.phase = LEVEL_3
                self._state.current_level = 3
                self._reset_level_tracking(ohlc_1h)
                logger.info("transition_to_level_3")

        elif phase == LEVEL_3:
            self._state.levels_completed = 3
            # Level 3 can transition to midweek reversal or friday trap
            if _is_midweek(ny_now):
                # Check for reversal formation at level 3
                rev_formation = self._detect_reversal_at_level3(ohlc_1h, direction)
                if rev_formation:
                    # Before flipping direction, check for market reset (Type 1).
                    # A reset means the formation is a CONTINUATION signal,
                    # not a reversal — keep direction and stay at Level 3.
                    reset = self.detect_market_reset(
                        ohlc_1h, ema_state=None, prior_direction=direction or "",
                    )
                    if reset is not None and reset.reset_type == 1:
                        logger.info(
                            "market_reset_blocks_reversal",
                            reset_type=reset.reset_type,
                            direction=reset.direction,
                            confidence=reset.confidence,
                        )
                        # Stay in LEVEL_3, don't flip direction
                    else:
                        self._state.phase = MIDWEEK_REVERSAL
                        logger.info("transition_to_midweek_reversal", formation=rev_formation)
            elif _is_friday_uk_session(ny_now):
                self._transition_to_friday_trap()

    def _handle_midweek_reversal(
        self,
        ohlc_1h: pd.DataFrame,
        ny_now: datetime,
    ) -> None:
        """Handle MIDWEEK_REVERSAL -> REVERSAL_LEVELS transition.

        At the midweek reversal, direction flips. A new M or W formation
        confirms the reversal, and then we run 3 levels in the opposite
        direction.
        """
        # Flip direction
        old_direction = self._state.direction
        if old_direction == "bullish":
            self._reversal_direction = "bearish"
        elif old_direction == "bearish":
            self._reversal_direction = "bullish"
        else:
            self._reversal_direction = None

        self._state.direction = self._reversal_direction
        self._state.phase = REVERSAL_LEVELS
        self._state.current_level = 1
        self._reversal_levels_completed = 0
        self._reset_level_tracking(ohlc_1h)

        logger.info(
            "midweek_reversal_activated",
            old_direction=old_direction,
            new_direction=self._reversal_direction,
        )

    def _handle_reversal_levels(
        self,
        ohlc_1h: pd.DataFrame,
        ny_now: datetime,
    ) -> None:
        """Handle REVERSAL_LEVELS phase.

        This repeats the 3-level pattern (LEVEL -> BOARD_MEETING -> LEVEL...)
        in the reversed direction. We track progress with _reversal_levels_completed.
        """
        direction = self._state.direction

        if self._reversal_levels_completed < 3:
            # Check for retrace (board meeting between levels)
            if self._check_level_retrace(ohlc_1h, direction):
                self._reversal_levels_completed += 1
                self._state.levels_completed = 3 + self._reversal_levels_completed
                self._state.current_level = min(self._reversal_levels_completed + 1, 3)
                self._reset_level_tracking(ohlc_1h)
                logger.info(
                    "reversal_level_completed",
                    reversal_levels=self._reversal_levels_completed,
                    total_levels=self._state.levels_completed,
                )

        # Transition to Friday Trap if it's time
        if _is_friday_uk_session(ny_now):
            self._transition_to_friday_trap()

    def _transition_to_friday_trap(self) -> None:
        """Transition to the FRIDAY_TRAP terminal phase."""
        self._state.phase = FRIDAY_TRAP
        self._state.take_profit_signal = True
        logger.info(
            "transition_to_friday_trap",
            levels_completed=self._state.levels_completed,
            direction=self._state.direction,
        )

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect_simple_formation(
        self,
        ohlc_1h: pd.DataFrame,
        direction: str,
    ) -> str | None:
        """Simplified M/W formation detection.

        Looks for a double-bottom (W) or double-top (M) pattern in the last
        20 candles using swing point analysis.

        Returns:
            "M" for double top (bearish), "W" for double bottom (bullish),
            or None if no formation detected.
        """
        recent = ohlc_1h.tail(20)
        if len(recent) < 10:
            return None

        highs = recent["high"].values
        lows = recent["low"].values
        closes = recent["close"].values

        if direction == "bullish":
            # Look for W bottom: two lows of similar depth with a higher middle
            min_idx = np.argmin(lows)
            min_val = lows[min_idx]

            # Look for second low of similar depth (within 0.5% of first)
            tolerance = min_val * 0.005
            similar_lows = np.where(
                (np.abs(lows - min_val) <= tolerance)
                & (np.arange(len(lows)) != min_idx)
            )[0]

            if len(similar_lows) > 0:
                # Check that there is a higher point between the two lows
                second_idx = similar_lows[0]
                start, end = sorted([min_idx, second_idx])
                if end - start >= 3:  # at least 3 candles between lows
                    middle_high = np.max(highs[start:end + 1])
                    if middle_high > min_val * 1.003:  # middle is at least 0.3% higher
                        return "W"

        elif direction == "bearish":
            # Look for M top: two highs of similar height with a lower middle
            max_idx = np.argmax(highs)
            max_val = highs[max_idx]

            tolerance = max_val * 0.005
            similar_highs = np.where(
                (np.abs(highs - max_val) <= tolerance)
                & (np.arange(len(highs)) != max_idx)
            )[0]

            if len(similar_highs) > 0:
                second_idx = similar_highs[0]
                start, end = sorted([max_idx, second_idx])
                if end - start >= 3:
                    middle_low = np.min(lows[start:end + 1])
                    if middle_low < max_val * 0.997:
                        return "M"

        return None

    def _check_ema_break(
        self,
        ohlc_1h: pd.DataFrame,
        ema_period: int,
        direction: str | None,
    ) -> bool:
        """Check if the most recent candle has broken an EMA with conviction.

        For a bullish break: close > EMA and candle body is mostly above EMA.
        For a bearish break: close < EMA and candle body is mostly below EMA.

        Args:
            ohlc_1h: 1H OHLCV DataFrame.
            ema_period: EMA period to check (50 or 200).
            direction: Expected direction of break.

        Returns:
            True if a valid break is detected.
        """
        if direction is None or len(ohlc_1h) < ema_period + 5:
            return False

        ema = _compute_ema(ohlc_1h["close"], ema_period)
        last_close = float(ohlc_1h["close"].iloc[-1])
        last_ema = float(ema.iloc[-1])
        prev_close = float(ohlc_1h["close"].iloc[-2])
        prev_ema = float(ema.iloc[-2])

        if direction == "bullish":
            # Close crossed above EMA
            return prev_close <= prev_ema and last_close > last_ema
        elif direction == "bearish":
            # Close crossed below EMA
            return prev_close >= prev_ema and last_close < last_ema

        return False

    def _check_level_retrace(
        self,
        ohlc_1h: pd.DataFrame,
        direction: str | None,
    ) -> bool:
        """Check if price has retraced sufficiently for a board meeting.

        A level is 'done' when price pulls back toward the 50 EMA after
        advancing. We measure the retrace as a % of the level's swing.

        Returns:
            True if retrace is sufficient for a board meeting.
        """
        if direction is None or len(ohlc_1h) < 10:
            return False

        recent = ohlc_1h.tail(10)
        ema_50 = _compute_ema(ohlc_1h["close"], EMA_50_PERIOD)

        if len(ema_50) < 1:
            return False

        current_ema = float(ema_50.iloc[-1])

        if direction == "bullish":
            # Track the swing: how far above EMA did we get?
            swing_high = float(recent["high"].max())
            self._level_swing_high = max(self._level_swing_high, swing_high)

            if self._level_swing_high <= current_ema:
                return False

            swing_range = self._level_swing_high - current_ema
            current_retrace = self._level_swing_high - float(recent["low"].iloc[-1])
            retrace_pct = current_retrace / swing_range if swing_range > 0 else 0

            return retrace_pct >= BOARD_MEETING_RETRACE_PCT

        elif direction == "bearish":
            swing_low = float(recent["low"].min())
            self._level_swing_low = min(self._level_swing_low, swing_low)

            if self._level_swing_low >= current_ema:
                return False

            swing_range = current_ema - self._level_swing_low
            current_retrace = float(recent["high"].iloc[-1]) - self._level_swing_low
            retrace_pct = current_retrace / swing_range if swing_range > 0 else 0

            return retrace_pct >= BOARD_MEETING_RETRACE_PCT

        return False

    def _detect_reversal_at_level3(
        self,
        ohlc_1h: pd.DataFrame,
        direction: str | None,
    ) -> str | None:
        """Detect a reversal formation at Level 3 during midweek.

        At Level 3, the original trend is exhausted and a reversal M or W
        should form. The expected formation is the OPPOSITE of the current
        direction:
          - Bullish trend -> expect M top -> reversal to bearish
          - Bearish trend -> expect W bottom -> reversal to bullish

        Returns:
            Formation type ("M" or "W") or None.
        """
        if direction is None:
            return None

        # For reversal, we look for the opposite formation
        reversal_direction = "bearish" if direction == "bullish" else "bullish"
        return self._detect_simple_formation(ohlc_1h, reversal_direction)

    def _reset_level_tracking(self, ohlc_1h: pd.DataFrame) -> None:
        """Reset swing tracking for a new level."""
        if ohlc_1h is not None and not ohlc_1h.empty:
            last = ohlc_1h.iloc[-1]
            self._level_swing_high = float(last["high"])
            self._level_swing_low = float(last["low"])
        else:
            self._level_swing_high = 0.0
            self._level_swing_low = float("inf")

    # ------------------------------------------------------------------
    # Market Resets (Lesson 15 — A6)
    # ------------------------------------------------------------------

    def detect_market_reset(
        self,
        candles_1h: pd.DataFrame,
        ema_state: dict | None,
        prior_direction: str,
    ) -> "MarketResetResult | None":
        """Detect a market reset pattern indicating CONTINUATION (not reversal).

        Three reset types from Lesson 15:

        Type 1: W fails to break 50 EMA → continuation downtrend.
            After a 3-level drop completes, a W forms but FAILS to break
            above the 50 EMA. The downtrend continues. Do NOT enter long.

        Type 2: Two consecutive Asia sessions at same price level.
            UK/US do fakeouts both days, close back in range. Final stop
            hunt indicates real direction. (Detection only — not wired
            into state machine yet.)

        Type 3: Full-day consolidation (Asia to Asia).
            After cycle completes, MM consolidates for an entire day.
            Stop hunt or fake move in opposite direction → continuation.
            (Detection only — not wired into state machine yet.)

        Args:
            candles_1h: 1H OHLCV DataFrame.
            ema_state: Dict with EMA values (e.g. {"ema50": float}) or None.
            prior_direction: The prior trend direction ("bullish" or "bearish").

        Returns:
            MarketResetResult if detected, else None.
        """
        if candles_1h is None or candles_1h.empty or len(candles_1h) < 50:
            return None

        # --- Type 1: W/M fails to break 50 EMA → continuation ---
        type1 = self._detect_reset_type1(candles_1h, prior_direction)
        if type1 is not None:
            return type1

        # --- Type 2: Two consecutive Asia sessions at same level ---
        type2 = self._detect_reset_type2(candles_1h, prior_direction)
        if type2 is not None:
            return type2

        # --- Type 3: Full-day consolidation ---
        type3 = self._detect_reset_type3(candles_1h, prior_direction)
        if type3 is not None:
            return type3

        return None

    def _detect_reset_type1(
        self,
        candles_1h: pd.DataFrame,
        prior_direction: str,
    ) -> "MarketResetResult | None":
        """Type 1: Formation fails to break 50 EMA → continuation.

        After a downtrend: W forms but fails to break above 50 EMA.
        After an uptrend: M forms but fails to break below 50 EMA.
        """
        if len(candles_1h) < EMA_50_PERIOD + 10:
            return None

        ema50 = _compute_ema(candles_1h["close"], EMA_50_PERIOD)
        recent = candles_1h.tail(20)
        recent_ema = ema50.tail(20)

        if prior_direction == "bearish":
            # After downtrend, look for a W that failed to break above 50 EMA
            # A failed W = price rallied (W shape) but the rally highs stayed
            # below the 50 EMA
            highs = recent["high"].values.astype(float)
            ema_vals = recent_ema.values.astype(float)

            # Find the highest point in the recent window
            max_high = float(np.max(highs))
            max_ema = float(np.max(ema_vals))

            # W attempt: price tried to go up but stayed below EMA
            # Check that there was a rally attempt (some highs approached EMA)
            # but never broke above it
            close_to_ema = np.any(highs > ema_vals * 0.995)
            broke_ema = np.any(highs > ema_vals * 1.005)

            if close_to_ema and not broke_ema:
                logger.info(
                    "market_reset_type1",
                    direction="bearish",
                    prior_direction=prior_direction,
                    max_high=round(max_high, 2),
                    max_ema=round(max_ema, 2),
                )
                return MarketResetResult(
                    detected=True,
                    reset_type=1,
                    direction=prior_direction,  # continuation
                    confidence=0.7,
                )

        elif prior_direction == "bullish":
            # After uptrend, look for an M that failed to break below 50 EMA
            lows = recent["low"].values.astype(float)
            ema_vals = recent_ema.values.astype(float)

            min_low = float(np.min(lows))

            close_to_ema = np.any(lows < ema_vals * 1.005)
            broke_ema = np.any(lows < ema_vals * 0.995)

            if close_to_ema and not broke_ema:
                logger.info(
                    "market_reset_type1",
                    direction="bullish",
                    prior_direction=prior_direction,
                    min_low=round(min_low, 2),
                )
                return MarketResetResult(
                    detected=True,
                    reset_type=1,
                    direction=prior_direction,
                    confidence=0.7,
                )

        return None

    def _detect_reset_type2(
        self,
        candles_1h: pd.DataFrame,
        prior_direction: str,
    ) -> "MarketResetResult | None":
        """Type 2: Two consecutive Asia sessions at same price level.

        UK/US do fakeouts high and low both days, close back in range.
        Detection only — logged for telemetry, not hard-wired into state machine.
        """
        if len(candles_1h) < 48:  # need ~2 days of data
            return None

        # Compare the last two Asia sessions (roughly hours 0-8 UTC each day)
        # For simplicity, compare the range of the last 24h vs prior 24h
        last_24 = candles_1h.iloc[-24:]
        prev_24 = candles_1h.iloc[-48:-24]

        if len(last_24) < 24 or len(prev_24) < 24:
            return None

        # Asia is roughly the first 8 candles of each 24h period
        asia_last = last_24.iloc[:8]
        asia_prev = prev_24.iloc[:8]

        # Check if both Asia sessions consolidated at the same level
        last_mid = (float(asia_last["high"].max()) + float(asia_last["low"].min())) / 2
        prev_mid = (float(asia_prev["high"].max()) + float(asia_prev["low"].min())) / 2

        if last_mid <= 0 or prev_mid <= 0:
            return None

        # Same level = midpoints within 0.5% of each other
        diff_pct = abs(last_mid - prev_mid) / last_mid
        if diff_pct < 0.005:
            logger.info(
                "market_reset_type2",
                prior_direction=prior_direction,
                last_mid=round(last_mid, 2),
                prev_mid=round(prev_mid, 2),
                diff_pct=round(diff_pct * 100, 3),
            )
            return MarketResetResult(
                detected=True,
                reset_type=2,
                direction=prior_direction,
                confidence=0.5,
            )

        return None

    def _detect_reset_type3(
        self,
        candles_1h: pd.DataFrame,
        prior_direction: str,
    ) -> "MarketResetResult | None":
        """Type 3: Full-day consolidation (Asia to Asia).

        After cycle completes, MM consolidates for an entire day. The
        range of the last 24 candles is very tight (< 1.5% of price).
        Detection only — logged for telemetry.
        """
        if len(candles_1h) < 24:
            return None

        last_24 = candles_1h.iloc[-24:]
        day_high = float(last_24["high"].max())
        day_low = float(last_24["low"].min())

        if day_low <= 0:
            return None

        day_range_pct = (day_high - day_low) / day_low * 100

        # Full-day consolidation: range < 1.5%
        if day_range_pct < 1.5:
            logger.info(
                "market_reset_type3",
                prior_direction=prior_direction,
                day_range_pct=round(day_range_pct, 3),
                day_high=round(day_high, 2),
                day_low=round(day_low, 2),
            )
            return MarketResetResult(
                detected=True,
                reset_type=3,
                direction=prior_direction,
                confidence=0.4,
            )

        return None

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        ohlc_1h: pd.DataFrame,
        ny_now: datetime,
    ) -> float:
        """Compute a confidence score (0-1) for the current cycle state.

        Factors:
          - FMWB detected and clear direction: +0.25
          - Formation confirmed: +0.20
          - Multiple levels completed: +0.05 per level (up to 0.15)
          - Midweek timing alignment: +0.15
          - Volume confirmation: +0.10
          - Weekend box quality: +0.15
        """
        score = 0.0

        # FMWB clarity
        if self._state.fmwb_detected and self._state.direction:
            score += 0.25
            # Bonus for strong FMWB magnitude
            if self._fmwb_result and self._fmwb_result.magnitude > 2.0:
                score += 0.05

        # Formation confirmed
        if self._state.formation_detected:
            score += 0.20

        # Levels completed
        score += min(self._state.levels_completed * 0.05, 0.15)

        # Midweek timing
        if _is_midweek(ny_now) and self._state.levels_completed >= 2:
            score += 0.15
        elif self._state.levels_completed >= 1:
            score += 0.05

        # Weekend box quality
        if self._weekend_box and self._weekend_box.high > 0:
            box_range_pct = (
                (self._weekend_box.high - self._weekend_box.low)
                / self._weekend_box.low * 100
                if self._weekend_box.low > 0
                else 0
            )
            # Tight box = clearer setup
            if 0.5 <= box_range_pct <= 3.0:
                score += 0.15
            elif box_range_pct > 0:
                score += 0.05

        return min(score, 1.0)
