"""Market Maker session timing for crypto markets.

Tracks MM session boundaries using New York timezone (America/New_York),
which automatically accounts for DST transitions. All session definitions
are anchored to NY wall-clock time.

Sessions (NY time):
    Dead Zone:  5:00pm - 8:00pm  (all MMs off duty)
    Asia Gap:   8:00pm - 8:30pm  (handover)
    Asia Open:  8:30pm - 3:00am  (next day)
    UK Gap:     3:00am - 3:30am
    UK Open:    3:30am - 9:00am
    US Gap:     9:00am - 9:30am  (handover)
    US Open:    9:30am - 5:00pm

Week boundaries:
    Start: Sunday 5:00pm NY
    End:   Friday 5:00pm NY

Day boundaries:
    5:00pm NY to 5:00pm NY (next calendar day)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from src.utils.logging import get_logger

logger = get_logger(__name__)

NY_TZ = ZoneInfo("America/New_York")

# --- Session boundary times (NY wall-clock) ---

_DEAD_ZONE_START = time(17, 0)   # 5:00pm
_DEAD_ZONE_END = time(20, 0)     # 8:00pm

_ASIA_GAP_START = time(20, 0)    # 8:00pm
_ASIA_GAP_END = time(20, 30)     # 8:30pm

_ASIA_OPEN_START = time(20, 30)  # 8:30pm
# Asia wraps midnight — ends at 3:00am next calendar day
_ASIA_OPEN_END = time(3, 0)      # 3:00am

_UK_GAP_START = time(3, 0)       # 3:00am
_UK_GAP_END = time(3, 30)        # 3:30am

_UK_OPEN_START = time(3, 30)     # 3:30am
_UK_OPEN_END = time(9, 0)        # 9:00am

_US_GAP_START = time(9, 0)       # 9:00am
_US_GAP_END = time(9, 30)        # 9:30am

_US_OPEN_START = time(9, 30)     # 9:30am
_US_OPEN_END = time(17, 0)       # 5:00pm


# Ordered session definitions: (name, start, end, is_gap, wraps_midnight)
_SESSIONS: list[tuple[str, time, time, bool, bool]] = [
    ("dead_zone",  _DEAD_ZONE_START,  _DEAD_ZONE_END,  False, False),
    ("asia_gap",   _ASIA_GAP_START,   _ASIA_GAP_END,   True,  False),
    ("asia",       _ASIA_OPEN_START,  _ASIA_OPEN_END,  False, True),
    ("uk_gap",     _UK_GAP_START,     _UK_GAP_END,     True,  False),
    ("uk",         _UK_OPEN_START,    _UK_OPEN_END,     False, False),
    ("us_gap",     _US_GAP_START,     _US_GAP_END,      True,  False),
    ("us",         _US_OPEN_START,    _US_OPEN_END,      False, False),
]


@dataclass
class SessionInfo:
    """Result of MM session timing analysis.

    Attributes:
        session_name: Current session identifier. One of: "asia", "uk", "us",
            "dead_zone", "asia_gap", "uk_gap", "us_gap".
        is_gap: True if in a 30-minute handover gap between sessions.
        session_start: Session start as a timezone-aware datetime (NY).
        session_end: Session end as a timezone-aware datetime (NY).
        minutes_remaining: Minutes remaining in the current session.
        day_of_week: ISO weekday mapped to 0=Mon ... 6=Sun.
        is_weekend: True if between Friday 5pm NY and Sunday 5pm NY.
    """

    session_name: str
    is_gap: bool
    session_start: datetime
    session_end: datetime
    minutes_remaining: float
    day_of_week: int
    is_weekend: bool


def _to_ny(dt: datetime) -> datetime:
    """Convert any datetime to America/New_York.

    Naive datetimes are assumed UTC before conversion.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(NY_TZ)


def _in_range(t: time, start: time, end: time, *, wraps_midnight: bool = False) -> bool:
    """Check if time ``t`` falls within [start, end).

    When *wraps_midnight* is True the range is treated as spanning midnight,
    e.g. 20:30 -> 03:00 means ``t >= 20:30 OR t < 03:00``.
    """
    if wraps_midnight:
        return t >= start or t < end
    return start <= t < end


def _make_session_dt(
    reference_ny: datetime,
    boundary_time: time,
    *,
    is_end_of_midnight_wrap: bool = False,
    is_start_of_midnight_wrap: bool = False,
) -> datetime:
    """Build a timezone-aware NY datetime for a session boundary.

    For sessions that wrap midnight (Asia Open 20:30-03:00):
      - If the reference time is *before* midnight (>= 20:30), the end at
        03:00 falls on the *next* calendar day.
      - If the reference time is *after* midnight (< 03:00), the start at
        20:30 falls on the *previous* calendar day.
    """
    base_date = reference_ny.date()

    if is_end_of_midnight_wrap and reference_ny.time() >= _ASIA_OPEN_START:
        # We are in the evening portion; end is tomorrow
        base_date = base_date + timedelta(days=1)
    elif is_start_of_midnight_wrap and reference_ny.time() < _ASIA_OPEN_END:
        # We are in the early-morning portion; start was yesterday
        base_date = base_date - timedelta(days=1)

    return datetime.combine(base_date, boundary_time, tzinfo=NY_TZ)


class MMSessionAnalyzer:
    """Analyzes Market Maker session timing for crypto markets.

    All computations use America/New_York wall-clock time via the stdlib
    ``zoneinfo`` module (Python 3.9+). DST transitions are handled
    automatically by the OS timezone database.

    Usage::

        analyzer = MMSessionAnalyzer()
        info = analyzer.get_current_session()
        print(info.session_name, info.minutes_remaining)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_session(self, dt: datetime | None = None) -> SessionInfo:
        """Return which MM session *dt* falls in.

        Args:
            dt: Timezone-aware (or naive-UTC) datetime. Defaults to now.

        Returns:
            :class:`SessionInfo` with session name, boundaries, and remaining
            time.
        """
        if dt is None:
            dt = datetime.now(NY_TZ)
        return self._resolve(dt)

    def get_session_for_candle(self, candle_time: datetime) -> SessionInfo:
        """Same as :meth:`get_current_session` but for a historical candle.

        Args:
            candle_time: Candle open time (typically UTC from exchange).
        """
        return self._resolve(candle_time)

    def is_session_changeover(self, dt: datetime | None = None) -> bool:
        """True if *dt* falls within a 30-minute gap period."""
        info = self.get_current_session(dt)
        return info.is_gap

    def get_week_boundaries(self, dt: datetime | None = None) -> tuple[datetime, datetime]:
        """Return (Sunday 5pm, Friday 5pm) bounding the trading week that
        contains *dt*.

        If *dt* is during the weekend (after Friday 5pm), boundaries refer
        to the *upcoming* week (next Sunday -> next Friday).
        """
        ny = _to_ny(dt) if dt is not None else datetime.now(NY_TZ)
        weekday = ny.weekday()  # Mon=0 ... Sun=6

        # Calculate days back to the most recent Sunday
        days_since_sunday = (weekday + 1) % 7  # Sun=0, Mon=1 ... Sat=6

        sunday = ny.date() - timedelta(days=days_since_sunday)
        week_start = datetime.combine(sunday, time(17, 0), tzinfo=NY_TZ)

        # Friday is 5 days after Sunday
        friday = sunday + timedelta(days=5)
        week_end = datetime.combine(friday, time(17, 0), tzinfo=NY_TZ)

        # If we are past Friday 5pm (weekend), shift to the *next* week
        if ny > week_end:
            sunday = sunday + timedelta(days=7)
            friday = friday + timedelta(days=7)
            week_start = datetime.combine(sunday, time(17, 0), tzinfo=NY_TZ)
            week_end = datetime.combine(friday, time(17, 0), tzinfo=NY_TZ)

        return week_start, week_end

    def get_day_boundaries(self, dt: datetime | None = None) -> tuple[datetime, datetime]:
        """Return (5pm today-or-yesterday, 5pm tomorrow-or-today) for the MM
        trading day containing *dt*.

        An MM day runs 5pm -> 5pm (next calendar day).
        """
        ny = _to_ny(dt) if dt is not None else datetime.now(NY_TZ)
        ny_time = ny.time()
        base_date = ny.date()

        if ny_time < time(17, 0):
            # Before 5pm — the trading day started at 5pm *yesterday*
            day_start = datetime.combine(
                base_date - timedelta(days=1), time(17, 0), tzinfo=NY_TZ,
            )
            day_end = datetime.combine(base_date, time(17, 0), tzinfo=NY_TZ)
        else:
            # At or after 5pm — the trading day starts now
            day_start = datetime.combine(base_date, time(17, 0), tzinfo=NY_TZ)
            day_end = datetime.combine(
                base_date + timedelta(days=1), time(17, 0), tzinfo=NY_TZ,
            )

        return day_start, day_end

    def is_weekend(self, dt: datetime | None = None) -> bool:
        """True if *dt* falls between Friday 5pm NY and Sunday 5pm NY."""
        ny = _to_ny(dt) if dt is not None else datetime.now(NY_TZ)
        weekday = ny.weekday()  # Mon=0 ... Sun=6
        ny_time = ny.time()

        # Friday after 5pm
        if weekday == 4 and ny_time >= time(17, 0):
            return True
        # All day Saturday
        if weekday == 5:
            return True
        # Sunday before 5pm
        if weekday == 6 and ny_time < time(17, 0):
            return True

        return False

    def is_dead_zone(self, dt: datetime | None = None) -> bool:
        """True if *dt* falls within the Dead Zone (5pm-8pm NY)."""
        ny = _to_ny(dt) if dt is not None else datetime.now(NY_TZ)
        return _in_range(ny.time(), _DEAD_ZONE_START, _DEAD_ZONE_END)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve(self, dt: datetime) -> SessionInfo:
        """Core session resolution logic."""
        ny = _to_ny(dt)
        t = ny.time()
        weekday = ny.weekday()

        # Weekend check first
        weekend = self.is_weekend(ny)

        for name, start, end, is_gap, wraps in _SESSIONS:
            if _in_range(t, start, end, wraps_midnight=wraps):
                sess_start = _make_session_dt(
                    ny, start,
                    is_start_of_midnight_wrap=(wraps and t < end),
                )
                sess_end = _make_session_dt(
                    ny, end,
                    is_end_of_midnight_wrap=(wraps and t >= start),
                )

                remaining_td = sess_end - ny
                minutes_remaining = max(remaining_td.total_seconds() / 60.0, 0.0)

                return SessionInfo(
                    session_name=name,
                    is_gap=is_gap,
                    session_start=sess_start,
                    session_end=sess_end,
                    minutes_remaining=round(minutes_remaining, 1),
                    day_of_week=weekday,
                    is_weekend=weekend,
                )

        # Should never reach here — sessions cover the full 24h cycle.
        # Defensive fallback: treat as dead zone.
        logger.warning(
            "mm_session_resolution_fallback",
            time=str(t),
            weekday=weekday,
        )
        day_start, day_end = self.get_day_boundaries(ny)
        return SessionInfo(
            session_name="dead_zone",
            is_gap=False,
            session_start=day_start,
            session_end=day_end,
            minutes_remaining=0.0,
            day_of_week=weekday,
            is_weekend=weekend,
        )
