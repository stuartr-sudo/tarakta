"""Session detection and kill zone analysis for Market Maker Method.

Identifies trading sessions, calculates Asian session range, detects
Asian range sweeps, and determines if we're in a kill zone (optimal
entry window during London/NY opens).

The AMD cycle: Accumulation (Asian consolidation) → Manipulation
(London sweep) → Distribution (NY move).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Session boundaries (UTC)
ASIAN_START = time(0, 0)
ASIAN_END = time(8, 0)

LONDON_START = time(7, 0)
LONDON_END = time(16, 0)

NY_START = time(12, 0)
NY_END = time(21, 0)

# Kill zones — optimal entry windows
LONDON_KZ_START = time(7, 0)
LONDON_KZ_END = time(10, 0)

NY_KZ_START = time(12, 0)
NY_KZ_END = time(15, 0)

# Post-kill zone windows — manipulation is DONE, real move begins
POST_LONDON_KZ_START = time(10, 0)
POST_LONDON_KZ_END = time(12, 0)

POST_NY_KZ_START = time(15, 0)
POST_NY_KZ_END = time(17, 0)


@dataclass
class SessionResult:
    """Result of session analysis."""
    current_session: str        # "asian", "london", "ny", "overlap", "off_hours"
    in_kill_zone: bool
    kill_zone_name: str | None  # "london_kz", "ny_kz", or None
    in_post_kill_zone: bool = False
    post_kill_zone_name: str | None = None  # "post_london_kz", "post_ny_kz"
    asian_high: float = 0.0
    asian_low: float = 0.0
    london_high: float = 0.0
    london_low: float = 0.0
    ny_high: float = 0.0
    ny_low: float = 0.0
    asian_range_swept: str | None = None  # "above" (bearish signal) or "below" (bullish) or None
    minutes_into_session: int = 0


def _in_range(t: time, start: time, end: time) -> bool:
    """Check if time `t` falls within [start, end)."""
    if start <= end:
        return start <= t < end
    # Wraps midnight (not used currently but defensive)
    return t >= start or t < end


class SessionAnalyzer:
    """Analyzes trading sessions, kill zones, and Asian range sweeps."""

    def analyze(
        self,
        candles_1h: pd.DataFrame,
        now: datetime | None = None,
    ) -> SessionResult:
        """Full session analysis.

        Args:
            candles_1h: 1H OHLCV candles with DatetimeIndex (UTC).
            now: Current time (UTC). Defaults to now.

        Returns:
            SessionResult with session, kill zone, and Asian range data.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        current_time = now.time()

        # --- Determine current session ---
        session = self._get_session(current_time)

        # --- Kill zone check ---
        in_kz, kz_name = self._check_kill_zone(current_time)

        # --- Post-kill zone check (manipulation done, real move begins) ---
        in_post_kz, post_kz_name = self._check_post_kill_zone(current_time)

        # --- Minutes into current session ---
        mins = self._minutes_into_session(now, session)

        # --- Session ranges ---
        asian_high, asian_low = self._get_asian_range(candles_1h, now)
        london_high, london_low = self._get_session_range(
            candles_1h, now, hour_start=7, hour_end=12, session_name="london",
        )
        ny_high, ny_low = self._get_session_range(
            candles_1h, now, hour_start=12, hour_end=17, session_name="ny",
        )

        # --- Asian sweep detection ---
        current_price = 0.0
        if candles_1h is not None and not candles_1h.empty:
            current_price = float(candles_1h["close"].iloc[-1])

        sweep = self._check_asian_sweep(current_price, asian_high, asian_low)

        return SessionResult(
            current_session=session,
            in_kill_zone=in_kz,
            kill_zone_name=kz_name,
            in_post_kill_zone=in_post_kz,
            post_kill_zone_name=post_kz_name,
            asian_high=asian_high,
            asian_low=asian_low,
            london_high=london_high,
            london_low=london_low,
            ny_high=ny_high,
            ny_low=ny_low,
            asian_range_swept=sweep,
            minutes_into_session=mins,
        )

    def _get_session(self, t: time) -> str:
        """Determine which forex session is active."""
        in_london = _in_range(t, LONDON_START, LONDON_END)
        in_ny = _in_range(t, NY_START, NY_END)
        in_asian = _in_range(t, ASIAN_START, ASIAN_END)

        if in_london and in_ny:
            return "overlap"
        if in_london:
            return "london"
        if in_ny:
            return "ny"
        if in_asian:
            return "asian"
        return "off_hours"

    def _check_kill_zone(self, t: time) -> tuple[bool, str | None]:
        """Check if current time is in a kill zone."""
        if _in_range(t, LONDON_KZ_START, LONDON_KZ_END):
            return True, "london_kz"
        if _in_range(t, NY_KZ_START, NY_KZ_END):
            return True, "ny_kz"
        return False, None

    def _check_post_kill_zone(self, t: time) -> tuple[bool, str | None]:
        """Check if current time is in a post-kill-zone window.

        Post-kill zones are when the manipulation phase is complete and
        the distribution (real) move begins:
        - Post-London: 10:00-12:00 UTC
        - Post-NY: 15:00-17:00 UTC
        """
        if _in_range(t, POST_LONDON_KZ_START, POST_LONDON_KZ_END):
            return True, "post_london_kz"
        if _in_range(t, POST_NY_KZ_START, POST_NY_KZ_END):
            return True, "post_ny_kz"
        return False, None

    def _minutes_into_session(self, now: datetime, session: str) -> int:
        """Calculate minutes elapsed since session start."""
        session_starts = {
            "asian": ASIAN_START,
            "london": LONDON_START,
            "ny": NY_START,
            "overlap": NY_START,  # Overlap starts when NY opens
        }
        start = session_starts.get(session)
        if start is None:
            return 0

        start_dt = now.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
        if now < start_dt:
            # Session started yesterday (shouldn't happen with current ranges, but defensive)
            return 0
        return int((now - start_dt).total_seconds() / 60)

    def _get_asian_range(
        self,
        candles_1h: pd.DataFrame,
        now: datetime,
    ) -> tuple[float, float]:
        """Calculate the Asian session high/low from 1H candles.

        Looks at the most recent completed Asian session (00:00-08:00 UTC).
        If we're currently in the Asian session, uses yesterday's.
        """
        if candles_1h is None or candles_1h.empty:
            return 0.0, 0.0

        # Ensure we have a proper DatetimeIndex
        idx = candles_1h.index
        if not isinstance(idx, pd.DatetimeIndex):
            return 0.0, 0.0

        current_time = now.time()

        # If we're in Asian session or London hasn't started yet,
        # use yesterday's Asian session
        target_date = now.date()
        if _in_range(current_time, ASIAN_START, ASIAN_END):
            # Still in Asian session — use yesterday's completed Asian
            target_date = (now - pd.Timedelta(days=1)).date()

        # Filter candles to the target Asian session
        # Asian: 00:00-08:00 UTC on target_date
        asian_start = pd.Timestamp(
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
            hour=0, minute=0,
            tz="UTC",
        )
        asian_end = pd.Timestamp(
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
            hour=8, minute=0,
            tz="UTC",
        )

        # Ensure index is tz-aware for comparison
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
            candles_1h = candles_1h.copy()
            candles_1h.index = idx

        asian_candles = candles_1h[
            (candles_1h.index >= asian_start) & (candles_1h.index < asian_end)
        ]

        if asian_candles.empty:
            # Try previous day as fallback
            asian_start -= pd.Timedelta(days=1)
            asian_end -= pd.Timedelta(days=1)
            asian_candles = candles_1h[
                (candles_1h.index >= asian_start) & (candles_1h.index < asian_end)
            ]

        if asian_candles.empty:
            return 0.0, 0.0

        asian_high = float(asian_candles["high"].max())
        asian_low = float(asian_candles["low"].min())

        return asian_high, asian_low

    def _get_session_range(
        self,
        candles_1h: pd.DataFrame,
        now: datetime,
        hour_start: int,
        hour_end: int,
        session_name: str,
    ) -> tuple[float, float]:
        """Calculate a session's high/low from 1H candles.

        Works for London (07:00-12:00 UTC) and NY (12:00-17:00 UTC).
        If we're currently inside the session, uses previous day's completed range.
        """
        if candles_1h is None or candles_1h.empty:
            return 0.0, 0.0

        idx = candles_1h.index
        if not isinstance(idx, pd.DatetimeIndex):
            return 0.0, 0.0

        current_time = now.time()
        target_date = now.date()

        # If we're currently inside this session window, use yesterday's completed range
        session_start_time = time(hour_start, 0)
        session_end_time = time(hour_end, 0)
        if _in_range(current_time, session_start_time, session_end_time):
            target_date = (now - pd.Timedelta(days=1)).date()

        # Build timestamps for the target session window
        range_start = pd.Timestamp(
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
            hour=hour_start, minute=0,
            tz="UTC",
        )
        range_end = pd.Timestamp(
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
            hour=hour_end, minute=0,
            tz="UTC",
        )

        # Ensure index is tz-aware for comparison
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
            candles_1h = candles_1h.copy()
            candles_1h.index = idx

        session_candles = candles_1h[
            (candles_1h.index >= range_start) & (candles_1h.index < range_end)
        ]

        if session_candles.empty:
            # Try previous day as fallback
            range_start -= pd.Timedelta(days=1)
            range_end -= pd.Timedelta(days=1)
            session_candles = candles_1h[
                (candles_1h.index >= range_start) & (candles_1h.index < range_end)
            ]

        if session_candles.empty:
            return 0.0, 0.0

        session_high = float(session_candles["high"].max())
        session_low = float(session_candles["low"].min())

        return session_high, session_low

    def _check_asian_sweep(
        self,
        current_price: float,
        asian_high: float,
        asian_low: float,
    ) -> str | None:
        """Check if price has swept the Asian range.

        A sweep above the Asian high during London/NY is a bearish signal
        (market makers grabbed sell-side liquidity).

        A sweep below the Asian low is a bullish signal
        (market makers grabbed buy-side liquidity).
        """
        if asian_high <= 0 or asian_low <= 0:
            return None
        if asian_high <= asian_low:
            return None

        if current_price > asian_high:
            return "above"
        if current_price < asian_low:
            return "below"
        return None
