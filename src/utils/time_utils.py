from __future__ import annotations

from datetime import datetime, timedelta, timezone


def ceil_to_interval(dt: datetime, interval_minutes: int) -> datetime:
    """Round a datetime up to the next interval boundary."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    minutes = dt.minute
    remainder = minutes % interval_minutes
    if remainder == 0 and dt.second == 0 and dt.microsecond == 0:
        return dt
    next_boundary = dt.replace(second=0, microsecond=0) + timedelta(
        minutes=interval_minutes - remainder
    )
    return next_boundary


def next_midnight_utc() -> datetime:
    """Return the next midnight UTC."""
    now = datetime.now(timezone.utc)
    tomorrow = now.date() + timedelta(days=1)
    return datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=timezone.utc)


def is_new_day(last_time: datetime | None) -> bool:
    """Check if we've crossed midnight UTC since last_time."""
    if last_time is None:
        return True
    now = datetime.now(timezone.utc)
    if last_time.tzinfo is None:
        last_time = last_time.replace(tzinfo=timezone.utc)
    return now.date() > last_time.date()
