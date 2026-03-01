from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from enum import Enum

from src.utils.logging import get_logger
from src.utils.time_utils import ceil_to_interval

logger = get_logger(__name__)


class TickType(Enum):
    PRIMARY = "primary"  # Full scan cycle
    MONITOR = "monitor"  # Position monitoring only


class Scheduler:
    """Tick-based scheduler aligned to candle close boundaries."""

    def __init__(self, primary_interval_minutes: int = 15, monitor_interval_seconds: int = 60) -> None:
        self.primary_interval = primary_interval_minutes
        self.monitor_interval = monitor_interval_seconds

    async def wait_for_next_tick(self) -> TickType:
        """Wait for the next tick and return its type."""
        now = datetime.now(timezone.utc)
        next_primary = ceil_to_interval(now, self.primary_interval)
        next_monitor = now.replace(microsecond=0) + timedelta(seconds=self.monitor_interval)

        if next_monitor < next_primary:
            wait_seconds = (next_monitor - now).total_seconds()
            if wait_seconds > 0:
                logger.debug("waiting_for_monitor_tick", seconds=round(wait_seconds, 1))
                await asyncio.sleep(wait_seconds)
            return TickType.MONITOR
        else:
            wait_seconds = (next_primary - now).total_seconds()
            if wait_seconds > 0:
                logger.info("waiting_for_primary_tick", seconds=round(wait_seconds, 1), next_tick=next_primary.isoformat())
                await asyncio.sleep(wait_seconds)
            return TickType.PRIMARY
