"""Trading hours management for non-24/7 markets."""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from src.exchange.protocol import MarketInfo, TradingHours
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TradingHoursManager:
    """Manages market open/close state for non-24/7 markets."""

    def is_market_open(self, market_info: MarketInfo | None) -> bool:
        """Check if the market is currently open for trading."""
        if market_info is None or market_info.trading_hours is None:
            return True  # 24/7 (crypto)

        hours = market_info.trading_hours
        tz = ZoneInfo(hours.timezone)
        now = datetime.now(tz)

        # Check trading day (0=Monday, 6=Sunday)
        if now.weekday() not in hours.trading_days:
            return False

        current_time = now.time()
        return hours.open_time <= current_time <= hours.close_time

    def next_open(self, market_info: MarketInfo | None) -> datetime | None:
        """When does the market next open? None if 24/7 or currently open."""
        if market_info is None or market_info.trading_hours is None:
            return None

        if self.is_market_open(market_info):
            return None

        hours = market_info.trading_hours
        tz = ZoneInfo(hours.timezone)
        now = datetime.now(tz)

        # Try today first, then iterate forward
        candidate = now.replace(
            hour=hours.open_time.hour,
            minute=hours.open_time.minute,
            second=0, microsecond=0,
        )

        for day_offset in range(8):  # Max 7 days forward (handles holidays)
            check = candidate + timedelta(days=day_offset)
            if check.weekday() in hours.trading_days and check > now:
                return check.astimezone(timezone.utc)

        return None

    def next_close(self, market_info: MarketInfo | None) -> datetime | None:
        """When does the market next close? None if 24/7 or currently closed."""
        if market_info is None or market_info.trading_hours is None:
            return None

        if not self.is_market_open(market_info):
            return None

        hours = market_info.trading_hours
        tz = ZoneInfo(hours.timezone)
        now = datetime.now(tz)

        close = now.replace(
            hour=hours.close_time.hour,
            minute=hours.close_time.minute,
            second=0, microsecond=0,
        )
        return close.astimezone(timezone.utc)

    def market_open_hours_between(
        self, start_utc: datetime, end_utc: datetime, trading_hours: TradingHours | None,
    ) -> float:
        """Calculate hours the market was open between two UTC timestamps.

        For 24/7 markets (trading_hours is None), returns wall-clock hours.
        For scheduled markets, sums only time within trading sessions.
        """
        if trading_hours is None:
            return (end_utc - start_utc).total_seconds() / 3600

        tz = ZoneInfo(trading_hours.timezone)
        start_local = start_utc.astimezone(tz)
        end_local = end_utc.astimezone(tz)

        total_seconds = 0.0
        current_date = start_local.date()
        end_date = end_local.date()

        while current_date <= end_date:
            if current_date.weekday() in trading_hours.trading_days:
                session_open = datetime.combine(current_date, trading_hours.open_time, tzinfo=tz)
                session_close = datetime.combine(current_date, trading_hours.close_time, tzinfo=tz)

                overlap_start = max(start_local, session_open)
                overlap_end = min(end_local, session_close)

                if overlap_start < overlap_end:
                    total_seconds += (overlap_end - overlap_start).total_seconds()

            current_date += timedelta(days=1)

        return total_seconds / 3600

    def should_scan(self, market_info: MarketInfo | None) -> bool:
        """Should we run a scan cycle? False if market is closed."""
        return self.is_market_open(market_info)
