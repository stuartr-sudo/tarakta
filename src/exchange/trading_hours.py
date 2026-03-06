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

    def should_scan(self, market_info: MarketInfo | None) -> bool:
        """Should we run a scan cycle? False if market is closed."""
        return self.is_market_open(market_info)
