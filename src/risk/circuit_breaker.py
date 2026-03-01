from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.config import Settings
from src.exchange.models import CircuitBreakerStatus
from src.utils.logging import get_logger
from src.utils.time_utils import next_midnight_utc

logger = get_logger(__name__)


class CircuitBreaker:
    """Halts trading when drawdown exceeds safety thresholds."""

    COOLDOWN_HOURS = 24

    def __init__(self, config: Settings) -> None:
        self.daily_limit = config.max_daily_drawdown
        self.total_limit = config.circuit_breaker_pct

    def check(
        self,
        current_balance: float,
        daily_start_balance: float,
        peak_balance: float,
    ) -> CircuitBreakerStatus:
        """Check if circuit breaker should trigger."""
        # Total drawdown from peak (critical)
        if peak_balance > 0:
            total_dd = (peak_balance - current_balance) / peak_balance
        else:
            total_dd = 0

        if total_dd >= self.total_limit:
            logger.critical(
                "circuit_breaker_critical",
                total_drawdown=f"{total_dd:.1%}",
                limit=f"{self.total_limit:.0%}",
                balance=current_balance,
                peak=peak_balance,
            )
            return CircuitBreakerStatus(
                triggered=True,
                reason=f"Total drawdown {total_dd:.1%} exceeds {self.total_limit:.0%} limit. Manual restart required.",
                severity="critical",
                resume_at=None,
            )

        # Daily drawdown (warning — auto-resume next day)
        if daily_start_balance > 0:
            daily_dd = (daily_start_balance - current_balance) / daily_start_balance
        else:
            daily_dd = 0

        if daily_dd >= self.daily_limit:
            resume_at = next_midnight_utc() + timedelta(hours=self.COOLDOWN_HOURS)
            logger.warning(
                "circuit_breaker_daily",
                daily_drawdown=f"{daily_dd:.1%}",
                limit=f"{self.daily_limit:.0%}",
                resume_at=resume_at.isoformat(),
            )
            return CircuitBreakerStatus(
                triggered=True,
                reason=f"Daily drawdown {daily_dd:.1%} exceeds {self.daily_limit:.0%}. Paused until {resume_at.strftime('%Y-%m-%d %H:%M UTC')}.",
                severity="warning",
                resume_at=resume_at,
            )

        return CircuitBreakerStatus(triggered=False)
