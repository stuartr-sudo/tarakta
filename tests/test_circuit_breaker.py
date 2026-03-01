import pytest

from src.config import Settings
from src.risk.circuit_breaker import CircuitBreaker


@pytest.fixture
def cb():
    config = Settings(
        kraken_api_key="test",
        kraken_api_secret="test",
        supabase_url="https://test.supabase.co",
        supabase_key="test",
        dashboard_password_hash="test",
    )
    return CircuitBreaker(config)


class TestCircuitBreaker:
    def test_no_trigger(self, cb):
        status = cb.check(current_balance=95.0, daily_start_balance=100.0, peak_balance=100.0)
        assert not status.triggered

    def test_daily_drawdown_trigger(self, cb):
        # 10% daily drawdown
        status = cb.check(current_balance=89.0, daily_start_balance=100.0, peak_balance=100.0)
        assert status.triggered
        assert status.severity == "warning"
        assert status.resume_at is not None

    def test_total_drawdown_trigger(self, cb):
        # 15% from peak
        status = cb.check(current_balance=84.0, daily_start_balance=100.0, peak_balance=100.0)
        assert status.triggered
        assert status.severity == "critical"
        assert status.resume_at is None  # requires manual restart

    def test_total_takes_priority(self, cb):
        # Both daily and total exceeded — total is critical
        status = cb.check(current_balance=80.0, daily_start_balance=100.0, peak_balance=100.0)
        assert status.triggered
        assert status.severity == "critical"

    def test_just_under_limit(self, cb):
        # 9.9% daily drawdown — should NOT trigger
        status = cb.check(current_balance=90.1, daily_start_balance=100.0, peak_balance=100.0)
        assert not status.triggered
