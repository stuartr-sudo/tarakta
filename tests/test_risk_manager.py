import pytest

from src.config import Settings
from src.exchange.models import SignalCandidate
from src.risk.manager import RiskManager


@pytest.fixture
def risk_manager():
    config = Settings(
        kraken_api_key="test",
        kraken_api_secret="test",
        supabase_url="https://test.supabase.co",
        supabase_key="test",
        dashboard_password_hash="test",
    )
    return RiskManager(config)


class TestPositionSizing:
    def test_normal_sizing(self, risk_manager):
        """With $100 and 2% risk = $2 max loss."""
        result = risk_manager.calculate_position_size(
            balance=100.0,
            entry_price=50.0,
            stop_loss_price=48.0,
        )
        assert result.valid
        assert result.risk_usd == pytest.approx(2.0, abs=0.01)
        assert result.quantity == pytest.approx(1.0, abs=0.01)  # $2 / $2 distance = 1 unit
        assert result.cost_usd == pytest.approx(50.0, abs=0.5)

    def test_position_capped_by_balance(self, risk_manager):
        """Position cost cannot exceed balance."""
        result = risk_manager.calculate_position_size(
            balance=10.0,
            entry_price=50.0,
            stop_loss_price=49.99,  # very tight SL = huge position
        )
        assert result.valid
        assert result.cost_usd <= 10.0

    def test_below_minimum(self, risk_manager):
        """Position too small for Kraken minimum."""
        result = risk_manager.calculate_position_size(
            balance=3.0,  # 2% of $3 = $0.06 risk
            entry_price=50.0,
            stop_loss_price=48.0,
        )
        assert not result.valid
        assert "minimum" in result.reason.lower()

    def test_zero_sl_distance(self, risk_manager):
        result = risk_manager.calculate_position_size(
            balance=100.0,
            entry_price=50.0,
            stop_loss_price=50.0,
        )
        assert not result.valid

    def test_invalid_prices(self, risk_manager):
        result = risk_manager.calculate_position_size(
            balance=100.0,
            entry_price=0,
            stop_loss_price=48.0,
        )
        assert not result.valid


class TestTradeValidation:
    def test_valid_trade(self, risk_manager):
        signal = SignalCandidate(score=70, direction="bullish", symbol="BTC/USD")
        result = risk_manager.validate_trade(
            open_position_count=1,
            open_position_symbols={"ETH/USD"},
            current_balance=95.0,
            daily_start_balance=100.0,
            daily_pnl=-5.0,
            signal=signal,
        )
        assert result.allowed

    def test_max_positions(self, risk_manager):
        signal = SignalCandidate(score=70, direction="bullish", symbol="BTC/USD")
        result = risk_manager.validate_trade(
            open_position_count=3,
            open_position_symbols={"ETH/USD", "SOL/USD", "AVAX/USD"},
            current_balance=80.0,
            daily_start_balance=100.0,
            daily_pnl=0,
            signal=signal,
        )
        assert not result.allowed
        assert "concurrent" in result.reason.lower()

    def test_daily_drawdown_exceeded(self, risk_manager):
        signal = SignalCandidate(score=70, direction="bullish", symbol="BTC/USD")
        result = risk_manager.validate_trade(
            open_position_count=0,
            open_position_symbols=set(),
            current_balance=89.0,  # 11% drawdown
            daily_start_balance=100.0,
            daily_pnl=-11.0,
            signal=signal,
        )
        assert not result.allowed
        assert "drawdown" in result.reason.lower()

    def test_duplicate_symbol(self, risk_manager):
        signal = SignalCandidate(score=70, direction="bullish", symbol="BTC/USD")
        result = risk_manager.validate_trade(
            open_position_count=1,
            open_position_symbols={"BTC/USD"},
            current_balance=98.0,
            daily_start_balance=100.0,
            daily_pnl=0,
            signal=signal,
        )
        assert not result.allowed
        assert "already" in result.reason.lower()

    def test_bearish_rejected(self, risk_manager):
        signal = SignalCandidate(score=70, direction="bearish", symbol="BTC/USD")
        result = risk_manager.validate_trade(
            open_position_count=0,
            open_position_symbols=set(),
            current_balance=100.0,
            daily_start_balance=100.0,
            daily_pnl=0,
            signal=signal,
        )
        assert not result.allowed
        assert "spot" in result.reason.lower()

    def test_low_balance(self, risk_manager):
        signal = SignalCandidate(score=70, direction="bullish", symbol="BTC/USD")
        result = risk_manager.validate_trade(
            open_position_count=0,
            open_position_symbols=set(),
            current_balance=5.0,
            daily_start_balance=100.0,
            daily_pnl=-95.0,
            signal=signal,
        )
        assert not result.allowed
