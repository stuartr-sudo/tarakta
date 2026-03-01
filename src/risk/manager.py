from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.config import Settings
from src.exchange.models import PositionSize, SignalCandidate, TradeValidation
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RiskManager:
    """Position sizing, trade validation, and risk controls."""

    def __init__(self, config: Settings) -> None:
        self.max_risk_pct = config.max_risk_pct
        self.max_position_pct = config.max_position_pct
        self.max_exposure_pct = config.max_exposure_pct
        self.max_concurrent = config.max_concurrent
        self.max_daily_drawdown = config.max_daily_drawdown
        self.min_rr_ratio = config.min_rr_ratio
        self.cooldown_hours = config.cooldown_hours
        self._symbol_cooldowns: dict[str, datetime] = {}

    def record_stop_out(self, symbol: str) -> None:
        """Record a stop-loss exit — symbol goes on cooldown."""
        until = datetime.now(timezone.utc) + timedelta(hours=self.cooldown_hours)
        self._symbol_cooldowns[symbol] = until
        logger.info("symbol_cooldown_set", symbol=symbol, until=until.isoformat(), hours=self.cooldown_hours)

    def _check_cooldown(self, symbol: str) -> str | None:
        """Return rejection reason if symbol is on cooldown, else None."""
        until = self._symbol_cooldowns.get(symbol)
        if until is None:
            return None
        now = datetime.now(timezone.utc)
        if now < until:
            remaining = until - now
            mins = int(remaining.total_seconds() / 60)
            return f"{symbol} on cooldown for {mins}m after stop-loss (until {until.strftime('%H:%M UTC')})"
        # Cooldown expired — clean up
        del self._symbol_cooldowns[symbol]
        return None

    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> PositionSize:
        """
        Calculate position size based on risk amount.

        With $100 balance and 2% risk = $2 max loss per trade.
        Position size = risk_amount / distance_to_SL
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return PositionSize(valid=False, reason="Invalid price")

        sl_distance = abs(entry_price - stop_loss_price)
        if sl_distance == 0:
            return PositionSize(valid=False, reason="SL distance is zero")

        risk_amount = balance * self.max_risk_pct
        quantity = risk_amount / sl_distance
        cost = quantity * entry_price

        # Account for fees + slippage (~0.36% total)
        fee_multiplier = 1.004
        total_cost = cost * fee_multiplier

        # Cap: max 5% of balance per position
        max_position_cost = balance * self.max_position_pct
        if total_cost > max_position_cost:
            quantity = max_position_cost / (entry_price * fee_multiplier)
            cost = quantity * entry_price
            total_cost = cost * fee_multiplier

        # Cannot exceed available balance
        if total_cost > balance:
            quantity = balance / (entry_price * fee_multiplier)
            cost = quantity * entry_price

        actual_risk = quantity * sl_distance

        # Check Kraken minimum order (~$5 for most pairs)
        # Use total_cost (incl. fees) since cost is the raw notional before fees
        if total_cost < 5.0:
            return PositionSize(valid=False, reason=f"Position cost ${total_cost:.2f} below Kraken minimum ~$5")

        return PositionSize(
            valid=True,
            quantity=quantity,
            cost_usd=quantity * entry_price,
            risk_usd=actual_risk,
            risk_pct=actual_risk / balance if balance > 0 else 0,
        )

    def validate_trade(
        self,
        open_position_count: int,
        open_position_symbols: set[str],
        current_balance: float,
        daily_start_balance: float,
        daily_pnl: float,
        signal: SignalCandidate,
        total_exposure_usd: float = 0.0,
    ) -> TradeValidation:
        """Pre-trade validation checks."""
        # Max total exposure (50% of equity = cash + open positions)
        equity = current_balance + total_exposure_usd
        if equity > 0:
            max_exposure = equity * self.max_exposure_pct
            if total_exposure_usd >= max_exposure:
                return TradeValidation(
                    allowed=False,
                    reason=f"Total exposure ${total_exposure_usd:.2f} exceeds {self.max_exposure_pct:.0%} of equity (${max_exposure:.2f})",
                )

        # Max concurrent positions
        if open_position_count >= self.max_concurrent:
            return TradeValidation(
                allowed=False, reason=f"Max {self.max_concurrent} concurrent positions reached"
            )

        # Daily drawdown check
        if daily_start_balance > 0:
            daily_dd = (daily_start_balance - current_balance) / daily_start_balance
            if daily_dd >= self.max_daily_drawdown:
                return TradeValidation(
                    allowed=False, reason=f"Daily drawdown {daily_dd:.1%} exceeds {self.max_daily_drawdown:.0%}"
                )

        # No duplicate symbol
        if signal.symbol in open_position_symbols:
            return TradeValidation(
                allowed=False, reason=f"Already in position for {signal.symbol}"
            )

        # Symbol cooldown after stop-loss
        cooldown_reason = self._check_cooldown(signal.symbol)
        if cooldown_reason:
            return TradeValidation(allowed=False, reason=cooldown_reason)

        # Minimum balance check
        if current_balance < 10:
            return TradeValidation(allowed=False, reason=f"Balance ${current_balance:.2f} too low")

        return TradeValidation(allowed=True)
