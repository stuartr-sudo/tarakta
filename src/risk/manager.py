from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.config import Settings
from src.exchange.models import PositionSize, SignalCandidate, TradeValidation
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RiskManager:
    """Position sizing, trade validation, and risk controls."""

    def __init__(self, config: Settings, exchange=None) -> None:
        self.max_risk_pct = config.max_risk_pct
        self.max_position_pct = config.max_position_pct
        self.max_exposure_pct = config.max_exposure_pct
        self.max_concurrent = config.max_concurrent
        self.max_daily_drawdown = config.max_daily_drawdown
        self.min_rr_ratio = config.min_rr_ratio
        self.cooldown_hours = config.cooldown_hours
        self._symbol_cooldowns: dict[str, datetime] = {}
        self._exchange_name = getattr(exchange, "exchange_name", config.exchange_name) if exchange else config.exchange_name
        self._min_order_usd = getattr(exchange, "min_order_usd", 5.0) if exchange else 5.0
        self._account_type = config.account_type
        self._leverage = config.leverage

        # Market-aware fee rate (from exchange.market_info if available)
        market_info = getattr(exchange, "market_info", None) if exchange else None
        self._supports_shorting = market_info.supports_shorting if market_info else (config.account_type != "spot")
        self._fee_rate_override = market_info.default_fee_rate if market_info else None

    def record_stop_out(self, symbol: str) -> None:
        """Record a stop-loss exit — symbol goes on cooldown."""
        until = datetime.now(timezone.utc) + timedelta(hours=self.cooldown_hours)
        self._symbol_cooldowns[symbol] = until
        logger.info("symbol_cooldown_set", symbol=symbol, until=until.isoformat(), hours=self.cooldown_hours)

    def clear_cooldown(self, symbol: str) -> None:
        """Remove stop-out cooldown for a symbol (used on reversal)."""
        self._symbol_cooldowns.pop(symbol, None)

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

        With $2000 balance: each trade allocated max 5% ($100),
        but can lose up to 10% ($200) if SL is hit.
        Position size = risk_amount / distance_to_SL, capped by max_position_pct.
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return PositionSize(valid=False, reason="Invalid price")

        sl_distance = abs(entry_price - stop_loss_price)
        if sl_distance == 0:
            return PositionSize(valid=False, reason="SL distance is zero")

        # For futures with leverage: ensure SL triggers before liquidation
        if self._leverage > 1:
            liq_distance = entry_price / self._leverage
            if sl_distance > liq_distance * 0.8:
                return PositionSize(
                    valid=False,
                    reason=f"SL distance ({sl_distance:.4f}) too wide for {self._leverage}x leverage "
                           f"(liquidation at {liq_distance:.4f}, 80% safety at {liq_distance * 0.8:.4f})",
                )

        risk_amount = balance * self.max_risk_pct
        quantity = risk_amount / sl_distance
        cost = quantity * entry_price

        # Account for fees + slippage
        if self._fee_rate_override is not None:
            # Market-aware: use actual fee rate * 2 (entry + exit) + 0.1% slippage
            fee_multiplier = 1 + (self._fee_rate_override * 2) + 0.001
        else:
            fee_multiplier = 1.002 if self._account_type == "futures" else 1.004
        total_cost = cost * fee_multiplier

        # For futures: the margin required is notional / leverage
        if self._leverage > 1:
            margin_cost = total_cost / self._leverage
        else:
            margin_cost = total_cost

        # Cap per position (percentage of balance)
        if self.max_position_pct < 1.0:
            max_margin = balance * self.max_position_pct
            if margin_cost > max_margin:
                # Scale down: margin_cost = max_margin, so total_cost = max_margin * leverage
                total_cost = max_margin * self._leverage
                quantity = total_cost / (entry_price * fee_multiplier)
                cost = quantity * entry_price
                margin_cost = total_cost / self._leverage if self._leverage > 1 else total_cost

        # Cannot exceed available balance (margin check)
        if margin_cost > balance:
            margin_cost = balance
            total_cost = margin_cost * self._leverage if self._leverage > 1 else margin_cost
            quantity = total_cost / (entry_price * fee_multiplier)
            cost = quantity * entry_price

        actual_risk = quantity * sl_distance

        # Check exchange minimum order size (notional value, not margin)
        notional = quantity * entry_price
        if notional < self._min_order_usd:
            return PositionSize(valid=False, reason=f"Position notional ${notional:.2f} below {self._exchange_name} minimum ~${self._min_order_usd:.0f}")

        return PositionSize(
            valid=True,
            quantity=quantity,
            cost_usd=notional,  # Full notional value
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
        # Cannot short on spot/data-only accounts — reject bearish signals
        if signal.direction == "bearish" and not self._supports_shorting:
            return TradeValidation(
                allowed=False,
                reason="This market/account does not support shorting. Bearish signal rejected.",
            )

        # Max total exposure (pct of equity = cash + open positions)
        # For futures: exposure is margin-based (notional / leverage)
        if self._leverage > 1:
            effective_exposure = total_exposure_usd / self._leverage
        else:
            effective_exposure = total_exposure_usd
        equity = current_balance + effective_exposure
        if equity > 0:
            max_exposure = equity * self.max_exposure_pct
            if effective_exposure >= max_exposure:
                return TradeValidation(
                    allowed=False,
                    reason=f"Total exposure ${effective_exposure:.2f} (margin) exceeds {self.max_exposure_pct:.0%} of equity (${max_exposure:.2f})",
                )

        # Max concurrent positions
        if open_position_count >= self.max_concurrent:
            return TradeValidation(
                allowed=False, reason=f"Max {self.max_concurrent} concurrent positions reached"
            )

        # Daily drawdown check — use equity (cash + positions), not just cash
        if daily_start_balance > 0:
            current_equity = current_balance + total_exposure_usd
            daily_dd = (daily_start_balance - current_equity) / daily_start_balance
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
