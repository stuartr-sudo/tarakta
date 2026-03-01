from __future__ import annotations

from datetime import datetime, timezone

from src.exchange.models import Position
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PortfolioTracker:
    """Tracks portfolio state, PnL, and balance."""

    def __init__(
        self,
        initial_balance: float,
        peak_balance: float | None = None,
        daily_start_balance: float | None = None,
    ) -> None:
        self.current_balance = initial_balance
        self.peak_balance = peak_balance or initial_balance
        self.daily_start_balance = daily_start_balance or initial_balance
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.open_positions: dict[str, Position] = {}

    def record_entry(self, position: Position) -> None:
        """Record a new position entry."""
        self.open_positions[position.symbol] = position
        self.current_balance -= position.cost_usd
        logger.info(
            "position_opened",
            symbol=position.symbol,
            entry_price=position.entry_price,
            quantity=position.quantity,
            cost=position.cost_usd,
            balance=self.current_balance,
        )

    def record_exit(self, symbol: str, exit_price: float, fee: float = 0.0) -> float:
        """Record a position exit and return PnL."""
        position = self.open_positions.pop(symbol, None)
        if not position:
            logger.warning("exit_no_position", symbol=symbol)
            return 0.0

        if position.direction == "short":
            # Short: profit when price drops. Return collateral +/- PnL.
            pnl = (position.entry_price - exit_price) * position.quantity - fee
            self.current_balance += position.cost_usd + pnl
        else:
            # Long: sold the asset, get revenue back
            revenue = position.quantity * exit_price
            pnl = revenue - position.cost_usd - fee
            self.current_balance += revenue - fee
        self.daily_pnl += pnl
        self.total_pnl += pnl

        # Update peak balance using equity (cash + remaining positions)
        equity = self.get_equity()
        if equity > self.peak_balance:
            self.peak_balance = equity

        logger.info(
            "position_closed",
            symbol=symbol,
            entry=position.entry_price,
            exit=exit_price,
            pnl=pnl,
            balance=self.current_balance,
        )
        return pnl

    def reset_daily(self) -> None:
        """Reset daily tracking at midnight UTC."""
        self.daily_start_balance = self.current_balance
        self.daily_pnl = 0.0
        logger.info("daily_reset", balance=self.current_balance)

    def get_equity(self, current_prices: dict[str, float] | None = None) -> float:
        """Calculate total equity = cash + open position costs.

        If current_prices is provided, uses mark-to-market (unrealized P&L).
        Otherwise uses entry cost as a conservative proxy.
        """
        if current_prices:
            unrealized = 0.0
            for symbol, pos in self.open_positions.items():
                if symbol in current_prices:
                    if pos.direction == "short":
                        unrealized += (pos.entry_price - current_prices[symbol]) * pos.quantity
                    else:
                        unrealized += (current_prices[symbol] - pos.entry_price) * pos.quantity
            return self.current_balance + sum(
                pos.cost_usd for pos in self.open_positions.values()
            ) + unrealized
        # No live prices — equity = cash + deployed capital (at-cost)
        return self.current_balance + sum(
            pos.cost_usd for pos in self.open_positions.values()
        )

    def get_drawdown_pct(self) -> float:
        """Drawdown from peak, measured against equity (not cash)."""
        if self.peak_balance <= 0:
            return 0.0
        equity = self.get_equity()
        return max(0.0, (self.peak_balance - equity) / self.peak_balance)

    def to_snapshot_dict(self, cycle_number: int, mode: str) -> dict:
        equity = self.get_equity()
        return {
            "balance_usd": equity,  # Show equity (cash + positions) as the headline number
            "equity_usd": equity,
            "open_positions": len(self.open_positions),
            "daily_pnl_usd": self.daily_pnl,
            "total_pnl_usd": self.total_pnl,
            "drawdown_pct": self.get_drawdown_pct(),
            "peak_balance": self.peak_balance,
            "mode": mode,
            "cycle_number": cycle_number,
        }

    def to_state_dict(self, status: str, mode: str, cycle_count: int) -> dict:
        positions_data = {}
        for sym, pos in self.open_positions.items():
            positions_data[sym] = {
                "trade_id": pos.trade_id,
                "symbol": pos.symbol,
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "high_water_mark": pos.high_water_mark,
                "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
                "cost_usd": pos.cost_usd,
                "direction": pos.direction,
            }
        equity = self.get_equity()
        # Update peak if equity has grown
        if equity > self.peak_balance:
            self.peak_balance = equity
        return {
            "status": status,
            "mode": mode,
            "open_positions": positions_data,
            "daily_pnl_usd": self.daily_pnl,
            "daily_start_bal": self.daily_start_balance,
            "peak_balance": self.peak_balance,
            "current_balance": self.current_balance,
            "cycle_count": cycle_count,
        }
