from __future__ import annotations

from src.exchange.models import ExitSignal, Position
from src.utils.logging import get_logger

logger = get_logger(__name__)

TRAILING_STOP_ACTIVATION_RR = 1.5
TRAILING_STOP_PCT = 0.02


class PositionMonitor:
    """Monitors open positions for SL/TP hits and trailing stop updates."""

    async def check_positions(
        self, positions: dict[str, Position], exchange
    ) -> list[ExitSignal]:
        """Check all open positions against current prices."""
        exits: list[ExitSignal] = []

        for symbol, pos in positions.items():
            try:
                ticker = await exchange.fetch_ticker(symbol)
                current_price = float(ticker["last"])
            except Exception as e:
                logger.warning("ticker_fetch_failed", symbol=symbol, error=str(e))
                continue

            exit_signal = self._evaluate_position(symbol, pos, current_price)
            if exit_signal:
                exits.append(exit_signal)

        return exits

    def _evaluate_position(
        self, symbol: str, pos: Position, current_price: float
    ) -> ExitSignal | None:
        """Evaluate a single position for exit conditions."""
        if pos.direction == "long":
            return self._evaluate_long(symbol, pos, current_price)
        else:
            return self._evaluate_short(symbol, pos, current_price)

    def _evaluate_long(
        self, symbol: str, pos: Position, current_price: float
    ) -> ExitSignal | None:
        # Update high water mark
        if current_price > pos.high_water_mark:
            pos.high_water_mark = current_price

        # Check stop loss (price drops below SL)
        if current_price <= pos.stop_loss:
            logger.info("sl_hit", symbol=symbol, direction="long", price=current_price, sl=pos.stop_loss)
            return ExitSignal(symbol=symbol, reason="sl_hit", price=current_price)

        # Check take profit (price rises above TP)
        if pos.take_profit and current_price >= pos.take_profit:
            logger.info("tp_hit", symbol=symbol, direction="long", price=current_price, tp=pos.take_profit)
            return ExitSignal(symbol=symbol, reason="tp_hit", price=current_price)

        # Trailing stop
        sl_distance = pos.entry_price - pos.stop_loss
        if sl_distance > 0:
            unrealized_rr = (current_price - pos.entry_price) / sl_distance
            if unrealized_rr >= TRAILING_STOP_ACTIVATION_RR:
                trailing_sl = pos.high_water_mark * (1 - TRAILING_STOP_PCT)
                if trailing_sl > pos.stop_loss:
                    pos.stop_loss = trailing_sl
                if current_price <= trailing_sl:
                    return ExitSignal(symbol=symbol, reason="trailing_stop", price=current_price)

        return None

    def _evaluate_short(
        self, symbol: str, pos: Position, current_price: float
    ) -> ExitSignal | None:
        # Update low water mark (for shorts, lower is better)
        if current_price < pos.high_water_mark:
            pos.high_water_mark = current_price

        # Check stop loss (price rises above SL)
        if current_price >= pos.stop_loss:
            logger.info("sl_hit", symbol=symbol, direction="short", price=current_price, sl=pos.stop_loss)
            return ExitSignal(symbol=symbol, reason="sl_hit", price=current_price)

        # Check take profit (price drops below TP)
        if pos.take_profit and current_price <= pos.take_profit:
            logger.info("tp_hit", symbol=symbol, direction="short", price=current_price, tp=pos.take_profit)
            return ExitSignal(symbol=symbol, reason="tp_hit", price=current_price)

        # Trailing stop for shorts
        sl_distance = pos.stop_loss - pos.entry_price
        if sl_distance > 0:
            unrealized_rr = (pos.entry_price - current_price) / sl_distance
            if unrealized_rr >= TRAILING_STOP_ACTIVATION_RR:
                trailing_sl = pos.high_water_mark * (1 + TRAILING_STOP_PCT)
                if trailing_sl < pos.stop_loss:
                    pos.stop_loss = trailing_sl
                if current_price >= trailing_sl:
                    return ExitSignal(symbol=symbol, reason="trailing_stop", price=current_price)

        return None
