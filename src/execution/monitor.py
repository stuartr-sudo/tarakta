from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from src.exchange.models import ExitSignal, Position
from src.exchange.protocol import TradingHours
from src.exchange.trading_hours import TradingHoursManager
from src.utils.logging import get_logger

logger = get_logger(__name__)

TRAILING_STOP_PCT_FALLBACK = 0.02  # Fallback if ATR not available
TICKER_FETCH_TIMEOUT = 45  # seconds — safety net above ccxt's 30s timeout


class PositionMonitor:
    """Monitors open positions for SL/TP hits and ATR-based trailing stop updates."""

    def __init__(
        self,
        trailing_activation_rr: float = 2.0,
        trailing_atr_multiplier: float = 1.5,
        breakeven_activation_rr: float = 0.5,
        max_hold_hours: float = 4.0,
        stale_close_below_rr: float = 0.0,
    ) -> None:
        self.trailing_activation_rr = trailing_activation_rr
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.breakeven_activation_rr = breakeven_activation_rr
        self.max_hold_hours = max_hold_hours
        self.stale_close_below_rr = stale_close_below_rr
        self._trading_hours_mgr = TradingHoursManager()
        # Throttle repeated ticker errors: only log per symbol once per 5 min
        self._ticker_error_last_logged: dict[str, datetime] = {}
        self._ticker_error_throttle_seconds = 300

    async def check_positions(
        self, positions: dict[str, Position], exchange, atr_values: dict[str, float] | None = None,
        trading_hours: TradingHours | None = None,
    ) -> list[ExitSignal]:
        """Check all open positions against current prices.

        Args:
            positions: Map of symbol -> Position
            exchange: Exchange client for ticker fetches
            atr_values: Map of symbol -> current ATR value (from 1H candles)
        """
        if atr_values is None:
            atr_values = {}
        exits: list[ExitSignal] = []

        for symbol, pos in positions.items():
            try:
                ticker = await asyncio.wait_for(
                    exchange.fetch_ticker(symbol), timeout=TICKER_FETCH_TIMEOUT
                )
                current_price = float(ticker["last"])
            except asyncio.TimeoutError:
                logger.warning("ticker_fetch_timeout", symbol=symbol, timeout=TICKER_FETCH_TIMEOUT)
                continue
            except Exception as e:
                # Throttle: only log same symbol's ticker error once per 5 min
                now_err = datetime.now(timezone.utc)
                last_logged = self._ticker_error_last_logged.get(symbol)
                if last_logged is None or (now_err - last_logged).total_seconds() >= self._ticker_error_throttle_seconds:
                    logger.warning("ticker_fetch_failed", symbol=symbol, error=str(e))
                    self._ticker_error_last_logged[symbol] = now_err
                continue

            atr = atr_values.get(symbol, 0.0)
            exit_signal = self._evaluate_position(symbol, pos, current_price, atr, trading_hours)
            if exit_signal:
                exits.append(exit_signal)

        return exits

    def _evaluate_position(
        self, symbol: str, pos: Position, current_price: float, atr: float,
        trading_hours: TradingHours | None = None,
    ) -> ExitSignal | None:
        """Evaluate a single position for exit conditions."""
        if pos.direction == "long":
            return self._evaluate_long(symbol, pos, current_price, atr, trading_hours)
        else:
            return self._evaluate_short(symbol, pos, current_price, atr, trading_hours)

    def _evaluate_long(
        self, symbol: str, pos: Position, current_price: float, atr: float,
        trading_hours: TradingHours | None = None,
    ) -> ExitSignal | None:
        # Update high water mark
        if current_price > pos.high_water_mark:
            pos.high_water_mark = current_price

        # 1. Check stop loss (price drops below SL) — full exit of remaining quantity
        if current_price <= pos.stop_loss:
            # If SL was moved up from original (by trailing stop or progressive TP),
            # this is a profitable exit — label it correctly
            if pos.original_stop_loss and pos.stop_loss > pos.original_stop_loss:
                reason = "trailing_stop"
            else:
                reason = "sl_hit"
            logger.info(reason, symbol=symbol, direction="long",
                        price=current_price, sl=pos.stop_loss, original_sl=pos.original_stop_loss)
            return ExitSignal(symbol=symbol, reason=reason, price=current_price)

        # 2. Progressive TP tiers (if enabled)
        if pos.tp_tiers:
            for tier in pos.tp_tiers:
                if tier.filled or tier.price is None:
                    continue
                if current_price >= tier.price:
                    logger.info(
                        f"tp{tier.level}_hit", symbol=symbol, direction="long",
                        price=current_price, tp=tier.price, quantity=tier.quantity,
                    )
                    return ExitSignal(
                        symbol=symbol,
                        reason=f"tp{tier.level}_hit",
                        price=current_price,
                        is_partial=True,
                        partial_quantity=tier.quantity,
                        tier=tier.level,
                    )
            # All priced tiers checked; fall through to trailing stop
        elif pos.take_profit and current_price >= pos.take_profit:
            # Legacy single TP mode
            logger.info("tp_hit", symbol=symbol, direction="long",
                        price=current_price, tp=pos.take_profit)
            return ExitSignal(symbol=symbol, reason="tp_hit", price=current_price)

        # 3. Early breakeven protection — move SL to entry once 0.5R in profit
        #    (only if SL hasn't already been moved above entry by TP1 hit)
        original_sl_dist = (pos.entry_price - pos.original_stop_loss) if pos.original_stop_loss else 0
        sl_distance = original_sl_dist if original_sl_dist > 0 else (pos.entry_price - pos.stop_loss)
        if sl_distance > 0:
            unrealized_rr = (current_price - pos.entry_price) / sl_distance
            if (
                unrealized_rr >= self.breakeven_activation_rr
                and pos.stop_loss < pos.entry_price  # Not already at/above breakeven
            ):
                pos.stop_loss = pos.entry_price
                logger.info(
                    "sl_moved_to_breakeven_early",
                    symbol=symbol,
                    direction="long",
                    unrealized_rr=f"{unrealized_rr:.2f}",
                    new_sl=pos.entry_price,
                )

        # 4. Stale trade auto-close — close losing trades open too long
        #    Uses market-open hours only (overnight/weekend doesn't count for stocks/commodities)
        if self.max_hold_hours > 0 and pos.entry_time:
            now_utc = datetime.now(timezone.utc)
            hours_open = self._trading_hours_mgr.market_open_hours_between(
                pos.entry_time, now_utc, trading_hours,
            )
            if hours_open >= self.max_hold_hours:
                stale_rr = (current_price - pos.entry_price) / sl_distance if sl_distance > 0 else 0
                if stale_rr < self.stale_close_below_rr:
                    logger.info(
                        "stale_trade_closed",
                        symbol=symbol,
                        direction="long",
                        hours_open=f"{hours_open:.1f}",
                        unrealized_rr=f"{stale_rr:.2f}",
                        max_hold_hours=self.max_hold_hours,
                    )
                    return ExitSignal(symbol=symbol, reason="stale_close", price=current_price)

        # 5. ATR-based trailing stop — use original SL distance for R:R calculation
        if sl_distance > 0:
            unrealized_rr = (current_price - pos.entry_price) / sl_distance
            if unrealized_rr >= self.trailing_activation_rr:
                # ATR-dynamic trailing: high_water_mark - ATR * multiplier
                if atr > 0:
                    atr_trail = atr * self.trailing_atr_multiplier
                else:
                    # Fallback to percentage if ATR unavailable
                    atr_trail = pos.high_water_mark * TRAILING_STOP_PCT_FALLBACK

                # After TP1 but before TP2: use wider trail (1R = SL distance)
                # so the position has room to reach TP2.  After TP2: tighten.
                tier = getattr(pos, "current_tier", 0) or 0
                if tier >= 2:
                    trail_dist = atr_trail          # Tight trail for runner
                else:
                    trail_dist = max(atr_trail, sl_distance)  # Wide trail until TP2

                trailing_sl = pos.high_water_mark - trail_dist
                if trailing_sl > pos.stop_loss:
                    pos.stop_loss = trailing_sl
                if current_price <= trailing_sl:
                    return ExitSignal(symbol=symbol, reason="trailing_stop",
                                      price=current_price)

        return None

    def _evaluate_short(
        self, symbol: str, pos: Position, current_price: float, atr: float,
        trading_hours: TradingHours | None = None,
    ) -> ExitSignal | None:
        # Update low water mark (for shorts, lower is better)
        if current_price < pos.high_water_mark:
            pos.high_water_mark = current_price

        # 1. Check stop loss (price rises above SL) — full exit of remaining quantity
        if current_price >= pos.stop_loss:
            # If SL was moved down from original (by trailing stop or progressive TP),
            # this is a profitable exit — label it correctly
            if pos.original_stop_loss and pos.stop_loss < pos.original_stop_loss:
                reason = "trailing_stop"
            else:
                reason = "sl_hit"
            logger.info(reason, symbol=symbol, direction="short",
                        price=current_price, sl=pos.stop_loss, original_sl=pos.original_stop_loss)
            return ExitSignal(symbol=symbol, reason=reason, price=current_price)

        # 2. Progressive TP tiers (if enabled)
        if pos.tp_tiers:
            for tier in pos.tp_tiers:
                if tier.filled or tier.price is None:
                    continue
                if current_price <= tier.price:
                    logger.info(
                        f"tp{tier.level}_hit", symbol=symbol, direction="short",
                        price=current_price, tp=tier.price, quantity=tier.quantity,
                    )
                    return ExitSignal(
                        symbol=symbol,
                        reason=f"tp{tier.level}_hit",
                        price=current_price,
                        is_partial=True,
                        partial_quantity=tier.quantity,
                        tier=tier.level,
                    )
        elif pos.take_profit and current_price <= pos.take_profit:
            # Legacy single TP mode
            logger.info("tp_hit", symbol=symbol, direction="short",
                        price=current_price, tp=pos.take_profit)
            return ExitSignal(symbol=symbol, reason="tp_hit", price=current_price)

        # 3. Early breakeven protection — move SL to entry once 0.5R in profit
        #    (only if SL hasn't already been moved below entry by TP1 hit)
        original_sl_dist = (pos.original_stop_loss - pos.entry_price) if pos.original_stop_loss else 0
        sl_distance = original_sl_dist if original_sl_dist > 0 else (pos.stop_loss - pos.entry_price)
        if sl_distance > 0:
            unrealized_rr = (pos.entry_price - current_price) / sl_distance
            if (
                unrealized_rr >= self.breakeven_activation_rr
                and pos.stop_loss > pos.entry_price  # Not already at/below breakeven
            ):
                pos.stop_loss = pos.entry_price
                logger.info(
                    "sl_moved_to_breakeven_early",
                    symbol=symbol,
                    direction="short",
                    unrealized_rr=f"{unrealized_rr:.2f}",
                    new_sl=pos.entry_price,
                )

        # 4. Stale trade auto-close — close losing trades open too long
        #    Uses market-open hours only (overnight/weekend doesn't count for stocks/commodities)
        if self.max_hold_hours > 0 and pos.entry_time:
            now_utc = datetime.now(timezone.utc)
            hours_open = self._trading_hours_mgr.market_open_hours_between(
                pos.entry_time, now_utc, trading_hours,
            )
            if hours_open >= self.max_hold_hours:
                stale_rr = (pos.entry_price - current_price) / sl_distance if sl_distance > 0 else 0
                if stale_rr < self.stale_close_below_rr:
                    logger.info(
                        "stale_trade_closed",
                        symbol=symbol,
                        direction="short",
                        hours_open=f"{hours_open:.1f}",
                        unrealized_rr=f"{stale_rr:.2f}",
                        max_hold_hours=self.max_hold_hours,
                    )
                    return ExitSignal(symbol=symbol, reason="stale_close", price=current_price)

        # 5. ATR-based trailing stop for shorts — use original SL distance for R:R calculation
        if sl_distance > 0:
            unrealized_rr = (pos.entry_price - current_price) / sl_distance
            if unrealized_rr >= self.trailing_activation_rr:
                # ATR-dynamic trailing: low_water_mark + ATR * multiplier
                if atr > 0:
                    atr_trail = atr * self.trailing_atr_multiplier
                else:
                    atr_trail = pos.high_water_mark * TRAILING_STOP_PCT_FALLBACK

                # After TP1 but before TP2: use wider trail (1R = SL distance)
                # so the position has room to reach TP2.  After TP2: tighten.
                tier = getattr(pos, "current_tier", 0) or 0
                if tier >= 2:
                    trail_dist = atr_trail          # Tight trail for runner
                else:
                    trail_dist = max(atr_trail, sl_distance)  # Wide trail until TP2

                trailing_sl = pos.high_water_mark + trail_dist
                if trailing_sl < pos.stop_loss:
                    pos.stop_loss = trailing_sl
                if current_price >= trailing_sl:
                    return ExitSignal(symbol=symbol, reason="trailing_stop",
                                      price=current_price)

        return None
