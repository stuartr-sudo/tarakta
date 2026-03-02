from __future__ import annotations

from datetime import datetime, timezone

from src.exchange.models import OrderResult, Position, SignalCandidate
from src.risk.manager import RiskManager
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OrderExecutor:
    """Handles entry and exit order placement with SL/TP calculation."""

    def __init__(self, exchange, risk_manager: RiskManager, min_rr: float = 2.0) -> None:
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.min_rr = min_rr

    async def execute_entry(
        self,
        signal: SignalCandidate,
        current_balance: float,
        mode: str,
    ) -> tuple[Position | None, OrderResult | None, dict | None]:
        """
        Calculate SL/TP, validate R:R, size position, place order.

        Returns (Position, OrderResult, trade_record) or (None, None, None) if skipped.
        """
        is_long = signal.direction == "bullish"

        # Calculate stop loss
        sl_price = self._calculate_stop_loss(signal)
        if sl_price is None:
            logger.info("skip_invalid_sl", symbol=signal.symbol, sl=sl_price)
            return None, None, None

        # Validate SL direction
        if is_long and sl_price >= signal.entry_price:
            logger.info("skip_sl_above_entry", symbol=signal.symbol, sl=sl_price, entry=signal.entry_price)
            return None, None, None
        if not is_long and sl_price <= signal.entry_price:
            logger.info("skip_sl_below_entry", symbol=signal.symbol, sl=sl_price, entry=signal.entry_price)
            return None, None, None

        # Calculate take profit
        tp_price = self._calculate_take_profit(signal, sl_price)

        # Validate R:R ratio
        sl_distance = abs(signal.entry_price - sl_price)
        if sl_distance <= 0:
            return None, None, None

        tp_distance = abs(tp_price - signal.entry_price) if tp_price else sl_distance * self.min_rr
        rr_ratio = tp_distance / sl_distance

        if rr_ratio < self.min_rr:
            logger.info(
                "skip_low_rr",
                symbol=signal.symbol,
                rr=f"{rr_ratio:.1f}",
                min_rr=self.min_rr,
            )
            return None, None, None

        # Calculate position size
        pos_size = self.risk_manager.calculate_position_size(
            balance=current_balance,
            entry_price=signal.entry_price,
            stop_loss_price=sl_price,
        )
        if not pos_size.valid:
            logger.info("skip_invalid_size", symbol=signal.symbol, reason=pos_size.reason)
            return None, None, None

        # Place order — long=buy, short=sell
        side = "buy" if is_long else "sell"
        try:
            result = await self.exchange.place_market_order(
                symbol=signal.symbol,
                side=side,
                quantity=pos_size.quantity,
            )
        except Exception as e:
            logger.error("order_failed", symbol=signal.symbol, error=str(e))
            return None, None, None

        # Build position
        direction = "long" if is_long else "short"
        entry_px = result.avg_price if result.avg_price > 0 else signal.entry_price
        leverage = getattr(self.exchange, "leverage", 1) or 1
        margin_used = pos_size.cost_usd / leverage if leverage > 1 else 0.0
        # Calculate liquidation price
        if leverage > 1:
            if is_long:
                liq_price = entry_px * (1 - 1 / leverage * 0.95)
            else:
                liq_price = entry_px * (1 + 1 / leverage * 0.95)
        else:
            liq_price = 0.0

        position = Position(
            trade_id="",
            symbol=signal.symbol,
            direction=direction,
            entry_price=entry_px,
            quantity=result.filled_quantity if result.filled_quantity > 0 else pos_size.quantity,
            stop_loss=sl_price,
            take_profit=tp_price,
            high_water_mark=entry_px,
            entry_time=datetime.now(timezone.utc),
            cost_usd=pos_size.cost_usd,
            leverage=leverage,
            margin_used=margin_used,
            liquidation_price=liq_price,
        )

        trade_record = {
            "symbol": signal.symbol,
            "direction": direction,
            "status": "open",
            "mode": mode,
            "entry_price": position.entry_price,
            "entry_quantity": position.quantity,
            "entry_cost_usd": position.cost_usd,
            "entry_order_id": result.order_id,
            "entry_time": position.entry_time.isoformat(),
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "risk_usd": pos_size.risk_usd,
            "risk_reward": round(rr_ratio, 2),
            "confluence_score": signal.score,
            "signal_reasons": signal.reasons,
            "timeframes_used": {"htf": "4h", "entry": "15m"},
            "fees_usd": result.fee,
            "leverage": leverage,
            "margin_used": margin_used,
            "liquidation_price": liq_price,
        }

        logger.info(
            "trade_entered",
            symbol=signal.symbol,
            direction=direction,
            entry_price=position.entry_price,
            quantity=position.quantity,
            sl=sl_price,
            tp=tp_price,
            rr=f"{rr_ratio:.1f}",
            risk_usd=pos_size.risk_usd,
            leverage=leverage,
            score=signal.score,
        )

        return position, result, trade_record

    async def execute_exit(
        self, symbol: str, position: Position, reason: str, current_price: float
    ) -> tuple[OrderResult | None, float]:
        """Execute an order to close a position. Returns (OrderResult, pnl_usd)."""
        # Long exit = sell, short exit = buy
        side = "sell" if position.direction == "long" else "buy"
        try:
            result = await self.exchange.place_market_order(
                symbol=symbol,
                side=side,
                quantity=position.quantity,
            )
        except Exception as e:
            logger.error("exit_order_failed", symbol=symbol, reason=reason, error=str(e))
            return None, 0.0

        exit_price = result.avg_price if result.avg_price > 0 else current_price

        # PnL: long = (exit - entry) * qty, short = (entry - exit) * qty
        if position.direction == "long":
            pnl = (exit_price - position.entry_price) * position.quantity - result.fee
        else:
            pnl = (position.entry_price - exit_price) * position.quantity - result.fee

        logger.info(
            "trade_exited",
            symbol=symbol,
            direction=position.direction,
            reason=reason,
            exit_price=exit_price,
            pnl=pnl,
            fee=result.fee,
        )
        return result, pnl

    def _calculate_stop_loss(self, signal: SignalCandidate) -> float | None:
        """Calculate SL based on direction and signal context."""
        entry = signal.entry_price
        is_long = signal.direction == "bullish"

        if is_long:
            return self._sl_for_long(signal, entry)
        else:
            return self._sl_for_short(signal, entry)

    def _sl_for_long(self, signal: SignalCandidate, entry: float) -> float:
        """SL below entry for long trades."""
        # Priority 1: Below order block
        if signal.ob_context and signal.ob_context.direction == "bullish":
            sl = signal.ob_context.bottom * 0.998
            if sl < entry:
                return sl

        # Priority 2: Below FVG
        if signal.fvg_context and signal.fvg_context.direction == "bullish":
            sl = signal.fvg_context.bottom * 0.998
            if sl < entry:
                return sl

        # Priority 3: Below swing lows
        for tf in ["1h_swing_low", "15m_swing_low"]:
            level = signal.key_levels.get(tf)
            if level and level < entry:
                return level * 0.998

        # Fallback: 9% below entry
        return entry * 0.91

    def _sl_for_short(self, signal: SignalCandidate, entry: float) -> float:
        """SL above entry for short trades."""
        # Priority 1: Above bearish order block top
        if signal.ob_context and signal.ob_context.direction == "bearish":
            sl = signal.ob_context.top * 1.002
            if sl > entry:
                return sl

        # Priority 2: Above FVG
        if signal.fvg_context and signal.fvg_context.direction == "bearish":
            sl = signal.fvg_context.top * 1.002
            if sl > entry:
                return sl

        # Priority 3: Above swing highs
        for tf in ["1h_swing_high", "15m_swing_high"]:
            level = signal.key_levels.get(tf)
            if level and level > entry:
                return level * 1.002

        # Fallback: 9% above entry
        return entry * 1.09

    def _calculate_take_profit(self, signal: SignalCandidate, sl_price: float) -> float | None:
        """Calculate TP based on direction."""
        entry = signal.entry_price
        sl_distance = abs(entry - sl_price)
        is_long = signal.direction == "bullish"

        if is_long:
            min_tp = entry + (sl_distance * self.min_rr)
            candidates = []
            for tf in ["1h_swing_high", "4h_swing_high", "1d_swing_high"]:
                level = signal.key_levels.get(tf)
                if level and level > entry:
                    candidates.append(level)
            if candidates:
                for tp in sorted(candidates):
                    if tp >= min_tp:
                        return tp
            return min_tp
        else:
            min_tp = entry - (sl_distance * self.min_rr)
            candidates = []
            for tf in ["1h_swing_low", "4h_swing_low", "1d_swing_low"]:
                level = signal.key_levels.get(tf)
                if level and level < entry:
                    candidates.append(level)
            if candidates:
                for tp in sorted(candidates, reverse=True):
                    if tp <= min_tp:
                        return tp
            return min_tp
