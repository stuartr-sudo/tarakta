"""Order execution with Trade Travel Chill SL/TP logic.

SL: Behind the sweep extreme (wick tip) + 0.5% buffer. This is the
safest SL because MMs already grabbed that liquidity and have no
reason to revisit it.

TP: Progressive 3-tier take-profit:
  TP1 (1R) → close 33%, move SL to breakeven
  TP2 (2R) → close 33%, move SL to TP1 price
  TP3      → remaining 34% via trailing stop
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.config import Settings
from src.exchange.models import OrderResult, Position, SignalCandidate, TakeProfitTier
from src.risk.manager import RiskManager
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OrderExecutor:
    """Handles entry and exit order placement with post-sweep SL/TP."""

    def __init__(self, exchange, risk_manager: RiskManager, config: Settings) -> None:
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.min_rr = config.min_rr_ratio
        self.config = config

    async def execute_entry(
        self,
        signal: SignalCandidate,
        current_balance: float,
        mode: str,
        sl_override: float | None = None,
        tp_override: float | None = None,
    ) -> tuple[Position | None, OrderResult | None, dict | None]:
        """
        Calculate SL/TP, validate R:R, size position, place order.

        Returns (Position, OrderResult, trade_record) or (None, None, None) if skipped.
        """
        is_long = signal.direction == "bullish"

        # Calculate stop loss (use override if provided)
        if sl_override is not None:
            sl_price = sl_override
            logger.info("using_llm_sl_override", symbol=signal.symbol, sl=sl_override)
        else:
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

        # Calculate take profit (use override if provided)
        if tp_override is not None:
            tp_price = tp_override
            logger.info("using_llm_tp_override", symbol=signal.symbol, tp=tp_override)
        else:
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

        # Minimum trade size: $min_trade_usd × leverage
        leverage = getattr(self.exchange, "leverage", 1) or 1
        min_notional = self.config.min_trade_usd * leverage
        if pos_size.cost_usd < min_notional:
            logger.info("skip_below_min_trade", symbol=signal.symbol,
                        cost_usd=f"{pos_size.cost_usd:.2f}", min_notional=f"{min_notional:.2f}",
                        min_trade_usd=self.config.min_trade_usd, leverage=leverage)
            return None, None, None

        # Place order — long=buy, short=sell
        side = "buy" if is_long else "sell"
        try:
            ob = await self.exchange.fetch_order_book(signal.symbol, limit=10)

            # --- Liquidity gate: check spread and depth before entering ---
            bids = ob.get("bids") or []
            asks = ob.get("asks") or []
            if not bids or not asks:
                logger.info("skip_no_orderbook", symbol=signal.symbol)
                return None, None, None

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
            spread_pct = (best_ask - best_bid) / mid_price if mid_price > 0 else 1.0

            if spread_pct > self.config.max_spread_pct:
                logger.info(
                    "skip_wide_spread",
                    symbol=signal.symbol,
                    spread_pct=f"{spread_pct:.4f}",
                    max_spread=f"{self.config.max_spread_pct:.4f}",
                    best_bid=best_bid,
                    best_ask=best_ask,
                )
                return None, None, None

            # Check depth across top 5 levels (not just best level)
            bid_depth_usd = sum(
                float(level[0]) * float(level[1]) for level in bids[:5]
            )
            ask_depth_usd = sum(
                float(level[0]) * float(level[1]) for level in asks[:5]
            )
            relevant_depth = bid_depth_usd if is_long else ask_depth_usd

            if relevant_depth < self.config.min_ob_depth_usd:
                logger.info(
                    "skip_thin_orderbook",
                    symbol=signal.symbol,
                    depth_usd=f"{relevant_depth:.2f}",
                    min_depth=self.config.min_ob_depth_usd,
                    side=side,
                )
                return None, None, None

            # --- 24h volume re-check at entry time ---
            try:
                ticker = await self.exchange.fetch_ticker(signal.symbol)
                quote_vol_24h = float(ticker.get("quoteVolume", 0) or 0)
                if quote_vol_24h < self.config.min_volume_usd:
                    logger.info(
                        "skip_low_volume_at_entry",
                        symbol=signal.symbol,
                        volume_24h=f"{quote_vol_24h:,.0f}",
                        min_volume=f"{self.config.min_volume_usd:,.0f}",
                    )
                    return None, None, None

                # --- Position size vs daily volume check ---
                notional = pos_size.quantity * signal.entry_price
                vol_pct = notional / quote_vol_24h if quote_vol_24h > 0 else 1.0
                if vol_pct > self.config.max_position_volume_pct:
                    logger.info(
                        "skip_position_too_large_for_volume",
                        symbol=signal.symbol,
                        notional=f"{notional:,.2f}",
                        volume_24h=f"{quote_vol_24h:,.0f}",
                        vol_pct=f"{vol_pct:.6f}",
                        max_pct=f"{self.config.max_position_volume_pct:.4f}",
                    )
                    return None, None, None
            except Exception as e:
                logger.warning("volume_recheck_failed", symbol=signal.symbol, error=str(e)[:100])
                # Conservative: skip entry if we can't verify volume
                return None, None, None

            logger.info(
                "liquidity_check_passed",
                symbol=signal.symbol,
                spread_pct=f"{spread_pct:.4f}",
                bid_depth_usd=f"{bid_depth_usd:.0f}",
                ask_depth_usd=f"{ask_depth_usd:.0f}",
                volume_24h=f"{quote_vol_24h:,.0f}",
            )

            if is_long:
                limit_price = best_bid
            else:
                limit_price = best_ask

            result = await self.exchange.place_limit_order(
                symbol=signal.symbol,
                side=side,
                quantity=pos_size.quantity,
                price=limit_price,
            )
        except Exception as e:
            logger.error("order_failed", symbol=signal.symbol, error=str(e))
            return None, None, None

        # Build position
        direction = "long" if is_long else "short"
        entry_px = result.avg_price if result.avg_price > 0 else signal.entry_price
        leverage = getattr(self.exchange, "leverage", 1) or 1
        margin_used = pos_size.cost_usd / leverage if leverage > 1 else 0.0
        if leverage > 1:
            if is_long:
                liq_price = entry_px * (1 - 1 / leverage * 0.95)
            else:
                liq_price = entry_px * (1 + 1 / leverage * 0.95)
        else:
            liq_price = 0.0

        qty = result.filled_quantity if result.filled_quantity > 0 else pos_size.quantity

        position = Position(
            trade_id="",
            symbol=signal.symbol,
            direction=direction,
            entry_price=entry_px,
            quantity=qty,
            stop_loss=sl_price,
            take_profit=tp_price,
            high_water_mark=entry_px,
            entry_time=datetime.now(timezone.utc),
            cost_usd=pos_size.cost_usd,
            leverage=leverage,
            margin_used=margin_used,
            liquidation_price=liq_price,
            original_quantity=qty,
            original_stop_loss=sl_price,
        )

        # Progressive TP tiers (disabled by default in Trade Travel Chill)
        tp_tiers_data = None
        if self.config.tp_tiers_enabled:
            position.tp_tiers = self._calculate_tp_tiers(
                entry_price=entry_px,
                sl_price=sl_price,
                quantity=qty,
                direction=direction,
            )
            position.take_profit = position.tp_tiers[0].price
            tp_tiers_data = [
                {"level": t.level, "price": t.price, "pct": t.pct, "quantity": t.quantity}
                for t in position.tp_tiers
            ]

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
            "take_profit": position.take_profit,
            "risk_usd": pos_size.risk_usd,
            "risk_reward": round(rr_ratio, 2),
            "confluence_score": signal.score,
            "signal_reasons": signal.reasons,
            "timeframes_used": {"htf": "4h", "entry": "1h"},
            "fees_usd": result.fee,
            "leverage": leverage,
            "margin_used": margin_used,
            "liquidation_price": liq_price,
            "tp_tiers": tp_tiers_data,
            "original_quantity": qty,
            "remaining_quantity": qty,
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
        """Execute an order to close a position using limit order."""
        side = "sell" if position.direction == "long" else "buy"
        try:
            ob = await self.exchange.fetch_order_book(symbol, limit=5)
            if side == "sell":
                limit_price = float(ob["asks"][0][0]) if ob.get("asks") else current_price
            else:
                limit_price = float(ob["bids"][0][0]) if ob.get("bids") else current_price
            result = await self.exchange.place_limit_order(
                symbol=symbol,
                side=side,
                quantity=position.quantity,
                price=limit_price,
            )
        except Exception as e:
            logger.error("exit_order_failed", symbol=symbol, reason=reason, error=str(e))
            return None, 0.0

        exit_price = result.avg_price if result.avg_price > 0 else current_price

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

    async def execute_partial_exit(
        self,
        symbol: str,
        position: Position,
        reason: str,
        current_price: float,
        quantity: float,
        tier: int,
    ) -> tuple[OrderResult | None, float]:
        """Execute a partial close (TP tier hit). Legacy — kept for backward compat."""
        side = "sell" if position.direction == "long" else "buy"
        try:
            ob = await self.exchange.fetch_order_book(symbol, limit=5)
            if side == "sell":
                limit_price = float(ob["asks"][0][0]) if ob.get("asks") else current_price
            else:
                limit_price = float(ob["bids"][0][0]) if ob.get("bids") else current_price
            result = await self.exchange.place_limit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=limit_price,
            )
        except Exception as e:
            logger.error("partial_exit_failed", symbol=symbol, reason=reason,
                         tier=tier, quantity=quantity, error=str(e))
            return None, 0.0

        exit_price = result.avg_price if result.avg_price > 0 else current_price

        if position.direction == "long":
            pnl = (exit_price - position.entry_price) * quantity - result.fee
        else:
            pnl = (position.entry_price - exit_price) * quantity - result.fee

        logger.info(
            "partial_exit_executed",
            symbol=symbol,
            direction=position.direction,
            reason=reason,
            tier=tier,
            exit_price=exit_price,
            quantity=quantity,
            remaining=position.quantity - quantity,
            pnl=round(pnl, 4),
            fee=result.fee,
        )
        return result, pnl

    def _calculate_tp_tiers(
        self,
        entry_price: float,
        sl_price: float,
        quantity: float,
        direction: str,
    ) -> list[TakeProfitTier]:
        """Build 3-tier progressive TP plan: 33% at 1R, 33% at 2R, 34% trailing."""
        sl_distance = abs(entry_price - sl_price)
        is_long = direction == "long"

        if is_long:
            tp1_price = entry_price + sl_distance * self.config.tp1_rr
            tp2_price = entry_price + sl_distance * self.config.tp2_rr
        else:
            tp1_price = entry_price - sl_distance * self.config.tp1_rr
            tp2_price = entry_price - sl_distance * self.config.tp2_rr

        tp1_qty = round(quantity * self.config.tp1_pct, 8)
        tp2_qty = round(quantity * self.config.tp2_pct, 8)
        tp3_qty = round(quantity - tp1_qty - tp2_qty, 8)

        return [
            TakeProfitTier(level=1, price=tp1_price, pct=self.config.tp1_pct, quantity=tp1_qty),
            TakeProfitTier(level=2, price=tp2_price, pct=self.config.tp2_pct, quantity=tp2_qty),
            TakeProfitTier(level=3, price=None, pct=self.config.tp3_pct, quantity=tp3_qty),
        ]

    def _calculate_stop_loss(self, signal: SignalCandidate) -> float | None:
        """SL behind the sweep extreme + 0.5% buffer.

        The sweep wick tip is the safest SL because MMs already grabbed
        that liquidity and have no reason to revisit it.

        Falls back to ATR-based SL if no sweep data available.
        """
        entry = signal.entry_price
        is_long = signal.direction == "bullish"
        sweep = getattr(signal, "sweep_result", None)

        if sweep is not None and sweep.sweep_detected and sweep.sweep_level > 0:
            if is_long:
                # SL below the sweep low (bearish sweep wick tip)
                sl = sweep.sweep_level * 0.995  # 0.5% buffer below
                if sl >= entry:
                    # Sweep level too close to or above entry — use ATR fallback
                    return self._atr_fallback_sl(signal, entry, is_long)
                return sl
            else:
                # SL above the sweep high (bullish sweep wick tip)
                sl = sweep.sweep_level * 1.005  # 0.5% buffer above
                if sl <= entry:
                    return self._atr_fallback_sl(signal, entry, is_long)
                return sl

        # No sweep data — ATR fallback
        return self._atr_fallback_sl(signal, entry, is_long)

    def _atr_fallback_sl(self, signal: SignalCandidate, entry: float, is_long: bool) -> float:
        """ATR-based SL fallback when sweep data is unavailable."""
        atr = getattr(signal, "atr_1h", 0.0) or 0.0
        if atr > 0:
            distance = atr * 2.0
        else:
            distance = entry * 0.03  # 3% fallback

        if is_long:
            return entry - distance
        else:
            return entry + distance

    def _calculate_take_profit(self, signal: SignalCandidate, sl_price: float) -> float | None:
        """TP at the opposite liquidity pool with minimum R:R.

        For longs (after bearish sweep): TP at swing high / Asian high
        For shorts (after bullish sweep): TP at swing low / Asian low
        """
        entry = signal.entry_price
        sl_distance = abs(entry - sl_price)
        is_long = signal.direction == "bullish"
        sweep = getattr(signal, "sweep_result", None)

        min_tp_distance = sl_distance * self.min_rr

        if is_long:
            min_tp = entry + min_tp_distance

            # Use sweep target_level (opposite side liquidity)
            if sweep and sweep.target_level and sweep.target_level > entry:
                tp = sweep.target_level
                if tp >= min_tp:
                    return tp

            # Fallback to structural levels
            for tf in ["1h_swing_high", "4h_swing_high", "1d_swing_high"]:
                level = signal.key_levels.get(tf)
                if level and level >= min_tp:
                    return level

            return min_tp
        else:
            min_tp = entry - min_tp_distance

            if sweep and sweep.target_level and sweep.target_level < entry:
                tp = sweep.target_level
                if tp <= min_tp:
                    return tp

            for tf in ["1h_swing_low", "4h_swing_low", "1d_swing_low"]:
                level = signal.key_levels.get(tf)
                if level and level <= min_tp:
                    return level

            return min_tp
