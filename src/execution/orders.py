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

import asyncio
from datetime import datetime, timezone

from src.config import Settings
from src.exchange.models import (
    OrderResult,
    Position,
    ProgressiveFillResult,
    SignalCandidate,
    TakeProfitTier,
    TrancheFill,
)
from src.risk.manager import RiskManager
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Order fill verification settings
FILL_CHECK_DELAY = 2.0      # seconds to wait before checking fill
FILL_CHECK_RETRIES = 3      # how many times to check
FILL_CHECK_INTERVAL = 3.0   # seconds between retries
CANCEL_UNFILLED = True       # cancel unfilled limit orders


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

            # ── Hard zone gate for PullbackPlan entries ──
            plan = getattr(signal, "pullback_plan", None)
            if plan is not None:
                ob_price = best_bid if is_long else best_ask
                reject_reason = ""
                if plan.is_expired:
                    reject_reason = "plan_expired"
                elif plan.invalidation_hit(ob_price):
                    reject_reason = "invalidation_hit"
                elif not plan.price_in_zone(ob_price):
                    reject_reason = "price_outside_zone"
                elif not plan.slippage_ok(ob_price):
                    reject_reason = "slippage_exceeded"

                if reject_reason:
                    logger.info(
                        "pullback_plan_entry_rejected",
                        symbol=signal.symbol,
                        reject_reason=reject_reason,
                        decision_price=round(signal.entry_price, 6),
                        live_price=round(ob_price, 6),
                        zone=plan.zone_str(),
                        invalidation=round(plan.invalidation_level, 6),
                        age_seconds=round(plan.age_seconds, 1),
                        path_taken="orders_hard_gate",
                    )
                    return None, None, None

                logger.info(
                    "pullback_plan_entry_accepted",
                    symbol=signal.symbol,
                    ob_price=round(ob_price, 6),
                    zone=plan.zone_str(),
                    limit_price=round(plan.limit_price, 6),
                    age_seconds=round(plan.age_seconds, 1),
                    zone_updates=plan.zone_updates,
                    path_taken="orders_hard_gate",
                )

            # --- Determine progressive vs single-order entry ---
            position_notional = pos_size.quantity * signal.entry_price
            num_tranches = self._calculate_tranche_count(position_notional, relevant_depth)

            if num_tranches > 1:
                # Progressive entry — split into tranches
                logger.info(
                    "progressive_entry_start",
                    symbol=signal.symbol,
                    side=side,
                    total_qty=pos_size.quantity,
                    notional=f"{position_notional:.2f}",
                    ob_depth=f"{relevant_depth:.0f}",
                    tranches=num_tranches,
                )
                prog_result = await self._progressive_fill(
                    symbol=signal.symbol,
                    side=side,
                    total_quantity=pos_size.quantity,
                    num_tranches=num_tranches,
                    is_exit=False,
                )

                if prog_result.total_filled <= 0:
                    logger.info("progressive_entry_no_fills", symbol=signal.symbol,
                                abort_reason=prog_result.abort_reason)
                    return None, None, None

                # Check if filled amount meets minimum trade size
                filled_notional = prog_result.total_filled * prog_result.vwap
                if filled_notional < min_notional:
                    logger.info("progressive_entry_below_min_after_fill",
                                symbol=signal.symbol,
                                filled_notional=f"{filled_notional:.2f}",
                                min_notional=f"{min_notional:.2f}")
                    # TODO: could close the partial fill here, but for now just skip
                    return None, None, None

                # Build a synthetic OrderResult from progressive fill
                result = OrderResult(
                    order_id=prog_result.tranches[0].order_id if prog_result.tranches else "",
                    symbol=signal.symbol,
                    side=side,
                    filled_quantity=prog_result.total_filled,
                    avg_price=prog_result.vwap,
                    fee=prog_result.total_fees,
                    status="closed",
                )
            else:
                # Single order — zone-aware pricing for PullbackPlan entries
                if plan is not None and plan.limit_price > 0 and self.config.pullback_use_limit_in_zone:
                    # Longs: buy at lower quarter of zone, but never worse than book
                    # Shorts: sell at upper quarter of zone, but never worse than book
                    if is_long:
                        limit_price = min(plan.limit_price, best_bid)
                    else:
                        limit_price = max(plan.limit_price, best_ask)
                    logger.info(
                        "pullback_plan_limit_price",
                        symbol=signal.symbol,
                        zone_limit=round(plan.limit_price, 6),
                        book_price=round(best_bid if is_long else best_ask, 6),
                        final_limit=round(limit_price, 6),
                    )
                else:
                    # Standard best-bid/ask pricing
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

                # --- Live order fill verification ---
                if hasattr(self.exchange, "fetch_order") and result.status != "closed":
                    verified = await self._verify_order_fill(
                        order_id=result.order_id,
                        symbol=signal.symbol,
                        expected_qty=pos_size.quantity,
                    )
                    if verified is None:
                        return None, None, None
                    result = verified

        except Exception as e:
            logger.error("order_failed", symbol=signal.symbol, error=str(e))
            return None, None, None

        # Build position
        direction = "long" if is_long else "short"
        entry_px = result.avg_price if result.avg_price > 0 else signal.entry_price
        leverage = getattr(self.exchange, "leverage", 1) or 1
        # Recalculate cost from actual fill (may differ from pos_size.cost_usd if partial)
        actual_cost = result.filled_quantity * entry_px if result.filled_quantity > 0 else pos_size.cost_usd
        margin_used = actual_cost / leverage if leverage > 1 else 0.0
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
            cost_usd=actual_cost,
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
            "progressive_entry": num_tranches > 1,
            "entry_tranches": num_tranches,
            "entry_vwap": entry_px if num_tranches > 1 else None,
            "entry_missed_qty": (pos_size.quantity - qty) if num_tranches > 1 else 0.0,
            "entry_fill_rate": f"{(qty / pos_size.quantity * 100):.1f}%" if num_tranches > 1 else "100%",
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
        """Execute an order to close a position, progressively if configured."""
        side = "sell" if position.direction == "long" else "buy"
        try:
            ob = await self.exchange.fetch_order_book(symbol, limit=10)
            bids = ob.get("bids") or []
            asks = ob.get("asks") or []

            if not bids or not asks:
                logger.warning("exit_no_orderbook", symbol=symbol)
                # Fall through to market order
                result = await self.exchange.place_market_order(
                    symbol=symbol, side=side, quantity=position.quantity,
                )
                exit_price = result.avg_price if result.avg_price > 0 else current_price
                if position.direction == "long":
                    pnl = (exit_price - position.entry_price) * position.quantity - result.fee
                else:
                    pnl = (position.entry_price - exit_price) * position.quantity - result.fee
                return result, pnl

            # Calculate depth for tranche decision
            if side == "sell":
                depth_usd = sum(float(l[0]) * float(l[1]) for l in bids[:5])
            else:
                depth_usd = sum(float(l[0]) * float(l[1]) for l in asks[:5])

            exit_notional = position.quantity * current_price
            num_tranches = 1
            if self.config.progressive_exit_enabled and depth_usd > 0:
                depth_ratio = exit_notional / depth_usd
                if depth_ratio >= self.config.progressive_depth_ratio:
                    raw = int(depth_ratio / self.config.progressive_depth_ratio) + 1
                    num_tranches = max(
                        self.config.progressive_min_tranches,
                        min(raw, self.config.progressive_max_tranches),
                    )

            if num_tranches > 1:
                # Progressive exit
                logger.info(
                    "progressive_exit_start",
                    symbol=symbol,
                    side=side,
                    total_qty=position.quantity,
                    notional=f"{exit_notional:.2f}",
                    ob_depth=f"{depth_usd:.0f}",
                    tranches=num_tranches,
                    reason=reason,
                )
                prog_result = await self._progressive_fill(
                    symbol=symbol,
                    side=side,
                    total_quantity=position.quantity,
                    num_tranches=num_tranches,
                    is_exit=True,  # enables market order fallback per tranche
                )

                if prog_result.total_filled <= 0:
                    # Nothing filled progressively — emergency market order for full qty
                    logger.error("progressive_exit_total_failure_market_fallback",
                                 symbol=symbol, reason=reason)
                    try:
                        result = await self.exchange.place_market_order(
                            symbol=symbol, side=side, quantity=position.quantity,
                        )
                    except Exception as e2:
                        logger.error("exit_market_fallback_failed", symbol=symbol, error=str(e2))
                        return None, 0.0
                else:
                    result = OrderResult(
                        order_id=prog_result.tranches[0].order_id if prog_result.tranches else "",
                        symbol=symbol,
                        side=side,
                        filled_quantity=prog_result.total_filled,
                        avg_price=prog_result.vwap,
                        fee=prog_result.total_fees,
                        status="closed",
                    )

                    # If partial fill on exit, market-order the remainder
                    if prog_result.total_missed > 0:
                        logger.warning(
                            "progressive_exit_partial_market_remainder",
                            symbol=symbol,
                            missed=prog_result.total_missed,
                        )
                        try:
                            remainder_result = await self.exchange.place_market_order(
                                symbol=symbol, side=side, quantity=prog_result.total_missed,
                            )
                            # Merge into result — recalculate VWAP
                            rem_qty = remainder_result.filled_quantity or prog_result.total_missed
                            rem_px = remainder_result.avg_price or current_price
                            new_total_qty = prog_result.total_filled + rem_qty
                            new_vwap = (
                                (prog_result.vwap * prog_result.total_filled + rem_px * rem_qty)
                                / new_total_qty
                            ) if new_total_qty > 0 else current_price
                            result = OrderResult(
                                order_id=result.order_id,
                                symbol=symbol,
                                side=side,
                                filled_quantity=new_total_qty,
                                avg_price=new_vwap,
                                fee=prog_result.total_fees + remainder_result.fee,
                                status="closed",
                            )
                        except Exception as e2:
                            logger.error("exit_remainder_market_failed", symbol=symbol, error=str(e2))
            else:
                # Single order exit — original behavior
                if side == "sell":
                    limit_price = float(asks[0][0])
                else:
                    limit_price = float(bids[0][0])

                result = await self.exchange.place_limit_order(
                    symbol=symbol, side=side, quantity=position.quantity, price=limit_price,
                )

                if hasattr(self.exchange, "fetch_order") and result.status != "closed":
                    verified = await self._verify_order_fill(
                        order_id=result.order_id, symbol=symbol,
                        expected_qty=position.quantity,
                    )
                    if verified is None:
                        logger.error("exit_fill_failed_trying_market",
                                     symbol=symbol, reason=reason, order_id=result.order_id)
                        try:
                            result = await self.exchange.place_market_order(
                                symbol=symbol, side=side, quantity=position.quantity,
                            )
                        except Exception as e2:
                            logger.error("exit_market_fallback_failed", symbol=symbol, error=str(e2))
                            return None, 0.0
                    else:
                        result = verified

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
            progressive=num_tranches > 1,
            tranches=num_tranches,
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
        """Execute a partial close (TP tier hit), progressively if large relative to book."""
        side = "sell" if position.direction == "long" else "buy"
        try:
            ob = await self.exchange.fetch_order_book(symbol, limit=10)
            bids = ob.get("bids") or []
            asks = ob.get("asks") or []

            if not bids or not asks:
                # No order book — market order
                result = await self.exchange.place_market_order(
                    symbol=symbol, side=side, quantity=quantity,
                )
                exit_price = result.avg_price if result.avg_price > 0 else current_price
                if position.direction == "long":
                    pnl = (exit_price - position.entry_price) * quantity - result.fee
                else:
                    pnl = (position.entry_price - exit_price) * quantity - result.fee
                return result, pnl

            # Check if progressive exit warranted for this partial
            if side == "sell":
                depth_usd = sum(float(l[0]) * float(l[1]) for l in bids[:5])
                limit_price = float(asks[0][0])
            else:
                depth_usd = sum(float(l[0]) * float(l[1]) for l in asks[:5])
                limit_price = float(bids[0][0])

            partial_notional = quantity * current_price
            num_tranches = 1
            if self.config.progressive_exit_enabled and depth_usd > 0:
                depth_ratio = partial_notional / depth_usd
                if depth_ratio >= self.config.progressive_depth_ratio:
                    raw = int(depth_ratio / self.config.progressive_depth_ratio) + 1
                    num_tranches = max(
                        self.config.progressive_min_tranches,
                        min(raw, self.config.progressive_max_tranches),
                    )

            if num_tranches > 1:
                prog_result = await self._progressive_fill(
                    symbol=symbol, side=side, total_quantity=quantity,
                    num_tranches=num_tranches, is_exit=True,
                )
                if prog_result.total_filled <= 0:
                    # Fallback to market
                    try:
                        result = await self.exchange.place_market_order(
                            symbol=symbol, side=side, quantity=quantity,
                        )
                    except Exception as e2:
                        logger.error("partial_exit_market_fallback_failed",
                                     symbol=symbol, error=str(e2))
                        return None, 0.0
                else:
                    result = OrderResult(
                        order_id=prog_result.tranches[0].order_id if prog_result.tranches else "",
                        symbol=symbol, side=side,
                        filled_quantity=prog_result.total_filled,
                        avg_price=prog_result.vwap,
                        fee=prog_result.total_fees, status="closed",
                    )
                    # Market-order any remainder
                    if prog_result.total_missed > 0:
                        try:
                            rem = await self.exchange.place_market_order(
                                symbol=symbol, side=side, quantity=prog_result.total_missed,
                            )
                            rem_qty = rem.filled_quantity or prog_result.total_missed
                            rem_px = rem.avg_price or current_price
                            new_total = prog_result.total_filled + rem_qty
                            new_vwap = (
                                (prog_result.vwap * prog_result.total_filled + rem_px * rem_qty)
                                / new_total
                            ) if new_total > 0 else current_price
                            result = OrderResult(
                                order_id=result.order_id, symbol=symbol, side=side,
                                filled_quantity=new_total, avg_price=new_vwap,
                                fee=prog_result.total_fees + rem.fee, status="closed",
                            )
                        except Exception as e2:
                            logger.error("partial_exit_remainder_failed",
                                         symbol=symbol, error=str(e2))
            else:
                # Single order — original behavior
                result = await self.exchange.place_limit_order(
                    symbol=symbol, side=side, quantity=quantity, price=limit_price,
                )
                if hasattr(self.exchange, "fetch_order") and result.status != "closed":
                    verified = await self._verify_order_fill(
                        order_id=result.order_id, symbol=symbol, expected_qty=quantity,
                    )
                    if verified is None:
                        logger.error("partial_exit_fill_failed_trying_market",
                                     symbol=symbol, tier=tier, quantity=quantity)
                        try:
                            result = await self.exchange.place_market_order(
                                symbol=symbol, side=side, quantity=quantity,
                            )
                        except Exception as e2:
                            logger.error("partial_exit_market_fallback_failed",
                                         symbol=symbol, error=str(e2))
                            return None, 0.0
                    else:
                        result = verified

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
            progressive=num_tranches > 1,
        )
        return result, pnl

    def _calculate_tranche_count(
        self, position_notional: float, ob_depth_usd: float,
    ) -> int:
        """Decide how many tranches based on position size vs order book depth.

        Returns 1 if progressive fill isn't warranted (small order relative to book).
        """
        if not self.config.progressive_entry_enabled:
            return 1
        if ob_depth_usd <= 0:
            return 1

        depth_ratio = position_notional / ob_depth_usd
        if depth_ratio < self.config.progressive_depth_ratio:
            # Position is small relative to book — no need to split
            return 1

        # Scale tranches: 10-20% of depth → 2 tranches, 20-40% → 3, etc.
        # Cap at max_tranches
        raw = int(depth_ratio / self.config.progressive_depth_ratio) + 1
        return max(
            self.config.progressive_min_tranches,
            min(raw, self.config.progressive_max_tranches),
        )

    async def _progressive_fill(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        num_tranches: int,
        is_exit: bool = False,
    ) -> ProgressiveFillResult:
        """Execute an order progressively across multiple tranches.

        For each tranche:
        1. Re-fetch order book to get fresh best price
        2. Check spread hasn't widened dangerously
        3. Place limit order for tranche quantity
        4. Verify fill (or cancel if unfilled)
        5. Wait before next tranche

        Returns ProgressiveFillResult with VWAP, fill totals, and per-tranche detail.
        """
        is_long_side = side == "buy"
        tranches: list[TrancheFill] = []
        total_filled = 0.0
        total_fees = 0.0
        weighted_price_sum = 0.0
        initial_spread = None
        abort_reason = ""

        # Split quantity into roughly equal chunks
        tranche_quantities = []
        base_qty = total_quantity / num_tranches
        for i in range(num_tranches):
            if i < num_tranches - 1:
                tranche_quantities.append(round(base_qty, 8))
            else:
                # Last tranche gets the remainder to avoid rounding dust
                placed_so_far = sum(tranche_quantities)
                tranche_quantities.append(round(total_quantity - placed_so_far, 8))

        for i, tranche_qty in enumerate(tranche_quantities):
            tranche_num = i + 1

            # Add any missed quantity from previous tranches to this one
            missed_from_prev = sum(
                t.quantity_requested - t.quantity_filled
                for t in tranches
                if t.status in ("missed", "partial")
            ) - sum(
                t.quantity_requested - t.quantity_filled
                for t in tranches[:max(0, len(tranches) - 1)]
                if t.status in ("missed", "partial")
            )
            # Actually, simpler: roll forward unfilled from last tranche only
            if tranches and tranches[-1].status in ("missed", "partial"):
                rollover = tranches[-1].quantity_requested - tranches[-1].quantity_filled
                tranche_qty = round(tranche_qty + rollover, 8)

            if tranche_qty <= 0:
                continue

            # --- Re-fetch order book for fresh pricing ---
            try:
                ob = await self.exchange.fetch_order_book(symbol, limit=10)
            except Exception as e:
                logger.warning("progressive_ob_fetch_failed", symbol=symbol,
                               tranche=tranche_num, error=str(e)[:100])
                abort_reason = f"order_book_fetch_failed_tranche_{tranche_num}"
                break

            bids = ob.get("bids") or []
            asks = ob.get("asks") or []
            if not bids or not asks:
                abort_reason = f"empty_orderbook_tranche_{tranche_num}"
                break

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
            current_spread = (best_ask - best_bid) / mid_price if mid_price > 0 else 1.0

            # Record initial spread on first tranche
            if initial_spread is None:
                initial_spread = current_spread

            # --- Abort check: spread widened dangerously ---
            if (initial_spread > 0
                    and current_spread > initial_spread * self.config.progressive_abort_spread_multiplier):
                logger.warning(
                    "progressive_abort_spread_widened",
                    symbol=symbol,
                    tranche=tranche_num,
                    initial_spread=f"{initial_spread:.5f}",
                    current_spread=f"{current_spread:.5f}",
                    multiplier=self.config.progressive_abort_spread_multiplier,
                )
                abort_reason = "spread_widened"
                break

            # --- Abort check: too little filled after half the tranches ---
            if (tranche_num > num_tranches // 2
                    and total_filled < total_quantity * self.config.progressive_min_fill_pct):
                logger.warning(
                    "progressive_abort_low_fill_rate",
                    symbol=symbol,
                    tranche=tranche_num,
                    filled=total_filled,
                    requested=total_quantity,
                    min_fill_pct=self.config.progressive_min_fill_pct,
                )
                abort_reason = "low_fill_rate"
                break

            # --- Place limit order for this tranche ---
            if is_long_side:
                limit_price = best_bid
            else:
                limit_price = best_ask

            try:
                result = await self.exchange.place_limit_order(
                    symbol=symbol,
                    side=side,
                    quantity=tranche_qty,
                    price=limit_price,
                )

                # Verify fill in live mode
                if hasattr(self.exchange, "fetch_order") and result.status != "closed":
                    verified = await self._verify_order_fill(
                        order_id=result.order_id,
                        symbol=symbol,
                        expected_qty=tranche_qty,
                    )
                    if verified is None:
                        # Tranche didn't fill
                        tranches.append(TrancheFill(
                            tranche_num=tranche_num,
                            order_id=result.order_id,
                            price=limit_price,
                            quantity_requested=tranche_qty,
                            quantity_filled=0.0,
                            fee=0.0,
                            status="missed",
                        ))
                        logger.info("progressive_tranche_missed", symbol=symbol,
                                    tranche=tranche_num, qty=tranche_qty, price=limit_price)

                        # On exit, if a tranche misses, try market order for that chunk
                        if is_exit:
                            try:
                                market_result = await self.exchange.place_market_order(
                                    symbol=symbol, side=side, quantity=tranche_qty,
                                )
                                fill_px = market_result.avg_price if market_result.avg_price > 0 else limit_price
                                fill_qty = market_result.filled_quantity if market_result.filled_quantity > 0 else tranche_qty
                                tranches[-1] = TrancheFill(
                                    tranche_num=tranche_num,
                                    order_id=market_result.order_id,
                                    price=fill_px,
                                    quantity_requested=tranche_qty,
                                    quantity_filled=fill_qty,
                                    fee=market_result.fee,
                                    status="filled",
                                )
                                total_filled += fill_qty
                                total_fees += market_result.fee
                                weighted_price_sum += fill_px * fill_qty
                                logger.info("progressive_tranche_market_fallback", symbol=symbol,
                                            tranche=tranche_num, qty=fill_qty, price=fill_px)
                            except Exception as e2:
                                logger.error("progressive_market_fallback_failed", symbol=symbol,
                                             tranche=tranche_num, error=str(e2)[:100])
                        continue
                    result = verified

                # Tranche filled
                fill_px = result.avg_price if result.avg_price > 0 else limit_price
                fill_qty = result.filled_quantity if result.filled_quantity > 0 else tranche_qty

                tranches.append(TrancheFill(
                    tranche_num=tranche_num,
                    order_id=result.order_id,
                    price=fill_px,
                    quantity_requested=tranche_qty,
                    quantity_filled=fill_qty,
                    fee=result.fee,
                    status="filled",
                ))
                total_filled += fill_qty
                total_fees += result.fee
                weighted_price_sum += fill_px * fill_qty

                logger.info(
                    "progressive_tranche_filled",
                    symbol=symbol,
                    tranche=f"{tranche_num}/{num_tranches}",
                    qty=fill_qty,
                    price=fill_px,
                    cumulative_filled=total_filled,
                    cumulative_pct=f"{(total_filled / total_quantity * 100):.1f}%",
                )

            except Exception as e:
                logger.error("progressive_tranche_order_failed", symbol=symbol,
                             tranche=tranche_num, error=str(e)[:200])
                tranches.append(TrancheFill(
                    tranche_num=tranche_num,
                    order_id="",
                    price=limit_price,
                    quantity_requested=tranche_qty,
                    quantity_filled=0.0,
                    fee=0.0,
                    status="missed",
                ))
                continue

            # --- Delay between tranches (skip after last) ---
            if tranche_num < num_tranches:
                await asyncio.sleep(self.config.progressive_tranche_delay_seconds)

        # --- Compute VWAP ---
        vwap = weighted_price_sum / total_filled if total_filled > 0 else 0.0
        total_missed = total_quantity - total_filled

        fill_result = ProgressiveFillResult(
            total_requested=total_quantity,
            total_filled=total_filled,
            total_missed=total_missed,
            vwap=vwap,
            tranches=tranches,
            total_fees=total_fees,
            fully_filled=abs(total_missed) < 1e-8,
            num_tranches_attempted=len(tranches),
            num_tranches_filled=sum(1 for t in tranches if t.status == "filled"),
            abort_reason=abort_reason,
        )

        logger.info(
            "progressive_fill_complete",
            symbol=symbol,
            side=side,
            requested=total_quantity,
            filled=total_filled,
            missed=total_missed,
            vwap=f"{vwap:.8f}" if vwap > 0 else "n/a",
            fill_rate=f"{(total_filled / total_quantity * 100):.1f}%" if total_quantity > 0 else "0%",
            tranches_filled=f"{fill_result.num_tranches_filled}/{fill_result.num_tranches_attempted}",
            abort_reason=abort_reason or "none",
            total_fees=total_fees,
        )

        return fill_result

    def _calculate_tp_tiers(
        self,
        entry_price: float,
        sl_price: float,
        quantity: float,
        direction: str,
    ) -> list[TakeProfitTier]:
        """Build 3-tier progressive TP plan: 33% at 0.70R, 33% at 0.95R, 34% at 1.5R."""
        sl_distance = abs(entry_price - sl_price)
        is_long = direction == "long"

        if is_long:
            tp1_price = entry_price + sl_distance * self.config.tp1_rr
            tp2_price = entry_price + sl_distance * self.config.tp2_rr
            tp3_price = entry_price + sl_distance * self.config.tp3_rr
        else:
            tp1_price = entry_price - sl_distance * self.config.tp1_rr
            tp2_price = entry_price - sl_distance * self.config.tp2_rr
            tp3_price = entry_price - sl_distance * self.config.tp3_rr

        tp1_qty = round(quantity * self.config.tp1_pct, 8)
        tp2_qty = round(quantity * self.config.tp2_pct, 8)
        tp3_qty = round(quantity - tp1_qty - tp2_qty, 8)

        return [
            TakeProfitTier(level=1, price=tp1_price, pct=self.config.tp1_pct, quantity=tp1_qty),
            TakeProfitTier(level=2, price=tp2_price, pct=self.config.tp2_pct, quantity=tp2_qty),
            TakeProfitTier(level=3, price=tp3_price, pct=self.config.tp3_pct, quantity=tp3_qty),
        ]

    async def _verify_order_fill(
        self, order_id: str, symbol: str, expected_qty: float
    ) -> OrderResult | None:
        """Verify a limit order filled in live trading.

        Polls the exchange for order status. If the order doesn't fill
        within the retry window, cancels it and returns None.

        Returns:
            OrderResult with actual fill data, or None if unfilled/cancelled.
        """
        await asyncio.sleep(FILL_CHECK_DELAY)

        for attempt in range(FILL_CHECK_RETRIES):
            try:
                order_data = await self.exchange.fetch_order(order_id, symbol)
                status = order_data.get("status", "unknown")
                filled = order_data.get("filled", 0)
                avg_price = order_data.get("average", 0)

                if status == "closed" and filled > 0:
                    # Fully filled
                    logger.info(
                        "order_fill_verified",
                        symbol=symbol,
                        order_id=order_id,
                        filled=filled,
                        avg_price=avg_price,
                        attempt=attempt + 1,
                    )
                    return OrderResult(
                        order_id=order_id,
                        symbol=symbol,
                        side=order_data.get("side", "buy"),
                        filled_quantity=filled,
                        avg_price=avg_price,
                        fee=0.0,  # Fees computed by exchange
                        status="closed",
                    )

                if status in ("canceled", "cancelled", "expired", "rejected"):
                    logger.warning(
                        "order_not_filled",
                        symbol=symbol,
                        order_id=order_id,
                        status=status,
                    )
                    return None

                # Still open — wait and retry
                logger.info(
                    "order_pending",
                    symbol=symbol,
                    order_id=order_id,
                    status=status,
                    filled=filled,
                    attempt=attempt + 1,
                    max_attempts=FILL_CHECK_RETRIES,
                )

                if attempt < FILL_CHECK_RETRIES - 1:
                    await asyncio.sleep(FILL_CHECK_INTERVAL)

            except Exception as e:
                logger.warning(
                    "order_check_failed",
                    symbol=symbol,
                    order_id=order_id,
                    error=str(e)[:200],
                    attempt=attempt + 1,
                )
                if attempt < FILL_CHECK_RETRIES - 1:
                    await asyncio.sleep(FILL_CHECK_INTERVAL)

        # Order still not filled after retries — cancel it
        if CANCEL_UNFILLED:
            try:
                await self.exchange.exchange.cancel_order(order_id, symbol)
                logger.warning(
                    "unfilled_order_cancelled",
                    symbol=symbol,
                    order_id=order_id,
                    reason="not_filled_within_timeout",
                )
            except Exception as e:
                logger.warning(
                    "cancel_failed",
                    symbol=symbol,
                    order_id=order_id,
                    error=str(e)[:200],
                )

        return None

    @staticmethod
    def _nudge_past_round(price: float, is_long: bool) -> float:
        """Push SL past nearby round/psychological price levels.

        Market makers hunt clustered stops sitting at round numbers like
        $60,000 or $0.50.  If our calculated SL lands within 0.4% of a
        round level, we nudge it *further away* from entry so the MM
        sweep is less likely to clip us.

        Round-level granularity scales with price magnitude:
          price >= 10000  → round to nearest 1000  (e.g. 60000, 61000)
          price >= 1000   → round to nearest 100   (e.g. 3500, 3600)
          price >= 100    → round to nearest 10     (e.g. 150, 160)
          price >= 10     → round to nearest 1      (e.g. 50, 51)
          price >= 1      → round to nearest 0.1    (e.g. 0.5, 0.6)
          price < 1       → round to nearest 0.01   (e.g. 0.05, 0.06)
        """
        if price <= 0:
            return price

        # Determine the round-number granularity for this price range
        if price >= 10_000:
            step = 1000
        elif price >= 1000:
            step = 100
        elif price >= 100:
            step = 10
        elif price >= 10:
            step = 1
        elif price >= 1:
            step = 0.1
        else:
            step = 0.01

        nearest_round = round(price / step) * step
        proximity_pct = abs(price - nearest_round) / price

        # If within 0.4% of a round level, nudge past it
        if proximity_pct < 0.004:
            nudge = step * 0.15  # Push 15% of one step beyond the round
            if is_long:
                # SL is below entry — push it further down
                price = nearest_round - nudge
            else:
                # SL is above entry — push it further up
                price = nearest_round + nudge

        return price

    def _calculate_stop_loss(self, signal: SignalCandidate) -> float | None:
        """SL behind the sweep extreme + configurable buffer (default 3%).

        The sweep wick tip is the safest SL because MMs already grabbed
        that liquidity and have no reason to revisit it. The buffer gives
        extra room for unexpected wicks.

        Falls back to ATR-based SL if no sweep data available.
        Enforces a minimum SL distance floor (default 2% of entry).
        Nudges final SL away from round psychological levels.
        """
        entry = signal.entry_price
        is_long = signal.direction == "bullish"
        sl_buffer = getattr(self.config, "sl_buffer", 0.03)
        min_sl_pct = getattr(self.config, "min_sl_pct", 0.02)
        min_distance = entry * min_sl_pct
        sweep = getattr(signal, "sweep_result", None)

        sl = None
        if sweep is not None and sweep.sweep_detected and sweep.sweep_level > 0:
            if is_long:
                # SL below the sweep low (bearish sweep wick tip)
                sl = sweep.sweep_level * (1 - sl_buffer)
                if sl >= entry:
                    sl = None  # Invalid — use ATR fallback
            else:
                # SL above the sweep high (bullish sweep wick tip)
                sl = sweep.sweep_level * (1 + sl_buffer)
                if sl <= entry:
                    sl = None  # Invalid — use ATR fallback

        # Enforce minimum SL distance
        if sl is not None:
            actual_distance = abs(entry - sl)
            if actual_distance < min_distance:
                sl = entry - min_distance if is_long else entry + min_distance

            # Enforce maximum SL distance — skip trades where SL is too far
            max_sl_pct = getattr(self.config, "max_sl_pct", 0.0)
            if max_sl_pct > 0:
                max_distance = entry * max_sl_pct
                actual_distance = abs(entry - sl)
                if actual_distance > max_distance:
                    logger.info(
                        "skip_sl_too_wide",
                        symbol=signal.symbol,
                        sl_distance_pct=f"{(actual_distance / entry):.2%}",
                        max_sl_pct=f"{max_sl_pct:.2%}",
                    )
                    return None

            return self._nudge_past_round(sl, is_long)

        # No sweep data or invalid — ATR fallback
        sl = self._atr_fallback_sl(signal, entry, is_long, min_distance)

        # Enforce maximum SL distance on ATR fallback too
        max_sl_pct = getattr(self.config, "max_sl_pct", 0.0)
        if max_sl_pct > 0 and sl is not None:
            max_distance = entry * max_sl_pct
            actual_distance = abs(entry - sl)
            if actual_distance > max_distance:
                logger.info(
                    "skip_sl_too_wide",
                    symbol=signal.symbol,
                    sl_distance_pct=f"{(actual_distance / entry):.2%}",
                    max_sl_pct=f"{max_sl_pct:.2%}",
                )
                return None

        return self._nudge_past_round(sl, is_long) if sl is not None else None

    def _atr_fallback_sl(
        self, signal: SignalCandidate, entry: float, is_long: bool,
        min_distance: float = 0.0,
    ) -> float:
        """ATR-based SL fallback when sweep data is unavailable."""
        atr = getattr(signal, "atr_1h", 0.0) or 0.0
        if atr > 0:
            distance = max(atr * 2.5, min_distance)
        else:
            distance = max(entry * 0.04, min_distance)  # 4% fallback

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
