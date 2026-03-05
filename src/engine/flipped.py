"""Flipped shadow trader — inverts every signal direction.

Runs alongside the main bot as a paper simulation. Same signal detection
(sweep + displacement + pullback), but every trade direction is flipped:
bullish → short, bearish → long.

Wider SL, higher leverage. No exchange orders — purely simulated using
ticker prices. Trades stored in DB with mode='flipped_paper'.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.config import Settings
from src.data.repository import Repository
from src.exchange.models import Position, SignalCandidate
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Simulated fee rate (taker fee on Binance futures)
SIM_FEE_RATE = 0.0004


class FlippedTrader:
    """Shadow bot that flips every signal and paper-trades it."""

    def __init__(self, config: Settings, repo: Repository) -> None:
        self.enabled = config.flipped_enabled
        self.leverage = config.flipped_leverage
        self.sl_buffer = config.flipped_sl_buffer
        self.min_rr = config.min_rr_ratio
        self.max_risk_pct = config.max_risk_pct
        self.max_position_pct = config.max_position_pct
        self.trailing_activation_rr = config.trailing_activation_rr
        self.trailing_atr_multiplier = config.trailing_atr_multiplier
        self.repo = repo

        # Separate paper balance — independent of main bot
        self.balance = config.flipped_initial_balance
        self.peak_balance = config.flipped_initial_balance
        self.daily_start_balance = config.flipped_initial_balance
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.positions: dict[str, Position] = {}
        self.daily_trade_count: int = 0

    async def process_signals(self, signals: list[SignalCandidate]) -> int:
        """Process signals with flipped direction. Returns count of entries."""
        if not self.enabled:
            return 0

        entered = 0
        for signal in signals:
            if signal.symbol in self.positions:
                continue
            try:
                if await self._try_enter(signal):
                    entered += 1
            except Exception as e:
                logger.warning("flipped_entry_error", symbol=signal.symbol, error=str(e))
        return entered

    async def _try_enter(self, signal: SignalCandidate) -> bool:
        """Flip direction and simulate entry."""
        # FLIP: bullish → short, bearish → long
        is_long = signal.direction == "bearish"
        direction = "long" if is_long else "short"

        # Calculate flipped SL
        sl_price = self._calculate_sl(signal, is_long)
        if sl_price is None:
            return False

        # Validate SL direction
        if is_long and sl_price >= signal.entry_price:
            return False
        if not is_long and sl_price <= signal.entry_price:
            return False

        # Calculate flipped TP
        tp_price = self._calculate_tp(signal, sl_price, is_long)

        # Validate R:R
        sl_distance = abs(signal.entry_price - sl_price)
        if sl_distance <= 0:
            return False
        tp_distance = abs(tp_price - signal.entry_price) if tp_price else sl_distance * self.min_rr
        rr_ratio = tp_distance / sl_distance
        if rr_ratio < self.min_rr:
            return False

        # Ensure SL within leverage safety (don't get liquidated before SL)
        liq_distance = signal.entry_price / self.leverage
        if sl_distance > liq_distance * 0.8:
            return False

        # Position sizing
        risk_amount = self.balance * self.max_risk_pct
        quantity = risk_amount / sl_distance
        cost_usd = quantity * signal.entry_price
        margin_used = cost_usd / self.leverage

        # Cap by max_position_pct
        max_margin = self.balance * self.max_position_pct
        if margin_used > max_margin:
            margin_used = max_margin
            cost_usd = margin_used * self.leverage
            quantity = cost_usd / signal.entry_price

        # Can't exceed available balance
        if margin_used > self.balance:
            margin_used = self.balance
            cost_usd = margin_used * self.leverage
            quantity = cost_usd / signal.entry_price

        if cost_usd < 5.0:
            return False  # Below exchange minimum

        # Liquidation price
        if is_long:
            liq_price = signal.entry_price * (1 - 1 / self.leverage * 0.95)
        else:
            liq_price = signal.entry_price * (1 + 1 / self.leverage * 0.95)

        # Build position
        position = Position(
            trade_id="",
            symbol=signal.symbol,
            direction=direction,
            entry_price=signal.entry_price,
            quantity=quantity,
            stop_loss=sl_price,
            take_profit=tp_price,
            high_water_mark=signal.entry_price,
            entry_time=datetime.now(timezone.utc),
            cost_usd=cost_usd,
            leverage=self.leverage,
            margin_used=margin_used,
            liquidation_price=liq_price,
            original_quantity=quantity,
            original_stop_loss=sl_price,
            confluence_score=signal.score,
        )

        # Entry fee
        entry_fee = cost_usd * SIM_FEE_RATE

        # Save to DB with mode="flipped_paper"
        trade_record = {
            "symbol": signal.symbol,
            "direction": direction,
            "status": "open",
            "mode": "flipped_paper",
            "entry_price": signal.entry_price,
            "entry_quantity": quantity,
            "entry_cost_usd": cost_usd,
            "entry_order_id": "flipped_sim",
            "entry_time": position.entry_time.isoformat(),
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "risk_usd": quantity * sl_distance,
            "risk_reward": round(rr_ratio, 2),
            "confluence_score": signal.score,
            "signal_reasons": [f"FLIPPED:{signal.direction}→{direction}"] + signal.reasons,
            "timeframes_used": {"htf": "4h", "entry": "1h"},
            "fees_usd": entry_fee,
            "leverage": self.leverage,
            "margin_used": margin_used,
            "liquidation_price": liq_price,
            "original_quantity": quantity,
            "remaining_quantity": quantity,
        }

        db_trade = await self.repo.insert_trade(trade_record)
        if db_trade:
            position.trade_id = db_trade.get("id", "")

        self.positions[signal.symbol] = position
        self.balance -= margin_used
        self.daily_trade_count += 1

        logger.info(
            "flipped_entry",
            symbol=signal.symbol,
            original_direction=signal.direction,
            flipped_direction=direction,
            entry=signal.entry_price,
            sl=sl_price,
            tp=tp_price,
            rr=f"{rr_ratio:.1f}",
            leverage=self.leverage,
            margin=round(margin_used, 2),
            balance=round(self.balance, 2),
        )
        return True

    async def monitor_positions(self, exchange, atr_values: dict[str, float] | None = None) -> None:
        """Check flipped positions for SL/TP/trailing stop using live ticker data."""
        if not self.enabled or not self.positions:
            return

        atr_values = atr_values or {}

        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            try:
                ticker = await exchange.fetch_ticker(symbol)
                current_price = float(ticker["last"])
            except Exception:
                continue

            # Update high water mark
            if pos.direction == "long":
                if current_price > pos.high_water_mark:
                    pos.high_water_mark = current_price
            else:
                if current_price < pos.high_water_mark:
                    pos.high_water_mark = current_price

            exit_reason = None

            # SL check
            if pos.direction == "long" and current_price <= pos.stop_loss:
                exit_reason = "sl_hit"
            elif pos.direction == "short" and current_price >= pos.stop_loss:
                exit_reason = "sl_hit"

            # TP check
            if not exit_reason and pos.take_profit:
                if pos.direction == "long" and current_price >= pos.take_profit:
                    exit_reason = "tp_hit"
                elif pos.direction == "short" and current_price <= pos.take_profit:
                    exit_reason = "tp_hit"

            # Trailing stop (same logic as main bot)
            if not exit_reason:
                sl_dist = abs(pos.entry_price - pos.original_stop_loss)
                if sl_dist > 0:
                    if pos.direction == "long":
                        r_multiple = (pos.high_water_mark - pos.entry_price) / sl_dist
                    else:
                        r_multiple = (pos.entry_price - pos.high_water_mark) / sl_dist

                    if r_multiple >= self.trailing_activation_rr:
                        atr = atr_values.get(symbol, 0)
                        if atr > 0:
                            trail_dist = atr * self.trailing_atr_multiplier
                        else:
                            trail_dist = sl_dist * 0.75  # fallback

                        if pos.direction == "long":
                            trail_sl = pos.high_water_mark - trail_dist
                            if trail_sl > pos.stop_loss:
                                pos.stop_loss = trail_sl
                            if current_price <= pos.stop_loss:
                                exit_reason = "trailing_stop"
                        else:
                            trail_sl = pos.high_water_mark + trail_dist
                            if trail_sl < pos.stop_loss:
                                pos.stop_loss = trail_sl
                            if current_price >= pos.stop_loss:
                                exit_reason = "trailing_stop"

            if exit_reason:
                await self._close_position(symbol, current_price, exit_reason)

    async def _close_position(self, symbol: str, exit_price: float, reason: str) -> None:
        """Simulate closing a flipped position."""
        pos = self.positions.pop(symbol, None)
        if not pos:
            return

        # Calculate PnL
        if pos.direction == "long":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        # Simulate exit fee
        exit_fee = pos.cost_usd * SIM_FEE_RATE
        pnl -= exit_fee

        # Return margin + pnl to balance
        margin = pos.margin_used
        self.balance += margin + pnl
        self.daily_pnl += pnl
        self.total_pnl += pnl

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        # Close in DB
        pnl_pct = (pnl / pos.cost_usd * 100) if pos.cost_usd > 0 else 0
        total_fees = pos.cost_usd * SIM_FEE_RATE + exit_fee
        await self.repo.close_trade(
            trade_id=pos.trade_id,
            exit_price=exit_price,
            exit_quantity=pos.quantity,
            exit_order_id="flipped_sim",
            exit_reason=reason,
            pnl_usd=round(pnl, 4),
            pnl_percent=round(pnl_pct, 2),
            fees_usd=round(total_fees, 4),
        )

        logger.info(
            "flipped_exit",
            symbol=symbol,
            direction=pos.direction,
            reason=reason,
            entry=pos.entry_price,
            exit=exit_price,
            pnl=round(pnl, 2),
            balance=round(self.balance, 2),
        )

    def _calculate_sl(self, signal: SignalCandidate, is_long: bool) -> float | None:
        """Calculate SL for flipped trade with wider buffer.

        For flipped LONG (original bearish → swept highs):
          SL below the target_level (swing low, original TP destination) with wider buffer
        For flipped SHORT (original bullish → swept lows):
          SL above the target_level (swing high, original TP destination) with wider buffer
        Falls back to ATR-based SL.
        """
        entry = signal.entry_price
        sweep = getattr(signal, "sweep_result", None)

        if sweep is not None and sweep.sweep_detected and sweep.target_level > 0:
            if is_long:
                # Original was bearish (swept highs) → we go LONG
                # SL below the target_level (swing low) with wide buffer
                sl = sweep.target_level * (1 - self.sl_buffer)
                if sl < entry:
                    return sl
            else:
                # Original was bullish (swept lows) → we go SHORT
                # SL above the target_level (swing high) with wide buffer
                sl = sweep.target_level * (1 + self.sl_buffer)
                if sl > entry:
                    return sl

        # ATR fallback with wider buffer
        atr = getattr(signal, "atr_1h", 0.0) or 0.0
        if atr > 0:
            distance = atr * 2.5  # wider than main bot's 2.0
        else:
            distance = entry * 0.04  # 4% fallback (vs 3% for main)

        if is_long:
            return entry - distance
        else:
            return entry + distance

    def _calculate_tp(self, signal: SignalCandidate, sl_price: float, is_long: bool) -> float | None:
        """Calculate TP for flipped trade.

        For flipped LONG: TP above sweep_level (the wick high the original shorted from)
        For flipped SHORT: TP below sweep_level (the wick low the original went long from)
        """
        entry = signal.entry_price
        sl_distance = abs(entry - sl_price)
        min_tp_distance = sl_distance * self.min_rr
        sweep = getattr(signal, "sweep_result", None)

        if is_long:
            min_tp = entry + min_tp_distance
            if sweep and sweep.sweep_level > entry:
                tp = sweep.sweep_level * 1.005
                if tp >= min_tp:
                    return tp
            # Structural levels
            for tf in ["1h_swing_high", "4h_swing_high", "1d_swing_high"]:
                level = signal.key_levels.get(tf)
                if level and level >= min_tp:
                    return level
            return min_tp
        else:
            min_tp = entry - min_tp_distance
            if sweep and sweep.sweep_level < entry:
                tp = sweep.sweep_level * 0.995
                if tp <= min_tp:
                    return tp
            for tf in ["1h_swing_low", "4h_swing_low", "1d_swing_low"]:
                level = signal.key_levels.get(tf)
                if level and level <= min_tp:
                    return level
            return min_tp

    def reset_daily(self) -> None:
        """Reset daily counters at midnight UTC."""
        self.daily_start_balance = self.balance
        self.daily_pnl = 0.0
        self.daily_trade_count = 0

    def to_state_dict(self) -> dict:
        """Serialize flipped trader state for crash recovery."""
        positions_data = {}
        for sym, pos in self.positions.items():
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
                "leverage": pos.leverage,
                "margin_used": pos.margin_used,
                "liquidation_price": pos.liquidation_price,
                "original_quantity": pos.original_quantity,
                "original_stop_loss": pos.original_stop_loss,
                "confluence_score": pos.confluence_score,
            }
        return {
            "balance": self.balance,
            "peak_balance": self.peak_balance,
            "daily_start_balance": self.daily_start_balance,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "daily_trade_count": self.daily_trade_count,
            "positions": positions_data,
        }

    def restore_state(self, data: dict) -> None:
        """Restore from persisted state dict."""
        if not data:
            return
        self.balance = float(data.get("balance", self.balance))
        self.peak_balance = float(data.get("peak_balance", self.peak_balance))
        self.daily_start_balance = float(data.get("daily_start_balance", self.daily_start_balance))
        self.daily_pnl = float(data.get("daily_pnl", 0))
        self.total_pnl = float(data.get("total_pnl", 0))
        self.daily_trade_count = int(data.get("daily_trade_count", 0))

        positions_data = data.get("positions", {})
        for sym, pd in positions_data.items():
            try:
                entry_time = datetime.now(timezone.utc)
                if pd.get("entry_time"):
                    entry_time = datetime.fromisoformat(str(pd["entry_time"]))
                self.positions[sym] = Position(
                    trade_id=pd.get("trade_id", ""),
                    symbol=sym,
                    entry_price=float(pd.get("entry_price", 0)),
                    quantity=float(pd.get("quantity", 0)),
                    stop_loss=float(pd.get("stop_loss", 0)),
                    take_profit=float(pd.get("take_profit")) if pd.get("take_profit") else None,
                    high_water_mark=float(pd.get("high_water_mark", pd.get("entry_price", 0))),
                    entry_time=entry_time,
                    cost_usd=float(pd.get("cost_usd", 0)),
                    direction=pd.get("direction", "long"),
                    leverage=int(pd.get("leverage", 1) or 1),
                    margin_used=float(pd.get("margin_used", 0) or 0),
                    liquidation_price=float(pd.get("liquidation_price", 0) or 0),
                    original_quantity=float(pd.get("original_quantity", 0) or 0),
                    original_stop_loss=float(pd.get("original_stop_loss", 0) or 0),
                    confluence_score=float(pd.get("confluence_score", 0) or 0),
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.warning("flipped_position_restore_failed", symbol=sym, error=str(e))

        logger.info(
            "flipped_state_restored",
            balance=self.balance,
            open_positions=len(self.positions),
            total_pnl=self.total_pnl,
        )
