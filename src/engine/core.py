from __future__ import annotations

import asyncio
import traceback
from datetime import datetime, timezone

from src.config import Settings
from src.data.candles import CandleManager
from src.data.repository import Repository
from src.engine.scheduler import Scheduler, TickType
from src.engine.state import EngineState
from src.exchange.models import Position
from src.execution.monitor import PositionMonitor
from src.execution.orders import OrderExecutor
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.manager import RiskManager
from src.risk.portfolio import PortfolioTracker
from src.strategy.scanner import AltcoinScanner
from src.utils.logging import get_logger
from src.utils.time_utils import is_new_day

logger = get_logger(__name__)


class TradingEngine:
    """Main trading engine — the heartbeat of Tarakta."""

    def __init__(
        self,
        config: Settings,
        exchange,
        repo: Repository,
        candle_manager: CandleManager,
    ) -> None:
        self.config = config
        self.exchange = exchange
        self.repo = repo
        self.candle_manager = candle_manager

        self.scheduler = Scheduler(
            primary_interval_minutes=config.scan_interval_minutes,
            monitor_interval_seconds=60,
        )
        self.risk_manager = RiskManager(config)
        self.circuit_breaker = CircuitBreaker(config)
        self.order_executor = OrderExecutor(exchange, self.risk_manager, config.min_rr_ratio)
        self.position_monitor = PositionMonitor()
        self.scanner = AltcoinScanner(candle_manager, config)

        self.state: EngineState | None = None
        self.portfolio: PortfolioTracker | None = None
        self._running = False

    async def startup(self) -> None:
        """Initialize engine state from DB or create fresh."""
        saved = await self.repo.get_engine_state()

        if saved:
            self.state = EngineState.from_db(saved, mode=self.config.trading_mode)
            logger.info("engine_recovered", cycle=self.state.cycle_count)
        else:
            self.state = EngineState(
                mode=self.config.trading_mode,
                initial_balance=self.config.initial_balance,
            )
            logger.info("engine_fresh_start", balance=self.config.initial_balance)

        self.portfolio = PortfolioTracker(
            initial_balance=self.state.current_balance,
            peak_balance=self.state.peak_balance,
            daily_start_balance=self.state.daily_start_balance,
        )
        self.portfolio.open_positions = dict(self.state.open_positions)
        self.portfolio.daily_pnl = self.state.daily_pnl
        self.portfolio.total_pnl = self.state.total_pnl

        # Reconcile open positions against trade records FIRST
        # (this may close stale positions and add to P&L counters)
        await self._reconcile_positions()

        # THEN set P&L from closed trades in DB — this is the ground truth
        # and overwrites whatever _reconcile_positions accumulated,
        # preventing double-counting.
        stats = await self.repo.get_trade_stats()
        db_total_pnl = stats["total_pnl"]
        db_daily_pnl = await self.repo.get_daily_realized_pnl()

        if abs(db_total_pnl - self.portfolio.total_pnl) > 0.01:
            logger.info(
                "total_pnl_reconciled",
                old=self.portfolio.total_pnl,
                from_db=db_total_pnl,
            )
        if abs(db_daily_pnl - self.portfolio.daily_pnl) > 0.01:
            logger.info(
                "daily_pnl_reconciled",
                old=self.portfolio.daily_pnl,
                from_db=db_daily_pnl,
            )

        self.portfolio.total_pnl = db_total_pnl
        self.state.total_pnl = db_total_pnl
        self.portfolio.daily_pnl = db_daily_pnl
        self.state.daily_pnl = db_daily_pnl

        self._running = True

    async def run(self) -> None:
        """Main event loop."""
        await self.startup()

        # Persist initial state immediately so the dashboard has data
        await self._persist_state()

        logger.info(
            "engine_started",
            mode=self.state.mode,
            balance=self.state.current_balance,
            positions=len(self.state.open_positions),
        )

        while self._running:
            try:
                tick_type = await self.scheduler.wait_for_next_tick()

                # Daily reset check
                if is_new_day(self.state.last_scan_time):
                    # Use equity (cash + open position costs) so daily drawdown
                    # comparison is consistent with risk manager's equity calc
                    equity = self.portfolio.current_balance + sum(
                        pos.cost_usd for pos in self.portfolio.open_positions.values()
                    )
                    self.portfolio.reset_daily()
                    self.portfolio.daily_start_balance = equity
                    self.state.daily_start_balance = equity
                    self.state.daily_pnl = 0.0
                    self.state.last_scan_time = datetime.now(timezone.utc)
                    logger.info(
                        "daily_reset",
                        daily_start_balance=self.state.daily_start_balance,
                        equity=equity,
                    )

                if tick_type == TickType.MONITOR:
                    await self._monitor_tick()
                else:
                    await self._primary_tick()

                self.state.errors_consecutive = 0

            except asyncio.CancelledError:
                logger.info("engine_cancelled")
                break
            except Exception as e:
                self.state.errors_consecutive += 1
                logger.error(
                    "engine_cycle_error",
                    error=str(e),
                    consecutive_errors=self.state.errors_consecutive,
                    exc_info=True,
                )
                await self.repo.log_error(
                    "engine", "error", str(e), stack_trace=traceback.format_exc()
                )

                # Back off on repeated errors
                if self.state.errors_consecutive >= 5:
                    logger.critical("too_many_consecutive_errors", count=self.state.errors_consecutive)
                    await asyncio.sleep(300)  # 5 min cooldown
                else:
                    await asyncio.sleep(30)

    async def _primary_tick(self) -> None:
        """Full scan cycle: scan → analyze → decide → execute → log."""
        self.state.cycle_count += 1
        cycle = self.state.cycle_count
        logger.info("primary_tick_start", cycle=cycle)

        # Circuit breaker check — use equity (cash + open positions), not just cash
        equity = self.portfolio.current_balance + sum(
            pos.cost_usd for pos in self.portfolio.open_positions.values()
        )
        cb_status = self.circuit_breaker.check(
            current_balance=equity,
            daily_start_balance=self.portfolio.daily_start_balance,
            peak_balance=self.portfolio.peak_balance,
        )
        if cb_status.triggered:
            self.state.status = "circuit_break"
            logger.warning("circuit_breaker_active", reason=cb_status.reason)
            await self._persist_state()
            return

        self.state.status = "running"

        # Monitor existing positions first
        await self._monitor_tick()

        # Get tradeable pairs
        try:
            pairs = await self.exchange.get_tradeable_pairs(
                min_volume_usd=self.config.min_volume_usd,
                quote_currencies=self.config.quote_currencies,
            )
        except Exception as e:
            logger.error("pair_scan_failed", error=str(e))
            await self._persist_state()
            return

        if not pairs:
            logger.warning("no_tradeable_pairs")
            await self._persist_state()
            return

        # Scan for signals
        signals = await self.scanner.scan(pairs)

        # Execute entries for top signals
        signals_saved = 0
        trades_entered = 0
        for signal in signals:
            try:
                position = None

                # Validate trade
                # Calculate total exposure across open positions
                total_exposure = sum(
                    pos.cost_usd for pos in self.portfolio.open_positions.values()
                )
                validation = self.risk_manager.validate_trade(
                    open_position_count=len(self.portfolio.open_positions),
                    open_position_symbols=set(self.portfolio.open_positions.keys()),
                    current_balance=self.portfolio.current_balance,
                    daily_start_balance=self.portfolio.daily_start_balance,
                    daily_pnl=self.portfolio.daily_pnl,
                    signal=signal,
                    total_exposure_usd=total_exposure,
                )

                if not validation.allowed:
                    logger.info("trade_rejected", symbol=signal.symbol, reason=validation.reason)
                else:
                    # Execute entry
                    position, order_result, trade_record = await self.order_executor.execute_entry(
                        signal=signal,
                        current_balance=self.portfolio.current_balance,
                        mode=self.state.mode,
                    )

                    if position and trade_record:
                        # Save to DB
                        db_trade = await self.repo.insert_trade(trade_record)
                        if db_trade:
                            position.trade_id = db_trade.get("id", "")

                        # Update portfolio
                        self.portfolio.record_entry(position)
                        self.state.open_positions[signal.symbol] = position
                        trades_entered += 1

                # Log the signal regardless of whether trade was executed
                vol_24h = self.exchange.get_24h_volume(signal.symbol)
                await self.repo.insert_signal(
                    {
                        "symbol": signal.symbol,
                        "direction": signal.direction or "none",
                        "score": signal.score,
                        "reasons": signal.reasons,
                        "components": {"volume_24h": vol_24h},
                        "current_price": signal.entry_price,
                        "acted_on": position is not None,
                        "trade_id": position.trade_id if position else None,
                        "scan_cycle": cycle,
                    }
                )
                signals_saved += 1

            except Exception as e:
                logger.error(
                    "signal_processing_failed",
                    symbol=signal.symbol,
                    score=signal.score,
                    error=str(e),
                )
                await self.repo.log_error(
                    "engine",
                    "error",
                    f"Signal processing failed for {signal.symbol}: {e}",
                    details={"symbol": signal.symbol, "score": signal.score},
                    stack_trace=traceback.format_exc(),
                )

        # Update state
        self.state.last_scan_time = datetime.now(timezone.utc)
        self.state.current_balance = self.portfolio.current_balance
        self.state.peak_balance = self.portfolio.peak_balance
        self.state.daily_pnl = self.portfolio.daily_pnl
        self.state.total_pnl = self.portfolio.total_pnl

        await self._persist_state()

        logger.info(
            "primary_tick_complete",
            cycle=cycle,
            balance=self.portfolio.current_balance,
            open_positions=len(self.portfolio.open_positions),
            signals_found=len(signals),
            signals_saved=signals_saved,
            trades_entered=trades_entered,
        )

    async def _monitor_tick(self) -> None:
        """Check open positions for SL/TP/trailing stop hits."""
        if not self.portfolio.open_positions:
            return

        exits = await self.position_monitor.check_positions(
            self.portfolio.open_positions, self.exchange
        )

        for exit_signal in exits:
            position = self.portfolio.open_positions.get(exit_signal.symbol)
            if not position:
                continue

            order_result, pnl = await self.order_executor.execute_exit(
                symbol=exit_signal.symbol,
                position=position,
                reason=exit_signal.reason,
                current_price=exit_signal.price,
            )

            if order_result:
                # Update DB
                pnl_pct = (pnl / position.cost_usd * 100) if position.cost_usd > 0 else 0
                await self.repo.close_trade(
                    trade_id=position.trade_id,
                    exit_price=order_result.avg_price or exit_signal.price,
                    exit_quantity=order_result.filled_quantity or position.quantity,
                    exit_order_id=order_result.order_id,
                    exit_reason=exit_signal.reason,
                    pnl_usd=pnl,
                    pnl_percent=pnl_pct,
                    fees_usd=order_result.fee,
                )

                # Update portfolio
                self.portfolio.record_exit(exit_signal.symbol, exit_signal.price, order_result.fee)

                # Record cooldown on stop-loss exits
                if exit_signal.reason == "sl_hit":
                    self.risk_manager.record_stop_out(exit_signal.symbol)

                # Remove from state
                self.state.open_positions.pop(exit_signal.symbol, None)

        # Sync state
        self.state.current_balance = self.portfolio.current_balance
        self.state.peak_balance = self.portfolio.peak_balance
        self.state.daily_pnl = self.portfolio.daily_pnl
        self.state.total_pnl = self.portfolio.total_pnl

    async def _persist_state(self) -> None:
        """Save engine state and portfolio snapshot to DB."""
        try:
            await self.repo.upsert_engine_state(
                self.portfolio.to_state_dict(
                    status=self.state.status,
                    mode=self.state.mode,
                    cycle_count=self.state.cycle_count,
                )
            )
            await self.repo.insert_snapshot(
                self.portfolio.to_snapshot_dict(
                    cycle_number=self.state.cycle_count,
                    mode=self.state.mode,
                )
            )
        except Exception as e:
            logger.error("state_persist_failed", error=str(e))

    async def _reconcile_positions(self) -> None:
        """Cross-check open positions in state against trade records in DB.

        Handles two cases on restart/deploy:
        1. Closed trades still in state — remove them and credit balance.
        2. Open trades in DB but missing from state — restore them.
        """
        changed = False

        # --- Case 1: Remove stale closed positions from state ---
        if self.state.open_positions:
            trade_ids = [
                pos.trade_id for pos in self.state.open_positions.values() if pos.trade_id
            ]
            if trade_ids:
                db_trades = await self.repo.get_trades_by_ids(trade_ids)
                trade_map = {t["id"]: t for t in db_trades}

                removed = []
                for symbol, position in list(self.state.open_positions.items()):
                    db_trade = trade_map.get(position.trade_id)
                    if not db_trade:
                        continue

                    if db_trade.get("status") == "closed":
                        pnl = float(db_trade.get("pnl_usd", 0))
                        fees = float(db_trade.get("fees_usd", 0))

                        if position.direction == "short":
                            self.portfolio.current_balance += position.cost_usd + pnl
                        else:
                            exit_price = float(db_trade.get("exit_price", position.entry_price))
                            exit_qty = float(db_trade.get("exit_quantity", position.quantity))
                            revenue = exit_qty * exit_price
                            self.portfolio.current_balance += revenue - fees

                        self.portfolio.daily_pnl += pnl
                        self.portfolio.total_pnl += pnl
                        self.portfolio.open_positions.pop(symbol, None)
                        self.state.open_positions.pop(symbol, None)
                        removed.append(symbol)

                        logger.info(
                            "reconciled_stale_position",
                            symbol=symbol,
                            trade_id=position.trade_id,
                            pnl=pnl,
                            balance=self.portfolio.current_balance,
                        )

                if removed:
                    changed = True
                    logger.info(
                        "reconciliation_removed",
                        removed=removed,
                        balance=self.portfolio.current_balance,
                    )

        # --- Case 2: Restore open trades missing from state ---
        db_open_trades = await self.repo.get_open_trades(mode=self.state.mode)
        state_trade_ids = {
            pos.trade_id for pos in self.state.open_positions.values() if pos.trade_id
        }

        restored = []
        for t in db_open_trades:
            if t["id"] in state_trade_ids:
                continue
            # This trade is open in DB but missing from engine state
            symbol = t["symbol"]
            if symbol in self.state.open_positions:
                continue  # Already tracked (different trade_id edge case)

            position = Position(
                trade_id=t["id"],
                symbol=symbol,
                direction=t.get("direction", "long"),
                entry_price=float(t.get("entry_price", 0)),
                quantity=float(t.get("entry_quantity", 0)),
                stop_loss=float(t.get("stop_loss", 0)),
                take_profit=float(t.get("take_profit")) if t.get("take_profit") else None,
                high_water_mark=float(t.get("entry_price", 0)),
                entry_time=datetime.fromisoformat(t["entry_time"]) if t.get("entry_time") else datetime.now(timezone.utc),
                cost_usd=float(t.get("entry_cost_usd", 0)),
            )

            self.state.open_positions[symbol] = position
            self.portfolio.open_positions[symbol] = position
            # Deduct cost from balance (was already deducted when trade was placed)
            self.portfolio.current_balance -= position.cost_usd
            restored.append(symbol)

            logger.info(
                "reconciled_missing_position",
                symbol=symbol,
                trade_id=t["id"],
                cost=position.cost_usd,
                balance=self.portfolio.current_balance,
            )

        if restored:
            changed = True
            logger.info(
                "reconciliation_restored",
                restored=restored,
                balance=self.portfolio.current_balance,
                total_positions=len(self.state.open_positions),
            )

        if changed:
            self.state.current_balance = self.portfolio.current_balance
            self.state.peak_balance = self.portfolio.peak_balance

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("engine_shutting_down")
        self._running = False
        await self._persist_state()
        logger.info("engine_shutdown_complete")
