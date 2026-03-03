from __future__ import annotations

import asyncio
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.config import Settings
from src.data.candles import CandleManager
from src.data.repository import Repository
from src.engine.scheduler import Scheduler, TickType
from src.engine.state import EngineState
from src.exchange.models import Position, TakeProfitTier
from src.execution.monitor import PositionMonitor
from src.execution.orders import OrderExecutor
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.manager import RiskManager
from src.risk.portfolio import PortfolioTracker
from src.strategy.adaptive_threshold import AdaptiveThreshold
from src.strategy.llm_analyst import LLMTradeAnalyst
from src.strategy.scanner import AltcoinScanner
from src.strategy.sentiment import SentimentFilter
from src.strategy.split_test import SplitTestManager
from src.strategy.trade_analyzer import TradeAnalyzer
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
        self.risk_manager = RiskManager(config, exchange=exchange)
        self.circuit_breaker = CircuitBreaker(config)
        self.order_executor = OrderExecutor(exchange, self.risk_manager, config)
        self.position_monitor = PositionMonitor()
        self.scanner = AltcoinScanner(candle_manager, config)

        # Self-improving components
        self.trade_analyzer = TradeAnalyzer(repo)
        self.adaptive_threshold = AdaptiveThreshold(config.entry_threshold)
        self.sentiment_filter = SentimentFilter(hf_api_token=config.hf_api_token)

        # LLM split test components
        self.split_test = SplitTestManager(config)
        self.llm_analyst: LLMTradeAnalyst | None = None
        if config.llm_api_key:
            self.llm_analyst = LLMTradeAnalyst(config)
            logger.info("llm_analyst_ready", model=config.llm_model, ratio=config.llm_split_ratio)

        self.state: EngineState | None = None
        self.portfolio: PortfolioTracker | None = None
        self._running = False
        self._reversal_cooldowns: dict[str, datetime] = {}

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

        # Restore PaperExchange internal position tracking after restart
        # Without this, partial exits would create phantom positions instead of closing
        if hasattr(self.exchange, "restore_positions"):
            self.exchange.restore_positions(self.state.open_positions)

        # Load historical trade data for self-improving components
        await self.trade_analyzer.load_history()

        # Bootstrap adaptive threshold from recent closed trades
        try:
            recent_trades = await self.repo.get_trades(status="closed", per_page=50)
            outcomes = [float(t.get("pnl_usd", 0)) > 0 for t in reversed(recent_trades)]
            if outcomes:
                self.adaptive_threshold.load_outcomes(outcomes)
                logger.info(
                    "adaptive_threshold_ready",
                    threshold=self.adaptive_threshold.threshold,
                    base=self.adaptive_threshold.base_threshold,
                )
        except Exception as e:
            logger.warning("adaptive_threshold_load_failed", error=str(e))

        self._running = True

    async def run(self) -> None:
        """Main event loop."""
        await self.startup()

        # Persist initial state immediately so the dashboard has data
        await self._persist_state()

        logger.info(
            "engine_started",
            mode=self.state.mode,
            account_type=self.config.account_type,
            leverage=self.config.leverage,
            balance=self.state.current_balance,
            positions=len(self.state.open_positions),
        )

        while self._running:
            try:
                tick_type = await self.scheduler.wait_for_next_tick()

                # Daily reset check
                if is_new_day(self.state.last_scan_time):
                    # Use equity (cash + deployed capital) — margin-aware
                    equity = self.portfolio.get_equity()
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

        # Circuit breaker check — use equity (cash + deployed capital, margin-aware)
        equity = self.portfolio.get_equity()
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

        # Apply adaptive threshold filter
        active_threshold = self.adaptive_threshold.threshold
        if active_threshold != self.config.entry_threshold:
            pre_count = len(signals)
            signals = [s for s in signals if s.score >= active_threshold]
            if pre_count != len(signals):
                logger.info(
                    "adaptive_threshold_filtered",
                    threshold=active_threshold,
                    before=pre_count,
                    after=len(signals),
                )

        # Execute entries for top signals
        signals_saved = 0
        trades_entered = 0
        for signal in signals:
            try:
                position = None

                # --- Sentiment filter (Strategy A) ---
                sentiment_score = await self.sentiment_filter.get_sentiment(signal.symbol)

                if signal.direction and self.sentiment_filter.should_block_trade(
                    sentiment_score, signal.direction
                ):
                    logger.info(
                        "trade_blocked_by_sentiment",
                        symbol=signal.symbol,
                        direction=signal.direction,
                        sentiment=round(sentiment_score, 2),
                    )
                    # Still log the signal but skip trade
                    vol_24h = self.exchange.get_24h_volume(signal.symbol)
                    await self.repo.insert_signal(
                        {
                            "symbol": signal.symbol,
                            "direction": signal.direction or "none",
                            "score": signal.score,
                            "reasons": signal.reasons + [f"BLOCKED:sentiment={sentiment_score:.1f}"],
                            "components": {"volume_24h": vol_24h, "sentiment": sentiment_score},
                            "current_price": signal.entry_price,
                            "acted_on": False,
                            "scan_cycle": cycle,
                        }
                    )
                    signals_saved += 1
                    continue

                # --- Critical event detection (zero-shot classification) ---
                critical_event = self.sentiment_filter.has_critical_event(signal.symbol)
                if critical_event:
                    logger.warning(
                        "trade_blocked_critical_event",
                        symbol=signal.symbol,
                        event=critical_event,
                    )
                    vol_24h = self.exchange.get_24h_volume(signal.symbol)
                    await self.repo.insert_signal(
                        {
                            "symbol": signal.symbol,
                            "direction": signal.direction or "none",
                            "score": signal.score,
                            "reasons": signal.reasons + [f"BLOCKED:critical_event={critical_event}"],
                            "components": {"volume_24h": vol_24h, "sentiment": sentiment_score},
                            "current_price": signal.entry_price,
                            "acted_on": False,
                            "scan_cycle": cycle,
                        }
                    )
                    signals_saved += 1
                    continue

                # Apply sentiment score adjustment to confluence score
                sentiment_adj = self.sentiment_filter.score_adjustment(
                    sentiment_score, signal.direction or "long"
                )
                adjusted_score = signal.score + sentiment_adj

                # Also check pattern-based score from trade analyzer (Strategy B)
                pattern_modifier = self.trade_analyzer.get_pattern_score(
                    signal.reasons, signal.direction or "long", signal.score
                )
                if pattern_modifier is not None and pattern_modifier < -0.5:
                    # Patterns suggest this is a losing setup — penalize
                    adjusted_score -= 5.0
                    logger.info(
                        "pattern_penalty_applied",
                        symbol=signal.symbol,
                        pattern_modifier=round(pattern_modifier, 2),
                    )

                # Re-check against threshold after adjustments
                if adjusted_score < active_threshold:
                    logger.info(
                        "trade_below_adjusted_threshold",
                        symbol=signal.symbol,
                        original_score=signal.score,
                        adjusted_score=round(adjusted_score, 2),
                        threshold=active_threshold,
                        sentiment_adj=sentiment_adj,
                    )
                    vol_24h = self.exchange.get_24h_volume(signal.symbol)
                    await self.repo.insert_signal(
                        {
                            "symbol": signal.symbol,
                            "direction": signal.direction or "none",
                            "score": signal.score,
                            "reasons": signal.reasons + [f"adj_score={adjusted_score:.1f}"],
                            "components": {"volume_24h": vol_24h, "sentiment": sentiment_score},
                            "current_price": signal.entry_price,
                            "acted_on": False,
                            "scan_cycle": cycle,
                        }
                    )
                    signals_saved += 1
                    continue

                # --- LLM Split Test: assign group and optionally analyze ---
                # Check runtime toggle from engine state (dashboard toggle)
                state_dict = await self.repo.get_engine_state() or {}
                llm_runtime_enabled = state_dict.get("llm_enabled", self.config.llm_enabled)

                if llm_runtime_enabled:
                    test_group = self.split_test.assign_group(signal)
                else:
                    test_group = "control"
                llm_analysis_data = None
                sl_override = None
                tp_override = None

                if test_group == "llm" and self.llm_analyst:
                    llm_context = {
                        "sentiment_score": sentiment_score,
                        "adjusted_score": adjusted_score,
                        "active_threshold": active_threshold,
                    }
                    llm_result = await self.llm_analyst.analyze_signal(signal, llm_context)
                    llm_analysis_data = {
                        "approve": llm_result.approve,
                        "confidence": llm_result.confidence,
                        "reasoning": llm_result.reasoning,
                        "suggested_sl": llm_result.suggested_sl,
                        "suggested_tp": llm_result.suggested_tp,
                        "latency_ms": round(llm_result.latency_ms, 1),
                        "error": llm_result.error,
                    }

                    if not llm_result.approve:
                        logger.info(
                            "trade_blocked_by_llm",
                            symbol=signal.symbol,
                            confidence=llm_result.confidence,
                            reasoning=llm_result.reasoning[:100],
                        )
                        vol_24h = self.exchange.get_24h_volume(signal.symbol)
                        await self.repo.insert_signal(
                            {
                                "symbol": signal.symbol,
                                "direction": signal.direction or "none",
                                "score": signal.score,
                                "reasons": signal.reasons + [f"BLOCKED:llm_reject(conf={llm_result.confidence:.0f})"],
                                "components": {
                                    "volume_24h": vol_24h,
                                    "sentiment": sentiment_score,
                                    "adjusted_score": round(adjusted_score, 2),
                                    "test_group": test_group,
                                    "llm_analysis": llm_analysis_data,
                                },
                                "current_price": signal.entry_price,
                                "acted_on": False,
                                "scan_cycle": cycle,
                            }
                        )
                        signals_saved += 1
                        continue

                    # Apply LLM SL/TP suggestions if provided
                    sl_override = llm_result.suggested_sl
                    tp_override = llm_result.suggested_tp

                # --- Opposite signal reversal check ---
                existing_pos = self.portfolio.open_positions.get(signal.symbol)
                if existing_pos:
                    is_opposite = (
                        (existing_pos.direction == "long" and signal.direction == "bearish")
                        or (existing_pos.direction == "short" and signal.direction == "bullish")
                    )
                    if is_opposite:
                        reversed_ok = await self._attempt_reversal(signal, existing_pos, cycle)
                        if reversed_ok:
                            trades_entered += 1
                            vol_24h = self.exchange.get_24h_volume(signal.symbol)
                            await self.repo.insert_signal(
                                {
                                    "symbol": signal.symbol,
                                    "direction": signal.direction or "none",
                                    "score": signal.score,
                                    "reasons": signal.reasons + ["reversal"],
                                    "components": {
                                        "volume_24h": vol_24h,
                                        "sentiment": sentiment_score,
                                        "adjusted_score": round(adjusted_score, 2),
                                        "test_group": test_group,
                                    },
                                    "current_price": signal.entry_price,
                                    "acted_on": True,
                                    "scan_cycle": cycle,
                                }
                            )
                            signals_saved += 1
                            continue
                        # Reversal guards failed — fall through to normal validation
                        # which will reject with "Already in position"

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
                    # --- Liquidity check: position size vs 24h volume ---
                    vol_24h_check = self.exchange.get_24h_volume(signal.symbol)
                    max_position_cost = self.portfolio.current_balance * self.config.max_position_pct
                    if vol_24h_check > 0 and max_position_cost / vol_24h_check > self.config.max_position_volume_pct:
                        logger.info(
                            "trade_rejected_low_liquidity",
                            symbol=signal.symbol,
                            position_cost=round(max_position_cost, 2),
                            volume_24h=round(vol_24h_check, 2),
                            pct_of_volume=round(max_position_cost / vol_24h_check * 100, 4),
                            max_pct=self.config.max_position_volume_pct * 100,
                        )
                    else:
                        # Execute entry (with optional LLM SL/TP overrides)
                        position, order_result, trade_record = await self.order_executor.execute_entry(
                            signal=signal,
                            current_balance=self.portfolio.current_balance,
                            mode=self.state.mode,
                            sl_override=sl_override,
                            tp_override=tp_override,
                        )

                        if position and trade_record:
                            # Tag trade with split test metadata
                            trade_record["test_group"] = test_group
                            if llm_analysis_data:
                                trade_record["llm_analysis"] = llm_analysis_data

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
                signal_components = {
                    "volume_24h": vol_24h,
                    "sentiment": sentiment_score,
                    "adjusted_score": round(adjusted_score, 2),
                    "test_group": test_group,
                }
                if llm_analysis_data:
                    signal_components["llm_analysis"] = llm_analysis_data
                await self.repo.insert_signal(
                    {
                        "symbol": signal.symbol,
                        "direction": signal.direction or "none",
                        "score": signal.score,
                        "reasons": signal.reasons,
                        "components": signal_components,
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
        """Check open positions for SL/TP/trailing stop + liquidation proximity."""
        if not self.portfolio.open_positions:
            return

        # Liquidation proximity check for leveraged positions
        from src.exchange.models import ExitSignal
        liq_exits: list[ExitSignal] = []
        for symbol, pos in list(self.portfolio.open_positions.items()):
            if pos.leverage <= 1 or pos.liquidation_price <= 0:
                continue
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                cur_price = float(ticker["last"])
                dist_pct = abs(cur_price - pos.liquidation_price) / cur_price
                if dist_pct < 0.02:
                    logger.warning(
                        "liquidation_proximity_exit",
                        symbol=symbol, current_price=cur_price,
                        liquidation_price=pos.liquidation_price,
                        distance_pct=round(dist_pct * 100, 2),
                    )
                    liq_exits.append(ExitSignal(symbol=symbol, reason="liquidation_proximity", price=cur_price))
            except Exception as e:
                logger.warning("liq_check_failed", symbol=symbol, error=str(e))

        for liq_exit in liq_exits:
            liq_pos = self.portfolio.open_positions.get(liq_exit.symbol)
            if not liq_pos:
                continue
            liq_result, liq_pnl = await self.order_executor.execute_exit(
                symbol=liq_exit.symbol, position=liq_pos,
                reason=liq_exit.reason, current_price=liq_exit.price,
            )
            if liq_result:
                liq_pnl_pct = (liq_pnl / liq_pos.cost_usd * 100) if liq_pos.cost_usd > 0 else 0
                await self.repo.close_trade(
                    trade_id=liq_pos.trade_id,
                    exit_price=liq_result.avg_price or liq_exit.price,
                    exit_quantity=liq_result.filled_quantity or liq_pos.quantity,
                    exit_order_id=liq_result.order_id,
                    exit_reason=liq_exit.reason,
                    pnl_usd=liq_pnl, pnl_percent=liq_pnl_pct, fees_usd=liq_result.fee,
                )
                self.portfolio.record_exit(liq_exit.symbol, liq_exit.price, liq_result.fee)
                self.risk_manager.record_stop_out(liq_exit.symbol)
                self.adaptive_threshold.record_outcome(is_win=(liq_pnl > 0))
                self.state.open_positions.pop(liq_exit.symbol, None)

        # Compute ATR values for open positions (15m candles)
        atr_values: dict[str, float] = {}
        for symbol in list(self.portfolio.open_positions.keys()):
            try:
                candles = await self.candle_manager.get_candles(symbol, "15m", limit=30)
                if candles is not None and len(candles) >= 15:
                    high = candles["high"].astype(float)
                    low = candles["low"].astype(float)
                    close = candles["close"].astype(float)
                    prev_close = close.shift(1)
                    tr = pd.concat(
                        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                        axis=1,
                    ).max(axis=1)
                    atr_series = tr.rolling(14).mean()
                    latest_atr = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
                    if latest_atr > 0:
                        atr_values[symbol] = latest_atr
            except Exception as e:
                logger.debug("atr_fetch_failed", symbol=symbol, error=str(e))

        exits = await self.position_monitor.check_positions(
            self.portfolio.open_positions, self.exchange, atr_values=atr_values
        )

        for exit_signal in exits:
            position = self.portfolio.open_positions.get(exit_signal.symbol)
            if not position:
                continue

            if exit_signal.is_partial and exit_signal.partial_quantity > 0:
                # --- PARTIAL EXIT (TP tier hit) ---
                order_result, pnl = await self.order_executor.execute_partial_exit(
                    symbol=exit_signal.symbol,
                    position=position,
                    reason=exit_signal.reason,
                    current_price=exit_signal.price,
                    quantity=exit_signal.partial_quantity,
                    tier=exit_signal.tier,
                )

                if order_result:
                    # Mark tier as filled
                    if position.tp_tiers:
                        for tier in position.tp_tiers:
                            if tier.level == exit_signal.tier and not tier.filled:
                                tier.filled = True
                                tier.fill_price = order_result.avg_price or exit_signal.price
                                tier.fill_time = datetime.now(timezone.utc)
                                break

                    position.current_tier = exit_signal.tier

                    # TP1: move SL to breakeven
                    if exit_signal.tier == 1 and self.config.move_sl_to_be_after_tp1:
                        position.stop_loss = position.entry_price
                        logger.info("sl_moved_to_breakeven", symbol=exit_signal.symbol,
                                    new_sl=position.entry_price)

                    # Record partial exit in portfolio (updates quantity, returns capital)
                    self.portfolio.record_partial_exit(
                        symbol=exit_signal.symbol,
                        exit_price=order_result.avg_price or exit_signal.price,
                        quantity=order_result.filled_quantity or exit_signal.partial_quantity,
                        fee=order_result.fee,
                    )

                    # Log partial exit to DB
                    cost_basis = exit_signal.partial_quantity * position.entry_price
                    pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
                    await self.repo.log_partial_exit(
                        trade_id=position.trade_id,
                        tier=exit_signal.tier,
                        exit_price=order_result.avg_price or exit_signal.price,
                        exit_quantity=order_result.filled_quantity or exit_signal.partial_quantity,
                        exit_order_id=order_result.order_id,
                        exit_reason=exit_signal.reason,
                        pnl_usd=round(pnl, 4),
                        pnl_percent=round(pnl_pct, 2),
                        fees_usd=order_result.fee,
                        remaining_quantity=position.quantity,
                        new_stop_loss=position.stop_loss,
                    )

                    # Update trade record with current tier state (including filled flags)
                    update_data = {
                        "current_tier": position.current_tier,
                        "remaining_quantity": position.quantity,
                        "stop_loss": position.stop_loss,
                        "margin_used": position.margin_used,
                    }
                    if position.tp_tiers:
                        update_data["tp_tiers"] = [
                            {
                                "level": t.level,
                                "price": t.price,
                                "pct": t.pct,
                                "quantity": t.quantity,
                                "filled": t.filled,
                                "fill_price": t.fill_price,
                                "fill_time": t.fill_time.isoformat() if t.fill_time else None,
                            }
                            for t in position.tp_tiers
                        ]
                    await self.repo.update_trade(position.trade_id, update_data)

                    # Position stays in state — NOT removed

            else:
                # --- FULL EXIT (SL, trailing stop, legacy TP, circuit breaker) ---
                order_result, pnl = await self.order_executor.execute_exit(
                    symbol=exit_signal.symbol,
                    position=position,
                    reason=exit_signal.reason,
                    current_price=exit_signal.price,
                )

                if order_result:
                    # Accumulate PnL from partial exits for total trade PnL
                    total_pnl = pnl
                    total_fees = order_result.fee
                    if position.tp_tiers and position.current_tier > 0:
                        try:
                            partial_exits = await self.repo.get_partial_exits(position.trade_id)
                            total_pnl += sum(float(pe.get("pnl_usd", 0)) for pe in partial_exits)
                            total_fees += sum(float(pe.get("fees_usd", 0)) for pe in partial_exits)
                        except Exception as e:
                            logger.warning("partial_exit_sum_failed", error=str(e))

                    orig_cost = (position.original_quantity * position.entry_price) if position.original_quantity else position.cost_usd
                    pnl_pct = (total_pnl / orig_cost * 100) if orig_cost > 0 else 0

                    await self.repo.close_trade(
                        trade_id=position.trade_id,
                        exit_price=order_result.avg_price or exit_signal.price,
                        exit_quantity=order_result.filled_quantity or position.quantity,
                        exit_order_id=order_result.order_id,
                        exit_reason=exit_signal.reason,
                        pnl_usd=round(total_pnl, 4),
                        pnl_percent=round(pnl_pct, 2),
                        fees_usd=round(total_fees, 4),
                    )

                    # Update portfolio
                    self.portfolio.record_exit(exit_signal.symbol, exit_signal.price, order_result.fee)

                    # Record cooldown on stop-loss exits
                    if exit_signal.reason == "sl_hit":
                        self.risk_manager.record_stop_out(exit_signal.symbol)

                    # --- Post-trade analysis (Strategy B) ---
                    holding_seconds = 0.0
                    if position.entry_time:
                        holding_seconds = (datetime.now(timezone.utc) - position.entry_time).total_seconds()

                    try:
                        await self.trade_analyzer.analyze_closed_trade(
                            trade_id=position.trade_id,
                            symbol=exit_signal.symbol,
                            direction=position.direction,
                            entry_price=position.entry_price,
                            exit_price=order_result.avg_price or exit_signal.price,
                            pnl_usd=total_pnl,
                            pnl_percent=pnl_pct,
                            exit_reason=exit_signal.reason,
                            confluence_score=0,
                            holding_seconds=holding_seconds,
                        )
                    except Exception as e:
                        logger.warning("post_trade_analysis_failed", error=str(e))

                    # --- Adaptive threshold update (Strategy C) ---
                    self.adaptive_threshold.record_outcome(is_win=(total_pnl > 0))

                    # Remove from state
                    self.state.open_positions.pop(exit_signal.symbol, None)

        # Sync state
        self.state.current_balance = self.portfolio.current_balance
        self.state.peak_balance = self.portfolio.peak_balance
        self.state.daily_pnl = self.portfolio.daily_pnl
        self.state.total_pnl = self.portfolio.total_pnl

    async def _attempt_reversal(
        self,
        signal,
        existing_position: Position,
        cycle: int,
    ) -> bool:
        """
        Attempt to close an existing position and open one in the opposite direction.

        Returns True if reversal succeeded (caller should skip normal entry flow).
        Returns False if reversal was blocked by guards (caller falls through to normal validation).
        """
        symbol = signal.symbol

        # Guard 1: Feature disabled
        if not self.config.reversal_enabled:
            return False

        # Guard 2: Score too low
        if signal.score < self.config.reversal_min_score:
            logger.info(
                "reversal_score_too_low", symbol=symbol,
                score=signal.score, min_score=self.config.reversal_min_score,
            )
            return False

        # Guard 3: Position held too briefly
        if existing_position.entry_time:
            held_minutes = (datetime.now(timezone.utc) - existing_position.entry_time).total_seconds() / 60
            if held_minutes < self.config.reversal_min_hold_minutes:
                logger.info(
                    "reversal_too_early", symbol=symbol,
                    held_minutes=round(held_minutes, 1),
                    min_minutes=self.config.reversal_min_hold_minutes,
                )
                return False

        # Guard 4: Reversal cooldown (separate from SL cooldown)
        cooldown_until = self._reversal_cooldowns.get(symbol)
        if cooldown_until and datetime.now(timezone.utc) < cooldown_until:
            remaining = (cooldown_until - datetime.now(timezone.utc)).total_seconds() / 60
            logger.info(
                "reversal_on_cooldown", symbol=symbol,
                remaining_minutes=round(remaining, 1),
            )
            return False

        # --- CLOSE EXISTING POSITION ---
        logger.info(
            "reversal_closing_position", symbol=symbol,
            old_direction=existing_position.direction,
            new_direction=signal.direction,
            signal_score=signal.score,
        )

        try:
            ticker = await asyncio.wait_for(
                self.exchange.fetch_ticker(symbol), timeout=45,
            )
            current_price = float(ticker["last"])
        except Exception as e:
            logger.error("reversal_ticker_failed", symbol=symbol, error=str(e))
            return False

        # Execute exit order
        order_result, pnl = await self.order_executor.execute_exit(
            symbol=symbol,
            position=existing_position,
            reason="opposite_signal_reversal",
            current_price=current_price,
        )

        if not order_result:
            logger.error("reversal_exit_failed", symbol=symbol)
            return False

        # Accumulate partial exit PnLs for total trade PnL
        total_pnl = pnl
        total_fees = order_result.fee
        if existing_position.tp_tiers and existing_position.current_tier > 0:
            try:
                partial_exits = await self.repo.get_partial_exits(existing_position.trade_id)
                total_pnl += sum(float(pe.get("pnl_usd", 0)) for pe in partial_exits)
                total_fees += sum(float(pe.get("fees_usd", 0)) for pe in partial_exits)
            except Exception as e:
                logger.warning("reversal_partial_sum_failed", error=str(e))

        orig_cost = (
            (existing_position.original_quantity * existing_position.entry_price)
            if existing_position.original_quantity
            else existing_position.cost_usd
        )
        pnl_pct = (total_pnl / orig_cost * 100) if orig_cost > 0 else 0

        # Close trade in DB
        await self.repo.close_trade(
            trade_id=existing_position.trade_id,
            exit_price=order_result.avg_price or current_price,
            exit_quantity=order_result.filled_quantity or existing_position.quantity,
            exit_order_id=order_result.order_id,
            exit_reason="opposite_signal_reversal",
            pnl_usd=round(total_pnl, 4),
            pnl_percent=round(pnl_pct, 2),
            fees_usd=round(total_fees, 4),
        )

        # Update portfolio & state
        self.portfolio.record_exit(symbol, current_price, order_result.fee)
        self.state.open_positions.pop(symbol, None)

        # Clear SL cooldown so the new entry isn't blocked
        self.risk_manager.clear_cooldown(symbol)

        # Post-trade analysis
        holding_seconds = 0.0
        if existing_position.entry_time:
            holding_seconds = (datetime.now(timezone.utc) - existing_position.entry_time).total_seconds()
        try:
            await self.trade_analyzer.analyze_closed_trade(
                trade_id=existing_position.trade_id,
                symbol=symbol,
                direction=existing_position.direction,
                entry_price=existing_position.entry_price,
                exit_price=order_result.avg_price or current_price,
                pnl_usd=total_pnl,
                pnl_percent=pnl_pct,
                exit_reason="opposite_signal_reversal",
                confluence_score=0,
                holding_seconds=holding_seconds,
            )
        except Exception as e:
            logger.warning("reversal_post_analysis_failed", error=str(e))

        self.adaptive_threshold.record_outcome(is_win=(total_pnl > 0))

        old_trade_id = existing_position.trade_id
        old_direction = existing_position.direction

        logger.info(
            "reversal_position_closed", symbol=symbol,
            old_direction=old_direction,
            pnl=round(total_pnl, 4),
        )

        # --- ENTER NEW POSITION ---
        new_position, new_order_result, new_trade_record = await self.order_executor.execute_entry(
            signal=signal,
            current_balance=self.portfolio.current_balance,
            mode=self.state.mode,
        )

        new_trade_id = ""
        if new_position and new_trade_record:
            db_trade = await self.repo.insert_trade(new_trade_record)
            if db_trade:
                new_position.trade_id = db_trade.get("id", "")
                new_trade_id = new_position.trade_id

            self.portfolio.record_entry(new_position)
            self.state.open_positions[symbol] = new_position

            logger.info(
                "reversal_new_position_opened", symbol=symbol,
                new_direction=new_position.direction,
                entry_price=new_position.entry_price,
                quantity=new_position.quantity,
            )
        else:
            logger.warning(
                "reversal_new_entry_failed", symbol=symbol,
                note="Old position closed but new entry failed — will retry on next scan",
            )

        # Record reversal event for analytics
        try:
            await self.repo.log_reversal(
                old_trade_id=old_trade_id,
                new_trade_id=new_trade_id,
                symbol=symbol,
                old_direction=old_direction,
                new_direction=signal.direction or "unknown",
                close_pnl=round(total_pnl, 4),
                signal_score=signal.score,
            )
        except Exception as e:
            logger.warning("log_reversal_failed", error=str(e))

        # Set reversal cooldown
        self._reversal_cooldowns[symbol] = datetime.now(timezone.utc) + timedelta(
            minutes=self.config.reversal_cooldown_minutes
        )

        return True

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

        Handles three cases on restart/deploy:
        1. Closed trades still in state — remove them and credit balance.
        2. Open trades in DB but missing from state — restore them.
        3. Orphaned positions (no trade_id or trade_id missing from DB) — remove them.
        """
        changed = False

        # --- Case 1: Remove stale closed positions from state ---
        # --- Case 3: Remove orphaned positions (no DB trade record) ---
        if self.state.open_positions:
            trade_ids = [
                pos.trade_id for pos in self.state.open_positions.values() if pos.trade_id
            ]
            trade_map = {}
            if trade_ids:
                db_trades = await self.repo.get_trades_by_ids(trade_ids)
                trade_map = {t["id"]: t for t in db_trades}

            removed = []
            orphaned = []
            for symbol, position in list(self.state.open_positions.items()):
                # Case 3: No trade_id or trade_id not found in DB — orphaned position
                if not position.trade_id or (position.trade_id and position.trade_id not in trade_map):
                    # Return margin/cost to balance
                    if position.leverage > 1:
                        margin = position.margin_used or (position.cost_usd / position.leverage)
                        self.portfolio.current_balance += margin
                    else:
                        self.portfolio.current_balance += position.cost_usd
                    self.portfolio.open_positions.pop(symbol, None)
                    self.state.open_positions.pop(symbol, None)
                    orphaned.append(symbol)
                    logger.warning(
                        "reconciled_orphaned_position",
                        symbol=symbol,
                        trade_id=position.trade_id or "(none)",
                        cost_usd=position.cost_usd,
                        balance=self.portfolio.current_balance,
                    )
                    continue

                db_trade = trade_map.get(position.trade_id)
                if not db_trade:
                    continue

                # Case 1: Trade is closed in DB but still in state
                if db_trade.get("status") == "closed":
                    pnl = float(db_trade.get("pnl_usd", 0))
                    fees = float(db_trade.get("fees_usd", 0))

                    if position.leverage > 1:
                        # Leveraged: return margin + PnL
                        margin = position.margin_used or (position.cost_usd / position.leverage)
                        self.portfolio.current_balance += margin + pnl
                    elif position.direction == "short":
                        self.portfolio.current_balance += position.cost_usd + pnl
                    else:
                        exit_price = db_trade.get("exit_price") or position.entry_price
                        exit_qty = db_trade.get("exit_quantity") or position.quantity
                        exit_price = float(exit_price)
                        exit_qty = float(exit_qty)
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

            if orphaned:
                changed = True
                logger.warning(
                    "reconciliation_orphaned",
                    orphaned=orphaned,
                    count=len(orphaned),
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

            leverage = int(t.get("leverage", 1) or 1)
            # Fallback: if DB says 1x but config is leveraged, use config leverage
            if leverage <= 1 and self.config.leverage > 1:
                leverage = self.config.leverage
            cost_usd = float(t.get("entry_cost_usd", 0))
            margin_used = float(t.get("margin_used") or 0) or (cost_usd / leverage if leverage > 1 else 0.0)

            # Restore TP tiers if present
            tp_tiers = None
            tp_tiers_data = t.get("tp_tiers")
            if tp_tiers_data and isinstance(tp_tiers_data, list):
                tp_tiers = []
                for td in tp_tiers_data:
                    fill_time = None
                    if td.get("fill_time"):
                        try:
                            fill_time = datetime.fromisoformat(str(td["fill_time"]))
                        except (ValueError, TypeError):
                            pass
                    tp_tiers.append(TakeProfitTier(
                        level=int(td["level"]),
                        price=float(td["price"]) if td.get("price") is not None else None,
                        pct=float(td.get("pct", 0.33)),
                        quantity=float(td.get("quantity", 0)),
                        filled=bool(td.get("filled", False)),
                        fill_price=float(td.get("fill_price", 0)),
                        fill_time=fill_time,
                    ))

            # Use remaining_quantity if partial exits have occurred
            qty = float(t.get("remaining_quantity") or t.get("entry_quantity", 0))

            position = Position(
                trade_id=t["id"],
                symbol=symbol,
                direction=t.get("direction", "long"),
                entry_price=float(t.get("entry_price", 0)),
                quantity=qty,
                stop_loss=float(t.get("stop_loss", 0)),
                take_profit=float(t.get("take_profit")) if t.get("take_profit") else None,
                high_water_mark=float(t.get("entry_price", 0)),
                entry_time=datetime.fromisoformat(t["entry_time"]) if t.get("entry_time") else datetime.now(timezone.utc),
                cost_usd=cost_usd,
                leverage=leverage,
                margin_used=margin_used,
                liquidation_price=float(t.get("liquidation_price", 0) or 0),
                tp_tiers=tp_tiers,
                original_quantity=float(t.get("original_quantity") or t.get("entry_quantity", 0)),
                original_stop_loss=float(t.get("stop_loss", 0)),
                current_tier=int(t.get("current_tier", 0) or 0),
            )

            self.state.open_positions[symbol] = position
            self.portfolio.open_positions[symbol] = position
            # Deduct margin (for leveraged) or full cost (for spot)
            deduction = margin_used if leverage > 1 else cost_usd
            self.portfolio.current_balance -= deduction
            restored.append(symbol)

            logger.info(
                "reconciled_missing_position",
                symbol=symbol,
                trade_id=t["id"],
                cost=position.cost_usd,
                margin=deduction,
                leverage=leverage,
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
        await self.sentiment_filter.close()
        if self.llm_analyst:
            await self.llm_analyst.close()
        logger.info("engine_shutdown_complete")
