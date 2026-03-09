from __future__ import annotations

import asyncio
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from src.config import Settings
from src.data.candles import CandleManager
from src.data.repository import Repository
from src.engine.consensus import ConsensusMonitor
from src.engine.entry_refiner import EntryRefiner
from src.engine.flipped import FlippedTrader
from src.engine.scheduler import Scheduler, TickType
from src.engine.watchlist import WatchlistMonitor
from src.engine.state import EngineState
from src.exchange.models import Position, TakeProfitTier
from src.exchange.trading_hours import TradingHoursManager
from src.execution.monitor import PositionMonitor
from src.execution.orders import OrderExecutor
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.manager import RiskManager
from src.risk.portfolio import PortfolioTracker
from src.strategy.adaptive_threshold import AdaptiveThreshold
from src.strategy.confluence import WEIGHTS
from src.strategy.dynamic_weights import DynamicWeightOptimizer
from src.strategy.agent_analyst import AgentEntryAnalyst
from src.strategy.llm_analyst import LLMTradeAnalyst
from src.strategy.market_filter import MarketFilter
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
        self.position_monitor = PositionMonitor(
            trailing_activation_rr=config.trailing_activation_rr,
            trailing_atr_multiplier=config.trailing_atr_multiplier,
            breakeven_activation_rr=config.breakeven_activation_rr,
            max_hold_hours=config.max_hold_hours,
            stale_close_below_rr=config.stale_close_below_rr,
        )
        self.scanner = AltcoinScanner(candle_manager, config)

        # Market-level cross-reference filters (BTC gate, breadth, persistence, funding, correlation)
        self.market_filter = MarketFilter(
            btc_macro_gate_enabled=getattr(config, "btc_macro_gate_enabled", True),
            market_breadth_enabled=getattr(config, "market_breadth_enabled", True),
            market_breadth_threshold=getattr(config, "market_breadth_threshold", 0.70),
            funding_gate_enabled=getattr(config, "funding_gate_enabled", True),
            funding_gate_threshold=getattr(config, "funding_gate_threshold", 0.0005),
            signal_persistence_scans=getattr(config, "signal_persistence_scans", 2),
            max_per_correlation_cluster=getattr(config, "max_per_correlation_cluster", 2),
        )
        self._scan_cycle = 0  # Track scan cycles for signal persistence

        # Hyper-Watchlist Monitor — fast 5m monitoring for near-miss signals
        self._watchlist_queue: asyncio.Queue = asyncio.Queue()
        self.watchlist_monitor: WatchlistMonitor | None = (
            WatchlistMonitor(
                candle_manager=candle_manager,
                config=config,
                signal_queue=self._watchlist_queue,
            )
            if config.watchlist_enabled
            else None
        )

        # Post-sweep entry refinement — 5m monitoring for better entries (main bot)
        self.main_entry_refiner: EntryRefiner | None = (
            EntryRefiner(candle_manager=candle_manager, config=config)
            if config.entry_refiner_enabled
            else None
        )

        # Market consensus monitor — portfolio + BTC alignment check
        self.consensus_monitor: ConsensusMonitor | None = (
            ConsensusMonitor(candle_manager=candle_manager, config=config)
            if config.consensus_enabled
            else None
        )

        # Trading hours (24/7 for crypto, restricted for stocks/commodities)
        self.trading_hours = TradingHoursManager()
        self._market_name: str = "crypto"  # Set by main.py for multi-market

        # Self-improving components
        self.trade_analyzer = TradeAnalyzer(repo)
        self.adaptive_threshold = AdaptiveThreshold(config.entry_threshold)
        self.sentiment_filter = SentimentFilter(hf_api_token=config.hf_api_token)

        # LLM split test components (legacy Anthropic gate)
        self.split_test = SplitTestManager(config)
        self.llm_analyst: LLMTradeAnalyst | None = None
        if config.llm_api_key:
            self.llm_analyst = LLMTradeAnalyst(config)
            logger.info("llm_analyst_ready", model=config.llm_model, ratio=config.llm_split_ratio)

        # AI entry agent (OpenAI — intelligent decision-maker)
        self.agent_analyst: AgentEntryAnalyst | None = None
        if config.agent_api_key:
            self.agent_analyst = AgentEntryAnalyst(config)
            logger.info(
                "agent_analyst_ready",
                model=config.agent_model,
                min_score=config.agent_min_score,
            )

        self.state: EngineState | None = None
        self.portfolio: PortfolioTracker | None = None
        self._running = False
        self._scanning_active = False  # Main bot starts PAUSED — user clicks Start
        self._reversal_cooldowns: dict[str, datetime] = {}
        # Throttle repeated error logs (e.g. liq_check_failed for same symbol)
        self._error_last_logged: dict[str, datetime] = {}

        # Dynamic weight optimizer
        self.dynamic_weights: DynamicWeightOptimizer | None = None
        if config.dynamic_weights_enabled:
            self.dynamic_weights = DynamicWeightOptimizer(WEIGHTS)
            logger.info("dynamic_weights_enabled")

        # Flipped shadow trader (independent scanner + loop)
        self.flipped_trader = FlippedTrader(
            config, repo, candle_manager=candle_manager, exchange=exchange,
        )
        if self.flipped_trader.enabled:
            logger.info(
                "flipped_trader_enabled",
                leverage=config.flipped_leverage,
                sl_buffer=config.flipped_sl_buffer,
                scan_interval=config.flipped_scan_interval_minutes,
            )

        # Custom configurable bot (user can toggle direction + margin via dashboard)
        self.custom_trader = FlippedTrader(
            config, repo, candle_manager=candle_manager, exchange=exchange,
            flip_direction=config.custom_flip_direction,
            mode_name="custom_paper",
            state_key="custom_trader",
        )
        if config.custom_enabled:
            self.custom_trader.enabled = True
            self.custom_trader.scan_interval = config.custom_scan_interval_minutes
            self.custom_trader.max_position_pct = config.custom_margin_pct
            self.custom_trader.max_risk_pct = config.custom_margin_pct
            self.custom_trader._initial_balance = config.custom_initial_balance
            self.custom_trader.balance = config.custom_initial_balance
            self.custom_trader.peak_balance = config.custom_initial_balance
            self.custom_trader.daily_start_balance = config.custom_initial_balance
            self.custom_trader.leverage = config.custom_leverage
            # Custom bot starts PAUSED — user must click "Begin" on dashboard
            self.custom_trader._scanning_active = False
            # Post-sweep entry refinement: drop to 5m after 1H sweep detection
            if config.entry_refiner_enabled:
                self.custom_trader.entry_refiner = EntryRefiner(
                    candle_manager=candle_manager,
                    config=config,
                )
            # Market consensus monitor for custom bot
            if config.consensus_enabled:
                self.custom_trader.consensus_monitor = ConsensusMonitor(
                    candle_manager=candle_manager,
                    config=config,
                )
            logger.info(
                "custom_trader_enabled",
                leverage=config.custom_leverage,
                flip_direction=config.custom_flip_direction,
                margin_pct=config.custom_margin_pct,
                scan_interval=config.custom_scan_interval_minutes,
                entry_refiner=config.entry_refiner_enabled,
                consensus=config.consensus_enabled,
            )

        # Background task infrastructure (for async post-mortems)
        self._postmortem_semaphore = asyncio.Semaphore(2)
        self._background_tasks: set[asyncio.Task] = set()

    def begin_scanning(self) -> None:
        """Start the main bot's scan loop (called from dashboard)."""
        self._scanning_active = True
        logger.info("main_bot_scanning_started")

    def stop_scanning(self) -> None:
        """Pause the main bot's scan loop (keeps monitoring open positions)."""
        self._scanning_active = False
        logger.info("main_bot_scanning_stopped")

    def update_settings(
        self,
        margin_pct: float | None = None,
        leverage: int | None = None,
    ) -> None:
        """Update main bot margin % and leverage at runtime (takes effect on next trade)."""
        if margin_pct is not None:
            self.config.max_position_pct = margin_pct
            self.risk_manager.max_position_pct = margin_pct
            logger.info("main_bot_margin_updated", margin_pct=margin_pct)
        if leverage is not None:
            self.config.leverage = leverage
            self.risk_manager._leverage = leverage
            # Also update PaperExchange leverage if paper trading
            if hasattr(self.exchange, "_leverage"):
                self.exchange._leverage = leverage
            logger.info("main_bot_leverage_updated", leverage=leverage)

    async def startup(self) -> None:
        """Initialize engine state from DB or create fresh."""

        # Force reset: wipe all DB data and start fresh (set FORCE_RESET=true env var)
        if self.config.force_reset:
            logger.info("force_reset_triggered", balance=self.config.initial_balance)
            await self.repo.wipe_all_data()
            self.state = EngineState(
                mode=self.config.trading_mode,
                initial_balance=self.config.initial_balance,
            )
            logger.info("engine_fresh_start_after_reset", balance=self.config.initial_balance)
        else:
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

        # Live exchange reconciliation — compare Binance positions vs DB state
        # This catches orphaned exchange positions (e.g. manual trades, failed exits)
        await self._reconcile_exchange_positions()

        # Restore flipped trader state (stored in config_overrides JSONB)
        if self.flipped_trader.enabled:
            saved_state = await self.repo.get_engine_state() or {}
            overrides = saved_state.get("config_overrides") or {}
            flipped_data = overrides.get("flipped_trader") if isinstance(overrides, dict) else None
            if flipped_data:
                self.flipped_trader.restore_state(flipped_data)
            else:
                # First run or after reset — reconcile from DB
                flipped_stats = await self.repo.get_trade_stats(mode="flipped_paper")
                self.flipped_trader.total_pnl = flipped_stats.get("total_pnl", 0)
            logger.info(
                "flipped_trader_restored",
                balance=self.flipped_trader.balance,
                positions=len(self.flipped_trader.positions),
                total_pnl=self.flipped_trader.total_pnl,
            )

        # Restore custom trader state (stored in config_overrides JSONB)
        if self.custom_trader.enabled:
            saved_state = await self.repo.get_engine_state() or {}
            overrides = saved_state.get("config_overrides") or {}
            # Use dynamic state_key (e.g. "custom_trader" for crypto, "custom_trader_stocks" for stocks)
            ct_key = self.custom_trader.state_key
            custom_data = overrides.get(ct_key) if isinstance(overrides, dict) else None
            if custom_data:
                self.custom_trader.restore_state(custom_data)
            else:
                custom_stats = await self.repo.get_trade_stats(mode="custom_paper")
                self.custom_trader.total_pnl = custom_stats.get("total_pnl", 0)

            # Apply custom_trader_settings (user's explicit dashboard choices) on top
            # These are SHARED across all markets (one set of user preferences)
            custom_settings = overrides.get("custom_trader_settings") if isinstance(overrides, dict) else None
            if custom_settings and isinstance(custom_settings, dict):
                if "flip_direction" in custom_settings:
                    self.custom_trader.flip_direction = bool(custom_settings["flip_direction"])
                if "margin_pct" in custom_settings:
                    self.custom_trader.max_position_pct = float(custom_settings["margin_pct"])
                    self.custom_trader.max_risk_pct = float(custom_settings["margin_pct"])
                if "flip_mode" in custom_settings:
                    mode = str(custom_settings["flip_mode"])
                    if mode in ("always_flip", "smart_flip", "normal"):
                        self.custom_trader.flip_mode = mode
                if "flip_threshold" in custom_settings:
                    self.custom_trader.flip_threshold = max(0.0, min(1.0, float(custom_settings["flip_threshold"])))
                if "leverage" in custom_settings:
                    saved_lev = int(custom_settings["leverage"])
                    self.custom_trader.leverage = saved_lev
                    if hasattr(self.custom_trader.exchange, "_leverage"):
                        self.custom_trader.exchange._leverage = saved_lev
                    logger.info("custom_leverage_restored", leverage=saved_lev)
                # Restore scanning_active from shared settings (applies to all markets)
                if "scanning_active" in custom_settings:
                    self.custom_trader._scanning_active = bool(custom_settings["scanning_active"])

            logger.info(
                "custom_trader_restored",
                market=self._market_name,
                state_key=ct_key,
                balance=self.custom_trader.balance,
                positions=len(self.custom_trader.positions),
                total_pnl=self.custom_trader.total_pnl,
                flip_direction=self.custom_trader.flip_direction,
                flip_mode=self.custom_trader.flip_mode,
                margin_pct=self.custom_trader.max_position_pct,
                scanning_active=self.custom_trader._scanning_active,
            )

        # Load historical trade data for self-improving components
        await self.trade_analyzer.load_history()

        # Bootstrap adaptive threshold from recent closed trades
        try:
            recent_trades = await self.repo.get_trades(status="closed", mode=self.state.mode, per_page=50)
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

        # Bootstrap dynamic weights from saved state (in config_overrides)
        if self.dynamic_weights:
            try:
                raw_state = await self.repo.get_engine_state() or {}
                overrides = raw_state.get("config_overrides") or {}
                dw_state = overrides.get("dynamic_weights") if isinstance(overrides, dict) else None
                if dw_state:
                    self.dynamic_weights.from_state(dw_state)
                    self.scanner.confluence_engine.update_weights(
                        self.dynamic_weights.get_weights()
                    )
            except Exception as e:
                logger.warning("dynamic_weights_load_failed", error=str(e))

        self._running = True

    async def run(self) -> None:
        """Main event loop."""
        await self.startup()

        # Persist initial state immediately so the dashboard has data
        await self._persist_state()

        # --- LIVE MODE SAFETY WARNINGS ---
        is_live = self.config.trading_mode == "live"
        if is_live:
            logger.warning(
                "LIVE_TRADING_MODE",
                message="⚠️  LIVE TRADING WITH REAL MONEY ⚠️",
                mode="LIVE",
                account_type=self.config.account_type,
                leverage=self.config.leverage,
                margin_mode=getattr(self.config, "margin_mode", "isolated"),
                max_position_pct=self.config.max_position_pct,
                min_trade_usd=self.config.min_trade_usd,
                max_concurrent=self.config.max_concurrent_positions,
            )
            # Fetch and log actual exchange balance
            try:
                balance = await self.exchange.get_balance()
                usdt = balance.get("USDT", 0)
                logger.warning(
                    "live_exchange_balance",
                    exchange_usdt=f"{usdt:.2f}",
                    db_balance=f"{self.state.current_balance:.2f}",
                    open_positions=len(self.state.open_positions),
                )
                if usdt < self.config.min_trade_usd:
                    logger.critical(
                        "insufficient_live_balance",
                        exchange_usdt=f"{usdt:.2f}",
                        min_trade_usd=self.config.min_trade_usd,
                        hint="Exchange balance is below minimum trade size. "
                             "Deposit funds or reduce min_trade_usd.",
                    )
            except Exception as e:
                logger.warning("live_balance_check_failed", error=str(e)[:200])
        else:
            logger.info("paper_trading_mode", mode="PAPER")

        logger.info(
            "engine_started",
            mode=self.state.mode,
            account_type=self.config.account_type,
            leverage=self.config.leverage,
            balance=self.state.current_balance,
            positions=len(self.state.open_positions),
        )

        # Start flipped trader's independent scan loop (runs every 15 min)
        if self.flipped_trader.enabled:
            self._flipped_task = asyncio.create_task(self.flipped_trader.run_loop())
            self._background_tasks.add(self._flipped_task)
            self._flipped_task.add_done_callback(self._background_tasks.discard)
            logger.info("flipped_loop_spawned", mode=self.flipped_trader.mode_name)

        # Start custom trader's independent scan loop (runs every 20 min)
        if self.custom_trader.enabled:
            self._custom_task = asyncio.create_task(self.custom_trader.run_loop())
            self._background_tasks.add(self._custom_task)
            self._custom_task.add_done_callback(self._background_tasks.discard)
            logger.info(
                "custom_loop_spawned",
                mode=self.custom_trader.mode_name,
                scanning_active=self.custom_trader._scanning_active,
                flip_mode=self.custom_trader.flip_mode,
                flip_direction=self.custom_trader.flip_direction,
                margin_pct=self.custom_trader.max_position_pct,
            )

        # Restore main bot settings from DB state
        db_state = await self.repo.get_engine_state()
        if db_state:
            overrides = db_state.get("config_overrides", {}) or {}
            main_settings = overrides.get("main_bot_settings", {}) or {}
            if "scanning_active" in main_settings:
                self._scanning_active = bool(main_settings["scanning_active"])
            # Restore margin_pct and leverage from saved settings
            if "margin_pct" in main_settings:
                saved_margin = float(main_settings["margin_pct"])
                self.config.max_position_pct = saved_margin
                self.risk_manager.max_position_pct = saved_margin
                logger.info("main_bot_margin_restored", margin_pct=saved_margin)
            if "leverage" in main_settings:
                saved_leverage = int(main_settings["leverage"])
                self.config.leverage = saved_leverage
                self.risk_manager._leverage = saved_leverage
                if hasattr(self.exchange, "_leverage"):
                    self.exchange._leverage = saved_leverage
                logger.info("main_bot_leverage_restored", leverage=saved_leverage)

        # Start hyper-watchlist monitor loop (checks every 2.5 min on 5m candles)
        if self.watchlist_monitor:
            # Restore watchlist state from DB
            if db_state:
                wl_overrides = (db_state.get("config_overrides") or {})
                wl_data = wl_overrides.get("watchlist_monitor")
                if wl_data:
                    self.watchlist_monitor.restore_state(wl_data)
            self._watchlist_task = asyncio.create_task(self.watchlist_monitor.run_loop())
            self._background_tasks.add(self._watchlist_task)
            self._watchlist_task.add_done_callback(self._background_tasks.discard)
            logger.info(
                "watchlist_monitor_spawned",
                entries=len(self.watchlist_monitor.entries),
            )

        # Restore entry refiner state (main bot)
        if self.main_entry_refiner and db_state:
            refiner_overrides = (db_state.get("config_overrides") or {})
            refiner_data = refiner_overrides.get("main_entry_refiner")
            if refiner_data:
                self.main_entry_refiner.restore_state(refiner_data)
                logger.info(
                    "main_entry_refiner_restored",
                    queued=len(self.main_entry_refiner.queue),
                )

        # Restore consensus monitor state (main bot)
        if self.consensus_monitor and db_state:
            consensus_overrides = (db_state.get("config_overrides") or {})
            consensus_data = consensus_overrides.get("consensus_monitor")
            if consensus_data:
                self.consensus_monitor.restore_state(consensus_data)
                logger.info(
                    "consensus_monitor_restored",
                    queued=len(self.consensus_monitor.queue),
                )

        # Run an immediate scan on startup if scanning is active and no recent scan
        if self._scanning_active:
            # Check trading hours first — don't scan on weekends for stocks/commodities
            market_info = getattr(self.exchange, "market_info", None)
            if not self.trading_hours.should_scan(market_info):
                next_open = self.trading_hours.next_open(market_info)
                logger.info(
                    "startup_scan_skipped_market_closed",
                    market=self._market_name,
                    next_open=next_open.isoformat() if next_open else "unknown",
                )
            else:
                should_startup_scan = False
                if self.state.last_scan_time is None:
                    should_startup_scan = True
                else:
                    elapsed = (datetime.now(timezone.utc) - self.state.last_scan_time).total_seconds()
                    if elapsed > self.config.scan_interval_minutes * 60:
                        should_startup_scan = True

                if should_startup_scan:
                    logger.info("startup_scan", reason="no recent scan, running immediately")
                    try:
                        await self._primary_tick()
                    except Exception as e:
                        logger.error("startup_scan_failed", error=str(e))
        else:
            logger.info("main_bot_paused_at_startup", reason="waiting for user to click Start")

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
                    self.state.daily_trade_count = 0
                    self.state.last_scan_time = datetime.now(timezone.utc)
                    if self.flipped_trader.enabled:
                        self.flipped_trader.reset_daily()
                    logger.info(
                        "daily_reset",
                        daily_start_balance=self.state.daily_start_balance,
                        equity=equity,
                    )

                if tick_type == TickType.MONITOR:
                    # Always monitor open positions (even when scanning is paused)
                    await self._monitor_tick()
                    # Process graduated signals from hyper-watchlist
                    await self._process_watchlist_graduations()
                    # Process refined entries from post-sweep 5m monitoring
                    await self._process_refined_entries()
                    # Process graduated signals from consensus monitor
                    await self._process_consensus_graduations()
                    await self._persist_state()
                else:
                    # Skip scanning for new trades if bot is paused
                    if not self._scanning_active:
                        continue
                    # Check trading hours — skip scan if market is closed
                    market_info = getattr(self.exchange, "market_info", None)
                    if not self.trading_hours.should_scan(market_info):
                        next_open = self.trading_hours.next_open(market_info)
                        if next_open:
                            logger.info("market_closed_skipping_scan", market=self._market_name, next_open=next_open.isoformat())
                        continue
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

    def _spawn_background(self, coro) -> None:
        """Fire-and-forget an async task with cleanup on completion."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _run_postmortem(self, trade_id: str, symbol: str) -> None:
        """Background task: run LLM post-mortem and persist result."""
        async with self._postmortem_semaphore:
            try:
                # Fetch full trade record for rich context
                trade_records = await self.repo.get_trades_by_ids([trade_id])
                trade_record = trade_records[0] if trade_records else None
                if not trade_record:
                    return

                # Fetch partial exits if applicable
                partial_exits = None
                tp_tiers = trade_record.get("tp_tiers")
                if tp_tiers:
                    try:
                        partial_exits = await self.repo.get_partial_exits(trade_id)
                    except Exception:
                        pass

                result = await self.llm_analyst.analyze_closed_trade(
                    trade_data=trade_record,
                    partial_exits=partial_exits,
                )

                if not result.error:
                    await self.repo.update_trade(trade_id, {
                        "llm_postmortem": {
                            "summary": result.summary,
                            "what_worked": result.what_worked,
                            "what_failed": result.what_failed,
                            "entry_quality": result.entry_quality,
                            "exit_quality": result.exit_quality,
                            "lesson": result.lesson,
                            "trade_grade": result.trade_grade,
                            "latency_ms": round(result.latency_ms, 1),
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                            "analyzed_at": datetime.now(timezone.utc).isoformat(),
                        }
                    })
                    logger.info(
                        "llm_postmortem_complete",
                        trade_id=trade_id,
                        symbol=symbol,
                        grade=result.trade_grade,
                        latency_ms=round(result.latency_ms, 1),
                    )
                else:
                    logger.warning(
                        "llm_postmortem_failed",
                        trade_id=trade_id,
                        error=result.error,
                    )
            except Exception as e:
                logger.warning(
                    "llm_postmortem_error",
                    trade_id=trade_id,
                    error=str(e),
                )

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
                quality_filter=self.config.quality_filter,
            )
        except Exception as e:
            logger.error("pair_scan_failed", error=str(e))
            await self._persist_state()
            return

        if not pairs:
            logger.warning("no_tradeable_pairs")
            await self._persist_state()
            return

        # Scan for signals (exclude symbols already on the hyper-watchlist)
        watchlist_exclude = (
            self.watchlist_monitor.get_excluded_symbols()
            if self.watchlist_monitor
            else set()
        )
        signals = await self.scanner.scan(pairs, exclude=watchlist_exclude)

        # Promote near-misses to hyper-watchlist for fast 5m monitoring
        if self.watchlist_monitor and self.scanner.last_near_misses:
            promoted = 0
            for near_miss in self.scanner.last_near_misses:
                sig_type = "breakout" if near_miss.breakout_result is not None else "sweep"
                if self.watchlist_monitor.add_entry(near_miss, sig_type):
                    promoted += 1
            if promoted:
                logger.info(
                    "watchlist_near_misses_promoted",
                    promoted=promoted,
                    total_near_misses=len(self.scanner.last_near_misses),
                    watchlist_size=len(self.watchlist_monitor.entries),
                )

        # ── Market-level cross-reference filters ──
        # BTC macro gate, market breadth, signal persistence, correlation clustering
        self._scan_cycle += 1
        signals, mf_result = await self.market_filter.apply_all(
            signals=signals,
            candle_manager=self.candle_manager,
            scan_cycle=self._scan_cycle,
            open_positions=(
                self.portfolio.open_positions if self.portfolio else {}
            ),
        )
        if mf_result.signals_before != mf_result.signals_after:
            logger.info(
                "market_filter_applied",
                before=mf_result.signals_before,
                after=mf_result.signals_after,
                btc_trend=mf_result.btc_trend,
                breadth=mf_result.breadth_direction,
                blocked_btc=len(mf_result.blocked_by_btc_gate),
                blocked_breadth=len(mf_result.blocked_by_breadth),
            )

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

        # Read runtime toggles ONCE per tick (not per signal)
        state_dict = await self.repo.get_engine_state() or {}
        llm_runtime_enabled = state_dict.get("llm_enabled", self.config.llm_enabled)
        agent_runtime_enabled = state_dict.get("agent_enabled", self.config.agent_enabled)
        logger.info(
            "ai_toggle_check",
            llm_runtime_enabled=llm_runtime_enabled,
            agent_runtime_enabled=agent_runtime_enabled,
            has_llm=self.llm_analyst is not None,
            has_agent=self.agent_analyst is not None,
        )

        # Pre-fetch recent performance for AI context (once per tick)
        llm_perf_context: dict[str, Any] = {}
        if (llm_runtime_enabled and self.llm_analyst) or (agent_runtime_enabled and self.agent_analyst):
            try:
                recent_trades = await self.repo.get_trades(status="closed", mode=self.state.mode, per_page=20)
                if recent_trades:
                    wins = sum(1 for t in recent_trades if (t.get("pnl_usd") or 0) > 0)
                    llm_perf_context["recent_win_rate"] = round(wins / len(recent_trades) * 100, 1)
                    llm_perf_context["recent_trade_count"] = len(recent_trades)
                    rr_values = [float(t.get("risk_reward") or 0) for t in recent_trades if t.get("risk_reward")]
                    if rr_values:
                        llm_perf_context["recent_avg_rr"] = round(sum(rr_values) / len(rr_values), 2)
                    # Calculate current streak
                    streak = 0
                    streak_type = None
                    for t in recent_trades:
                        is_win = (t.get("pnl_usd") or 0) > 0
                        if streak_type is None:
                            streak_type = is_win
                            streak = 1
                        elif is_win == streak_type:
                            streak += 1
                        else:
                            break
                    if streak_type is True:
                        llm_perf_context["winning_streak"] = streak
                    elif streak_type is False:
                        llm_perf_context["losing_streak"] = streak
            except Exception as e:
                logger.debug("llm_perf_context_failed", error=str(e))

        # Execute entries for top signals
        signals_saved = 0
        trades_entered = 0
        refiner_queued = 0
        for signal in signals:
            try:
                position = None
                sig_type = "breakout" if signal.breakout_result is not None else "sweep"

                # --- AI Agent Early Decision (runs BEFORE refiner so agent controls the flow) ---
                agent_early_decision = None  # None = agent not active, let refiner decide
                agent_analysis_data = None
                if agent_runtime_enabled and self.agent_analyst:
                    # Pre-calculate SL/TP for agent context
                    pre_sl = self.order_executor._calculate_stop_loss(signal)
                    pre_tp = None
                    pre_rr = None
                    if pre_sl is not None:
                        pre_tp = self.order_executor._calculate_take_profit(signal, pre_sl)
                        sl_dist = abs(signal.entry_price - pre_sl)
                        if sl_dist > 0 and pre_tp is not None:
                            pre_rr = abs(pre_tp - signal.entry_price) / sl_dist

                    # Get sentiment early for agent context
                    early_sentiment = await self.sentiment_filter.get_sentiment(signal.symbol)
                    early_headlines = self.sentiment_filter.get_recent_headlines(signal.symbol)

                    ai_context = {
                        "sentiment_score": early_sentiment,
                        "adjusted_score": signal.score,
                        "active_threshold": active_threshold,
                        "sl_price": pre_sl,
                        "tp_price": pre_tp,
                        "rr_ratio": round(pre_rr, 2) if pre_rr else None,
                        "open_position_count": len(self.portfolio.open_positions),
                        "recent_headlines": early_headlines,
                        "ml_win_probability": None,
                        **llm_perf_context,
                    }

                    agent_result = await self.agent_analyst.analyze_signal(signal, ai_context)
                    agent_early_decision = agent_result.action

                    agent_analysis_data = {
                        "action": agent_result.action,
                        "confidence": agent_result.confidence,
                        "reasoning": agent_result.reasoning,
                        "market_regime": agent_result.market_regime,
                        "risk_assessment": agent_result.risk_assessment,
                        "suggested_entry": agent_result.suggested_entry,
                        "suggested_sl": agent_result.suggested_sl,
                        "suggested_tp": agent_result.suggested_tp,
                        "latency_ms": round(agent_result.latency_ms, 1),
                        "error": agent_result.error,
                        "tokens": agent_result.input_tokens + agent_result.output_tokens,
                    }

                    if agent_result.action == "SKIP":
                        logger.info(
                            "trade_skipped_by_agent",
                            symbol=signal.symbol,
                            confidence=agent_result.confidence,
                            risk=agent_result.risk_assessment,
                            reasoning=agent_result.reasoning[:120],
                        )
                        vol_24h = self.exchange.get_24h_volume(signal.symbol)
                        await self.repo.insert_signal(
                            {
                                "symbol": signal.symbol,
                                "direction": signal.direction or "none",
                                "score": signal.score,
                                "reasons": signal.reasons + [
                                    f"AGENT:SKIP(conf={agent_result.confidence:.0f},"
                                    f"risk={agent_result.risk_assessment})"
                                ],
                                "components": {
                                    "volume_24h": vol_24h,
                                    "test_group": "agent",
                                    "agent_analysis": agent_analysis_data,
                                    "signal_type": sig_type,
                                },
                                "current_price": signal.entry_price,
                                "acted_on": False,
                                "scan_cycle": cycle,
                            }
                        )
                        signals_saved += 1
                        continue

                    elif agent_result.action == "WAIT_PULLBACK":
                        # Agent wants to wait — queue in entry refiner
                        if (
                            self.main_entry_refiner
                            and signal.symbol not in self.portfolio.open_positions
                            and signal.symbol not in self.main_entry_refiner.get_queued_symbols()
                        ):
                            signal.original_1h_price = signal.entry_price
                            if agent_result.suggested_entry is not None:
                                signal.agent_target_entry = agent_result.suggested_entry
                            queued = self.main_entry_refiner.add(signal)
                            if queued:
                                refiner_queued += 1
                                logger.info(
                                    "agent_wait_pullback_queued",
                                    symbol=signal.symbol,
                                    target_entry=agent_result.suggested_entry,
                                    confidence=agent_result.confidence,
                                )
                                vol_24h = self.exchange.get_24h_volume(signal.symbol)
                                await self.repo.insert_signal(
                                    {
                                        "symbol": signal.symbol,
                                        "direction": signal.direction or "none",
                                        "score": signal.score,
                                        "reasons": signal.reasons + [
                                            f"AGENT:WAIT_PULLBACK(conf={agent_result.confidence:.0f},"
                                            f"target={agent_result.suggested_entry})"
                                        ],
                                        "components": {
                                            "volume_24h": vol_24h,
                                            "test_group": "agent",
                                            "agent_analysis": agent_analysis_data,
                                            "signal_type": sig_type,
                                        },
                                        "current_price": signal.entry_price,
                                        "acted_on": False,
                                        "scan_cycle": cycle,
                                    }
                                )
                                signals_saved += 1
                                continue
                        # If refiner unavailable/full, fall through to ENTER_NOW

                    # Agent says ENTER_NOW (or WAIT_PULLBACK fell through)
                    # Store overrides for later use in the execution path
                    logger.info(
                        "agent_enter_now",
                        symbol=signal.symbol,
                        confidence=agent_result.confidence,
                        regime=agent_result.market_regime,
                        risk=agent_result.risk_assessment,
                    )

                # --- OTE Entry Refinement (only when agent is NOT active) ---
                # When agent is active, it already decided whether to use the refiner.
                # When agent is inactive, use the original formula-based refiner logic.
                if agent_early_decision is None:
                    if (
                        self.main_entry_refiner
                        and signal.symbol not in self.portfolio.open_positions
                        and signal.symbol not in self.main_entry_refiner.get_queued_symbols()
                    ):
                        queued = False
                        signal.original_1h_price = signal.entry_price

                        if signal.sweep_result and signal.sweep_result.sweep_detected:
                            queued = self.main_entry_refiner.add(signal)
                        elif signal.breakout_result and signal.breakout_result.breakout_detected:
                            queued = self.main_entry_refiner.add_breakout(signal)

                        if queued:
                            refiner_queued += 1
                            vol_24h = self.exchange.get_24h_volume(signal.symbol)
                            await self.repo.insert_signal(
                                {
                                    "symbol": signal.symbol,
                                    "direction": signal.direction or "none",
                                    "score": signal.score,
                                    "reasons": signal.reasons + [f"QUEUED:entry_refiner({sig_type})"],
                                    "components": {"volume_24h": vol_24h, "signal_type": sig_type},
                                    "current_price": signal.entry_price,
                                    "acted_on": False,
                                    "scan_cycle": cycle,
                                }
                            )
                            signals_saved += 1
                            continue
                        # If refiner queue is full, fall through to immediate entry

                # --- Daily trade limit (Chill phase) ---
                if self.state.daily_trade_count >= self.config.max_daily_trades:
                    logger.info(
                        "daily_trade_limit_reached",
                        symbol=signal.symbol,
                        daily_trades=self.state.daily_trade_count,
                        max_daily=self.config.max_daily_trades,
                    )
                    break  # Stop processing more signals today

                # --- Sentiment filter (Strategy A) ---
                sentiment_score = await self.sentiment_filter.get_sentiment(signal.symbol)
                recent_headlines = self.sentiment_filter.get_recent_headlines(signal.symbol)

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
                            "components": {"volume_24h": vol_24h, "sentiment": sentiment_score, "signal_type": sig_type},
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
                            "components": {"volume_24h": vol_24h, "sentiment": sentiment_score, "signal_type": sig_type},
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
                            "components": {"volume_24h": vol_24h, "sentiment": sentiment_score, "signal_type": sig_type},
                            "current_price": signal.entry_price,
                            "acted_on": False,
                            "scan_cycle": cycle,
                        }
                    )
                    signals_saved += 1
                    continue

                # --- Funding Rate Gate: block crowded-direction signals ---
                if self.market_filter.check_funding_gate(signal):
                    vol_24h = self.exchange.get_24h_volume(signal.symbol)
                    await self.repo.insert_signal(
                        {
                            "symbol": signal.symbol,
                            "direction": signal.direction or "none",
                            "score": signal.score,
                            "reasons": signal.reasons + ["BLOCKED:funding_gate"],
                            "components": {"volume_24h": vol_24h, "signal_type": sig_type},
                            "current_price": signal.entry_price,
                            "acted_on": False,
                            "scan_cycle": cycle,
                        }
                    )
                    signals_saved += 1
                    continue

                # --- Market Consensus Check: portfolio + BTC alignment ---
                if self.consensus_monitor and signal.direction:
                    consensus_result = await self.consensus_monitor.compute_consensus(
                        signal=signal,
                        open_positions=self.portfolio.open_positions,
                        exchange=self.exchange,
                        candle_manager=self.candle_manager,
                    )
                    if consensus_result.applied and consensus_result.penalty > 0:
                        adjusted_score -= consensus_result.penalty
                        signal.reasons.append(
                            f"consensus_penalty=-{consensus_result.penalty:.0f} "
                            f"(portfolio={consensus_result.portfolio_bias}, "
                            f"btc={consensus_result.btc_trend})"
                        )
                        if adjusted_score < active_threshold:
                            # Score dropped below threshold — move to consensus monitor
                            if self.consensus_monitor.add(signal, consensus_result):
                                # Remove from other queues
                                if self.main_entry_refiner:
                                    self.main_entry_refiner.queue.pop(signal.symbol, None)
                                if self.watchlist_monitor:
                                    self.watchlist_monitor.remove_entry(signal.symbol)
                                vol_24h = self.exchange.get_24h_volume(signal.symbol)
                                await self.repo.insert_signal(
                                    {
                                        "symbol": signal.symbol,
                                        "direction": signal.direction or "none",
                                        "score": signal.score,
                                        "reasons": signal.reasons + [
                                            f"DEFERRED:consensus(adj={adjusted_score:.1f})"
                                        ],
                                        "components": {
                                            "volume_24h": vol_24h,
                                            "sentiment": sentiment_score,
                                            "adjusted_score": round(adjusted_score, 2),
                                            "consensus_penalty": consensus_result.penalty,
                                            "portfolio_bias": consensus_result.portfolio_bias,
                                            "btc_trend": consensus_result.btc_trend,
                                            "signal_type": sig_type,
                                        },
                                        "current_price": signal.entry_price,
                                        "acted_on": False,
                                        "scan_cycle": cycle,
                                    }
                                )
                                signals_saved += 1
                                continue
                            # If consensus queue is full, fall through to normal entry

                # --- AI Entry Agent / Legacy LLM Gate ---
                # Agent already ran in the early decision block above.
                # Here we only set up variables for downstream execution.
                llm_analysis_data = None
                sl_override = None
                tp_override = None

                if agent_early_decision is not None:
                    # Agent already decided — use its results (no duplicate API call)
                    test_group = "agent"
                    sl_override = agent_result.suggested_sl
                    tp_override = agent_result.suggested_tp
                elif llm_runtime_enabled:
                    test_group = self.split_test.assign_group(signal)
                else:
                    test_group = "control"

                # --- Legacy LLM Gate (Anthropic) — only when agent is NOT active ---
                if agent_early_decision is None and test_group == "llm" and self.llm_analyst:
                    # Pre-calculate SL/TP for LLM context
                    pre_sl = self.order_executor._calculate_stop_loss(signal)
                    pre_tp = None
                    pre_rr = None
                    if pre_sl is not None:
                        pre_tp = self.order_executor._calculate_take_profit(signal, pre_sl)
                        sl_dist = abs(signal.entry_price - pre_sl)
                        if sl_dist > 0 and pre_tp is not None:
                            pre_rr = abs(pre_tp - signal.entry_price) / sl_dist

                    ai_context = {
                        "sentiment_score": sentiment_score,
                        "adjusted_score": adjusted_score,
                        "active_threshold": active_threshold,
                        "sl_price": pre_sl,
                        "tp_price": pre_tp,
                        "rr_ratio": round(pre_rr, 2) if pre_rr else None,
                        "open_position_count": len(self.portfolio.open_positions),
                        "recent_headlines": recent_headlines,
                        "ml_win_probability": round((pattern_modifier + 1) / 2 * 100, 1) if pattern_modifier is not None else None,
                        **llm_perf_context,
                    }
                    llm_result = await self.llm_analyst.analyze_signal(signal, ai_context)

                    # If LLM errored, demote to control group
                    if llm_result.error and llm_result.error != "api_backoff":
                        test_group = "control"
                        logger.info(
                            "llm_error_demoted_to_control",
                            symbol=signal.symbol,
                            error=llm_result.error,
                        )
                    else:
                        llm_analysis_data = {
                            "approve": llm_result.approve,
                            "confidence": llm_result.confidence,
                            "reasoning": llm_result.reasoning,
                            "suggested_sl": llm_result.suggested_sl,
                            "suggested_tp": llm_result.suggested_tp,
                            "latency_ms": round(llm_result.latency_ms, 1),
                            "error": llm_result.error,
                            "input_tokens": llm_result.input_tokens,
                            "output_tokens": llm_result.output_tokens,
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
                                    "signal_type": sig_type,
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
                                        "signal_type": sig_type,
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
                            # Store confluence score on position for post-trade analysis
                            position.confluence_score = signal.score

                            # Tag trade with split test metadata
                            trade_record["test_group"] = test_group
                            # Note: llm_analysis and entry_headlines stored in signal_components,
                            # not trade record (columns don't exist in Supabase trades table)

                            # Save to DB
                            db_trade = await self.repo.insert_trade(trade_record)
                            if db_trade:
                                position.trade_id = db_trade.get("id", "")

                            # Update portfolio
                            self.portfolio.record_entry(position)
                            self.state.open_positions[signal.symbol] = position
                            trades_entered += 1
                            self.state.daily_trade_count += 1

                # Log the signal regardless of whether trade was executed
                vol_24h = self.exchange.get_24h_volume(signal.symbol)
                signal_components = {
                    "volume_24h": vol_24h,
                    "sentiment": sentiment_score,
                    "adjusted_score": round(adjusted_score, 2),
                    "test_group": test_group,
                    "signal_type": sig_type,
                }
                if llm_analysis_data:
                    signal_components["llm_analysis"] = llm_analysis_data
                if agent_analysis_data:
                    signal_components["agent_analysis"] = agent_analysis_data
                await self.repo.insert_signal(
                    {
                        "symbol": signal.symbol,
                        "direction": signal.direction or "none",
                        "score": signal.score,
                        "reasons": signal.reasons,
                        "components": signal_components,
                        "current_price": signal.entry_price,
                        "acted_on": position is not None,
                        "trade_id": position.trade_id if position and position.trade_id else None,
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
            refiner_queued=refiner_queued,
        )

    async def _process_refined_entries(self) -> None:
        """Process signals refined on 5m candles (post-sweep entry refinement).

        Runs every 60s (on MONITOR tick). Checks the entry refiner for signals
        that have confirmed a sweep reclaim on 5m, or have expired.
        Follows the same pipeline as watchlist graduations.
        """
        if not self.main_entry_refiner or not self.main_entry_refiner.queue:
            return

        ready_signals = await self.main_entry_refiner.check_all()
        for signal in ready_signals:
            # Skip if we already have a position in this symbol
            if signal.symbol in self.portfolio.open_positions:
                continue

            # Skip if in cooldown
            cooldown_until = self._reversal_cooldowns.get(signal.symbol)
            if cooldown_until and datetime.now(timezone.utc) < cooldown_until:
                continue

            # Daily trade limit
            if self.state.daily_trade_count >= self.config.max_daily_trades:
                continue

            # Risk validation
            try:
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
                    logger.info(
                        "refiner_signal_risk_rejected",
                        symbol=signal.symbol,
                        reason=validation.reason,
                    )
                    continue
            except Exception as e:
                logger.warning("refiner_risk_check_failed", error=str(e))
                continue

            # Execute the trade
            try:
                position, order_result, trade_record = (
                    await self.order_executor.execute_entry(
                        signal=signal,
                        current_balance=self.portfolio.current_balance,
                        mode=self.state.mode,
                    )
                )

                if position and trade_record:
                    position.confluence_score = signal.score
                    db_trade = await self.repo.insert_trade(trade_record)
                    if db_trade:
                        position.trade_id = db_trade.get("id", "")
                    self.portfolio.record_entry(position)
                    self.state.open_positions[signal.symbol] = position
                    self.state.daily_trade_count += 1

                    improvement = ""
                    if signal.original_1h_price > 0 and signal.refined_entry:
                        imp_pct = abs(
                            signal.entry_price - signal.original_1h_price
                        ) / signal.original_1h_price * 100
                        improvement = f"{imp_pct:.2f}%"

                    logger.info(
                        "refiner_trade_entered",
                        symbol=signal.symbol,
                        direction=signal.direction,
                        entry=position.entry_price,
                        sl=position.stop_loss,
                        tp=position.take_profit,
                        score=signal.score,
                        refined=signal.refined_entry,
                        original_1h_price=round(signal.original_1h_price, 6),
                        improvement_pct=improvement,
                        duration_seconds=round(signal.refinement_duration_seconds, 0),
                    )
            except Exception as e:
                logger.warning(
                    "refiner_entry_failed",
                    symbol=signal.symbol,
                    error=str(e),
                )

    async def _process_consensus_graduations(self) -> None:
        """Process signals that graduated from the consensus monitor.

        Runs every 60s (on MONITOR tick). Graduated signals cleared consensus
        (portfolio/BTC now agree) or had their direction flipped.
        """
        if not self.consensus_monitor or not self.consensus_monitor.queue:
            return

        ready_signals = await self.consensus_monitor.check_all(
            open_positions=self.portfolio.open_positions,
            exchange=self.exchange,
            candle_manager=self.candle_manager,
        )
        for signal in ready_signals:
            if signal.symbol in self.portfolio.open_positions:
                continue
            if self.state.daily_trade_count >= self.config.max_daily_trades:
                continue

            try:
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
                    logger.info(
                        "consensus_signal_risk_rejected",
                        symbol=signal.symbol,
                        reason=validation.reason,
                    )
                    continue
            except Exception as e:
                logger.warning("consensus_risk_check_failed", error=str(e))
                continue

            try:
                position, order_result, trade_record = (
                    await self.order_executor.execute_entry(
                        signal=signal,
                        current_balance=self.portfolio.current_balance,
                        mode=self.state.mode,
                    )
                )

                if position and trade_record:
                    position.confluence_score = signal.score
                    db_trade = await self.repo.insert_trade(trade_record)
                    if db_trade:
                        position.trade_id = db_trade.get("id", "")
                    self.portfolio.record_entry(position)
                    self.state.open_positions[signal.symbol] = position
                    self.state.daily_trade_count += 1

                    logger.info(
                        "consensus_trade_entered",
                        symbol=signal.symbol,
                        direction=signal.direction,
                        entry=position.entry_price,
                        sl=position.stop_loss,
                        tp=position.take_profit,
                        score=signal.score,
                    )
            except Exception as e:
                logger.warning(
                    "consensus_entry_failed",
                    symbol=signal.symbol,
                    error=str(e),
                )

    async def _process_watchlist_graduations(self) -> None:
        """Process signals that graduated from the hyper-watchlist.

        Runs every 60s (on MONITOR tick). Graduated signals are fed through
        the same execution pipeline as regular signals (risk validation,
        order execution, etc.).
        """
        if not self.watchlist_monitor:
            return

        while not self._watchlist_queue.empty():
            try:
                graduated = self._watchlist_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            logger.info(
                "watchlist_signal_processing",
                symbol=graduated.symbol,
                score=graduated.score,
                direction=graduated.direction,
                duration_seconds=graduated.watchlist_duration_seconds,
            )

            # Skip if we already have a position in this symbol
            if graduated.symbol in self.portfolio.open_positions:
                logger.info(
                    "watchlist_signal_skipped_open_position",
                    symbol=graduated.symbol,
                )
                continue

            # Skip if in cooldown
            cooldown_until = self._reversal_cooldowns.get(graduated.symbol)
            if cooldown_until and datetime.now(timezone.utc) < cooldown_until:
                logger.info(
                    "watchlist_signal_skipped_cooldown",
                    symbol=graduated.symbol,
                )
                continue

            # Daily trade limit check
            if self.state.daily_trade_count >= self.config.max_daily_trades:
                logger.info("watchlist_signal_skipped_daily_limit")
                continue

            # Risk validation (max concurrent, balance, exposure, etc.)
            try:
                total_exposure = sum(
                    pos.cost_usd for pos in self.portfolio.open_positions.values()
                )
                validation = self.risk_manager.validate_trade(
                    open_position_count=len(self.portfolio.open_positions),
                    open_position_symbols=set(self.portfolio.open_positions.keys()),
                    current_balance=self.portfolio.current_balance,
                    daily_start_balance=self.portfolio.daily_start_balance,
                    daily_pnl=self.portfolio.daily_pnl,
                    signal=graduated,
                    total_exposure_usd=total_exposure,
                )
                if not validation.allowed:
                    logger.info(
                        "watchlist_signal_risk_rejected",
                        symbol=graduated.symbol,
                        reason=validation.reason,
                    )
                    continue
            except Exception as e:
                logger.warning("watchlist_risk_check_failed", error=str(e))
                continue

            # --- AI Agent gate for watchlist signals ---
            agent_skip = False
            if self.config.agent_enabled and self.agent_analyst:
                try:
                    pre_sl = self.order_executor._calculate_stop_loss(graduated)
                    pre_tp = None
                    pre_rr = None
                    if pre_sl is not None:
                        pre_tp = self.order_executor._calculate_take_profit(graduated, pre_sl)
                        sl_dist = abs(graduated.entry_price - pre_sl)
                        if sl_dist > 0 and pre_tp is not None:
                            pre_rr = abs(pre_tp - graduated.entry_price) / sl_dist

                    wl_sentiment = await self.sentiment_filter.get_sentiment(graduated.symbol)
                    wl_headlines = self.sentiment_filter.get_recent_headlines(graduated.symbol)

                    wl_context = {
                        "sentiment_score": wl_sentiment,
                        "adjusted_score": graduated.score,
                        "active_threshold": 60.0,
                        "sl_price": pre_sl,
                        "tp_price": pre_tp,
                        "rr_ratio": round(pre_rr, 2) if pre_rr else None,
                        "open_position_count": len(self.portfolio.open_positions),
                        "recent_headlines": wl_headlines,
                        "ml_win_probability": None,
                    }

                    wl_agent_result = await self.agent_analyst.analyze_signal(graduated, wl_context)
                    logger.info(
                        "watchlist_agent_decision",
                        symbol=graduated.symbol,
                        action=wl_agent_result.action,
                        confidence=wl_agent_result.confidence,
                        risk=wl_agent_result.risk_assessment,
                        latency_ms=round(wl_agent_result.latency_ms, 1),
                        reasoning=wl_agent_result.reasoning[:120],
                    )

                    if wl_agent_result.action == "SKIP":
                        agent_skip = True
                        logger.info(
                            "watchlist_signal_skipped_by_agent",
                            symbol=graduated.symbol,
                            confidence=wl_agent_result.confidence,
                        )
                    elif wl_agent_result.action == "WAIT_PULLBACK":
                        # Agent says wait — don't enter now, let it expire or re-trigger
                        agent_skip = True
                        logger.info(
                            "watchlist_signal_deferred_by_agent",
                            symbol=graduated.symbol,
                            target_entry=wl_agent_result.suggested_entry,
                        )
                except Exception as e:
                    logger.warning("watchlist_agent_failed", symbol=graduated.symbol, error=str(e))
                    # On agent failure, proceed with the trade (fallback)

            if agent_skip:
                continue

            # Execute the trade
            try:
                position, order_result, trade_record = (
                    await self.order_executor.execute_entry(
                        signal=graduated,
                        current_balance=self.portfolio.current_balance,
                        mode=self.state.mode,
                    )
                )

                if position and order_result.filled:
                    self.portfolio.add_position(position)
                    self.state.daily_trade_count += 1
                    logger.info(
                        "watchlist_trade_entered",
                        symbol=graduated.symbol,
                        direction=graduated.direction,
                        entry=position.entry_price,
                        sl=position.stop_loss,
                        tp=position.take_profit,
                        score=graduated.score,
                        watchlist_duration=graduated.watchlist_duration_seconds,
                    )
            except Exception as e:
                logger.warning(
                    "watchlist_entry_failed",
                    symbol=graduated.symbol,
                    error=str(e),
                )

    async def _monitor_tick(self) -> None:
        """Check open positions for SL/TP/trailing stop + liquidation proximity."""
        # Non-crypto engines must NOT monitor main bot positions — those are crypto
        # symbols that only the crypto engine's Binance connector can handle.
        # Without this guard, stocks/commodities engines try to fetch crypto tickers
        # from Yahoo Finance → 500 errors + triple API call volume.
        has_main_positions = bool(self.portfolio.open_positions) and self._market_name == "crypto"
        has_flipped_positions = self.flipped_trader.enabled and bool(self.flipped_trader.positions)

        if not has_main_positions and not has_flipped_positions:
            return

        # --- Flipped shadow trader: monitor positions every tick (60s) ---
        if has_flipped_positions:
            try:
                await self.flipped_trader.monitor_positions()
            except Exception as e:
                logger.warning("flipped_monitor_failed", error=str(e))

        if not has_main_positions:
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
                # Throttle: only log same symbol's error once per 5 min
                now_err = datetime.now(timezone.utc)
                err_key = f"liq_{symbol}"
                last = self._error_last_logged.get(err_key)
                if last is None or (now_err - last).total_seconds() >= 300:
                    logger.warning("liq_check_failed", symbol=symbol, error=str(e))
                    self._error_last_logged[err_key] = now_err

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

        # Compute ATR values for open positions (1H candles)
        atr_values: dict[str, float] = {}
        for symbol in list(self.portfolio.open_positions.keys()):
            try:
                candles = await self.candle_manager.get_candles(symbol, "1h", limit=30)
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

                    # Progressive SL: move SL to previous TP level after each tier
                    if exit_signal.tier == 1 and self.config.move_sl_to_be_after_tp1:
                        # TP1 hit → move SL to breakeven (entry price)
                        position.stop_loss = position.entry_price
                        logger.info("sl_moved_to_breakeven", symbol=exit_signal.symbol,
                                    new_sl=position.entry_price)
                    elif exit_signal.tier >= 2 and position.tp_tiers:
                        # TP2+ hit → move SL to previous tier's price
                        prev_tier = next(
                            (t for t in position.tp_tiers if t.level == exit_signal.tier - 1),
                            None,
                        )
                        if prev_tier and prev_tier.price:
                            position.stop_loss = prev_tier.price
                            logger.info("sl_moved_to_prev_tp", symbol=exit_signal.symbol,
                                        tier=exit_signal.tier, new_sl=prev_tier.price)

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

                    # Record cooldown on stop-loss and stale exits
                    if exit_signal.reason in ("sl_hit", "stale_close"):
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
                            confluence_score=position.confluence_score,
                            holding_seconds=holding_seconds,
                        )
                    except Exception as e:
                        logger.warning("post_trade_analysis_failed", error=str(e))

                    # --- LLM Post-mortem (non-blocking background task) ---
                    if self.llm_analyst and position.trade_id:
                        self._spawn_background(
                            self._run_postmortem(position.trade_id, exit_signal.symbol)
                        )

                    # --- Adaptive threshold update (Strategy C) ---
                    self.adaptive_threshold.record_outcome(is_win=(total_pnl > 0))

                    # --- Dynamic weight update (Strategy D) ---
                    if self.dynamic_weights and position.trade_id:
                        try:
                            trade_data = await self.repo.get_trades_by_ids([position.trade_id])
                            if trade_data:
                                components = trade_data[0].get("confluence_components", {})
                                if components:
                                    self.dynamic_weights.record_outcome(components, is_win=(total_pnl > 0))
                                    self.scanner.confluence_engine.update_weights(
                                        self.dynamic_weights.get_weights()
                                    )
                        except Exception as e:
                            logger.debug("dynamic_weights_update_failed", error=str(e))

                    # Remove from state
                    self.state.open_positions.pop(exit_signal.symbol, None)

        # Sync state
        self.state.current_balance = self.portfolio.current_balance
        self.state.peak_balance = self.portfolio.peak_balance
        self.state.daily_pnl = self.portfolio.daily_pnl
        self.state.total_pnl = self.portfolio.total_pnl

        # Flipped monitoring now handled in _monitor_tick (every 60s)

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
                confluence_score=existing_position.confluence_score,
                holding_seconds=holding_seconds,
            )
        except Exception as e:
            logger.warning("reversal_post_analysis_failed", error=str(e))

        # LLM post-mortem for the reversed trade
        if self.llm_analyst and existing_position.trade_id:
            self._spawn_background(
                self._run_postmortem(existing_position.trade_id, symbol)
            )

        self.adaptive_threshold.record_outcome(is_win=(total_pnl > 0))

        # Dynamic weight update
        if self.dynamic_weights and existing_position.trade_id:
            try:
                trade_data = await self.repo.get_trades_by_ids([existing_position.trade_id])
                if trade_data:
                    components = trade_data[0].get("confluence_components", {})
                    if components:
                        self.dynamic_weights.record_outcome(components, is_win=(total_pnl > 0))
                        self.scanner.confluence_engine.update_weights(
                            self.dynamic_weights.get_weights()
                        )
            except Exception as e:
                logger.debug("dynamic_weights_reversal_update_failed", error=str(e))

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
        """Save engine state and portfolio snapshot to DB.

        Only the primary (crypto) engine writes full portfolio state.
        Non-crypto engines save their custom trader state independently
        via FlippedTrader._save_state() to avoid overwriting shared DB row.
        """
        try:
            # Non-crypto engines: only save custom_trader state, skip full portfolio write
            if self._market_name != "crypto":
                return

            state_dict = self.portfolio.to_state_dict(
                status=self.state.status,
                mode=self.state.mode,
                cycle_count=self.state.cycle_count,
            )
            # Read existing overrides first to preserve custom_trader and other engines' state
            existing = await self.repo.get_engine_state()
            overrides = {}
            if existing:
                overrides = existing.get("config_overrides") or {}
                if not isinstance(overrides, dict):
                    overrides = {}
                # Preserve dashboard-toggled fields that aren't in portfolio state
                for key in ("llm_enabled",):
                    if key in existing and key not in state_dict:
                        state_dict[key] = existing[key]

            # Update only the keys this engine manages (preserving custom_trader* keys etc.)
            if self.dynamic_weights:
                overrides["dynamic_weights"] = self.dynamic_weights.to_state()
            if self.flipped_trader.enabled:
                overrides["flipped_trader"] = self.flipped_trader.to_state_dict()
            # Persist hyper-watchlist state
            if self.watchlist_monitor:
                overrides["watchlist_monitor"] = self.watchlist_monitor.get_state()
            # Persist entry refiner state (main bot)
            if self.main_entry_refiner:
                overrides["main_entry_refiner"] = self.main_entry_refiner.get_state()
            # Persist consensus monitor state (main bot)
            if self.consensus_monitor:
                overrides["consensus_monitor"] = self.consensus_monitor.get_state()
            # Persist main bot scanning state
            main_settings = overrides.get("main_bot_settings", {})
            if not isinstance(main_settings, dict):
                main_settings = {}
            main_settings["scanning_active"] = self._scanning_active
            overrides["main_bot_settings"] = main_settings
            state_dict["config_overrides"] = overrides
            await self.repo.upsert_engine_state(state_dict)
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
                original_stop_loss=float(t.get("original_stop_loss") or t.get("stop_loss", 0)),
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

    async def _reconcile_exchange_positions(self) -> None:
        """Compare live exchange positions with DB/engine state on startup.

        This is critical for live trading safety:
        - Detects positions on the exchange that the bot doesn't know about
          (e.g. manual trades, failed exit orders, restart during order placement)
        - Detects positions in the DB that don't exist on the exchange
          (e.g. positions closed manually on Binance UI)
        - Logs warnings for any mismatch so the operator can investigate

        Only runs for exchanges that support fetch_all_positions() (BinanceFuturesClient).
        PaperExchange and data-only connectors skip this.
        """
        exchange = self.exchange
        # For PaperExchange, get the live exchange underneath
        if hasattr(exchange, "live"):
            exchange = exchange.live

        if not hasattr(exchange, "fetch_all_positions"):
            return

        try:
            live_positions = await exchange.fetch_all_positions()
        except Exception as e:
            logger.warning("exchange_reconciliation_failed", error=str(e)[:200])
            return

        # Build maps for comparison
        live_map: dict[str, dict] = {}
        for pos in live_positions:
            symbol = pos.get("symbol")
            if symbol:
                live_map[symbol] = pos

        db_symbols = set(self.state.open_positions.keys())
        live_symbols = set(live_map.keys())

        # Positions on exchange but NOT in our DB state
        orphaned_on_exchange = live_symbols - db_symbols
        for symbol in orphaned_on_exchange:
            pos = live_map[symbol]
            logger.warning(
                "exchange_position_not_in_db",
                symbol=symbol,
                side=pos.get("side"),
                contracts=pos.get("contracts"),
                entry_price=pos.get("entry_price"),
                unrealized_pnl=pos.get("unrealized_pnl"),
                action="MANUAL_REVIEW_REQUIRED",
                hint="Position exists on exchange but bot doesn't track it. "
                     "May be a manual trade or failed exit. Check Binance UI.",
            )

        # Positions in our DB state but NOT on exchange (exchange closed them)
        missing_from_exchange = db_symbols - live_symbols
        for symbol in missing_from_exchange:
            db_pos = self.state.open_positions[symbol]
            # Only flag if this is a futures position (spot positions aren't tracked as "positions")
            if db_pos.leverage > 1:
                logger.warning(
                    "db_position_not_on_exchange",
                    symbol=symbol,
                    direction=db_pos.direction,
                    entry_price=db_pos.entry_price,
                    quantity=db_pos.quantity,
                    action="POSITION_MAY_HAVE_BEEN_LIQUIDATED_OR_CLOSED_EXTERNALLY",
                    hint="Position tracked in DB but not found on exchange. "
                         "May have been liquidated or closed via Binance UI.",
                )

        # Positions in both — check for quantity mismatches
        common = db_symbols & live_symbols
        for symbol in common:
            db_pos = self.state.open_positions[symbol]
            live_pos = live_map[symbol]
            live_qty = live_pos.get("contracts", 0)
            db_qty = db_pos.quantity

            # Allow 1% tolerance for rounding
            if abs(live_qty - db_qty) / max(db_qty, 0.0001) > 0.01:
                logger.warning(
                    "position_quantity_mismatch",
                    symbol=symbol,
                    db_quantity=db_qty,
                    exchange_quantity=live_qty,
                    difference=abs(live_qty - db_qty),
                    action="QUANTITY_SYNC_NEEDED",
                    hint="Position quantity differs between DB and exchange. "
                         "Partial fill or manual adjustment may have occurred.",
                )

        # Log the balance from exchange for verification
        try:
            balance = await exchange.get_balance()
            usdt_balance = balance.get("USDT", 0)
            logger.info(
                "exchange_reconciliation_complete",
                live_positions=len(live_positions),
                db_positions=len(db_symbols),
                orphaned_on_exchange=len(orphaned_on_exchange),
                missing_from_exchange=len(missing_from_exchange),
                exchange_usdt_balance=f"{usdt_balance:.2f}",
                db_balance=f"{self.state.current_balance:.2f}",
            )
        except Exception:
            logger.info(
                "exchange_reconciliation_complete",
                live_positions=len(live_positions),
                db_positions=len(db_symbols),
                orphaned_on_exchange=len(orphaned_on_exchange),
                missing_from_exchange=len(missing_from_exchange),
            )

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("engine_shutting_down")
        self._running = False

        # Wait for pending post-mortems to complete
        if self._background_tasks:
            logger.info("waiting_for_background_tasks", count=len(self._background_tasks))
            _, pending = await asyncio.wait(self._background_tasks, timeout=30)
            for task in pending:
                task.cancel()

        # Check if DB was externally reset (e.g. data wipe) before persisting
        # If cycle_count in DB is 0 but our in-memory state is higher,
        # someone reset the DB — don't overwrite it.
        try:
            db_state = await self.repo.get_engine_state()
            db_cycle = db_state.get("cycle_count", 0) if db_state else 0
            if db_cycle == 0 and self.state.cycle_count > 0:
                logger.info(
                    "shutdown_skip_persist_db_was_reset",
                    db_cycle=db_cycle,
                    mem_cycle=self.state.cycle_count,
                )
            else:
                await self._persist_state()
        except Exception:
            # If we can't check, persist anyway to be safe
            await self._persist_state()

        await self.sentiment_filter.close()
        if self.llm_analyst:
            await self.llm_analyst.close()
        if self.agent_analyst:
            await self.agent_analyst.close()
        logger.info("engine_shutdown_complete")
