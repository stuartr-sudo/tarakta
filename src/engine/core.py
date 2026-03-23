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
from src.engine.scheduler import Scheduler, TickType
from src.engine.watchlist import WatchlistMonitor
from src.engine.state import EngineState
from src.exchange.models import Position, PullbackPlan, TakeProfitTier
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
from src.strategy.refiner_agent import RefinerMonitorAgent
from src.strategy.market_filter import MarketFilter
from src.strategy.scanner import AltcoinScanner
from src.strategy.sentiment import SentimentFilter
from src.strategy.trade_analyzer import TradeAnalyzer
from src.data.rag import TradeRAG
from src.strategy.lesson_generator import TradeLessonGenerator, format_lessons_for_prompt
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
        # Agent 3 (Position Manager) — always instantiate if API key exists
        # Runtime toggle via settings page controls whether it's wired into the monitor
        self.position_agent = None
        if config.openai_api_key:
            from src.strategy.position_agent import PositionManagerAgent
            self.position_agent = PositionManagerAgent(config)
            self.position_agent._repo = repo  # usage tracking

        self.position_monitor = PositionMonitor(
            trailing_activation_rr=config.trailing_activation_rr,
            trailing_atr_multiplier=config.trailing_atr_multiplier,
            breakeven_activation_rr=config.breakeven_activation_rr,
            max_hold_hours=config.max_hold_hours,
            stale_close_below_rr=config.stale_close_below_rr,
            position_agent=self.position_agent,
            position_agent_interval_minutes=getattr(
                config, "position_agent_check_interval_minutes", 5.0
            ),
            repo=repo,
        )
        # Req 8: wire shadow mode from config
        self.position_monitor._agent3_shadow_mode = getattr(
            config, "agent3_shadow_mode", False
        )
        self.scanner = AltcoinScanner(candle_manager, config)

        # Footprint analyzer — order flow confirmation gate
        self.footprint_analyzer = None
        if config.footprint_enabled:
            from src.strategy.footprint import FootprintAnalyzer
            self.footprint_analyzer = FootprintAnalyzer(
                min_delta_pct=config.footprint_min_delta_pct,
                absorption_threshold=config.footprint_absorption_threshold,
                min_confidence=config.footprint_min_confidence,
            )
            logger.info("footprint_analyzer_ready")

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

        # Symbol entry lock — prevents duplicate entries across async execution gaps
        self._entering_symbols: set[str] = set()

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

        # Agent 2 — Refiner Monitor (tactical entry timing)
        # Always create if API key exists so runtime toggle works instantly
        self.refiner_agent: RefinerMonitorAgent | None = None
        if config.openai_api_key:
            self.refiner_agent = RefinerMonitorAgent(config)
            self.refiner_agent._repo = repo  # usage tracking
            logger.info(
                "refiner_agent_created",
                model=config.refiner_agent_model,
                enabled=getattr(config, "refiner_agent_enabled", False),
                check_interval=f"{getattr(config, 'refiner_agent_check_interval_minutes', 5.0)}min",
            )

        # Post-sweep entry refinement — 5m monitoring for better entries (main bot)
        # Only pass Agent 2 to refiner if enabled in config (runtime toggle syncs later)
        _initial_refiner_agent = (
            self.refiner_agent
            if getattr(config, "refiner_agent_enabled", False)
            else None
        )
        self.main_entry_refiner: EntryRefiner | None = (
            EntryRefiner(
                candle_manager=candle_manager,
                config=config,
                refiner_agent=_initial_refiner_agent,
                market_filter=self.market_filter,
                exchange=exchange,
            )
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

        # RAG Knowledge Base — trade history retrieval for agents
        self.trade_rag: TradeRAG | None = None
        if getattr(config, "rag_enabled", False) and config.openai_api_key:
            from src.data.db import Database as _DB
            self.trade_rag = TradeRAG(repo.db, config.openai_api_key)
            logger.info("rag_knowledge_base_ready")

        # Trade Lesson Generator — AI post-mortem after every closed trade
        self.lesson_generator: TradeLessonGenerator | None = None
        if config.openai_api_key:
            self.lesson_generator = TradeLessonGenerator(config, repo)
            logger.info("lesson_generator_ready")

        # AI entry agent
        self.agent_analyst: AgentEntryAnalyst | None = None
        if config.openai_api_key:
            self.agent_analyst = AgentEntryAnalyst(config)
            self.agent_analyst._repo = repo  # usage tracking
            logger.info(
                "agent_analyst_ready",
                model=config.agent_model,
                min_score=config.agent_min_score,
            )

        self.state: EngineState | None = None
        self.portfolio: PortfolioTracker | None = None
        self._running = False
        self._scanning_active = False  # Main bot starts PAUSED — user clicks Start
        # Throttle repeated error logs (e.g. liq_check_failed for same symbol)
        self._error_last_logged: dict[str, datetime] = {}
        # Cache current prices from monitor tick for accurate drawdown calculation
        self._last_monitor_prices: dict[str, float] = {}

        # Dynamic weight optimizer
        self.dynamic_weights: DynamicWeightOptimizer | None = None
        if config.dynamic_weights_enabled:
            self.dynamic_weights = DynamicWeightOptimizer(WEIGHTS)
            logger.info("dynamic_weights_enabled")

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
        max_concurrent: int | None = None,
        max_risk_pct: float | None = None,
        max_daily_drawdown: float | None = None,
        entry_threshold: float | None = None,
        min_rr_ratio: float | None = None,
        max_hold_hours: float | None = None,
        circuit_breaker_pct: float | None = None,
        max_sl_pct: float | None = None,
        cooldown_hours: float | None = None,
        max_exposure_pct: float | None = None,
        # Weekly cycle settings
        monday_manipulation_penalty: float | None = None,
        monday_manipulation_hours: float | None = None,
        midweek_reversal_bonus: float | None = None,
        midweek_reversal_delay_hours: float | None = None,
        weekly_cycle_enabled: bool | None = None,
    ) -> None:
        """Update trading settings at runtime (takes effect on next trade/cycle)."""
        if margin_pct is not None:
            self.config.max_position_pct = margin_pct
            self.risk_manager.max_position_pct = margin_pct
            logger.info("setting_updated", key="margin_pct", value=margin_pct)
        if leverage is not None:
            self.config.leverage = leverage
            self.risk_manager._leverage = leverage
            if hasattr(self.exchange, "_leverage"):
                self.exchange._leverage = leverage
            logger.info("setting_updated", key="leverage", value=leverage)
        if max_concurrent is not None:
            self.config.max_concurrent = max_concurrent
            self.risk_manager.max_concurrent = max_concurrent
            logger.info("setting_updated", key="max_concurrent", value=max_concurrent)
        if max_risk_pct is not None:
            self.config.max_risk_pct = max_risk_pct
            self.risk_manager.max_risk_pct = max_risk_pct
            logger.info("setting_updated", key="max_risk_pct", value=max_risk_pct)
        if max_daily_drawdown is not None:
            self.config.max_daily_drawdown = max_daily_drawdown
            self.risk_manager.max_daily_drawdown = max_daily_drawdown
            self.circuit_breaker.daily_limit = max_daily_drawdown
            logger.info("setting_updated", key="max_daily_drawdown", value=max_daily_drawdown)
        if entry_threshold is not None:
            self.config.entry_threshold = entry_threshold
            self.adaptive_threshold.base_threshold = entry_threshold
            logger.info("setting_updated", key="entry_threshold", value=entry_threshold)
        if min_rr_ratio is not None:
            self.config.min_rr_ratio = min_rr_ratio
            self.risk_manager.min_rr_ratio = min_rr_ratio
            logger.info("setting_updated", key="min_rr_ratio", value=min_rr_ratio)
        if max_hold_hours is not None:
            self.config.max_hold_hours = max_hold_hours
            self.position_monitor.max_hold_hours = max_hold_hours
            logger.info("setting_updated", key="max_hold_hours", value=max_hold_hours)
        if circuit_breaker_pct is not None:
            self.config.circuit_breaker_pct = circuit_breaker_pct
            self.circuit_breaker.total_limit = circuit_breaker_pct
            logger.info("setting_updated", key="circuit_breaker_pct", value=circuit_breaker_pct)
        if max_sl_pct is not None:
            self.config.max_sl_pct = max_sl_pct
            logger.info("setting_updated", key="max_sl_pct", value=max_sl_pct)
        if cooldown_hours is not None:
            self.config.cooldown_hours = cooldown_hours
            self.risk_manager.cooldown_hours = cooldown_hours
            logger.info("setting_updated", key="cooldown_hours", value=cooldown_hours)
        if max_exposure_pct is not None:
            self.config.max_exposure_pct = max_exposure_pct
            self.risk_manager.max_exposure_pct = max_exposure_pct
            logger.info("setting_updated", key="max_exposure_pct", value=max_exposure_pct)
        # Weekly cycle settings
        if monday_manipulation_penalty is not None:
            self.config.monday_manipulation_penalty = monday_manipulation_penalty
            self.scanner.weekly_cycle.monday_penalty = monday_manipulation_penalty
            logger.info("setting_updated", key="monday_manipulation_penalty", value=monday_manipulation_penalty)
        if monday_manipulation_hours is not None:
            self.config.monday_manipulation_hours = monday_manipulation_hours
            self.scanner.weekly_cycle.monday_hours = monday_manipulation_hours
            logger.info("setting_updated", key="monday_manipulation_hours", value=monday_manipulation_hours)
        if midweek_reversal_bonus is not None:
            self.config.midweek_reversal_bonus = midweek_reversal_bonus
            self.scanner.weekly_cycle.midweek_bonus = midweek_reversal_bonus
            logger.info("setting_updated", key="midweek_reversal_bonus", value=midweek_reversal_bonus)
        if midweek_reversal_delay_hours is not None:
            self.config.midweek_reversal_delay_hours = midweek_reversal_delay_hours
            self.scanner.weekly_cycle.midweek_delay_hours = midweek_reversal_delay_hours
            logger.info("setting_updated", key="midweek_reversal_delay_hours", value=midweek_reversal_delay_hours)
        if weekly_cycle_enabled is not None:
            self.config.weekly_cycle_enabled = weekly_cycle_enabled
            self.scanner.weekly_cycle_enabled = weekly_cycle_enabled
            logger.info("setting_updated", key="weekly_cycle_enabled", value=weekly_cycle_enabled)

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

        # Restore agent models from DB overrides (saved via Settings page)
        if saved:
            overrides = saved.get("config_overrides") or {}
            if isinstance(overrides, dict):
                agent_models = overrides.get("agent_models") or {}
                from src.strategy.llm_client import MODEL_PRICING
                if self.agent_analyst and agent_models.get("agent1"):
                    saved_model = agent_models["agent1"]
                    if saved_model in MODEL_PRICING:
                        self.agent_analyst.set_model(saved_model)
                        logger.info("agent1_model_restored", model=saved_model)
                    else:
                        logger.warning("agent1_model_stale", saved=saved_model, using=self.config.agent_model)
                elif self.agent_analyst:
                    logger.info("agent1_model_default", model=self.config.agent_model)
                if self.refiner_agent and agent_models.get("agent2"):
                    saved_model = agent_models["agent2"]
                    if saved_model in MODEL_PRICING:
                        self.refiner_agent.set_model(saved_model)
                        logger.info("agent2_model_restored", model=saved_model)
                    else:
                        logger.warning("agent2_model_stale", saved=saved_model, using=self.config.refiner_agent_model)
                elif self.refiner_agent:
                    logger.info("agent2_model_default", model=self.config.refiner_agent_model)
                if self.position_agent and agent_models.get("agent3"):
                    saved_model = agent_models["agent3"]
                    if saved_model in MODEL_PRICING:
                        self.position_agent._model = saved_model
                        self.position_agent._api_key = self.config.openai_api_key
                        logger.info("agent3_model_restored", model=saved_model)
                    else:
                        logger.warning("agent3_model_stale", saved=saved_model, using=self.config.position_agent_model)

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
        stats = await self.repo.get_trade_stats(mode=self.state.mode)
        db_total_pnl = stats["total_pnl"]
        db_daily_pnl = await self.repo.get_daily_realized_pnl(mode=self.state.mode)

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

        # Reconcile current_balance from ground truth.
        # correct_cash = initial_balance + realized_pnl - margin_for_open_positions
        # The stored current_balance can drift due to accounting bugs.
        deployed_margin = 0.0
        for pos in self.portfolio.open_positions.values():
            if pos.leverage > 1:
                deployed_margin += pos.margin_used or (pos.cost_usd / pos.leverage)
            else:
                deployed_margin += pos.cost_usd
        correct_cash = self.config.initial_balance + db_total_pnl - deployed_margin
        if abs(correct_cash - self.portfolio.current_balance) > 0.01:
            logger.warning(
                "current_balance_reconciled",
                old=round(self.portfolio.current_balance, 2),
                new=round(correct_cash, 2),
                initial=self.config.initial_balance,
                realized_pnl=db_total_pnl,
                deployed_margin=round(deployed_margin, 2),
            )
            self.portfolio.current_balance = correct_cash
            self.state.current_balance = correct_cash

        # Sanitise peak_balance: correct_equity = correct_cash + deployed_margin
        # = initial_balance + realized_pnl. Peak can never exceed this (plus
        # a small tolerance for unrealized gains).
        correct_equity = correct_cash + deployed_margin  # = initial + realized_pnl
        if self.portfolio.peak_balance > correct_equity * 1.05:
            logger.warning(
                "peak_balance_clamped",
                old_peak=self.portfolio.peak_balance,
                new_peak=correct_equity,
                reason="inflated by previous equity bug",
            )
            self.portfolio.peak_balance = correct_equity
            self.state.peak_balance = correct_equity

        # Restore PaperExchange internal position tracking after restart
        # Without this, partial exits would create phantom positions instead of closing
        if hasattr(self.exchange, "restore_positions"):
            self.exchange.restore_positions(self.state.open_positions)

        # Live exchange reconciliation — compare Binance positions vs DB state
        # This catches orphaned exchange positions (e.g. manual trades, failed exits)
        await self._reconcile_exchange_positions()

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
                max_concurrent=self.config.max_concurrent,
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
                logger.critical(
                    "LIVE_PREFLIGHT_FAILED",
                    error=str(e)[:200],
                    hint=(
                        "Cannot reach Binance authenticated endpoints. "
                        "If you moved regions (e.g. iad → ams) you MUST update "
                        "your Binance API key IP whitelist to include the new "
                        "server IP. Live trading will NOT work until this is fixed."
                    ),
                )
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
            # Restore all other runtime settings via update_settings()
            _restore_keys = [
                "max_concurrent", "max_risk_pct", "max_daily_drawdown",
                "entry_threshold", "min_rr_ratio", "max_hold_hours",
                "circuit_breaker_pct", "max_sl_pct", "cooldown_hours",
                "max_exposure_pct", "monday_manipulation_penalty",
                "monday_manipulation_hours", "midweek_reversal_bonus",
                "midweek_reversal_delay_hours", "weekly_cycle_enabled",
            ]
            restore_kwargs = {k: main_settings[k] for k in _restore_keys if k in main_settings}
            if restore_kwargs:
                self.update_settings(**restore_kwargs)
                logger.info("runtime_settings_restored", keys=list(restore_kwargs.keys()))

        # Start hyper-watchlist monitor loop (checks every 2.5 min on 5m candles)
        # NOTE: Watchlist is NOT restored across restarts — clean slate every time.
        # Restored entries lose critical signal data (sweep_result, agent_context)
        # and can graduate as hollow placeholders, leading to blind trades.
        if self.watchlist_monitor:
            self._watchlist_task = asyncio.create_task(self.watchlist_monitor.run_loop())
            self._background_tasks.add(self._watchlist_task)
            self._watchlist_task.add_done_callback(self._background_tasks.discard)
            logger.info(
                "watchlist_monitor_spawned",
                entries=len(self.watchlist_monitor.entries),
            )

        # Do NOT restore entry refiner queue on restart.
        # Pending signals are stale after a restart — the market has moved on.
        # Fresh signals will come through naturally from the scanner.
        # Only open positions (managed by portfolio) survive restarts.
        if self.main_entry_refiner:
            logger.info("main_entry_refiner_fresh_start", msg="Queue starts empty on restart")

        # RAG backfill — ingest recent closed trades into knowledge base
        if self.trade_rag and getattr(self.config, "rag_backfill_on_startup", True):
            try:
                ingested = await self.trade_rag.backfill_trades(limit=200)
                logger.info("rag_startup_backfill", ingested=ingested)
            except Exception as e:
                logger.warning("rag_backfill_failed", error=str(e)[:100])

        # NOTE: Consensus monitor is NOT restored across restarts — clean slate.
        # Same reasoning as watchlist: restored entries use hollow placeholder signals.

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
                    logger.info(
                        "daily_reset",
                        daily_start_balance=self.state.daily_start_balance,
                        equity=equity,
                    )
                    # Trigger daily advisor analysis in background
                    asyncio.create_task(self._maybe_run_advisor())

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
                sig_type = "sweep"
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
        agent_runtime_enabled = state_dict.get("agent_enabled", self.config.agent_enabled)
        logger.info(
            "ai_toggle_check",
            agent_runtime_enabled=agent_runtime_enabled,
            has_agent=self.agent_analyst is not None,
        )

        # Pre-fetch recent performance for AI context (once per tick)
        ai_perf_context: dict[str, Any] = {}
        if agent_runtime_enabled and self.agent_analyst:
            try:
                recent_trades = await self.repo.get_trades(status="closed", mode=self.state.mode, per_page=20)
                if recent_trades:
                    wins = sum(1 for t in recent_trades if (t.get("pnl_usd") or 0) > 0)
                    ai_perf_context["recent_win_rate"] = round(wins / len(recent_trades) * 100, 1)
                    ai_perf_context["recent_trade_count"] = len(recent_trades)
                    rr_values = [float(t.get("risk_reward") or 0) for t in recent_trades if t.get("risk_reward")]
                    if rr_values:
                        ai_perf_context["recent_avg_rr"] = round(sum(rr_values) / len(rr_values), 2)
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
                        ai_perf_context["winning_streak"] = streak
                    elif streak_type is False:
                        ai_perf_context["losing_streak"] = streak
            except Exception as e:
                logger.debug("ai_perf_context_failed", error=str(e))

        # Execute entries for top signals
        signals_saved = 0
        trades_entered = 0
        refiner_queued = 0
        for signal in signals:
            try:
                position = None
                sig_type = "sweep"

                # --- AI Agent Early Decision (runs BEFORE refiner so agent controls the flow) ---
                agent_early_decision = None  # None = agent not active, let refiner decide
                agent_analysis_data = None

                # Skip Agent 1 entirely if this symbol is already queued for Agent 2.
                # Agent 1's job is done once it says WAIT_PULLBACK — Agent 2 monitors from here.
                # This prevents wasting GPT-5 calls (and timeouts) on repeat scanner detections.
                if (
                    self.main_entry_refiner
                    and signal.symbol in self.main_entry_refiner.get_queued_symbols()
                ):
                    logger.info(
                        "agent1_skipped_already_in_refiner",
                        symbol=signal.symbol,
                    )
                    continue

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

                    # Calculate Fibonacci retracement levels from sweep displacement
                    if signal.sweep_result and signal.sweep_result.sweep_detected and signal.sweep_result.sweep_level:
                        sweep_level = signal.sweep_result.sweep_level
                        current = signal.entry_price
                        if signal.direction in ("bullish", "long") and current > sweep_level:
                            # Bullish: displacement up from sweep low → pullback retraces downward
                            disp_low, disp_high = sweep_level, current
                            span = disp_high - disp_low
                            signal.fibonacci_levels = {
                                "displacement_low": round(disp_low, 8),
                                "displacement_high": round(disp_high, 8),
                                "fib_50": round(disp_high - span * 0.50, 8),
                                "fib_618": round(disp_high - span * 0.618, 8),
                                "fib_786": round(disp_high - span * 0.786, 8),
                            }
                        elif signal.direction in ("bearish", "short") and current < sweep_level:
                            # Bearish: displacement down from sweep high → pullback retraces upward
                            disp_low, disp_high = current, sweep_level
                            span = disp_high - disp_low
                            signal.fibonacci_levels = {
                                "displacement_low": round(disp_low, 8),
                                "displacement_high": round(disp_high, 8),
                                "fib_50": round(disp_low + span * 0.50, 8),
                                "fib_618": round(disp_low + span * 0.618, 8),
                                "fib_786": round(disp_low + span * 0.786, 8),
                            }

                    # Fetch per-symbol trade history for Agent 1 feedback loop
                    symbol_history = []
                    if getattr(self.config, "symbol_history_enabled", True):
                        try:
                            symbol_history = await self.repo.get_recent_trades_for_symbol(
                                signal.symbol, limit=5, mode=self.state.mode
                            )
                            # No-lookahead: exclude trades closed < 5 min ago (Req 5)
                            if symbol_history:
                                cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
                                symbol_history = [
                                    t for t in symbol_history
                                    if datetime.fromisoformat(t.get("exit_time", "2000-01-01T00:00:00+00:00")) < cutoff
                                ]
                        except Exception:
                            pass  # Non-critical — agent works without history

                    # Attach trade history to signal so Agent 2 can also access it
                    signal._symbol_history = symbol_history

                    # RAG knowledge retrieval — similar past trades
                    rag_context = ""
                    if self.trade_rag:
                        try:
                            sweep_type = signal.sweep_result.sweep_type if signal.sweep_result else ""
                            rag_results = await self.trade_rag.retrieve_for_symbol(
                                signal.symbol,
                                direction=signal.direction,
                                setup_context=f"sweep {sweep_type}",
                                k=getattr(self.config, "rag_max_results", 5),
                            )
                            rag_context = self.trade_rag.format_context(rag_results)
                        except Exception:
                            pass

                    # Attach RAG context to signal so Agent 2 can also access it
                    signal._rag_context = rag_context

                    # Retrieve learned lessons for Agent 1
                    lessons_context = ""
                    try:
                        lessons_context = await format_lessons_for_prompt(
                            self.repo, "agent1", symbol=signal.symbol, max_lessons=8,
                        )
                    except Exception:
                        pass

                    # Attach lessons to signal so Agent 2 can also access them
                    signal._lessons_context = lessons_context

                    # Fetch advisor insights for Agent 1 context
                    advisor_insights_text = ""
                    try:
                        from src.advisor.insights import get_recent_insights, format_insights_for_agent
                        insights = await get_recent_insights(self.repo.db, self.config.instance_id)
                        advisor_insights_text = format_insights_for_agent(insights)
                    except Exception:
                        pass  # Advisor insights are optional

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
                        "symbol_history": symbol_history,
                        "rag_context": rag_context,
                        "lessons_context": lessons_context,
                        "advisor_insights": advisor_insights_text,
                        **ai_perf_context,
                    }

                    agent_result = await self.agent_analyst.analyze_signal(signal, ai_context)
                    agent_early_decision = agent_result.action

                    # ── Agent 1 is the direction authority ──
                    # If Agent 1 chose a direction, override the scanner's suggestion
                    new_direction = signal.direction or ""
                    if agent_result.direction in ("LONG", "SHORT"):
                        new_direction = "bullish" if agent_result.direction == "LONG" else "bearish"
                        if new_direction != signal.direction:
                            logger.info(
                                "agent1_direction_override",
                                symbol=signal.symbol,
                                scanner_direction=signal.direction,
                                agent_direction=new_direction,
                            )
                            signal.direction = new_direction
                            # Re-compute SL/TP with the flipped direction
                            pre_sl = self.order_executor._calculate_stop_loss(signal)
                            pre_tp = None
                            pre_rr = None
                            if pre_sl is not None:
                                pre_tp = self.order_executor._calculate_take_profit(signal, pre_sl)
                                sl_dist = abs(signal.entry_price - pre_sl)
                                if sl_dist > 0 and pre_tp is not None:
                                    pre_rr = abs(pre_tp - signal.entry_price) / sl_dist

                    # ── SL/TP Priority: Agent 1 is primary, risk_manager is fallback ──
                    agent_sl = agent_result.suggested_sl
                    agent_tp = agent_result.suggested_tp

                    if agent_sl and agent_sl > 0:
                        sl_valid = True
                        if new_direction == "bullish" and agent_sl >= signal.entry_price:
                            sl_valid = False
                        elif new_direction == "bearish" and agent_sl <= signal.entry_price:
                            sl_valid = False
                        elif signal.entry_price > 0:
                            sl_dist = abs(signal.entry_price - agent_sl) / signal.entry_price
                            if sl_dist > self.config.max_sl_pct:
                                sl_valid = False
                        if sl_valid:
                            pre_sl = agent_sl

                    if agent_tp and agent_tp > 0:
                        tp_valid = True
                        if new_direction == "bullish" and agent_tp <= signal.entry_price:
                            tp_valid = False
                        elif new_direction == "bearish" and agent_tp >= signal.entry_price:
                            tp_valid = False
                        if tp_valid:
                            pre_tp = agent_tp

                    agent_analysis_data = {
                        "action": agent_result.action,
                        "direction": agent_result.direction,
                        "confidence": agent_result.confidence,
                        "reasoning": agent_result.reasoning,
                        "market_regime": agent_result.market_regime,
                        "risk_assessment": agent_result.risk_assessment,
                        "suggested_entry": agent_result.suggested_entry,
                        "entry_zone_high": agent_result.entry_zone_high,
                        "entry_zone_low": agent_result.entry_zone_low,
                        "suggested_sl": agent_result.suggested_sl,
                        "suggested_tp": agent_result.suggested_tp,
                        "must_reach_price": agent_result.must_reach_price,
                        "invalidation_level": agent_result.invalidation_level,
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
                        # Agent wants to wait — queue in entry refiner with formal PullbackPlan
                        if (
                            self.main_entry_refiner
                            and signal.symbol not in self.portfolio.open_positions
                            and signal.symbol not in self.main_entry_refiner.get_queued_symbols()
                        ):
                            signal.original_1h_price = signal.entry_price
                            if agent_result.suggested_entry is not None:
                                signal.agent_target_entry = agent_result.suggested_entry
                            if agent_result.entry_zone_high is not None:
                                signal.agent_entry_zone_high = agent_result.entry_zone_high
                            if agent_result.entry_zone_low is not None:
                                signal.agent_entry_zone_low = agent_result.entry_zone_low

                            # ── Fail-safe: reject if zone is missing or inverted ──
                            zone_h = agent_result.entry_zone_high
                            zone_l = agent_result.entry_zone_low
                            if not zone_h or not zone_l or zone_h <= zone_l or zone_h <= 0:
                                logger.warning(
                                    "wait_pullback_rejected_invalid_zone",
                                    symbol=signal.symbol,
                                    zone_high=zone_h,
                                    zone_low=zone_l,
                                    reason="zone missing, inverted, or zero",
                                )
                                continue

                            # ── Build PullbackPlan ──
                            now = datetime.now(timezone.utc)
                            expiry_seconds = self.config.pullback_valid_candles * 5 * 60  # candles × 5m
                            # Invalidation level: Agent 1 primary, sweep level fallback
                            invalidation = 0.0
                            if agent_result.invalidation_level and agent_result.invalidation_level > 0:
                                invalidation = agent_result.invalidation_level
                            elif signal.sweep_result and signal.sweep_result.sweep_detected:
                                invalidation = signal.sweep_result.sweep_level

                            # Use sweep_direction for invalidation logic (not signal.direction
                            # which may have been flipped by HTF override)
                            sweep_dir = (
                                signal.sweep_result.sweep_direction
                                if signal.sweep_result and signal.sweep_result.sweep_direction
                                else ("swing_low" if signal.direction == "bullish" else "swing_high" if signal.direction == "bearish" else "")
                            )
                            plan = PullbackPlan(
                                zone_low=zone_l,
                                zone_high=zone_h,
                                created_at=now,
                                expires_at=now + timedelta(seconds=expiry_seconds),
                                invalidation_level=invalidation,
                                max_chase_bps=self.config.pullback_max_chase_bps,
                                zone_tolerance_bps=self.config.pullback_zone_tolerance_bps,
                                valid_for_candles=self.config.pullback_valid_candles,
                                direction=sweep_dir,
                                original_suggested_entry=agent_result.suggested_entry or 0.0,
                                must_reach_price=agent_result.must_reach_price or 0.0,
                            )
                            plan.limit_price = plan.compute_limit_price()
                            signal.pullback_plan = plan

                            # Store pre-computed SL/TP for Agent 2 to reference
                            signal._pre_sl = pre_sl
                            signal._pre_tp = pre_tp
                            # Attach agent analysis to signal components so
                            # the refiner queue dashboard can display reasoning
                            if agent_analysis_data:
                                signal.components["agent_analysis"] = agent_analysis_data
                                # Store SL/TP in components for Agent 2 context
                                signal.components["sl_price"] = pre_sl
                                signal.components["tp_price"] = pre_tp
                                signal.components["rr_ratio"] = round(pre_rr, 2) if pre_rr else None
                            # Queue in refiner
                            queued = self.main_entry_refiner.add(signal)
                            if queued:
                                refiner_queued += 1
                                logger.info(
                                    "agent_wait_pullback_queued",
                                    symbol=signal.symbol,
                                    target_entry=agent_result.suggested_entry,
                                    entry_zone=plan.zone_str(),
                                    limit_price=round(plan.limit_price, 6),
                                    invalidation=round(plan.invalidation_level, 6),
                                    expires_in_seconds=expiry_seconds,
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
                        # If refiner unavailable/full, skip instead of forcing
                        # entry — the agent said to WAIT, so respect that.
                        logger.info(
                            "agent_wait_pullback_skipped_refiner_full",
                            symbol=signal.symbol,
                            confidence=agent_result.confidence,
                            refiner_size=len(self.main_entry_refiner.queue) if self.main_entry_refiner else 0,
                        )
                        vol_24h = self.exchange.get_24h_volume(signal.symbol)
                        await self.repo.insert_signal(
                            {
                                "symbol": signal.symbol,
                                "direction": signal.direction or "none",
                                "score": signal.score,
                                "reasons": signal.reasons + [
                                    f"AGENT:WAIT_PULLBACK(conf={agent_result.confidence:.0f},"
                                    f"refiner_full)"
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

                    # Agent says SETUP_CONFIRMED — route through Agent 2 for confirmation
                    # Agent 2 checks on the very first tick (no 5-minute wait)
                    # but MUST verify the 5m candle confirmation criteria Agent 1 specified
                    if (
                        self.main_entry_refiner
                        and signal.symbol not in self.portfolio.open_positions
                        and signal.symbol not in self.main_entry_refiner.get_queued_symbols()
                    ):
                        signal.original_1h_price = signal.entry_price
                        if agent_result.suggested_entry is not None:
                            signal.agent_target_entry = agent_result.suggested_entry
                        if agent_result.entry_zone_high is not None:
                            signal.agent_entry_zone_high = agent_result.entry_zone_high
                        if agent_result.entry_zone_low is not None:
                            signal.agent_entry_zone_low = agent_result.entry_zone_low

                        # Build a PullbackPlan so Agent 2 has zone context
                        zone_h = agent_result.entry_zone_high
                        zone_l = agent_result.entry_zone_low
                        if zone_h and zone_l and zone_h > zone_l and zone_h > 0:
                            now = datetime.now(timezone.utc)
                            # SETUP_CONFIRMED gets 4 hours to find confirmation
                            expiry_seconds = 4 * 60 * 60  # 4 hours
                            invalidation = 0.0
                            if agent_result.invalidation_level and agent_result.invalidation_level > 0:
                                invalidation = agent_result.invalidation_level
                            elif signal.sweep_result and signal.sweep_result.sweep_detected:
                                invalidation = signal.sweep_result.sweep_level
                            sweep_dir = (
                                signal.sweep_result.sweep_direction
                                if signal.sweep_result and signal.sweep_result.sweep_direction
                                else ("swing_low" if signal.direction == "bullish" else "swing_high" if signal.direction == "bearish" else "")
                            )
                            plan = PullbackPlan(
                                zone_low=zone_l,
                                zone_high=zone_h,
                                created_at=now,
                                expires_at=now + timedelta(seconds=expiry_seconds),
                                invalidation_level=invalidation,
                                max_chase_bps=self.config.pullback_max_chase_bps,
                                zone_tolerance_bps=self.config.pullback_zone_tolerance_bps,
                                valid_for_candles=3,  # Short window
                                direction=sweep_dir,
                                original_suggested_entry=agent_result.suggested_entry or 0.0,
                                must_reach_price=0.0,  # No must_reach_price gate for SETUP_CONFIRMED
                            )
                            plan.limit_price = plan.compute_limit_price()
                            signal.pullback_plan = plan

                        # Store pre-computed SL/TP for Agent 2 to reference
                        signal._pre_sl = pre_sl
                        signal._pre_tp = pre_tp
                        if agent_analysis_data:
                            signal.components["agent_analysis"] = agent_analysis_data
                            signal.components["sl_price"] = pre_sl
                            signal.components["tp_price"] = pre_tp
                            signal.components["rr_ratio"] = round(pre_rr, 2) if pre_rr else None

                        queued = self.main_entry_refiner.add_setup_confirmed(signal)
                        if queued:
                            refiner_queued += 1
                            logger.info(
                                "agent_setup_confirmed_queued",
                                symbol=signal.symbol,
                                confidence=agent_result.confidence,
                                regime=agent_result.market_regime,
                                risk=agent_result.risk_assessment,
                                entry_zone=f"{zone_l}-{zone_h}" if zone_l and zone_h else "none",
                            )
                            vol_24h = self.exchange.get_24h_volume(signal.symbol)
                            await self.repo.insert_signal(
                                {
                                    "symbol": signal.symbol,
                                    "direction": signal.direction or "none",
                                    "score": signal.score,
                                    "reasons": signal.reasons + [
                                        f"AGENT:SETUP_CONFIRMED→AGENT2(conf={agent_result.confidence:.0f})"
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

                    # Fallback: if refiner unavailable, skip (don't blindly execute)
                    logger.warning(
                        "agent_setup_confirmed_no_refiner",
                        symbol=signal.symbol,
                        confidence=agent_result.confidence,
                        reason="refiner unavailable or symbol already queued/in position",
                    )
                    continue

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
                if self.config.max_daily_trades > 0 and self.state.daily_trade_count >= self.config.max_daily_trades:
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

                # --- AI Entry Agent results ---
                # Agent already ran in the early decision block above.
                # Here we only set up variables for downstream execution.
                sl_override = None
                tp_override = None
                test_group = "control"

                if agent_early_decision is not None:
                    # Agent already decided — use its results (no duplicate API call)
                    test_group = "agent"
                    sl_override = agent_result.suggested_sl
                    tp_override = agent_result.suggested_tp

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
                    elif not self._can_enter_symbol(signal.symbol):
                        logger.info("primary_duplicate_blocked", symbol=signal.symbol)
                    elif self.footprint_analyzer and signal.sweep_result and not (
                        self.config.footprint_longs_only and signal.direction == "bearish"
                    ):
                        # Footprint order flow confirmation gate
                        footprint_passed = True
                        footprint_result = None
                        try:
                            footprint_result = await self.footprint_analyzer.analyze(
                                exchange=self.exchange,
                                symbol=signal.symbol,
                                sweep_direction=getattr(signal.sweep_result, "sweep_type", "") or "",
                                sweep_level=getattr(signal.sweep_result, "sweep_level", 0.0) or 0.0,
                                current_price=signal.entry_price,
                                trade_limit=self.config.footprint_trade_limit,
                                sweep_oi_usd=getattr(signal, "sweep_oi_usd", 0.0) or 0.0,
                            )
                            footprint_passed = footprint_result.passed
                            logger.info(
                                "footprint_check",
                                symbol=signal.symbol,
                                passed=footprint_passed,
                                confidence=round(footprint_result.confidence, 3),
                                delta_pct=round(footprint_result.delta_pct, 4),
                                absorption=round(footprint_result.absorption_score, 3),
                            )
                        except Exception as e:
                            logger.warning("footprint_check_error", symbol=signal.symbol, error=str(e))

                        if not footprint_passed:
                            logger.info(
                                "trade_rejected_footprint",
                                symbol=signal.symbol,
                                direction=signal.direction,
                                reasons=footprint_result.reasons if footprint_result else [],
                            )
                            # Store footprint data on the signal for analysis
                            if footprint_result:
                                signal_components["footprint"] = {
                                    "passed": False,
                                    "confidence": footprint_result.confidence,
                                    "delta_pct": footprint_result.delta_pct,
                                    "absorption": footprint_result.absorption_score,
                                    "cum_delta": footprint_result.cumulative_delta_confirms,
                                }
                        else:
                            # Footprint passed — store data and proceed to entry
                            if footprint_result:
                                signal_components["footprint"] = {
                                    "passed": True,
                                    "confidence": footprint_result.confidence,
                                    "delta_pct": footprint_result.delta_pct,
                                    "absorption": footprint_result.absorption_score,
                                    "cum_delta": footprint_result.cumulative_delta_confirms,
                                }
                            self._entering_symbols.add(signal.symbol)
                            try:
                                position, order_result, trade_record = await self.order_executor.execute_entry(
                                    signal=signal,
                                    current_balance=self.portfolio.current_balance,
                                    mode=self.state.mode,
                                    sl_override=sl_override,
                                    tp_override=tp_override,
                                )

                                if position and trade_record:
                                    position.confluence_score = signal.score
                                    if agent_analysis_data:
                                        position.agent1_reasoning = agent_analysis_data.get("reasoning", "")
                                    trade_record["test_group"] = test_group
                                    db_trade = await self.repo.insert_trade(trade_record)
                                    if db_trade:
                                        position.trade_id = db_trade.get("id", "")
                                    self.portfolio.record_entry(position)
                                    self.state.open_positions[signal.symbol] = position
                                    trades_entered += 1
                                    self.state.daily_trade_count += 1
                            finally:
                                self._entering_symbols.discard(signal.symbol)
                    else:
                        # Execute entry (no footprint gate or footprint skipped for shorts)
                        self._entering_symbols.add(signal.symbol)
                        try:
                            position, order_result, trade_record = await self.order_executor.execute_entry(
                                signal=signal,
                                current_balance=self.portfolio.current_balance,
                                mode=self.state.mode,
                                sl_override=sl_override,
                                tp_override=tp_override,
                            )
                        finally:
                            self._entering_symbols.discard(signal.symbol)

                        if position and trade_record:
                            # Store confluence score on position for post-trade analysis
                            position.confluence_score = signal.score
                            if agent_analysis_data:
                                position.agent1_reasoning = agent_analysis_data.get("reasoning", "")

                            # Tag trade with split test metadata
                            trade_record["test_group"] = test_group
                            # Note: agent_analysis and entry_headlines stored in signal_components,
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

    async def _rag_ingest_closed_trade(self, trade_id: str) -> None:
        """Ingest a freshly closed trade into the RAG knowledge base."""
        if not self.trade_rag:
            return
        try:
            trades = await self.repo.get_trades_by_ids([trade_id])
            if trades:
                await self.trade_rag.ingest_trade(trades[0])
        except Exception as e:
            logger.debug("rag_post_close_ingest_failed", error=str(e)[:80])

    async def _run_postmortem(self, position, exit_price: float, total_pnl: float,
                               pnl_pct: float, exit_reason: str, holding_seconds: float) -> None:
        """Run AI post-mortem lesson generation and ingest lessons into RAG.

        Runs under semaphore to limit concurrent API calls. Designed to be
        launched as a background task so it doesn't block trade execution.
        """
        if not self.lesson_generator:
            return

        async with self._postmortem_semaphore:
            try:
                # Fetch full trade record for agent reasoning fields
                trade_data = await self.repo.get_trades_by_ids([position.trade_id])
                trade = trade_data[0] if trade_data else {}

                # Get recent lessons to avoid repeating them
                recent_lessons = await self.repo.get_recent_lessons(limit=5)

                lessons = await self.lesson_generator.generate_lessons(
                    trade_id=position.trade_id,
                    symbol=position.symbol,
                    direction=position.direction,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    pnl_usd=total_pnl,
                    pnl_percent=pnl_pct,
                    exit_reason=exit_reason,
                    confluence_score=position.confluence_score,
                    holding_seconds=holding_seconds,
                    stop_loss=float(trade.get("stop_loss", 0) or 0),
                    take_profit=float(trade.get("take_profit", 0) or 0),
                    agent1_reasoning=position.agent1_reasoning or trade.get("agent1_reasoning", ""),
                    agent2_reasoning=trade.get("last_agent2_reasoning", ""),
                    agent3_reasoning=position.last_agent3_reasoning or "",
                    btc_trend=trade.get("btc_trend", ""),
                    signal_reasons=trade.get("signal_reasons") or [],
                    current_tier=position.current_tier,
                    recent_lessons=recent_lessons,
                )

                # Ingest each lesson into RAG as a knowledge chunk
                if lessons and self.trade_rag:
                    for lesson in lessons:
                        await self._rag_ingest_lesson(position, lesson)

                if lessons:
                    logger.info(
                        "postmortem_complete",
                        trade_id=position.trade_id,
                        symbol=position.symbol,
                        num_lessons=len(lessons),
                    )

            except Exception as e:
                logger.warning(
                    "postmortem_failed",
                    trade_id=position.trade_id,
                    error=str(e)[:100],
                )

    async def _rag_ingest_lesson(self, position, lesson: dict) -> None:
        """Ingest a single post-mortem lesson into RAG as a knowledge chunk."""
        if not self.trade_rag:
            return
        try:
            # Build a synthetic trade-like dict for the RAG ingestion
            lesson_text = lesson.get("lesson", "")
            severity = lesson.get("severity", "medium")
            lesson_type = lesson.get("lesson_type", "entry")
            outcome = lesson.get("outcome", "loss")

            lesson_trade = {
                "id": f"{position.trade_id}_lesson_{lesson_type}",
                "symbol": position.symbol,
                "direction": position.direction,
                "pnl_usd": lesson.get("pnl_usd", 0),
                "pnl_percent": lesson.get("pnl_percent", 0),
                "entry_price": position.entry_price,
                "exit_price": 0,
                "exit_reason": f"lesson:{lesson_type}",
                "confluence_score": lesson.get("confluence_score", 0),
                "stop_loss": 0,
                "take_profit": 0,
                "signal_reasons": f"[{severity.upper()} LESSON] {lesson_text}",
                "last_agent2_reasoning": "",
            }
            await self.trade_rag.ingest_trade(lesson_trade)
        except Exception as e:
            logger.debug("rag_lesson_ingest_failed", error=str(e)[:80])

    def _launch_postmortem(self, position, exit_price: float, total_pnl: float,
                            pnl_pct: float, exit_reason: str, holding_seconds: float) -> None:
        """Launch post-mortem as a background task."""
        task = asyncio.create_task(
            self._run_postmortem(position, exit_price, total_pnl, pnl_pct, exit_reason, holding_seconds)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _process_refined_entries(self) -> None:
        """Process signals refined on 5m candles (post-sweep entry refinement).

        Called on MONITOR ticks but only runs every 5 minutes (aligned with
        Agent 2's check interval). Agent 2 is the SOLE checker of queued
        entries — no redundant polling from primary tick.
        """
        if not self.main_entry_refiner:
            return

        # 5-minute cooldown — Agent 2 checks every 5 min, no need to poll faster
        now = datetime.now(timezone.utc)
        if hasattr(self, "_last_refiner_run"):
            elapsed = (now - self._last_refiner_run).total_seconds()
            refiner_interval = getattr(
                self.config, "refiner_agent_check_interval_minutes", 5.0
            ) * 60
            if elapsed < refiner_interval:
                return
        self._last_refiner_run = now

        # Sync Agent 2 toggle from engine state (dashboard can flip at runtime)
        # Check every 5th monitor tick (~5 min) to avoid excess DB reads
        if self.refiner_agent and self.main_entry_refiner.total_checks % 5 == 0:
            try:
                db_state = await self.repo.get_engine_state() or {}
                agent2_enabled = db_state.get(
                    "refiner_agent_enabled",
                    getattr(self.config, "refiner_agent_enabled", False),
                )
                current_agent = self.main_entry_refiner.refiner_agent
                if agent2_enabled and current_agent is None:
                    self.main_entry_refiner.refiner_agent = self.refiner_agent
                    logger.info("refiner_agent_enabled_at_runtime")
                elif not agent2_enabled and current_agent is not None:
                    self.main_entry_refiner.refiner_agent = None
                    logger.info("refiner_agent_disabled_at_runtime")
            except Exception:
                pass  # Non-critical — toggle will sync on next check

        if not self.main_entry_refiner.queue:
            return

        ready_signals = await self.main_entry_refiner.check_all(
            open_position_symbols=set(self.portfolio.open_positions.keys()),
        )
        for signal in ready_signals:

            # ── Expired signal — drop it ──────────────────────────
            # If the refiner expired without a pullback, the market has
            # moved on.  Do NOT re-evaluate or enter at current price —
            # chasing missed pullbacks leads to bad entries.
            if getattr(signal, "_expired_from_refiner", False):
                logger.info(
                    "refiner_expired_dropped",
                    symbol=signal.symbol,
                    reason="pullback_never_came",
                )
                continue

            # Daily trade limit
            if self.config.max_daily_trades > 0 and self.state.daily_trade_count >= self.config.max_daily_trades:
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

            # ── Pre-dispatch revalidation ──
            # Two paths: PullbackPlan signals get structured zone enforcement,
            # SETUP_CONFIRMED signals get the legacy drift check.
            try:
                ticker = await self.exchange.fetch_ticker(signal.symbol)
                live_price = float(ticker.get("last", 0) or 0)
                if live_price <= 0:
                    logger.warning("refiner_live_price_zero", symbol=signal.symbol)
                    continue

                plan = getattr(signal, "pullback_plan", None)

                if plan is not None:
                    # ── PullbackPlan structured enforcement ──
                    reject_reason = ""
                    if plan.is_expired:
                        reject_reason = "plan_expired"
                    elif plan.invalidation_hit(live_price):
                        reject_reason = "invalidation_hit"
                    elif not plan.price_in_zone(live_price):
                        reject_reason = "price_outside_zone"
                    elif not plan.slippage_ok(live_price):
                        reject_reason = "slippage_exceeded"

                    if reject_reason:
                        logger.info(
                            "pullback_plan_entry_rejected",
                            symbol=signal.symbol,
                            reject_reason=reject_reason,
                            decision_price=round(signal.entry_price, 6),
                            live_price=round(live_price, 6),
                            zone=plan.zone_str(),
                            invalidation=round(plan.invalidation_level, 6),
                            age_seconds=round(plan.age_seconds, 1),
                            path_taken="pre_dispatch",
                        )
                        # Re-queue soft rejections back to refiner so Agent 2
                        # can keep watching until the 4h expiry.
                        # Only price_outside_zone and slippage_exceeded are
                        # re-queueable — invalidation_hit and plan_expired
                        # are permanent kills.
                        if reject_reason in ("price_outside_zone", "slippage_exceeded"):
                            refiner_entry = getattr(signal, "_refiner_entry", None)
                            if refiner_entry and self.main_entry_refiner:
                                requeued = self.main_entry_refiner.requeue(signal, refiner_entry)
                                if requeued:
                                    logger.info(
                                        "pullback_plan_requeued",
                                        symbol=signal.symbol,
                                        reject_reason=reject_reason,
                                    )
                        continue  # DO_NOT_ENTER — no market fallback

                    # Plan passed all checks — update entry to live price
                    signal.entry_price = live_price

                else:
                    # ── No PullbackPlan — reject entry ──
                    # Every trade MUST have a PullbackPlan with zone enforcement.
                    # Signals without a plan have no structural confirmation and
                    # must not enter.
                    logger.warning(
                        "refiner_no_pullback_plan_rejected",
                        symbol=signal.symbol,
                        live_price=round(live_price, 6),
                        decision_price=round(signal.entry_price, 6),
                        reason="no_pullback_plan",
                    )
                    continue
            except Exception as e:
                logger.warning("refiner_live_price_check_failed", symbol=signal.symbol, error=str(e)[:80])
                continue  # Do NOT fall through to execution without validation

            # Duplicate entry guard
            if not self._can_enter_symbol(signal.symbol):
                logger.info("refiner_duplicate_blocked", symbol=signal.symbol)
                continue

            # Footprint order flow confirmation gate (same as primary path)
            if self.footprint_analyzer and signal.sweep_result and not (
                self.config.footprint_longs_only and signal.direction == "bearish"
            ):
                try:
                    fp_result = await self.footprint_analyzer.analyze(
                        exchange=self.exchange,
                        symbol=signal.symbol,
                        sweep_direction=getattr(signal.sweep_result, "sweep_type", "") or "",
                        sweep_level=getattr(signal.sweep_result, "sweep_level", 0.0) or 0.0,
                        current_price=signal.entry_price,
                        trade_limit=self.config.footprint_trade_limit,
                        sweep_oi_usd=getattr(signal, "sweep_oi_usd", 0.0) or 0.0,
                    )
                    logger.info(
                        "footprint_check",
                        symbol=signal.symbol,
                        passed=fp_result.passed,
                        confidence=round(fp_result.confidence, 3),
                        delta_pct=round(fp_result.delta_pct, 4),
                        path="refiner",
                    )
                    if not fp_result.passed:
                        logger.info(
                            "trade_rejected_footprint",
                            symbol=signal.symbol,
                            direction=signal.direction,
                            path="refiner",
                            reasons=fp_result.reasons,
                        )
                        continue
                except Exception as e:
                    logger.warning("footprint_check_error", symbol=signal.symbol, error=str(e))

            # Execute the trade (with Agent 2 SL/TP overrides if available)
            self._entering_symbols.add(signal.symbol)
            try:
                sl_override = getattr(signal, "_agent2_sl", None)
                tp_override = getattr(signal, "_agent2_tp", None)
                size_modifier = getattr(signal, "_agent2_size_modifier", 1.0)

                # Scale balance for position sizing if Agent 2 specified a modifier
                effective_balance = self.portfolio.current_balance
                if size_modifier != 1.0 and size_modifier > 0:
                    effective_balance = self.portfolio.current_balance * size_modifier

                position, order_result, trade_record = (
                    await self.order_executor.execute_entry(
                        signal=signal,
                        current_balance=effective_balance,
                        mode=self.state.mode,
                        sl_override=sl_override,
                        tp_override=tp_override,
                    )
                )

                if position and trade_record:
                    position.confluence_score = signal.score
                    # Set agent1_reasoning from signal components if available
                    agent_data = signal.components.get("agent_analysis") if hasattr(signal, "components") else None
                    if agent_data:
                        position.agent1_reasoning = agent_data.get("reasoning", "")
                    # Attach Agent 2 (Refiner) decision data to trade record for dashboard
                    if getattr(signal, "_agent2_reasoning", None):
                        trade_record["last_agent2_action"] = getattr(signal, "_agent2_action", "")
                        trade_record["last_agent2_reasoning"] = getattr(signal, "_agent2_reasoning", "")
                        trade_record["last_agent2_urgency"] = getattr(signal, "_agent2_urgency", "")
                        trade_record["agent2_confidence"] = getattr(signal, "_agent2_confidence", 0)
                        trade_record["agent2_check_count"] = getattr(signal, "_agent2_check_count", 0)
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

                    # Link the original WAIT_PULLBACK signal to the trade
                    if position.trade_id:
                        await self.repo.link_signal_to_trade(
                            signal.symbol, position.trade_id
                        )
            except Exception as e:
                logger.warning(
                    "refiner_entry_failed",
                    symbol=signal.symbol,
                    error=str(e),
                )
            finally:
                self._entering_symbols.discard(signal.symbol)

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
            if self.config.max_daily_trades > 0 and self.state.daily_trade_count >= self.config.max_daily_trades:
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

            if not self._can_enter_symbol(signal.symbol):
                logger.info("consensus_duplicate_blocked", symbol=signal.symbol)
                continue

            self._entering_symbols.add(signal.symbol)
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
                    # Set agent1_reasoning from signal components if available
                    agent_data = signal.components.get("agent_analysis") if hasattr(signal, "components") else None
                    if agent_data:
                        position.agent1_reasoning = agent_data.get("reasoning", "")
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
            finally:
                self._entering_symbols.discard(signal.symbol)

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

            # Daily trade limit check
            if self.config.max_daily_trades > 0 and self.state.daily_trade_count >= self.config.max_daily_trades:
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
            # Skip Agent 1 if symbol is already queued for Agent 2
            if (
                self.main_entry_refiner
                and graduated.symbol in self.main_entry_refiner.get_queued_symbols()
            ):
                logger.info(
                    "watchlist_agent1_skipped_already_in_refiner",
                    symbol=graduated.symbol,
                )
                continue

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

                    # Fetch per-symbol trade history for Agent 1 feedback loop
                    wl_symbol_history = []
                    if getattr(self.config, "symbol_history_enabled", True):
                        try:
                            wl_symbol_history = await self.repo.get_recent_trades_for_symbol(
                                graduated.symbol, limit=5, mode=self.state.mode
                            )
                            # No-lookahead: exclude trades closed < 5 min ago (Req 5)
                            if wl_symbol_history:
                                cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
                                wl_symbol_history = [
                                    t for t in wl_symbol_history
                                    if datetime.fromisoformat(t.get("exit_time", "2000-01-01T00:00:00+00:00")) < cutoff
                                ]
                        except Exception:
                            pass

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
                        "symbol_history": wl_symbol_history,
                    }

                    wl_agent_result = await self.agent_analyst.analyze_signal(graduated, wl_context)

                    # ── Agent 1 is the direction authority ──
                    new_dir = graduated.direction or ""
                    if wl_agent_result.direction in ("LONG", "SHORT"):
                        new_dir = "bullish" if wl_agent_result.direction == "LONG" else "bearish"
                        if new_dir != graduated.direction:
                            logger.info(
                                "watchlist_agent1_direction_override",
                                symbol=graduated.symbol,
                                scanner_direction=graduated.direction,
                                agent_direction=new_dir,
                            )
                            graduated.direction = new_dir
                            # Re-compute SL/TP with the flipped direction
                            pre_sl = self.order_executor._calculate_stop_loss(graduated)
                            pre_tp = None
                            pre_rr = None
                            if pre_sl is not None:
                                pre_tp = self.order_executor._calculate_take_profit(graduated, pre_sl)
                                sl_dist = abs(graduated.entry_price - pre_sl)
                                if sl_dist > 0 and pre_tp is not None:
                                    pre_rr = abs(pre_tp - graduated.entry_price) / sl_dist

                    # ── SL/TP Priority: Agent 1 is primary, risk_manager is fallback ──
                    agent_sl = wl_agent_result.suggested_sl
                    agent_tp = wl_agent_result.suggested_tp

                    if agent_sl and agent_sl > 0:
                        sl_valid = True
                        if new_dir == "bullish" and agent_sl >= graduated.entry_price:
                            sl_valid = False
                        elif new_dir == "bearish" and agent_sl <= graduated.entry_price:
                            sl_valid = False
                        elif graduated.entry_price > 0:
                            sl_dist = abs(graduated.entry_price - agent_sl) / graduated.entry_price
                            if sl_dist > self.config.max_sl_pct:
                                sl_valid = False
                        if sl_valid:
                            pre_sl = agent_sl

                    if agent_tp and agent_tp > 0:
                        tp_valid = True
                        if new_dir == "bullish" and agent_tp <= graduated.entry_price:
                            tp_valid = False
                        elif new_dir == "bearish" and agent_tp >= graduated.entry_price:
                            tp_valid = False
                        if tp_valid:
                            pre_tp = agent_tp

                    logger.info(
                        "watchlist_agent_decision",
                        symbol=graduated.symbol,
                        action=wl_agent_result.action,
                        direction=wl_agent_result.direction,
                        confidence=wl_agent_result.confidence,
                        risk=wl_agent_result.risk_assessment,
                        latency_ms=round(wl_agent_result.latency_ms, 1),
                        reasoning=wl_agent_result.reasoning[:120],
                    )

                    # Build agent_analysis_data for signal persistence
                    wl_agent_analysis_data = {
                        "action": wl_agent_result.action,
                        "direction": wl_agent_result.direction,
                        "confidence": wl_agent_result.confidence,
                        "reasoning": wl_agent_result.reasoning,
                        "market_regime": wl_agent_result.market_regime,
                        "risk_assessment": wl_agent_result.risk_assessment,
                        "suggested_entry": wl_agent_result.suggested_entry,
                        "entry_zone_high": wl_agent_result.entry_zone_high,
                        "entry_zone_low": wl_agent_result.entry_zone_low,
                        "suggested_sl": wl_agent_result.suggested_sl,
                        "suggested_tp": wl_agent_result.suggested_tp,
                        "must_reach_price": wl_agent_result.must_reach_price,
                        "invalidation_level": wl_agent_result.invalidation_level,
                        "latency_ms": round(wl_agent_result.latency_ms, 1),
                        "error": wl_agent_result.error,
                        "tokens": wl_agent_result.input_tokens + wl_agent_result.output_tokens,
                    }

                    if wl_agent_result.action == "SKIP":
                        agent_skip = True
                        logger.info(
                            "watchlist_signal_skipped_by_agent",
                            symbol=graduated.symbol,
                            confidence=wl_agent_result.confidence,
                            risk=wl_agent_result.risk_assessment,
                            reasoning=wl_agent_result.reasoning[:120],
                        )
                        # Save SKIP signal for dashboard history
                        vol_24h = self.exchange.get_24h_volume(graduated.symbol)
                        await self.repo.insert_signal(
                            {
                                "symbol": graduated.symbol,
                                "direction": graduated.direction or "none",
                                "score": graduated.score,
                                "reasons": graduated.reasons + [
                                    f"AGENT:SKIP(conf={wl_agent_result.confidence:.0f},watchlist)"
                                ],
                                "components": {
                                    "volume_24h": vol_24h,
                                    "test_group": "agent",
                                    "agent_analysis": wl_agent_analysis_data,
                                    "signal_type": "watchlist",
                                },
                                "current_price": graduated.entry_price,
                                "acted_on": False,
                                "scan_cycle": 0,
                            }
                        )
                    elif wl_agent_result.action == "SETUP_CONFIRMED":
                        # Agent says setup is valid — route through Agent 2 for 5m confirmation
                        # NEVER execute directly from watchlist — Agent 2 MUST confirm
                        agent_skip = True
                        if (
                            self.main_entry_refiner
                            and graduated.symbol not in self.portfolio.open_positions
                            and graduated.symbol not in self.main_entry_refiner.get_queued_symbols()
                        ):
                            zone_h = wl_agent_result.entry_zone_high
                            zone_l = wl_agent_result.entry_zone_low
                            if not zone_h or not zone_l or zone_h <= zone_l or zone_h <= 0:
                                logger.warning(
                                    "watchlist_setup_confirmed_rejected_invalid_zone",
                                    symbol=graduated.symbol,
                                    zone_high=zone_h,
                                    zone_low=zone_l,
                                )
                            else:
                                now = datetime.now(timezone.utc)
                                expiry_seconds = 4 * 60 * 60  # 4 hours for SETUP_CONFIRMED
                                invalidation = 0.0
                                if wl_agent_result.invalidation_level and wl_agent_result.invalidation_level > 0:
                                    invalidation = wl_agent_result.invalidation_level
                                elif (
                                    graduated.sweep_result
                                    and graduated.sweep_result.sweep_detected
                                ):
                                    invalidation = graduated.sweep_result.sweep_level

                                sweep_dir = (
                                    graduated.sweep_result.sweep_direction
                                    if graduated.sweep_result and graduated.sweep_result.sweep_direction
                                    else ("swing_low" if graduated.direction == "bullish" else "swing_high" if graduated.direction == "bearish" else "")
                                )
                                plan = PullbackPlan(
                                    zone_low=zone_l,
                                    zone_high=zone_h,
                                    created_at=now,
                                    expires_at=now + timedelta(seconds=expiry_seconds),
                                    invalidation_level=invalidation,
                                    max_chase_bps=self.config.pullback_max_chase_bps,
                                    zone_tolerance_bps=self.config.pullback_zone_tolerance_bps,
                                    valid_for_candles=3,  # Short window
                                    direction=sweep_dir,
                                    original_suggested_entry=wl_agent_result.suggested_entry or 0.0,
                                    must_reach_price=0.0,  # No must_reach for SETUP_CONFIRMED
                                )
                                plan.limit_price = plan.compute_limit_price()
                                graduated.pullback_plan = plan
                                graduated.original_1h_price = graduated.entry_price
                                graduated.agent_target_entry = wl_agent_result.suggested_entry
                                graduated.agent_entry_zone_high = zone_h
                                graduated.agent_entry_zone_low = zone_l

                                # Attach agent analysis for dashboard
                                graduated.components["agent_analysis"] = wl_agent_analysis_data
                                graduated.components["sl_price"] = pre_sl
                                graduated.components["tp_price"] = pre_tp
                                graduated.components["rr_ratio"] = round(pre_rr, 2) if pre_rr else None

                                # Store pre-computed SL/TP for Agent 2
                                graduated._pre_sl = pre_sl
                                graduated._pre_tp = pre_tp

                                queued = self.main_entry_refiner.add_setup_confirmed(graduated)
                                if queued:
                                    logger.info(
                                        "watchlist_setup_confirmed_queued_agent2",
                                        symbol=graduated.symbol,
                                        confidence=wl_agent_result.confidence,
                                        regime=wl_agent_result.market_regime,
                                        risk=wl_agent_result.risk_assessment,
                                        entry_zone=plan.zone_str(),
                                    )
                                    vol_24h = self.exchange.get_24h_volume(graduated.symbol)
                                    await self.repo.insert_signal(
                                        {
                                            "symbol": graduated.symbol,
                                            "direction": graduated.direction or "none",
                                            "score": graduated.score,
                                            "reasons": graduated.reasons + [
                                                f"AGENT:SETUP_CONFIRMED→AGENT2(conf={wl_agent_result.confidence:.0f},watchlist)"
                                            ],
                                            "components": {
                                                "volume_24h": vol_24h,
                                                "test_group": "agent",
                                                "agent_analysis": wl_agent_analysis_data,
                                                "signal_type": "watchlist",
                                            },
                                            "current_price": graduated.entry_price,
                                            "acted_on": False,
                                            "scan_cycle": 0,
                                        }
                                    )
                                else:
                                    logger.info(
                                        "watchlist_setup_confirmed_not_queued",
                                        symbol=graduated.symbol,
                                        reason="refiner_queue_full_or_duplicate",
                                    )
                        else:
                            logger.warning(
                                "watchlist_setup_confirmed_no_refiner",
                                symbol=graduated.symbol,
                                reason="refiner unavailable or symbol already queued/in position",
                            )
                    elif wl_agent_result.action == "WAIT_PULLBACK":
                        # Agent says wait — queue a PullbackPlan in the refiner (never chase now)
                        agent_skip = True
                        if (
                            self.main_entry_refiner
                            and graduated.symbol not in self.portfolio.open_positions
                            and graduated.symbol not in self.main_entry_refiner.get_queued_symbols()
                        ):
                            zone_h = wl_agent_result.entry_zone_high
                            zone_l = wl_agent_result.entry_zone_low
                            if not zone_h or not zone_l or zone_h <= zone_l or zone_h <= 0:
                                logger.warning(
                                    "watchlist_wait_pullback_rejected_invalid_zone",
                                    symbol=graduated.symbol,
                                    zone_high=zone_h,
                                    zone_low=zone_l,
                                )
                            else:
                                now = datetime.now(timezone.utc)
                                expiry_seconds = self.config.pullback_valid_candles * 5 * 60
                                invalidation = 0.0
                                if wl_agent_result.invalidation_level and wl_agent_result.invalidation_level > 0:
                                    invalidation = wl_agent_result.invalidation_level
                                elif (
                                    graduated.sweep_result
                                    and graduated.sweep_result.sweep_detected
                                ):
                                    invalidation = graduated.sweep_result.sweep_level

                                # Use sweep_direction for invalidation logic (not graduated.direction
                                # which may have been flipped by HTF override)
                                sweep_dir = (
                                    graduated.sweep_result.sweep_direction
                                    if graduated.sweep_result and graduated.sweep_result.sweep_direction
                                    else ("swing_low" if graduated.direction == "bullish" else "swing_high" if graduated.direction == "bearish" else "")
                                )
                                plan = PullbackPlan(
                                    zone_low=zone_l,
                                    zone_high=zone_h,
                                    created_at=now,
                                    expires_at=now + timedelta(seconds=expiry_seconds),
                                    invalidation_level=invalidation,
                                    max_chase_bps=self.config.pullback_max_chase_bps,
                                    zone_tolerance_bps=self.config.pullback_zone_tolerance_bps,
                                    valid_for_candles=self.config.pullback_valid_candles,
                                    direction=sweep_dir,
                                    original_suggested_entry=wl_agent_result.suggested_entry or 0.0,
                                    must_reach_price=wl_agent_result.must_reach_price or 0.0,
                                )
                                plan.limit_price = plan.compute_limit_price()
                                graduated.pullback_plan = plan
                                graduated.original_1h_price = graduated.entry_price
                                graduated.agent_target_entry = wl_agent_result.suggested_entry
                                graduated.agent_entry_zone_high = zone_h
                                graduated.agent_entry_zone_low = zone_l

                                # Attach agent analysis so dashboard can display reasoning
                                graduated.components["agent_analysis"] = wl_agent_analysis_data
                                graduated.components["sl_price"] = pre_sl
                                graduated.components["tp_price"] = pre_tp
                                graduated.components["rr_ratio"] = round(pre_rr, 2) if pre_rr else None

                                # Store pre-computed SL/TP for Agent 2
                                graduated._pre_sl = pre_sl
                                graduated._pre_tp = pre_tp

                                queued = self.main_entry_refiner.add(graduated)

                                if queued:
                                    logger.info(
                                        "watchlist_wait_pullback_queued",
                                        symbol=graduated.symbol,
                                        target_entry=wl_agent_result.suggested_entry,
                                        entry_zone=plan.zone_str(),
                                        limit_price=round(plan.limit_price, 6),
                                        expires_in_seconds=expiry_seconds,
                                    )
                                    # Persist signal for dashboard linkage
                                    vol_24h = self.exchange.get_24h_volume(graduated.symbol)
                                    await self.repo.insert_signal(
                                        {
                                            "symbol": graduated.symbol,
                                            "direction": graduated.direction or "none",
                                            "score": graduated.score,
                                            "reasons": graduated.reasons + [
                                                f"AGENT:WAIT_PULLBACK(conf={wl_agent_result.confidence:.0f},"
                                                f"target={wl_agent_result.suggested_entry},watchlist)"
                                            ],
                                            "components": {
                                                "volume_24h": vol_24h,
                                                "test_group": "agent",
                                                "agent_analysis": wl_agent_analysis_data,
                                                "signal_type": "watchlist",
                                            },
                                            "current_price": graduated.entry_price,
                                            "acted_on": False,
                                            "scan_cycle": 0,
                                        }
                                    )
                                else:
                                    logger.info(
                                        "watchlist_wait_pullback_skipped_refiner_full",
                                        symbol=graduated.symbol,
                                    )
                        else:
                            logger.info(
                                "watchlist_wait_pullback_not_queued",
                                symbol=graduated.symbol,
                                reason="refiner_unavailable_or_symbol_already_tracked",
                            )
                except Exception as e:
                    logger.warning("watchlist_agent_failed", symbol=graduated.symbol, error=str(e))
                    # On agent failure, SKIP the trade — never enter blind
                    agent_skip = True

            if agent_skip:
                continue

            # Idempotency guard: skip if symbol already in open positions or being entered
            if not self._can_enter_symbol(graduated.symbol):
                logger.warning(
                    "watchlist_duplicate_entry_blocked",
                    symbol=graduated.symbol,
                )
                continue

            # Execute the trade
            self._entering_symbols.add(graduated.symbol)
            try:
                position, order_result, trade_record = (
                    await self.order_executor.execute_entry(
                        signal=graduated,
                        current_balance=self.portfolio.current_balance,
                        mode=self.state.mode,
                    )
                )

                if position and order_result.filled_quantity > 0:
                    # Second idempotency check after async execution gap
                    if graduated.symbol in self.state.open_positions:
                        logger.warning(
                            "watchlist_duplicate_post_exec_blocked",
                            symbol=graduated.symbol,
                        )
                        self._entering_symbols.discard(graduated.symbol)
                        continue

                    # Match all other entry paths: confluence score, DB persistence, portfolio, state
                    position.confluence_score = graduated.score
                    # Set agent1_reasoning from signal components if available
                    agent_data = graduated.components.get("agent_analysis") if hasattr(graduated, "components") else None
                    if agent_data:
                        position.agent1_reasoning = agent_data.get("reasoning", "")

                    if trade_record:
                        db_trade = await self.repo.insert_trade(trade_record)
                        if db_trade:
                            position.trade_id = db_trade.get("id", "")

                    self.portfolio.record_entry(position)
                    self.state.open_positions[graduated.symbol] = position
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
            finally:
                self._entering_symbols.discard(graduated.symbol)

    async def _monitor_tick(self) -> None:
        """Check open positions for SL/TP/trailing stop + liquidation proximity."""
        # Non-crypto engines must NOT monitor main bot positions — those are crypto
        # symbols that only the crypto engine's Binance connector can handle.
        # Without this guard, stocks/commodities engines try to fetch crypto tickers
        # from Yahoo Finance → 500 errors + triple API call volume.
        has_main_positions = bool(self.portfolio.open_positions) and self._market_name == "crypto"

        if not has_main_positions:
            return

        # Sync Agent 3 runtime toggle from DB (check every ~5 monitor ticks)
        if self.position_agent and hasattr(self, "_agent3_sync_counter"):
            self._agent3_sync_counter += 1
        else:
            self._agent3_sync_counter = 0
        if self.position_agent and self._agent3_sync_counter % 5 == 0:
            try:
                db_state = await self.repo.get_engine_state() or {}
                agent3_enabled = db_state.get(
                    "position_agent_enabled",
                    getattr(self.config, "position_agent_enabled", False),
                )
                current_agent = self.position_monitor._position_agent
                if agent3_enabled and current_agent is None:
                    self.position_monitor._position_agent = self.position_agent
                    logger.info("position_agent_enabled_at_runtime")
                elif not agent3_enabled and current_agent is not None:
                    self.position_monitor._position_agent = None
                    logger.info("position_agent_disabled_at_runtime")
            except Exception:
                pass

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
                self.portfolio.record_exit(liq_exit.symbol, liq_result.avg_price or liq_exit.price, liq_result.fee)
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

        market_info = getattr(self.exchange, "market_info", None)
        trading_hours = market_info.trading_hours if market_info else None
        exits = await self.position_monitor.check_positions(
            self.portfolio.open_positions, self.exchange,
            atr_values=atr_values, trading_hours=trading_hours,
            market_filter=self.market_filter,
            candle_manager=self.candle_manager,
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

                    # Check if ALL tiers are filled → close the trade (dust cleanup)
                    all_filled = all(t.filled for t in position.tp_tiers)
                    if all_filled:
                        logger.info("all_tp_tiers_filled_closing", symbol=exit_signal.symbol,
                                    remaining_qty=position.quantity)
                        # Accumulate PnL from all partial exits
                        total_pnl = 0.0
                        total_fees = 0.0
                        try:
                            partial_exits = await self.repo.get_partial_exits(position.trade_id)
                            total_pnl = sum(float(pe.get("pnl_usd", 0)) for pe in partial_exits)
                            total_fees = sum(float(pe.get("fees_usd", 0)) for pe in partial_exits)
                        except Exception as e:
                            logger.warning("partial_exit_sum_failed", error=str(e))
                        orig_cost = (position.original_quantity * position.entry_price) if position.original_quantity else position.cost_usd
                        pnl_pct = (total_pnl / orig_cost * 100) if orig_cost > 0 else 0
                        last_fill_price = order_result.avg_price or exit_signal.price
                        await self.repo.close_trade(
                            trade_id=position.trade_id,
                            exit_price=last_fill_price,
                            exit_quantity=position.original_quantity or position.quantity,
                            pnl_usd=round(total_pnl, 4),
                            pnl_percent=round(pnl_pct, 2),
                            exit_reason="all_tp_hit",
                            fees_usd=total_fees,
                        )
                        self.portfolio.close_position(exit_signal.symbol)
                        self._active_symbols.discard(exit_signal.symbol)

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
                    self.portfolio.record_exit(exit_signal.symbol, order_result.avg_price or exit_signal.price, order_result.fee)

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

                    # --- RAG knowledge ingestion ---
                    await self._rag_ingest_closed_trade(position.trade_id)

                    # --- AI Post-mortem (lesson generation + RAG ingestion) ---
                    self._launch_postmortem(
                        position, order_result.avg_price or exit_signal.price,
                        total_pnl, pnl_pct, exit_signal.reason, holding_seconds,
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

                    # --- Post-win reassessment (all TPs hit → rescan symbol) ---
                    await self._post_win_reassessment(
                        exit_signal.symbol, position, pnl_pct,
                    )

        # --- Agent 3 (Position Manager) — runs AFTER algorithmic exits are executed ---
        # This ensures slow LLM calls never block time-critical TP/SL execution.
        exited_symbols = {e.symbol for e in exits}
        agent3_exits = await self.position_monitor.run_agent3_checks(
            self.portfolio.open_positions, self.exchange,
            atr_values=atr_values,
            exited_symbols=exited_symbols,
            market_filter=self.market_filter,
            candle_manager=self.candle_manager,
        )
        for exit_signal in agent3_exits:
            position = self.portfolio.open_positions.get(exit_signal.symbol)
            if not position:
                continue
            if exit_signal.is_partial and exit_signal.partial_quantity > 0:
                order_result, pnl = await self.order_executor.execute_partial_exit(
                    symbol=exit_signal.symbol,
                    position=position,
                    reason=exit_signal.reason,
                    current_price=exit_signal.price,
                    quantity=exit_signal.partial_quantity,
                    tier=exit_signal.tier,
                )
                if order_result:
                    self.portfolio.record_partial_exit(
                        symbol=exit_signal.symbol,
                        exit_price=order_result.avg_price or exit_signal.price,
                        quantity=order_result.filled_quantity or exit_signal.partial_quantity,
                        fee=order_result.fee,
                    )
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

                    # Check if ALL tiers are filled → close the trade (dust cleanup)
                    if position.tp_tiers and all(t.filled for t in position.tp_tiers):
                        logger.info("all_tp_tiers_filled_closing", symbol=exit_signal.symbol,
                                    remaining_qty=position.quantity)
                        total_pnl = 0.0
                        total_fees = 0.0
                        try:
                            partial_exits = await self.repo.get_partial_exits(position.trade_id)
                            total_pnl = sum(float(pe.get("pnl_usd", 0)) for pe in partial_exits)
                            total_fees = sum(float(pe.get("fees_usd", 0)) for pe in partial_exits)
                        except Exception as e:
                            logger.warning("partial_exit_sum_failed", error=str(e))
                        orig_cost = (position.original_quantity * position.entry_price) if position.original_quantity else position.cost_usd
                        pnl_pct = (total_pnl / orig_cost * 100) if orig_cost > 0 else 0
                        last_fill_price = order_result.avg_price or exit_signal.price
                        await self.repo.close_trade(
                            trade_id=position.trade_id,
                            exit_price=last_fill_price,
                            exit_quantity=position.original_quantity or position.quantity,
                            pnl_usd=round(total_pnl, 4),
                            pnl_percent=round(pnl_pct, 2),
                            exit_reason="all_tp_hit",
                            fees_usd=total_fees,
                        )
                        self.portfolio.close_position(exit_signal.symbol)
                        self._active_symbols.discard(exit_signal.symbol)
            else:
                order_result, pnl = await self.order_executor.execute_exit(
                    symbol=exit_signal.symbol,
                    position=position,
                    reason=exit_signal.reason,
                    current_price=exit_signal.price,
                )
                if order_result:
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
                    self.portfolio.record_exit(exit_signal.symbol, order_result.avg_price or exit_signal.price, order_result.fee)
                    if exit_signal.reason in ("sl_hit", "stale_close"):
                        self.risk_manager.record_stop_out(exit_signal.symbol)
                    self.adaptive_threshold.record_outcome(is_win=(total_pnl > 0))
                    self.state.open_positions.pop(exit_signal.symbol, None)

        # Persist SL changes and Agent 3 data for open positions
        for symbol, pos in self.portfolio.open_positions.items():
            if not pos.trade_id:
                continue
            update_fields: dict = {}

            # Track last persisted SL to avoid redundant DB writes
            last_sl = getattr(pos, "_last_persisted_sl", None)
            if last_sl is None:
                # First tick — initialize tracking without writing
                pos._last_persisted_sl = pos.stop_loss
            elif pos.stop_loss != last_sl:
                update_fields["stop_loss"] = pos.stop_loss
                pos._last_persisted_sl = pos.stop_loss

            # Persist Agent 3 reasoning when it has a new decision
            last_a3 = getattr(pos, "_last_persisted_agent3", None)
            if pos.last_agent3_action and pos.last_agent3_action != last_a3:
                update_fields["last_agent3_action"] = pos.last_agent3_action
                update_fields["last_agent3_reasoning"] = pos.last_agent3_reasoning
                update_fields["agent3_confidence"] = pos.agent3_confidence
                if pos.last_agent3_sl:
                    update_fields["last_agent3_sl"] = pos.last_agent3_sl
                pos._last_persisted_agent3 = pos.last_agent3_action

            # Persist EXTEND_TP3 changes to the database
            if getattr(pos, "_tp_extended", False) and pos.tp_tiers:
                update_fields["tp_tiers"] = [
                    {
                        "level": t.level,
                        "price": t.price,
                        "pct": t.pct,
                        "quantity": t.quantity,
                        "filled": t.filled,
                        "fill_price": t.fill_price,
                        "fill_time": t.fill_time.isoformat() if t.fill_time else None,
                    }
                    for t in pos.tp_tiers
                ]
                pos._tp_extended = False

            if update_fields:
                try:
                    await self.repo.update_trade(pos.trade_id, update_fields)
                    if "stop_loss" in update_fields:
                        logger.info(
                            "sl_persisted_to_db",
                            symbol=symbol,
                            old_sl=last_sl,
                            new_sl=pos.stop_loss,
                        )
                except Exception as e:
                    logger.warning("position_persist_failed", symbol=symbol, error=str(e))

        # Sync state
        self.state.current_balance = self.portfolio.current_balance
        self.state.peak_balance = self.portfolio.peak_balance
        self.state.daily_pnl = self.portfolio.daily_pnl
        self.state.total_pnl = self.portfolio.total_pnl

    async def _post_win_reassessment(
        self,
        symbol: str,
        closed_position: Position,
        pnl_pct: float,
    ) -> None:
        """Reassess symbol for a new trade after all TP tiers were filled.

        When a trade hits all 3 TPs (a clean win), the same symbol is rescanned
        through the full Agent 1 → Agent 2 pipeline for a potential new position
        in either direction.  If the scanner finds no actionable signal, the
        symbol simply returns to normal scan rotation.
        """
        if not self.agent_analyst or not self.scanner:
            return

        # Only trigger when ALL TP tiers were filled
        if (
            not closed_position.tp_tiers
            or not all(getattr(t, "filled", False) for t in closed_position.tp_tiers)
        ):
            return

        logger.info(
            "post_win_reassessment_start",
            symbol=symbol,
            prev_direction=closed_position.direction,
            pnl_pct=round(pnl_pct, 2),
            tiers_hit=len(closed_position.tp_tiers),
        )

        try:
            # Fresh scan of just this symbol
            signals = await self.scanner.scan([symbol])
            if not signals:
                logger.info("post_win_reassessment_no_signal", symbol=symbol)
                return

            signal = signals[0]

            # ── Build Agent 1 context (mirrors _primary_tick inline block) ──
            pre_sl = self.order_executor._calculate_stop_loss(signal)
            pre_tp = None
            pre_rr = None
            if pre_sl:
                pre_tp = self.order_executor._calculate_take_profit(signal, pre_sl)
                sl_dist = abs(signal.entry_price - pre_sl)
                if sl_dist > 0 and pre_tp:
                    pre_rr = abs(pre_tp - signal.entry_price) / sl_dist

            # Symbol history for Agent 1 feedback
            symbol_history = []
            try:
                symbol_history = await self.repo.get_recent_trades_for_symbol(
                    signal.symbol, limit=5, mode=self.state.mode,
                )
            except Exception:
                pass

            ai_context = {
                "sentiment_score": None,
                "adjusted_score": signal.score,
                "active_threshold": self.adaptive_threshold.current_threshold,
                "sl_price": pre_sl,
                "tp_price": pre_tp,
                "rr_ratio": round(pre_rr, 2) if pre_rr else None,
                "open_position_count": len(self.portfolio.open_positions),
                "recent_headlines": None,
                "ml_win_probability": None,
                "symbol_history": symbol_history,
                # Reassessment context for Agent 1
                "post_win_reassessment": True,
                "previous_direction": closed_position.direction,
                "previous_pnl_pct": round(pnl_pct, 2),
            }

            # ── Run Agent 1 ──
            agent_result = await self.agent_analyst.analyze_signal(signal, ai_context)

            logger.info(
                "post_win_reassessment_decision",
                symbol=symbol,
                action=agent_result.action,
                confidence=agent_result.confidence,
                reasoning=agent_result.reasoning[:150],
            )

            if agent_result.action == "SKIP":
                return

            # ── SL/TP Priority: Agent 1 is primary, risk_manager is fallback ──
            agent_sl = agent_result.suggested_sl
            agent_tp = agent_result.suggested_tp
            new_direction = signal.direction or ""

            if agent_sl and agent_sl > 0:
                sl_valid = True
                if new_direction == "bullish" and agent_sl >= signal.entry_price:
                    sl_valid = False
                elif new_direction == "bearish" and agent_sl <= signal.entry_price:
                    sl_valid = False
                elif signal.entry_price > 0:
                    sl_dist = abs(signal.entry_price - agent_sl) / signal.entry_price
                    if sl_dist > self.config.max_sl_pct:
                        sl_valid = False
                if sl_valid:
                    pre_sl = agent_sl

            if agent_tp and agent_tp > 0:
                tp_valid = True
                if new_direction == "bullish" and agent_tp <= signal.entry_price:
                    tp_valid = False
                elif new_direction == "bearish" and agent_tp >= signal.entry_price:
                    tp_valid = False
                if tp_valid:
                    pre_tp = agent_tp

            # Recompute R:R if we now have both
            if pre_sl and pre_tp and signal.entry_price > 0:
                sl_dist = abs(signal.entry_price - pre_sl)
                if sl_dist > 0:
                    pre_rr = abs(pre_tp - signal.entry_price) / sl_dist

            # ── WAIT_PULLBACK → queue in refiner (same as primary tick) ──
            if agent_result.action == "WAIT_PULLBACK" and self.main_entry_refiner:
                if (
                    signal.symbol not in self.portfolio.open_positions
                    and signal.symbol not in self.main_entry_refiner.get_queued_symbols()
                ):
                    signal.original_1h_price = signal.entry_price
                    if agent_result.suggested_entry is not None:
                        signal.agent_target_entry = agent_result.suggested_entry
                    if agent_result.entry_zone_high is not None:
                        signal.agent_entry_zone_high = agent_result.entry_zone_high
                    if agent_result.entry_zone_low is not None:
                        signal.agent_entry_zone_low = agent_result.entry_zone_low

                    zone_h = agent_result.entry_zone_high
                    zone_l = agent_result.entry_zone_low
                    if not zone_h or not zone_l or zone_h <= zone_l or zone_h <= 0:
                        logger.warning(
                            "post_win_reassessment_invalid_zone",
                            symbol=symbol, zone_high=zone_h, zone_low=zone_l,
                        )
                        return

                    now = datetime.now(timezone.utc)
                    expiry_seconds = self.config.pullback_valid_candles * 5 * 60
                    invalidation = 0.0
                    if agent_result.invalidation_level and agent_result.invalidation_level > 0:
                        invalidation = agent_result.invalidation_level
                    elif signal.sweep_result and signal.sweep_result.sweep_detected:
                        invalidation = signal.sweep_result.sweep_level

                    sweep_dir = (
                        signal.sweep_result.sweep_direction
                        if signal.sweep_result and signal.sweep_result.sweep_direction
                        else signal.direction or ""
                    )
                    plan = PullbackPlan(
                        zone_low=zone_l,
                        zone_high=zone_h,
                        created_at=now,
                        expires_at=now + timedelta(seconds=expiry_seconds),
                        invalidation_level=invalidation,
                        max_chase_bps=self.config.pullback_max_chase_bps,
                        zone_tolerance_bps=self.config.pullback_zone_tolerance_bps,
                        valid_for_candles=self.config.pullback_valid_candles,
                        direction=sweep_dir,
                        original_suggested_entry=agent_result.suggested_entry or 0.0,
                        must_reach_price=agent_result.must_reach_price or 0.0,
                    )
                    plan.limit_price = plan.compute_limit_price()
                    signal.pullback_plan = plan
                    signal._pre_sl = pre_sl
                    signal._pre_tp = pre_tp

                    agent_analysis_data = {
                        "action": agent_result.action,
                        "confidence": agent_result.confidence,
                        "reasoning": agent_result.reasoning,
                        "market_regime": agent_result.market_regime,
                        "risk_assessment": agent_result.risk_assessment,
                        "suggested_entry": agent_result.suggested_entry,
                        "entry_zone_high": agent_result.entry_zone_high,
                        "entry_zone_low": agent_result.entry_zone_low,
                        "suggested_sl": agent_result.suggested_sl,
                        "suggested_tp": agent_result.suggested_tp,
                        "must_reach_price": agent_result.must_reach_price,
                        "invalidation_level": agent_result.invalidation_level,
                    }
                    signal.components["agent_analysis"] = agent_analysis_data
                    signal.components["sl_price"] = pre_sl
                    signal.components["tp_price"] = pre_tp
                    signal.components["rr_ratio"] = round(pre_rr, 2) if pre_rr else None

                    queued = self.main_entry_refiner.add(signal)
                    if queued:
                        logger.info(
                            "post_win_reassessment_queued",
                            symbol=symbol,
                            direction=signal.direction,
                            zone=f"{zone_l}-{zone_h}",
                            confidence=agent_result.confidence,
                        )

        except Exception as e:
            logger.warning(
                "post_win_reassessment_failed",
                symbol=symbol,
                error=str(e),
            )

    async def _maybe_run_advisor(self) -> None:
        """Run the trade advisor if it hasn't run today. Non-blocking background task."""
        try:
            from src.advisor.insights import get_recent_insights
            from src.advisor.runner import run_advisor

            # Check if already run today
            recent = await get_recent_insights(self.repo.db, self.config.instance_id, limit=1)
            if recent:
                from datetime import date

                last_run = recent[0].get("run_date")
                if last_run and str(last_run) == date.today().isoformat():
                    return  # Already ran today

            logger.info("advisor_daily_run_starting", instance=self.config.instance_id)
            result = await run_advisor(self.repo.db, instance_id=self.config.instance_id)
            logger.info(
                "advisor_daily_run_complete",
                cost_usd=result.get("cost_usd", 0),
                has_structured=bool(result.get("structured")),
            )
        except Exception as e:
            logger.warning("advisor_daily_run_failed", error=str(e))

    async def _persist_state(self) -> None:
        """Save engine state and portfolio snapshot to DB.

        Only the primary (crypto) engine writes full portfolio state.
        """
        try:
            # Non-crypto engines: skip full portfolio write
            if self._market_name != "crypto":
                return

            state_dict = self.portfolio.to_state_dict(
                status=self.state.status,
                mode=self.state.mode,
                cycle_count=self.state.cycle_count,
            )
            # Read existing overrides first to preserve other engines' state
            existing = await self.repo.get_engine_state()
            overrides = {}
            if existing:
                overrides = existing.get("config_overrides") or {}
                if not isinstance(overrides, dict):
                    overrides = {}
            # Update only the keys this engine manages
            if self.dynamic_weights:
                overrides["dynamic_weights"] = self.dynamic_weights.to_state()
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
            snapshot = self.portfolio.to_snapshot_dict(
                cycle_number=self.state.cycle_count,
                mode=self.state.mode,
            )
            # True equity = cash + deployed capital + unrealized P&L (mark-to-market)
            # Use prices cached from the latest monitor tick for accuracy
            monitor_prices = getattr(self.position_monitor, "last_prices", {}) or {}
            true_equity = self.portfolio.get_equity(current_prices=monitor_prices if monitor_prices else None)
            snapshot["balance_usd"] = true_equity
            # Drawdown = peak-to-trough decline. Only positive when equity is below peak.
            peak = self.portfolio.peak_balance
            if peak > 0 and true_equity < peak:
                snapshot["drawdown_pct"] = (peak - true_equity) / peak
            else:
                snapshot["drawdown_pct"] = 0.0
                # Update peak if equity has grown
                if true_equity > peak:
                    self.portfolio.peak_balance = true_equity
            await self.repo.insert_snapshot(snapshot)
        except Exception as e:
            logger.error("state_persist_failed", error=str(e))

    def _can_enter_symbol(self, symbol: str) -> bool:
        """Check if a symbol can be entered (not already open or being entered)."""
        if symbol in self.portfolio.open_positions:
            return False
        if symbol in self.state.open_positions:
            return False
        if symbol in self._entering_symbols:
            return False
        return True

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

            # Fetch current price so high_water_mark isn't reset to entry
            # on restart — prevents trailing stops from firing incorrectly
            entry_px = float(t.get("entry_price", 0))
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                current_px = float(ticker.get("last", entry_px))
            except Exception:
                current_px = entry_px
            direction_str = t.get("direction", "long")
            if direction_str == "long":
                hwm = max(entry_px, current_px)
            else:
                hwm = min(entry_px, current_px) if current_px > 0 else entry_px

            position = Position(
                trade_id=t["id"],
                symbol=symbol,
                direction=direction_str,
                entry_price=entry_px,
                quantity=qty,
                stop_loss=float(t.get("stop_loss", 0)),
                take_profit=float(t.get("take_profit")) if t.get("take_profit") else None,
                high_water_mark=hwm,
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
        if self.agent_analyst:
            await self.agent_analyst.close()
        if self.refiner_agent:
            await self.refiner_agent.close()
        if self.position_agent:
            await self.position_agent.close()
        logger.info("engine_shutdown_complete")
