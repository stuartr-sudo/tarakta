"""Standalone Market Makers Method trading engine.

This engine runs completely independently from the existing SMC/footprint
pipeline. It uses the 10 MM Method modules to make purely algorithmic
trading decisions with ZERO LLM/agent calls.

Architecture:
  - Runs as a parallel async task alongside the existing TradingEngine
  - Shares the exchange connection and database, but nothing else
  - Has its own scan cycle, entry logic, position management, and exit rules
  - All decisions are mechanical, based on the MM Method course rules

Flow per scan cycle:
  1. Session check — are we in a tradeable session?
  2. Weekly cycle update — what phase are we in?
  3. Per-pair analysis — formations, levels, EMA, confluence
  4. Entry decisions — purely algorithmic (no LLM)
  5. Position management — SL tightening, partial profits, exits

The engine tags all its trades with strategy='mm_method' so they can be
distinguished from SMC trades in the database and dashboard.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pandas as pd
from src.strategy.mm_board_meetings import BoardMeetingDetector
from src.strategy.mm_confluence import MMConfluenceScorer, MMContext
from src.strategy.mm_ema_framework import EMAFramework
from src.strategy.mm_formations import FormationDetector
from src.strategy.mm_levels import LevelTracker
from src.strategy.mm_risk import MMRiskCalculator
from src.strategy.mm_sessions import MMSessionAnalyzer
from src.strategy.mm_targets import TargetAnalyzer
from src.strategy.mm_weekly_cycle import WeeklyCycleTracker
from src.strategy.mm_weekend_trap import WeekendTrapAnalyzer
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Scan interval (minutes) — how often the MM engine runs
DEFAULT_SCAN_INTERVAL = 5

# Minimum confluence score (%) to consider an entry — course requires Grade C+ (40%)
MIN_CONFLUENCE_PCT = 40.0

# Minimum R:R ratio (to Level 1 target)
# Course lesson 53: 1.4 is the "don't get out of bed" floor — below that the
# trade isn't worth taking regardless of setup quality. 3.0 is the typical
# course default for new traders. We engine-default to 1.4 (the absolute
# floor per the course) and let users raise it via /mm/settings.
MIN_RR = 3.0
MIN_RR_COURSE_FLOOR = 1.4  # Lesson 53 "don't get out of bed" threshold
MIN_RR_AGGRESSIVE = 1.4    # was 1.0 — 1.0 is below the course floor

# Minimum formation quality to act on
MIN_FORMATION_QUALITY = 0.4

# Maximum concurrent MM positions (6 for bots — 3 was for human traders)
MAX_MM_POSITIONS = 6

# Maximum margin utilization — don't open new positions if margin > 60% of balance
MAX_MARGIN_UTILIZATION = 0.60

# Asia session range threshold — skip day if BTC range > 2%
ASIA_RANGE_SKIP_PCT = 2.0

# Maximum SL distance (%) — skip formations with structures too wide to trade
# A 5% wide formation means $100 risk only buys $2000 notional at 10x = $200 margin
# Below ~$200 margin, exchange fees eat the edge
MAX_SL_DISTANCE_PCT = 5.0

# Position sizing: risk per trade as % of balance
RISK_PER_TRADE_PCT = 1.0

# Cooldown after a trade closes on a symbol (hours) — prevents re-entry churn
SYMBOL_COOLDOWN_HOURS = 4

# Partial profit schedule (cumulative close %)
PROFIT_SCHEDULE = {
    1: 0.30,   # Close 30% at Level 1
    2: 0.50,   # Close to 50% total at Level 2
    3: 1.00,   # Close remaining at Level 3
}

# Valid phases for new entries (weekly cycle phase machine)
VALID_ENTRY_PHASES = {
    "FORMATION_PENDING", "LEVEL_1", "LEVEL_2", "LEVEL_3",
    "BOARD_MEETING_1", "BOARD_MEETING_2",
}

# Strategy tag for database
STRATEGY_TAG = "mm_method"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MMSignal:
    """A potential MM Method trade signal."""
    symbol: str = ""
    direction: str = ""         # "long" or "short"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_l1: float = 0.0
    target_l2: float = 0.0
    target_l3: float = 0.0
    risk_reward: float = 0.0

    # MM analysis results
    formation_type: str = ""     # "M" or "W"
    formation_variant: str = ""
    formation_quality: float = 0.0
    cycle_phase: str = ""
    current_level: int = 0
    confluence_score: float = 0.0
    confluence_grade: str = ""
    session_name: str = ""
    ema_alignment: str = ""
    fmwb_direction: str = ""
    weekend_bias: str = ""

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""

    # Course-faithful lifecycle carriers (2026-04 audit)
    # Course lesson 49 Refund Zone — only 2nd-peak aggressive entries use it
    entry_type: str = "conservative"  # "aggressive" (2nd-peak) or "conservative" (post-L1)
    peak2_wick_price: float = 0.0     # For W: 2nd-peak low wick; for M: 2nd-peak high wick


@dataclass
class MMPosition:
    """An open MM Method position with level tracking."""
    trade_id: str = ""
    symbol: str = ""
    direction: str = ""         # "long" or "short"
    entry_price: float = 0.0
    quantity: float = 0.0
    stop_loss: float = 0.0
    current_level: int = 0      # Track which level we're at
    last_level_checked: int = 0
    partial_closed_pct: float = 0.0  # How much has been closed (0-1)
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cost_usd: float = 0.0
    margin_used: float = 0.0

    # Targets
    target_l1: float = 0.0
    target_l2: float = 0.0
    target_l3: float = 0.0

    # Entry metadata (for dashboard display)
    entry_reason: str = ""
    formation_type: str = ""
    confluence_grade: str = ""
    cycle_phase: str = ""
    confluence_score: float = 0.0

    # Course-faithful lifecycle fields (2026-04 audit fix)
    # Course lesson 49 (Refund Zone): only applies to 2nd-peak aggressive entries.
    # If price CLOSES past the peak2 wick, cut the trade for a small loss.
    entry_type: str = "conservative"     # "aggressive" or "conservative"
    peak2_wick_price: float = 0.0        # For W: 2nd-peak low wick; for M: 2nd-peak high wick
    original_stop_loss: float = 0.0      # Immutable original SL for monotonic-SL enforcement
    sl_moved_to_breakeven: bool = False  # Track breakeven move after L2 starts
    sl_moved_under_50ema: bool = False   # Track "SL under 50 EMA once L2 running"


# ---------------------------------------------------------------------------
# MM Engine
# ---------------------------------------------------------------------------

class MMEngine:
    """Standalone algorithmic MM Method trading engine.

    Runs independently from the existing TradingEngine. Makes all decisions
    mechanically based on the MM Method rules — no LLM calls.
    """

    def __init__(
        self,
        exchange,
        repo,
        candle_manager,
        config=None,
        scan_interval_minutes: float = DEFAULT_SCAN_INTERVAL,
    ):
        self.exchange = exchange
        self.repo = repo
        self.candle_manager = candle_manager
        self.config = config
        self.scan_interval = scan_interval_minutes * 60  # Convert to seconds
        self.max_positions = getattr(config, "mm_max_positions", MAX_MM_POSITIONS)

        # Tunable parameters (overridable via settings page)
        self.risk_pct = RISK_PER_TRADE_PCT
        self.leverage = 10
        self.min_rr = MIN_RR_AGGRESSIVE
        self.min_confluence = MIN_CONFLUENCE_PCT
        self.min_formation_quality = MIN_FORMATION_QUALITY
        self.max_sl_pct = MAX_SL_DISTANCE_PCT

        # MM Method modules
        self.session_analyzer = MMSessionAnalyzer()
        self.ema_framework = EMAFramework()
        self.formation_detector = FormationDetector(session_analyzer=self.session_analyzer)
        self.level_tracker = LevelTracker(ema_framework=self.ema_framework)
        self.weekly_cycle_tracker = WeeklyCycleTracker()
        self.confluence_scorer = MMConfluenceScorer(min_rr=MIN_RR, min_score=MIN_CONFLUENCE_PCT)
        self.weekend_trap_analyzer = WeekendTrapAnalyzer()
        self.board_meeting_detector = BoardMeetingDetector()
        self.target_analyzer = TargetAnalyzer()
        self.risk_calculator = MMRiskCalculator(risk_per_trade=RISK_PER_TRADE_PCT / 100)

        # State
        self.positions: dict[str, MMPosition] = {}
        self._cooldowns: dict[str, datetime] = {}  # symbol -> earliest re-entry time
        self._cooldown_hours: float = SYMBOL_COOLDOWN_HOURS  # configurable via settings
        self._last_prices: dict[str, float] = {}  # symbol -> last known price (survives fetch failures)
        self._oi_cache: dict[str, float] = {}     # symbol -> last Open Interest reading (for rise/fall detection)

        # Course F3: pluggable data-feed registry (all stubs by default).
        # When real subscriptions are wired (Hyblock, TradingLite, Forex Factory,
        # basedmoney.io, CoinGecko for dominance, TradingView for correlation,
        # alternative.me for sentiment), swap stubs for real providers.
        from src.strategy.mm_data_feeds import DataFeedRegistry
        self.data_feeds = getattr(config, "mm_data_feeds", None) or DataFeedRegistry()

        # Course E1 (lesson 54): "if you have $10K OKX and $10K Binance,
        # 1% of $20K" — multi-exchange combined balance for 1% risk.
        # Expect config.mm_extra_exchanges: list of additional exchange clients
        # to sum alongside self.exchange. Empty by default = single-exchange.
        self.mm_extra_exchanges = getattr(config, "mm_extra_exchanges", []) or []
        self.cycle_count = 0
        self._running = True
        self._scanning_active = True  # MM Engine starts active (unlike main bot)

        # Per-cycle funnel state — all reset at start of each scan, reported in mm_scan_funnel.
        # Rejection reasons (why pairs drop out).
        self._scan_reject_counts: dict[str, int] = {}
        # Stage counters (how many pairs PASS each gate). Answers the positive
        # question: "are the scoring stages actually being reached?"
        self._scan_stage_counts: dict[str, int] = {}
        # Among pairs that reached the scoring stage, how often did each
        # confluence factor fire (score > 0). Proves the scoring factors are
        # actively being evaluated, not just the hard gates.
        self._scan_factor_hits: dict[str, int] = {}
        # Confluence score percentages seen in this cycle (pre-retest).
        # Used to compute min/avg/max/median + grade distribution.
        self._scan_score_samples: list[float] = []
        # Grade distribution (A/B/C/F) from this cycle's confluence calls.
        self._scan_grade_counts: dict[str, int] = {}
        # Retest-conditions-met distribution (0,1,2,3,4).
        self._scan_retest_counts: dict[int, int] = {}
        # Last funnel snapshot surfaced to the dashboard (updated at end of each scan)
        self.last_funnel: dict | None = None

        logger.info("mm_engine_initialized", scan_interval=scan_interval_minutes)

    def _reject(self, reason: str, symbol: str, **kwargs) -> None:
        """Log a rejection and increment the per-cycle funnel counter.

        Replaces the prior pattern ``logger.info("mm_reject_X", ...); return None``
        so ``mm_scan_funnel`` at end of scan can report how many pairs dropped
        off at each filter stage. Returns ``None`` so callers can ``return self._reject(...)``.
        """
        logger.info(f"mm_reject_{reason}", symbol=symbol, **kwargs)
        self._scan_reject_counts[reason] = self._scan_reject_counts.get(reason, 0) + 1
        return None

    def _advance(self, stage: str) -> None:
        """Increment the per-cycle stage counter when a pair passes a gate.

        Paired with ``_reject`` so the funnel shows BOTH sides: how many pairs
        dropped off at each reason (``_scan_reject_counts``) and how many
        advanced past each stage (``_scan_stage_counts``). This is what proves
        to the dashboard viewer that the scoring stages are actually being
        reached, not just the hard gates.
        """
        self._scan_stage_counts[stage] = self._scan_stage_counts.get(stage, 0) + 1

    def _try_three_hits_formation(self, candles_1h, cycle_state):
        """Course lesson-18 alternative: 3 hits at HOW/LOW at Level 3 replace M/W.

        Per lesson 18 (direct quote):
            "after three levels have completed. There are also other reversal
            signals that you could get, that would replace the M or W. If the
            Market Maker comes to test a weekly High or Low three times, and
            they don't break it, it's likely a reversal is imminent. This
            would also depend on price being at level three."

        Returns a synthesized :class:`Formation` when the conditions match, or
        ``None`` otherwise. The synthesized formation is consumed by the same
        pipeline as a real M/W so SL/target/confluence/retest gates still run.
        """
        from src.strategy.mm_formations import Formation  # local — avoids cycle

        # --- HOW 3-hits → bearish reversal → short setup (M-style) ---
        if cycle_state.how and cycle_state.how > 0:
            hits_how = self.formation_detector.detect_three_hits(candles_1h, cycle_state.how)
            if hits_how.detected and hits_how.expected_outcome == "reversal":
                # Course: the prior trend up to HOW must be a completed 3-level
                # swing. Measure bullish levels over the full recent window.
                level = self.level_tracker.analyze(candles_1h, direction="bullish")
                if level.current_level >= 3 and len(hits_how.hit_indices) >= 3:
                    mid = hits_how.hit_indices[len(hits_how.hit_indices) // 2]
                    return Formation(
                        type="M",
                        variant="three_hits_how",
                        peak1_idx=hits_how.hit_indices[0],
                        peak1_price=cycle_state.how,
                        peak2_idx=hits_how.hit_indices[-1],
                        peak2_price=cycle_state.how,
                        trough_idx=mid,
                        trough_price=cycle_state.how * 0.99,  # approximate
                        direction="bearish",
                        quality_score=0.5,   # synthetic — gate removed anyway
                        at_key_level=True,   # HOW is a key level by definition
                    )

        # --- LOW 3-hits → bullish reversal → long setup (W-style) ---
        if cycle_state.low and cycle_state.low < float("inf") and cycle_state.low > 0:
            hits_low = self.formation_detector.detect_three_hits(candles_1h, cycle_state.low)
            if hits_low.detected and hits_low.expected_outcome == "reversal":
                level = self.level_tracker.analyze(candles_1h, direction="bearish")
                if level.current_level >= 3 and len(hits_low.hit_indices) >= 3:
                    mid = hits_low.hit_indices[len(hits_low.hit_indices) // 2]
                    return Formation(
                        type="W",
                        variant="three_hits_low",
                        peak1_idx=hits_low.hit_indices[0],
                        peak1_price=cycle_state.low,
                        peak2_idx=hits_low.hit_indices[-1],
                        peak2_price=cycle_state.low,
                        trough_idx=mid,
                        trough_price=cycle_state.low * 1.01,
                        direction="bullish",
                        quality_score=0.5,
                        at_key_level=True,
                    )

        return None

    def begin_scanning(self) -> None:
        """Resume the MM engine's scan loop (called from dashboard)."""
        self._scanning_active = True
        logger.info("mm_engine_scanning_started")

    def stop_scanning(self) -> None:
        """Pause the MM engine's scan loop (keeps managing open positions)."""
        self._scanning_active = False
        logger.info("mm_engine_scanning_stopped")

    async def run(self) -> None:
        """Main loop — runs scan/trade/manage cycles."""
        # Restore settings from DB
        try:
            state = await self.repo.get_engine_state()
            if state:
                overrides = state.get("config_overrides", {}) or {}
                mm_settings = overrides.get("mm_engine_settings", {})
                if isinstance(mm_settings, dict):
                    if "scanning_active" in mm_settings:
                        self._scanning_active = bool(mm_settings["scanning_active"])
                    if "mm_max_positions" in mm_settings:
                        self.max_positions = int(mm_settings["mm_max_positions"])
                    if "mm_scan_interval" in mm_settings:
                        self.scan_interval = float(mm_settings["mm_scan_interval"]) * 60
                    if "mm_cooldown_hours" in mm_settings:
                        self._cooldown_hours = float(mm_settings["mm_cooldown_hours"])
                    if "mm_risk_pct" in mm_settings:
                        self.risk_pct = float(mm_settings["mm_risk_pct"])
                        self.risk_calculator = MMRiskCalculator(risk_per_trade=self.risk_pct / 100)
                    if "mm_leverage" in mm_settings:
                        self.leverage = int(mm_settings["mm_leverage"])
                    if "mm_min_rr" in mm_settings:
                        self.min_rr = float(mm_settings["mm_min_rr"])
                    if "mm_min_confluence" in mm_settings:
                        self.min_confluence = float(mm_settings["mm_min_confluence"])
                    if "mm_min_formation_quality" in mm_settings:
                        self.min_formation_quality = float(mm_settings["mm_min_formation_quality"])
                    if "mm_max_sl_pct" in mm_settings:
                        self.max_sl_pct = float(mm_settings["mm_max_sl_pct"])
                    logger.info("mm_engine_restored_settings",
                                scanning_active=self._scanning_active,
                                max_positions=self.max_positions,
                                scan_interval_min=self.scan_interval / 60)
        except Exception as e:
            logger.debug("mm_engine_state_restore_failed", error=str(e))

        # Restore open MM positions from DB (survives restarts).
        # Keep only the LATEST trade per symbol; close older duplicates as orphans.
        try:
            open_trades = await self.repo.get_open_trades()
            mm_trades = [t for t in open_trades if t.get("strategy") == STRATEGY_TAG]
            # Sort newest first so we keep the latest per symbol
            mm_trades.sort(key=lambda t: t.get("created_at", ""), reverse=True)
            restored = 0
            orphaned = 0
            for t in mm_trades:
                symbol = t["symbol"]
                if symbol in self.positions:
                    # Duplicate open trade for same symbol — close it as orphan
                    try:
                        await self.repo.update_trade(str(t["id"]), {
                            "status": "closed",
                            "exit_reason": "orphan_cleanup",
                            "pnl_usd": 0,
                            "exit_time": datetime.now(timezone.utc).isoformat(),
                        })
                        orphaned += 1
                    except Exception:
                        pass
                    continue
                orig_qty = t.get("original_quantity") or t.get("entry_quantity", 0)
                remain_qty = t.get("remaining_quantity") or orig_qty
                closed_pct = 1.0 - (remain_qty / orig_qty) if orig_qty > 0 else 0.0
                # Restore L2/L3 targets from tp_tiers JSON
                tp_l2, tp_l3 = 0.0, 0.0
                raw_tiers = t.get("tp_tiers")
                if raw_tiers:
                    try:
                        tiers = json.loads(raw_tiers) if isinstance(raw_tiers, str) else raw_tiers
                        tp_l2 = float(tiers.get("l2", 0))
                        tp_l3 = float(tiers.get("l3", 0))
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        pass
                self.positions[symbol] = MMPosition(
                    trade_id=str(t["id"]),
                    symbol=symbol,
                    direction=t.get("direction", "long"),
                    entry_price=float(t.get("entry_price", 0)),
                    quantity=float(remain_qty),
                    stop_loss=float(t.get("stop_loss", 0)),
                    current_level=int(t.get("current_tier") or 0),
                    partial_closed_pct=closed_pct,
                    entry_time=datetime.fromisoformat(t["entry_time"]) if t.get("entry_time") else datetime.now(timezone.utc),
                    cost_usd=float(t.get("entry_cost_usd", 0)),
                    margin_used=float(t.get("margin_used") or 0),
                    target_l1=float(t.get("take_profit", 0)),
                    target_l2=tp_l2,
                    target_l3=tp_l3,
                    entry_reason=t.get("entry_reason", ""),
                    formation_type=t.get("mm_formation", ""),
                    confluence_grade=t.get("mm_confluence_grade", ""),
                    cycle_phase=t.get("mm_cycle_phase", ""),
                    confluence_score=float(t.get("confluence_score") or 0),
                )
                restored += 1
            if mm_trades:
                logger.info("mm_positions_restored", restored=restored, orphaned=orphaned,
                            symbols=list(self.positions.keys()))
        except Exception as e:
            logger.warning("mm_position_restore_failed", error=str(e))

        logger.info("mm_engine_started", scanning_active=self._scanning_active)

        while self._running:
            try:
                await self._cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("mm_engine_cycle_error", error=str(e), exc_info=True)

            await asyncio.sleep(self.scan_interval)

        logger.info("mm_engine_stopped")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False

    async def get_status(self) -> dict:
        """Return a status snapshot for the dashboard API.

        Fetches live prices for all open positions in parallel, then
        falls back to cached prices if any individual fetch fails.
        """
        # Refresh prices for all open positions in parallel
        if self.positions:
            async def _fetch_price(symbol: str) -> tuple[str, float | None]:
                try:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    return symbol, float(ticker["last"])
                except Exception:
                    return symbol, None

            results = await asyncio.gather(
                *[_fetch_price(s) for s in self.positions], return_exceptions=True
            )
            for r in results:
                if isinstance(r, tuple) and r[1] is not None:
                    self._last_prices[r[0]] = r[1]

        session = self.session_analyzer.get_current_session()
        positions_out = []
        total_unrealized = 0.0
        for pos in self.positions.values():
            current_price = self._last_prices.get(pos.symbol, pos.entry_price)
            if pos.direction == "long":
                unrealized = (current_price - pos.entry_price) * pos.quantity
            else:
                unrealized = (pos.entry_price - current_price) * pos.quantity
            total_unrealized += unrealized
            positions_out.append({
                "symbol": pos.symbol,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "current_price": current_price,
                "stop_loss": pos.stop_loss,
                "current_level": pos.current_level,
                "partial_closed_pct": round(pos.partial_closed_pct * 100),
                "trade_id": pos.trade_id,
                "unrealized_pnl": round(unrealized, 4),
                "quantity": pos.quantity,
                "cost_usd": round(pos.cost_usd, 2),
                "margin_used": round(pos.margin_used, 2),
                "leverage": 10,
                "target_l1": pos.target_l1,
                "entry_reason": pos.entry_reason,
                "formation_type": pos.formation_type,
                "confluence_grade": pos.confluence_grade,
                "cycle_phase": pos.cycle_phase,
                "confluence_score": pos.confluence_score,
                "entry_time": pos.entry_time.isoformat() if pos.entry_time else "",
            })
        total_margin = sum(p.margin_used for p in self.positions.values())
        total_notional = sum(p.cost_usd for p in self.positions.values())
        return {
            "scanning_active": self._scanning_active,
            "running": self._running,
            "cycle_count": self.cycle_count,
            "open_positions": len(self.positions),
            "session": session.session_name if session else "unknown",
            "is_weekend": session.is_weekend if session else False,
            "total_unrealized_pnl": round(total_unrealized, 4),
            "total_margin_used": round(total_margin, 2),
            "total_notional": round(total_notional, 2),
            "positions": positions_out,
            "last_funnel": self.last_funnel,
        }

    async def _cycle(self) -> None:
        """One complete scan/trade/manage cycle."""
        self.cycle_count += 1
        now = datetime.now(timezone.utc)

        # 1. Session check
        session = self.session_analyzer.get_current_session()
        if session.is_weekend:
            logger.info("mm_engine_weekend_skip", cycle=self.cycle_count)
            return

        if session.session_name == "dead_zone":
            logger.info("mm_engine_dead_zone_skip", cycle=self.cycle_count)
            return

        logger.info(
            "mm_engine_cycle",
            cycle=self.cycle_count,
            session=session.session_name,
            is_gap=session.is_gap,
            positions=len(self.positions),
        )

        # 2. Manage existing positions FIRST (before scanning)
        #    This ensures SL exits happen before we consider new entries,
        #    preventing the scan-then-manage churn where a stopped-out
        #    position gets re-entered in the same cycle.
        for symbol in list(self.positions.keys()):
            try:
                await self._manage_position(symbol)
            except Exception as e:
                logger.warning("mm_manage_error", symbol=symbol, error=str(e))

        # 3. Only scan for new entries if scanning is active
        if self._scanning_active:
            # Get tradeable pairs
            try:
                pairs = await self._get_pairs()
            except Exception as e:
                logger.warning("mm_engine_pairs_error", error=str(e))
                pairs = []

            # 4. Scan each pair — reset ALL per-cycle telemetry so mm_scan_funnel
            # shows just this cycle's activity.
            self._scan_reject_counts = {}
            self._scan_stage_counts = {}
            self._scan_factor_hits = {}
            self._scan_score_samples = []
            self._scan_grade_counts = {}
            self._scan_retest_counts = {}
            exceptions_raised = 0
            signals: list[MMSignal] = []
            for pair in pairs:
                try:
                    signal = await self._analyze_pair(pair, session, now)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    exceptions_raised += 1
                    logger.warning("mm_analyze_pair_error", symbol=pair, error=str(e), error_type=type(e).__name__)

            # Funnel summary — shows exactly where symbols dropped off this cycle
            rejected_total = sum(self._scan_reject_counts.values())
            unaccounted = len(pairs) - rejected_total - len(signals) - exceptions_raised
            funnel_rejects = dict(sorted(self._scan_reject_counts.items(), key=lambda kv: -kv[1]))

            # Score distribution stats for pairs that reached the scoring stage
            scored_count = len(self._scan_score_samples)
            if scored_count:
                sorted_scores = sorted(self._scan_score_samples)
                score_stats = {
                    "count": scored_count,
                    "min": round(sorted_scores[0], 2),
                    "max": round(sorted_scores[-1], 2),
                    "avg": round(sum(sorted_scores) / scored_count, 2),
                    "median": round(sorted_scores[scored_count // 2], 2),
                }
            else:
                score_stats = {"count": 0, "min": 0, "max": 0, "avg": 0, "median": 0}

            # Stages in canonical pipeline order (matches _analyze_pair flow)
            stage_order = [
                "candles_ok", "formation_found", "level_ok", "phase_valid",
                "direction_ok", "target_acquired", "rr_passed", "scored",
                "confluence_passed", "retest_passed", "signal_built",
            ]
            stages = {k: self._scan_stage_counts.get(k, 0) for k in stage_order}

            funnel_factors = dict(sorted(self._scan_factor_hits.items(), key=lambda kv: -kv[1]))

            logger.info(
                "mm_scan_funnel",
                cycle=self.cycle_count,
                pairs_scanned=len(pairs),
                signals_found=len(signals),
                rejected_total=rejected_total,
                exceptions=exceptions_raised,
                unaccounted=unaccounted,
                rejects=funnel_rejects,
                stages=stages,
                factor_hits=funnel_factors,
                score_stats=score_stats,
                grades=dict(self._scan_grade_counts),
                retest_counts={str(k): v for k, v in sorted(self._scan_retest_counts.items())},
            )
            # Stash for dashboard — lets the UI render live selectivity per cycle
            self.last_funnel = {
                "cycle": self.cycle_count,
                "timestamp": now.isoformat(),
                "pairs_scanned": len(pairs),
                "signals_found": len(signals),
                "rejected_total": rejected_total,
                "exceptions": exceptions_raised,
                "unaccounted": unaccounted,
                "rejects": funnel_rejects,
                "stages": stages,
                "factor_hits": funnel_factors,
                "score_stats": score_stats,
                "grades": dict(self._scan_grade_counts),
                "retest_counts": {str(k): v for k, v in sorted(self._scan_retest_counts.items())},
            }
            # Keep mm_scan_summary for existing alerts/dashboards
            logger.info(
                "mm_scan_summary",
                pairs_scanned=len(pairs),
                signals_found=len(signals),
                cycle=self.cycle_count,
            )

            # 5. Rank signals and enter best ones
            if signals:
                signals.sort(key=lambda s: s.confluence_score, reverse=True)
                await self._process_entries(signals)

    async def _get_pairs(self) -> list[str]:
        """Get the list of pairs to scan."""
        pairs = await self.exchange.get_tradeable_pairs(
            quote_currencies=["USDT"],
            min_volume_usd=getattr(self.config, "min_volume_usd", 5_000_000),
        )
        now = datetime.now(timezone.utc)
        # Expire old cooldowns
        self._cooldowns = {s: t for s, t in self._cooldowns.items() if t > now}
        # Filter out pairs with open positions or on cooldown
        return [p for p in pairs if p not in self.positions and p not in self._cooldowns]

    async def _analyze_pair(
        self,
        symbol: str,
        session,
        now: datetime,
    ) -> MMSignal | None:
        """Run the full MM Method analysis on a single pair.

        Returns an MMSignal if entry conditions are met, None otherwise.
        """
        # Fetch candles (1H primary, 4H for EMA trend, 15m for entry refinement)
        # Course C1 (lesson 13): "drop down to the 15 minute time frame for
        # the actual Entry... then switch back to the one hour" — we fetch
        # 15m so final-damage hammer/inverted-hammer checks (lesson 21) and
        # other 15m-sensitive validations have data available.
        try:
            candles_1h = await self.candle_manager.get_candles(symbol, "1h", limit=500)
            candles_4h = await self.candle_manager.get_candles(symbol, "4h", limit=250)
            try:
                candles_15m = await self.candle_manager.get_candles(symbol, "15m", limit=200)
            except Exception:
                candles_15m = None  # best-effort — some markets won't have 15m
        except Exception as e:
            return self._reject("candle_fetch", symbol, error=str(e))

        if candles_1h is None or candles_1h.empty or len(candles_1h) < 50:
            return self._reject("insufficient_candles", symbol, count=0 if candles_1h is None or (hasattr(candles_1h, 'empty') and candles_1h.empty) else len(candles_1h))

        self._advance("candles_ok")

        # Course A8 (lesson 20 "wait for candle CLOSE before entering"):
        # Reject if the latest 1H bar hasn't closed yet. We use a 5-minute
        # tolerance because exchange candle endpoints sometimes return the
        # just-closed bar with a timestamp a few seconds before `now`.
        try:
            last_bar_ts = candles_1h.index[-1]
            # Convert to UTC aware datetime if needed
            if hasattr(last_bar_ts, "to_pydatetime"):
                last_bar_dt = last_bar_ts.to_pydatetime()
            else:
                last_bar_dt = last_bar_ts
            if last_bar_dt.tzinfo is None:
                last_bar_dt = last_bar_dt.replace(tzinfo=timezone.utc)
            # Bar close time = bar_start + 1h
            bar_close_dt = last_bar_dt + timedelta(hours=1)
            if now < bar_close_dt - timedelta(minutes=5):
                # Still inside the current 1H bar — don't score yet.
                return self._reject("bar_not_closed", symbol,
                                    bar_start=last_bar_dt.isoformat(),
                                    bar_close_due=bar_close_dt.isoformat())
        except Exception:
            pass  # Best-effort only — don't block on timestamp parsing issues

        current_price = float(candles_1h.iloc[-1]["close"])

        # Course A9 (lesson 9): midweek reversal for crypto happens Wed/Thu.
        # Reversal-only formations (three_hits_*, final_damage) on Mon/Tue are
        # fighting the primary weekly trend start. Subsumed by the
        # "no counter-trend after L1" gate below once the level tracker has
        # counted a level of move in the new week.

        # Course A3 (lesson 12): Asia range >2% for BTC → skip the whole day.
        # Only apply to BTC (course is specific) — don't penalize alts whose
        # Asia range is inherently wider.
        if "BTC" in symbol.upper().split("/")[0]:
            asia_range_pct = self._compute_asia_range_pct(candles_1h, now)
            if asia_range_pct is not None and asia_range_pct > ASIA_RANGE_SKIP_PCT:
                return self._reject("asia_range_too_wide", symbol,
                                    asia_range_pct=round(asia_range_pct, 2),
                                    threshold=ASIA_RANGE_SKIP_PCT)

        # --- Run all MM modules ---

        # EMA Framework (4H for macro trend)
        ema_state = None
        ema_values = {}
        if candles_4h is not None and not candles_4h.empty and len(candles_4h) > 200:
            ema_state = self.ema_framework.calculate(candles_4h)
            _trend_state = self.ema_framework.get_trend_state(candles_4h)  # noqa: F841 — kept for future use
            ema_values = ema_state.values
            ema_break = self.ema_framework.detect_ema_break(candles_4h, ema_period=50)
        else:
            ema_break = None

        # Weekly cycle — computed up front so both the formation path and the
        # lesson-18 three-hits alternative can use HOW/LOW and phase data.
        cycle_state = self.weekly_cycle_tracker.update(candles_1h, now)

        # Course F2 (lesson 29): Open Interest as trapped-trader detector.
        # Best-effort fetch — many alt perps support it via CCXT; if not we
        # leave the OI flag None and the confluence factor scores 0.
        oi_increasing: bool | None = None
        try:
            oi_data = await self.exchange.fetch_open_interest(symbol)
            # The CCXT shape is {openInterestAmount, timestamp, datetime, info, ...}
            # We compare against our cache from the previous cycle.
            if oi_data and "openInterestAmount" in oi_data:
                current_oi = float(oi_data["openInterestAmount"])
                prev_oi = self._oi_cache.get(symbol)
                if prev_oi is not None:
                    oi_increasing = current_oi > prev_oi * 1.005  # >0.5% rise = increasing
                self._oi_cache[symbol] = current_oi
        except Exception:
            pass  # OI not critical; confluence factor defaults to 0

        # Formation detection (1H)
        formations = self.formation_detector.detect(candles_1h)

        if formations:
            best_formation = formations[0]
        else:
            # Course lesson 18 alternative: 3 hits at HOW/LOW at Level 3
            # "replace the M or W". Synthesize a Formation-shaped object so
            # the rest of the pipeline works uniformly.
            best_formation = self._try_three_hits_formation(candles_1h, cycle_state)
            if best_formation is None:
                return self._reject("no_formation", symbol)
            logger.info("mm_three_hits_formation_synthesized",
                        symbol=symbol, type=best_formation.type,
                        variant=best_formation.variant,
                        level=cycle_state.how if best_formation.type == "M" else cycle_state.low)

        self._advance("formation_found")

        # Course B2 (lesson 21): Final Damage M/W must be a hammer (W) or
        # inverted hammer (M) on the 15m timeframe at the 2nd peak, otherwise
        # it's not a valid Final Damage signal.
        if best_formation.variant == "final_damage" and candles_15m is not None:
            if not self._final_damage_hammer_15m(
                best_formation=best_formation,
                candles_15m=candles_15m,
            ):
                return self._reject("final_damage_no_hammer_15m", symbol,
                                    formation_type=best_formation.type)

        # Course-faithful change (2026-04): the raw course (lesson 20) does not
        # score formation "quality" — an M/W either satisfies its three MM
        # appearances or it is not an M/W. quality_score is still used by
        # FormationDetector to rank candidates, but we no longer hard-reject
        # a weak-quality formation that passes the other course rules.

        # Level tracking (1H) — count levels AFTER the formation, not all history.
        # peak2_idx is relative to the last DEFAULT_LOOKBACK (40) candles,
        # so translate to the full DataFrame index.
        direction = best_formation.direction or "bullish"
        # Derive trade_direction early — used below in weekly bias gating before
        # SL/target calc redefines it. Must match: bullish (W) → long, bearish (M) → short.
        trade_direction = "long" if direction == "bullish" else "short"
        lookback_start = max(0, len(candles_1h) - 40)  # same window formation detector used
        formation_abs_idx = lookback_start + best_formation.peak2_idx
        candles_post_formation = candles_1h.iloc[formation_abs_idx:]
        level_analysis = self.level_tracker.analyze(candles_post_formation, direction=direction)

        # Don't enter if Level 3+ already reached (expect reversal)
        if level_analysis.current_level >= 3:
            return self._reject("level_too_advanced", symbol, level=level_analysis.current_level, post_formation_candles=len(candles_post_formation))

        self._advance("level_ok")

        # Course A4 (lesson 12): "We do not counter Trend Trade after level 1 Rise"
        # — called out as "the major one that you don't want to break".
        # If we're already past Level 1 in one direction, don't open a trade
        # in the OPPOSITE direction (unless an M/W reversal formation is the
        # explicit trigger — which we handle via the lesson-18 path / level=3
        # reject above). For the in-trend direction, this is a no-op.
        # Detection: examine the WHOLE-chart trend direction up to now.
        try:
            # "Chart-level" direction — count bullish vs bearish levels on the
            # full recent window, regardless of our formation.
            full_window_bullish = self.level_tracker.analyze(candles_1h.tail(60), direction="bullish")
            full_window_bearish = self.level_tracker.analyze(candles_1h.tail(60), direction="bearish")
            # If the prior direction is bullish past L1 and we're trying to short,
            # or bearish past L1 and we're trying to long — block it.
            if full_window_bullish.current_level >= 1 and trade_direction == "short":
                # Unless we have a 3-hit reversal or a high-confluence reversal formation,
                # the plain "short into an ongoing bullish move" path is blocked.
                # The lesson-18 three-hits path already sets formation.variant properly,
                # so we let those through. Same for Final Damage variants at HOW.
                if best_formation.variant not in ("three_hits_how", "final_damage", "multi_session") \
                   and not best_formation.at_key_level:
                    return self._reject("counter_trend_after_l1", symbol,
                                        trend_level=full_window_bullish.current_level,
                                        trade_dir=trade_direction,
                                        formation_variant=best_formation.variant)
            if full_window_bearish.current_level >= 1 and trade_direction == "long":
                if best_formation.variant not in ("three_hits_low", "final_damage", "multi_session") \
                   and not best_formation.at_key_level:
                    return self._reject("counter_trend_after_l1", symbol,
                                        trend_level=full_window_bearish.current_level,
                                        trade_dir=trade_direction,
                                        formation_variant=best_formation.variant)
        except Exception:
            pass  # Best-effort — don't block on tracker errors

        # Don't enter during FMWB (it's the false move)
        if cycle_state.phase == "FMWB":
            return self._reject("fmwb_phase", symbol, phase=cycle_state.phase)

        # Don't enter during Friday trap phase
        if cycle_state.phase == "FRIDAY_TRAP":
            return self._reject("friday_trap", symbol, phase=cycle_state.phase)

        # Bug 3: Phase machine enforcement — only enter during valid phases
        if cycle_state.phase not in VALID_ENTRY_PHASES:
            return self._reject("wrong_phase", symbol, phase=cycle_state.phase)

        self._advance("phase_valid")

        # Weekend trap analysis
        weekend = self.weekend_trap_analyzer.analyze(candles_1h, now)

        # Bug 6: Weekly bias gating — FMWB direction determines allowed trade direction
        # FMWB direction is the FALSE move. Real move is opposite.
        if weekend.fmwb.detected:
            # "up" false move → real direction bearish → only shorts
            # "down" false move → real direction bullish → only longs
            real_direction = "short" if weekend.fmwb.direction == "up" else "long"
            if trade_direction != real_direction:
                return self._reject("against_weekly_bias", symbol,
                                    trade_dir=trade_direction, fmwb_dir=weekend.fmwb.direction,
                                    real_dir=real_direction)
        else:
            logger.info("mm_warn_no_weekly_bias", symbol=symbol, direction=trade_direction)

        self._advance("direction_ok")

        # Bug 5: Three hits rule — check HOW and LOW for reversal/continuation signals
        three_hits_at_how = None
        three_hits_at_low = None
        if cycle_state.how > 0:
            three_hits_at_how = self.formation_detector.detect_three_hits(candles_1h, cycle_state.how)
            if three_hits_at_how.detected:
                log_event = "mm_signal_four_hits" if three_hits_at_how.hit_count >= 4 else "mm_signal_three_hits"
                logger.info(log_event, symbol=symbol, level=cycle_state.how,
                            hits=three_hits_at_how.hit_count, outcome=three_hits_at_how.expected_outcome)
        if cycle_state.low < float("inf"):
            three_hits_at_low = self.formation_detector.detect_three_hits(candles_1h, cycle_state.low)
            if three_hits_at_low.detected:
                log_event = "mm_signal_four_hits" if three_hits_at_low.hit_count >= 4 else "mm_signal_three_hits"
                logger.info(log_event, symbol=symbol, level=cycle_state.low,
                            hits=three_hits_at_low.hit_count, outcome=three_hits_at_low.expected_outcome)

        # Target identification
        target_analysis = self.target_analyzer.analyze(
            ohlc=candles_1h,
            direction=direction,
            entry_price=current_price,
            stop_loss=current_price * (0.99 if direction == "bullish" else 1.01),
            current_level=level_analysis.current_level,
            ema_values=ema_values,
            how=cycle_state.how if cycle_state.how > 0 else None,
            low=cycle_state.low if cycle_state.low < float("inf") else None,
        )

        # Calculate entry, SL, and targets
        entry_price = current_price

        # Stop loss placement per MM Method course rules (lesson 10, 20):
        # W-bottom (long):  "SL below the low of the candle PRECEDING the 1st spike, OR below LOD"
        # M-top (short):    "SL above the high of the candle PRECEDING the 1st spike, OR above HOD"
        # We look at peak1_idx-1 for the preceding candle's low (W) / high (M),
        # and fall back to the peak itself if the preceding candle is unavailable.
        peak1_abs_idx = lookback_start + best_formation.peak1_idx
        try:
            if peak1_abs_idx > 0 and peak1_abs_idx - 1 < len(candles_1h):
                preceding = candles_1h.iloc[peak1_abs_idx - 1]
            else:
                preceding = None
        except Exception:
            preceding = None

        if best_formation.type.upper() == "W":
            # Prefer the low of the PRECEDING candle (course rule).
            # Fallback: min of peak1/peak2 prices (previous behaviour).
            if preceding is not None:
                sl_ref = min(float(preceding["low"]),
                             best_formation.peak1_price,
                             best_formation.peak2_price)
            else:
                sl_ref = min(best_formation.peak1_price, best_formation.peak2_price)
            sl_price = sl_ref * 0.998  # 0.2% buffer below invalidation
            trade_direction = "long"
        else:
            if preceding is not None:
                sl_ref = max(float(preceding["high"]),
                             best_formation.peak1_price,
                             best_formation.peak2_price)
            else:
                sl_ref = max(best_formation.peak1_price, best_formation.peak2_price)
            sl_price = sl_ref * 1.002  # 0.2% buffer above invalidation
            trade_direction = "short"

        # Course-faithful change (2026-04): the course (lesson 53) is explicit —
        # "SL goes where it needs to go. NEVER tighten SL to improve R:R."
        # Risk is controlled by position sizing (1% of account / SL distance),
        # NOT by refusing to trade wide-SL setups. The previous 5% cap was a
        # code invention not found anywhere in the course. Logged for telemetry
        # only so we can see when SL is unusually wide.
        sl_distance_pct = abs(entry_price - sl_price) / entry_price * 100
        if sl_distance_pct > self.max_sl_pct:
            logger.info("mm_wide_sl_warning", symbol=symbol,
                        sl_distance_pct=round(sl_distance_pct, 2), threshold=self.max_sl_pct)

        # Targets from target analyzer
        t_l1 = target_analysis.primary_l1.price if target_analysis.primary_l1 else None
        t_l2 = target_analysis.primary_l2.price if target_analysis.primary_l2 else None
        t_l3 = target_analysis.primary_l3.price if target_analysis.primary_l3 else None

        # Course-faithful change (2026-04): the course target hierarchy (lesson
        # 47, spec section 8) is "50 EMA (primary) → 200 EMA (if 50 already
        # broken) → first unrecovered Vector candle". The prior implementation
        # rejected outright when primary_l1 was missing (e.g., when 50 EMA is
        # on the wrong side of price because it already broke). Instead, fall
        # back to the L2 target and use it as the effective L1.
        if not t_l1 and t_l2:
            logger.info("mm_l1_target_fallback_to_l2", symbol=symbol, entry=entry_price,
                        ema_50=ema_values.get(50), target_l2=t_l2,
                        formation=best_formation.type)
            t_l1 = t_l2

        # Only reject if NO target is available at any level (no EMAs, no
        # vectors, no HOW/LOW in direction).
        if not t_l1:
            return self._reject("no_target_available", symbol, direction=trade_direction,
                                ema_50=ema_values.get(50), ema_200=ema_values.get(200),
                                vectors=len(target_analysis.unrecovered_vectors),
                                entry=entry_price, formation=best_formation.type)

        self._advance("target_acquired")

        # R:R check — try L1 first, fall back to L2 if L1 R:R is too low
        risk = abs(entry_price - sl_price)
        if risk <= 0:
            return self._reject("zero_risk", symbol)

        reward = abs(t_l1 - entry_price)
        rr = reward / risk

        # If L1 target gives poor R:R, try L2 as the primary target
        if rr < self.min_rr and t_l2 and self._is_valid_target(t_l2, trade_direction, entry_price):
            reward_l2 = abs(t_l2 - entry_price)
            rr_l2 = reward_l2 / risk
            if rr_l2 >= self.min_rr:
                logger.info("mm_target_upgraded_to_l2", symbol=symbol, rr_l1=round(rr, 2), rr_l2=round(rr_l2, 2))
                t_l1 = t_l2  # Use L2 as the effective target
                rr = rr_l2

        if rr < self.min_rr:
            return self._reject("low_rr", symbol, rr=round(rr, 2), min_required=self.min_rr, entry=entry_price, sl=sl_price, t1=t_l1)

        self._advance("rr_passed")

        # Confluence scoring
        at_how = abs(current_price - cycle_state.how) / current_price < 0.005 if cycle_state.how > 0 else False
        at_low = abs(current_price - cycle_state.low) / current_price < 0.005 if cycle_state.low < float("inf") else False

        # Bug 5: Three hits boosts at_key_level — 3-hit reversal at HOW/LOW strengthens confluence
        three_hit_boost = False
        if three_hits_at_how and three_hits_at_how.detected and three_hits_at_how.expected_outcome == "reversal":
            three_hit_boost = True
        if three_hits_at_low and three_hits_at_low.detected and three_hits_at_low.expected_outcome == "reversal":
            three_hit_boost = True

        mm_ctx = MMContext(
            formation={
                "type": best_formation.type,
                "variant": best_formation.variant,
                "quality": best_formation.quality_score,
                "at_key_level": best_formation.at_key_level,
                "session": session.session_name,
            },
            ema_state={
                "alignment": ema_state.alignment if ema_state else "mixed",
                "broke_50": ema_break.broke_ema if ema_break else False,
                "volume_confirmed": ema_break.volume_confirmed if ema_break else False,
            } if ema_state else None,
            level_state={
                "current_level": level_analysis.current_level,
                "svc_detected": level_analysis.svc.detected if level_analysis.svc else False,
                "volume_degrading": level_analysis.volume_degrading,
            },
            cycle_state={
                "phase": cycle_state.phase,
                "direction": cycle_state.direction,
            },
            entry_price=entry_price,
            stop_loss=sl_price,
            target_price=t_l1,
            at_session_changeover=session.is_gap,
            at_how_low=at_how or at_low or three_hit_boost,
            has_unrecovered_vector=len(target_analysis.unrecovered_vectors) > 0,
            moon_phase_aligned=self._moon_phase_aligned(trade_direction, now),
            oi_increasing=oi_increasing,
        )

        confluence_result = self.confluence_scorer.score(mm_ctx)
        self._advance("scored")

        # Telemetry: record which confluence factors actually fired (score > 0)
        # plus the score percentage, grade, and retest-conditions-met count.
        # This is the "other areas are being considered" proof — the dashboard
        # can render factor hit rates and score distribution from this.
        for factor_name, factor_score in confluence_result.factors.items():
            if factor_score > 0:
                self._scan_factor_hits[factor_name] = self._scan_factor_hits.get(factor_name, 0) + 1
        self._scan_score_samples.append(confluence_result.score_pct)
        self._scan_grade_counts[confluence_result.grade] = self._scan_grade_counts.get(confluence_result.grade, 0) + 1
        self._scan_retest_counts[confluence_result.retest_conditions_met] = \
            self._scan_retest_counts.get(confluence_result.retest_conditions_met, 0) + 1

        # Check confluence meets minimum
        if confluence_result.score_pct < self.min_confluence:
            return self._reject("low_confluence", symbol, score=confluence_result.score_pct, min_required=self.min_confluence, formation=best_formation.type)

        self._advance("confluence_passed")

        # Bug 1: Retest conditions — course requires 2+ of 4 retest conditions met
        if confluence_result.retest_conditions_met < 2:
            return self._reject("low_retest", symbol, retest_met=confluence_result.retest_conditions_met, confluence=confluence_result.score_pct)

        self._advance("retest_passed")

        # Course-faithful: determine entry type & peak2 wick for Refund Zone
        # Aggressive = 2nd-peak entry (refund zone applies).
        # Conservative = post-L1 retest entry (refund zone NOT applicable per lesson 49).
        # We classify post-50-EMA-break as conservative; otherwise aggressive.
        is_post_l1 = ema_break is not None and getattr(ema_break, "broke_ema", False)
        entry_type = "conservative" if is_post_l1 else "aggressive"
        # peak2 wick: for W, the LOW wick of 2nd peak; for M, the HIGH wick.
        # best_formation.peak2_idx is relative to the 40-bar lookback.
        try:
            peak2_abs_idx = lookback_start + best_formation.peak2_idx
            peak2_candle = candles_1h.iloc[peak2_abs_idx]
            if best_formation.type.upper() == "W":
                peak2_wick_price = float(peak2_candle["low"])
            else:
                peak2_wick_price = float(peak2_candle["high"])
        except Exception:
            peak2_wick_price = float(best_formation.peak2_price)

        # Build signal
        signal = MMSignal(
            symbol=symbol,
            direction=trade_direction,
            entry_price=entry_price,
            stop_loss=sl_price,
            target_l1=t_l1 or 0,
            target_l2=t_l2 or 0,
            target_l3=t_l3 or 0,
            risk_reward=round(rr, 2),
            formation_type=best_formation.type,
            formation_variant=best_formation.variant,
            formation_quality=round(best_formation.quality_score, 3),
            cycle_phase=cycle_state.phase,
            current_level=level_analysis.current_level,
            confluence_score=round(confluence_result.total_score, 1),
            confluence_grade=confluence_result.grade,
            session_name=session.session_name,
            ema_alignment=ema_state.alignment if ema_state else "mixed",
            fmwb_direction=weekend.fmwb.direction if weekend.fmwb.detected else "",
            weekend_bias=weekend.bias,
            entry_type=entry_type,
            peak2_wick_price=peak2_wick_price,
            reason=f"{best_formation.type} formation ({best_formation.variant}) "
                   f"grade={confluence_result.grade} R:R={rr:.1f} "
                   f"phase={cycle_state.phase} entry={entry_type}",
        )

        logger.info(
            "mm_signal_generated",
            symbol=symbol,
            direction=trade_direction,
            formation=best_formation.type,
            grade=confluence_result.grade,
            rr=round(rr, 2),
            confluence=round(confluence_result.score_pct, 1),
        )
        self._advance("signal_built")

        return signal

    async def _process_entries(self, signals: list[MMSignal]) -> None:
        """Process entry signals — execute the best ones."""
        open_count = len(self.positions)

        # Get balance for margin utilization check
        try:
            balance = await self.exchange.get_balance()
            account_balance = balance.get("USDT", 0) or balance.get("USD", 0)
            account_balance = await self._combined_balance(account_balance)
        except Exception:
            account_balance = 0

        for signal in signals:
            if open_count >= self.max_positions:
                break

            # Check in-memory positions AND cooldowns
            if signal.symbol in self.positions:
                continue
            if signal.symbol in self._cooldowns:
                continue

            # Margin utilization check — don't exceed 60% of balance
            if account_balance > 0:
                total_margin = sum(p.margin_used for p in self.positions.values())
                utilization = total_margin / account_balance
                if utilization > MAX_MARGIN_UTILIZATION:
                    logger.info("mm_reject_margin_limit", symbol=signal.symbol,
                                utilization_pct=round(utilization * 100, 1),
                                max_pct=round(MAX_MARGIN_UTILIZATION * 100, 0))
                    break  # Stop trying — all remaining signals would also fail

            try:
                await self._enter_trade(signal)
                open_count += 1
            except Exception as e:
                logger.warning("mm_entry_failed", symbol=signal.symbol, error=str(e))

    async def _enter_trade(self, signal: MMSignal) -> None:
        """Execute an MM Method trade entry."""
        # Get balance for position sizing — course E1 (lesson 54): 1% of TOTAL
        # trading account across all exchanges.
        balance = await self.exchange.get_balance()
        account_balance = balance.get("USDT", 0) or balance.get("USD", 0)
        account_balance = await self._combined_balance(account_balance)

        if account_balance <= 0:
            logger.info("mm_entry_skip_no_balance", symbol=signal.symbol, balance=account_balance)
            return

        # Calculate position size (1% risk)
        pos_result = self.risk_calculator.calculate_position_size(
            account_balance_usd=account_balance,
            entry_price=signal.entry_price,
            stop_loss_price=signal.stop_loss,
        )

        if not pos_result.is_viable or pos_result.position_size_usd < 10:
            logger.info("mm_position_too_small", symbol=signal.symbol, size=pos_result.position_size_usd, viable=pos_result.is_viable, balance=account_balance)
            return

        # Calculate quantity
        quantity = pos_result.position_size_usd / signal.entry_price

        # Place order
        side = "buy" if signal.direction == "long" else "sell"
        try:
            result = await self.exchange.place_market_order(
                symbol=signal.symbol,
                side=side,
                quantity=quantity,
            )
        except Exception as e:
            logger.warning("mm_order_failed", symbol=signal.symbol, error=str(e))
            return

        if not result or result.status != "closed":
            logger.info("mm_order_not_filled", symbol=signal.symbol, status=getattr(result, 'status', None))
            return

        fill_price = result.avg_price or signal.entry_price
        cost_usd = result.filled_quantity * fill_price

        # Log to database first to get the DB-generated id
        trade_id = str(uuid4())  # valid UUID fallback if DB insert fails
        risk_usd = abs(fill_price - signal.stop_loss) * result.filled_quantity
        margin = pos_result.position_size_usd / pos_result.recommended_leverage
        try:
            db_row = await self.repo.insert_trade({
                "symbol": signal.symbol,
                "direction": signal.direction,
                "entry_price": fill_price,
                "entry_quantity": result.filled_quantity,
                "original_quantity": result.filled_quantity,
                "remaining_quantity": result.filled_quantity,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.target_l1,
                "tp_tiers": json.dumps({"l2": signal.target_l2, "l3": signal.target_l3}),
                "margin_used": round(margin, 2),
                "entry_cost_usd": cost_usd,
                "risk_usd": round(risk_usd, 2),
                "leverage": getattr(self.config, 'markets', {}).get('crypto', None) and self.config.markets['crypto'].leverage or 10,
                "instance_id": getattr(self.config, 'instance_id', 'footprint'),
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "strategy": STRATEGY_TAG,
                "entry_reason": signal.reason,
                "confluence_score": signal.confluence_score,
                "mm_formation": signal.formation_type,
                "mm_cycle_phase": signal.cycle_phase,
                "mm_confluence_grade": signal.confluence_grade,
                "mode": self.config.trading_mode if hasattr(self.config, 'trading_mode') else "paper",
                "status": "open",
            })
            if db_row and db_row.get("id"):
                trade_id = str(db_row["id"])
        except Exception as e:
            logger.warning("mm_trade_db_insert_failed", symbol=signal.symbol, error=str(e))

        # Create MM position with DB id for later updates
        # Course-faithful lifecycle fields captured from signal so we can
        # enforce Refund Zone (lesson 49) and monotonic SL (lesson 51).
        position = MMPosition(
            trade_id=trade_id,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=fill_price,
            quantity=result.filled_quantity,
            stop_loss=signal.stop_loss,
            current_level=0,
            cost_usd=cost_usd,
            margin_used=margin,
            target_l1=signal.target_l1,
            target_l2=signal.target_l2,
            target_l3=signal.target_l3,
            entry_reason=signal.reason,
            formation_type=signal.formation_type,
            confluence_grade=signal.confluence_grade,
            cycle_phase=signal.cycle_phase,
            confluence_score=signal.confluence_score,
            # Lifecycle tracking
            entry_type=getattr(signal, "entry_type", "conservative"),
            peak2_wick_price=getattr(signal, "peak2_wick_price", 0.0),
            original_stop_loss=signal.stop_loss,
        )

        self.positions[signal.symbol] = position

        logger.info(
            "mm_trade_entered",
            trade_id=trade_id,
            symbol=signal.symbol,
            direction=signal.direction,
            entry=fill_price,
            sl=signal.stop_loss,
            tp1=signal.target_l1,
            rr=signal.risk_reward,
            grade=signal.confluence_grade,
            formation=signal.formation_type,
            cost=round(position.cost_usd, 2),
        )

    async def _manage_position(self, symbol: str) -> None:
        """Manage an existing MM position — SL, partials, exits."""
        pos = self.positions.get(symbol)
        if not pos:
            return

        # Get current price
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = float(ticker["last"])
            self._last_prices[symbol] = current_price
        except Exception:
            return

        # Course-faithful: REFUND ZONE check (lesson 49).
        # Only for 2nd-peak aggressive entries. If price CLOSES past the 2nd
        # peak's wick, cut immediately at a small loss rather than riding to SL.
        # Requires a closed candle — check last 1h bar close.
        if pos.entry_type == "aggressive" and pos.peak2_wick_price > 0 and pos.current_level == 0:
            try:
                candles_short = await self.candle_manager.get_candles(symbol, "1h", limit=5)
                if candles_short is not None and not candles_short.empty and len(candles_short) >= 2:
                    # Use the PENULTIMATE candle's close — guaranteed closed
                    last_close = float(candles_short.iloc[-2]["close"])
                    refund_check = self.risk_calculator.check_refund_zone(
                        entry_price=pos.entry_price,
                        current_price=last_close,
                        formation_type=pos.formation_type,
                        peak2_wick_price=pos.peak2_wick_price,
                    )
                    if refund_check.should_cut:
                        logger.info("mm_refund_zone_cut", symbol=symbol,
                                    entry=pos.entry_price, last_close=last_close,
                                    peak2_wick=pos.peak2_wick_price,
                                    formation=pos.formation_type,
                                    loss_pct=refund_check.loss_pct)
                        await self._close_position(pos, current_price, "refund_zone")
                        return
            except Exception as e:
                logger.debug("mm_refund_zone_check_failed", symbol=symbol, error=str(e))

        # Check stop loss hit
        if self._is_stopped_out(pos, current_price):
            await self._close_position(pos, current_price, "stop_loss")
            return

        # Bug 2: Target-based level advancement — check if price has reached targets.
        # This complements the PVSRA vector-based level tracker which needs time
        # to accumulate enough candles. Price hitting the EMA target IS the level.
        target_level = pos.current_level
        is_long = pos.direction == "long"
        if pos.target_l1 and target_level < 1:
            if (is_long and current_price >= pos.target_l1) or (not is_long and current_price <= pos.target_l1):
                target_level = 1
                logger.info("mm_level_target_hit", symbol=symbol, level=1, target=pos.target_l1, price=current_price)
        if pos.target_l2 and target_level < 2:
            if (is_long and current_price >= pos.target_l2) or (not is_long and current_price <= pos.target_l2):
                target_level = 2
                logger.info("mm_level_target_hit", symbol=symbol, level=2, target=pos.target_l2, price=current_price)
        if pos.target_l3 and target_level < 3:
            if (is_long and current_price >= pos.target_l3) or (not is_long and current_price <= pos.target_l3):
                target_level = 3
                logger.info("mm_level_target_hit", symbol=symbol, level=3, target=pos.target_l3, price=current_price)

        # Fetch fresh candles for PVSRA level analysis
        try:
            candles_1h = await self.candle_manager.get_candles(symbol, "1h", limit=100)
        except Exception:
            candles_1h = None

        # PVSRA vector-based level tracker (complementary to target-based)
        new_level = target_level
        if candles_1h is not None and not candles_1h.empty:
            direction = "bullish" if is_long else "bearish"
            if pos.entry_time:
                candles_since_entry = candles_1h[candles_1h.index >= pos.entry_time] if hasattr(candles_1h.index, 'tz') else candles_1h
                if candles_since_entry.empty or len(candles_since_entry) < 3:
                    candles_since_entry = candles_1h.tail(10)
            else:
                candles_since_entry = candles_1h.tail(10)
            level_analysis = self.level_tracker.analyze(candles_since_entry, direction=direction)
            # Use the higher of target-based and PVSRA-based level
            new_level = max(target_level, level_analysis.current_level)
        else:
            level_analysis = None

        # Level progression — take partials and tighten SL
        if new_level > pos.current_level:
            logger.info(
                "mm_level_advanced",
                symbol=symbol,
                old_level=pos.current_level,
                new_level=new_level,
            )

            # Take partial profit
            await self._take_partial(pos, new_level, current_price)

            # Tighten SL per MM rules
            if candles_1h is not None:
                self._tighten_sl(pos, new_level, candles_1h)

            # Bug 4: Board meeting detection after level advance (logging only for now)
            # Check BEFORE updating pos.current_level so the condition fires.
            if candles_1h is not None and new_level in (1, 2):
                direction_str = "bullish" if is_long else "bearish"
                bm_detection = self.board_meeting_detector.detect(candles_1h, level_direction=direction_str)
                if bm_detection.detected:
                    logger.info("mm_board_meeting_detected", symbol=symbol,
                                level_before=new_level, bm_type=bm_detection.bm_type,
                                duration=bm_detection.duration_candles,
                                stop_hunt=bm_detection.stop_hunt_detected,
                                has_entry=bm_detection.entry is not None,
                                fib_entry=bm_detection.entry.entry_price if bm_detection.entry else None,
                                fib_sl=bm_detection.entry.stop_loss if bm_detection.entry else None,
                                fib_rr=bm_detection.entry.risk_reward if bm_detection.entry else None)

            pos.current_level = new_level

            # Persist level advance + SL change to DB
            try:
                await self.repo.update_trade(pos.trade_id, {
                    "stop_loss": pos.stop_loss,
                    "current_tier": new_level,
                })
            except Exception as e:
                logger.debug("mm_level_db_update_failed", error=str(e))

        # Check for Stopping Volume Candle at Level 3
        if new_level >= 3 and level_analysis and level_analysis.svc and level_analysis.svc.detected:
            logger.info("mm_svc_detected", symbol=symbol, level=new_level)
            # Close remaining position
            if pos.partial_closed_pct < 1.0:
                await self._close_position(pos, current_price, "svc_level_3")
                return

        # Check Friday UK session exit
        session = self.session_analyzer.get_current_session()
        if session.session_name == "uk" and session.day_of_week == 4:  # Friday
            if new_level >= 2:
                logger.info("mm_friday_uk_exit", symbol=symbol)
                await self._close_position(pos, current_price, "friday_uk_exit")
                return

        # Volume degradation at Level 3 = exit
        if new_level >= 3 and level_analysis and level_analysis.volume_degrading:
            logger.info("mm_volume_degradation_exit", symbol=symbol)
            await self._close_position(pos, current_price, "volume_degradation")
            return

    def _is_stopped_out(self, pos: MMPosition, current_price: float) -> bool:
        """Check if position has hit its stop loss."""
        if pos.direction == "long":
            return current_price <= pos.stop_loss
        else:
            return current_price >= pos.stop_loss

    def _tighten_sl(self, pos: MMPosition, new_level: int, candles_1h: pd.DataFrame) -> None:
        """Tighten stop loss per MM Method level rules (course lessons 47, 48).

        Course rules:
        - Level 1 complete: SL stays at entry/structure (lesson 47: "only after
          retracement can you move stop loss to break even").
        - Level 2 starts: move SL to breakeven (entry price).
        - Level 2 running: SL just under the 50 EMA (lesson 48 / spec §7).
        - Level 3: SL trails under recent structure.

        Monotonic SL enforcement (lesson 51: "never increase your stop loss
        because that'll mean you'll lose more money"): a new SL proposal is
        only accepted if it TIGHTENS the stop (reduces risk), never widens it.
        """
        def _tightens(new_sl: float) -> bool:
            """Return True if new_sl is a tightening move (reduces risk)."""
            if pos.direction == "long":
                return new_sl > pos.stop_loss
            return new_sl < pos.stop_loss

        def _apply(new_sl: float, reason: str) -> bool:
            """Apply SL move if it tightens AND doesn't cross current price."""
            # Safety: never move SL to the wrong side of current price
            # (would be an instant stop-out).
            if pos.direction == "long" and new_sl >= pos.entry_price * 1.1:
                return False  # sanity cap: don't move SL absurdly high on long
            if pos.direction == "short" and new_sl <= pos.entry_price * 0.9:
                return False
            if _tightens(new_sl):
                old = pos.stop_loss
                pos.stop_loss = new_sl
                logger.info("mm_sl_tightened", symbol=pos.symbol,
                            level=new_level, reason=reason,
                            old_sl=old, new_sl=new_sl,
                            direction=pos.direction)
                return True
            return False

        if new_level <= 1:
            # Lesson 47: "A lot of times, I'll hear people move their stop loss
            # before they've hit their initial target, that is wrong"
            return

        if new_level == 2:
            # (a) Move SL to breakeven first if we haven't yet
            if not pos.sl_moved_to_breakeven:
                if _apply(pos.entry_price, "breakeven_at_l2"):
                    pos.sl_moved_to_breakeven = True

            # (b) Once at breakeven, try to move SL just under the 50 EMA
            # Lesson 48: "Once Level 2 is running: can place SL just under 50 EMA"
            if pos.sl_moved_to_breakeven and not pos.sl_moved_under_50ema:
                ema_50 = self._compute_50ema(candles_1h)
                if ema_50 is not None:
                    if pos.direction == "long":
                        candidate_sl = ema_50 * 0.998  # 0.2% buffer below
                    else:
                        candidate_sl = ema_50 * 1.002  # 0.2% buffer above
                    if _apply(candidate_sl, "under_50ema_at_l2"):
                        pos.sl_moved_under_50ema = True

        elif new_level >= 3:
            # Tighten under/above recent structure (last 10 candles)
            recent = candles_1h.tail(10)
            if pos.direction == "long":
                new_sl = float(recent["low"].min()) * 0.998
                _apply(new_sl, f"l{new_level}_recent_structure")
            else:
                new_sl = float(recent["high"].max()) * 1.002
                _apply(new_sl, f"l{new_level}_recent_structure")

    def _detect_ema_flatten(self, candles: pd.DataFrame, lookback: int = 20) -> bool:
        """Course D1 (lesson 24): EMAs flattening = trend ending signal.

        Detect if the 50 EMA's slope over the last `lookback` bars is near
        zero (less than 0.1% change). Returns True if EMAs are flattening.
        """
        if candles is None or candles.empty or len(candles) < 50 + lookback:
            return False
        try:
            ema50 = candles["close"].ewm(span=50, adjust=False).mean()
            recent = ema50.tail(lookback)
            start, end = float(recent.iloc[0]), float(recent.iloc[-1])
            if start <= 0:
                return False
            slope_pct = abs(end - start) / start * 100
            return slope_pct < 0.1  # less than 0.1% change over lookback
        except Exception:
            return False

    def _detect_ema_fan_out(self, candles: pd.DataFrame) -> bool:
        """Course D2 (lesson 24): EMAs fanning out = trend acceleration / L3 hint.

        "trend acceleration... EMAs fan out... price moves away from the EMAs".
        Detect when:
          (a) the 10-200 EMA spread (normalised by price) is > 2% today, AND
          (b) that is a material expansion vs recent history
              (current > max(prior_median * 2, 0.5% of price)).
        """
        if candles is None or candles.empty or len(candles) < 250:
            return False
        try:
            ema10 = candles["close"].ewm(span=10, adjust=False).mean()
            ema200 = candles["close"].ewm(span=200, adjust=False).mean()
            spread = (ema10 - ema200).abs()
            current = float(spread.iloc[-1])
            last_price = float(candles["close"].iloc[-1])
            if last_price <= 0:
                return False
            current_pct = current / last_price
            # Use prior window MEDIAN as the baseline. If prior is near-zero
            # (flat period), use 0.5% of price as a practical floor so we don't
            # always trip False on "came from zero" data.
            prior_median = float(spread.iloc[-200:-50].median())
            baseline = max(prior_median * 2.0, 0.005 * last_price)
            wide_today = current_pct > 0.02
            grew = current > baseline
            return bool(wide_today and grew)
        except Exception:
            return False

    def _compute_50ema(self, candles_1h: pd.DataFrame) -> float | None:
        """Compute the 50 EMA of the provided 1H candles. None if insufficient data."""
        if candles_1h is None or candles_1h.empty or len(candles_1h) < 50:
            return None
        try:
            ema = candles_1h["close"].ewm(span=50, adjust=False).mean()
            return float(ema.iloc[-1])
        except Exception:
            return None

    async def _combined_balance(self, primary_balance: float) -> float:
        """Course E1 (lesson 54): sum USDT balance across ALL configured
        trading-portfolio exchanges. 1% risk is computed against the combined
        total per the course.

        Falls back to the primary balance if extra exchanges aren't configured
        or their balance fetch fails.
        """
        total = float(primary_balance or 0.0)
        for exch in self.mm_extra_exchanges:
            try:
                bal = await exch.get_balance()
                total += float(bal.get("USDT", 0) or bal.get("USD", 0) or 0.0)
            except Exception as e:
                logger.debug("mm_extra_exchange_balance_failed", error=str(e))
        return total

    def _final_damage_hammer_15m(self, best_formation, candles_15m) -> bool:
        """Course B2 (lesson 21): Final Damage M/W requires hammer/inverted
        hammer on the 15m timeframe at the 2nd peak.

        W Final Damage → require hammer (long lower wick, close near high)
        M Final Damage → require inverted hammer (long upper wick, close near low)
        """
        try:
            from src.strategy.mm_formations import _is_hammer, _is_inverted_hammer
            # Get the most recent 3 15m candles (the 2nd-peak area)
            if candles_15m is None or candles_15m.empty or len(candles_15m) < 3:
                return False
            recent = candles_15m.tail(4)  # look at last 4 for tolerance
            is_w = best_formation.type.upper() == "W"
            for _, row in recent.iterrows():
                o, h, low, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
                if is_w and _is_hammer(o, h, low, c):
                    return True
                if not is_w and _is_inverted_hammer(o, h, low, c):
                    return True
            return False
        except Exception:
            # If 15m is unavailable, fall back to allowing the trade (don't block on data issue)
            return True

    def _moon_phase_aligned(self, direction: str, now: datetime) -> bool:
        """Course lesson 37 moon signal alignment with trade direction.

        Returns True if within ±3 days of a primary phase (new/full) whose
        signal supports the requested direction. Feeds the `moon_cycle`
        confluence factor (2 pts LOW per `WEIGHTS`).
        """
        try:
            from src.strategy.mm_moon import compute_moon_phase, moon_signal_aligns_with_direction
            info = compute_moon_phase(now)
            return moon_signal_aligns_with_direction(info, direction)
        except Exception:
            return False

    def _compute_asia_range_pct(self, candles_1h: pd.DataFrame, now: datetime) -> float | None:
        """Compute today's Asia-session range as a % of the session open price.

        Course lesson 9/12: "Asia's job is just to create the spread and the
        initial HIGH and LOW of the Day. This typically is no more than a 2%
        range for Bitcoin in particular. If Asia runs a wider range, skip
        the whole day."

        Asia session: 8:30pm–3am NY. For this simple check we use the last
        ~7 closed 1H bars preceding 3am NY if we're past Asia close, otherwise
        the Asia bars seen so far today. Returns ``None`` if Asia hasn't
        started or we lack data.
        """
        if candles_1h is None or candles_1h.empty or len(candles_1h) < 8:
            return None
        try:
            # Take the last ~7 1H bars (Asia session is 6.5h). This is a
            # pragmatic approximation — a more accurate implementation would
            # align to exact NY-time Asia boundaries.
            asia = candles_1h.tail(7)
            high = float(asia["high"].max())
            low = float(asia["low"].min())
            if low <= 0:
                return None
            return (high - low) / low * 100.0
        except Exception:
            return None

    async def _take_partial(self, pos: MMPosition, level: int, current_price: float) -> None:
        """Take partial profit at level completion."""
        target_close_pct = PROFIT_SCHEDULE.get(level, 0)
        if target_close_pct <= pos.partial_closed_pct:
            return  # Already closed enough

        close_pct = target_close_pct - pos.partial_closed_pct
        close_qty = pos.quantity * close_pct

        if close_qty <= 0:
            return

        side = "sell" if pos.direction == "long" else "buy"
        try:
            result = await self.exchange.place_market_order(
                symbol=pos.symbol,
                side=side,
                quantity=close_qty,
            )
            if result and result.status == "closed":
                pos.partial_closed_pct = target_close_pct
                pos.quantity -= close_qty
                # Persist to DB so restarts don't re-close the same partials
                try:
                    await self.repo.update_trade(pos.trade_id, {
                        "remaining_quantity": pos.quantity,
                        "current_tier": level,
                    })
                except Exception as e:
                    logger.debug("mm_partial_db_update_failed", error=str(e))
                logger.info(
                    "mm_partial_profit",
                    symbol=pos.symbol,
                    level=level,
                    closed_pct=round(target_close_pct * 100),
                    qty=round(close_qty, 6),
                    price=current_price,
                )
        except Exception as e:
            logger.warning("mm_partial_failed", symbol=pos.symbol, error=str(e))

    async def _close_position(self, pos: MMPosition, price: float, reason: str) -> None:
        """Fully close an MM position."""
        if pos.quantity <= 0:
            self.positions.pop(pos.symbol, None)
            return

        side = "sell" if pos.direction == "long" else "buy"
        try:
            await self.exchange.place_market_order(
                symbol=pos.symbol,
                side=side,
                quantity=pos.quantity,
            )
        except Exception as e:
            logger.warning("mm_close_failed", symbol=pos.symbol, error=str(e))
            return

        # Calculate PnL
        if pos.direction == "long":
            pnl = (price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - price) * pos.quantity

        # Update database
        try:
            await self.repo.update_trade(pos.trade_id, {
                "status": "closed",
                "exit_price": price,
                "exit_quantity": pos.quantity,
                "exit_reason": reason,
                "pnl_usd": round(pnl, 4),
                "remaining_quantity": 0,
                "exit_time": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            logger.debug("mm_trade_db_update_failed", error=str(e))

        logger.info(
            "mm_trade_closed",
            trade_id=pos.trade_id,
            symbol=pos.symbol,
            direction=pos.direction,
            entry=pos.entry_price,
            exit=price,
            pnl=round(pnl, 4),
            reason=reason,
            level=pos.current_level,
        )

        self.positions.pop(pos.symbol, None)
        # Cooldown: don't re-enter this symbol for _cooldown_hours
        self._cooldowns[pos.symbol] = datetime.now(timezone.utc) + timedelta(hours=self._cooldown_hours)

    @staticmethod
    def _is_valid_target(price: float, direction: str, entry: float) -> bool:
        """Check if a target price is in the right direction."""
        if direction == "long":
            return price > entry
        else:
            return price < entry
