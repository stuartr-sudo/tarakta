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
from src.strategy.mm_brinks import BrinksDetector
from src.strategy.mm_confluence import MMConfluenceScorer, MMContext
from src.strategy.mm_ema_framework import EMAFramework
from src.strategy.mm_formations import (
    FormationDetector,
    classify_london_pattern,
    detect_half_batman,
    detect_nyc_reversal,
    detect_stophunt_entry,
)
from src.strategy.mm_levels import LevelTracker
from src.strategy.mm_risk import MMRiskCalculator
from src.strategy.mm_adr import ADRAnalyzer
from src.strategy.mm_rsi import RSIAnalyzer
from src.strategy.mm_sessions import MMSessionAnalyzer
from src.strategy.mm_targets import TargetAnalyzer
from src.strategy.mm_weekly_cycle import WeeklyCycleTracker
from src.strategy.mm_weekend_trap import WeekendTrapAnalyzer
from src.strategy.mm_scalp_vwap_rsi import VWAPRSIScalper, ScalpSignal
from src.strategy.mm_scalp_ribbon import RibbonAnalyzer, RibbonSignal
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Scan interval (minutes) — how often the MM engine runs
DEFAULT_SCAN_INTERVAL = 5

# Minimum confluence score (%) to consider an entry — course requires Grade C+ (40%)
MIN_CONFLUENCE_PCT = 35.0  # Lowered from 40: with 7 stubbed data feeds (18 pts unavailable), 40 was too restrictive

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

# Hard-ceiling sanity backstop for concurrent MM positions. The actual
# limit is the aggregate-risk budget (MAX_AGGREGATE_RISK_PCT below). 20
# is ~the majors universe size, so this never fires unless something
# else is broken. Raised 3→6→20 on 2026-04-20.
MAX_MM_POSITIONS = 20

# Aggregate open risk cap across ALL concurrent positions, as % of
# account balance. Course rule is 1% per trade; this expresses the same
# principle at portfolio level. If proposed trade would push aggregate
# risk over this cap, the trade is rejected (reason=aggregate_risk).
MAX_AGGREGATE_RISK_PCT = 5.0

# Max distance from entry to TP1, as % of entry price. Engineering cap
# — NOT an explicit course rule. Rejects setups whose computed TP1 is
# so far from entry that the formation-timeframe (1H) has little chance
# of reaching it in a sane hold period. 0 disables.
# Tuneable via config.mm_max_tp1_distance_pct or settings UI.
MAX_TP1_DISTANCE_PCT = 10.0

# Max slippage (%) between current market price and the 2nd-peak wick
# (the course's retest entry level per Lesson 20 / 47). Setups where
# price has bounced further than this from peak2 wick are rejected as
# "missed entry window". 0 disables the gate. Configurable via
# config.mm_max_entry_slippage_pct (settings UI).
MAX_ENTRY_SLIPPAGE_PCT = 1.0

# 2h scratch rule — Max Favorable Excursion threshold in R-multiples.
# Course Lesson 13 [47:00]: "If you're not in substantial profit within
# two hours you scratch the trade." "Within two hours" is a window, not
# an instant — we track the highest R-multiple reached during the trade
# and only scratch if peak R never cleared this threshold by the 2h
# mark. 0.3R reads "substantial" conservatively: enough to rule out
# noise, small enough that any trade starting to work will have crossed
# it. Tuneable via config.mm_scratch_mfe_threshold_r. 0 effectively
# disables the scratch rule (any trade with 0 favorable excursion is
# already 0R so would always scratch at 0; 0 means "scratch nothing").
SCRATCH_MFE_THRESHOLD_R = 0.3

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

# Valid phases for new entries (weekly cycle phase machine).
#
# All 11 phases defined in mm_weekly_cycle.py:
#   WEEKEND_TRAP      — Sat/Sun consolidation before week open. No trading
#                       window per lesson 2 ("MM are not around"). BLOCK.
#   FMWB              — False Move Weekend Bait near Sun 5pm NY. Explicit
#                       reject at a separate gate (fmwb_phase). BLOCK.
#   FORMATION_PENDING — Between phases, formations forming. ALLOW.
#   LEVEL_1/2/3       — Inside a level progression. ALLOW.
#   BOARD_MEETING_1/2 — Retracement between levels (fib entries). ALLOW.
#   MIDWEEK_REVERSAL  — Wed/Thu reversal window (lesson 9). The course is
#                       explicit: "A new M or W formation confirms the
#                       reversal." This IS an entry window. ALLOW.
#   REVERSAL_LEVELS   — Repeats 3-level progression post-reversal. Same
#                       entry rules as LEVEL_1/2/3. ALLOW.
#   FRIDAY_TRAP       — Post Friday-UK-close. Terminal state for the week,
#                       holding is the trap. Explicit reject at
#                       friday_trap gate. BLOCK.
#
# Bug fixed 2026-04-15: MIDWEEK_REVERSAL and REVERSAL_LEVELS were missing
# from this set. On midweek (Wed/Thu), 96.8% of all scanned pairs were
# rejected by the wrong_phase gate — so the engine scanned for hours and
# couldn't open a single trade. See dashboard funnel screenshot in
# session history.
VALID_ENTRY_PHASES = {
    "FORMATION_PENDING",
    "LEVEL_1", "LEVEL_2", "LEVEL_3",
    "BOARD_MEETING_1", "BOARD_MEETING_2",
    "MIDWEEK_REVERSAL",
    "REVERSAL_LEVELS",
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

    # D3: London pattern classification (Lesson 09)
    london_pattern_type: str | None = None   # "type_1" | "type_2" | "type_3" | None

    # Course-faithful lifecycle carriers (2026-04 audit)
    # Course lesson 49 Refund Zone — only 2nd-peak aggressive entries use it
    entry_type: str = "conservative"  # "aggressive" (2nd-peak) or "conservative" (post-L1)
    peak2_wick_price: float = 0.0     # For W: 2nd-peak low wick; for M: 2nd-peak high wick
    # Course lessons 20, 23 — SVC "Trapped Traders" zone for post-entry invalidation
    svc_high: float = 0.0
    svc_low: float = 0.0

    # HTF trend snapshot at entry (2026-04 audit — migration 018).
    # Recorded on every entry so post-mortems can distinguish trend-aligned
    # losses from counter-trend losses without having to reconstruct the
    # 4H EMA state from historical candle data.
    htf_trend_4h: str = "unknown"  # "bullish" | "bearish" | "sideways" | "unknown"
    htf_trend_1d: str = "unknown"  # "bullish" | "bearish" | "sideways" | "unknown"
    counter_trend: bool = False    # True when trade_direction opposes a non-sideways 4H trend

    # MM Sanity Agent verdict (migration 019). VETOs never reach here —
    # only APPROVEs get persisted. ``decision`` is None when the agent
    # was disabled / unavailable, which is indistinguishable from "not
    # yet reviewed" — either way treat as "not reviewed by agent".
    mm_agent_decision: str | None = None      # "APPROVE" | None
    mm_agent_reason: str = ""
    mm_agent_confidence: float = 0.0
    mm_agent_model: str = ""
    mm_agent_concerns: list[str] = field(default_factory=list)


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

    # Course lesson 20, 23 — Stopping Volume Candle "Trapped Traders zone".
    # "We always want to see price fail to get back to the wick of a stopping
    # volume candle." If a 1H CLOSE returns into [svc_low, svc_high], the
    # formation is invalidated and we cut the position.
    svc_high: float = 0.0   # Top of the SVC range (body+wick)
    svc_low: float = 0.0    # Bottom of the SVC range
    # Track whether we've already taken the 200-EMA hammer partial (C6)
    took_200ema_partial: bool = False

    # Max Favorable Excursion in R-multiples (P3 fix 2026-04-22).
    # R = distance from entry to original_stop_loss. MFE is the highest
    # R-multiple the trade has reached at any point in its lifetime.
    # Used by the 2h scratch rule (course Lesson 13 [47:00]):
    #   "If you're not in substantial profit within two hours you scratch."
    # The course says "within two hours" — meaning at ANY point during
    # that window the trade must have reached substantial profit, not
    # "at the 2h mark". The prior rule (gross > round_trip_fees at the
    # 2h instant) closed winners that had been +1R mid-flight but
    # pulled back to break-even by the check, defeating the intent.
    # MFE tracking + a threshold lets us honour the course quote
    # faithfully: a trade that ever reached +0.3R during its first 2h
    # is not scratched, even if it's currently at break-even.
    max_favorable_excursion_r: float = 0.0


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
        # Aggregate-risk budget (portfolio-level expression of the
        # course's 1%-per-trade rule). See MAX_AGGREGATE_RISK_PCT.
        self.max_aggregate_risk_pct = float(
            getattr(config, "mm_max_aggregate_risk_pct", MAX_AGGREGATE_RISK_PCT)
        )
        # Max TP1 distance cap. See MAX_TP1_DISTANCE_PCT.
        self.max_tp1_distance_pct = float(
            getattr(config, "mm_max_tp1_distance_pct", MAX_TP1_DISTANCE_PCT)
        )
        # Max entry slippage from retest wick. See MAX_ENTRY_SLIPPAGE_PCT.
        self.max_entry_slippage_pct = float(
            getattr(config, "mm_max_entry_slippage_pct", MAX_ENTRY_SLIPPAGE_PCT)
        )
        # 2h scratch MFE threshold in R-multiples. See SCRATCH_MFE_THRESHOLD_R.
        self.scratch_mfe_threshold_r = float(
            getattr(config, "mm_scratch_mfe_threshold_r", SCRATCH_MFE_THRESHOLD_R)
        )

        # Tunable parameters (overridable via settings page)
        self.risk_pct = RISK_PER_TRADE_PCT
        self.leverage = 10
        self.min_rr = MIN_RR_AGGRESSIVE
        self.min_confluence = MIN_CONFLUENCE_PCT
        self.min_formation_quality = MIN_FORMATION_QUALITY
        self.max_sl_pct = MAX_SL_DISTANCE_PCT
        # Gate threshold: 0 = disabled (default, backwards-compat).
        # 1-5 = require N of 5 course-cited gates (multi-session,
        # hammer-at-peak2, at LOD/LOW, course-variant, HTF). Recommended
        # production value: 3 — see docs/STATUS_2026-04-26 backtest results.
        self._gate_threshold = 0

        # MM Method modules
        self.session_analyzer = MMSessionAnalyzer()
        self.ema_framework = EMAFramework()
        self.rsi_analyzer = RSIAnalyzer()
        self.adr_analyzer = ADRAnalyzer()
        self.formation_detector = FormationDetector(session_analyzer=self.session_analyzer)
        self.level_tracker = LevelTracker(ema_framework=self.ema_framework)
        self.weekly_cycle_tracker = WeeklyCycleTracker()
        self.confluence_scorer = MMConfluenceScorer(min_rr=MIN_RR, min_score=MIN_CONFLUENCE_PCT)
        self.weekend_trap_analyzer = WeekendTrapAnalyzer()
        self.board_meeting_detector = BoardMeetingDetector()
        self.brinks_detector = BrinksDetector()
        self.target_analyzer = TargetAnalyzer()
        self.risk_calculator = MMRiskCalculator(risk_per_trade=RISK_PER_TRADE_PCT / 100)

        # C4: BBWP volatility timing indicator (course Trading Strategies lesson 04).
        # Timing-only — does not affect entry decisions. Used for logging/telemetry.
        from src.strategy.mm_bbwp import BBWPAnalyzer
        self.bbwp_analyzer = BBWPAnalyzer()

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

        # Course G1 (lesson 55): Linda Trade multi-TF level cascade tracker.
        # Tracks per-symbol 3-level cycle completion across 15m → 1h → 4h →
        # 1d → 1w. Fed from `_analyze_pair` / `_manage_position` when a
        # level is observed.
        from src.strategy.mm_linda import LindaTracker
        self.linda = LindaTracker()

        # A7: VWAP+RSI(2) Scalp Strategy — fallback entry path when no
        # MM formation is found. Fires on 15m chart with VWAP + RSI(2)
        # extreme + reversal candlestick at pullback.
        self.scalper = VWAPRSIScalper()

        # A8: Ribbon (Multi-EMA) Scalp Strategy — second fallback entry path
        # when neither MM formation nor VWAP+RSI scalp is found. Uses a
        # 9-EMA ribbon (periods 2-100) with trend-flip + yellow-EMA pullback.
        self.ribbon_analyzer = RibbonAnalyzer()

        # MM Sanity Agent (Agent 4) — Opus 4.7 LLM guardrail that reviews every
        # setup surviving the deterministic rules and vetoes judgement-call
        # failures rules can't catch (e.g. "three_hits_how exemption voided by
        # accelerating 4H trend"). See docs/MM_SANITY_AGENT_DESIGN.md.
        #
        # When the API key or SDK is missing, .review() returns None and the
        # engine fails open (approves). Same for timeouts / API errors.
        from src.strategy.mm_sanity_agent import MMSanityAgent
        self.sanity_agent = MMSanityAgent(config=config, repo=repo)

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

    def _try_board_meeting_formation(self, candles_1h):
        """Course lesson 22: board-meeting M/W as an entry trigger.

        "within a board meeting you should be looking for either stop hunt,
        for your indication that price is going to break to the next level,
        or an M or W."

        Uses the existing BoardMeetingDetector; when it yields a concrete
        `entry` suggestion we synthesize a Formation with variant='board_meeting'.
        """
        from src.strategy.mm_formations import Formation

        if candles_1h is None or candles_1h.empty or len(candles_1h) < 30:
            return None

        # Try both directions — the BM detector is direction-sensitive
        for direction in ("bullish", "bearish"):
            try:
                bm = self.board_meeting_detector.detect(candles_1h, level_direction=direction)
            except Exception:
                continue
            if not bm.detected or bm.entry is None:
                continue
            # Need at least 2 retracement candidates in the BM to call it "M/W-shaped"
            entry = bm.entry
            if entry.risk_reward <= 0 or entry.entry_price <= 0:
                continue
            # Map direction → W (bullish long) / M (bearish short)
            ftype = "W" if direction == "bullish" else "M"
            # peak1/peak2 prices — use SL reference as the invalidation extreme.
            # For W (long): peaks = lows (SL goes below them).
            # For M (short): peaks = highs (SL goes above them).
            # entry.stop_loss from BM detector is the correct invalidation level.
            p1 = float(entry.stop_loss) if entry.stop_loss else float(entry.entry_price)
            p2 = float(entry.entry_price)
            trough = float(entry.target) if entry.target else p2
            return Formation(
                type=ftype,
                variant="board_meeting",
                peak1_idx=max(0, bm.start_idx) if bm.start_idx >= 0 else 0,
                peak1_price=p1,
                peak2_idx=max(0, bm.end_idx) if bm.end_idx >= 0 else len(candles_1h) - 1,
                peak2_price=p2,
                trough_idx=max(0, bm.start_idx) if bm.start_idx >= 0 else 0,
                trough_price=trough,
                direction=direction,
                quality_score=float(entry.confidence),
                at_key_level=True,
                confirmed=True,  # BM detector validates the entry independently
            )
        return None

    def _try_brinks_formation(self, candles_15m, now, cycle_state):
        """Course lesson 06: Brinks Trade — highest R:R setup (6:1 to 18:1).

        The Brinks Trade fires ONLY at two 15-min candle close times in NY:
        3:30-3:45am (UK open) or 9:30-9:45am (US open). Requires a hammer/
        inverted hammer at HOD/LOD with a prior peak 30-90 minutes back.
        """
        from src.strategy.mm_formations import Formation
        from zoneinfo import ZoneInfo

        if candles_15m is None or candles_15m.empty or len(candles_15m) < 3:
            return None

        ny_tz = ZoneInfo("America/New_York")
        now_ny = now.astimezone(ny_tz) if now.tzinfo else now.replace(tzinfo=ny_tz)

        hod = getattr(cycle_state, "hod", 0) or 0
        lod = getattr(cycle_state, "lod", 0) or 0

        if hod <= 0 and lod <= 0:
            return None

        result = self.brinks_detector.detect(candles_15m, hod, lod, now_ny)
        if result is None:
            return None

        ftype = result.formation_type  # "W" or "M"
        direction = "bullish" if result.direction == "long" else "bearish"

        # Synthesize a Formation so the rest of the pipeline works uniformly.
        peak2_idx = len(candles_15m) - 1
        # Estimate peak1_idx from separation.
        if result.peak1_time and result.peak2_time:
            delta = result.peak2_time - result.peak1_time
            bar_sep = max(1, int(delta.total_seconds() / 900))
        else:
            bar_sep = 3  # fallback ~45 min

        peak1_idx = max(0, peak2_idx - bar_sep)

        return Formation(
            type=ftype,
            variant="brinks",
            peak1_idx=peak1_idx,
            peak1_price=result.stop_loss,   # first leg extreme
            peak2_idx=peak2_idx,
            peak2_price=result.entry_price,
            trough_idx=peak1_idx,
            trough_price=result.entry_price,
            direction=direction,
            quality_score=result.quality,
            at_key_level=True,
            confirmed=True,
        )

    def _try_nyc_reversal_formation(self, candles_1h, session, cycle_state, now):
        """Course lesson 10 (A2): NYC Reversal trade.

        Within the first 3 hours of the US session (9:30am-12:30pm NY),
        if price is at Level 3 with HOD/LOD formed and shows a reversal
        candlestick pattern (hammer, inverted hammer, railroad tracks),
        enter a reversal trade targeting the 50 EMA or range recovery.

        This is the L3 reversal — it NEEDS the L3 gate to be bypassed.
        """
        from src.strategy.mm_formations import Formation
        from zoneinfo import ZoneInfo

        if session is None or session.session_name != "us":
            return None

        # Compute level analysis inline (like three_hits does)
        # Check both directions and pick the one with higher level
        level_bull = self.level_tracker.analyze(candles_1h, direction="bullish")
        level_bear = self.level_tracker.analyze(candles_1h, direction="bearish")
        current_level = max(level_bull.current_level, level_bear.current_level)
        if current_level < 3:
            return None

        ny_tz = ZoneInfo("America/New_York")
        now_ny = now.astimezone(ny_tz) if now.tzinfo else now.replace(tzinfo=ny_tz)

        hod = getattr(cycle_state, "hod", 0) or 0
        lod = getattr(cycle_state, "lod", 0) or 0

        result = detect_nyc_reversal(
            candles_1h=candles_1h,
            session_name=session.session_name,
            current_level=current_level,
            hod=hod, lod=lod, now_ny=now_ny,
        )

        if result is None:
            return None

        ftype = "W" if result.direction == "bullish" else "M"
        slice_len = min(40, len(candles_1h))
        p_last_idx = max(0, slice_len - 1)
        p_prev_idx = max(0, slice_len - 2)

        logger.info(
            "mm_nyc_reversal_detected",
            direction=result.direction,
            pattern=result.pattern,
            entry_price=result.entry_price,
            level=result.level,
        )

        return Formation(
            type=ftype,
            variant="nyc_reversal",
            peak1_idx=p_prev_idx,
            peak1_price=hod if result.direction == "bearish" else lod,
            peak2_idx=p_last_idx,
            peak2_price=result.entry_price,
            trough_idx=p_last_idx,
            trough_price=result.entry_price,
            direction=result.direction,
            quality_score=0.6,
            at_key_level=True,
            confirmed=True,
        )

    def _try_stophunt_formation(self, candles_1h):
        """Course lesson 15 (A4): Stop hunt entry at Level 3.

        At Level 3 in a board meeting, a vector candle (stop hunt) fires
        with high volume and big wick. Entry is 1-2 candles AFTER the hunt,
        once we verify the wick is "left alone".
        """
        from src.strategy.mm_formations import Formation

        # Compute level analysis inline (like three_hits does)
        level_bull = self.level_tracker.analyze(candles_1h, direction="bullish")
        level_bear = self.level_tracker.analyze(candles_1h, direction="bearish")
        current_level = max(level_bull.current_level, level_bear.current_level)
        if current_level < 3:
            return None

        # Check if a board meeting is currently active
        board_meeting_active = False
        try:
            bm_detection = self.board_meeting_detector.detect(
                candles_1h, level_direction="bullish"
            )
            if bm_detection is not None and getattr(bm_detection, "detected", False):
                board_meeting_active = True
            if not board_meeting_active:
                bm_detection = self.board_meeting_detector.detect(
                    candles_1h, level_direction="bearish"
                )
                if bm_detection is not None and getattr(bm_detection, "detected", False):
                    board_meeting_active = True
        except Exception:
            pass

        result = detect_stophunt_entry(
            candles_1h=candles_1h,
            current_level=current_level,
            board_meeting_active=board_meeting_active,
        )

        if result is None:
            return None

        ftype = "W" if result.direction == "bullish" else "M"
        slice_len = min(40, len(candles_1h))
        p_last_idx = max(0, slice_len - 1)
        hunt_rel_idx = max(0, slice_len - (len(candles_1h) - result.hunt_candle_idx))

        logger.info(
            "mm_stophunt_entry_detected",
            direction=result.direction,
            entry_price=result.entry_price,
            stop_loss=result.stop_loss,
            hunt_candle_idx=result.hunt_candle_idx,
        )

        return Formation(
            type=ftype,
            variant="stophunt_l3",
            peak1_idx=hunt_rel_idx,
            peak1_price=result.stop_loss,
            peak2_idx=p_last_idx,
            peak2_price=result.entry_price,
            trough_idx=p_last_idx,
            trough_price=result.entry_price,
            direction=result.direction,
            quality_score=0.6,
            at_key_level=True,
            confirmed=True,
        )

    def _try_half_batman_formation(self, candles_1h):
        """Course lesson 15 (A3): Half Batman pattern.

        After a 3-level move, only ONE peak forms (no second peak for M/W).
        Very tight sideways consolidation follows — no stop hunts, equal
        highs/lows. Entry on break of the consolidation range.
        """
        from src.strategy.mm_formations import Formation

        # Compute level analysis inline
        level_bull = self.level_tracker.analyze(candles_1h, direction="bullish")
        level_bear = self.level_tracker.analyze(candles_1h, direction="bearish")
        current_level = max(level_bull.current_level, level_bear.current_level)

        result = detect_half_batman(
            candles_1h=candles_1h,
            current_level=current_level,
        )

        if result is None:
            return None

        ftype = "M" if result.direction == "bearish" else "W"
        slice_len = min(40, len(candles_1h))
        p_prev_idx = max(0, slice_len - 2)
        p_last_idx = max(0, slice_len - 1)

        logger.info(
            "mm_half_batman_detected",
            direction=result.direction,
            peak_price=result.peak_price,
            consol_high=result.consolidation_high,
            consol_low=result.consolidation_low,
            entry_price=result.entry_price,
        )

        return Formation(
            type=ftype,
            variant="half_batman",
            peak1_idx=p_prev_idx,
            peak1_price=result.peak_price,
            peak2_idx=p_last_idx,
            peak2_price=result.entry_price,
            trough_idx=p_last_idx,
            trough_price=result.entry_price,
            direction="bullish" if result.direction == "bullish" else "bearish",
            quality_score=0.55,
            at_key_level=True,
            confirmed=True,
        )

    def _try_33_trade_formation(self, candles_1h, cycle_state):
        """Course lesson 12 (A5): 33 Trade.

        Three rises over three days AND three hits to high on Day 3 AND
        EMAs fanning out (trend acceleration). Short off inverted hammer
        at Rise Level 3. Target: 50 EMA first, then 200 EMA.

        All three components already exist individually:
        1. level >= 3 (from level tracker)
        2. three hits at HOW (from formation detector)
        3. EMA fan-out (from _detect_ema_fan_out)
        """
        from src.strategy.mm_formations import Formation

        # Condition 1: level >= 3
        level_bull = self.level_tracker.analyze(candles_1h, direction="bullish")
        level_bear = self.level_tracker.analyze(candles_1h, direction="bearish")
        current_level = max(level_bull.current_level, level_bear.current_level)
        if current_level < 3:
            return None

        # Condition 2: three hits at HOW (bearish) or LOW (bullish)
        three_hits_how = None
        three_hits_low = None
        if cycle_state.how and cycle_state.how > 0:
            three_hits_how = self.formation_detector.detect_three_hits(
                candles_1h, cycle_state.how,
            )
        if cycle_state.low and cycle_state.low < float("inf") and cycle_state.low > 0:
            three_hits_low = self.formation_detector.detect_three_hits(
                candles_1h, cycle_state.low,
            )

        how_detected = three_hits_how is not None and three_hits_how.detected
        low_detected = three_hits_low is not None and three_hits_low.detected

        if not how_detected and not low_detected:
            return None

        # Condition 3: EMA fan-out
        if not self._detect_ema_fan_out(candles_1h):
            return None

        # All three conditions met — synthesize formation
        slice_len = min(40, len(candles_1h))
        p_prev_idx = max(0, slice_len - 2)
        p_last_idx = max(0, slice_len - 1)
        last_price = float(candles_1h["close"].iloc[-1])

        if how_detected:
            # Three hits at HOW → bearish reversal (short)
            return Formation(
                type="M",
                variant="33_trade",
                peak1_idx=p_prev_idx,
                peak1_price=cycle_state.how,
                peak2_idx=p_last_idx,
                peak2_price=last_price,
                trough_idx=p_last_idx,
                trough_price=last_price,
                direction="bearish",
                quality_score=0.6,
                at_key_level=True,
                confirmed=True,
            )
        else:
            # Three hits at LOW → bullish reversal (long)
            return Formation(
                type="W",
                variant="33_trade",
                peak1_idx=p_prev_idx,
                peak1_price=cycle_state.low,
                peak2_idx=p_last_idx,
                peak2_price=last_price,
                trough_idx=p_last_idx,
                trough_price=last_price,
                direction="bullish",
                quality_score=0.6,
                at_key_level=True,
                confirmed=True,
            )

    def _is_scalp_candidate(self, candles_1h) -> bool:
        """Pre-filter: only run scalp analysis on coins with RSI extremes on 1H.

        Course (Scalp Lesson 07): use screener to find oversold (<20) or
        overbought (>80) coins on 1H before analyzing.
        """
        if candles_1h is None or candles_1h.empty or len(candles_1h) < 15:
            return False
        rsi_state = self.rsi_analyzer.calculate(candles_1h)
        if rsi_state is None:
            return False
        return rsi_state.rsi_value < 20 or rsi_state.rsi_value > 80

    def _try_scalp_signal(self, candles_15m, candles_1h, cycle_state):
        """A7: VWAP+RSI Scalp — scan for a scalp setup on the 15m chart.

        Gets targets from the weekly cycle (HOW, LOW, HOD, LOD) and calls
        the VWAPRSIScalper. Returns a ScalpSignal or None.
        """
        targets = []
        for attr in ("how", "low", "hod", "lod"):
            val = getattr(cycle_state, attr, None)
            if val and val > 0 and val != float("inf"):
                targets.append(float(val))

        scalp = self.scalper.scan(candles_15m, candles_1h, targets or None)
        if scalp is not None and scalp.detected:
            logger.info(
                "mm_scalp_signal_detected",
                direction=scalp.direction,
                entry=scalp.entry_price,
                sl=scalp.stop_loss,
                tp=scalp.target,
                rr=scalp.risk_reward,
                rsi2=scalp.rsi_2_value,
                pattern=scalp.pattern,
                bias=scalp.rsi_14_bias,
            )
        return scalp

    def _formation_from_scalp(self, scalp: "ScalpSignal", candles_15m) -> "Formation":
        """Synthesize a Formation from a ScalpSignal for pipeline compatibility.

        Tags the formation with variant='scalp_vwap_rsi' so it's
        distinguishable in the trade log.
        """
        from src.strategy.mm_formations import Formation

        ftype = "W" if scalp.direction == "long" else "M"
        direction = "bullish" if scalp.direction == "long" else "bearish"

        n = len(candles_15m)
        peak2_idx = max(0, n - 1)
        peak1_idx = max(0, n - 3)  # ~30 min prior

        return Formation(
            type=ftype,
            variant="scalp_vwap_rsi",
            peak1_idx=peak1_idx,
            peak1_price=scalp.stop_loss,
            peak2_idx=peak2_idx,
            peak2_price=scalp.entry_price,
            trough_idx=peak2_idx,
            trough_price=scalp.entry_price,
            direction=direction,
            quality_score=0.5,
            at_key_level=False,
            confirmed=True,
        )

    def _try_ribbon_signal(self, candles_15m, cycle_state):
        """A8: Ribbon Scalp — scan for a ribbon setup on the 15m chart.

        Gets targets from the weekly cycle (HOW, LOW, HOD, LOD) and calls
        the RibbonAnalyzer. Returns a RibbonSignal or None.
        """
        targets = []
        for attr in ("how", "low", "hod", "lod"):
            val = getattr(cycle_state, attr, None)
            if val and val > 0 and val != float("inf"):
                targets.append(float(val))

        ribbon = self.ribbon_analyzer.scan(candles_15m, targets or None)
        if ribbon is not None and ribbon.detected:
            logger.info(
                "mm_ribbon_signal_detected",
                direction=ribbon.direction,
                entry=ribbon.entry_price,
                sl=ribbon.stop_loss,
                tp=ribbon.target,
                rr=ribbon.risk_reward,
                trend=ribbon.trend,
                squeezed=ribbon.squeezed,
            )
        return ribbon

    def _formation_from_ribbon(self, ribbon: "RibbonSignal", candles_15m) -> "Formation":
        """Synthesize a Formation from a RibbonSignal for pipeline compatibility.

        Tags the formation with variant='scalp_ribbon' so it's
        distinguishable in the trade log.
        """
        from src.strategy.mm_formations import Formation

        ftype = "W" if ribbon.direction == "long" else "M"
        direction = "bullish" if ribbon.direction == "long" else "bearish"

        n = len(candles_15m)
        peak2_idx = max(0, n - 1)
        peak1_idx = max(0, n - 3)  # ~30 min prior

        return Formation(
            type=ftype,
            variant="scalp_ribbon",
            peak1_idx=peak1_idx,
            peak1_price=ribbon.stop_loss,
            peak2_idx=peak2_idx,
            peak2_price=ribbon.entry_price,
            trough_idx=peak2_idx,
            trough_price=ribbon.entry_price,
            direction=direction,
            quality_score=0.5,
            at_key_level=False,
            confirmed=True,
        )

    def _try_200ema_rejection_formation(self, candles_1h, candles_4h, candles_15m):
        """Course lesson 18 alternative #2: 200 EMA rejection trade.

        "Besides these M or W replacement trigger, there is one more. And
        this is a 200 EMA rejection trade. The 200 EMA rejection trade is
        the 2nd setup in the weekly cycle."

        Triggers when:
          - Price is currently within 1% of the 4H 200 EMA
          - The last 15m bar is a hammer (long setup) or inverted hammer (short)
          - The hammer/inverted hammer is the recent reversal candle

        Returns a synthesized Formation object (treated as "200ema_rejection"
        variant) or ``None`` if criteria not met.
        """
        from src.strategy.mm_formations import Formation, _is_hammer, _is_inverted_hammer

        if candles_1h is None or candles_1h.empty or len(candles_1h) < 5:
            return None
        if candles_15m is None or candles_15m.empty or len(candles_15m) < 3:
            return None

        try:
            # Prefer 4H 200 EMA (macro-level rejection signal per lesson 24).
            if candles_4h is not None and not candles_4h.empty and len(candles_4h) >= 200:
                ema200 = candles_4h["close"].ewm(span=200, adjust=False).mean()
                ema200_now = float(ema200.iloc[-1])
            else:
                # Fall back to 1H 200 EMA
                if len(candles_1h) < 200:
                    return None
                ema200 = candles_1h["close"].ewm(span=200, adjust=False).mean()
                ema200_now = float(ema200.iloc[-1])

            last_price = float(candles_1h.iloc[-1]["close"])
            if ema200_now <= 0 or last_price <= 0:
                return None

            # Within 1% of 200 EMA?
            distance_pct = abs(last_price - ema200_now) / last_price
            if distance_pct > 0.01:
                return None

            # Index convention: downstream code computes
            # `peak_abs_idx = lookback_start + formation.peak_idx`
            # where lookback_start = max(0, len(candles_1h) - 40). We must
            # return peak indices RELATIVE to that slice, not absolute.
            # Previously we returned `len(candles_1h) - 2` which created an
            # out-of-bounds absolute index (peak_abs_idx = lookback_start +
            # len(candles_1h) - 2) and the SL calculation then grabbed the
            # WRONG candle — producing a nonsense SL on every
            # 200-EMA-rejection entry.
            slice_len = min(40, len(candles_1h))
            p_prev_idx = max(0, slice_len - 2)
            p_last_idx = max(0, slice_len - 1)

            # Check last 15m candle for hammer (price approached from above → bullish bounce)
            # or inverted hammer (price approached from below → bearish rejection)
            recent_15m = candles_15m.tail(4)
            for _, row in recent_15m.iloc[::-1].iterrows():
                o, h, low, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
                # Hammer = bullish reversal (W-style) — typically when price dropped to 200 EMA and bounced
                if _is_hammer(o, h, low, c) and last_price > ema200_now:
                    return Formation(
                        type="W", variant="200ema_rejection",
                        peak1_idx=p_prev_idx, peak1_price=ema200_now,
                        peak2_idx=p_last_idx, peak2_price=ema200_now,
                        trough_idx=p_last_idx, trough_price=low,
                        direction="bullish", quality_score=0.5, at_key_level=True,
                    )
                # Inverted hammer = bearish reversal (M-style) — price rallied to 200 EMA and got rejected
                if _is_inverted_hammer(o, h, low, c) and last_price < ema200_now:
                    return Formation(
                        type="M", variant="200ema_rejection",
                        peak1_idx=p_prev_idx, peak1_price=ema200_now,
                        peak2_idx=p_last_idx, peak2_price=ema200_now,
                        trough_idx=p_last_idx, trough_price=h,
                        direction="bearish", quality_score=0.5, at_key_level=True,
                    )
            return None
        except Exception:
            return None

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
                    if "mm_max_aggregate_risk_pct" in mm_settings:
                        self.max_aggregate_risk_pct = float(
                            mm_settings["mm_max_aggregate_risk_pct"]
                        )
                    if "mm_max_tp1_distance_pct" in mm_settings:
                        self.max_tp1_distance_pct = float(
                            mm_settings["mm_max_tp1_distance_pct"]
                        )
                    if "mm_max_entry_slippage_pct" in mm_settings:
                        self.max_entry_slippage_pct = float(
                            mm_settings["mm_max_entry_slippage_pct"]
                        )
                    if "mm_scratch_mfe_threshold_r" in mm_settings:
                        self.scratch_mfe_threshold_r = float(
                            mm_settings["mm_scratch_mfe_threshold_r"]
                        )
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
                    if "mm_gate_threshold" in mm_settings:
                        # 0..5; 0 = disabled. See gate framework below.
                        self._gate_threshold = max(0, min(5, int(mm_settings["mm_gate_threshold"])))
                    logger.info("mm_engine_restored_settings",
                                scanning_active=self._scanning_active,
                                max_positions=self.max_positions,
                                scan_interval_min=self.scan_interval / 60,
                                gate_threshold=self._gate_threshold)
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
                    # Per-trade MM lifecycle state — MUST restore or SL
                    # tightening / SVC invalidation / Refund Zone / 200 EMA
                    # partial deduplication are all silently disabled on
                    # restart. See migration 017.
                    original_stop_loss=float(t.get("original_stop_loss") or t.get("stop_loss") or 0),
                    entry_type=str(t.get("mm_entry_type") or "conservative"),
                    peak2_wick_price=float(t.get("mm_peak2_wick_price") or 0),
                    svc_high=float(t.get("mm_svc_high") or 0),
                    svc_low=float(t.get("mm_svc_low") or 0),
                    sl_moved_to_breakeven=bool(t.get("mm_sl_moved_to_breakeven") or False),
                    sl_moved_under_50ema=bool(t.get("mm_sl_moved_under_50ema") or False),
                    took_200ema_partial=bool(t.get("mm_took_200ema_partial") or False),
                    # Max Favorable Excursion in R (migration 020 / P3 fix).
                    # Without this, a mid-trade restart loses the fact
                    # that the trade already cleared the scratch bar,
                    # leading to a false-positive scratch at 2h.
                    max_favorable_excursion_r=float(
                        t.get("mm_max_favorable_excursion_r") or 0
                    ),
                )
                restored += 1
            if mm_trades:
                logger.info("mm_positions_restored", restored=restored, orphaned=orphaned,
                            symbols=list(self.positions.keys()))
        except Exception as e:
            logger.warning("mm_position_restore_failed", error=str(e))

        logger.info("mm_engine_started", scanning_active=self._scanning_active)

        # Start the dedicated price-refresh task in THIS event loop. The
        # dashboard reads self._last_prices but must not call self.exchange
        # directly because it runs in a different (uvicorn) thread/loop and
        # the exchange's semaphore is bound to THIS loop.
        price_task = asyncio.create_task(self._price_refresh_loop())

        try:
            while self._running:
                try:
                    await self._cycle()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("mm_engine_cycle_error", error=str(e), exc_info=True)

                await asyncio.sleep(self.scan_interval)
        finally:
            price_task.cancel()
            try:
                await price_task
            except (asyncio.CancelledError, Exception):
                pass

        logger.info("mm_engine_stopped")

    async def _price_refresh_loop(self) -> None:
        """Refresh _last_prices for all open positions every few seconds.

        Runs in the engine's main event loop so the exchange semaphore is
        in the right place. The dashboard's /api/mm/status reads the
        cached values — no cross-loop exchange calls.
        """
        interval_s = 5.0
        while self._running:
            try:
                symbols = list(self.positions.keys())
                if symbols:
                    # Parallel ticker fetch, swallow individual failures.
                    async def _one(sym: str) -> tuple[str, float | None]:
                        try:
                            t = await self.exchange.fetch_ticker(sym)
                            return sym, float(t.get("last") or 0) or None
                        except Exception:
                            return sym, None
                    results = await asyncio.gather(
                        *[_one(s) for s in symbols], return_exceptions=True
                    )
                    for r in results:
                        if isinstance(r, tuple) and r[1]:
                            self._last_prices[r[0]] = r[1]
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("mm_price_refresh_error", error=str(e))
            await asyncio.sleep(interval_s)

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False

    async def get_status(self) -> dict:
        """Return a status snapshot for the dashboard API.

        IMPORTANT: This runs in the dashboard's uvicorn thread event loop.
        We CANNOT call self.exchange.* here — the exchange client's
        asyncio.Semaphore was created in the MAIN event loop (where the
        engine cycle runs), so any call from this thread hangs forever
        with "bound to a different event loop". Symptom: the semaphore
        fills with waiters, eventually starving the engine's own scans
        ("mm_engine_pairs_error" with pairs_scanned:0).

        Instead: read from self._last_prices only. The engine loop's
        dedicated _price_refresh_loop keeps those fresh every few seconds.
        """
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
                "candles_ok", "formation_found", "htf_aligned", "level_ok", "phase_valid",
                "direction_ok", "target_acquired", "rr_passed", "scored",
                "confluence_passed", "retest_passed", "sanity_agent_passed", "signal_built",
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
        """Get the list of pairs to scan.

        Course lesson 1-3 and 53: MM Method works best on high-liquidity
        majors. Low-cap alts are manipulated, produce false signals, and
        rarely respect EMAs / market-maker cycles. We enforce:
          - Base volume floor much higher than generic (was 5M, now 50M).
          - Optional majors-only whitelist (mm_majors_only=True) that
            restricts to BTC/ETH/SOL/etc. — the safest MM Method setups.
        Both knobs are config-driven so the user can relax them if desired.
        """
        min_vol = float(getattr(self.config, "mm_min_volume_usd",
                                 getattr(self.config, "min_volume_usd", 50_000_000)))
        pairs = await self.exchange.get_tradeable_pairs(
            quote_currencies=["USDT"],
            min_volume_usd=min_vol,
        )
        # Optional majors-only gate (default False so existing behaviour
        # is preserved; flip to True in config for safest operation).
        majors_only = bool(getattr(self.config, "mm_majors_only", False))
        if majors_only:
            majors = {
                "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX",
                "DOT", "LINK", "MATIC", "LTC", "BCH", "TRX", "ATOM", "NEAR",
                "UNI", "APT", "ARB", "OP",
            }
            pairs = [p for p in pairs if p.split("/")[0].upper() in majors]
            logger.info("mm_majors_only_filter", total=len(pairs))
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
        # Fetch candles (1H primary, 4H and 1D for HTF trend, 15m for entry refinement)
        # Course C1 (lesson 13): "drop down to the 15 minute time frame for
        # the actual Entry... then switch back to the one hour" — we fetch
        # 15m so final-damage hammer/inverted-hammer checks (lesson 21) and
        # other 15m-sensitive validations have data available.
        # Course lessons 03 / 10 / 12: "shift to higher timeframes" — the 4H
        # and 1D EMA stacks are the HTF trend-alignment gate (see below).
        try:
            candles_1h = await self.candle_manager.get_candles(symbol, "1h", limit=500)
            # Need >=800 bars for EMA-800 in mm_ema_framework (largest period in
            # DEFAULT_EMA_PERIODS). Below that, get_trend_state returns sideways
            # and the HTF veto silently disables.
            candles_4h = await self.candle_manager.get_candles(symbol, "4h", limit=1000)
            try:
                candles_15m = await self.candle_manager.get_candles(symbol, "15m", limit=200)
            except Exception:
                candles_15m = None  # best-effort — some markets won't have 15m
            try:
                # 1D: 250 bars = ~8 months, plenty for EMA 50/200 daily trend.
                # Same EMA-800 requirement as 4H; insufficient history makes
                # 1D trend always read "sideways".
                candles_1d = await self.candle_manager.get_candles(symbol, "1d", limit=1000)
            except Exception:
                candles_1d = None  # best-effort — not all markets expose 1D
        except Exception as e:
            return self._reject("candle_fetch", symbol, error=str(e))

        if candles_1h is None or candles_1h.empty or len(candles_1h) < 50:
            return self._reject("insufficient_candles", symbol, count=0 if candles_1h is None or (hasattr(candles_1h, 'empty') and candles_1h.empty) else len(candles_1h))

        self._advance("candles_ok")

        # D4 course-faithful: only analyze closed candles (Lesson 13)
        # Course A8 (lesson 20 "wait for candle CLOSE before entering"):
        # Exchange candle endpoints include the CURRENT in-progress bar.
        # The course rule is "evaluate on the most recent CLOSED bar" — so if
        # the last row of the dataframe is still forming, drop it and use the
        # prior bar as the closed bar. Previously this gate rejected the pair
        # outright, which meant the engine could only ever pass this check
        # during the last 5 minutes of each hour (before the forming bar
        # ticks over) — i.e. effectively never.
        try:
            last_bar_ts = candles_1h.index[-1]
            if hasattr(last_bar_ts, "to_pydatetime"):
                last_bar_dt = last_bar_ts.to_pydatetime()
            else:
                last_bar_dt = last_bar_ts
            if last_bar_dt.tzinfo is None:
                last_bar_dt = last_bar_dt.replace(tzinfo=timezone.utc)
            # Bar close time = bar_start + 1h
            bar_close_dt = last_bar_dt + timedelta(hours=1)
            if now < bar_close_dt - timedelta(seconds=30):
                # Current bar still forming — drop it and work off closed bars.
                candles_1h = candles_1h.iloc[:-1]
                if len(candles_1h) < 50:
                    return self._reject(
                        "insufficient_candles_after_trim",
                        symbol,
                        count=len(candles_1h),
                    )
        except Exception:
            pass  # Best-effort only — don't block on timestamp parsing issues

        # Also trim the in-progress bar from 4H and 15m so downstream analysis
        # (formations, EMAs, inside-hits) operates on fully closed candles.
        for _tf_name, _df in (("4h", candles_4h), ("15m", candles_15m)):
            if _df is None or _df.empty:
                continue
            try:
                _last_ts = _df.index[-1]
                if hasattr(_last_ts, "to_pydatetime"):
                    _last_dt = _last_ts.to_pydatetime()
                else:
                    _last_dt = _last_ts
                if _last_dt.tzinfo is None:
                    _last_dt = _last_dt.replace(tzinfo=timezone.utc)
                _tf_secs = {"4h": 4 * 3600, "15m": 15 * 60}[_tf_name]
                _close_dt = _last_dt + timedelta(seconds=_tf_secs)
                if now < _close_dt - timedelta(seconds=30):
                    if _tf_name == "4h":
                        candles_4h = _df.iloc[:-1]
                    else:
                        candles_15m = _df.iloc[:-1]
            except Exception:
                pass

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

        # --- EMA timeframe hierarchy (course-faithful as of 2026-04-20) ----
        #
        # Course Lesson 12 / 16: the 50 EMA / 200 EMA / 800 EMA targets are
        # defined "on the chart you're trading". Formations are detected on
        # 1H here, so 1H is the trade-timeframe and supplies:
        #   - ema_state       → confluence scorer (ema_alignment, ema50_break)
        #   - ema_values      → target analyzer for L1 (200 EMA) and L2 (800 EMA)
        #
        # 4H is the HTF-alignment / trend-veto context (Lessons 03, 10, 12).
        # Gives us `trend_state_4h` → hard-veto counter-trend setups.
        # NOT used for TP targets anymore — previously was, which caused
        # 1H retest entries to inherit 4H-EMA targets that were often
        # beyond reach in intraday windows (BTC 2026-04-20 TP1 at +22%).
        #
        # 1D is the true higher-timeframe per Lesson 12 "a 200 or 800 EMA
        # on a higher time frame as a Target". Gives us:
        #   - trend_state_1d   → post-mortem / directional context
        #   - htf_ema_values   → target analyzer for L3 (1D 200 / 1D 800)
        #
        # When 1D history is insufficient (< 200 bars), 4H EMAs fall back
        # in to fill htf_ema_values so the L3 target slot isn't empty.
        ema_state = None
        ema_values: dict[int, float] = {}
        ema_break = None
        if candles_1h is not None and not candles_1h.empty and len(candles_1h) > 200:
            ema_state = self.ema_framework.calculate(candles_1h)
            ema_values = ema_state.values
            ema_break = self.ema_framework.detect_ema_break(candles_1h, ema_period=50)

        trend_state_4h = None
        if candles_4h is not None and not candles_4h.empty and len(candles_4h) > 200:
            trend_state_4h = self.ema_framework.get_trend_state(candles_4h)

        trend_state_1d = None
        if (
            candles_1d is not None
            and not candles_1d.empty
            and len(candles_1d) >= 50
        ):
            try:
                trend_state_1d = self.ema_framework.get_trend_state(candles_1d)
            except Exception:
                trend_state_1d = None

        # C2: RSI state (1H candles) — best-effort; None if insufficient data.
        rsi_state = None
        try:
            if candles_1h is not None and not candles_1h.empty:
                rsi_state = self.rsi_analyzer.calculate(candles_1h)
        except Exception:
            pass  # RSI not critical; confluence factor defaults to 0

        # C3: ADR state (1H candles resampled to daily) — best-effort; None if insufficient data.
        adr_state = None
        try:
            if candles_1h is not None and not candles_1h.empty:
                current_close = float(candles_1h["close"].iloc[-1])
                adr_state = self.adr_analyzer.calculate(candles_1h, current_close)
        except Exception:
            pass  # ADR not critical; confluence factor defaults to 0

        # C4: BBWP volatility timing (course Trading Strategies lesson 04).
        # Timing-only: "breakout_imminent" (BBWP ≤5) means consolidation maturing,
        # big move coming. "extreme_reached" (BBWP ≥95) means be careful.
        # Does NOT affect entry decisions — logged for telemetry only.
        try:
            bbwp_state = self.bbwp_analyzer.calculate(candles_1h)
            if bbwp_state is not None:
                logger.info(
                    "mm_bbwp",
                    symbol=symbol,
                    bbwp=round(bbwp_state.bbwp_value, 2),
                    ma=round(bbwp_state.ma_value, 2),
                    signal=bbwp_state.signal,
                )
        except Exception:
            pass  # BBWP not critical — never block on this

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
            # Course lesson 18 alternative #1: 3 hits at HOW/LOW at Level 3
            # "replace the M or W". Synthesize a Formation-shaped object so
            # the rest of the pipeline works uniformly.
            best_formation = self._try_three_hits_formation(candles_1h, cycle_state)
            if best_formation is None:
                # Course lesson 18 alternative #2: 200 EMA rejection trade.
                best_formation = self._try_200ema_rejection_formation(
                    candles_1h=candles_1h,
                    candles_4h=candles_4h,
                    candles_15m=candles_15m,
                )
                if best_formation is not None:
                    logger.info("mm_200ema_rejection_synthesized",
                                symbol=symbol, type=best_formation.type,
                                variant=best_formation.variant)
            else:
                logger.info("mm_three_hits_formation_synthesized",
                            symbol=symbol, type=best_formation.type,
                            variant=best_formation.variant,
                            level=cycle_state.how if best_formation.type == "M" else cycle_state.low)

            # Course lesson 22 alternative #3: board-meeting M/W.
            # "Within a board meeting you should be looking for either stop
            # hunt... or an M or W." Detector already runs — if it has a
            # concrete entry suggestion, use it.
            if best_formation is None:
                best_formation = self._try_board_meeting_formation(candles_1h)
                if best_formation is not None:
                    logger.info("mm_board_meeting_formation_synthesized",
                                symbol=symbol, type=best_formation.type,
                                variant=best_formation.variant)

            # Course lesson 06 alternative #5: Brinks Trade.
            # Highest R:R (6:1 to 18:1) — fires only at 3:30-3:45am or
            # 9:30-9:45am NY with hammer/inverted hammer at HOD/LOD.
            if best_formation is None and candles_15m is not None:
                best_formation = self._try_brinks_formation(candles_15m, now, cycle_state)
                if best_formation is not None:
                    logger.info("mm_brinks_formation_synthesized",
                                symbol=symbol, type=best_formation.type,
                                variant=best_formation.variant)

            # Course lesson 10 alternative #6: NYC Reversal (A2).
            # US open first 3 hours, Level 3+, reversal pattern at HOD/LOD.
            if best_formation is None:
                best_formation = self._try_nyc_reversal_formation(
                    candles_1h, session, cycle_state, now,
                )
                if best_formation is not None:
                    logger.info("mm_nyc_reversal_formation_synthesized",
                                symbol=symbol, type=best_formation.type,
                                variant=best_formation.variant)

            # Course lesson 15 alternative #7: Stop Hunt at Level 3 (A4).
            # Vector candle (stop hunt) in a board meeting at Level 3.
            if best_formation is None:
                best_formation = self._try_stophunt_formation(
                    candles_1h,
                )
                if best_formation is not None:
                    logger.info("mm_stophunt_formation_synthesized",
                                symbol=symbol, type=best_formation.type,
                                variant=best_formation.variant)

            # Course lesson 15 alternative #8: Half Batman (A3).
            # Single peak + tight consolidation after 3-level move.
            if best_formation is None:
                best_formation = self._try_half_batman_formation(candles_1h)
                if best_formation is not None:
                    logger.info("mm_half_batman_formation_synthesized",
                                symbol=symbol, type=best_formation.type,
                                variant=best_formation.variant)

            # Course lesson 12 alternative #9: 33 Trade (A5).
            # Three rises over three days AND three hits to high on Day 3
            # AND EMAs fanning out (trend acceleration). All three conditions
            # already exist individually — this combines them.
            if best_formation is None:
                best_formation = self._try_33_trade_formation(
                    candles_1h, cycle_state,
                )
                if best_formation is not None:
                    logger.info("mm_33_trade_formation_synthesized",
                                symbol=symbol, type=best_formation.type,
                                variant=best_formation.variant)

            # A7: VWAP+RSI Scalp fallback when no MM formation found.
            # Scans 15m chart for a pullback-to-VWAP/255-EMA setup with
            # extreme RSI(2) and a reversal candlestick pattern.
            # Gate: only run scalp on coins with RSI extremes on 1H (Scalp Lesson 07).
            if best_formation is None and candles_15m is not None and len(candles_15m) >= 30 and self._is_scalp_candidate(candles_1h):
                # Scalp Lesson 03: warn if near 1H candle close (last 10 min)
                if self.session_analyzer.is_near_1h_candle_close():
                    logger.info("mm_scalp_near_1h_close_warning", symbol=symbol)
                scalp = self._try_scalp_signal(candles_15m, candles_1h, cycle_state)
                if scalp is not None and scalp.detected:
                    best_formation = self._formation_from_scalp(scalp, candles_15m)

            # A8: Ribbon Scalp — second fallback after VWAP+RSI scalp.
            # Requires 150+ candles (slowest EMA period 100 * 1.5).
            if best_formation is None and candles_15m is not None and len(candles_15m) >= 150:
                ribbon = self._try_ribbon_signal(candles_15m, cycle_state)
                if ribbon is not None and ribbon.detected:
                    best_formation = self._formation_from_ribbon(ribbon, candles_15m)

            if best_formation is None:
                return self._reject("no_formation", symbol)

        self._advance("formation_found")

        # ---------------------------------------------------------------
        # at_key_level enrichment for standard W/M formations.
        # Synthesized variants (board_meeting, brinks, three_hits, etc.)
        # already set this flag in their constructors. The plain W/M
        # detector returns Formation(at_key_level=False) by default; here
        # we check whether peak2 sits at LOD/LOW (W) or HOD/HOW (M).
        # Lesson 7 [09:30]: "As long as it's at the Low of the day, or
        # Low of the week. So remember it has to be in the right place."
        # ---------------------------------------------------------------
        if best_formation is not None and not best_formation.at_key_level:
            p2 = float(best_formation.peak2_price)
            if p2 > 0 and len(candles_1h) >= 24:
                last_24h = candles_1h.tail(24)
                lod = float(last_24h["low"].min())
                hod = float(last_24h["high"].max())
                low_w = (
                    float(cycle_state.low)
                    if cycle_state.low and cycle_state.low > 0 and cycle_state.low < float("inf")
                    else 0.0
                )
                how_w = (
                    float(cycle_state.how)
                    if cycle_state.how and cycle_state.how > 0
                    else 0.0
                )
                AT_KEY_TOL_PCT = 0.005  # 0.5% tolerance
                is_at_key = False
                if best_formation.type.upper() == "W":
                    if lod > 0 and abs(p2 - lod) / lod < AT_KEY_TOL_PCT:
                        is_at_key = True
                    if low_w > 0 and abs(p2 - low_w) / low_w < AT_KEY_TOL_PCT:
                        is_at_key = True
                elif best_formation.type.upper() == "M":
                    if hod > 0 and abs(p2 - hod) / hod < AT_KEY_TOL_PCT:
                        is_at_key = True
                    if how_w > 0 and abs(p2 - how_w) / how_w < AT_KEY_TOL_PCT:
                        is_at_key = True
                if is_at_key:
                    best_formation.at_key_level = True

        # ---------------------------------------------------------------
        # EXPERIMENTAL: Course-faithful gate framework
        # ---------------------------------------------------------------
        # Replaces additive confluence scoring with hard gates derived
        # directly from Lesson 7. Configurable threshold lets us A/B test
        # 3/5, 4/5, 5/5 of the gates required to pass.
        #
        # Gates (each binary):
        #   1. Valid M/W formation         — passed by reaching here
        #   2. HTF trend aligned           — checked downstream by HTF veto
        #   3. Course-specific variant     — not bare "standard"
        #   4. Hammer/engulfing at peak2   — formation.confirmed
        #   5. At LOD/LOW or HOD/HOW       — formation.at_key_level
        #
        # Set `_gate_threshold` to N (1..5) to require N of 5. 0 disables.
        # Backwards compat: `_quality_gates_enabled` still honoured.
        gate_threshold = int(getattr(self, "_gate_threshold", 0) or 0)
        if gate_threshold > 0:
            gates_total = 5
            passed = 1  # gate 1: formation valid (already here)
            failed: list[str] = []
            # Gate 2: HTF aligned — accept if cycle_state.phase isn't a
            # hard-veto phase. Real HTF veto fires later; here we just
            # don't re-check (count as passed for simplicity).
            passed += 1
            # Gate 3: course-specific variant
            if best_formation.variant != "standard":
                passed += 1
            else:
                failed.append("standard_variant")
            # Gate 4: hammer/engulfing at peak2
            if best_formation.confirmed:
                passed += 1
            else:
                failed.append("no_peak2_confirmation")
            # Gate 5: at LOD/LOW or HOD/HOW
            if best_formation.at_key_level:
                passed += 1
            else:
                failed.append("not_at_key_level")
            if passed < gate_threshold:
                return self._reject(
                    "gate_threshold",
                    symbol,
                    passed=passed,
                    of=gates_total,
                    required=gate_threshold,
                    failed=",".join(failed) or "none",
                    variant=best_formation.variant,
                )
        elif getattr(self, "_quality_gates_enabled", False):
            # Legacy two-gate experiment (kept for the older wrapper).
            if best_formation.variant == "standard":
                return self._reject(
                    "single_session_standard",
                    symbol,
                    variant=best_formation.variant,
                )
            if best_formation.variant in ("standard", "multi_session") and not best_formation.confirmed:
                return self._reject(
                    "no_hammer_at_peak2",
                    symbol,
                    variant=best_formation.variant,
                    type=best_formation.type,
                )

        # D2: Asia closing spike directional hint (Lesson 09).
        # The spike that Asia makes in its final 30 minutes (2:00-2:30am NY)
        # predicts the London opening structure. We apply this as a soft
        # filter during the early UK session only — warn when the formation
        # direction conflicts with the Asia spike bias, but never hard-reject.
        try:
            asia_spike = self.session_analyzer.detect_asia_closing_spike(candles_1h, now)
            if (
                asia_spike.detected
                and session.session_name == "uk"
                and session.minutes_remaining > 270  # early UK: first 60 min (90-min gap subtr.)
            ):
                formation_direction = best_formation.direction  # "bullish" | "bearish"
                if asia_spike.bias and asia_spike.bias != "none" and asia_spike.bias != formation_direction:
                    logger.info(
                        "mm_asia_spike_bias_conflict",
                        symbol=symbol,
                        spike_direction=asia_spike.direction,
                        spike_bias=asia_spike.bias,
                        formation_direction=formation_direction,
                        magnitude_pct=asia_spike.magnitude_pct,
                        note="soft_filter_only",
                    )
                elif asia_spike.bias == formation_direction:
                    logger.info(
                        "mm_asia_spike_bias_aligned",
                        symbol=symbol,
                        spike_bias=asia_spike.bias,
                        formation_direction=formation_direction,
                        magnitude_pct=asia_spike.magnitude_pct,
                    )
        except Exception:
            pass  # Best-effort telemetry — never block a trade on this

        # D3: London pattern classification (Lesson 09).
        # Classify the formation as Type 1/2/3 for telemetry and signal metadata.
        # Type 1 (multi-session) is the highest-probability setup.
        london_pattern_type: str | None = None
        try:
            hod_val = float(cycle_state.how) if cycle_state.how and cycle_state.how > 0 else 0.0
            lod_val = float(cycle_state.low) if cycle_state.low and cycle_state.low < float("inf") else 0.0
            london_pattern_type = classify_london_pattern(
                formation=best_formation,
                session_info=session,
                hod=hod_val,
                lod=lod_val,
            )
            logger.info(
                "mm_london_pattern_classified",
                symbol=symbol,
                pattern_type=london_pattern_type,
                formation_variant=best_formation.variant,
                session=session.session_name,
            )
        except Exception:
            pass  # Telemetry only — classification failure never blocks a trade

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

        # Course B1 (lesson 20): "3 inside right-side hits" requirement.
        # After the Stopping Volume Candle (1st MM appearance) the market maker
        # must show up THREE times on the inside right side of the formation
        # — visible only by dropping to a lower timeframe. We enforce this on
        # standard formations; relax for synthetic variants where the concept
        # doesn't cleanly apply (three_hits_*, board_meeting — which the course
        # itself says "don't follow the same criteria").
        if best_formation.variant in ("standard", "multi_session", "final_damage"):
            if candles_1h is not None and candles_15m is not None:
                _lookback_start = max(0, len(candles_1h) - 40)
                if not self._check_inside_hits_15m(
                    best_formation=best_formation,
                    candles_1h=candles_1h,
                    candles_15m=candles_15m,
                    lookback_start=_lookback_start,
                ):
                    return self._reject("no_inside_hits_15m", symbol,
                                        formation_type=best_formation.type,
                                        variant=best_formation.variant)

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

        # ---------------------------------------------------------------
        # HTF trend-alignment hard-veto (course lessons 03 / 10 / 12).
        # ---------------------------------------------------------------
        # Prior to 2026-04 the 4H trend state was computed and thrown away.
        # That's how the BNB short on 2026-04-17 slipped through: the 4H
        # stack was cleanly bullish and the bot still entered a short on a
        # 1H "three hits at HOW" reversal. Price never came back.
        #
        # Rule: if the 4H trend direction is bullish/bearish (strength is
        # non-trivial) AND it opposes the trade, reject — unless the
        # formation is a recognised reversal variant AND the trend is
        # NOT accelerating (i.e. trend may be exhausting, so a reversal is
        # at least plausible). The course explicitly allows counter-trend
        # entries at Level-3 exhaustion (lessons 10, 14) but not earlier.
        htf_4h_dir = trend_state_4h.direction if trend_state_4h is not None else "unknown"
        htf_4h_strength = trend_state_4h.strength if trend_state_4h is not None else 0.0
        htf_4h_accel = trend_state_4h.is_accelerating if trend_state_4h is not None else False
        htf_1d_dir = trend_state_1d.direction if trend_state_1d is not None else "unknown"

        is_counter_trend_4h = (
            htf_4h_dir == "bullish" and trade_direction == "short"
        ) or (
            htf_4h_dir == "bearish" and trade_direction == "long"
        )

        # Reversal variants that the course allows against HTF trend at
        # exhaustion. "three_hits_*" and "final_damage" are named reversal
        # setups; "half_batman", "nyc_reversal", and "200ema_rejection" are
        # the other explicit counter-trend setups the course teaches.
        HTF_REVERSAL_EXEMPT_VARIANTS = {
            "three_hits_how", "three_hits_low",
            "final_damage",
            "half_batman",
            "nyc_reversal",
            "200ema_rejection",
            "stophunt",
        }

        # Only veto on non-trivial trend strength. A weak/noisy 4H trend
        # (strength < 0.5) isn't a reliable directional filter.
        HTF_VETO_STRENGTH_THRESHOLD = 0.5

        if (
            is_counter_trend_4h
            and htf_4h_strength >= HTF_VETO_STRENGTH_THRESHOLD
        ):
            exempt = (
                best_formation.variant in HTF_REVERSAL_EXEMPT_VARIANTS
                and not htf_4h_accel  # accelerating trend = no exemption
            )
            if not exempt:
                return self._reject(
                    "htf_4h_counter_trend",
                    symbol,
                    htf_4h=htf_4h_dir,
                    htf_4h_strength=round(htf_4h_strength, 3),
                    htf_4h_accelerating=htf_4h_accel,
                    htf_1d=htf_1d_dir,
                    trade_dir=trade_direction,
                    formation_variant=best_formation.variant,
                )
            # Exempt reversal setup against non-accelerating HTF trend.
            # Log it loudly — these trades deserve scrutiny and want to be
            # easy to find in logs when they lose.
            logger.info(
                "mm_htf_counter_trend_exempt",
                symbol=symbol,
                htf_4h=htf_4h_dir,
                htf_4h_strength=round(htf_4h_strength, 3),
                trade_dir=trade_direction,
                formation_variant=best_formation.variant,
            )

        self._advance("htf_aligned")
        # ---------------------------------------------------------------

        lookback_start = max(0, len(candles_1h) - 40)  # same window formation detector used
        formation_abs_idx = lookback_start + best_formation.peak2_idx
        candles_post_formation = candles_1h.iloc[formation_abs_idx:]
        level_analysis = self.level_tracker.analyze(candles_post_formation, direction=direction)

        # Feed the Linda multi-TF level tracker on EVERY available TF
        # (course G1 / lesson 55). Cascades start at the lowest available
        # TF and tick upward as 3-level cycles complete. Previously we only
        # recorded 1H — which meant cascades into 4H/Daily required 9
        # full 1H cycles (~9 hours) before the first cascade fired. By
        # also recording 15m, a 15m 3-level cycle (45 min) cascades to
        # 1H L1 immediately. By recording 4H, completed 4H cycles cascade
        # to Daily etc.
        linda_cascade_15m_to_1h = False
        linda_cascade_1h_to_4h = False
        linda_cascade_4h_to_1d = False
        try:
            # 15m level from the 15m candles we already have (B1 data)
            if candles_15m is not None and not candles_15m.empty and len(candles_15m) >= 10:
                try:
                    lvl_15m = self.level_tracker.analyze(
                        candles_15m.tail(60), direction=direction,
                    )
                    self.linda.record(
                        symbol=symbol, timeframe="15m",
                        level=int(lvl_15m.current_level),
                        direction=direction, now=now,
                    )
                except Exception:
                    pass

            # 1H — same as before
            self.linda.record(
                symbol=symbol, timeframe="1h",
                level=int(level_analysis.current_level),
                direction=direction, now=now,
            )

            # 4H level from the 4H candles we already have (EMA fetch)
            if candles_4h is not None and not candles_4h.empty and len(candles_4h) >= 10:
                try:
                    lvl_4h = self.level_tracker.analyze(
                        candles_4h.tail(60), direction=direction,
                    )
                    self.linda.record(
                        symbol=symbol, timeframe="4h",
                        level=int(lvl_4h.current_level),
                        direction=direction, now=now,
                    )
                except Exception:
                    pass

            # Read cascade state at all three boundaries so downstream
            # confluence / exit logic can use it. Lesson 55:
            #   15m→1H cascade  = move is starting to size up
            #   1H→4H cascade   = "bigger than it looks", hold runner
            #   4H→1d cascade   = multi-week Linda trade — full hold
            linda_cascade_15m_to_1h = self.linda.cascade_detected(
                symbol, from_tf="15m", to_tf="1h"
            )
            linda_cascade_1h_to_4h = self.linda.cascade_detected(
                symbol, from_tf="1h", to_tf="4h"
            )
            linda_cascade_4h_to_1d = self.linda.cascade_detected(
                symbol, from_tf="4h", to_tf="1d"
            )
            if linda_cascade_15m_to_1h or linda_cascade_1h_to_4h or linda_cascade_4h_to_1d:
                logger.info("mm_linda_cascade_active", symbol=symbol,
                            fifteen_to_one=linda_cascade_15m_to_1h,
                            one_to_four=linda_cascade_1h_to_4h,
                            four_to_daily=linda_cascade_4h_to_1d,
                            direction=direction)
        except Exception:
            pass  # Telemetry only — don't break scan on Linda errors

        # D7: Session-specific entry biases (Lessons 04, 05).
        try:
            self._log_session_entry_bias(
                symbol=symbol,
                session_name=session.session_name,
                current_level=level_analysis.current_level,
                formation_variant=best_formation.variant,
            )
        except Exception:
            pass  # Best-effort telemetry — never block a trade on this

        # Don't enter if Level 3+ already reached (expect reversal)
        # Exception: NYC Reversal (A2) and Stop Hunt (A4) ARE the L3 reversal
        # trades — they explicitly require Level 3+ to fire.
        _l3_bypass_variants = (
            "three_hits_how", "three_hits_low",
            "nyc_reversal", "stophunt_l3",
            "half_batman", "33_trade",
        )
        if level_analysis.current_level >= 3 and best_formation.variant not in _l3_bypass_variants:
            return self._reject("level_too_advanced", symbol, level=level_analysis.current_level, post_formation_candles=len(candles_post_formation))

        self._advance("level_ok")

        # Course A4 (lesson 12): "We do not counter Trend Trade after level 1 Rise"
        # — called out as "the major one that you don't want to break".
        # If we're already past Level 1 in one direction, don't open a trade
        # in the OPPOSITE direction (unless an M/W reversal formation is the
        # explicit trigger — which we handle via the lesson-18 path / level=3
        # reject above). For the in-trend direction, this is a no-op.
        #
        # Previously analyzed BOTH directions simultaneously on the same
        # window — in choppy markets both bullish.current_level AND
        # bearish.current_level report >=1, blocking every signal regardless
        # of trade direction. Fix: use the weekly-cycle-tracked dominant
        # direction, and only block signals that fight it.
        try:
            dominant_dir = getattr(cycle_state, "direction", None)
            if dominant_dir in ("bullish", "bearish"):
                # Only analyze the dominant direction; pick a conservative
                # counter-trend level so we don't block very mild retraces.
                dom = self.level_tracker.analyze(candles_1h.tail(60), direction=dominant_dir)
                if dom.current_level >= 1:
                    is_fighting = (
                        (dominant_dir == "bullish" and trade_direction == "short")
                        or (dominant_dir == "bearish" and trade_direction == "long")
                    )
                    if is_fighting:
                        allowed_variants = (
                            "three_hits_how", "three_hits_low",
                            "final_damage", "multi_session",
                        )
                        if best_formation.variant not in allowed_variants \
                           and not best_formation.at_key_level:
                            return self._reject("counter_trend_after_l1", symbol,
                                                trend_level=dom.current_level,
                                                dominant_dir=dominant_dir,
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
            # Course C2 (lesson 15): "if you don't see the false breakout in
            # your weekend box, also look for W's and Ms. Look for W's and
            # Ms if there is no spike out of the weekend box." The absence of
            # an FMWB doesn't invalidate a valid M/W — we explicitly allow
            # the trade through. Log for telemetry.
            # WeekendAnalysis exposes the box as `trap_box` (not `box`).
            trap_box = getattr(weekend, "trap_box", None)
            if trap_box is not None and getattr(trap_box, "detected", False):
                logger.info("mm_weekend_box_no_fmwb_accepting_mw",
                            symbol=symbol, direction=trade_direction,
                            formation_variant=best_formation.variant,
                            box_high=float(getattr(trap_box, "box_high", 0.0)),
                            box_low=float(getattr(trap_box, "box_low", 0.0)))
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

        # Higher-TF EMAs for L3 targets (course Lesson 12 — "a 200 or 800
        # EMA on a higher time frame as a Target"). Prefer 1D EMAs because
        # they are the true "higher TF" relative to a 1H trade-timeframe
        # (4H is only one step up and is already used by trend_state_4h
        # for HTF-alignment veto). If 1D history is insufficient, fall
        # back to 4H so the L3 slot isn't empty.
        htf_ema_values: dict[int, float] | None = None
        try:
            # Prefer 1D
            if candles_1d is not None and not candles_1d.empty and len(candles_1d) >= 200:
                htf_ema_values = {
                    200: float(candles_1d["close"].ewm(span=200, adjust=False).mean().iloc[-1]),
                }
                if len(candles_1d) >= 800:
                    htf_ema_values[800] = float(
                        candles_1d["close"].ewm(span=800, adjust=False).mean().iloc[-1]
                    )
            # Fallback: 4H if 1D insufficient
            elif candles_4h is not None and not candles_4h.empty and len(candles_4h) >= 200:
                htf_ema_values = {
                    200: float(candles_4h["close"].ewm(span=200, adjust=False).mean().iloc[-1]),
                }
                if len(candles_4h) >= 800:
                    htf_ema_values[800] = float(
                        candles_4h["close"].ewm(span=800, adjust=False).mean().iloc[-1]
                    )
        except Exception:
            htf_ema_values = None

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
            htf_ema_values=htf_ema_values,
        )

        # --- Entry price & retest-slippage gate (2026-04-22 fix) ---
        #
        # Course Lesson 20 / 47 entry for M/W retests: the entry is a limit
        # order placed at the 2nd peak wick — the retest level. For an at-
        # market scanning bot (which we are), the closest honest
        # approximation is: only take the setup when current price is
        # CLOSE to peak2 wick. If price has already bounced far from the
        # retest, we've missed the course's intended entry window.
        #
        # Pre-fix behaviour: entry_price = current_price unconditionally.
        # Measured impact on 5 days of trades:
        #   BTC long  : entry was +4.39% ABOVE peak2 wick  → SL 12.47% (course-correct: 8.45%)
        #   NEAR long : entry was +4.67% ABOVE peak2 wick  → SL 15.77% (course-correct: 11.65%)
        #   BNB long  : entry was +1.70% ABOVE peak2 wick  → SL  8.02% (course-correct: 6.42%)
        # "Wide SL" rejections upstream were symptoms of this entry
        # inflation, not real structural problems.
        #
        # Fix: compute peak2_wick_price early and reject setups where the
        # current market price has slipped past it by more than
        # mm_max_entry_slippage_pct (default 1.0%). Below that threshold,
        # current_price is close enough to peak2_wick that entering at
        # market effectively is entering at the retest.
        peak2_abs_idx_early = lookback_start + best_formation.peak2_idx
        peak2_wick_price_early = 0.0
        try:
            if 0 <= peak2_abs_idx_early < len(candles_1h):
                p2 = candles_1h.iloc[peak2_abs_idx_early]
                if best_formation.type.upper() == "W":
                    peak2_wick_price_early = float(p2["low"])
                else:
                    peak2_wick_price_early = float(p2["high"])
        except Exception:
            peak2_wick_price_early = float(best_formation.peak2_price)

        if peak2_wick_price_early > 0 and self.max_entry_slippage_pct > 0:
            # For a W-long retest, "favourable slippage" = current ABOVE peak2 wick.
            # For an M-short retest, "favourable slippage" = current BELOW peak2 wick.
            if best_formation.type.upper() == "W":
                slip_pct = (current_price - peak2_wick_price_early) / current_price * 100
            else:
                slip_pct = (peak2_wick_price_early - current_price) / current_price * 100
            if slip_pct > self.max_entry_slippage_pct:
                return self._reject(
                    "entry_slipped_from_retest",
                    symbol,
                    current_price=round(current_price, 6),
                    peak2_wick=round(peak2_wick_price_early, 6),
                    slippage_pct=round(slip_pct, 2),
                    threshold_pct=self.max_entry_slippage_pct,
                    formation=best_formation.type,
                    direction=(
                        "long" if best_formation.type.upper() == "W" else "short"
                    ),
                )

        # Calculate entry, SL, and targets
        # EXPERIMENTAL: when `_hypothetical_perfect_entry` is set, simulate
        # filling at the peak2 wick (the course-correct retest price)
        # rather than the live tick. Isolates entry-quality impact from
        # downstream payoff issues. SL distance and TP ladder cascade
        # naturally from this entry.
        if (
            getattr(self, "_hypothetical_perfect_entry", False)
            and peak2_wick_price_early > 0
        ):
            entry_price = peak2_wick_price_early
        else:
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

        # Re-added 2026-04-23 as a REJECT (not a tighten). Earlier removal
        # (2026-04) conflated two distinct actions:
        #   - Tightening SL to improve RR — prohibited per Lesson 53
        #     ("SL goes where it needs to go. NEVER tighten SL.")
        #   - Rejecting a setup because SL distance indicates we MISSED
        #     the retest — course-compatible (Lesson 9: "enter at the
        #     proper swing level"). A wide SL relative to entry means
        #     current_price has drifted far past the formation's
        #     invalidation point — i.e. we are LATE, not at the retest.
        # Keeping the rule-change at the reject path (not the SL
        # placement) is course-faithful. SL is untouched; we simply
        # don't enter when price is no longer at the retest level.
        #
        # Concrete impact (2026-04-22 funnel): LTC long SL 7.15%, SOL
        # long 9.96%, XRP long 10.06% all produce RR < 0.4 that waste
        # agent calls ($0.08 each, re-billed every 5-min scan) even
        # though they will never pass downstream gates.
        sl_distance_pct = abs(entry_price - sl_price) / entry_price * 100
        if sl_distance_pct > self.max_sl_pct:
            return self._reject(
                "sl_too_wide",
                symbol,
                sl_distance_pct=round(sl_distance_pct, 2),
                threshold=self.max_sl_pct,
                entry=entry_price,
                sl=sl_price,
            )

        # Minimum SL distance guard (added 2026-04-15 after STRK entered with
        # SL $0.00000317 above entry — 0.01% of price — and got stopped out
        # on the first tick down). Below a floor the trade isn't meaningfully
        # an M/W retest, it's noise.
        #
        # Course rationale: lesson 9 ("tight stops at retests are correct —
        # ONLY at the proper swing level") implies a real structural SL, not
        # a decimal-rounding SL. We enforce a floor per asset class:
        #   - BTC / ETH: 0.30%
        #   - everything else: 0.50%
        # BOTH floors are still within the course's "1% per trade" risk
        # framework because 1%-risk / 0.3%-SL = ~3.3x leverage, normal.
        floor_pct = 0.30 if symbol.upper().startswith(("BTC/", "ETH/")) else 0.50
        if sl_distance_pct < floor_pct:
            return self._reject(
                "sl_too_tight",
                symbol,
                sl_distance_pct=round(sl_distance_pct, 4),
                floor_pct=floor_pct,
                entry=entry_price,
                sl=sl_price,
            )
        # Also reject if SL is on the wrong side of entry (long SL above
        # entry or short SL below entry) — which was the exact STRK bug:
        # a "long" was opened with SL ABOVE entry, so the first tick down
        # was an instant stop-out.
        is_long_signal = direction == "bullish"
        wrong_side = (is_long_signal and sl_price >= entry_price) or (
            (not is_long_signal) and sl_price <= entry_price
        )
        if wrong_side:
            return self._reject(
                "sl_wrong_side",
                symbol,
                direction=direction,
                entry=entry_price,
                sl=sl_price,
            )

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

        # L1 ≡ L2 collision fix (2026-04-15 — exposed by RENDER trade).
        # When the 50 EMA and 800 EMA converge (tight consolidation), the
        # target analyzer returns the same price for L1 and L2. That caused
        # both partial-close levels to fire simultaneously at the same
        # tick and the trailing SL was placed between the collapsed L1/L2
        # and the real L3 — any normal pullback stopped the runner out.
        #
        # Course spec: L1 (50 EMA) is the first cup, L2 (800 EMA) is a
        # meaningful extension. If they're within 0.2% of entry price,
        # treat L2 as a distance-based extension of L1 (1.6x the L1 reward,
        # capped so it can't exceed t_l3). This keeps the three targets
        # ordered and distinct.
        def _nearly_equal(a: float, b: float, ref: float) -> bool:
            if not a or not b or ref <= 0:
                return False
            return abs(a - b) / ref < 0.002  # 0.2% of entry

        if t_l1 and t_l2 and _nearly_equal(t_l1, t_l2, entry_price):
            is_long_tgt = (trade_direction == "long") if 'trade_direction' in locals() else (direction == "bullish")
            base_reward = abs(t_l1 - entry_price)
            spread_reward = max(base_reward * 1.6, entry_price * 0.005)  # at least +0.5% further
            new_l2 = (entry_price + spread_reward) if is_long_tgt else (entry_price - spread_reward)
            # If there's already a distinct L3 further out, respect it.
            # But if L3 has ALSO collapsed onto L1, this clamp is a no-op —
            # detect that via the separate three-way guard below.
            if t_l3 and not _nearly_equal(t_l1, t_l3, entry_price):
                # Clamp L2 to halfway between L1 and L3 so L3 still dominates.
                halfway = (t_l1 + t_l3) / 2
                new_l2 = (min(new_l2, halfway) if is_long_tgt else max(new_l2, halfway))
            logger.info("mm_l1_l2_collision_spread", symbol=symbol,
                        original_l1=t_l1, original_l2=t_l2, new_l2=new_l2,
                        entry=entry_price)
            t_l2 = new_l2

        # Three-way collapse guard (2026-04-20 — exposed by NEAR trade).
        # When the 50 EMA is unavailable (fallback from primary_l1 to l2)
        # AND the target analyzer returns the same underlying level for
        # l2 and l3 (common when there's only one unrecovered vector or
        # one HOW/LOD in the direction), all three TPs collapse to the
        # same price. The dashboard shows "one TP" and no staggered
        # partial exits ever fire. Course Lesson 47 fallback: when EMA-
        # based levels aren't resolvable, use R-multiples of the SL
        # distance for intermediate exits. Keep the identified target as
        # the far L3 and synthesize L1 at 2R, L2 at 3R.
        if (
            t_l1 and t_l2 and t_l3
            and _nearly_equal(t_l1, t_l3, entry_price)
        ):
            is_long_tgt = (trade_direction == "long") if 'trade_direction' in locals() else (direction == "bullish")
            r = abs(entry_price - sl_price)
            if r > 0:
                if is_long_tgt:
                    synthesized_l1 = entry_price + 2.0 * r
                    synthesized_l2 = entry_price + 3.0 * r
                    in_order = synthesized_l1 < synthesized_l2 < t_l3
                else:
                    synthesized_l1 = entry_price - 2.0 * r
                    synthesized_l2 = entry_price - 3.0 * r
                    in_order = synthesized_l1 > synthesized_l2 > t_l3
                if in_order:
                    logger.info(
                        "mm_targets_staggered_from_r_multiples",
                        symbol=symbol,
                        original_l1=t_l1, original_l2=t_l2, original_l3=t_l3,
                        new_l1=synthesized_l1, new_l2=synthesized_l2,
                        r_multiples_used=(2.0, 3.0),
                        entry=entry_price, sl=sl_price, direction=trade_direction,
                    )
                    t_l1 = synthesized_l1
                    t_l2 = synthesized_l2
                    # t_l3 stays as the originally-identified far target
                else:
                    # Identified L3 is too close (< 3R from entry) — fall
                    # back to pure R-multiples for all three. This is
                    # rare; logs it prominently so we can investigate.
                    logger.warning(
                        "mm_targets_l3_too_close_using_pure_r",
                        symbol=symbol, original_l3=t_l3, risk_r=r,
                        entry=entry_price,
                    )
                    if is_long_tgt:
                        t_l1 = entry_price + 2.0 * r
                        t_l2 = entry_price + 3.0 * r
                        t_l3 = entry_price + 5.0 * r
                    else:
                        t_l1 = entry_price - 2.0 * r
                        t_l2 = entry_price - 3.0 * r
                        t_l3 = entry_price - 5.0 * r

        # Only reject if NO target is available at any level (no EMAs, no
        # vectors, no HOW/LOW in direction).
        if not t_l1:
            return self._reject("no_target_available", symbol, direction=trade_direction,
                                ema_50=ema_values.get(50), ema_200=ema_values.get(200),
                                vectors=len(target_analysis.unrecovered_vectors),
                                entry=entry_price, formation=best_formation.type)

        # TP1-distance cap (2026-04-20). Engineering cap, NOT in course —
        # rejects formation-timeframe entries whose computed TP1 is far
        # enough from entry that the setup would need a multi-week move
        # to reach it. Concrete trigger: BTC 2026-04-20 10:04 UTC entered
        # at $75,251 with TP1 at $92,319 (+22.68%) because all 1H/4H EMAs
        # were below entry — cascade landed on a historical vector 22%
        # away. Disable by setting mm_max_tp1_distance_pct to 0.
        if self.max_tp1_distance_pct > 0 and entry_price > 0:
            tp1_dist_pct = abs(t_l1 - entry_price) / entry_price * 100
            if tp1_dist_pct > self.max_tp1_distance_pct:
                return self._reject(
                    "tp1_too_far",
                    symbol,
                    tp1=round(t_l1, 6),
                    entry=round(entry_price, 6),
                    tp1_distance_pct=round(tp1_dist_pct, 2),
                    max_pct=self.max_tp1_distance_pct,
                    formation=best_formation.type,
                    direction=trade_direction,
                )

        self._advance("target_acquired")

        # R:R check — try L1 first, fall back to L2 if L1 R:R is too low.
        # Do NOT overwrite t_l1 with t_l2 on the R:R-upgrade path — doing so
        # collapsed the 3-tier partial schedule into a 2-tier (L1 partial
        # never fired because t_l1 had been rewritten to L2). Instead, we
        # use L2 for the R:R GATE but preserve the original targets.
        risk = abs(entry_price - sl_price)
        if risk <= 0:
            return self._reject("zero_risk", symbol)

        reward = abs(t_l1 - entry_price)
        rr = reward / risk

        # B4 course-faithful: Linda cascade lowers the min R:R threshold (Lesson 55).
        # When the 1H→4H or 4H→Daily cascade is active AND in our trade direction,
        # the move is "bigger than it looks" — accept the course floor (1.4) instead
        # of the normal aggressive threshold.
        linda_cascade_same_dir = False
        try:
            if linda_cascade_1h_to_4h or linda_cascade_4h_to_1d:
                higher_tf = "4h" if linda_cascade_1h_to_4h else "1d"
                tf_state = self.linda.get(symbol, higher_tf)
                expected_dir = "bullish" if trade_direction == "long" else "bearish"
                if tf_state.direction == expected_dir:
                    linda_cascade_same_dir = True
        except Exception:
            pass

        effective_min_rr = MIN_RR_COURSE_FLOOR if linda_cascade_same_dir else self.min_rr

        if rr < effective_min_rr and t_l2 and self._is_valid_target(t_l2, trade_direction, entry_price):
            reward_l2 = abs(t_l2 - entry_price)
            rr_l2 = reward_l2 / risk
            if rr_l2 >= effective_min_rr:
                logger.info("mm_target_rr_upgraded_to_l2", symbol=symbol,
                            rr_l1=round(rr, 2), rr_l2=round(rr_l2, 2),
                            t_l1=t_l1, t_l2=t_l2)
                # Keep t_l1 untouched so the 30% partial fires at L1.
                # The R:R gate below uses rr_l2 just to pass validation.
                rr = rr_l2

        if rr < effective_min_rr:
            if linda_cascade_same_dir:
                logger.info("mm_linda_cascade_rr_threshold", symbol=symbol,
                            effective_min_rr=effective_min_rr, rr=round(rr, 2))
            return self._reject("low_rr", symbol, rr=round(rr, 2), min_required=effective_min_rr, entry=entry_price, sl=sl_price, t1=t_l1)

        self._advance("rr_passed")

        # Confluence scoring
        at_how = abs(current_price - cycle_state.how) / current_price < 0.005 if cycle_state.how > 0 else False
        at_low = abs(current_price - cycle_state.low) / current_price < 0.005 if cycle_state.low < float("inf") else False

        # D1: iHOD/iLOD confirmation — when using HOD/LOD as key_level confluence,
        # check that the level was confirmed by 30-90 min of sideways holding.
        # This is logging/telemetry only; the `at_how_or_low` confluence flag
        # is already populated above and is not gated on this check.
        try:
            if at_how and cycle_state.how > 0:
                ihod_conf = self.weekly_cycle_tracker.confirm_ihod_ilod(
                    candles_1h, cycle_state.how, "ihod"
                )
                logger.info(
                    "mm_ihod_confirmation",
                    symbol=symbol,
                    level=round(cycle_state.how, 4),
                    confirmed=ihod_conf["confirmed"],
                    hold_minutes=ihod_conf["hold_minutes"],
                    touch_count=ihod_conf["touch_count"],
                )
            if at_low and cycle_state.low < float("inf") and cycle_state.low > 0:
                ilod_conf = self.weekly_cycle_tracker.confirm_ihod_ilod(
                    candles_1h, cycle_state.low, "ilod"
                )
                logger.info(
                    "mm_ilod_confirmation",
                    symbol=symbol,
                    level=round(cycle_state.low, 4),
                    confirmed=ilod_conf["confirmed"],
                    hold_minutes=ilod_conf["hold_minutes"],
                    touch_count=ilod_conf["touch_count"],
                )
        except Exception:
            pass  # Telemetry only — never block a trade on this

        # Bug 5: Three hits boosts at_key_level — 3-hit reversal at HOW/LOW strengthens confluence
        three_hit_boost = False
        if three_hits_at_how and three_hits_at_how.detected and three_hits_at_how.expected_outcome == "reversal":
            three_hit_boost = True
        if three_hits_at_low and three_hits_at_low.detected and three_hits_at_low.expected_outcome == "reversal":
            three_hit_boost = True

        # Course-faithful retest-gate inputs (2026-04 critical fix).
        # Prior code never populated the 4 flags the retest gate reads,
        # capping `retest_conditions_met` at 1 → no trade could ever clear
        # the ≥2/4 threshold. Populate all four from data we already have.

        # (1) at_50_ema: price within 0.3% of the 50 EMA
        ema50_now = self._compute_50ema(candles_1h)
        at_50_ema = False
        if ema50_now and ema50_now > 0:
            at_50_ema = abs(entry_price - ema50_now) / ema50_now < 0.003

        # (3) higher_low (W) / lower_high (M): derived from formation peaks
        # Standard W: two LOWS; a higher-low means peak2_price > peak1_price
        # Standard M: two HIGHS; a lower-high means peak2_price < peak1_price
        hl_lh = False
        try:
            p1 = float(best_formation.peak1_price)
            p2 = float(best_formation.peak2_price)
            if best_formation.type.upper() == "W":
                hl_lh = p2 > p1
            elif best_formation.type.upper() == "M":
                hl_lh = p2 < p1
        except Exception:
            hl_lh = False

        # (4) at_liquidity_cluster: two sources:
        #   (a) Price proximity — HOW/LOW or unrecovered-vector proximity
        #       (within 0.5%). Course lesson 27 treats these as where MMs
        #       pick up liquidity in the absence of Hyblock data.
        #   (b) Binance Long/Short Ratio (free API, no key required).
        #       Course Lesson 11: when delta is HIGH/EXTREME the over-positioned
        #       side will get hunted. We trade contrarian (against the crowd).
        at_liq_cluster = False
        try:
            for lvl in (cycle_state.how, cycle_state.low):
                if lvl and lvl > 0 and lvl != float("inf"):
                    if abs(entry_price - lvl) / entry_price < 0.005:
                        at_liq_cluster = True
                        break
            if not at_liq_cluster and target_analysis.unrecovered_vectors:
                for v in target_analysis.unrecovered_vectors[:3]:
                    vp = getattr(v, "midpoint", None) or getattr(v, "high", None)
                    if vp and abs(entry_price - float(vp)) / entry_price < 0.005:
                        at_liq_cluster = True
                        break
        except Exception:
            at_liq_cluster = False

        # Liquidation cluster check (Course Lesson 11) — Binance free API
        if not at_liq_cluster:
            try:
                liq_data = await self.data_feeds.hyblock.fetch_liquidation_data(symbol)
                if liq_data.available and liq_data.delta_level in ("high", "extreme"):
                    # When delta is HIGH or EXTREME, the over-positioned side will
                    # get hunted. Trade in the direction of the hunt (contrarian).
                    delta_dir = "short" if (liq_data.delta or 0) > 0 else "long"
                    if trade_direction == delta_dir:
                        at_liq_cluster = True
                        logger.info(
                            "mm_liq_cluster_aligned",
                            symbol=symbol,
                            delta=liq_data.delta,
                            level=liq_data.delta_level,
                            trade_dir=trade_direction,
                        )
            except Exception:
                pass  # Best-effort; falls back to proximity-based value above

        # Course lesson 15: M/W inside the weekend trap box is a named
        # high-probability setup. True when (a) the analyzer detected a
        # weekend box this week, (b) FMWB has NOT fired (we're in the
        # "no spike out" branch of the lesson), and (c) the formation's
        # peak/trough prices are all inside the box.
        mw_inside_box = False
        try:
            tb = getattr(weekend, "trap_box", None)
            if (
                tb is not None
                and getattr(tb, "detected", False)
                and not weekend.fmwb.detected
            ):
                bh = float(getattr(tb, "box_high", 0.0))
                bl = float(getattr(tb, "box_low", 0.0))
                if bh > 0 and bl > 0:
                    p1 = float(best_formation.peak1_price or 0)
                    p2 = float(best_formation.peak2_price or 0)
                    tp = float(best_formation.trough_price or 0)
                    pts = [v for v in (p1, p2, tp) if v > 0]
                    if pts and all(bl <= v <= bh for v in pts):
                        mw_inside_box = True
        except Exception:
            mw_inside_box = False

        # B2: Fibonacci alignment — check if entry_price is within 0.3% of any
        # Fibonacci retracement level from the board meeting detector.
        # BoardMeetingDetector is already instantiated as self.board_meeting_detector.
        has_fib_alignment = False
        try:
            bm_detection = self.board_meeting_detector.detect(
                candles_1h, level_direction=direction
            )
            if bm_detection.fib is not None and bm_detection.fib.levels:
                for fib_level in bm_detection.fib.levels:
                    if fib_level.price > 0:
                        if abs(entry_price - fib_level.price) / fib_level.price < 0.003:
                            has_fib_alignment = True
                            break
        except Exception:
            has_fib_alignment = False

        # C2: RSI confirmation — True if RSI divergence aligns with trade OR
        # RSI trend_bias aligns with trade direction (course confirming factor).
        rsi_confirmed: bool | None = None
        if rsi_state is not None:
            bias_match = (
                (trade_direction == "long" and rsi_state.trend_bias == "bullish")
                or (trade_direction == "short" and rsi_state.trend_bias == "bearish")
            )
            divergence_match = (
                rsi_state.divergence_detected
                and (
                    (trade_direction == "long" and rsi_state.divergence_type == "bullish")
                    or (trade_direction == "short" and rsi_state.divergence_type == "bearish")
                )
            )
            rsi_confirmed = bias_match or divergence_match

        # D9: Correlation pre-positioning check (Lesson 19).
        # DXY moves before BTC — if DXY diverges from BTC, log the implied
        # direction. When provider is available, pass result to MMContext.
        correlation_confirmed_flag: bool | None = None
        try:
            corr_signal = await self._check_correlation_signal()
            if corr_signal is not None:
                # Align: if implied BTC direction matches trade direction, confirmed
                if corr_signal.dxy_divergence and corr_signal.confidence > 0:
                    implied_long = corr_signal.implied_btc_direction == "up"
                    trade_is_long = trade_direction == "long"
                    correlation_confirmed_flag = implied_long == trade_is_long
                    logger.info(
                        "mm_correlation_signal",
                        symbol=symbol,
                        dxy_direction=corr_signal.dxy_direction,
                        implied_btc=corr_signal.implied_btc_direction,
                        sp500_aligned=corr_signal.sp500_aligned,
                        confidence=round(corr_signal.confidence, 3),
                        trade_direction=trade_direction,
                        correlation_confirmed=correlation_confirmed_flag,
                    )
        except Exception:
            pass  # Provider unavailable — leave as None

        mm_ctx = MMContext(
            # Passed so _score_ema_alignment can gate its 8 pts on direction
            # match — prior to 2026-04 a bullish stack awarded the same 8 pts
            # to shorts as to longs, directly inflating counter-trend setup
            # scores. See `_score_ema_alignment` docstring.
            trade_direction=trade_direction,
            formation={
                "type": best_formation.type,
                "variant": best_formation.variant,
                "quality": best_formation.quality_score,
                "at_key_level": best_formation.at_key_level,
                "session": session.session_name,
                # Retest condition 3 inputs
                "higher_low": hl_lh and best_formation.type.upper() == "W",
                "lower_high": hl_lh and best_formation.type.upper() == "M",
                "higher_low_or_lower_high": hl_lh,
            },
            ema_state={
                "alignment": ema_state.alignment if ema_state else "mixed",
                "broke_50": ema_break.broke_ema if ema_break else False,
                "volume_confirmed": ema_break.volume_confirmed if ema_break else False,
                # Retest condition 1 input
                "at_50_ema": at_50_ema,
            } if ema_state else {"at_50_ema": at_50_ema},
            level_state={
                "current_level": level_analysis.current_level,
                "svc_detected": level_analysis.svc.detected if level_analysis.svc else False,
                "volume_degrading": level_analysis.volume_degrading,
                # Retest condition 2 input (already covered by has_unrecovered_vector fallback,
                # but populate explicitly for clarity)
                "at_level1_vector": len(target_analysis.unrecovered_vectors) > 0,
            },
            cycle_state={
                "phase": cycle_state.phase,
                "direction": cycle_state.direction,
            },
            # Retest condition 4 input
            has_liquidation_cluster=at_liq_cluster,
            entry_price=entry_price,
            stop_loss=sl_price,
            target_price=t_l1,
            at_session_changeover=session.is_gap,
            at_how_low=at_how or at_low or three_hit_boost,
            has_unrecovered_vector=len(target_analysis.unrecovered_vectors) > 0,
            mw_inside_weekend_box=mw_inside_box,
            moon_phase_aligned=self._moon_phase_aligned(trade_direction, now),
            oi_increasing=oi_increasing,
            has_fib_alignment=has_fib_alignment,
            rsi_confirmed=rsi_confirmed,
            adr_at_fifty_pct=adr_state.at_fifty_pct if adr_state is not None else None,
            correlation_confirmed=correlation_confirmed_flag,
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

        # ---------------------------------------------------------------
        # MM Sanity Agent (Agent 4) — LLM veto layer.
        # ---------------------------------------------------------------
        # Fires on every setup that survives all deterministic rules.
        # Returns None on failure (API error / timeout / SDK missing / no
        # key) — we fail OPEN (approve) rather than halt trading. A VETO
        # decision is binding; a LOW-CONFIDENCE veto is already downgraded
        # inside the agent per `mm_sanity_agent_min_confidence` config.
        #
        # Derived-feature context is assembled via build_context() so the
        # LLM reasons over pre-computed facts rather than raw candles.
        agent_verdict = None  # type: ignore[assignment]
        if getattr(self.config, "mm_sanity_agent_enabled", True):
            try:
                from src.strategy.mm_sanity_agent import build_context
                # Asia spike direction — use the existing detector that
                # already runs as a soft filter earlier in this function.
                # We don't have a separate "asia range pct" computation
                # here; leave None and let the agent work without it.
                asia_range_pct = None
                try:
                    _asia_spike = self.session_analyzer.detect_asia_closing_spike(
                        candles_1h, now,
                    )
                    asia_spike_dir = (
                        _asia_spike.direction if getattr(_asia_spike, "detected", False)
                        else "none"
                    )
                    asia_range_pct = round(
                        float(getattr(_asia_spike, "magnitude_pct", 0.0) or 0.0), 3,
                    )
                except Exception:
                    asia_spike_dir = "none"

                # Recent trades on this symbol — regime signal per Rubric 7.
                try:
                    recent_trades = await self.repo.get_recent_trades_for_symbol(
                        symbol, limit=5,
                    )
                except Exception:
                    recent_trades = []

                agent_ctx = build_context(
                    symbol=symbol,
                    trade_direction=trade_direction,
                    best_formation=best_formation,
                    confluence_result=confluence_result,
                    entry_price=entry_price,
                    sl_ref=sl_price,
                    trend_state_4h=trend_state_4h,
                    trend_state_1d=trend_state_1d,
                    ema_state=ema_state,
                    ema_values=ema_values,
                    session=session,
                    cycle_state=cycle_state,
                    candles_4h=candles_4h,
                    candles_1h=candles_1h,
                    candles_15m=candles_15m,
                    asia_range_pct=asia_range_pct,
                    asia_spike_dir=asia_spike_dir,
                    recent_trades=recent_trades,
                    cycle_count=self.cycle_count,
                    now=now,
                )
                agent_verdict = await self.sanity_agent.review(agent_ctx)
            except Exception as e:
                # Assembly failure — log but don't block. This should be rare.
                logger.warning("mm_sanity_agent_context_failed",
                               symbol=symbol, error=str(e))
                agent_verdict = None

        if agent_verdict is not None and agent_verdict.decision == "VETO":
            return self._reject(
                "sanity_agent_veto",
                symbol,
                agent_reason=agent_verdict.reason,
                agent_confidence=round(agent_verdict.confidence, 3),
                concerns=",".join(agent_verdict.concerns),
                htf_4h=agent_verdict.htf_trend_4h,
                htf_1d=agent_verdict.htf_trend_1d,
                counter_trend=agent_verdict.counter_trend,
                formation_variant=best_formation.variant,
                grade=confluence_result.grade,
            )

        self._advance("sanity_agent_passed")
        # ---------------------------------------------------------------

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

        # Course B4 (lessons 20, 23): capture SVC zone for post-entry
        # "Trapped Traders" invalidation check in _manage_position.
        # Use the Stopping Volume Candle if detected, else fall back to the
        # 1st peak candle as a reasonable proxy.
        svc_high = svc_low = 0.0
        try:
            if level_analysis.svc and level_analysis.svc.detected:
                # mm_levels SVC carries index + high/low
                svc_high = float(getattr(level_analysis.svc, "candle_high", 0.0))
                svc_low = float(getattr(level_analysis.svc, "candle_low", 0.0))
            if svc_high == 0.0 or svc_low == 0.0:
                # Fallback: the 1st-peak candle's full range
                peak1_abs_idx = lookback_start + best_formation.peak1_idx
                if 0 <= peak1_abs_idx < len(candles_1h):
                    p1 = candles_1h.iloc[peak1_abs_idx]
                    svc_high = float(p1["high"])
                    svc_low = float(p1["low"])
        except Exception:
            svc_high = svc_low = 0.0

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
            svc_high=svc_high,
            svc_low=svc_low,
            london_pattern_type=london_pattern_type,
            htf_trend_4h=htf_4h_dir,
            htf_trend_1d=htf_1d_dir,
            counter_trend=is_counter_trend_4h,
            # MM Sanity Agent verdict (only APPROVE reaches here; VETO
            # would have returned earlier via `sanity_agent_veto` reject)
            mm_agent_decision=(
                agent_verdict.decision if agent_verdict is not None else None
            ),
            mm_agent_reason=(
                agent_verdict.reason if agent_verdict is not None else ""
            ),
            mm_agent_confidence=(
                float(agent_verdict.confidence) if agent_verdict is not None else 0.0
            ),
            mm_agent_model=(
                agent_verdict.model if agent_verdict is not None else ""
            ),
            mm_agent_concerns=(
                list(agent_verdict.concerns) if agent_verdict is not None else []
            ),
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

        # Persist to signals table for observability — captures every full
        # signal with its factor breakdown so we can analyze why setups
        # win or lose after the fact. Fire-and-forget; failures only log.
        try:
            await self.repo.insert_signal({
                "symbol": symbol,
                "direction": trade_direction,
                "score": float(confluence_result.score_pct),
                "reasons": [
                    f"formation={best_formation.type}/{best_formation.variant}",
                    f"grade={confluence_result.grade}",
                    f"rr={round(rr, 2)}",
                    f"phase={cycle_state.phase}",
                    f"entry_type={entry_type}",
                    f"htf_4h={htf_4h_dir}",
                    f"htf_1d={htf_1d_dir}",
                    f"counter_trend={is_counter_trend_4h}",
                    f"at_key_level={best_formation.at_key_level}",
                    f"confirmed={best_formation.confirmed}",
                ],
                "components": {
                    factor: round(float(score), 2)
                    for factor, score in confluence_result.factors.items()
                },
                "current_price": float(current_price),
                "acted_on": False,  # set True later when trade is created
                "scan_cycle": int(self.cycle_count),
            })
        except Exception as e:  # pragma: no cover — telemetry best-effort
            logger.warning("insert_signal_failed", symbol=symbol, error=str(e))

        return signal

    def _calculate_signal_density(self, signals: list[MMSignal]) -> dict:
        """Detect market-wide noise vs isolated premium signals.

        The course teaches humans to judge "is this clean?" — a setup where
        only ONE or TWO pairs signal out of the full universe is premium
        (isolated, probably real MM activity). When MANY pairs signal the
        same direction simultaneously it is likely a broad market move or
        manipulation: market-wide noise.

        Rules:
          - noise:   >50% of scanned pairs have signals AND >70% same direction
          - premium: <20% of scanned pairs signal AND top signal has score >60
        """
        if not signals:
            return {"density_pct": 0.0, "is_noise": False, "is_premium": False}

        pairs_scanned = (self.last_funnel or {}).get("pairs_scanned", 0)
        if not pairs_scanned:
            # Fallback: estimate from open positions + signals + rejections
            rejected_total = (self.last_funnel or {}).get("rejected_total", 0)
            pairs_scanned = len(self.positions) + len(signals) + rejected_total

        if pairs_scanned == 0:
            return {"density_pct": 0.0, "is_noise": False, "is_premium": False}

        density_pct = (len(signals) / pairs_scanned) * 100

        long_count = sum(1 for s in signals if s.direction == "long")
        short_count = len(signals) - long_count
        direction_alignment = max(long_count, short_count) / len(signals) if signals else 0

        is_noise = density_pct > 50 and direction_alignment > 0.7
        is_premium = (
            density_pct < 20
            and len(signals) >= 1
            and signals[0].confluence_score > 60
        )

        return {
            "density_pct": round(density_pct, 1),
            "long_count": long_count,
            "short_count": short_count,
            "direction_alignment": round(direction_alignment, 2),
            "is_noise": is_noise,
            "is_premium": is_premium,
        }

    async def _process_entries(self, signals: list[MMSignal]) -> None:
        """Process entry signals — execute the best ones."""
        open_count = len(self.positions)

        # --- Signal density edge (course: clean vs noisy market) ---
        density = self._calculate_signal_density(signals)
        effective_min_confluence = self.min_confluence
        effective_min_rr = self.min_rr

        if density["is_noise"]:
            effective_min_confluence = self.min_confluence + 10
            logger.warning(
                "mm_density_noise",
                density_pct=density["density_pct"],
                direction_alignment=density["direction_alignment"],
                raising_confluence_by=10,
                effective_min_confluence=effective_min_confluence,
            )
        elif density["is_premium"]:
            effective_min_rr = max(MIN_RR_COURSE_FLOOR, self.min_rr - 0.1)
            logger.info(
                "mm_density_premium",
                density_pct=density["density_pct"],
                top_score=signals[0].confluence_score if signals else 0,
                effective_min_rr=effective_min_rr,
            )

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

            # Density-adjusted confluence filter
            if signal.confluence_score < effective_min_confluence:
                logger.info(
                    "mm_reject_density_confluence",
                    symbol=signal.symbol,
                    score=signal.confluence_score,
                    required=effective_min_confluence,
                )
                continue

            # Density-adjusted R:R filter
            if signal.risk_reward < effective_min_rr:
                logger.info(
                    "mm_reject_density_rr",
                    symbol=signal.symbol,
                    rr=signal.risk_reward,
                    required=effective_min_rr,
                )
                continue

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

    def _aggregate_open_risk_usd(self) -> float:
        """Sum of dollar risk (entry→current SL) across all open positions.

        Uses each position's *current* SL, not the original — so trades
        that have had SL tightened to breakeven/better contribute zero
        (or even negative, clamped to 0) to aggregate risk. That is the
        correct live number: a locked-in-profit position doesn't consume
        new-trade risk budget.
        """
        total = 0.0
        for pos in self.positions.values():
            if pos.entry_price <= 0 or pos.quantity <= 0:
                continue
            # Risk = |entry - current SL| * remaining qty, but only if
            # SL is still in the loss direction. Past breakeven (long
            # with SL > entry, or short with SL < entry) risk is 0.
            if pos.direction == "long":
                if pos.stop_loss >= pos.entry_price:
                    continue  # SL is at/above entry → no open risk
                risk = (pos.entry_price - pos.stop_loss) * pos.quantity
            else:  # short
                if pos.stop_loss <= pos.entry_price:
                    continue  # SL is at/below entry → no open risk
                risk = (pos.stop_loss - pos.entry_price) * pos.quantity
            total += max(0.0, risk)
        return round(total, 2)

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

        # Aggregate-risk budget gate (course 1%/trade expressed at
        # portfolio level — see MAX_AGGREGATE_RISK_PCT). This replaces
        # the old mm_max_positions=3 hard cap, which was a human-
        # attention limit, with a risk-budget cap that scales with SL
        # tightness and a bot's ability to track many positions.
        cap_pct = self.max_aggregate_risk_pct
        if cap_pct > 0:
            current_risk = self._aggregate_open_risk_usd()
            proposed_risk = float(pos_result.risk_amount_usd or 0.0)
            projected_risk = current_risk + proposed_risk
            cap_usd = account_balance * (cap_pct / 100.0)
            if projected_risk > cap_usd:
                logger.info(
                    "mm_reject_aggregate_risk",
                    symbol=signal.symbol,
                    current_risk_usd=round(current_risk, 2),
                    proposed_risk_usd=round(proposed_risk, 2),
                    projected_risk_usd=round(projected_risk, 2),
                    cap_usd=round(cap_usd, 2),
                    cap_pct=cap_pct,
                    open_positions=len(self.positions),
                )
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
                "original_stop_loss": signal.stop_loss,  # baseline for monotonic SL
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
                # Per-trade MM lifecycle state so restarts preserve the
                # Refund Zone trigger (lesson 49), the SVC invalidation zone
                # (lessons 20/23), and the SL progression flags (47/48).
                "mm_entry_type": signal.entry_type,
                "mm_peak2_wick_price": signal.peak2_wick_price,
                "mm_svc_high": signal.svc_high,
                "mm_svc_low": signal.svc_low,
                "mm_sl_moved_to_breakeven": False,
                "mm_sl_moved_under_50ema": False,
                "mm_took_200ema_partial": False,
                # MFE baseline at entry (migration 020 / P3 fix). Live
                # updates happen in _manage_position as price moves.
                "mm_max_favorable_excursion_r": 0.0,
                # HTF trend snapshot at entry (migration 018) — so post-mortems
                # can separate trend-aligned losses from counter-trend losses
                # without re-fetching historical 4H/1D candles.
                "htf_trend_4h": getattr(signal, "htf_trend_4h", "unknown"),
                "htf_trend_1d": getattr(signal, "htf_trend_1d", "unknown"),
                "counter_trend": bool(getattr(signal, "counter_trend", False)),
                # MM Sanity Agent verdict (migration 019). Only APPROVE
                # verdicts reach insert; VETOs rejected upstream.
                "mm_agent_decision": getattr(signal, "mm_agent_decision", None),
                "mm_agent_reason": getattr(signal, "mm_agent_reason", "") or "",
                "mm_agent_confidence": float(getattr(signal, "mm_agent_confidence", 0.0) or 0.0),
                "mm_agent_model": getattr(signal, "mm_agent_model", "") or "",
                "mm_agent_concerns": list(getattr(signal, "mm_agent_concerns", []) or []),
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
            svc_high=getattr(signal, "svc_high", 0.0),
            svc_low=getattr(signal, "svc_low", 0.0),
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

        # D5 (lessons 05, 16): log stagger entry prices for manual reference.
        # In the current implementation we execute a single market order at
        # fill_price. The stagger prices show where 2nd and 3rd limit orders
        # would optimally sit. These can be placed manually or will be
        # automated when the exchange interface adds limit order support.
        stagger_prices = self._calculate_stagger_entries(fill_price, signal.stop_loss, signal.direction)
        logger.info(
            "mm_stagger_entry_prices",
            symbol=signal.symbol,
            direction=signal.direction,
            stagger_1={"price": stagger_prices[0]["price"], "weight": stagger_prices[0]["weight"]},
            stagger_2={"price": stagger_prices[1]["price"], "weight": stagger_prices[1]["weight"]},
            stagger_3={"price": stagger_prices[2]["price"], "weight": stagger_prices[2]["weight"]},
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

        # Course B4 (lessons 20, 23): SVC "Trapped Traders" zone invalidation.
        # Course verbatim: "We always want to see price fail to get back to
        # the wick of a stopping volume candle." This is a CONTINUOUS
        # guarantee over the life of the trade (until L1 is cleared), not
        # a one-off check of the most recent bar.
        #
        # 2026-04-20 NEAR fix: the course rule is "break out of SVC then
        # return = invalidation." Previously we fired on ANY close inside
        # the zone since entry — but if the entry price itself is inside
        # the SVC zone (common for W-retest entries at the wick), the
        # trade would cut on the very first 1H close near entry. NEAR
        # closed at +$553 profit but missed TP1 (+$1,473) because of this.
        # Require a clean breakout ABOVE svc_high (for longs) / BELOW
        # svc_low (for shorts) BEFORE the return-to-SVC invalidation
        # becomes active. 0.2% buffer matches the style used in
        # _tighten_sl for similar level-proximity checks.
        #
        # Still only pre-L1: once the 50 EMA breaks with volume and price
        # clears L1, the move is in progress and the SVC stops being the
        # relevant invalidation zone.
        if pos.current_level == 0 and pos.svc_high > 0 and pos.svc_low > 0:
            try:
                candles_short = await self.candle_manager.get_candles(symbol, "1h", limit=50)
                if candles_short is not None and not candles_short.empty and len(candles_short) >= 2:
                    # Trim the in-progress bar — only check fully closed candles.
                    closed_only = candles_short.iloc[:-1]
                    # Only look at bars closed AFTER we entered (can't invalidate
                    # on bars older than the trade).
                    if pos.entry_time and hasattr(closed_only.index, "tz"):
                        closed_only = closed_only[closed_only.index >= pos.entry_time]
                    broke_out = False
                    returned = False
                    return_close = None
                    return_ts = None
                    for ts, row in closed_only.iterrows():
                        c = float(row["close"])
                        # Phase 1: waiting for clean breakout out of the
                        # SVC zone. The breakout candle itself by
                        # definition closes outside the zone, so we
                        # don't need to also check it for "return."
                        if not broke_out:
                            if pos.direction == "long" and c > pos.svc_high * 1.002:
                                broke_out = True
                            elif pos.direction == "short" and c < pos.svc_low * 0.998:
                                broke_out = True
                            continue
                        # Phase 2: post-breakout — any close returning
                        # INTO the SVC zone invalidates the formation.
                        if pos.svc_low <= c <= pos.svc_high:
                            returned = True
                            return_close = c
                            return_ts = ts
                            break
                    if returned:
                        logger.info("mm_svc_wick_return_cut", symbol=symbol,
                                    return_close=return_close,
                                    return_ts=str(return_ts),
                                    svc_high=pos.svc_high, svc_low=pos.svc_low)
                        await self._close_position(pos, current_price, "svc_wick_return")
                        return
                    elif not broke_out:
                        # Still in pre-breakout phase — log once per cycle
                        # for telemetry, no action.
                        logger.debug("mm_svc_pre_breakout",
                                     symbol=symbol,
                                     svc_high=pos.svc_high, svc_low=pos.svc_low,
                                     current_price=current_price)
            except Exception as e:
                logger.debug("mm_svc_check_failed", symbol=symbol, error=str(e))

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

        # B1: Scratch rule — course Lesson 13 [47:00] verbatim:
        #
        #   "If you're not in substantial profit within two hours you
        #    scratch the trade. It means the Market Maker has a different
        #    plan. That's the rule. Market Maker only holds the
        #    consolidation level to get more contracts."
        #
        # P3 FIX 2026-04-22 — "within two hours" is a window, not an
        # instant.
        #
        # The prior rule measured unrealized P&L at the 2h mark only.
        # That closed trades that had been +1R mid-flight but pulled
        # back to break-even by the 2h check — the trade HAD shown
        # "substantial profit within two hours" but we closed it anyway
        # because we only looked at the snapshot. User reported the
        # pattern: "if the initial trade goes in the wrong direction,
        # but comes back in the direction we expect, it gets closed...
        # before it can actually realized."
        #
        # The fix: track Max Favorable Excursion (MFE) in R-multiples
        # continuously. "R" = distance from entry to original_stop_loss.
        # A trade's MFE is the highest R-value it has reached at any
        # point during its lifetime. At the 2h mark, we check whether
        # MFE ever cleared `scratch_mfe_threshold_r` (default 0.3R).
        # If so, the trade has shown "substantial profit within two
        # hours" per the course and is safe from scratch.
        #
        # Threshold rationale: 0.3R is conservative — low enough that
        # any trade starting to work has crossed it, high enough that
        # noise alone won't trigger it. Tunable via
        # config.mm_scratch_mfe_threshold_r.
        #
        # MFE persistence: on every tick we update pos.MFE and if it
        # increased by ≥0.1R since last persist, write to DB so a
        # mid-trade restart doesn't lose the fact that the bar was
        # already cleared.
        #
        # Prior anti-patterns removed here (commit 2a04c2e, reverted):
        # dynamic-by-SL scratch + board-meeting exemption. Both were
        # inventions not in the course.
        if pos.original_stop_loss > 0:
            risk_per_unit = abs(pos.entry_price - pos.original_stop_loss)
            if risk_per_unit > 0:
                if pos.direction == "long":
                    current_r = (current_price - pos.entry_price) / risk_per_unit
                else:
                    current_r = (pos.entry_price - current_price) / risk_per_unit
                if current_r > pos.max_favorable_excursion_r:
                    prev_mfe = pos.max_favorable_excursion_r
                    pos.max_favorable_excursion_r = current_r
                    # Persist in ~0.1R buckets to keep DB writes bounded
                    # — worst case 10 writes per trade from 0 to 1R.
                    if current_r - prev_mfe >= 0.1 and pos.trade_id:
                        try:
                            await self.repo.update_trade(pos.trade_id, {
                                "mm_max_favorable_excursion_r": round(
                                    pos.max_favorable_excursion_r, 3,
                                ),
                            })
                        except Exception as e:
                            logger.debug("mm_mfe_persist_failed",
                                         symbol=symbol, error=str(e))

        now = datetime.now(timezone.utc)
        elapsed = (now - pos.entry_time).total_seconds()
        if elapsed >= 7200:  # 2 hours
            # Scratch iff MFE never cleared the threshold during the
            # two-hour window. A trade that was once in substantial
            # profit but has since retraced is NOT scratched — that's
            # a separate management concern (breakeven stop, partial
            # take) handled elsewhere.
            if pos.max_favorable_excursion_r < self.scratch_mfe_threshold_r:
                # Recompute gross for log continuity with the old format
                if pos.direction == "long":
                    gross = (current_price - pos.entry_price) * pos.quantity
                else:
                    gross = (pos.entry_price - current_price) * pos.quantity
                logger.info(
                    "mm_scratch_rule",
                    symbol=symbol,
                    entry_time=pos.entry_time.isoformat(),
                    elapsed_hours=round(elapsed / 3600, 2),
                    mfe_r=round(pos.max_favorable_excursion_r, 3),
                    mfe_threshold_r=self.scratch_mfe_threshold_r,
                    unrealized_gross_usd=round(gross, 2),
                    current_price=current_price,
                    entry_price=pos.entry_price,
                )
                await self._close_position(pos, current_price, "scratch_2h")
                return

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

                    # B8 (lesson 13): Board meeting re-entry opportunity.
                    # Course: "Beginners should take profit at board meetings, then
                    # re-enter after." When a board meeting is detected between levels
                    # and we have already taken a partial profit, log the re-entry
                    # opportunity. The log enables manual monitoring and future
                    # automation — we do not auto-execute re-entry here because that
                    # requires additional exchange logic (limit orders, sizing).
                    if pos.partial_closed_pct > 0 and new_level in (1, 2):
                        logger.info(
                            "mm_board_meeting_reentry_opportunity",
                            symbol=symbol,
                            level=new_level,
                            partial_closed=pos.partial_closed_pct,
                        )

            pos.current_level = new_level

            # Persist level advance + SL change to DB
            try:
                await self.repo.update_trade(pos.trade_id, {
                    "stop_loss": pos.stop_loss,
                    "current_tier": new_level,
                })
            except Exception as e:
                logger.debug("mm_level_db_update_failed", error=str(e))

        # Linda cascade check — course lesson 55: when the 1H 3-level cycle
        # completes and cascades up to 4H (or 4H → Daily), the move is
        # multi-week. Don't close on L3 completion if the cascade is active
        # in our direction; the runner is the whole point of a Linda Trade.
        linda_same_dir = False
        try:
            cascade_1h = self.linda.cascade_detected(symbol, from_tf="1h", to_tf="4h")
            cascade_4h = self.linda.cascade_detected(symbol, from_tf="4h", to_tf="1d")
            if (cascade_1h or cascade_4h):
                tf_state = self.linda.get(symbol, "4h")
                expected_dir = "bullish" if pos.direction == "long" else "bearish"
                if tf_state.direction == expected_dir:
                    linda_same_dir = True
        except Exception:
            pass

        # B3 (lessons 12, 18): EMA fan-out at L3 = imminent reversal warning.
        # Informational only — SVC / vol-degradation checks below own the close.
        self._maybe_log_ema_fan_out_warning(pos, candles_1h)

        # B5 (lessons 08, 18): Wick direction change at L3 = exhaustion warning.
        # During a 3-level rise, wicks should be at the BOTTOM of candles (stop
        # hunting longs). When wicks turn to the TOP at L3, it signals exhaustion
        # and an imminent reversal. Informational only — existing SVC / vol-degradation
        # checks handle actual exits.
        if pos.current_level >= 3 and candles_1h is not None:
            pos_direction = "bullish" if pos.direction == "long" else "bearish"
            wick_reversal = self._detect_wick_direction_change(candles_1h, pos_direction)
            if wick_reversal:
                logger.info(
                    "mm_wick_direction_warning",
                    symbol=symbol,
                    level=pos.current_level,
                    direction=pos.direction,
                )

        # D6: MM Candle Reframing at Level 3.
        # Course teaches: a big green candle at L3 during a rise = BEARISH
        # (Market Maker distributing / selling into retail buying). The large
        # body candle is the MM reframe signal — informational warning here.
        if pos.current_level >= 3 and candles_1h is not None:
            pos_direction = "bullish" if pos.direction == "long" else "bearish"
            reframe = self._detect_mm_candle_reframe(candles_1h, pos_direction, pos.current_level)
            if reframe:
                logger.info(
                    "mm_candle_reframe_warning",
                    symbol=symbol,
                    level=pos.current_level,
                    direction=pos.direction,
                    msg="Large-body candle at L3 = MM distribution/absorption. Imminent reversal likely.",
                )

        # Check for Stopping Volume Candle at Level 3 (unless a Linda cascade
        # is running in our direction — lesson 55 — in which case SVC in the
        # OPPOSITE direction is the exit, but matching-direction SVCs are
        # just a pause inside a bigger trend).
        if new_level >= 3 and level_analysis and level_analysis.svc and level_analysis.svc.detected:
            if linda_same_dir:
                logger.info("mm_svc_ignored_linda_cascade", symbol=symbol, level=new_level)
            else:
                logger.info("mm_svc_detected", symbol=symbol, level=new_level)
                if pos.partial_closed_pct < 1.0:
                    await self._close_position(pos, current_price, "svc_level_3")
                    return

        # Course C6 (lessons 10, 24, 48): 200 EMA rejection with hammer /
        # inverted hammer = partial TP trigger. "Usually it will then make a
        # run for the 200 and reject there. That rejection starts to pull
        # back into the board meeting."
        # Only fires once per position (took_200ema_partial guard).
        if not pos.took_200ema_partial and candles_1h is not None and len(candles_1h) >= 200:
            try:
                ema200 = candles_1h["close"].ewm(span=200, adjust=False).mean()
                ema200_now = float(ema200.iloc[-1])
                last = candles_1h.iloc[-2] if len(candles_1h) >= 2 else None
                if last is not None and ema200_now > 0:
                    o = float(last["open"])
                    h = float(last["high"])
                    low = float(last["low"])
                    c = float(last["close"])
                    # Distance to 200 EMA as % of price
                    dist_pct = abs(current_price - ema200_now) / current_price
                    near_200ema = dist_pct < 0.015  # within 1.5%
                    if near_200ema:
                        from src.strategy.mm_formations import _is_hammer, _is_inverted_hammer
                        rejected = False
                        # Long position → look for hammer at 200 EMA (support bounce)
                        if pos.direction == "long" and _is_hammer(o, h, low, c):
                            rejected = True
                        # Short position → look for inverted hammer at 200 EMA
                        elif pos.direction == "short" and _is_inverted_hammer(o, h, low, c):
                            rejected = True
                        if rejected:
                            logger.info("mm_200ema_hammer_partial", symbol=symbol,
                                        ema_200=ema200_now, price=current_price,
                                        direction=pos.direction)
                            await self._take_partial(pos, level=max(pos.current_level, 1), current_price=current_price)
                            pos.took_200ema_partial = True
                            self._persist_lifecycle_flags(pos)
            except Exception as e:
                logger.debug("mm_200ema_partial_check_failed", symbol=symbol, error=str(e))

        # Check Friday UK session exit.
        # Course lesson 12 ("start closing by Friday UK close") applies
        # regardless of level — holding L1 trades into Friday US open is
        # exactly the Friday-Trap liquidity grab the course warns about.
        # Previously only fired at L>=2; now closes any open trade at
        # Friday UK close so the Friday-Trap phase starts flat.
        #
        # B6 course-faithful: conditional weekend hold (Lesson 12 nuance).
        # Exception: if the position has made progress (current_level >= 1)
        # AND the SL has been moved to breakeven, we are "playing with house
        # money" — the course allows holding through the weekend in this case.
        session = self.session_analyzer.get_current_session()
        if session.session_name == "uk" and session.day_of_week == 4:  # Friday
            # D8: Detect and log the Friday trap pattern progression before closing.
            # The existing UK exit already closes positions at the right time;
            # this adds pattern-awareness logging so post-trade analysis can
            # identify which phase of the Friday trap the exit occurred in.
            if candles_1h is not None:
                try:
                    now_dt = datetime.now(timezone.utc)
                    friday_trap = self.weekly_cycle_tracker.detect_friday_trap_pattern(
                        candles_1h, now_dt
                    )
                    if friday_trap is not None:
                        logger.info(
                            "mm_friday_trap_pattern_detected",
                            symbol=symbol,
                            phase=friday_trap["phase"],
                            direction=friday_trap["direction"],
                            position_direction=pos.direction,
                            level=new_level,
                        )
                except Exception:
                    pass  # Telemetry only

            if pos.current_level >= 1 and pos.sl_moved_to_breakeven:
                logger.info("mm_weekend_hold_allowed", symbol=symbol,
                            level=new_level, sl_at_breakeven=True)
                return
            logger.info("mm_friday_uk_exit", symbol=symbol, level=new_level)
            await self._close_position(pos, current_price, "friday_uk_exit")
            return

        # Volume degradation at Level 3 = exit.
        # Guards:
        #   (a) Only fires after we've taken the L2 partial (>=50% closed).
        #       Previously could dump a freshly-opened L3 entry whose first
        #       candle just happened to show lower volume than the last one.
        #   (b) Skipped when a Linda cascade is running in our direction
        #       (course lesson 55 — the big multi-week trade).
        if (
            new_level >= 3
            and level_analysis
            and level_analysis.volume_degrading
            and pos.partial_closed_pct >= 0.5
            and not linda_same_dir
        ):
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
            """Apply SL move if it tightens AND stays on the correct side.

            (Previously the cap was `new_sl >= entry * 1.1` for longs — that
            DID allow a long's SL to be moved to entry+9%, which is the
            wrong side of entry. For a long, SL above entry is a guaranteed
            instant stop; the only time we legitimately move SL to-or-above
            entry is the deliberate breakeven-at-L2 step — handled below.)
            """
            # Hard side guard: long SL must stay below entry UNLESS we're
            # explicitly going to breakeven at L2; short SL must stay above.
            is_breakeven_move = reason.startswith("breakeven")
            if pos.direction == "long":
                if new_sl > pos.entry_price and not is_breakeven_move:
                    return False
            else:
                if new_sl < pos.entry_price and not is_breakeven_move:
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
            # (a) Move SL to breakeven first if we haven't yet.
            # Account for round-trip fees so the "breakeven" actually breaks
            # even: entry * (1 + 2 * fee_rate) for longs. Previously moved
            # to pure entry, which guaranteed a small loss via fees.
            fee_rate = 0.0004  # 4bps one-way — typical futures taker
            if not pos.sl_moved_to_breakeven:
                if pos.direction == "long":
                    be_price = pos.entry_price * (1 + 2 * fee_rate)
                else:
                    be_price = pos.entry_price * (1 - 2 * fee_rate)
                if _apply(be_price, "breakeven_at_l2"):
                    pos.sl_moved_to_breakeven = True
                    # Persist immediately — otherwise a restart between
                    # breakeven-move and under-50ema-move would re-try both.
                    self._persist_lifecycle_flags(pos)

            # (b) Once at breakeven, try to move SL just under the 50 EMA
            # Lesson 48: "Once Level 2 is running: can place SL just under 50 EMA"
            if pos.sl_moved_to_breakeven and not pos.sl_moved_under_50ema:
                ema_50 = self._compute_50ema(candles_1h)
                if ema_50 is not None:
                    if pos.direction == "long":
                        candidate_sl = ema_50 * 0.998  # 0.2% buffer below
                    else:
                        candidate_sl = ema_50 * 1.002  # 0.2% buffer above
                    # Wrong-side guard: if price has already moved past the
                    # 50 EMA (e.g. a long where price is now BELOW the EMA),
                    # skipping — otherwise candidate_sl lands above current
                    # price and stops us out instantly.
                    try:
                        current_close = float(candles_1h.iloc[-1]["close"])
                        if pos.direction == "long" and candidate_sl >= current_close:
                            return
                        if pos.direction == "short" and candidate_sl <= current_close:
                            return
                    except Exception:
                        pass
                    if _apply(candidate_sl, "under_50ema_at_l2"):
                        pos.sl_moved_under_50ema = True
                        self._persist_lifecycle_flags(pos)

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

    def _log_session_entry_bias(
        self,
        symbol: str,
        session_name: str,
        current_level: int,
        formation_variant: str,
    ) -> None:
        """D7: Log session-specific entry bias warnings (Lessons 04, 05).

        UK session → trend-following bias. Counter-trend reversal entries at
        L3 during UK open are the NYC Reversal pattern — a US-session setup,
        not a UK setup. Multi-session formations are exempt (they bridge
        sessions and are inherently high-probability).

        US first 3 hours → reversal bias. No action here; that gate lives in
        the Phase 2 NYC Reversal module.

        This is a SOFT filter — never rejects a trade, only logs.
        """
        if session_name == "uk" and current_level >= 3:
            if formation_variant not in ("multi_session",):
                logger.info(
                    "mm_uk_reversal_warning",
                    symbol=symbol,
                    level=current_level,
                    formation_variant=formation_variant,
                    note="L3_reversal_during_UK_is_typically_US_setup",
                )

    def _maybe_log_ema_fan_out_warning(self, pos: "MMPosition", candles_1h: pd.DataFrame) -> None:
        """Course B3 (lessons 12, 18): log EMA_FAN_OUT_L3_WARNING when a position
        reaches Level 3 and the EMAs are visibly fanning out.

        "EMA fan-out at Level 3 = imminent reversal" — the warning is informational
        only; the actual close decision is left to the existing SVC / vol-degradation
        checks that follow in _manage_position().
        """
        if pos.current_level < 3:
            return
        if self._detect_ema_fan_out(candles_1h):
            logger.info(
                "EMA_FAN_OUT_L3_WARNING",
                symbol=pos.symbol,
                level=pos.current_level,
            )

    def _detect_wick_direction_change(self, candles: pd.DataFrame, direction: str) -> bool:
        """Course B5 (lessons 08, 18): detect wick direction change at Level 3.

        During a 3-level rise (bullish), wicks should be at the BOTTOM of candles
        (market makers stop-hunting longs beneath). At Level 3 top, wicks turn to
        the TOP of candles — this is exhaustion and signals a reversal.

        Two wicks together after an aggressive move = MM stopping momentum.

        Args:
            candles: 1H OHLCV DataFrame.
            direction: "bullish" (long) or "bearish" (short).

        Returns:
            True if wicks have turned in the reversal direction (warning signal).
        """
        if candles is None or candles.empty or len(candles) < 5:
            return False
        try:
            last5 = candles.iloc[-5:]
            top_wick_ratios: list[float] = []
            for _, row in last5.iterrows():
                h = float(row["high"])
                low = float(row["low"])
                o = float(row["open"])
                c = float(row["close"])
                candle_range = h - low
                if candle_range <= 0:
                    top_wick_ratios.append(0.0)
                    continue
                top_wick = h - max(o, c)
                ratio = top_wick / candle_range
                top_wick_ratios.append(ratio)

            avg_top_wick = sum(top_wick_ratios) / len(top_wick_ratios) if top_wick_ratios else 0.0

            if direction == "bullish":
                # During bullish rise wicks should be at bottom; if avg top wick
                # ratio > 0.5, wicks have turned to the top → exhaustion warning
                return avg_top_wick > 0.5
            else:
                # For short: wicks should be at top during descent; if top wick
                # ratio < 0.5 (wicks shifted to bottom), reversal warning
                return avg_top_wick < 0.5
        except Exception:
            return False

    def _detect_mm_candle_reframe(
        self,
        candles_1h: pd.DataFrame,
        direction: str,
        current_level: int,
    ) -> bool:
        """D6: MM Candle Reframing at Level 3.

        Course teaches: a big green candle at Level 3 during a rise is BEARISH
        — the Market Maker is distributing / selling into retail's bullish
        enthusiasm. Conversely a big red candle at L3 during a drop is
        BULLISH (MM absorbing into fear). Treat large-body candles as the
        OPPOSITE of what they look like.

        Conditions:
          - At Level 3 (current_level >= 3).
          - Check last 3 closed candles for a large-body candle:
              body > 70% of full range AND body > 2x average body size.
          - During a rise (long): bullish large candle → bearish reframe.
          - During a drop (short): bearish large candle → bullish reframe.

        Returns:
            True if a MM reframe signal is detected (warning, not hard exit).
        """
        if current_level < 3:
            return False
        if candles_1h is None or candles_1h.empty or len(candles_1h) < 10:
            return False
        try:
            recent = candles_1h.iloc[-4:-1]  # last 3 closed candles
            if len(recent) < 3:
                return False

            # Average body size over last 10 candles (normalised base)
            last10 = candles_1h.tail(10)
            avg_body = float((abs(last10["close"] - last10["open"])).mean())

            for _, row in recent.iterrows():
                o = float(row["open"])
                h = float(row["high"])
                lo = float(row["low"])
                c = float(row["close"])
                full_range = h - lo
                body = abs(c - o)
                if full_range <= 0 or avg_body <= 0:
                    continue
                # Large body: body > 70% of range AND > 2x average body
                is_large = (body / full_range) > 0.70 and body > (2.0 * avg_body)
                if not is_large:
                    continue
                is_bullish_candle = c > o
                # Reframe signal: large bullish candle during a rise (long) → bearish warning
                if direction == "bullish" and is_bullish_candle:
                    return True
                # Reframe signal: large bearish candle during a drop (short) → bullish warning
                if direction == "bearish" and not is_bullish_candle:
                    return True
            return False
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

    def _calculate_stagger_entries(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str,
    ) -> list[dict]:
        """Course D5 (lessons 05, 16): calculate 3 stagger prices across entry zone.

        Instead of a single market order at entry, the course recommends 2-3 limit
        orders spread across the entry zone toward the stop loss. This gets a better
        average fill and increases the probability of being in the trade at the
        optimal price.

        For long:  stagger at entry, entry-0.3*(entry-sl), entry-0.5*(entry-sl)
        For short: stagger at entry, entry+0.3*(entry-sl), entry+0.5*(entry-sl)
        Weights: 50% at entry, 30% at mid-zone, 20% at deep-zone.

        Note: This method calculates prices only. Actual limit order execution is
        left for future implementation when exchange interface supports it.

        Args:
            entry_price: The primary entry price.
            stop_loss: Stop loss price.
            direction: "long" or "short".

        Returns:
            List of 3 dicts, each with 'price' (float) and 'weight' (float, 0-1).
        """
        distance = abs(entry_price - stop_loss)

        if direction == "long":
            prices = [
                entry_price,
                entry_price - 0.3 * distance,
                entry_price - 0.5 * distance,
            ]
        else:
            prices = [
                entry_price,
                entry_price + 0.3 * distance,
                entry_price + 0.5 * distance,
            ]

        weights = [0.50, 0.30, 0.20]

        return [{"price": p, "weight": w} for p, w in zip(prices, weights)]

    def _compute_50ema(self, candles_1h: pd.DataFrame) -> float | None:
        """Compute the 50 EMA of the provided 1H candles. None if insufficient data."""
        if candles_1h is None or candles_1h.empty or len(candles_1h) < 50:
            return None
        try:
            ema = candles_1h["close"].ewm(span=50, adjust=False).mean()
            return float(ema.iloc[-1])
        except Exception:
            return None

    def _check_inside_hits_15m(
        self,
        best_formation,
        candles_1h: pd.DataFrame,
        candles_15m: pd.DataFrame,
        lookback_start: int,
    ) -> bool:
        """Course B1 (lesson 20): verify ≥3 TRAP candles on the 15m
        timeframe between the pullback valley and the 2nd peak.

        Course definition of a trap candle on the inside right side:
          - W formation: a RED-bodied 15m candle that makes a NEW LOW
            (since the 1H trough). Each such candle takes out resting
            stops below the prior swing low — inducing shorts who get
            trapped when the W reversal completes.
          - M formation: a GREEN-bodied 15m candle that makes a NEW HIGH
            (since the 1H trough). Mirror logic — induces longs.

        Stricter than the previous green/red-count approximation, which
        accepted any directional candle including non-trap ones. The
        course is specific that each trap MUST sweep stops via a new
        directional extreme.

        Returns False only when we have enough 15m data to decide AND the
        trap count < 3. If 15m data is insufficient, return True
        (fail-open — don't block on missing data; lesson 13 treats the
        15m drop-down as an inspection tool, not a disqualifier when
        the tool is missing).
        """
        try:
            if candles_15m is None or candles_15m.empty or len(candles_15m) < 4:
                return True  # fail-open on missing 15m data

            # Resolve peak1/peak2 absolute indices on the 1H chart
            peak1_abs = lookback_start + best_formation.peak1_idx
            peak2_abs = lookback_start + best_formation.peak2_idx
            if peak1_abs >= len(candles_1h) or peak2_abs >= len(candles_1h):
                return True

            trough_abs = lookback_start + best_formation.trough_idx
            trough_abs = max(peak1_abs, min(trough_abs, peak2_abs))

            # Get the timestamp window (inside right-side = trough → peak2)
            try:
                ts_start = candles_1h.index[trough_abs]
                ts_end = candles_1h.index[peak2_abs]
            except Exception:
                return True

            # Select 15m candles in this window
            try:
                m15_window = candles_15m.loc[ts_start:ts_end]
            except Exception:
                return True

            if m15_window.empty or len(m15_window) < 3:
                # Not enough 15m candles in the inside-right-side window
                return True  # fail-open

            opens = m15_window["open"].values
            highs = m15_window["high"].values
            lows = m15_window["low"].values
            closes = m15_window["close"].values

            trap_count = 0
            formation_type = best_formation.type.upper()

            if formation_type == "W":
                # W = bullish reversal. The inside right side is the second
                # DOWN leg (trough → 2nd low). A trap candle here must:
                #   (a) close red (close < open) — pushing down
                #   (b) make a new low since trough_ts — sweeping stops
                running_low = float("inf")
                for i in range(len(m15_window)):
                    is_red = closes[i] < opens[i]
                    is_new_low = lows[i] < running_low
                    if is_red and is_new_low:
                        trap_count += 1
                    if lows[i] < running_low:
                        running_low = lows[i]
            else:  # "M"
                # M = bearish reversal. Inside right side = 2nd UP leg.
                # Trap candle must:
                #   (a) close green (close > open) — pushing up
                #   (b) make a new high since trough_ts — sweeping stops
                running_high = 0.0
                for i in range(len(m15_window)):
                    is_green = closes[i] > opens[i]
                    is_new_high = highs[i] > running_high
                    if is_green and is_new_high:
                        trap_count += 1
                    if highs[i] > running_high:
                        running_high = highs[i]

            return trap_count >= 3
        except Exception:
            return True  # fail-open on unexpected errors

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

    async def _check_correlation_signal(self) -> "CorrelationSignal | None":
        """D9: Check for DXY/NASDAQ correlation pre-positioning signal (Lesson 19).

        Course teaches: DXY moves before BTC (inverse). If DXY diverges from BTC
        (DXY moved but BTC hasn't reacted yet) → pre-position before BTC catches up.
        S&P/NASDAQ confirmation adds confidence.

        When the CorrelationProvider is still stubbed (available=False), logs and
        returns None so callers leave `correlation_confirmed` as None (neutral score).

        Returns:
            CorrelationSignal if the provider is available and a divergence exists,
            else None.
        """
        try:
            provider = self.data_feeds.correlation
            if hasattr(provider, "fetch_correlation_signal"):
                signal = await provider.fetch_correlation_signal()
                if signal.confidence <= 0:
                    # Provider available but no active signal
                    return None
                return signal
            else:
                # Old-style provider without the new method — skip
                logger.debug("mm_correlation_provider_no_signal_method")
                return None
        except Exception as e:
            logger.debug("mm_correlation_signal_failed", error=str(e))
            return None

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

    def _persist_lifecycle_flags(self, pos: MMPosition) -> None:
        """Fire-and-forget persistence of MMPosition lifecycle flags to the
        trade row. Called immediately after any in-memory flag mutation so a
        restart between cycles can't lose the progression state.

        Per the course (lessons 47/48/49/C6), each flag represents a
        one-shot gate that must not re-fire. Losing them on restart would
        cause SL to be re-moved to breakeven, partials to be retaken, etc.
        """
        if not pos.trade_id:
            return
        updates = {
            "stop_loss": pos.stop_loss,
            "mm_entry_type": pos.entry_type,
            "mm_peak2_wick_price": pos.peak2_wick_price,
            "mm_svc_high": pos.svc_high,
            "mm_svc_low": pos.svc_low,
            "mm_sl_moved_to_breakeven": pos.sl_moved_to_breakeven,
            "mm_sl_moved_under_50ema": pos.sl_moved_under_50ema,
            "mm_took_200ema_partial": pos.took_200ema_partial,
        }

        async def _do() -> None:
            try:
                await self.repo.update_trade(pos.trade_id, updates)
            except Exception as e:
                logger.debug("mm_persist_lifecycle_failed",
                             symbol=pos.symbol, error=str(e))

        try:
            asyncio.create_task(_do())
        except Exception:
            pass  # No running loop — ignore (rare; caller is always async).

    async def _take_partial(self, pos: MMPosition, level: int, current_price: float) -> None:
        """Take partial profit at level completion.

        Also writes a row to the `partial_exits` table so each TP tier hit is
        auditable (was previously missing — current_tier advanced on the trade
        row but no history was kept, so the dashboard could not surface which
        levels fired).
        """
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
                # Compute tier-level PnL so the partial_exits audit row is meaningful
                if pos.direction == "long":
                    tier_pnl_usd = (current_price - pos.entry_price) * close_qty
                else:
                    tier_pnl_usd = (pos.entry_price - current_price) * close_qty
                entry_notional = pos.entry_price * close_qty
                tier_pnl_pct = (tier_pnl_usd / entry_notional * 100) if entry_notional > 0 else 0.0
                tier_fees = float(getattr(result, "fee", 0.0) or 0.0)

                pos.partial_closed_pct = target_close_pct
                pos.quantity -= close_qty
                # Persist trade row (current_tier + remaining_quantity)
                try:
                    await self.repo.update_trade(pos.trade_id, {
                        "remaining_quantity": pos.quantity,
                        "current_tier": level,
                    })
                except Exception as e:
                    logger.debug("mm_partial_db_update_failed", error=str(e))
                # Persist audit row to partial_exits so dashboards can show
                # which tiers fired on each trade.
                try:
                    await self.repo.log_partial_exit(
                        trade_id=pos.trade_id,
                        tier=level,
                        exit_price=float(current_price),
                        exit_quantity=float(close_qty),
                        exit_order_id=str(getattr(result, "order_id", "") or ""),
                        exit_reason=f"tp_l{level}",
                        pnl_usd=round(tier_pnl_usd, 4),
                        pnl_percent=round(tier_pnl_pct, 4),
                        fees_usd=round(tier_fees, 4),
                        remaining_quantity=float(pos.quantity),
                        new_stop_loss=float(pos.stop_loss),
                    )
                except Exception as e:
                    logger.debug("mm_partial_exit_audit_failed", error=str(e))
                logger.info(
                    "mm_partial_profit",
                    symbol=pos.symbol,
                    level=level,
                    closed_pct=round(target_close_pct * 100),
                    qty=round(close_qty, 6),
                    price=current_price,
                    tier_pnl_usd=round(tier_pnl_usd, 2),
                    tier_pnl_pct=round(tier_pnl_pct, 2),
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
