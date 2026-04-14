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
MIN_RR = 3.0
MIN_RR_AGGRESSIVE = 1.0

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
        self.cycle_count = 0
        self._running = True
        self._scanning_active = True  # MM Engine starts active (unlike main bot)

        logger.info("mm_engine_initialized", scan_interval=scan_interval_minutes)

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

        Uses cached prices only — no network calls. The main cycle
        updates _last_prices via fetch_ticker during _manage_position.
        This keeps the API fast and avoids cross-thread event loop issues.
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

            # 4. Scan each pair
            signals: list[MMSignal] = []
            for pair in pairs:
                try:
                    signal = await self._analyze_pair(pair, session, now)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.debug("mm_analyze_pair_error", symbol=pair, error=str(e))

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
        # Fetch candles (1H primary, 4H for EMA trend)
        try:
            candles_1h = await self.candle_manager.get_candles(symbol, "1h", limit=500)
            candles_4h = await self.candle_manager.get_candles(symbol, "4h", limit=250)
        except Exception as e:
            logger.info("mm_reject_candle_fetch", symbol=symbol, error=str(e))
            return None

        if candles_1h is None or candles_1h.empty or len(candles_1h) < 50:
            logger.info("mm_reject_insufficient_candles", symbol=symbol, count=0 if candles_1h is None or (hasattr(candles_1h, 'empty') and candles_1h.empty) else len(candles_1h))
            return None

        current_price = float(candles_1h.iloc[-1]["close"])

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

        # Formation detection (1H)
        formations = self.formation_detector.detect(candles_1h)

        if not formations:
            logger.info("mm_reject_no_formation", symbol=symbol)
            return None  # No formation = no entry

        best_formation = formations[0]

        if best_formation.quality_score < self.min_formation_quality:
            logger.info("mm_reject_low_formation_quality", symbol=symbol, quality=best_formation.quality_score, min_required=self.min_formation_quality)
            return None

        # Level tracking (1H) — count levels AFTER the formation, not all history.
        # peak2_idx is relative to the last DEFAULT_LOOKBACK (40) candles,
        # so translate to the full DataFrame index.
        direction = best_formation.direction or "bullish"
        lookback_start = max(0, len(candles_1h) - 40)  # same window formation detector used
        formation_abs_idx = lookback_start + best_formation.peak2_idx
        candles_post_formation = candles_1h.iloc[formation_abs_idx:]
        level_analysis = self.level_tracker.analyze(candles_post_formation, direction=direction)

        # Don't enter if Level 3+ already reached (expect reversal)
        if level_analysis.current_level >= 3:
            logger.info("mm_reject_level_too_advanced", symbol=symbol, level=level_analysis.current_level, post_formation_candles=len(candles_post_formation))
            return None

        # Weekly cycle
        cycle_state = self.weekly_cycle_tracker.update(candles_1h, now)

        # Don't enter during FMWB (it's the false move)
        if cycle_state.phase == "FMWB":
            logger.info("mm_reject_fmwb_phase", symbol=symbol, phase=cycle_state.phase)
            return None

        # Don't enter during Friday trap phase
        if cycle_state.phase == "FRIDAY_TRAP":
            logger.info("mm_reject_friday_trap", symbol=symbol, phase=cycle_state.phase)
            return None

        # Bug 3: Phase machine enforcement — only enter during valid phases
        if cycle_state.phase not in VALID_ENTRY_PHASES:
            logger.info("mm_reject_wrong_phase", symbol=symbol, phase=cycle_state.phase)
            return None

        # Weekend trap analysis
        weekend = self.weekend_trap_analyzer.analyze(candles_1h, now)

        # Bug 6: Weekly bias gating — FMWB direction determines allowed trade direction
        # FMWB direction is the FALSE move. Real move is opposite.
        if weekend.fmwb.detected:
            # "up" false move → real direction bearish → only shorts
            # "down" false move → real direction bullish → only longs
            real_direction = "short" if weekend.fmwb.direction == "up" else "long"
            if trade_direction != real_direction:
                logger.info("mm_reject_against_weekly_bias", symbol=symbol,
                            trade_dir=trade_direction, fmwb_dir=weekend.fmwb.direction,
                            real_dir=real_direction)
                return None
        else:
            logger.info("mm_warn_no_weekly_bias", symbol=symbol, direction=trade_direction)

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

        # Stop loss placement per MM Method course rules:
        # W-bottom (long): SL below the W's lowest low (min of peak1, peak2)
        # M-top (short): SL above the M's highest high (max of peak1, peak2)
        # No artificial cap — position sizing adjusts size for wider stops.
        if best_formation.type.upper() == "W":
            # W: peak1_price & peak2_price are the two lows. SL below the lowest.
            lowest_low = min(best_formation.peak1_price, best_formation.peak2_price)
            sl_price = lowest_low * 0.998  # 0.2% buffer below invalidation
            trade_direction = "long"
        else:
            # M: peak1_price & peak2_price are the two highs. SL above the highest.
            highest_high = max(best_formation.peak1_price, best_formation.peak2_price)
            sl_price = highest_high * 1.002  # 0.2% buffer above invalidation
            trade_direction = "short"

        # Check SL distance isn't too wide (formation structure too large to trade)
        sl_distance_pct = abs(entry_price - sl_price) / entry_price * 100
        if sl_distance_pct > self.max_sl_pct:
            logger.info("mm_reject_sl_too_wide", symbol=symbol, sl_distance_pct=round(sl_distance_pct, 2), max=self.max_sl_pct)
            return None

        # Targets from target analyzer
        t_l1 = target_analysis.primary_l1.price if target_analysis.primary_l1 else None
        t_l2 = target_analysis.primary_l2.price if target_analysis.primary_l2 else None
        t_l3 = target_analysis.primary_l3.price if target_analysis.primary_l3 else None

        # No valid L1 target = no trade. Per the course, the 50 EMA is the
        # primary L1 target. If it's not in a valid position, the setup doesn't
        # qualify. Don't force entry with a weaker target — an occupied slot
        # with a weak target blocks a valid setup from entering.
        if not t_l1:
            logger.info("mm_reject_no_l1_target", symbol=symbol, direction=trade_direction,
                        ema_50=ema_values.get(50), entry=entry_price,
                        formation=best_formation.type, quality=best_formation.quality_score)
            return None

        # R:R check — try L1 first, fall back to L2 if L1 R:R is too low
        risk = abs(entry_price - sl_price)
        if risk <= 0:
            logger.info("mm_reject_zero_risk", symbol=symbol)
            return None

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
            logger.info("mm_reject_low_rr", symbol=symbol, rr=round(rr, 2), min_required=self.min_rr, entry=entry_price, sl=sl_price, t1=t_l1)
            return None

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
        )

        confluence_result = self.confluence_scorer.score(mm_ctx)

        # Check confluence meets minimum
        if confluence_result.score_pct < self.min_confluence:
            logger.info("mm_reject_low_confluence", symbol=symbol, score=confluence_result.score_pct, min_required=self.min_confluence, formation=best_formation.type)
            return None

        # Bug 1: Retest conditions — course requires 2+ of 4 retest conditions met
        if confluence_result.retest_conditions_met < 2:
            logger.info("mm_reject_low_retest", symbol=symbol, retest_met=confluence_result.retest_conditions_met, confluence=confluence_result.score_pct)
            return None

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
            reason=f"{best_formation.type} formation ({best_formation.variant}) "
                   f"grade={confluence_result.grade} R:R={rr:.1f} "
                   f"phase={cycle_state.phase}",
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

        return signal

    async def _process_entries(self, signals: list[MMSignal]) -> None:
        """Process entry signals — execute the best ones."""
        open_count = len(self.positions)

        # Get balance for margin utilization check
        try:
            balance = await self.exchange.get_balance()
            account_balance = balance.get("USDT", 0) or balance.get("USD", 0)
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
        # Get balance for position sizing
        balance = await self.exchange.get_balance()
        account_balance = balance.get("USDT", 0) or balance.get("USD", 0)

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
        """Tighten stop loss per MM Method level rules.

        Rules from the course:
        - Level 1 complete: SL stays at entry (do NOT move to breakeven yet)
        - Level 2 starts: NOW move SL to breakeven (entry price)
        - Level 2 running: SL under 50 EMA
        - Level 3: SL trails under recent structure
        """
        if new_level <= 1:
            # Don't tighten yet — course says wait for Level 2
            return

        if new_level == 2:
            # Move to breakeven
            if pos.direction == "long" and pos.stop_loss < pos.entry_price:
                pos.stop_loss = pos.entry_price
                logger.info("mm_sl_breakeven", symbol=pos.symbol, sl=pos.stop_loss)
            elif pos.direction == "short" and pos.stop_loss > pos.entry_price:
                pos.stop_loss = pos.entry_price
                logger.info("mm_sl_breakeven", symbol=pos.symbol, sl=pos.stop_loss)

        elif new_level >= 3:
            # Tighten under/above recent structure (last 10 candles)
            recent = candles_1h.tail(10)
            if pos.direction == "long":
                new_sl = float(recent["low"].min()) * 0.998  # Small buffer
                if new_sl > pos.stop_loss:
                    pos.stop_loss = new_sl
                    logger.info("mm_sl_tightened", symbol=pos.symbol, sl=new_sl, level=new_level)
            else:
                new_sl = float(recent["high"].max()) * 1.002
                if new_sl < pos.stop_loss:
                    pos.stop_loss = new_sl
                    logger.info("mm_sl_tightened", symbol=pos.symbol, sl=new_sl, level=new_level)

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
