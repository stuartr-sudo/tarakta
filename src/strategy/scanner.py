"""Trade Travel Chill scanner — post-sweep displacement pipeline.

Scans all tradeable pairs through a sweep-only pipeline:

1. Fetch candles (1H, 4H, 1D)
2. Market structure on all TFs (for HTF trend + swing levels)
3. Session analysis (Asian, London, NY ranges + post-KZ timing)
4. Sweep detection on 1H (completed sweep = entry signal)
5. Displacement check on 1H (confirms institutional commitment)
6. Score with PostSweepEngine

Removed from pipeline: CRT, Order Blocks, FVGs, Premium/Discount,
LiquidityAnalyzer, MarketRegimeAnalyzer, BreakoutDetector — these are
the retail signals that market makers hunt.
"""
from __future__ import annotations

import asyncio
import gc

import pandas as pd

from src.config import Settings
from src.data.candles import CandleManager
from src.exchange.models import SignalCandidate
from src.exchange.protocol import FuturesCapable
from src.strategy.confluence import PostSweepEngine
from src.strategy.fair_value_gaps import FairValueGapAnalyzer
from src.strategy.leverage import LeverageAnalyzer
from src.strategy.liquidity import LiquidityAnalyzer
from src.strategy.order_blocks import OrderBlockAnalyzer
from src.strategy.market_structure import MarketStructureAnalyzer
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
from src.strategy.pullback import PullbackAnalyzer
from src.strategy.sessions import SessionAnalyzer
from src.strategy.sweep_detector import SweepDetector
from src.strategy.volume import VolumeAnalyzer
from src.strategy.weekly_cycle import WeeklyCycleAnalyzer
from src.utils.logging import get_logger

logger = get_logger(__name__)

TIMEFRAMES = ["1h", "4h", "1d"]
BATCH_SIZE = 8
BATCH_DELAY = 1.5  # seconds between batches


class AltcoinScanner:
    """Scans all tradeable pairs, runs strategy pipeline, ranks by score."""

    def __init__(self, candle_manager: CandleManager, config: Settings) -> None:
        self.candles = candle_manager
        self.config = config
        self.ms_analyzer = MarketStructureAnalyzer()
        self.vol_analyzer = VolumeAnalyzer()
        self.session_analyzer = SessionAnalyzer()
        self.sweep_detector = SweepDetector()
        self.pullback_analyzer = PullbackAnalyzer(
            min_retracement=config.pullback_min_retracement,
            max_retracement=config.pullback_max_retracement,
        )
        self.confluence = PostSweepEngine(entry_threshold=config.entry_threshold)
        self.leverage_analyzer = LeverageAnalyzer()
        # ICT/SMC context analyzers (for agent prompt enrichment only, not scoring)
        self.ob_analyzer = OrderBlockAnalyzer()
        self.fvg_analyzer = FairValueGapAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()
        # Weekly cycle — Fake Move Monday & Mid-Week Reversal
        self.weekly_cycle_enabled = getattr(config, "weekly_cycle_enabled", True)
        self.weekly_cycle = WeeklyCycleAnalyzer(
            monday_penalty_pts=getattr(config, "monday_manipulation_penalty", 15.0),
            monday_manipulation_hours=getattr(config, "monday_manipulation_hours", 8.0),
            midweek_reversal_bonus_pts=getattr(config, "midweek_reversal_bonus", 10.0),
        )
        # MM Method modules — Market Makers Method analysis pipeline
        self.mm_method_enabled = getattr(config, "mm_method_enabled", True)
        if self.mm_method_enabled:
            self.mm_session_analyzer = MMSessionAnalyzer()
            self.mm_ema_framework = EMAFramework()
            self.mm_formation_detector = FormationDetector(
                session_analyzer=self.mm_session_analyzer,
            )
            self.mm_level_tracker = LevelTracker(ema_framework=self.mm_ema_framework)
            self.mm_weekly_cycle_tracker = WeeklyCycleTracker()
            self.mm_confluence_scorer = MMConfluenceScorer(
                min_rr=getattr(config, "mm_min_rr", 3.0),
                min_score=getattr(config, "mm_min_confluence_score", 40.0),
            )
            self.mm_weekend_trap_analyzer = WeekendTrapAnalyzer()
            self.mm_board_meeting_detector = BoardMeetingDetector()
            self.mm_target_analyzer = TargetAnalyzer()
            self.mm_risk_calculator = MMRiskCalculator()
            self.mm_method_weight = getattr(config, "mm_method_weight", 15.0)
        # Near-misses for hyper-watchlist promotion (populated each scan cycle)
        self.last_near_misses: list[SignalCandidate] = []

    async def scan(
        self, pairs: list[str], exclude: set[str] | None = None,
    ) -> list[SignalCandidate]:
        """Scan all pairs through the post-sweep displacement pipeline."""
        # Exclude symbols currently on the hyper-watchlist
        if exclude:
            pairs = [p for p in pairs if p not in exclude]

        all_signals: list[SignalCandidate] = []
        self.last_near_misses = []  # Reset each scan cycle
        total = len(pairs)

        # ── Diagnostic counters (reset each scan) ──
        diag_sweeps = 0
        diag_displacements = 0
        diag_sweep_and_disp = 0
        diag_near_misses: list[tuple[str, float, str]] = []  # (symbol, score, type)
        diag_errors = 0

        for batch_idx in range(0, total, BATCH_SIZE):
            batch = pairs[batch_idx : batch_idx + BATCH_SIZE]
            batch_num = batch_idx // BATCH_SIZE + 1
            total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

            logger.info(
                "scanning_batch",
                batch=batch_num,
                total_batches=total_batches,
                pairs=len(batch),
            )

            tasks = [self._analyze_pair(pair) for pair in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.warning("pair_analysis_failed", error=str(result))
                    diag_errors += 1
                    continue
                if isinstance(result, SignalCandidate):
                    # ── Diagnostic tracking ──
                    comp = result.components
                    has_sweep = comp.get("sweep_detected", 0) > 0
                    has_disp = comp.get("displacement_confirmed", 0) > 0

                    if has_sweep:
                        diag_sweeps += 1
                    if has_disp:
                        diag_displacements += 1
                    if has_sweep and has_disp:
                        diag_sweep_and_disp += 1

                    # ── Threshold check ──
                    threshold = self.config.entry_threshold
                    if result.score >= threshold:
                        all_signals.append(result)
                    elif result.score > 0:
                        # Near miss — scored but didn't qualify
                        diag_near_misses.append(
                            (result.symbol, round(result.score, 1), "sweep")
                        )
                        # Collect full signal for hyper-watchlist promotion
                        if result.score >= self.config.watchlist_min_score:
                            self.last_near_misses.append(result)

            # Free memory between batches
            gc.collect()

            # Rate limit between batches
            if batch_idx + BATCH_SIZE < total:
                await asyncio.sleep(BATCH_DELAY)

        # Leverage enrichment pass — only for qualifying signals on futures (minimal API calls)
        if all_signals and isinstance(self.candles.exchange, FuturesCapable):
            await self._enrich_with_leverage(all_signals)

        # ICT/SMC context enrichment — attaches OB/FVG/liquidity/structure data for agent
        if all_signals:
            self._enrich_agent_context(all_signals)

        # Sort by score descending (re-sort after leverage bonus)
        all_signals.sort(key=lambda s: s.score, reverse=True)

        # ── Diagnostic summary ──
        diag_near_misses.sort(key=lambda x: x[1], reverse=True)
        top_misses = [
            f"{sym}={score}({typ})" for sym, score, typ in diag_near_misses[:5]
        ]

        logger.info(
            "scan_diagnostics",
            pairs_scanned=total,
            sweeps_found=diag_sweeps,
            displacements_found=diag_displacements,
            sweep_plus_displacement=diag_sweep_and_disp,
            signals_qualified=len(all_signals),
            near_misses=len(diag_near_misses),
            top_near_misses=top_misses if top_misses else "none",
            errors=diag_errors,
        )

        logger.info(
            "scan_complete",
            pairs_scanned=total,
            signals_above_threshold=len(all_signals),
            top_signal=all_signals[0].symbol if all_signals else None,
            top_score=all_signals[0].score if all_signals else 0,
        )

        return all_signals

    async def _analyze_pair(self, symbol: str) -> SignalCandidate:
        """Simplified post-sweep pipeline for a single pair."""
        # 1. Fetch candles for all timeframes
        candles: dict[str, pd.DataFrame] = {}
        for tf in TIMEFRAMES:
            candles[tf] = await self.candles.get_candles(symbol, tf, limit=200)

        # 2. Market structure on all TFs (for HTF trend + swing levels)
        ms_results = {}
        for tf, df in candles.items():
            ms_results[tf] = self.ms_analyzer.analyze(df, timeframe=tf)

        # 3. Session analysis (Asian range + post-KZ timing)
        session_result = self.session_analyzer.analyze(candles["1h"])

        # 4. Extract swing levels from 1H market structure
        swing_high = ms_results["1h"].key_levels.get("swing_high")
        swing_low = ms_results["1h"].key_levels.get("swing_low")

        # 5. Displacement check on 1H (detect first to get direction filter)
        vol_profile = self.vol_analyzer.analyze(candles["1h"])
        displacement_confirmed = vol_profile.displacement_detected
        displacement_direction = vol_profile.displacement_direction

        # 5.5 Volume sustainability check: if volume is declining after the
        # displacement spike, treat it as unreliable (likely a one-off liquidation
        # cascade or single whale, not sustained institutional interest).
        if displacement_confirmed and vol_profile.volume_trend == "decreasing":
            displacement_confirmed = False
            displacement_direction = None

        # 6. Sweep detection on 1H (prefer sweeps matching displacement direction
        #    to avoid pullback candles being misread as new conflicting sweeps)
        sweep_result = self.sweep_detector.detect(
            candles_1h=candles["1h"],
            asian_high=session_result.asian_high,
            asian_low=session_result.asian_low,
            swing_high=swing_high,
            swing_low=swing_low,
            lookback=8,
            prefer_direction=displacement_direction,
            london_high=session_result.london_high,
            london_low=session_result.london_low,
            ny_high=session_result.ny_high,
            ny_low=session_result.ny_low,
        )

        # 6.5 Pullback detection (requires displacement)
        pullback_result = None
        if displacement_confirmed and vol_profile.displacement_candle_idx is not None:
            pullback_result = self.pullback_analyzer.analyze(
                candles_1h=candles["1h"],
                displacement_candle_idx=vol_profile.displacement_candle_idx,
                direction=displacement_direction,
            )

        # Get current price
        current_price = float(candles["1h"]["close"].iloc[-1]) if not candles["1h"].empty else 0

        # Compute 14-period ATR on 1H candles for SL floor
        atr_1h = 0.0
        df_1h = candles.get("1h")
        if df_1h is not None and len(df_1h) >= 15:
            _high = df_1h["high"].astype(float)
            _low = df_1h["low"].astype(float)
            _close = df_1h["close"].astype(float)
            _prev = _close.shift(1)
            _tr = pd.concat(
                [_high - _low, (_high - _prev).abs(), (_low - _prev).abs()],
                axis=1,
            ).max(axis=1)
            _atr_series = _tr.rolling(14).mean()
            if pd.notna(_atr_series.iloc[-1]):
                atr_1h = float(_atr_series.iloc[-1])

        # Resolve HTF direction
        htf_direction = self._resolve_htf_direction(ms_results)

        # 7. Score with PostSweepEngine (sweep path)
        signal = self.confluence.score_signal(
            symbol=symbol,
            current_price=current_price,
            sweep_result=sweep_result,
            displacement_confirmed=displacement_confirmed,
            displacement_direction=displacement_direction,
            htf_direction=htf_direction,
            in_post_kill_zone=session_result.in_post_kill_zone,
            ms_results=ms_results,
            pullback_result=pullback_result,
        )

        # ── Weekly Cycle: Fake Move Monday & Mid-Week Reversal ──
        if self.weekly_cycle_enabled and signal.score > 0:
            weekly_result = self.weekly_cycle.analyze(
                candles_1d=candles.get("1d"),
                signal_direction=signal.direction,
            )
            if weekly_result.score_adjustment != 0:
                signal.score = max(0, signal.score + weekly_result.score_adjustment)
                signal.reasons.extend(weekly_result.reasons)
                signal.components["weekly_cycle"] = weekly_result.score_adjustment
                logger.info(
                    "weekly_cycle_applied",
                    symbol=symbol,
                    day=weekly_result.day_name,
                    adjustment=weekly_result.score_adjustment,
                    new_score=signal.score,
                    monday_manipulation=weekly_result.in_monday_manipulation,
                    midweek_reversal=weekly_result.signal_aligns_with_reversal,
                )

        # ── MM Method: sessions, EMA, formations, levels, weekly cycle ──
        if self.mm_method_enabled and signal.score > 0:
            try:
                mm_data = self._run_mm_analysis(symbol, candles, signal.direction)
                signal._scan_mm_data = mm_data  # Stashed for agent context enrichment

                # Apply MM bonus scoring
                mm_bonus = self._score_mm_method(mm_data)
                if mm_bonus > 0:
                    signal.score += mm_bonus
                    signal.components["mm_method"] = mm_bonus
                    signal.reasons.append(
                        f"MM Method confluence: grade {mm_data.get('confluence_grade', '?')} "
                        f"({mm_data.get('confluence_score_pct', 0):.0f}%) +{mm_bonus:.0f}pts"
                    )
                    logger.info(
                        "mm_method_applied",
                        symbol=symbol,
                        mm_phase=mm_data.get("cycle_phase"),
                        mm_grade=mm_data.get("confluence_grade"),
                        mm_score_pct=mm_data.get("confluence_score_pct"),
                        mm_level=mm_data.get("current_level"),
                        mm_formation=mm_data.get("formation_type"),
                        bonus=mm_bonus,
                        new_score=signal.score,
                    )
            except Exception as e:
                logger.warning("mm_method_failed", symbol=symbol, error=str(e))

        # Attach sweep data and ATR for order execution
        signal.sweep_result = sweep_result
        signal.atr_1h = atr_1h
        signal.session_result = session_result
        signal.htf_direction_cache = htf_direction  # Preserved for hyper-watchlist

        # Stash intermediate data for agent context enrichment (cleaned up after use)
        signal._scan_candles_1h = candles["1h"]
        signal._scan_ms_results = ms_results
        signal._scan_vol_profile = vol_profile
        signal._scan_pullback_result = pullback_result

        # Log qualifying signals
        if signal.score >= self.config.entry_threshold:
            logger.info(
                "signal_detected",
                symbol=symbol,
                score=signal.score,
                direction=signal.direction,
                signal_type="sweep",
                htf_continuation=sweep_result.htf_continuation if sweep_result.sweep_detected else False,
                reasons=signal.reasons,
            )

        return signal

    async def _enrich_with_leverage(self, signals: list[SignalCandidate]) -> None:
        """Fetch leverage data and apply bonus scoring for qualifying signals.

        Only called for signals that already passed sweep + displacement + pullback
        (typically 1-5 per scan). This keeps API calls to 3 per signal.
        """
        exchange = self.candles.exchange
        for signal in signals:
            try:
                oi_data = await exchange.fetch_open_interest(signal.symbol)
                fr_data = await exchange.fetch_funding_rate(signal.symbol)
                ls_data = await exchange.fetch_long_short_ratio(signal.symbol)

                sweep_direction = None
                if signal.sweep_result and signal.sweep_result.sweep_detected:
                    sweep_direction = signal.sweep_result.sweep_direction

                session = signal.session_result

                profile = self.leverage_analyzer.analyze(
                    current_price=signal.entry_price,
                    open_interest_usd=oi_data["open_interest_usd"],
                    funding_rate=fr_data["funding_rate"],
                    long_short_ratio=ls_data["long_short_ratio"],
                    sweep_direction=sweep_direction,
                    in_kill_zone=session.in_kill_zone if session else False,
                    in_post_kill_zone=session.in_post_kill_zone if session else False,
                )

                signal.leverage_profile = profile
                # Cache OI at sweep time for footprint comparison
                signal.sweep_oi_usd = oi_data.get("open_interest_usd", 0.0)

                # Apply leverage bonus scoring
                bonus = self._score_leverage(profile)
                if bonus > 0:
                    signal.score += bonus
                    signal.components["leverage_aligned"] = bonus
                    signal.reasons.append(
                        f"Leverage aligned: {profile.crowded_side}s crowded "
                        f"(funding={profile.funding_rate:.4%}, "
                        f"intensity={profile.crowding_intensity:.0%}) +{bonus:.0f}pts"
                    )

                logger.info(
                    "leverage_enriched",
                    symbol=signal.symbol,
                    funding=f"{fr_data['funding_rate']:.6f}",
                    oi_usd=f"{oi_data['open_interest_usd']:,.0f}",
                    ls_ratio=ls_data["long_short_ratio"],
                    crowded=profile.crowded_side,
                    intensity=f"{profile.crowding_intensity:.2f}",
                    sweep_aligns=profile.sweep_aligns_with_crowding,
                    bonus=bonus,
                )

            except Exception as e:
                logger.warning("leverage_enrichment_failed", symbol=signal.symbol, error=str(e))

    @staticmethod
    def _score_leverage(profile) -> float:
        """Score leverage alignment (0-10 bonus points).

        - Sweep aligns with crowded side: +5 base
        - High crowding intensity (>0.5): +2
        - Extreme crowding intensity (>0.8): +3 instead
        - Judas swing probability > 0.6: +2
        Max: 10 points.
        """
        if not profile.sweep_aligns_with_crowding:
            return 0.0

        bonus = 5.0  # Base: sweep grabbed the crowded side's liquidity

        if profile.crowding_intensity > 0.8:
            bonus += 3.0
        elif profile.crowding_intensity > 0.5:
            bonus += 2.0

        if profile.judas_swing_probability > 0.6:
            bonus += 2.0

        return min(bonus, 10.0)

    def _run_mm_analysis(
        self, symbol: str, candles: dict[str, pd.DataFrame], signal_direction: str | None,
    ) -> dict:
        """Run all MM Method modules and return a consolidated results dict.

        Called per-pair during _analyze_pair when mm_method_enabled is True.
        Uses 1H candles for formation/level detection and 4H for EMA trend.

        Args:
            symbol: Trading pair symbol.
            candles: Dict of timeframe -> DataFrame (must include "1h").
            signal_direction: Signal direction from sweep scoring ("bullish"/"bearish").

        Returns:
            Dict with all MM results keyed for agent context and scoring.
        """
        from datetime import datetime, timezone

        df_1h = candles.get("1h")
        df_4h = candles.get("4h")
        mm: dict = {}

        # 1. Session timing
        session_info = self.mm_session_analyzer.get_current_session()
        mm["session_name"] = session_info.session_name
        mm["session_is_gap"] = session_info.is_gap
        mm["session_minutes_remaining"] = session_info.minutes_remaining
        mm["is_weekend"] = session_info.is_weekend

        # 2. EMA framework (use 4H for macro trend, 1H for entry timing)
        if df_4h is not None and not df_4h.empty:
            ema_state_4h = self.mm_ema_framework.calculate(df_4h)
            trend_state_4h = self.mm_ema_framework.get_trend_state(df_4h)
            ema_break_50 = self.mm_ema_framework.detect_ema_break(df_4h, ema_period=50)
            mm["ema_alignment_4h"] = ema_state_4h.alignment
            mm["ema_fan_out_4h"] = round(ema_state_4h.fan_out_score, 3)
            mm["ema_price_dist_50"] = round(ema_state_4h.price_distance_from_50, 2)
            mm["ema_price_dist_200"] = round(ema_state_4h.price_distance_from_200, 2)
            mm["trend_direction_4h"] = trend_state_4h.direction
            mm["trend_strength_4h"] = round(trend_state_4h.strength, 3)
            mm["trend_is_accelerating"] = trend_state_4h.is_accelerating
            mm["trend_is_flattening"] = trend_state_4h.is_flattening
            mm["ema50_broke"] = ema_break_50.broke_ema
            mm["ema50_break_volume_confirmed"] = ema_break_50.volume_confirmed
            mm["ema50_break_direction"] = ema_break_50.direction
        else:
            mm["ema_alignment_4h"] = "mixed"
            mm["trend_direction_4h"] = "sideways"
            mm["trend_strength_4h"] = 0.0

        # 3. Formation detection (1H)
        if df_1h is not None and not df_1h.empty:
            formations = self.mm_formation_detector.detect(
                df_1h, direction_bias=signal_direction,
            )
            if formations:
                best = formations[0]  # Already sorted by quality score
                mm["formation_type"] = best.type
                mm["formation_variant"] = best.variant
                mm["formation_quality"] = round(best.quality_score, 3)
                mm["formation_direction"] = best.direction
                mm["formation_at_key_level"] = best.at_key_level
                mm["formation_confirmed"] = best.confirmed
                mm["formation_session_peak1"] = best.session_peak1
                mm["formation_session_peak2"] = best.session_peak2
            else:
                mm["formation_type"] = None

        # 4. Level tracking (1H)
        if df_1h is not None and not df_1h.empty:
            level_analysis = self.mm_level_tracker.analyze(
                df_1h, direction=signal_direction,
            )
            mm["current_level"] = level_analysis.current_level
            mm["level_direction"] = level_analysis.direction
            mm["is_extended"] = level_analysis.is_extended
            mm["volume_degrading"] = level_analysis.volume_degrading
            mm["num_board_meetings"] = len(level_analysis.board_meetings)
            if level_analysis.svc and level_analysis.svc.detected:
                mm["svc_detected"] = True
                mm["svc_confirmed"] = not level_analysis.svc.price_returned_to_wick
                mm["svc_volume_ratio"] = round(level_analysis.svc.volume_ratio, 2)
            else:
                mm["svc_detected"] = False
                mm["svc_confirmed"] = False
        else:
            mm["current_level"] = 0

        # 5. Weekly cycle state (1H)
        if df_1h is not None and not df_1h.empty:
            now_utc = datetime.now(timezone.utc)
            cycle_state = self.mm_weekly_cycle_tracker.update(df_1h, now_utc)
            mm["cycle_phase"] = cycle_state.phase
            mm["cycle_direction"] = cycle_state.direction
            mm["cycle_how"] = cycle_state.how
            mm["cycle_low"] = cycle_state.low
            mm["cycle_hod"] = cycle_state.hod
            mm["cycle_lod"] = cycle_state.lod
            mm["cycle_fmwb_detected"] = cycle_state.fmwb_detected
            mm["cycle_fmwb_direction"] = cycle_state.fmwb_direction
            mm["cycle_midweek_reversal_expected"] = cycle_state.midweek_reversal_expected
            mm["cycle_take_profit_signal"] = cycle_state.take_profit_signal
            mm["cycle_confidence"] = round(cycle_state.confidence, 3)
        else:
            mm["cycle_phase"] = "unknown"

        # 6. Weekend Trap Box + FMWB detection (1H)
        if df_1h is not None and not df_1h.empty:
            now_utc = datetime.now(timezone.utc)
            weekend_analysis = self.mm_weekend_trap_analyzer.analyze(df_1h, now_utc)
            mm["weekend_trap_detected"] = weekend_analysis.trap_box.detected
            mm["weekend_box_high"] = weekend_analysis.trap_box.box_high
            mm["weekend_box_low"] = weekend_analysis.trap_box.box_low
            mm["weekend_box_range_pct"] = weekend_analysis.trap_box.box_range_pct
            mm["weekend_trap_direction"] = weekend_analysis.trap_box.primary_trap_direction
            mm["fmwb_detected"] = weekend_analysis.fmwb.detected
            mm["fmwb_direction"] = weekend_analysis.fmwb.direction
            mm["fmwb_real_direction"] = weekend_analysis.fmwb.real_direction
            mm["fmwb_magnitude_pct"] = weekend_analysis.fmwb.magnitude_pct
            mm["weekend_bias"] = weekend_analysis.bias
            mm["weekend_confidence"] = weekend_analysis.confidence

        # 7. Target identification (1H)
        current_price = float(df_1h.iloc[-1]["close"]) if df_1h is not None and not df_1h.empty else 0.0
        if df_1h is not None and not df_1h.empty and current_price > 0:
            # Get EMA values for target calc
            ema_vals = {}
            if df_4h is not None and not df_4h.empty:
                ema_state = self.mm_ema_framework.calculate(df_4h)
                ema_vals = ema_state.values
            target_analysis = self.mm_target_analyzer.analyze(
                ohlc=df_1h,
                direction=signal_direction or "bullish",
                entry_price=current_price,
                stop_loss=current_price * 0.99,  # Placeholder SL for vector scanning
                current_level=mm.get("current_level", 1),
                ema_values=ema_vals,
                how=mm.get("cycle_how"),
                low=mm.get("cycle_low"),
            )
            mm["unrecovered_vectors"] = len(target_analysis.unrecovered_vectors)
            if target_analysis.primary_l1:
                mm["target_l1_price"] = target_analysis.primary_l1.price
                mm["target_l1_source"] = target_analysis.primary_l1.source
            if target_analysis.primary_l2:
                mm["target_l2_price"] = target_analysis.primary_l2.price
                mm["target_l2_source"] = target_analysis.primary_l2.source
            if target_analysis.primary_l3:
                mm["target_l3_price"] = target_analysis.primary_l3.price
                mm["target_l3_source"] = target_analysis.primary_l3.source
            mm["rr_to_l1"] = target_analysis.risk_reward_l1
            has_unrecovered = len(target_analysis.unrecovered_vectors) > 0
        else:
            has_unrecovered = False

        # 8. MM Confluence scoring (assembles MMContext from all the above)
        # Check HOW/LOW proximity for confluence
        at_how_low = False
        at_hod_lod = False
        if current_price > 0:
            cycle_how = mm.get("cycle_how", 0)
            cycle_low = mm.get("cycle_low", float("inf"))
            if cycle_how and abs(current_price - cycle_how) / current_price < 0.005:
                at_how_low = True
            if cycle_low and cycle_low < float("inf") and abs(current_price - cycle_low) / current_price < 0.005:
                at_how_low = True
            cycle_hod = mm.get("cycle_hod", 0)
            cycle_lod = mm.get("cycle_lod", float("inf"))
            if cycle_hod and abs(current_price - cycle_hod) / current_price < 0.003:
                at_hod_lod = True
            if cycle_lod and cycle_lod < float("inf") and abs(current_price - cycle_lod) / current_price < 0.003:
                at_hod_lod = True

        mm_ctx = MMContext(
            formation=mm if mm.get("formation_type") else None,
            ema_state={
                "alignment": mm.get("ema_alignment_4h", "mixed"),
                "broke_50": mm.get("ema50_broke", False),
                "volume_confirmed": mm.get("ema50_break_volume_confirmed", False),
                "at_50_ema": abs(mm.get("ema_price_dist_50", 999)) < 0.5,
            } if mm.get("ema_alignment_4h") else None,
            level_state={
                "current_level": mm.get("current_level", 0),
                "svc_detected": mm.get("svc_detected", False),
                "volume_degrading": mm.get("volume_degrading", False),
            },
            cycle_state={
                "phase": mm.get("cycle_phase", "unknown"),
                "direction": mm.get("cycle_direction"),
            },
            at_session_changeover=mm.get("session_is_gap", False),
            at_how_low=at_how_low,
            at_hod_lod=at_hod_lod,
            has_unrecovered_vector=has_unrecovered,
            has_liquidation_cluster=False,  # Requires Hyblock API integration (Phase 3c)
        )
        confluence_result = self.mm_confluence_scorer.score(mm_ctx)
        mm["confluence_score"] = round(confluence_result.total_score, 2)
        mm["confluence_score_pct"] = round(confluence_result.score_pct, 2)
        mm["confluence_grade"] = confluence_result.grade
        mm["confluence_rr"] = round(confluence_result.risk_reward, 2)
        mm["confluence_factors"] = confluence_result.factors

        return mm

    def _score_mm_method(self, mm_data: dict) -> float:
        """Convert MM Method analysis into bonus points for the signal score.

        Scales the MM confluence score percentage into 0-mm_method_weight bonus.
        Only contributes if the MM confluence meets the minimum threshold.

        Args:
            mm_data: Dict from _run_mm_analysis.

        Returns:
            Bonus points (0 to mm_method_weight).
        """
        score_pct = mm_data.get("confluence_score_pct", 0.0)
        min_pct = self.config.mm_min_confluence_score if hasattr(self.config, "mm_min_confluence_score") else 40.0
        max_bonus = self.mm_method_weight

        if score_pct < min_pct:
            return 0.0

        # Scale linearly: min_pct -> 0 bonus, 100% -> full bonus
        scale = (score_pct - min_pct) / (100.0 - min_pct) if min_pct < 100 else 0.0
        bonus = round(scale * max_bonus, 1)
        return min(bonus, max_bonus)

    def _enrich_agent_context(self, signals: list[SignalCandidate]) -> None:
        """Attach ICT/SMC context data to signals for the AI agent prompt.

        Runs AFTER scoring — this data is contextual only, it does NOT affect the
        formula score. Reads stashed intermediate data from _analyze_pair and cleans
        it up after use.
        """
        for signal in signals:
            try:
                candles_1h = getattr(signal, "_scan_candles_1h", None)
                ms_results = getattr(signal, "_scan_ms_results", None)
                vol_profile = getattr(signal, "_scan_vol_profile", None)
                pullback_result = getattr(signal, "_scan_pullback_result", None)

                if candles_1h is None or ms_results is None:
                    continue

                ctx: dict = {}
                current_price = signal.entry_price

                # ── Order Blocks (top 5 nearest by distance) ──
                swing_hl = ms_results["1h"].swing_highs_lows if ms_results.get("1h") else None
                ob_result = self.ob_analyzer.analyze(candles_1h, swing_hl)
                nearest_obs = sorted(
                    ob_result.active_order_blocks,
                    key=lambda ob: abs((ob.top + ob.bottom) / 2 - current_price),
                )[:5]
                ctx["order_blocks"] = [
                    {
                        "direction": ob.direction,
                        "top": ob.top,
                        "bottom": ob.bottom,
                        "strength": round(ob.strength, 3),
                        "distance_pct": round(
                            abs((ob.top + ob.bottom) / 2 - current_price) / current_price * 100, 2
                        ) if current_price > 0 else 0,
                    }
                    for ob in nearest_obs
                ]
                ctx["price_in_ob"] = ob_result.price_in_order_block is not None

                # ── Fair Value Gaps (top 5 nearest) ──
                fvg_result = self.fvg_analyzer.analyze(candles_1h)
                nearest_fvgs = sorted(
                    fvg_result.active_fvgs,
                    key=lambda f: abs(f.midpoint - current_price),
                )[:5]
                ctx["fair_value_gaps"] = [
                    {
                        "direction": fvg.direction,
                        "top": fvg.top,
                        "bottom": fvg.bottom,
                        "midpoint": fvg.midpoint,
                        "distance_pct": round(
                            abs(fvg.midpoint - current_price) / current_price * 100, 2
                        ) if current_price > 0 else 0,
                    }
                    for fvg in nearest_fvgs
                ]
                ctx["price_in_fvg"] = fvg_result.price_in_fvg is not None

                # ── Liquidity Pools ──
                liq_result = self.liquidity_analyzer.analyze(candles_1h, swing_hl)
                ctx["liquidity"] = {
                    "nearest_buy": liq_result.nearest_buy_liquidity,
                    "nearest_sell": liq_result.nearest_sell_liquidity,
                    "buy_distance_pct": round(
                        abs(liq_result.nearest_buy_liquidity - current_price) / current_price * 100, 2
                    ) if liq_result.nearest_buy_liquidity and current_price > 0 else None,
                    "sell_distance_pct": round(
                        abs(liq_result.nearest_sell_liquidity - current_price) / current_price * 100, 2
                    ) if liq_result.nearest_sell_liquidity and current_price > 0 else None,
                    "active_pool_count": len(liq_result.active_pools),
                    "recent_sweeps": len(liq_result.recent_sweeps),
                }

                # ── Market Structure per TF ──
                ms_context = {}
                for tf in ["1h", "4h", "1d"]:
                    ms = ms_results.get(tf)
                    if ms:
                        bos_dir = "bullish" if ms.last_bos_direction == 1 else (
                            "bearish" if ms.last_bos_direction == -1 else "none"
                        )
                        choch_dir = "bullish" if ms.last_choch_direction == 1 else (
                            "bearish" if ms.last_choch_direction == -1 else "none"
                        )
                        ms_context[tf] = {
                            "trend": ms.trend,
                            "strength": round(ms.structure_strength, 2),
                            "last_bos": bos_dir,
                            "last_choch": choch_dir,
                        }
                ctx["market_structure"] = ms_context

                # ── Volume Profile ──
                if vol_profile:
                    ctx["volume"] = {
                        "relative_volume": round(vol_profile.relative_volume, 2),
                        "volume_trend": vol_profile.volume_trend,
                        "displacement_detected": vol_profile.displacement_detected,
                        "displacement_strength": round(vol_profile.displacement_strength, 2),
                        "displacement_direction": vol_profile.displacement_direction,
                    }

                # ── Pullback Metrics ──
                if pullback_result and pullback_result.pullback_detected:
                    ctx["pullback"] = {
                        "retracement_pct": round(pullback_result.retracement_pct * 100, 1),
                        "thrust_extreme": pullback_result.thrust_extreme,
                        "optimal_entry": pullback_result.optimal_entry,
                        "pullback_status": pullback_result.pullback_status,
                        "displacement_open": pullback_result.displacement_open,
                    }

                # ── Leverage Profile (already attached by _enrich_with_leverage) ──
                lp = signal.leverage_profile
                if lp:
                    ctx["leverage"] = {
                        "open_interest_usd": lp.open_interest_usd,
                        "funding_rate": lp.funding_rate,
                        "long_short_ratio": lp.long_short_ratio,
                        "crowded_side": lp.crowded_side,
                        "crowding_intensity": round(lp.crowding_intensity, 2),
                        "funding_bias": lp.funding_bias,
                        "nearest_long_liq": lp.nearest_long_liq,
                        "nearest_short_liq": lp.nearest_short_liq,
                        "judas_swing_probability": round(lp.judas_swing_probability, 2),
                    }

                # ── MM Method (stashed from _analyze_pair) ──
                mm_data = getattr(signal, "_scan_mm_data", None)
                if mm_data:
                    ctx["mm_method"] = {
                        "session": {
                            "name": mm_data.get("session_name"),
                            "is_gap": mm_data.get("session_is_gap"),
                            "minutes_remaining": mm_data.get("session_minutes_remaining"),
                        },
                        "ema": {
                            "alignment_4h": mm_data.get("ema_alignment_4h"),
                            "fan_out_4h": mm_data.get("ema_fan_out_4h"),
                            "price_dist_50": mm_data.get("ema_price_dist_50"),
                            "price_dist_200": mm_data.get("ema_price_dist_200"),
                            "broke_50": mm_data.get("ema50_broke"),
                            "break_volume_confirmed": mm_data.get("ema50_break_volume_confirmed"),
                        },
                        "trend": {
                            "direction_4h": mm_data.get("trend_direction_4h"),
                            "strength_4h": mm_data.get("trend_strength_4h"),
                            "is_accelerating": mm_data.get("trend_is_accelerating"),
                            "is_flattening": mm_data.get("trend_is_flattening"),
                        },
                        "formation": {
                            "type": mm_data.get("formation_type"),
                            "variant": mm_data.get("formation_variant"),
                            "quality": mm_data.get("formation_quality"),
                            "direction": mm_data.get("formation_direction"),
                            "at_key_level": mm_data.get("formation_at_key_level"),
                            "confirmed": mm_data.get("formation_confirmed"),
                        } if mm_data.get("formation_type") else None,
                        "levels": {
                            "current": mm_data.get("current_level"),
                            "direction": mm_data.get("level_direction"),
                            "is_extended": mm_data.get("is_extended"),
                            "volume_degrading": mm_data.get("volume_degrading"),
                            "svc_detected": mm_data.get("svc_detected"),
                            "svc_confirmed": mm_data.get("svc_confirmed"),
                        },
                        "weekly_cycle": {
                            "phase": mm_data.get("cycle_phase"),
                            "direction": mm_data.get("cycle_direction"),
                            "how": mm_data.get("cycle_how"),
                            "low": mm_data.get("cycle_low"),
                            "fmwb_detected": mm_data.get("cycle_fmwb_detected"),
                            "midweek_reversal_expected": mm_data.get("cycle_midweek_reversal_expected"),
                            "take_profit_signal": mm_data.get("cycle_take_profit_signal"),
                        },
                        "weekend_trap": {
                            "detected": mm_data.get("weekend_trap_detected"),
                            "box_high": mm_data.get("weekend_box_high"),
                            "box_low": mm_data.get("weekend_box_low"),
                            "box_range_pct": mm_data.get("weekend_box_range_pct"),
                            "trap_direction": mm_data.get("weekend_trap_direction"),
                            "fmwb_detected": mm_data.get("fmwb_detected"),
                            "fmwb_direction": mm_data.get("fmwb_direction"),
                            "fmwb_real_direction": mm_data.get("fmwb_real_direction"),
                            "bias": mm_data.get("weekend_bias"),
                        } if mm_data.get("weekend_trap_detected") else None,
                        "targets": {
                            "l1_price": mm_data.get("target_l1_price"),
                            "l1_source": mm_data.get("target_l1_source"),
                            "l2_price": mm_data.get("target_l2_price"),
                            "l2_source": mm_data.get("target_l2_source"),
                            "l3_price": mm_data.get("target_l3_price"),
                            "l3_source": mm_data.get("target_l3_source"),
                            "unrecovered_vectors": mm_data.get("unrecovered_vectors", 0),
                            "rr_to_l1": mm_data.get("rr_to_l1"),
                        },
                        "confluence": {
                            "score": mm_data.get("confluence_score"),
                            "score_pct": mm_data.get("confluence_score_pct"),
                            "grade": mm_data.get("confluence_grade"),
                        },
                    }

                signal.agent_context = ctx

                logger.info(
                    "agent_context_enriched",
                    symbol=signal.symbol,
                    obs=len(ctx.get("order_blocks", [])),
                    fvgs=len(ctx.get("fair_value_gaps", [])),
                    liq_pools=ctx.get("liquidity", {}).get("active_pool_count", 0),
                    mm_method="yes" if ctx.get("mm_method") else "no",
                )

            except Exception as e:
                logger.warning("agent_context_enrichment_failed", symbol=signal.symbol, error=str(e))

            finally:
                # Persist 1H thrust data for the entry refiner (survives cleanup)
                _pb = getattr(signal, "_scan_pullback_result", None)
                if _pb is not None:
                    signal.thrust_extreme_1h = _pb.thrust_extreme
                    signal.displacement_open_1h = _pb.displacement_open

                # Clean up stashed data to free memory
                for attr in (
                    "_scan_candles_1h", "_scan_ms_results", "_scan_vol_profile",
                    "_scan_pullback_result", "_scan_mm_data",
                ):
                    if hasattr(signal, attr):
                        delattr(signal, attr)

    def _resolve_htf_direction(self, ms_results: dict) -> str | None:
        """Quick pre-pass to determine HTF directional bias."""
        htf_4h = ms_results.get("4h")
        htf_1d = ms_results.get("1d")

        trend_4h = htf_4h.trend if htf_4h else "ranging"
        trend_1d = htf_1d.trend if htf_1d else "ranging"

        if trend_4h == trend_1d and trend_4h != "ranging":
            return trend_4h
        if trend_4h != "ranging":
            return trend_4h
        if trend_1d != "ranging":
            return trend_1d
        return None
