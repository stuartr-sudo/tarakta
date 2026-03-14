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

                signal.agent_context = ctx

                logger.info(
                    "agent_context_enriched",
                    symbol=signal.symbol,
                    obs=len(ctx.get("order_blocks", [])),
                    fvgs=len(ctx.get("fair_value_gaps", [])),
                    liq_pools=ctx.get("liquidity", {}).get("active_pool_count", 0),
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
                for attr in ("_scan_candles_1h", "_scan_ms_results", "_scan_vol_profile", "_scan_pullback_result"):
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
