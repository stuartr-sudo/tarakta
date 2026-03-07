"""Trade Travel Chill scanner — post-sweep displacement + breakout pipeline.

Scans all tradeable pairs through a dual-path pipeline:

Path A — Sweep & Reverse (primary):
1. Fetch candles (1H, 4H, 1D)
2. Market structure on all TFs (for HTF trend + swing levels)
3. Session analysis (Asian, London, NY ranges + post-KZ timing)
4. Sweep detection on 1H (completed sweep = entry signal)
5. Displacement check on 1H (confirms institutional commitment)
6. Score with PostSweepEngine

Path B — Breakout (complementary):
7. If no sweep detected, check for genuine breakouts
8. Price broke AND held beyond a key level with volume
9. Score with PostSweepEngine.score_breakout()

Removed from pipeline: CRT, Order Blocks, FVGs, Premium/Discount,
LiquidityAnalyzer, MarketRegimeAnalyzer — these are the retail signals
that market makers hunt.
"""
from __future__ import annotations

import asyncio
import gc

import pandas as pd

from src.config import Settings
from src.data.candles import CandleManager
from src.exchange.models import SignalCandidate
from src.exchange.protocol import FuturesCapable
from src.strategy.breakout_detector import BreakoutDetector
from src.strategy.confluence import BREAKOUT_THRESHOLD, BREAKOUT_WEIGHTS, PostSweepEngine
from src.strategy.leverage import LeverageAnalyzer
from src.strategy.market_structure import MarketStructureAnalyzer
from src.strategy.pullback import PullbackAnalyzer
from src.strategy.sessions import SessionAnalyzer
from src.strategy.sweep_detector import SweepDetector
from src.strategy.volume import VolumeAnalyzer
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
        self.breakout_detector = BreakoutDetector()
        self.pullback_analyzer = PullbackAnalyzer(
            min_retracement=config.pullback_min_retracement,
            max_retracement=config.pullback_max_retracement,
        )
        self.confluence = PostSweepEngine(entry_threshold=config.entry_threshold)
        self.leverage_analyzer = LeverageAnalyzer()

    async def scan(self, pairs: list[str]) -> list[SignalCandidate]:
        """Scan all pairs through the post-sweep displacement pipeline."""
        all_signals: list[SignalCandidate] = []
        total = len(pairs)

        # ── Diagnostic counters (reset each scan) ──
        diag_sweeps = 0
        diag_displacements = 0
        diag_sweep_and_disp = 0
        diag_breakouts = 0
        diag_breakout_vol = 0
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
                    has_breakout = comp.get("breakout_confirmed", 0) > 0
                    has_bo_vol = comp.get("volume_confirmed", 0) > 0

                    if has_sweep:
                        diag_sweeps += 1
                    if has_disp:
                        diag_displacements += 1
                    if has_sweep and has_disp:
                        diag_sweep_and_disp += 1
                    if has_breakout:
                        diag_breakouts += 1
                    if has_breakout and has_bo_vol:
                        diag_breakout_vol += 1

                    # ── Threshold check ──
                    # Breakout signals use lower threshold (45) vs sweep (60)
                    threshold = (
                        BREAKOUT_THRESHOLD
                        if result.breakout_result is not None
                        else self.config.entry_threshold
                    )
                    if result.score >= threshold:
                        all_signals.append(result)
                    elif result.score > 0:
                        # Near miss — scored but didn't qualify
                        sig_type = "breakout" if has_breakout else "sweep"
                        diag_near_misses.append(
                            (result.symbol, round(result.score, 1), sig_type)
                        )

            # Free memory between batches
            gc.collect()

            # Rate limit between batches
            if batch_idx + BATCH_SIZE < total:
                await asyncio.sleep(BATCH_DELAY)

        # Leverage enrichment pass — only for qualifying signals on futures (minimal API calls)
        if all_signals and isinstance(self.candles.exchange, FuturesCapable):
            await self._enrich_with_leverage(all_signals)

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
            breakouts_found=diag_breakouts,
            breakouts_with_volume=diag_breakout_vol,
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

        # 8. If sweep didn't produce a qualifying signal, try breakout path
        if signal.score < self.config.entry_threshold:
            breakout_result = self.breakout_detector.detect(
                candles_1h=candles["1h"],
                asian_high=session_result.asian_high,
                asian_low=session_result.asian_low,
                london_high=session_result.london_high,
                london_low=session_result.london_low,
                ny_high=session_result.ny_high,
                ny_low=session_result.ny_low,
                swing_high=swing_high,
                swing_low=swing_low,
            )

            if breakout_result.breakout_detected:
                breakout_signal = self.confluence.score_breakout(
                    symbol=symbol,
                    current_price=current_price,
                    breakout_result=breakout_result,
                    htf_direction=htf_direction,
                    in_post_kill_zone=session_result.in_post_kill_zone,
                    ms_results=ms_results,
                )

                # Use breakout signal if it scores above breakout threshold
                if breakout_signal.score >= BREAKOUT_THRESHOLD:
                    breakout_signal.breakout_result = breakout_result
                    breakout_signal.atr_1h = atr_1h
                    breakout_signal.session_result = session_result
                    signal = breakout_signal
                else:
                    # Breakout found but too low — tag sweep signal for diagnostics
                    signal.components["breakout_confirmed"] = BREAKOUT_WEIGHTS["breakout_confirmed"]
                    if breakout_result.volume_confirmed:
                        signal.components["volume_confirmed"] = BREAKOUT_WEIGHTS["volume_confirmed"]

        # Attach sweep data and ATR for order execution
        signal.sweep_result = sweep_result
        signal.atr_1h = atr_1h
        signal.session_result = session_result

        # Log qualifying signals (sweep threshold=60, breakout threshold=45)
        log_threshold = (
            BREAKOUT_THRESHOLD
            if signal.breakout_result is not None
            else self.config.entry_threshold
        )
        if signal.score >= log_threshold:
            logger.info(
                "signal_detected",
                symbol=symbol,
                score=signal.score,
                direction=signal.direction,
                signal_type="breakout" if signal.breakout_result is not None else "sweep",
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
