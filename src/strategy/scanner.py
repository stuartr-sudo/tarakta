"""Trade Travel Chill scanner — simplified post-sweep displacement pipeline.

Scans all tradeable pairs through a streamlined pipeline:
1. Fetch candles (1H, 4H, 1D)
2. Market structure on all TFs (for HTF trend + swing levels)
3. Session analysis (for Asian range + post-KZ timing)
4. Sweep detection on 1H (completed sweep = entry signal)
5. Displacement check on 1H (confirms institutional commitment)
6. Score with PostSweepEngine

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
from src.strategy.confluence import PostSweepEngine
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
        self.pullback_analyzer = PullbackAnalyzer(
            min_retracement=config.pullback_min_retracement,
            max_retracement=config.pullback_max_retracement,
        )
        self.confluence = PostSweepEngine(entry_threshold=config.entry_threshold)

    async def scan(self, pairs: list[str]) -> list[SignalCandidate]:
        """Scan all pairs through the post-sweep displacement pipeline."""
        all_signals: list[SignalCandidate] = []
        total = len(pairs)

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
                    continue
                if isinstance(result, SignalCandidate) and result.score >= self.config.entry_threshold:
                    all_signals.append(result)

            # Free memory between batches
            gc.collect()

            # Rate limit between batches
            if batch_idx + BATCH_SIZE < total:
                await asyncio.sleep(BATCH_DELAY)

        # Sort by score descending
        all_signals.sort(key=lambda s: s.score, reverse=True)

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

        # 7. Score with PostSweepEngine
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

        # Attach sweep data and ATR for order execution
        signal.sweep_result = sweep_result
        signal.atr_1h = atr_1h
        signal.session_result = session_result

        if signal.score >= self.config.entry_threshold:
            logger.info(
                "signal_detected",
                symbol=symbol,
                score=signal.score,
                direction=signal.direction,
                reasons=signal.reasons,
            )

        return signal

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
