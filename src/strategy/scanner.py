from __future__ import annotations

import asyncio
import gc

import pandas as pd

from src.config import Settings
from src.data.candles import CandleManager
from src.exchange.models import SignalCandidate
from src.strategy.confluence import ConfluenceEngine
from src.strategy.fair_value_gaps import FairValueGapAnalyzer
from src.strategy.liquidity import LiquidityAnalyzer
from src.strategy.market_structure import MarketStructureAnalyzer
from src.strategy.order_blocks import OrderBlockAnalyzer
from src.strategy.premium_discount import PremiumDiscountAnalyzer
from src.strategy.volume import VolumeAnalyzer
from src.utils.logging import get_logger

logger = get_logger(__name__)

TIMEFRAMES = ["15m", "1h", "4h", "1d"]
BATCH_SIZE = 8
BATCH_DELAY = 1.5  # seconds between batches


class AltcoinScanner:
    """Scans all tradeable pairs, runs strategy pipeline, ranks by score."""

    def __init__(self, candle_manager: CandleManager, config: Settings) -> None:
        self.candles = candle_manager
        self.config = config
        self.ms_analyzer = MarketStructureAnalyzer()
        self.liq_analyzer = LiquidityAnalyzer()
        self.ob_analyzer = OrderBlockAnalyzer()
        self.fvg_analyzer = FairValueGapAnalyzer()
        self.vol_analyzer = VolumeAnalyzer()
        self.pd_analyzer = PremiumDiscountAnalyzer()
        self.confluence = ConfluenceEngine(entry_threshold=config.entry_threshold)

    async def scan(self, pairs: list[str]) -> list[SignalCandidate]:
        """Scan all pairs through the full strategy pipeline."""
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
        """Full strategy pipeline for a single pair."""
        # Fetch candles for all timeframes
        candles: dict[str, pd.DataFrame] = {}
        for tf in TIMEFRAMES:
            candles[tf] = await self.candles.get_candles(symbol, tf, limit=200)

        # --- Market Structure on all timeframes ---
        ms_results = {}
        for tf, df in candles.items():
            ms_results[tf] = self.ms_analyzer.analyze(df, timeframe=tf)

        # --- Liquidity on all timeframes ---
        liq_results = {}
        for tf, df in candles.items():
            swing_hl = ms_results[tf].swing_highs_lows
            liq_results[tf] = self.liq_analyzer.analyze(df, swing_hl)

        # --- Order Blocks on entry timeframes ---
        ob_results = {}
        for tf in ["15m", "1h"]:
            swing_hl = ms_results[tf].swing_highs_lows
            ob_results[tf] = self.ob_analyzer.analyze(candles[tf], swing_hl)

        # --- Fair Value Gaps on entry timeframes ---
        fvg_results = {}
        for tf in ["15m", "1h"]:
            fvg_results[tf] = self.fvg_analyzer.analyze(candles[tf])

        # --- Determine HTF directional bias (pre-pass for volume/PD scoring) ---
        direction = self._resolve_htf_direction(ms_results)

        # --- Volume / Displacement analysis on key timeframes ---
        volume_profiles = {}
        for tf in ["15m", "1h", "4h"]:
            df = candles.get(tf)
            if df is not None and not df.empty:
                volume_profiles[tf] = self.vol_analyzer.analyze(df)
        volume_result = self.vol_analyzer.score_volume(volume_profiles, direction)

        # --- Premium / Discount zone analysis ---
        pd_results = {}
        for tf in ["15m", "1h", "4h"]:
            df = candles.get(tf)
            if df is not None and not df.empty:
                swing_hl = ms_results.get(tf, None)
                swing_hl_data = swing_hl.swing_highs_lows if swing_hl else None
                pd_results[tf] = self.pd_analyzer.analyze(df, swing_hl_data)
        pd_score = self.pd_analyzer.score(pd_results, direction)

        # Get current price from most recent 15m close
        current_price = float(candles["15m"]["close"].iloc[-1]) if not candles["15m"].empty else 0

        # --- Score the full signal ---
        signal = self.confluence.score_signal(
            symbol=symbol,
            current_price=current_price,
            ms_results=ms_results,
            liq_results=liq_results,
            ob_results=ob_results,
            fvg_results=fvg_results,
            volume_result=volume_result,
            pd_result=pd_score,
        )

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
        """Quick pre-pass to determine HTF directional bias.

        Uses the same logic as ConfluenceEngine._score_htf_trend but only
        returns the direction (needed for volume and P/D scoring before
        the full confluence pass).
        """
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
