from __future__ import annotations

import pandas as pd
from smartmoneyconcepts import smc

from src.exchange.models import MarketStructureResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Swing length tuned per timeframe
SWING_LENGTHS = {
    "15m": 5,
    "1h": 10,
    "4h": 10,
    "1d": 15,
}


class MarketStructureAnalyzer:
    """Identifies BOS, CHoCH, swing points, and current trend state."""

    def analyze(self, ohlc: pd.DataFrame, timeframe: str = "1h") -> MarketStructureResult:
        if ohlc.empty or len(ohlc) < 30:
            return self._empty_result()

        swing_length = SWING_LENGTHS.get(timeframe, 10)

        try:
            swing_hl = smc.swing_highs_lows(ohlc, swing_length=swing_length)
            bos_choch = smc.bos_choch(ohlc, swing_hl, close_break=True)
        except Exception as e:
            logger.warning("market_structure_error", timeframe=timeframe, error=str(e))
            return self._empty_result()

        trend = self._determine_trend(bos_choch)
        key_levels = self._extract_key_levels(swing_hl, ohlc)
        last_bos = self._last_signal(bos_choch, "BOS")
        last_choch = self._last_signal(bos_choch, "CHOCH")
        strength = self._calc_structure_strength(bos_choch)

        return MarketStructureResult(
            swing_highs_lows=swing_hl,
            bos_choch=bos_choch,
            trend=trend,
            key_levels=key_levels,
            last_bos_direction=last_bos,
            last_choch_direction=last_choch,
            structure_strength=strength,
        )

    def _determine_trend(self, bos_choch: pd.DataFrame) -> str:
        """Determine trend from recent BOS/CHoCH signals.

        A CHoCH (Change of Character) signals a trend reversal and IS directional.
        A BOS (Break of Structure) signals trend continuation.
        The most recent signal determines the current bias.
        """
        bos_col = bos_choch.get("BOS")
        choch_col = bos_choch.get("CHOCH")

        if bos_col is None:
            return "ranging"

        recent_bos = bos_col.dropna().tail(3)
        recent_choch = choch_col.dropna().tail(2) if choch_col is not None else pd.Series(dtype=float)

        if recent_bos.empty and recent_choch.empty:
            return "ranging"

        # Find the most recent signal of any type
        last_bos_idx = recent_bos.index[-1] if not recent_bos.empty else -1
        last_choch_idx = recent_choch.index[-1] if not recent_choch.empty else -1

        # CHoCH is more recent — trend is shifting in the CHoCH direction
        if last_choch_idx > last_bos_idx:
            choch_dir = float(recent_choch.iloc[-1])
            return "bullish" if choch_dir > 0 else "bearish"

        # BOS is most recent — check BOS direction consistency
        if not recent_bos.empty:
            last_bos_dir = float(recent_bos.iloc[-1])
            # If last 2+ BOS agree, strong trend
            if len(recent_bos) >= 2 and (recent_bos.tail(2) > 0).all():
                return "bullish"
            if len(recent_bos) >= 2 and (recent_bos.tail(2) < 0).all():
                return "bearish"
            # Single BOS — use it as directional bias
            return "bullish" if last_bos_dir > 0 else "bearish"

        return "ranging"

    def _extract_key_levels(self, swing_hl: pd.DataFrame, ohlc: pd.DataFrame) -> dict:
        """Extract most recent swing high and swing low prices."""
        levels: dict = {"swing_high": None, "swing_low": None}

        if swing_hl is None or "HighLow" not in swing_hl.columns:
            return levels

        highs = swing_hl[swing_hl["HighLow"] == 1]["Level"].dropna()
        lows = swing_hl[swing_hl["HighLow"] == -1]["Level"].dropna()

        if not highs.empty:
            levels["swing_high"] = float(highs.iloc[-1])
        if not lows.empty:
            levels["swing_low"] = float(lows.iloc[-1])

        return levels

    def _last_signal(self, bos_choch: pd.DataFrame, col: str) -> int | None:
        if col not in bos_choch.columns:
            return None
        series = bos_choch[col].dropna()
        if series.empty:
            return None
        return int(series.iloc[-1])

    def _calc_structure_strength(self, bos_choch: pd.DataFrame) -> float:
        """0-1 score based on BOS consistency."""
        bos_col = bos_choch.get("BOS")
        if bos_col is None:
            return 0.0
        recent = bos_col.dropna().tail(5)
        if recent.empty:
            return 0.0
        # How consistent are the BOS directions?
        if (recent > 0).all() or (recent < 0).all():
            return 1.0
        dominant = max((recent > 0).sum(), (recent < 0).sum())
        return dominant / len(recent)

    def _empty_result(self) -> MarketStructureResult:
        return MarketStructureResult(
            swing_highs_lows=None,
            bos_choch=None,
            trend="ranging",
            key_levels={"swing_high": None, "swing_low": None},
            last_bos_direction=None,
            last_choch_direction=None,
            structure_strength=0.0,
        )
