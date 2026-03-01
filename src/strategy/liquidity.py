from __future__ import annotations

import pandas as pd
from smartmoneyconcepts import smc

from src.exchange.models import LiquidityResult, SweepEvent
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LiquidityAnalyzer:
    """Detects equal highs/lows (liquidity pools) and sweep events."""

    def __init__(self, range_percent: float = 0.005) -> None:
        self.range_percent = range_percent

    def analyze(self, ohlc: pd.DataFrame, swing_hl: pd.DataFrame | None) -> LiquidityResult:
        if ohlc.empty or swing_hl is None or len(ohlc) < 30:
            return self._empty_result()

        try:
            liq = smc.liquidity(ohlc, swing_hl, range_percent=self.range_percent)
        except Exception as e:
            logger.warning("liquidity_analysis_error", error=str(e))
            return self._empty_result()

        active_pools = self._get_active_pools(liq)
        sweeps = self._detect_sweeps(liq, ohlc)
        current_price = float(ohlc["close"].iloc[-1])

        nearest_above = self._nearest_pool(active_pools, current_price, "above")
        nearest_below = self._nearest_pool(active_pools, current_price, "below")

        # A sweep is "recent" if it happened in the last 10 candles
        recent_sweeps = [s for s in sweeps if s.candle_idx >= len(ohlc) - 10]

        return LiquidityResult(
            active_pools=active_pools,
            recent_sweeps=recent_sweeps,
            nearest_buy_liquidity=nearest_below,
            nearest_sell_liquidity=nearest_above,
            sweep_detected_recently=len(recent_sweeps) > 0,
        )

    def _get_active_pools(self, liq: pd.DataFrame) -> list[dict]:
        """Get liquidity pools that haven't been swept yet."""
        pools = []
        if "Liquidity" not in liq.columns:
            return pools

        for idx in liq.index:
            if pd.notna(liq.loc[idx, "Liquidity"]) and pd.isna(liq.get("Swept", pd.Series()).get(idx, float("nan"))):
                pools.append(
                    {
                        "level": float(liq.loc[idx, "Level"]),
                        "direction": "buy_side" if liq.loc[idx, "Liquidity"] == 1 else "sell_side",
                        "index": int(idx),
                    }
                )
        return pools

    def _detect_sweeps(self, liq: pd.DataFrame, ohlc: pd.DataFrame) -> list[SweepEvent]:
        """Detect liquidity sweeps (price pierces level then reverses)."""
        sweeps = []
        if "Swept" not in liq.columns:
            return sweeps

        swept_mask = liq["Swept"].notna()
        for idx in liq[swept_mask].index:
            try:
                sweep_candle_idx = int(liq.loc[idx, "Swept"])
                level = float(liq.loc[idx, "Level"])
                direction = int(liq.loc[idx, "Liquidity"])

                # Check if price reversed after the sweep
                if sweep_candle_idx < len(ohlc) - 1:
                    post_sweep = ohlc.iloc[sweep_candle_idx + 1]
                    if direction == 1 and float(post_sweep["close"]) > level:
                        sweeps.append(
                            SweepEvent(level=level, direction="bullish_sweep", candle_idx=sweep_candle_idx)
                        )
                    elif direction == -1 and float(post_sweep["close"]) < level:
                        sweeps.append(
                            SweepEvent(level=level, direction="bearish_sweep", candle_idx=sweep_candle_idx)
                        )
            except (ValueError, KeyError, IndexError):
                continue
        return sweeps

    def _nearest_pool(self, pools: list[dict], price: float, direction: str) -> float | None:
        if direction == "above":
            above = [p["level"] for p in pools if p["level"] > price]
            return min(above) if above else None
        else:
            below = [p["level"] for p in pools if p["level"] < price]
            return max(below) if below else None

    def _empty_result(self) -> LiquidityResult:
        return LiquidityResult(
            active_pools=[],
            recent_sweeps=[],
            nearest_buy_liquidity=None,
            nearest_sell_liquidity=None,
            sweep_detected_recently=False,
        )
