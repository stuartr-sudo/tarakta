from __future__ import annotations

import pandas as pd
from smartmoneyconcepts import smc

from src.exchange.models import FVGResult, FairValueGap
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FairValueGapAnalyzer:
    """Detects FVGs (imbalances) as potential entry/target zones."""

    def analyze(self, ohlc: pd.DataFrame) -> FVGResult:
        if ohlc.empty or len(ohlc) < 10:
            return self._empty_result()

        try:
            fvg = smc.fvg(ohlc, join_consecutive=True)
        except Exception as e:
            logger.warning("fvg_analysis_error", error=str(e))
            return self._empty_result()

        active_fvgs: list[FairValueGap] = []
        if "FVG" not in fvg.columns:
            return self._empty_result()

        mitigated_col = fvg.get("MitigatedIndex", pd.Series(dtype=float))

        for idx in fvg.index:
            if pd.notna(fvg.loc[idx, "FVG"]):
                # smc library: MitigatedIndex=0 means NOT mitigated,
                # positive integer = candle index where it was mitigated
                mitigated_val = mitigated_col.get(idx, 0) if isinstance(mitigated_col, pd.Series) else 0
                is_mitigated = pd.notna(mitigated_val) and float(mitigated_val) > 0

                if not is_mitigated:
                    top = float(fvg.loc[idx, "Top"])
                    bottom = float(fvg.loc[idx, "Bottom"])
                    active_fvgs.append(
                        FairValueGap(
                            direction="bullish" if fvg.loc[idx, "FVG"] == 1 else "bearish",
                            top=top,
                            bottom=bottom,
                            candle_idx=int(idx),
                            midpoint=(top + bottom) / 2,
                        )
                    )

        current_price = float(ohlc["close"].iloc[-1])

        # Check if price is in an FVG
        price_in_fvg = None
        for gap in active_fvgs:
            if gap.bottom <= current_price <= gap.top:
                price_in_fvg = gap
                break

        return FVGResult(
            active_fvgs=active_fvgs,
            price_in_fvg=price_in_fvg,
            nearest_bullish_fvg=self._nearest(active_fvgs, current_price, "bullish"),
            nearest_bearish_fvg=self._nearest(active_fvgs, current_price, "bearish"),
        )

    def _nearest(
        self, fvgs: list[FairValueGap], price: float, direction: str
    ) -> FairValueGap | None:
        candidates = [f for f in fvgs if f.direction == direction]

        if direction == "bullish":
            below = [f for f in candidates if f.top <= price]
            if below:
                return max(below, key=lambda f: f.top)
            inside = [f for f in candidates if f.bottom <= price <= f.top]
            return inside[0] if inside else None
        else:
            above = [f for f in candidates if f.bottom >= price]
            if above:
                return min(above, key=lambda f: f.bottom)
            inside = [f for f in candidates if f.bottom <= price <= f.top]
            return inside[0] if inside else None

    def _empty_result(self) -> FVGResult:
        return FVGResult(
            active_fvgs=[],
            price_in_fvg=None,
            nearest_bullish_fvg=None,
            nearest_bearish_fvg=None,
        )
