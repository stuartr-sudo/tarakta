from __future__ import annotations

import pandas as pd
from smartmoneyconcepts import smc

from src.exchange.models import OrderBlock, OrderBlockResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OrderBlockAnalyzer:
    """Identifies and tracks order blocks for potential entry zones."""

    def analyze(self, ohlc: pd.DataFrame, swing_hl: pd.DataFrame | None) -> OrderBlockResult:
        if ohlc.empty or swing_hl is None or len(ohlc) < 30:
            return self._empty_result()

        try:
            ob = smc.ob(ohlc, swing_hl, close_mitigation=False)
        except Exception as e:
            logger.warning("order_block_error", error=str(e))
            return self._empty_result()

        # Extract active (unmitigated) order blocks
        active_obs: list[OrderBlock] = []
        if "OB" not in ob.columns:
            return self._empty_result()

        mitigated_col = ob.get("MitigatedIndex", pd.Series(dtype=float))

        for idx in ob.index:
            if pd.notna(ob.loc[idx, "OB"]):
                # smc library: MitigatedIndex=0 means NOT mitigated,
                # positive integer = candle index where it was mitigated
                mitigated_val = mitigated_col.get(idx, 0) if isinstance(mitigated_col, pd.Series) else 0
                is_mitigated = pd.notna(mitigated_val) and float(mitigated_val) > 0

                if not is_mitigated:
                    active_obs.append(
                        OrderBlock(
                            direction="bullish" if ob.loc[idx, "OB"] == 1 else "bearish",
                            top=float(ob.loc[idx, "Top"]),
                            bottom=float(ob.loc[idx, "Bottom"]),
                            volume=float(ob.loc[idx, "OBVolume"]) if pd.notna(ob.loc[idx, "OBVolume"]) else 0.0,
                            strength=float(ob.loc[idx, "Percentage"]) if pd.notna(ob.loc[idx, "Percentage"]) else 0.0,
                            candle_idx=int(idx),
                        )
                    )

        current_price = float(ohlc["close"].iloc[-1])

        # Check if price is currently in an order block
        price_in_ob = None
        for block in active_obs:
            if block.bottom <= current_price <= block.top:
                price_in_ob = block
                break

        return OrderBlockResult(
            active_order_blocks=active_obs,
            price_in_order_block=price_in_ob,
            nearest_bullish_ob=self._nearest(active_obs, current_price, "bullish"),
            nearest_bearish_ob=self._nearest(active_obs, current_price, "bearish"),
        )

    def _nearest(
        self, obs: list[OrderBlock], price: float, direction: str
    ) -> OrderBlock | None:
        """Find nearest OB of given direction below (bullish) or above (bearish) price."""
        candidates = [ob for ob in obs if ob.direction == direction]

        if direction == "bullish":
            # Bullish OBs below current price (potential support / buy zone)
            below = [ob for ob in candidates if ob.top <= price]
            if below:
                return max(below, key=lambda ob: ob.top)
            # Also consider ones we're currently in
            inside = [ob for ob in candidates if ob.bottom <= price <= ob.top]
            return inside[0] if inside else None
        else:
            # Bearish OBs above current price (potential resistance / sell zone)
            above = [ob for ob in candidates if ob.bottom >= price]
            if above:
                return min(above, key=lambda ob: ob.bottom)
            inside = [ob for ob in candidates if ob.bottom <= price <= ob.top]
            return inside[0] if inside else None

    def _empty_result(self) -> OrderBlockResult:
        return OrderBlockResult(
            active_order_blocks=[],
            price_in_order_block=None,
            nearest_bullish_ob=None,
            nearest_bearish_ob=None,
        )
