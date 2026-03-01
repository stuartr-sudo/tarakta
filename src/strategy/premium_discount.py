from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PremiumDiscountResult:
    """Premium/Discount zone analysis for a single timeframe.

    In ICT/SMC methodology, the range between a significant swing high
    and swing low is divided into premium (above 50% equilibrium) and
    discount (below 50% equilibrium) zones.

    - Smart money buys in discount and sells in premium.
    - The equilibrium (50%) level often acts as support/resistance.
    - Entries in the optimal zone (OTE: 62-79% retracement) are ideal.
    """

    range_high: float
    range_low: float
    equilibrium: float  # 50% of the range
    current_zone: str  # "premium", "discount", "equilibrium"
    zone_depth: float  # 0-1 how deep into the zone (0 = equilibrium, 1 = extreme)
    ote_zone: tuple[float, float] | None  # Optimal Trade Entry (62-79% fib)
    price_in_ote: bool
    fib_levels: dict[str, float]  # key fib levels


@dataclass
class PremiumDiscountScore:
    """Aggregated premium/discount scoring across timeframes."""

    results: dict[str, PremiumDiscountResult]  # keyed by timeframe
    score: float  # 0-10 points for confluence
    reasons: list[str]


class PremiumDiscountAnalyzer:
    """Analyzes where price sits within the current dealing range.

    Key concepts (ICT methodology):
    - A dealing range is defined by the most recent significant swing high
      and swing low on the timeframe.
    - The 50% level (equilibrium / CE) divides premium from discount.
    - For LONGS: We want price in DISCOUNT zone (below 50%). Best at OTE
      (61.8%-79% retracement from the high).
    - For SHORTS: We want price in PREMIUM zone (above 50%). Best at OTE
      (61.8%-79% retracement from the low).
    - The deeper into the correct zone, the higher the score.
    """

    # Fibonacci levels for OTE and key zones
    FIB_LEVELS = {
        "0.0": 0.0,    # swing low
        "0.236": 0.236,
        "0.382": 0.382,
        "0.5": 0.5,      # equilibrium
        "0.618": 0.618,  # OTE start
        "0.705": 0.705,  # OTE sweet spot
        "0.786": 0.786,  # OTE end
        "1.0": 1.0,      # swing high
    }

    def analyze(self, ohlc: pd.DataFrame, swing_hl: object = None) -> PremiumDiscountResult:
        """Analyze premium/discount for a single timeframe.

        Args:
            ohlc: OHLCV DataFrame
            swing_hl: swing_highs_lows result from smc library (optional)
        """
        if ohlc.empty or len(ohlc) < 20:
            return self._empty_result()

        high = ohlc["high"].astype(float)
        low = ohlc["low"].astype(float)
        close = ohlc["close"].astype(float)
        current_price = float(close.iloc[-1])

        # Determine the dealing range
        range_high, range_low = self._find_dealing_range(high, low, swing_hl)

        if range_high <= range_low or range_high == 0:
            return self._empty_result()

        range_size = range_high - range_low
        equilibrium = range_low + range_size * 0.5

        # Calculate fib levels (measured from low to high)
        fib_levels = {}
        for name, ratio in self.FIB_LEVELS.items():
            fib_levels[name] = range_low + range_size * ratio

        # Determine current zone
        if abs(current_price - equilibrium) / range_size < 0.05:
            zone = "equilibrium"
            zone_depth = 0.0
        elif current_price > equilibrium:
            zone = "premium"
            zone_depth = min(1.0, (current_price - equilibrium) / (range_high - equilibrium))
        else:
            zone = "discount"
            zone_depth = min(1.0, (equilibrium - current_price) / (equilibrium - range_low))

        # OTE zone (61.8% to 78.6% retracement)
        # For bullish OTE: price retraces DOWN into 61.8-78.6% of range from HIGH
        # = range_high - 0.618 * range_size  to  range_high - 0.786 * range_size
        # = range_low + (1-0.618)*range_size to range_low + (1-0.786)*range_size
        # = fib 0.214 to 0.382 (discount side)
        bullish_ote_top = range_high - 0.618 * range_size  # = fib 0.382
        bullish_ote_bottom = range_high - 0.786 * range_size  # = fib 0.214

        # For bearish OTE: price retraces UP into 61.8-78.6% of range from LOW
        bearish_ote_bottom = range_low + 0.618 * range_size  # = fib 0.618
        bearish_ote_top = range_low + 0.786 * range_size  # = fib 0.786

        # Check if price is in either OTE
        price_in_ote = (
            (bullish_ote_bottom <= current_price <= bullish_ote_top)
            or (bearish_ote_bottom <= current_price <= bearish_ote_top)
        )

        # Provide the relevant OTE based on zone
        if zone == "discount":
            ote_zone = (bullish_ote_bottom, bullish_ote_top)
        elif zone == "premium":
            ote_zone = (bearish_ote_bottom, bearish_ote_top)
        else:
            ote_zone = None

        return PremiumDiscountResult(
            range_high=round(range_high, 8),
            range_low=round(range_low, 8),
            equilibrium=round(equilibrium, 8),
            current_zone=zone,
            zone_depth=round(zone_depth, 3),
            ote_zone=ote_zone,
            price_in_ote=price_in_ote,
            fib_levels=fib_levels,
        )

    def score(
        self,
        results: dict[str, PremiumDiscountResult],
        direction: str | None,
    ) -> PremiumDiscountScore:
        """Score premium/discount positioning across timeframes.

        Returns up to 10 points for the confluence engine:
        - Correct zone on HTF: 0-4 pts
        - Correct zone on entry TF: 0-3 pts
        - Price in OTE: 0-3 pts
        """
        if direction is None:
            return PremiumDiscountScore(results=results, score=0, reasons=[])

        score = 0.0
        reasons: list[str] = []

        # What zone do we want?
        # Bullish → discount zone (buying low)
        # Bearish → premium zone (selling high)
        desired_zone = "discount" if direction == "bullish" else "premium"

        # --- HTF zone alignment (0-4 pts) ---
        for tf in ["4h", "1h"]:
            r = results.get(tf)
            if not r or r.range_high <= r.range_low:
                continue

            if r.current_zone == desired_zone:
                pts = 2 + r.zone_depth * 2  # 2-4 pts based on depth
                score += pts
                reasons.append(f"Price in {desired_zone} zone on {tf} (depth {r.zone_depth:.0%})")
                break  # Count best HTF only
            elif r.current_zone == "equilibrium":
                score += 1
                reasons.append(f"Price at equilibrium on {tf}")
                break

        # --- Entry TF zone (0-3 pts) ---
        for tf in ["15m", "1h"]:
            r = results.get(tf)
            if not r or r.range_high <= r.range_low:
                continue

            if r.current_zone == desired_zone:
                pts = 1.5 + r.zone_depth * 1.5  # 1.5-3 pts
                score += pts
                reasons.append(f"Entry TF {tf} in {desired_zone} (depth {r.zone_depth:.0%})")
                break

        # --- OTE bonus (0-3 pts) ---
        for tf in ["15m", "1h"]:
            r = results.get(tf)
            if not r:
                continue
            if r.price_in_ote and r.current_zone == desired_zone:
                score += 3
                reasons.append(f"Price in OTE on {tf}")
                break

        score = min(score, 10)

        return PremiumDiscountScore(
            results=results,
            score=round(score, 1),
            reasons=reasons,
        )

    def _find_dealing_range(
        self,
        high: pd.Series,
        low: pd.Series,
        swing_hl: object = None,
    ) -> tuple[float, float]:
        """Find the current dealing range (swing high to swing low).

        Uses swing_hl from smc library if available, otherwise falls
        back to a rolling window approach.
        """
        # Try using swing highs/lows from smc library
        if swing_hl is not None and hasattr(swing_hl, "columns"):
            try:
                df = swing_hl
                if "HighLow" in df.columns:
                    # Find recent swing high and swing low
                    recent_highs = []
                    recent_lows = []

                    for i in range(len(df) - 1, max(len(df) - 50, -1), -1):
                        val = df["HighLow"].iloc[i]
                        if val == 1 and len(recent_highs) < 2:  # swing high
                            recent_highs.append(float(high.iloc[i]))
                        elif val == -1 and len(recent_lows) < 2:  # swing low
                            recent_lows.append(float(low.iloc[i]))

                        if recent_highs and recent_lows:
                            break

                    if recent_highs and recent_lows:
                        return max(recent_highs), min(recent_lows)
            except Exception:
                pass  # Fall through to rolling approach

        # Fallback: use rolling window for range detection
        lookback = min(50, len(high))
        recent_high = high.iloc[-lookback:]
        recent_low = low.iloc[-lookback:]

        range_high = float(np.max(recent_high.values))
        range_low = float(np.min(recent_low.values))

        return range_high, range_low

    def _empty_result(self) -> PremiumDiscountResult:
        return PremiumDiscountResult(
            range_high=0.0,
            range_low=0.0,
            equilibrium=0.0,
            current_zone="equilibrium",
            zone_depth=0.0,
            ote_zone=None,
            price_in_ote=False,
            fib_levels={},
        )
