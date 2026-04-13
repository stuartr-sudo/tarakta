"""Market Makers Method 5-EMA Framework.

Implements the 5-EMA system (10, 20, 50, 200, 800) used in the Market Makers
Method. Applied identically on all timeframes.

EMA behavior through the level cycle:
  1. Entry signal: Price crosses 10 and 20 EMA (usually on 2nd M/W peak)
  2. Level 1: Price breaks 50 EMA with high volume (MUST have volume)
  3. Board meeting after L1: Price retraces, should hold 50 EMA.
     If only reaches 10 EMA and counter to macro trend = caution.
  4. Level 2: Takes price to 200 EMA. Rejection at 200 EMA with hammer = TP zone.
  5. Level 2 board meeting: Retests 50 EMA.
  6. Level 3: Trend acceleration — EMAs fan out, price extends far from EMAs.
     Gap between price and EMAs = magnetic attraction, reversion coming.

Trend ending signals:
  - EMAs flatten (stop pointing in trend direction)
  - Gap between price and EMAs widening = Level 3 acceleration (trap)

PVSRA-style volume classification:
  - Vector 200 (Green/Red): volume >= 200% of avg of prior 10 candles
  - Vector 150 (Blue/Magenta): volume >= 150% of avg of prior 10 candles
  - Normal: below both thresholds
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default EMA periods and volume lookback
# ---------------------------------------------------------------------------
DEFAULT_EMA_PERIODS: list[int] = [10, 20, 50, 200, 800]
PVSRA_LOOKBACK: int = 10  # candles for PVSRA average volume baseline

# Slope lookback — number of candles used to measure EMA slope direction
SLOPE_LOOKBACK: int = 5

# Flattening threshold — absolute pct slope below which an EMA is "flat"
FLAT_SLOPE_THRESHOLD: float = 0.02  # 0.02%

# Retest proximity — price must come within this % of the EMA to count
RETEST_PROXIMITY_PCT: float = 0.15  # 0.15% of EMA value

# Number of recent candles to scan for retests / breaks
LOOKBACK_WINDOW: int = 20


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class EMAState:
    """Snapshot of all EMAs and their derived metrics."""

    values: dict[int, float]  # {10: 95123.4, 20: 94800.1, ...}
    slopes: dict[int, float]  # pct change over SLOPE_LOOKBACK candles per EMA
    alignment: str  # "bullish", "bearish", or "mixed"
    fan_out_score: float  # 0-1, how spread out the EMAs are relative to each other
    price_distance_from_50: float  # % distance of current price from 50 EMA
    price_distance_from_200: float  # % distance of current price from 200 EMA


@dataclass
class EMABreakResult:
    """Result of checking whether price broke through an EMA with volume."""

    broke_ema: bool
    direction: str  # "bullish" or "bearish"
    volume_confirmed: bool  # True if break candle had PVSRA vector_200 volume
    volume_ratio: float  # actual volume / avg volume of prior PVSRA_LOOKBACK candles
    break_candle_idx: int  # absolute index in the DataFrame (-1 if no break)


@dataclass
class TrendState:
    """High-level trend analysis derived from EMA alignment and slope."""

    direction: str  # "bullish", "bearish", or "sideways"
    strength: float  # 0-1, composite measure of trend conviction
    ema_alignment_score: float  # 0-1, how perfectly stacked the EMAs are
    is_accelerating: bool  # Level 3 condition — fan-out widening
    is_flattening: bool  # trend-ending condition — slopes near zero


@dataclass
class RetestResult:
    """Result of checking whether price retested an EMA after breaking it."""

    retested: bool
    held: bool  # True if price touched/wicked the EMA but closed on the break side
    ema_value: float  # EMA value at the retest candle
    retest_candle_idx: int  # absolute index in the DataFrame (-1 if no retest)


# ---------------------------------------------------------------------------
# EMA Framework
# ---------------------------------------------------------------------------
class EMAFramework:
    """5-EMA framework for Market Makers Method analysis.

    Computes the 10, 20, 50, 200, 800 EMAs and provides methods to detect
    EMA breaks with volume confirmation, trend state, and retests.
    """

    def __init__(self, periods: list[int] | None = None) -> None:
        self.periods: list[int] = periods if periods is not None else list(DEFAULT_EMA_PERIODS)
        # Keep sorted ascending so alignment checks work correctly
        self.periods.sort()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calculate(self, ohlc: pd.DataFrame) -> EMAState:
        """Compute all EMAs and return a full state snapshot.

        Args:
            ohlc: DataFrame with at least ``close`` column.

        Returns:
            EMAState with current values, slopes, alignment, fan-out, and
            price-distance metrics.
        """
        if ohlc.empty or len(ohlc) < max(self.periods):
            return self._empty_ema_state()

        close = ohlc["close"].astype(float)

        # Calculate EMAs
        ema_series: dict[int, pd.Series] = {}
        for period in self.periods:
            ema_series[period] = close.ewm(span=period, adjust=False).mean()

        # Latest values
        values: dict[int, float] = {p: float(ema_series[p].iloc[-1]) for p in self.periods}

        # Slopes — pct change over SLOPE_LOOKBACK candles
        slopes: dict[int, float] = {}
        for p in self.periods:
            s = ema_series[p]
            if len(s) >= SLOPE_LOOKBACK + 1:
                old_val = float(s.iloc[-(SLOPE_LOOKBACK + 1)])
                new_val = float(s.iloc[-1])
                slopes[p] = ((new_val - old_val) / old_val) * 100.0 if old_val != 0 else 0.0
            else:
                slopes[p] = 0.0

        # Alignment: bullish = 10 > 20 > 50 > 200 > 800 (price stacked above)
        alignment = self._compute_alignment(values)

        # Fan-out score: normalized spread between fastest and slowest EMA
        fan_out_score = self._compute_fan_out(values, float(close.iloc[-1]))

        # Price distances
        current_price = float(close.iloc[-1])
        dist_50 = self._pct_distance(current_price, values.get(50, current_price))
        dist_200 = self._pct_distance(current_price, values.get(200, current_price))

        return EMAState(
            values=values,
            slopes=slopes,
            alignment=alignment,
            fan_out_score=fan_out_score,
            price_distance_from_50=dist_50,
            price_distance_from_200=dist_200,
        )

    def detect_ema_break(
        self,
        ohlc: pd.DataFrame,
        ema_period: int = 50,
        volume_threshold: float = 2.0,
    ) -> EMABreakResult:
        """Detect if price broke an EMA with sufficient volume.

        Scans the last ``LOOKBACK_WINDOW`` candles for a close that crossed
        the EMA. Volume confirmation uses PVSRA logic: the break candle's
        volume must be >= ``volume_threshold`` * avg(prior PVSRA_LOOKBACK
        candles).

        Args:
            ohlc: OHLCV DataFrame with ``close`` and ``volume`` columns.
            ema_period: Which EMA to check for a break (default 50).
            volume_threshold: Multiple of average volume required (default 2.0
                = 200% of avg, i.e. PVSRA vector green/red).

        Returns:
            EMABreakResult describing the break (or lack thereof).
        """
        empty = EMABreakResult(
            broke_ema=False, direction="", volume_confirmed=False,
            volume_ratio=0.0, break_candle_idx=-1,
        )

        if ohlc.empty or len(ohlc) < ema_period + PVSRA_LOOKBACK:
            return empty

        close = ohlc["close"].astype(float)
        volume = ohlc["volume"].astype(float)
        ema = close.ewm(span=ema_period, adjust=False).mean()

        # Scan last LOOKBACK_WINDOW candles for a crossover
        start = max(1, len(ohlc) - LOOKBACK_WINDOW)
        for i in range(len(ohlc) - 1, start - 1, -1):
            prev_close = float(close.iloc[i - 1])
            curr_close = float(close.iloc[i])
            prev_ema = float(ema.iloc[i - 1])
            curr_ema = float(ema.iloc[i])

            # Bullish break: previous close below EMA, current close above
            bullish = prev_close <= prev_ema and curr_close > curr_ema
            # Bearish break: previous close above EMA, current close below
            bearish = prev_close >= prev_ema and curr_close < curr_ema

            if not bullish and not bearish:
                continue

            direction = "bullish" if bullish else "bearish"

            # Volume check — average of PVSRA_LOOKBACK candles before break
            vol_start = max(0, i - PVSRA_LOOKBACK)
            avg_vol = float(volume.iloc[vol_start:i].mean()) if i > vol_start else 0.0
            curr_vol = float(volume.iloc[i])
            vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 0.0
            vol_confirmed = vol_ratio >= volume_threshold

            logger.debug(
                "ema_break_detected",
                ema_period=ema_period,
                direction=direction,
                volume_ratio=round(vol_ratio, 2),
                volume_confirmed=vol_confirmed,
                candle_idx=i,
            )

            return EMABreakResult(
                broke_ema=True,
                direction=direction,
                volume_confirmed=vol_confirmed,
                volume_ratio=round(vol_ratio, 2),
                break_candle_idx=i,
            )

        return empty

    def get_trend_state(self, ohlc: pd.DataFrame) -> TrendState:
        """Analyze EMA alignment, fan-out, and flattening for trend state.

        Returns a composite view of the current trend derived from the EMAs:
        - direction: bullish/bearish/sideways based on alignment
        - strength: 0-1 composite of alignment + slope consistency
        - is_accelerating: Level 3 condition (fan-out widening rapidly)
        - is_flattening: trend-ending condition (slopes converging to zero)
        """
        empty = TrendState(
            direction="sideways", strength=0.0,
            ema_alignment_score=0.0, is_accelerating=False, is_flattening=False,
        )

        if ohlc.empty or len(ohlc) < max(self.periods):
            return empty

        state = self.calculate(ohlc)

        # Alignment score: count how many adjacent EMA pairs are in order
        alignment_score = self._alignment_score(state.values)

        # Direction from alignment
        if state.alignment == "bullish":
            direction = "bullish"
        elif state.alignment == "bearish":
            direction = "bearish"
        else:
            direction = "sideways"

        # Slope consistency — all slopes pointing the same way?
        slopes_list = list(state.slopes.values())
        bullish_slopes = sum(1 for s in slopes_list if s > FLAT_SLOPE_THRESHOLD)
        bearish_slopes = sum(1 for s in slopes_list if s < -FLAT_SLOPE_THRESHOLD)
        flat_slopes = sum(1 for s in slopes_list if abs(s) <= FLAT_SLOPE_THRESHOLD)
        total = len(slopes_list) or 1

        slope_consistency = max(bullish_slopes, bearish_slopes) / total

        # Strength = weighted combination of alignment + slope consistency
        strength = 0.6 * alignment_score + 0.4 * slope_consistency
        strength = min(1.0, max(0.0, strength))

        # Flattening: majority of EMAs have near-zero slope
        is_flattening = flat_slopes >= (total * 0.6)

        # Acceleration (Level 3): fan-out score is high AND increasing
        # We check current fan-out against a threshold
        is_accelerating = (
            state.fan_out_score > 0.6
            and not is_flattening
            and alignment_score > 0.7
        )

        if is_accelerating:
            logger.info(
                "ema_level3_acceleration",
                fan_out=round(state.fan_out_score, 3),
                alignment=state.alignment,
            )
        if is_flattening:
            logger.info(
                "ema_trend_flattening",
                flat_count=flat_slopes,
                total_emas=total,
            )

        return TrendState(
            direction=direction,
            strength=round(strength, 3),
            ema_alignment_score=round(alignment_score, 3),
            is_accelerating=is_accelerating,
            is_flattening=is_flattening,
        )

    def detect_retest(
        self,
        ohlc: pd.DataFrame,
        ema_period: int = 50,
    ) -> RetestResult:
        """Detect if price retested an EMA after previously breaking it.

        A retest is when price returns to the EMA zone (within
        RETEST_PROXIMITY_PCT) after breaking away. The retest "holds" if the
        candle closes back on the side of the break (e.g. broke above 50 EMA,
        retested it, and closed above it again).

        Scans the last LOOKBACK_WINDOW candles. Requires a prior break to
        exist before the retest (break must precede retest).

        Args:
            ohlc: OHLCV DataFrame.
            ema_period: Which EMA to check for a retest.

        Returns:
            RetestResult describing the retest or absence thereof.
        """
        empty = RetestResult(retested=False, held=False, ema_value=0.0, retest_candle_idx=-1)

        if ohlc.empty or len(ohlc) < ema_period + LOOKBACK_WINDOW:
            return empty

        close = ohlc["close"].astype(float)
        low = ohlc["low"].astype(float)
        high = ohlc["high"].astype(float)
        ema = close.ewm(span=ema_period, adjust=False).mean()

        # First find the most recent break
        break_result = self.detect_ema_break(ohlc, ema_period=ema_period, volume_threshold=0.0)
        if not break_result.broke_ema or break_result.break_candle_idx < 0:
            return empty

        break_idx = break_result.break_candle_idx
        break_dir = break_result.direction

        # Scan candles after the break for a retest
        for i in range(break_idx + 1, len(ohlc)):
            ema_val = float(ema.iloc[i])
            proximity = ema_val * (RETEST_PROXIMITY_PCT / 100.0)

            candle_low = float(low.iloc[i])
            candle_high = float(high.iloc[i])
            candle_close = float(close.iloc[i])

            # Did price wick into the EMA zone?
            touched_ema = False
            if break_dir == "bullish":
                # After a bullish break, a retest means price dipped back near the EMA
                touched_ema = candle_low <= ema_val + proximity
            else:
                # After a bearish break, a retest means price rallied back near the EMA
                touched_ema = candle_high >= ema_val - proximity

            if not touched_ema:
                continue

            # Check if it held — close stays on the break side
            if break_dir == "bullish":
                held = candle_close > ema_val
            else:
                held = candle_close < ema_val

            logger.debug(
                "ema_retest_detected",
                ema_period=ema_period,
                break_direction=break_dir,
                held=held,
                candle_idx=i,
                ema_value=round(ema_val, 2),
            )

            return RetestResult(
                retested=True,
                held=held,
                ema_value=round(ema_val, 8),
                retest_candle_idx=i,
            )

        return empty

    def classify_volume(self, ohlc: pd.DataFrame) -> pd.Series:
        """Classify each candle's volume using PVSRA logic.

        Returns a Series with the same index as ``ohlc``, containing:
          - ``"vector_200"`` — volume >= 200% of avg of prior 10 candles
            (Green candle if bullish, Red if bearish in PVSRA parlance)
          - ``"vector_150"`` — volume >= 150% of avg of prior 10 candles
            (Blue candle if bullish, Magenta if bearish)
          - ``"normal"`` — below both thresholds
        """
        if ohlc.empty or "volume" not in ohlc.columns:
            return pd.Series(dtype=str)

        volume = ohlc["volume"].astype(float)
        result = pd.Series("normal", index=ohlc.index)

        # Rolling average of prior PVSRA_LOOKBACK candles (shifted so current
        # candle is NOT included in its own average)
        avg_vol = volume.rolling(window=PVSRA_LOOKBACK, min_periods=PVSRA_LOOKBACK).mean().shift(1)

        # Compute ratio
        ratio = volume / avg_vol

        # Classify — check 200% first (more restrictive), then 150%
        result = result.where(ratio < 1.5, "vector_150")
        result = result.where(ratio < 2.0, "vector_200")

        # First PVSRA_LOOKBACK candles don't have a valid average — mark normal
        result.iloc[:PVSRA_LOOKBACK] = "normal"

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _compute_alignment(self, values: dict[int, float]) -> str:
        """Determine if EMAs are stacked bullish, bearish, or mixed.

        Bullish alignment: shorter EMAs above longer EMAs (10 > 20 > 50 > 200 > 800).
        Bearish alignment: shorter EMAs below longer EMAs (10 < 20 < 50 < 200 < 800).
        """
        sorted_periods = sorted(values.keys())  # ascending: 10, 20, 50, 200, 800
        vals = [values[p] for p in sorted_periods]

        # Check bullish: each value should be > the next (shorter EMA above longer)
        bullish_pairs = sum(1 for i in range(len(vals) - 1) if vals[i] > vals[i + 1])
        bearish_pairs = sum(1 for i in range(len(vals) - 1) if vals[i] < vals[i + 1])
        total_pairs = len(vals) - 1

        if total_pairs == 0:
            return "mixed"

        if bullish_pairs == total_pairs:
            return "bullish"
        if bearish_pairs == total_pairs:
            return "bearish"
        return "mixed"

    def _alignment_score(self, values: dict[int, float]) -> float:
        """Score 0-1 for how well EMAs are aligned (bullish or bearish).

        1.0 = perfect stacking in one direction.
        0.0 = completely mixed / no alignment.
        """
        sorted_periods = sorted(values.keys())
        vals = [values[p] for p in sorted_periods]
        total_pairs = len(vals) - 1

        if total_pairs == 0:
            return 0.0

        bullish_pairs = sum(1 for i in range(total_pairs) if vals[i] > vals[i + 1])
        bearish_pairs = sum(1 for i in range(total_pairs) if vals[i] < vals[i + 1])

        return max(bullish_pairs, bearish_pairs) / total_pairs

    def _compute_fan_out(self, values: dict[int, float], current_price: float) -> float:
        """Score 0-1 measuring how spread out the EMAs are.

        Uses the percentage spread between the fastest EMA (10) and the
        slowest EMA (800) normalized by current price. A spread of >= 10%
        maps to a fan_out_score of 1.0.
        """
        if not values or current_price == 0:
            return 0.0

        sorted_periods = sorted(values.keys())
        fastest = values[sorted_periods[0]]
        slowest = values[sorted_periods[-1]]

        spread_pct = abs(fastest - slowest) / current_price * 100.0

        # Normalize: 0% spread -> 0.0, >=10% spread -> 1.0
        max_spread = 10.0
        score = min(spread_pct / max_spread, 1.0)
        return round(score, 4)

    @staticmethod
    def _pct_distance(price: float, ema_value: float) -> float:
        """Percentage distance of price from an EMA value.

        Positive = price above EMA, negative = price below.
        """
        if ema_value == 0:
            return 0.0
        return round(((price - ema_value) / ema_value) * 100.0, 4)

    def _empty_ema_state(self) -> EMAState:
        """Return an empty/neutral EMAState when data is insufficient.

        Uses empty dicts so downstream code (target analyzer) doesn't
        treat zero-price entries as valid targets.
        """
        return EMAState(
            values={},
            slopes={},
            alignment="mixed",
            fan_out_score=0.0,
            price_distance_from_50=0.0,
            price_distance_from_200=0.0,
        )
