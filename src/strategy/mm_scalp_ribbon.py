"""Ribbon (Multi-EMA) Scalp Strategy (A8) for the MM Engine.

Implements a scalp trading strategy based on the TBD Scalp Trading course
(Lesson 01). This is an ALTERNATIVE entry path that runs alongside the weekly-
cycle MM engine and the VWAP+RSI(2) scalper — it fires only when no standard
M/W formation or VWAP+RSI signal is found.

Setup Components:
  - Multi-EMA ribbon: periods 2, 5, 8, 13, 21, 34, 55, 89, 100
  - Green ribbon  = bullish trend (fast EMAs above slow EMAs)
  - Red ribbon    = bearish trend (fast EMAs below slow EMAs)
  - Yellow EMAs   = middle-of-ribbon (21, 34) — key retest / pullback zones
  - Top 3 (fast)  = EMAs 2, 5, 8 — used for exhaustion detection

Entry Rules (Long):
  1. Ribbon is GREEN — all fast EMAs above slow EMAs
  2. EMAs squeezed tight (trend-change zone = small stop loss)
  3. Price pulls back to yellow EMAs (21, 34 average)
  4. Hammer candlestick pattern at the pullback
  5. Stop loss below the full ribbon (lowest EMA)
  6. Target: next S/R level on a higher timeframe
  7. Minimum R:R = 3:1

Entry Rules (Short):
  1. Ribbon is RED — all fast EMAs below slow EMAs
  2. EMAs squeezed tight
  3. Price pulls back up to yellow EMAs
  4. Inverted hammer / shooting star at the pullback
  5. Stop loss above the full ribbon (highest EMA)
  6. Target: next S/R level
  7. Minimum R:R = 3:1

Exit / No-Trade:
  - Top 3 EMAs fan out away from others = exhaustion warning
  - If price crosses through yellow EMAs in a long = exit signal
  - NO TRADE ZONE: when all EMAs are flat/sideways (range-bound)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.strategy.mm_formations import _is_hammer, _is_inverted_hammer
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All ribbon EMA periods (fastest → slowest)
RIBBON_PERIODS: list[int] = [2, 5, 8, 13, 21, 34, 55, 89, 100]

# "Yellow" EMAs — middle of the ribbon, used as retest/pullback zone
YELLOW_EMAS: list[int] = [21, 34]

# Top 3 fastest EMAs — divergence here signals exhaustion
FAST_EMAS: list[int] = [2, 5, 8]

# Minimum candle count for reliable EMA calculation (longest EMA * 1.5)
MIN_CANDLES = 150  # 100 * 1.5 = 150

# Squeeze: max spread between fastest and slowest EMA as % of price
SQUEEZE_THRESHOLD_PCT = 0.005   # 0.5%

# Fan-out: min spread between top-3 fastest EMAs as % of price = exhaustion
FAN_OUT_THRESHOLD_PCT = 0.02    # 2%

# No-trade zone: when ribbon slope is flat (max EMA change over last N candles)
FLAT_SLOPE_THRESHOLD_PCT = 0.001  # 0.1% change over look-back

# Pullback proximity: price must be within this % of yellow EMA average
PULLBACK_PROXIMITY_PCT = 0.005   # 0.5%

# Minimum acceptable risk:reward ratio
MIN_RR = 3.0

# Buffer beyond the ribbon for stop-loss placement
SL_BUFFER_PCT = 0.001  # 0.1%


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RibbonState:
    """Snapshot of the current multi-EMA ribbon state."""

    trend: str              # "bullish" | "bearish" | "flat"
    squeezed: bool          # EMAs close together (small stop, entry zone)
    fan_out: bool           # Top 3 EMAs fanning away = exhaustion
    yellow_ema_avg: float   # Average of EMAs 21 and 34 (retest zone)
    ribbon_high: float      # Highest EMA value (used for short SL)
    ribbon_low: float       # Lowest EMA value (used for long SL)


@dataclass
class RibbonSignal:
    """Detected Ribbon scalp setup."""

    detected: bool
    direction: str          # "long" | "short"
    entry_price: float
    stop_loss: float
    target: float
    risk_reward: float
    trend: str              # "bullish" | "bearish"
    squeezed: bool
    yellow_ema_avg: float
    reason: str             # Human-readable entry reason


# ---------------------------------------------------------------------------
# Main Analyzer
# ---------------------------------------------------------------------------


class RibbonAnalyzer:
    """Multi-EMA ribbon signal detector.

    Scans 15-minute candles for a trend-flip → pullback-to-yellow setup
    with a reversal candlestick, confirming the entry as close to the trend
    change as possible (squeezed EMAs = tight stop loss).
    """

    MIN_RR = MIN_RR

    def calculate_ribbon(self, ohlc: pd.DataFrame) -> RibbonState | None:
        """Calculate all ribbon EMAs and classify the current state.

        Args:
            ohlc: OHLCV DataFrame with at least a ``close`` column.
                  Needs >= MIN_CANDLES rows (150) for the slowest EMA (100).

        Returns:
            RibbonState, or None if insufficient data.
        """
        if ohlc is None or len(ohlc) < MIN_CANDLES:
            return None

        close = ohlc["close"].astype(float)

        # Calculate all EMAs
        ema_values: dict[int, float] = {}
        for period in RIBBON_PERIODS:
            ema_series = close.ewm(span=period, adjust=False).mean()
            ema_values[period] = float(ema_series.iloc[-1])

        # Determine trend: bullish if ALL fast EMAs > ALL slow EMAs
        # We check the overall order: EMA[2] > EMA[5] > ... > EMA[100]
        fast_vals = [ema_values[p] for p in FAST_EMAS]
        slow_vals = [ema_values[p] for p in [55, 89, 100]]

        min_fast = min(fast_vals)
        max_slow = max(slow_vals)

        # Flat check: is the slowest EMA barely moving?
        look_back = min(20, len(ohlc) - 1)
        ema_100_series = close.ewm(span=100, adjust=False).mean()
        ema_100_recent = float(ema_100_series.iloc[-1])
        ema_100_prior = float(ema_100_series.iloc[-(look_back + 1)])
        if ema_100_prior == 0:
            slope_pct = 0.0
        else:
            slope_pct = abs(ema_100_recent - ema_100_prior) / ema_100_prior

        if slope_pct < FLAT_SLOPE_THRESHOLD_PCT:
            trend = "flat"
        elif min_fast > max_slow:
            trend = "bullish"
        elif max(fast_vals) < min(slow_vals):
            trend = "bearish"
        else:
            trend = "flat"

        # Yellow EMA average (retest zone)
        yellow_avg = sum(ema_values[p] for p in YELLOW_EMAS) / len(YELLOW_EMAS)

        # Ribbon high / low (outermost EMAs for SL placement)
        all_vals = list(ema_values.values())
        ribbon_high = max(all_vals)
        ribbon_low = min(all_vals)

        # Squeeze: is the spread between fastest and slowest EMA narrow?
        current_price = float(close.iloc[-1])
        spread = abs(ema_values[RIBBON_PERIODS[0]] - ema_values[RIBBON_PERIODS[-1]])
        squeezed = (spread / current_price) < SQUEEZE_THRESHOLD_PCT

        # Fan-out: are the top-3 fastest EMAs spreading apart (exhaustion)?
        top3_vals = [ema_values[p] for p in FAST_EMAS]
        top3_spread = max(top3_vals) - min(top3_vals)
        fan_out = (top3_spread / current_price) > FAN_OUT_THRESHOLD_PCT

        return RibbonState(
            trend=trend,
            squeezed=squeezed,
            fan_out=fan_out,
            yellow_ema_avg=yellow_avg,
            ribbon_high=ribbon_high,
            ribbon_low=ribbon_low,
        )

    def scan(
        self,
        candles_15m: pd.DataFrame,
        targets: list[float] | None = None,
    ) -> RibbonSignal | None:
        """Scan for a Ribbon strategy setup on the 15-minute chart.

        Steps:
          1. Calculate all ribbon EMAs
          2. Require an established trend (bullish or bearish, not flat)
          3. Reject if exhaustion is detected (top-3 fanning out)
          4. Require a pullback to the yellow EMA zone
          5. Require a reversal candlestick at the pullback
          6. Calculate SL (beyond full ribbon), find target, check R:R >= 3

        Args:
            candles_15m: 15-minute OHLCV candles (needs >= MIN_CANDLES rows).
            targets: Significant price levels for target selection (HOW, LOW,
                     S/R levels). If None, a minimum 3:1 projection is used.

        Returns:
            RibbonSignal if a valid setup is detected, None otherwise.
        """
        # ---- 1. Data validation ----
        if candles_15m is None or len(candles_15m) < MIN_CANDLES:
            logger.debug(
                "ribbon_insufficient_data",
                rows=0 if candles_15m is None else len(candles_15m),
                required=MIN_CANDLES,
            )
            return None

        # ---- 2. Calculate ribbon state ----
        state = self.calculate_ribbon(candles_15m)
        if state is None:
            return None

        # ---- 3. No-trade zone: flat ribbon ----
        if state.trend == "flat":
            logger.debug("ribbon_no_trade_flat")
            return None

        # ---- 4. Reject exhaustion ----
        if state.fan_out:
            logger.debug(
                "ribbon_exhaustion_fan_out",
                trend=state.trend,
            )
            return None

        # ---- 5. Determine direction ----
        direction = "long" if state.trend == "bullish" else "short"

        # ---- 6. Current price ----
        close = candles_15m["close"].astype(float)
        price = float(close.iloc[-1])

        # ---- 7. Pullback to yellow EMA zone ----
        if not self._is_pullback_to_yellow(price, state.yellow_ema_avg, direction):
            logger.debug(
                "ribbon_no_pullback",
                price=price,
                yellow_avg=round(state.yellow_ema_avg, 4),
                direction=direction,
            )
            return None

        # ---- 8. Reversal candlestick ----
        pattern = self._detect_pattern(candles_15m, direction)
        if pattern is None:
            logger.debug("ribbon_no_pattern", direction=direction)
            return None

        # ---- 9. Calculate SL ----
        stop_loss = self._calculate_stop_loss(state, direction)

        # ---- 10. Find target ----
        target = self._find_target(price, stop_loss, direction, targets)
        if target is None:
            logger.debug("ribbon_no_target")
            return None

        # ---- 11. Check R:R ----
        risk = abs(price - stop_loss)
        if risk == 0:
            return None
        reward = abs(target - price)
        rr = reward / risk

        if rr < self.MIN_RR:
            logger.debug("ribbon_low_rr", rr=round(rr, 2), min_rr=self.MIN_RR)
            return None

        # ---- 12. Build signal ----
        reason = (
            f"Ribbon scalp {direction.upper()}: {state.trend} trend, "
            f"{pattern} at yellow EMA ({state.yellow_ema_avg:.4f}), "
            f"squeezed={state.squeezed}, R:R={rr:.1f}"
        )

        signal = RibbonSignal(
            detected=True,
            direction=direction,
            entry_price=round(price, 8),
            stop_loss=round(stop_loss, 8),
            target=round(target, 8),
            risk_reward=round(rr, 2),
            trend=state.trend,
            squeezed=state.squeezed,
            yellow_ema_avg=round(state.yellow_ema_avg, 8),
            reason=reason,
        )

        logger.info(
            "ribbon_signal_detected",
            direction=direction,
            entry=price,
            sl=stop_loss,
            tp=target,
            rr=round(rr, 2),
            trend=state.trend,
            squeezed=state.squeezed,
            pattern=pattern,
        )

        return signal

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_pullback_to_yellow(
        self, price: float, yellow_avg: float, direction: str
    ) -> bool:
        """Check if price has pulled back to the yellow EMA zone.

        For longs: price came down to near the yellow EMAs from above.
        For shorts: price bounced up to near the yellow EMAs from below.
        Both require price to be within PULLBACK_PROXIMITY_PCT of yellow_avg.
        """
        if yellow_avg == 0:
            return False
        proximity = abs(price - yellow_avg) / yellow_avg
        return proximity <= PULLBACK_PROXIMITY_PCT

    def _detect_pattern(
        self, candles: pd.DataFrame, direction: str
    ) -> str | None:
        """Detect a reversal candlestick on the last candle.

        For longs: hammer pattern.
        For shorts: inverted hammer (shooting star) pattern.
        Reuses _is_hammer / _is_inverted_hammer from mm_formations.py.
        """
        last = candles.iloc[-1]
        o = float(last["open"])
        h = float(last["high"])
        lo = float(last["low"])
        c = float(last["close"])

        if direction == "long":
            if _is_hammer(o, h, lo, c):
                return "hammer"
        else:
            if _is_inverted_hammer(o, h, lo, c):
                return "inverted_hammer"

        return None

    def _calculate_stop_loss(self, state: RibbonState, direction: str) -> float:
        """Place stop loss beyond the full ribbon with a small buffer.

        For longs: SL below the lowest ribbon EMA.
        For shorts: SL above the highest ribbon EMA.
        """
        if direction == "long":
            return state.ribbon_low * (1 - SL_BUFFER_PCT)
        else:
            return state.ribbon_high * (1 + SL_BUFFER_PCT)

    def _find_target(
        self,
        price: float,
        stop_loss: float,
        direction: str,
        targets: list[float] | None,
    ) -> float | None:
        """Find the nearest valid target level with >= MIN_RR risk:reward.

        Prefers the closest valid S/R level from the supplied ``targets`` list.
        Falls back to a 3:1 projection if no external targets qualify.
        """
        risk = abs(price - stop_loss)
        if risk == 0:
            return None

        min_target_dist = risk * self.MIN_RR

        if targets:
            valid: list[float] = []
            for t in targets:
                if direction == "long" and t > price + min_target_dist:
                    valid.append(t)
                elif direction == "short" and t < price - min_target_dist:
                    valid.append(t)

            if valid:
                return min(valid) if direction == "long" else max(valid)

        # No external target qualifies — use minimum 3:1 projection
        if direction == "long":
            return price + min_target_dist
        else:
            return price - min_target_dist
