"""VWAP + RSI(2) Scalp Strategy (A7) for the MM Engine.

Implements a scalp trading strategy based on the TBD Scalp Trading course
(Lessons 02-10). This is an ALTERNATIVE entry path that runs alongside the
weekly-cycle MM engine — it fires only when no standard M/W formation is found.

Setup Components:
  - VWAP (Volume Weighted Average Price), resets daily at midnight UTC
  - RSI(2) on 15-minute chart for entry timing (oversold < 10, overbought > 90)
  - RSI(14) on 1-hour chart for directional bias
  - 255 EMA on both timeframes
  - Candlestick reversal patterns (hammer, inverted hammer) at VWAP/EMA pullback

Entry Rules (Long):
  1. Price ABOVE both VWAP and 255 EMA
  2. Price pulls back to retest VWAP or 255 EMA
  3. RSI(2) drops BELOW 10 (oversold) during pullback
  4. Hammer candlestick pattern at the pullback
  5. Stop loss below both VWAP and 255 EMA
  6. Target: next significant level
  7. Minimum R:R = 3:1

Entry Rules (Short):
  1. Price BELOW both VWAP and 255 EMA
  2. Price pulls back up to retest VWAP or 255 EMA
  3. RSI(2) rises ABOVE 90 (overbought) during pullback
  4. Inverted hammer / shooting star at the pullback
  5. Stop loss above both VWAP and 255 EMA
  6. Target: next significant level
  7. Minimum R:R = 3:1

No-Trade Zone:
  - Price is between VWAP and 255 EMA
  - VWAP and 255 EMA far apart (fanned out)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.strategy.mm_formations import _is_hammer, _is_inverted_hammer
from src.strategy.mm_rsi import RSIAnalyzer
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RSI_SCALP_PERIOD = 2
RSI_BIAS_PERIOD = 14
EMA_PERIOD = 255
RSI_OVERBOUGHT = 90
RSI_OVERSOLD = 10
MIN_RR = 3.0

# If price is between VWAP and 255 EMA and the gap is < 0.1% of price,
# it's a no-trade zone (too tight / ambiguous trend).
NO_TRADE_ZONE_PCT = 0.001

# Maximum fan-out: if VWAP and 255 EMA are > 2% apart, the trend is
# overextended — a pullback to the moving average is unlikely.
MAX_FANOUT_PCT = 0.02

# Pullback proximity: price must be within 0.3% of VWAP or 255 EMA
# to count as a "retest".
PULLBACK_PROXIMITY_PCT = 0.003

# Minimum candles needed for reliable calculation
MIN_CANDLES_15M = 30
MIN_CANDLES_1H = 15


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScalpSignal:
    """Detected VWAP + RSI scalp setup."""

    detected: bool
    direction: str           # "long" or "short"
    entry_price: float
    stop_loss: float
    target: float
    risk_reward: float
    rsi_2_value: float       # RSI(2) value at entry
    rsi_14_bias: str         # "bullish" | "bearish" | "neutral" from 1H RSI(14)
    vwap_value: float
    ema_255_value: float
    pattern: str             # "hammer", "inverted_hammer", "morning_star", "evening_star"
    reason: str              # Human-readable entry reason


# ---------------------------------------------------------------------------
# VWAP Calculator
# ---------------------------------------------------------------------------

class VWAPCalculator:
    """Calculate Volume Weighted Average Price with daily reset.

    VWAP = cumsum(typical_price * volume) / cumsum(volume)
    typical_price = (high + low + close) / 3

    Resets at midnight UTC.
    """

    def calculate(self, ohlcv: pd.DataFrame) -> float | None:
        """Calculate current VWAP from intraday candles.

        Expects OHLCV DataFrame with columns: open, high, low, close, volume.
        The DataFrame should contain candles from the current day (since last
        midnight UTC). If a 'timestamp' column exists, only candles from the
        current UTC day are used.

        Returns:
            Current VWAP value, or None if insufficient data.
        """
        if ohlcv is None or ohlcv.empty:
            return None

        for col in ("high", "low", "close", "volume"):
            if col not in ohlcv.columns:
                return None

        df = ohlcv.copy()

        # Filter to current UTC day if timestamps are available
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            latest_ts = df["timestamp"].iloc[-1]
            day_start = latest_ts.normalize()  # midnight UTC
            df = df[df["timestamp"] >= day_start]
            if df.empty:
                return None

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # Avoid division by zero
        cum_vol = volume.cumsum()
        if cum_vol.iloc[-1] == 0:
            return None

        typical_price = (high + low + close) / 3.0
        cum_tp_vol = (typical_price * volume).cumsum()

        vwap = cum_tp_vol / cum_vol
        return float(vwap.iloc[-1])

    def calculate_series(self, ohlcv: pd.DataFrame) -> pd.Series | None:
        """Calculate VWAP series (all values, not just the latest).

        Returns a pandas Series aligned with the input DataFrame index.
        """
        if ohlcv is None or ohlcv.empty:
            return None

        for col in ("high", "low", "close", "volume"):
            if col not in ohlcv.columns:
                return None

        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)
        close = ohlcv["close"].astype(float)
        volume = ohlcv["volume"].astype(float)

        cum_vol = volume.cumsum()
        typical_price = (high + low + close) / 3.0
        cum_tp_vol = (typical_price * volume).cumsum()

        # Replace zero cum_vol with NaN to avoid div/0
        vwap = cum_tp_vol / cum_vol.replace(0, float("nan"))
        return vwap


# ---------------------------------------------------------------------------
# Scalp RSI (wraps RSIAnalyzer with configurable period)
# ---------------------------------------------------------------------------

class ScalpRSI:
    """RSI calculator with configurable period.

    Reuses the Wilder-smoothing RSI from mm_rsi.py.
    """

    def __init__(self, period: int = RSI_SCALP_PERIOD) -> None:
        self._analyzer = RSIAnalyzer(period=period)

    def calculate_series(self, closes: pd.Series) -> pd.Series | None:
        """Calculate full RSI series from close prices.

        Returns:
            RSI series (NaNs dropped from initial warmup), or None.
        """
        return self._analyzer.compute_rsi_series(closes)

    def current_value(self, closes: pd.Series) -> float | None:
        """Get the most recent RSI value."""
        series = self.calculate_series(closes)
        if series is None or series.empty:
            return None
        return float(series.iloc[-1])


# ---------------------------------------------------------------------------
# Main Scalper
# ---------------------------------------------------------------------------

class VWAPRSIScalper:
    """VWAP + RSI(2) scalp signal detector.

    Scans 15-minute candles for a pullback-to-VWAP/255-EMA setup with an
    extreme RSI(2) reading and a reversal candlestick pattern.
    """

    MIN_RR = MIN_RR
    RSI_OVERBOUGHT = RSI_OVERBOUGHT
    RSI_OVERSOLD = RSI_OVERSOLD
    EMA_PERIOD = EMA_PERIOD

    def __init__(self) -> None:
        self.vwap_calc = VWAPCalculator()
        self.rsi_2 = ScalpRSI(period=RSI_SCALP_PERIOD)
        self.rsi_14 = ScalpRSI(period=RSI_BIAS_PERIOD)

    def scan(
        self,
        candles_15m: pd.DataFrame,
        candles_1h: pd.DataFrame,
        targets: list[float] | None = None,
    ) -> ScalpSignal | None:
        """Scan for VWAP+RSI scalp setup on 15-min chart.

        Args:
            candles_15m: 15-minute OHLCV candles (needs >= 30 rows).
            candles_1h: 1-hour candles for RSI(14) bias (needs >= 15 rows).
            targets: Significant price levels (S/R, HOW/LOW) for target selection.

        Returns:
            ScalpSignal if a valid setup is detected, None otherwise.
        """
        # ---- Validation ----
        if candles_15m is None or len(candles_15m) < MIN_CANDLES_15M:
            logger.debug("scalp_insufficient_15m_data",
                         rows=0 if candles_15m is None else len(candles_15m))
            return None

        if candles_1h is None or len(candles_1h) < MIN_CANDLES_1H:
            logger.debug("scalp_insufficient_1h_data",
                         rows=0 if candles_1h is None else len(candles_1h))
            return None

        # ---- 1. Calculate indicators on 15m ----
        close_15m = candles_15m["close"].astype(float)

        vwap = self.vwap_calc.calculate(candles_15m)
        if vwap is None:
            logger.debug("scalp_no_vwap")
            return None

        ema_255 = self._calc_ema_value(close_15m, self.EMA_PERIOD)
        if ema_255 is None:
            logger.debug("scalp_no_ema255")
            return None

        rsi_2_val = self.rsi_2.current_value(close_15m)
        if rsi_2_val is None:
            logger.debug("scalp_no_rsi2")
            return None

        # ---- 2. 1H RSI(14) bias ----
        rsi_14_bias = self._get_1h_bias(candles_1h)

        # ---- 3. Current price ----
        price = float(close_15m.iloc[-1])

        # ---- 4. No-trade zone: price between VWAP and 255 EMA ----
        if self._in_no_trade_zone(price, vwap, ema_255):
            logger.debug("scalp_no_trade_zone",
                         price=price, vwap=vwap, ema=ema_255)
            return None

        # ---- 5. Fan-out check ----
        if self._is_fanned_out(vwap, ema_255, price):
            logger.debug("scalp_fanned_out",
                         vwap=vwap, ema=ema_255, gap_pct=abs(vwap - ema_255) / price)
            return None

        # ---- 6. Determine direction from price vs VWAP/EMA ----
        above_vwap = price > vwap
        above_ema = price > ema_255

        if above_vwap and above_ema:
            direction = "long"
        elif not above_vwap and not above_ema:
            direction = "short"
        else:
            # Price above one but below the other — ambiguous, but if at least
            # one is satisfied and structure has changed, proceed with caution.
            # Course says "at least one after structure change". Use the RSI
            # bias to break the tie.
            if rsi_14_bias == "bullish" and above_ema:
                direction = "long"
            elif rsi_14_bias == "bearish" and not above_ema:
                direction = "short"
            else:
                logger.debug("scalp_ambiguous_direction",
                             above_vwap=above_vwap, above_ema=above_ema,
                             bias=rsi_14_bias)
                return None

        # ---- 7. Check RSI(2) extreme ----
        if direction == "long" and rsi_2_val >= self.RSI_OVERSOLD:
            logger.debug("scalp_rsi2_not_oversold", rsi2=rsi_2_val)
            return None
        if direction == "short" and rsi_2_val <= self.RSI_OVERBOUGHT:
            logger.debug("scalp_rsi2_not_overbought", rsi2=rsi_2_val)
            return None

        # ---- 8. Check pullback proximity to VWAP or 255 EMA ----
        if not self._is_pullback(price, vwap, ema_255, direction):
            logger.debug("scalp_no_pullback", price=price, vwap=vwap, ema=ema_255)
            return None

        # ---- 9. Candlestick reversal pattern ----
        pattern = self._detect_pattern(candles_15m, direction)
        if pattern is None:
            logger.debug("scalp_no_pattern", direction=direction)
            return None

        # ---- 10. Calculate SL ----
        stop_loss = self._calculate_stop_loss(price, vwap, ema_255, direction)

        # ---- 11. Find target ----
        target = self._find_target(price, stop_loss, direction, targets)
        if target is None:
            logger.debug("scalp_no_target")
            return None

        # ---- 12. Check R:R ----
        risk = abs(price - stop_loss)
        if risk == 0:
            return None
        reward = abs(target - price)
        rr = reward / risk

        if rr < self.MIN_RR:
            logger.debug("scalp_low_rr", rr=round(rr, 2), min_rr=self.MIN_RR)
            return None

        # ---- 13. Build signal ----
        reason = (
            f"VWAP+RSI scalp {direction.upper()}: "
            f"RSI(2)={rsi_2_val:.1f}, {pattern} at "
            f"{'VWAP' if self._near(price, vwap) else '255 EMA'} pullback, "
            f"1H bias={rsi_14_bias}, R:R={rr:.1f}"
        )

        signal = ScalpSignal(
            detected=True,
            direction=direction,
            entry_price=round(price, 8),
            stop_loss=round(stop_loss, 8),
            target=round(target, 8),
            risk_reward=round(rr, 2),
            rsi_2_value=round(rsi_2_val, 2),
            rsi_14_bias=rsi_14_bias,
            vwap_value=round(vwap, 8),
            ema_255_value=round(ema_255, 8),
            pattern=pattern,
            reason=reason,
        )

        logger.info(
            "scalp_vwap_rsi_signal",
            direction=direction,
            entry=price,
            sl=stop_loss,
            tp=target,
            rr=round(rr, 2),
            rsi2=round(rsi_2_val, 2),
            pattern=pattern,
            bias=rsi_14_bias,
        )

        return signal

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calc_ema_value(self, series: pd.Series, period: int) -> float | None:
        """Calculate current EMA value. Returns None if insufficient data."""
        if len(series) < period:
            return None
        ema = series.ewm(span=period, adjust=False).mean()
        return float(ema.iloc[-1])

    def _get_1h_bias(self, candles_1h: pd.DataFrame) -> str:
        """Determine directional bias from 1H RSI(14).

        Course rules for scalp bias:
          - Uptrend: RSI 40-80 → bullish bias
          - Downtrend: RSI 20-60 → bearish bias
          - Use thresholds: > 60 = bullish, < 40 = bearish, 40-60 = neutral
        """
        close_1h = candles_1h["close"].astype(float)
        rsi_val = self.rsi_14.current_value(close_1h)
        if rsi_val is None:
            return "neutral"
        if rsi_val > 60:
            return "bullish"
        if rsi_val < 40:
            return "bearish"
        return "neutral"

    def _in_no_trade_zone(self, price: float, vwap: float, ema: float) -> bool:
        """Check if price is stuck between VWAP and 255 EMA (no-trade zone).

        When price is between the two, the trend is ambiguous.
        """
        upper = max(vwap, ema)
        lower = min(vwap, ema)

        # Only a no-trade zone if VWAP and EMA have meaningful separation
        gap = upper - lower
        if gap / price < NO_TRADE_ZONE_PCT:
            # VWAP and EMA are basically on top of each other — not a no-trade
            # zone, just convergence. Direction is determined by price vs both.
            return False

        return lower <= price <= upper

    def _is_fanned_out(self, vwap: float, ema: float, price: float) -> bool:
        """Check if VWAP and 255 EMA are too far apart (overextended trend)."""
        gap_pct = abs(vwap - ema) / price
        return gap_pct > MAX_FANOUT_PCT

    def _is_pullback(
        self, price: float, vwap: float, ema: float, direction: str
    ) -> bool:
        """Check if price has pulled back to retest VWAP or 255 EMA.

        For longs: price should be near (within PULLBACK_PROXIMITY_PCT) of
        VWAP or EMA from above.
        For shorts: price should be near from below.
        """
        if direction == "long":
            # Price above both but has come back down near one of them
            anchor = min(vwap, ema)  # whichever is lower is the pullback target
            return self._near(price, anchor) or self._near(price, max(vwap, ema))
        else:
            # Price below both but has risen up near one of them
            anchor = max(vwap, ema)  # whichever is higher is the pullback target
            return self._near(price, anchor) or self._near(price, min(vwap, ema))

    def _near(self, a: float, b: float) -> bool:
        """Check if two values are within PULLBACK_PROXIMITY_PCT of each other."""
        if b == 0:
            return False
        return abs(a - b) / b <= PULLBACK_PROXIMITY_PCT

    def _detect_pattern(
        self, candles: pd.DataFrame, direction: str
    ) -> str | None:
        """Detect a candlestick reversal pattern on the last 3 candles.

        For longs: hammer or morning star (3-candle bullish reversal).
        For shorts: inverted hammer (shooting star) or evening star.

        Reuses _is_hammer / _is_inverted_hammer from mm_formations.py.
        """
        n = len(candles)
        last = candles.iloc[-1]
        o = float(last["open"])
        h = float(last["high"])
        lo = float(last["low"])
        c = float(last["close"])

        if direction == "long":
            if _is_hammer(o, h, lo, c):
                return "hammer"
            # Morning star: 3-candle pattern (bearish, small body, bullish)
            if n >= 3:
                if self._is_morning_star(candles.iloc[-3], candles.iloc[-2], last):
                    return "morning_star"
            # Bullish engulfing on last candle
            if n >= 2:
                prev = candles.iloc[-2]
                if self._is_bullish_engulfing(prev, last):
                    return "bullish_engulfing"
        else:
            if _is_inverted_hammer(o, h, lo, c):
                return "inverted_hammer"
            if n >= 3:
                if self._is_evening_star(candles.iloc[-3], candles.iloc[-2], last):
                    return "evening_star"
            if n >= 2:
                prev = candles.iloc[-2]
                if self._is_bearish_engulfing(prev, last):
                    return "bearish_engulfing"

        return None

    def _is_morning_star(self, c1, c2, c3) -> bool:
        """3-candle bullish reversal: bearish, small body doji, bullish."""
        o1, c1v = float(c1["open"]), float(c1["close"])
        o2, h2, l2, c2v = float(c2["open"]), float(c2["high"]), float(c2["low"]), float(c2["close"])
        o3, c3v = float(c3["open"]), float(c3["close"])

        body1 = abs(c1v - o1)
        body2 = abs(c2v - o2)
        range2 = h2 - l2

        if range2 == 0 or body1 == 0:
            return False

        # c1: bearish, c2: small body (< 30% of range), c3: bullish
        return (
            c1v < o1  # bearish first candle
            and body2 / range2 < 0.3  # small body middle
            and c3v > o3  # bullish third candle
            and c3v > (o1 + c1v) / 2  # closes above midpoint of first
        )

    def _is_evening_star(self, c1, c2, c3) -> bool:
        """3-candle bearish reversal: bullish, small body doji, bearish."""
        o1, c1v = float(c1["open"]), float(c1["close"])
        o2, h2, l2, c2v = float(c2["open"]), float(c2["high"]), float(c2["low"]), float(c2["close"])
        o3, c3v = float(c3["open"]), float(c3["close"])

        body1 = abs(c1v - o1)
        body2 = abs(c2v - o2)
        range2 = h2 - l2

        if range2 == 0 or body1 == 0:
            return False

        return (
            c1v > o1  # bullish first candle
            and body2 / range2 < 0.3  # small body middle
            and c3v < o3  # bearish third candle
            and c3v < (o1 + c1v) / 2  # closes below midpoint of first
        )

    def _is_bullish_engulfing(self, prev, cur) -> bool:
        """Bullish engulfing: red prev candle, green cur candle wraps it."""
        po, pc = float(prev["open"]), float(prev["close"])
        co, cc = float(cur["open"]), float(cur["close"])
        return pc < po and cc > co and cc > po and co < pc

    def _is_bearish_engulfing(self, prev, cur) -> bool:
        """Bearish engulfing: green prev candle, red cur candle wraps it."""
        po, pc = float(prev["open"]), float(prev["close"])
        co, cc = float(cur["open"]), float(cur["close"])
        return pc > po and cc < co and cc < po and co > pc

    def _calculate_stop_loss(
        self, price: float, vwap: float, ema: float, direction: str
    ) -> float:
        """Place stop loss beyond both VWAP and 255 EMA.

        For longs: SL below the lower of (VWAP, 255 EMA) minus a small buffer.
        For shorts: SL above the higher of (VWAP, 255 EMA) plus a small buffer.
        """
        buffer_pct = 0.001  # 0.1% buffer beyond the levels

        if direction == "long":
            sl_ref = min(vwap, ema)
            return sl_ref * (1 - buffer_pct)
        else:
            sl_ref = max(vwap, ema)
            return sl_ref * (1 + buffer_pct)

    def _find_target(
        self,
        price: float,
        stop_loss: float,
        direction: str,
        targets: list[float] | None,
    ) -> float | None:
        """Find the best target from significant levels that gives >= 3:1 R:R.

        If no external targets are provided, uses a default 3:1 projection.
        """
        risk = abs(price - stop_loss)
        if risk == 0:
            return None

        min_target_dist = risk * self.MIN_RR

        if targets:
            # Filter targets that are in the right direction and far enough
            valid_targets = []
            for t in targets:
                if direction == "long" and t > price + min_target_dist:
                    valid_targets.append(t)
                elif direction == "short" and t < price - min_target_dist:
                    valid_targets.append(t)

            if valid_targets:
                # Pick the nearest valid target (conservative)
                if direction == "long":
                    return min(valid_targets)
                else:
                    return max(valid_targets)

        # No valid external target — use minimum 3:1 projection
        if direction == "long":
            return price + min_target_dist
        else:
            return price - min_target_dist
