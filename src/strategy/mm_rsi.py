"""Market Makers Method RSI Indicator (C2).

Implements RSI(14) analysis for the MM Method confluence scoring system.
RSI is used as a confirming factor (C2) — it contributes 6 points to the
confluence score when the RSI state aligns with the trade direction.

Alignment logic (course rules):
  - Bullish divergence at a W formation = strong confirmation
  - Bearish divergence at an M formation = strong confirmation
  - RSI trend bias aligned with trade direction = confirmation
  - RSI crossing 50 in trade direction = additional confirmation
  - Neutral / misaligned = no confirmation

RSI formula uses Wilder smoothing (exponential with alpha = 1/period),
identical to the standard RSI(14) definition.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RSI_PERIOD: int = 14

# Trend bias thresholds (course rules)
BULLISH_THRESHOLD: float = 60.0  # RSI > 60 → bullish bias
BEARISH_THRESHOLD: float = 40.0  # RSI < 40 → bearish bias
# 40–60: neutral

# How many recent candles to scan for a 50-level cross
CROSS_50_LOOKBACK: int = 5

# How many recent candles to scan for divergence peaks/troughs
DIVERGENCE_LOOKBACK: int = 20


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class RSIState:
    """Snapshot of RSI state derived from recent OHLC data.

    All fields are computed from 1H candles via RSIAnalyzer.calculate().
    """

    rsi_value: float            # Current RSI(14) value (0–100)
    trend_bias: str             # "bullish" | "bearish" | "neutral"
    divergence_detected: bool   # True if price vs RSI divergence found
    divergence_type: str | None  # "bullish" | "bearish" | None
    crossed_50: bool            # RSI crossed the 50 level in recent candles


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
class RSIAnalyzer:
    """Computes RSI state for the MM Method confluence scoring system.

    Wilder smoothing (alpha = 1/period), identical to the standard RSI(14).

    Usage::

        analyzer = RSIAnalyzer(period=14)
        state = analyzer.calculate(candles_1h)
        if state and state.divergence_detected:
            # Divergence confirmed
    """

    def __init__(self, period: int = DEFAULT_RSI_PERIOD) -> None:
        self.period = period

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, ohlc: pd.DataFrame) -> RSIState | None:
        """Calculate RSI state from OHLC candles.

        Requires at least ``period + 1`` rows to produce a valid result.

        Args:
            ohlc: DataFrame with at least a ``close`` column.

        Returns:
            RSIState if sufficient data exists, otherwise None.
        """
        if ohlc is None or ohlc.empty:
            return None

        if len(ohlc) < self.period + 1:
            logger.debug(
                "rsi_insufficient_data",
                rows=len(ohlc),
                required=self.period + 1,
            )
            return None

        close = ohlc["close"].astype(float)
        rsi_series = self._compute_rsi_series(close)

        if rsi_series is None or rsi_series.empty:
            return None

        current_rsi = float(rsi_series.iloc[-1])
        trend_bias = self._classify_trend_bias(current_rsi)
        divergence_detected, divergence_type = self._detect_divergence(ohlc, rsi_series)
        crossed_50 = self._detect_50_cross(rsi_series)

        state = RSIState(
            rsi_value=round(current_rsi, 2),
            trend_bias=trend_bias,
            divergence_detected=divergence_detected,
            divergence_type=divergence_type,
            crossed_50=crossed_50,
        )

        logger.debug(
            "rsi_state",
            rsi=state.rsi_value,
            bias=state.trend_bias,
            divergence=state.divergence_type,
            crossed_50=state.crossed_50,
        )

        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_rsi_series(self, close: pd.Series) -> pd.Series | None:
        """Compute the full RSI series using Wilder smoothing.

        Wilder smoothing: alpha = 1/period (equivalent to EWM with
        com = period - 1, i.e. adjust=False and alpha = 1/period).

        Steps:
          1. price changes = close.diff()
          2. gains = changes.clip(lower=0), losses = (-changes).clip(lower=0)
          3. First avg: simple mean of the first `period` values
          4. Subsequent: prev_avg * (period-1)/period + current/period  (Wilder)
          5. RSI = 100 - 100 / (1 + avg_gain / avg_loss)
        """
        if len(close) < self.period + 1:
            return None

        delta = close.diff()
        gains = delta.clip(lower=0.0)
        losses = (-delta).clip(lower=0.0)

        alpha = 1.0 / self.period

        # Use pandas EWM with adjust=False, which gives Wilder smoothing
        # when alpha = 1/period. (alpha = 1/(period)) => com = period-1)
        avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()

        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0.0, float("nan"))
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Where avg_loss == 0 and avg_gain > 0 → RSI = 100
        rsi = rsi.where(avg_loss > 0, 100.0)

        # Drop the first NaN from .diff()
        return rsi.dropna()

    def _classify_trend_bias(self, rsi: float) -> str:
        """Classify the current RSI value into a trend bias.

        Course rules:
          - Uptrend range: RSI 40–80
          - Downtrend range: RSI 20–60
          - Bullish bias: RSI > 60 (above both ranges' midpoints)
          - Bearish bias: RSI < 40 (below both ranges' midpoints)
          - Neutral: 40 ≤ RSI ≤ 60
        """
        if rsi > BULLISH_THRESHOLD:
            return "bullish"
        if rsi < BEARISH_THRESHOLD:
            return "bearish"
        return "neutral"

    def _detect_divergence(
        self,
        ohlc: pd.DataFrame,
        rsi_series: pd.Series,
        lookback: int = DIVERGENCE_LOOKBACK,
    ) -> tuple[bool, str | None]:
        """Detect classic price vs RSI divergence.

        Bullish divergence: price makes a lower low AND RSI makes a higher low.
        Bearish divergence: price makes a higher high AND RSI makes a lower high.

        Uses the last ``lookback`` candles. Compares the most recent
        local extreme against the prior extreme in the same direction.

        Args:
            ohlc: OHLCV DataFrame aligned with rsi_series.
            rsi_series: Full RSI series (same length as close after dropna).
            lookback: How many candles to scan.

        Returns:
            (divergence_detected, divergence_type)  where divergence_type is
            "bullish", "bearish", or None.
        """
        # Need at least 2 * (lookback // 2) candles for meaningful comparison
        if len(rsi_series) < lookback or len(ohlc) < lookback:
            return False, None

        try:
            # Align: rsi_series may be shorter than ohlc due to .dropna()
            # Take the tail of both matching the shorter one
            n = min(len(rsi_series), len(ohlc), lookback)
            price_window = ohlc["close"].astype(float).values[-n:]
            low_window = ohlc["low"].astype(float).values[-n:]
            high_window = ohlc["high"].astype(float).values[-n:]
            rsi_window = rsi_series.values[-n:]

            mid = n // 2  # split window into two halves

            # --- Bullish divergence: price lower low, RSI higher low ---
            price_low_first = float(low_window[:mid].min())
            price_low_second = float(low_window[mid:].min())
            rsi_low_first = float(rsi_window[:mid].min())
            rsi_low_second = float(rsi_window[mid:].min())

            if price_low_second < price_low_first and rsi_low_second > rsi_low_first:
                logger.debug(
                    "rsi_bullish_divergence",
                    price_ll=(price_low_first, price_low_second),
                    rsi_hl=(rsi_low_first, rsi_low_second),
                )
                return True, "bullish"

            # --- Bearish divergence: price higher high, RSI lower high ---
            price_high_first = float(high_window[:mid].max())
            price_high_second = float(high_window[mid:].max())
            rsi_high_first = float(rsi_window[:mid].max())
            rsi_high_second = float(rsi_window[mid:].max())

            if price_high_second > price_high_first and rsi_high_second < rsi_high_first:
                logger.debug(
                    "rsi_bearish_divergence",
                    price_hh=(price_high_first, price_high_second),
                    rsi_lh=(rsi_high_first, rsi_high_second),
                )
                return True, "bearish"

        except Exception as exc:
            logger.debug("rsi_divergence_error", error=str(exc))

        return False, None

    def _detect_50_cross(
        self,
        rsi_series: pd.Series,
        lookback: int = CROSS_50_LOOKBACK,
    ) -> bool:
        """Check if RSI crossed the 50 level in the recent candles.

        A cross is detected when consecutive RSI values straddle 50 —
        one side is below and the other is above (or vice versa).

        Args:
            rsi_series: Full RSI series.
            lookback: How many recent candles to scan.

        Returns:
            True if a 50-level cross occurred in the last ``lookback`` candles.
        """
        if len(rsi_series) < 2:
            return False

        window = rsi_series.values[-min(lookback + 1, len(rsi_series)):]

        for i in range(1, len(window)):
            prev = float(window[i - 1])
            curr = float(window[i])
            if (prev < 50.0 and curr >= 50.0) or (prev >= 50.0 and curr < 50.0):
                logger.debug("rsi_50_cross", prev=round(prev, 2), curr=round(curr, 2))
                return True

        return False
