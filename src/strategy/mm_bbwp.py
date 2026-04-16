"""Bollinger Band Width Percentile (BBWP) indicator.

Course Trading Strategies Lesson 04 (C4):
  - BBWP = Bollinger Band Width expressed as a percentile of its historical range.
  - Length: 252 (lookback for percentile), MA Length: 10 (EMA smoothing),
    MA Type: EMA, MA Data Length: 20 (BB calculation period).
  - Alert at 95 (top): local top OR bottom found — be careful.
  - Alert at 5 (bottom): consolidation maturing, breakout is coming soon.
  - Does NOT tell direction — only WHEN moves are coming.

This is a timing indicator only. It does not contribute to confluence
scoring or entry decisions directly. It is wired into mm_engine.py for
logging / telemetry so scan logs show current volatility timing state.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BBWP_LENGTH = 252          # Lookback period for percentile calculation
BBWP_MA_LENGTH = 10        # EMA smoothing length applied to BBWP values
BBWP_BB_PERIOD = 20        # Period for Bollinger Band SMA and StdDev
BBWP_EXTREME_HIGH = 95.0   # Alert threshold: top = extreme reached (be careful)
BBWP_EXTREME_LOW = 5.0     # Alert threshold: bottom = breakout coming


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BBWPState:
    """Current BBWP state.

    Attributes:
        bbwp_value: Current BBWP percentile (0-100). Higher = wider bands
            relative to history (volatility expansion). Lower = narrower bands
            (consolidation / compression).
        signal: Interpretation:
            "breakout_imminent"  — BBWP <= 5: consolidation maturing, big move soon.
            "extreme_reached"    — BBWP >= 95: volatility at extreme, local top/bottom.
            "neutral"            — 5 < BBWP < 95: no timing signal.
        ma_value: 10-period EMA of the BBWP value (smoothed percentile).
    """
    bbwp_value: float          # 0-100 percentile
    signal: str                # "breakout_imminent" | "extreme_reached" | "neutral"
    ma_value: float            # EMA-smoothed BBWP


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BBWPAnalyzer:
    """Bollinger Band Width Percentile analyzer.

    Course lesson C4 parameters:
      - length=252  (lookback for percentile)
      - ma_length=10 (EMA smoothing)
      - BB period=20 (SMA + stddev window for band width)

    Usage:
        analyzer = BBWPAnalyzer()
        state = analyzer.calculate(ohlc)
        if state and state.signal == "breakout_imminent":
            # Big move coming soon — look for entry confirmation
    """

    def __init__(
        self,
        length: int = BBWP_LENGTH,
        ma_length: int = BBWP_MA_LENGTH,
        bb_period: int = BBWP_BB_PERIOD,
    ) -> None:
        self.length = length
        self.ma_length = ma_length
        self.bb_period = bb_period

    def calculate(self, ohlc: pd.DataFrame) -> BBWPState | None:
        """Calculate the current BBWP state.

        Steps:
          1. Compute Bollinger Band Width for each bar:
               width = (upper - lower) / middle
               where:
                 middle = SMA(close, 20)
                 upper  = middle + 2 * stdev(close, 20)
                 lower  = middle - 2 * stdev(close, 20)
               Simplifies to: width = (4 * stdev) / SMA = 4 * (stdev / SMA)
          2. Compute the BBWP: percentile rank of the current width within
             the last `length` (252) width values.
          3. Apply `ma_length` (10)-period EMA smoothing to the BBWP series.
          4. Classify: >= 95 = "extreme_reached", <= 5 = "breakout_imminent",
             else "neutral".

        Args:
            ohlc: OHLCV DataFrame with at least `bb_period + length` rows and
                  a 'close' column. Fewer rows → returns None.

        Returns:
            BBWPState with the current reading, or None if insufficient data.
        """
        min_required = self.bb_period + self.length
        if ohlc is None or ohlc.empty or "close" not in ohlc.columns:
            return None
        if len(ohlc) < min_required:
            logger.debug(
                "bbwp_insufficient_data",
                have=len(ohlc),
                need=min_required,
            )
            return None

        try:
            close = ohlc["close"].astype(float)

            # --- Step 1: Bollinger Band Width series ---
            sma = close.rolling(window=self.bb_period).mean()
            std = close.rolling(window=self.bb_period).std(ddof=0)
            # width = (4 * std) / sma — avoids storing upper/lower explicitly
            width = (4.0 * std) / sma
            # Drop NaN rows from the BB calculation period
            width = width.dropna()

            if len(width) < self.length:
                logger.debug(
                    "bbwp_insufficient_width_series",
                    have=len(width),
                    need=self.length,
                )
                return None

            # --- Step 2: Percentile of current width within last `length` values ---
            # Calculate a rolling percentile series for the last `length` observations.
            # For each bar i, BBWP[i] = percentile_rank of width[i] in width[i-length+1..i].
            bbwp_series = width.rolling(window=self.length).apply(
                _percentile_rank, raw=True
            )
            bbwp_series = bbwp_series.dropna()

            if bbwp_series.empty:
                return None

            # --- Step 3: Apply EMA smoothing ---
            bbwp_ema = bbwp_series.ewm(span=self.ma_length, adjust=False).mean()

            current_bbwp = float(bbwp_series.iloc[-1])
            current_ma = float(bbwp_ema.iloc[-1])

            # --- Step 4: Classify ---
            if current_bbwp >= BBWP_EXTREME_HIGH:
                signal = "extreme_reached"
            elif current_bbwp <= BBWP_EXTREME_LOW:
                signal = "breakout_imminent"
            else:
                signal = "neutral"

            logger.debug(
                "bbwp_calculated",
                bbwp=round(current_bbwp, 2),
                ma=round(current_ma, 2),
                signal=signal,
            )

            return BBWPState(
                bbwp_value=current_bbwp,
                signal=signal,
                ma_value=current_ma,
            )

        except Exception as e:
            logger.debug("bbwp_calculation_error", error=str(e))
            return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _percentile_rank(arr: "object") -> float:
    """Return the percentile rank (0-100) of the last element in arr.

    Used as the rolling apply function. For a window of N values,
    returns: 100 * (count of values < last) / (N - 1).

    Edge case: if all values are equal, returns 0.0 (current is not wider than any).
    """
    import numpy as np  # local to avoid heavy import at module level
    a = arr  # arr is already a numpy array when raw=True
    n = len(a)
    if n <= 1:
        return 50.0
    current = a[-1]
    count_below = float(np.sum(a[:-1] < current))
    return (count_below / (n - 1)) * 100.0
