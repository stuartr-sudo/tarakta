"""Market Makers Method ADR Indicator (C3).

Implements Average Daily Range (ADR) analysis for the MM Method confluence
scoring system.

ADR is used as an optional confirming factor (C3) — it contributes 4 points
to the confluence score when price is near the 50% ADR line.

Course rules (Lesson 14):
  - 14-day ADR measures how far price typically travels each day.
  - The 50% ADR line = current day's low + 0.5 * ADR.
  - Price near the 50% line = "cheap" for longs, "expensive" for shorts.
  - Used mainly for scalping targets and entries at discounted/premium zones.
  - Confluence with EMAs at the 50% line strengthens the signal.

Note: the instructor mentions she doesn't personally use ADR much — it's
more of a Forex tool. This is a low-weight (4 pt) confirming factor.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ADR_PERIOD: int = 14

# Price is considered "at" the 50% line when within this percentage
AT_FIFTY_PCT_TOLERANCE: float = 0.003  # 0.3%


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ADRState:
    """Snapshot of ADR state derived from recent OHLC data.

    All fields are computed from 1H candles via ADRAnalyzer.calculate().
    """

    adr_value: float        # Average daily range in price units
    adr_pct: float          # ADR as percentage of current price (0.0–1.0)
    fifty_pct_line: float   # current day's low + 0.5 * ADR (midpoint of daily range)
    at_fifty_pct: bool      # Price within 0.3% of the 50% line


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
class ADRAnalyzer:
    """Computes ADR state for the MM Method confluence scoring system.

    Calculates the 14-day Average Daily Range from 1H candles by resampling
    to daily and averaging the last ``period`` daily high-low ranges.

    Usage::

        analyzer = ADRAnalyzer(period=14)
        state = analyzer.calculate(ohlc_1h, current_price=95000.0)
        if state and state.at_fifty_pct:
            # Price is at the ADR 50% line — potential confluence
    """

    def __init__(self, period: int = DEFAULT_ADR_PERIOD) -> None:
        self.period = period

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, ohlc_1h: pd.DataFrame, current_price: float) -> ADRState | None:
        """Calculate ADR state from 1H OHLC candles.

        Steps:
          1. Resample 1H candles to daily (midnight UTC boundary).
          2. Calculate daily range = high - low for each day.
          3. ADR = mean of last ``period`` complete daily ranges.
          4. 50% line = current day's low + 0.5 * ADR.
          5. at_fifty_pct = abs(current_price - fifty_pct_line) / current_price < 0.003

        Requires at least ``period + 1`` daily candles (to ensure ``period``
        complete days plus a partial current day for the low).

        Args:
            ohlc_1h:       DataFrame with open/high/low/close columns and a
                           DatetimeIndex (or a timestamp column). At least
                           (period + 1) * 24 rows gives best accuracy.
            current_price: Current mid-price used for proximity and pct calculations.

        Returns:
            ADRState if sufficient data exists, otherwise None.
        """
        if ohlc_1h is None or ohlc_1h.empty:
            return None

        if current_price <= 0:
            logger.debug("adr_invalid_price", price=current_price)
            return None

        daily = self._resample_to_daily(ohlc_1h)
        if daily is None or len(daily) < self.period + 1:
            logger.debug(
                "adr_insufficient_data",
                daily_rows=0 if daily is None else len(daily),
                required=self.period + 1,
            )
            return None

        # Use the last ``period`` complete days (exclude the current partial day)
        complete_days = daily.iloc[-(self.period + 1):-1]
        if len(complete_days) < self.period:
            logger.debug(
                "adr_insufficient_complete_days",
                complete_days=len(complete_days),
                required=self.period,
            )
            return None

        daily_ranges = (complete_days["high"] - complete_days["low"]).astype(float)
        adr_value = float(daily_ranges.mean())

        # Current day's low (the last/partial day)
        current_day_low = float(daily.iloc[-1]["low"])

        fifty_pct_line = current_day_low + 0.5 * adr_value

        adr_pct = adr_value / current_price if current_price > 0 else 0.0

        proximity = abs(current_price - fifty_pct_line) / current_price
        at_fifty_pct = proximity < AT_FIFTY_PCT_TOLERANCE

        state = ADRState(
            adr_value=round(adr_value, 8),
            adr_pct=round(adr_pct, 6),
            fifty_pct_line=round(fifty_pct_line, 8),
            at_fifty_pct=at_fifty_pct,
        )

        logger.debug(
            "adr_state",
            adr=state.adr_value,
            adr_pct_pct=round(adr_pct * 100, 2),
            fifty_pct_line=state.fifty_pct_line,
            current_price=current_price,
            proximity_pct=round(proximity * 100, 3),
            at_fifty_pct=state.at_fifty_pct,
        )

        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resample_to_daily(self, ohlc_1h: pd.DataFrame) -> pd.DataFrame | None:
        """Resample 1H OHLC candles to daily candles (midnight UTC boundary).

        Handles both DatetimeIndex DataFrames and DataFrames with a
        'timestamp' column (unix ms or datetime-like).

        Args:
            ohlc_1h: 1H OHLCV DataFrame.

        Returns:
            Daily OHLCV DataFrame, or None on failure.
        """
        try:
            df = ohlc_1h.copy()

            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
                    if ts.isna().all():
                        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                    df.index = ts
                    df = df.drop(columns=["timestamp"], errors="ignore")
                else:
                    # Try converting the existing index
                    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

            df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            df = df.sort_index()

            daily = df.resample("1D").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }).dropna(subset=["high", "low"])

            return daily if not daily.empty else None

        except Exception as exc:
            logger.debug("adr_resample_error", error=str(exc))
            return None
