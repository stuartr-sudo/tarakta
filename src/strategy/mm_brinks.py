"""Brinks Trade detection for the Market Makers Method.

The Brinks Trade is the HIGHEST R:R setup (6:1 to 18:1) in the entire method.
It occurs ONLY at two specific 15-minute candle close times in New York:
  - 3:30-3:45am (UK open)
  - 9:30-9:45am (US open)

Rules (Lesson 06):
  - The second leg of an M/W must form at exactly those times
  - Must be at HOD (for M/short) or LOD (for W/long)
  - Entry candle must be hammer (W) or inverted hammer (M) at 3:45 or 9:45
  - Time between first and second peak: 30-90 minutes
  - Usually 3 fast quick bursts precede the entry
  - Expect extended 3-level drop/rise following
  - Scratch rule: not in profit within 2 hours -> close
  - Works on ALL pairs (not just Bitcoin)
  - Approximately once per week across top 10-20 coins
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from src.strategy.mm_formations import _is_hammer, _is_inverted_hammer
from src.utils.logging import get_logger

logger = get_logger(__name__)

NY_TZ = ZoneInfo("America/New_York")


@dataclass
class BrinksResult:
    """Result of Brinks trade detection."""

    detected: bool
    window: str  # "uk_open" or "us_open"
    entry_price: float
    stop_loss: float
    direction: str  # "long" or "short"
    formation_type: str  # "W" or "M"
    peak1_time: datetime | None
    peak2_time: datetime | None
    quality: float  # 0-1 quality score


class BrinksDetector:
    """Detects Brinks trade setups at the two magic 15-min windows.

    The Brinks Trade triggers when:
    1. Current NY time is in a Brinks window (3:30-3:45 or 9:30-9:45)
    2. The last closed 15m candle is at HOD (short) or LOD (long)
    3. That candle is a hammer (long) or inverted hammer (short)
    4. A prior peak/trough exists 30-90 minutes back (the first leg)
    """

    # 15-min candle close times in NY timezone.
    # Each tuple: (window_start, window_end, label)
    BRINKS_WINDOWS_NY = [
        (time(3, 30), time(3, 45), "uk_open"),
        (time(9, 30), time(9, 45), "us_open"),
    ]

    MIN_PEAK_SEP_MINUTES = 30
    MAX_PEAK_SEP_MINUTES = 90

    # Within 0.3% of HOD/LOD counts as "at" that extreme.
    HOD_LOD_TOLERANCE_PCT = 0.003

    def detect(
        self,
        candles_15m: pd.DataFrame,
        hod: float,
        lod: float,
        now_ny: datetime,
    ) -> BrinksResult | None:
        """Detect a Brinks trade setup.

        Args:
            candles_15m: 15-minute OHLCV candles with columns
                ``open``, ``high``, ``low``, ``close``, ``volume``.
                Must be sorted by time ascending. Index or a ``timestamp``
                column should carry timezone-aware datetimes.
            hod: High of Day.
            lod: Low of Day.
            now_ny: Current time in New York timezone.

        Returns:
            ``BrinksResult`` if a setup is detected, ``None`` otherwise.
        """
        if candles_15m is None or candles_15m.empty or len(candles_15m) < 3:
            return None

        # 1. Check if current time is within a Brinks window.
        current_time_ny = now_ny.time()
        active_window: str | None = None
        for win_start, win_end, label in self.BRINKS_WINDOWS_NY:
            if win_start <= current_time_ny <= win_end:
                active_window = label
                break

        if active_window is None:
            return None

        # 2. Get the last closed 15m candle.
        last = candles_15m.iloc[-1]
        o, h, low, c = (
            float(last["open"]),
            float(last["high"]),
            float(last["low"]),
            float(last["close"]),
        )

        # 3. Determine if candle is at HOD or LOD within tolerance.
        at_hod = hod > 0 and abs(h - hod) / hod <= self.HOD_LOD_TOLERANCE_PCT
        at_lod = lod > 0 and abs(low - lod) / lod <= self.HOD_LOD_TOLERANCE_PCT

        if not at_hod and not at_lod:
            return None

        # 4. Check hammer / inverted hammer pattern.
        is_long_setup = at_lod and _is_hammer(o, h, low, c)
        is_short_setup = at_hod and _is_inverted_hammer(o, h, low, c)

        if not is_long_setup and not is_short_setup:
            return None

        # Resolve direction.
        if is_long_setup:
            direction = "long"
            formation_type = "W"
        else:
            direction = "short"
            formation_type = "M"

        # 5. Look back 30-90 minutes for the first peak (2-6 candles at 15m).
        min_bars = self.MIN_PEAK_SEP_MINUTES // 15  # 2
        max_bars = self.MAX_PEAK_SEP_MINUTES // 15  # 6

        lookback_end = len(candles_15m) - 1  # exclusive of the entry candle
        lookback_start = max(0, lookback_end - max_bars)

        if lookback_end - lookback_start < min_bars:
            return None

        window_slice = candles_15m.iloc[lookback_start:lookback_end]

        if window_slice.empty:
            return None

        # Find the first peak/trough in the lookback window.
        if formation_type == "W":
            # For W (long): first leg is the lowest low.
            peak1_iloc = int(window_slice["low"].values.argmin()) + lookback_start
            peak1_price = float(candles_15m.iloc[peak1_iloc]["low"])
        else:
            # For M (short): first leg is the highest high.
            peak1_iloc = int(window_slice["high"].values.argmax()) + lookback_start
            peak1_price = float(candles_15m.iloc[peak1_iloc]["high"])

        # 6. Verify peak separation is within [30, 90] minutes.
        peak2_iloc = len(candles_15m) - 1
        bar_sep = peak2_iloc - peak1_iloc
        minute_sep = bar_sep * 15

        if minute_sep < self.MIN_PEAK_SEP_MINUTES or minute_sep > self.MAX_PEAK_SEP_MINUTES:
            return None

        # 7. Calculate stop loss.
        if formation_type == "W":
            # Below the lowest wick of the entry candle or peak1, whichever is lower.
            stop_loss = min(low, peak1_price)
        else:
            # Above the highest wick of the entry candle or peak1, whichever is higher.
            stop_loss = max(h, peak1_price)

        # 8. Resolve peak timestamps (best effort).
        peak1_time = self._get_candle_time(candles_15m, peak1_iloc)
        peak2_time = self._get_candle_time(candles_15m, peak2_iloc)

        # Quality score: higher when separation is closer to 60m (ideal midpoint)
        # and candle body is decisive.
        ideal_sep = (self.MIN_PEAK_SEP_MINUTES + self.MAX_PEAK_SEP_MINUTES) / 2
        sep_quality = 1.0 - abs(minute_sep - ideal_sep) / ideal_sep
        body = abs(c - o)
        full_range = h - low
        body_ratio = body / full_range if full_range > 0 else 0
        quality = round(0.6 * sep_quality + 0.4 * (1.0 - body_ratio), 3)
        quality = max(0.0, min(1.0, quality))

        logger.info(
            "mm_brinks_detected",
            window=active_window,
            direction=direction,
            formation=formation_type,
            entry_price=c,
            stop_loss=stop_loss,
            peak_sep_min=minute_sep,
            quality=quality,
        )

        return BrinksResult(
            detected=True,
            window=active_window,
            entry_price=c,
            stop_loss=stop_loss,
            direction=direction,
            formation_type=formation_type,
            peak1_time=peak1_time,
            peak2_time=peak2_time,
            quality=quality,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_candle_time(df: pd.DataFrame, iloc_idx: int) -> datetime | None:
        """Extract a timezone-aware datetime for a candle by iloc index."""
        if "timestamp" in df.columns:
            ts = df.iloc[iloc_idx]["timestamp"]
            if isinstance(ts, pd.Timestamp):
                return ts.to_pydatetime()
            return ts
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index[iloc_idx]
            if isinstance(ts, pd.Timestamp):
                return ts.to_pydatetime()
            return ts
        return None
