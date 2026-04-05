"""Weekend Trap Box detection and FMWB (False Move Week Beginning) identification.

The Weekend Trap is the first phase of the MM weekly cycle. From Friday 5pm NY
(market close) through to Sunday ~5pm NY, Market Makers create a sideways box
that traps weekend traders. Near the end, a spike (stop hunt) clears liquidity
in one direction, then the FMWB begins.

Key concepts:
- Weekend Trap Box = the high/low range of candle *closes* (not wicks) from
  Friday US close through Sunday dead zone. Wicks represent the traps.
- FMWB = the aggressive move at/near the open of the new week (Sun 5pm NY for
  crypto, UK/US open Mon for forex). This is the FALSE direction.
- The trap direction is identified by which side of the box the wicks extend
  more — that tells us where stops were hunted.
- After the FMWB, the REAL move begins in the opposite direction.

Marking process (from the course):
1. Mark from US Friday close candle (last 1H candle before 5pm NY Friday)
2. Weekend = everything from next candle through Dead Zone (5pm NY Sunday)
3. Mark candle close extremes (not wicks) as the box boundaries
4. Wicks outside the box = trap zones (who got stopped out)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

NY_TZ = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WeekendTrapBox:
    """The Weekend Trap Box boundaries and analysis.

    The box is defined by the range of candle CLOSES during the weekend
    period. Wicks outside the box represent stop hunts / traps.
    """
    # Box boundaries (candle closes)
    box_high: float = 0.0          # Highest candle close in the weekend
    box_low: float = float("inf")  # Lowest candle close in the weekend
    box_range_pct: float = 0.0     # Box range as % of midpoint

    # Wick extremes (including wicks — the actual traps)
    wick_high: float = 0.0         # Highest wick in the weekend
    wick_low: float = float("inf") # Lowest wick in the weekend

    # Trap analysis
    trap_above_pct: float = 0.0    # How far wicks extend above box (% of box range)
    trap_below_pct: float = 0.0    # How far wicks extend below box (% of box range)
    primary_trap_direction: str = ""  # "long" (stopped longs) or "short" (stopped shorts)

    # Timing
    box_start: datetime | None = None  # Friday 5pm NY
    box_end: datetime | None = None    # Sunday 5pm NY
    num_candles: int = 0               # Number of 1H candles in the box

    # Metadata
    detected: bool = False


@dataclass
class FMWBResult:
    """Result of FMWB (False Move Week Beginning) detection.

    FMWB is the aggressive trap move that happens at/near the weekly open.
    For crypto: Sunday ~5pm NY. For forex: UK/US open Monday.
    """
    detected: bool = False
    direction: str = ""            # Direction of the FALSE move ("up" or "down")
    real_direction: str = ""       # Expected REAL direction (opposite of FMWB)
    magnitude_pct: float = 0.0    # Size of the FMWB move as % of price
    peak_price: float = 0.0       # Extreme price reached by FMWB
    started_at: datetime | None = None
    ended_at: datetime | None = None
    broke_box: bool = False        # Did FMWB break outside the weekend trap box?
    broke_box_side: str = ""       # "above" or "below" if broke_box=True


@dataclass
class WeekendAnalysis:
    """Complete weekend analysis combining trap box and FMWB."""
    trap_box: WeekendTrapBox = field(default_factory=WeekendTrapBox)
    fmwb: FMWBResult = field(default_factory=FMWBResult)

    # Context for agents
    bias: str = ""                 # Expected real move direction after FMWB
    confidence: float = 0.0        # 0-1 confidence in the analysis


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many hours after Sunday 5pm NY to look for the FMWB
FMWB_WINDOW_HOURS = 8  # Look for FMWB within 8 hours of weekly open

# Minimum % move to qualify as FMWB (relative to weekend box range)
FMWB_MIN_MOVE_PCT = 0.3  # FMWB must move at least 30% of weekend box range

# Minimum box range to consider meaningful (very tight boxes = no trap)
MIN_BOX_RANGE_PCT = 0.001  # 0.1% minimum


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class WeekendTrapAnalyzer:
    """Detects and analyzes the Weekend Trap Box and FMWB.

    Usage:
        analyzer = WeekendTrapAnalyzer()
        result = analyzer.analyze(ohlc_1h, current_time)
    """

    def detect_trap_box(
        self,
        ohlc_1h: pd.DataFrame,
        current_time: datetime,
    ) -> WeekendTrapBox:
        """Detect the Weekend Trap Box from 1H candle data.

        Looks back from current_time to find the most recent Friday 5pm NY →
        Sunday 5pm NY window and calculates box boundaries.

        Args:
            ohlc_1h: 1H OHLCV DataFrame with DatetimeIndex (UTC expected).
            current_time: Current wall-clock time (tz-aware).

        Returns:
            WeekendTrapBox with boundaries and trap analysis.
        """
        if ohlc_1h is None or ohlc_1h.empty:
            return WeekendTrapBox()

        ny_now = _to_ny(current_time)

        # Find most recent Friday 5pm NY
        friday_5pm = _last_friday_5pm(ny_now)
        sunday_5pm = friday_5pm + timedelta(days=2)  # Sunday 5pm NY

        # Convert to UTC for DataFrame filtering
        friday_utc = friday_5pm.astimezone(ZoneInfo("UTC"))
        sunday_utc = sunday_5pm.astimezone(ZoneInfo("UTC"))

        # Filter candles within the weekend window
        idx = ohlc_1h.index
        if idx.tz is None:
            # Assume UTC if naive
            friday_utc = friday_utc.replace(tzinfo=None)
            sunday_utc = sunday_utc.replace(tzinfo=None)

        weekend_candles = ohlc_1h[(idx >= friday_utc) & (idx <= sunday_utc)]

        if weekend_candles.empty:
            return WeekendTrapBox()

        # Box boundaries: CLOSES only (not wicks)
        closes = weekend_candles["close"]
        box_high = float(closes.max())
        box_low = float(closes.min())
        box_mid = (box_high + box_low) / 2

        box_range_pct = ((box_high - box_low) / box_mid * 100) if box_mid > 0 else 0.0

        # Wick extremes (actual trap extent)
        wick_high = float(weekend_candles["high"].max())
        wick_low = float(weekend_candles["low"].min())

        # Trap analysis: how far do wicks extend beyond the close box?
        box_range = box_high - box_low
        if box_range > 0:
            trap_above = max(0, wick_high - box_high)
            trap_below = max(0, box_low - wick_low)
            trap_above_pct = (trap_above / box_range) * 100
            trap_below_pct = (trap_below / box_range) * 100
        else:
            trap_above_pct = 0.0
            trap_below_pct = 0.0

        # Primary trap direction: which side has more wick extension?
        # Bigger wicks above = longs got trapped (stopped), so shorts are the real move
        # Bigger wicks below = shorts got trapped (stopped), so longs are the real move
        if trap_above_pct > trap_below_pct * 1.3:
            primary_trap = "long"   # Longs got trapped → expect bearish
        elif trap_below_pct > trap_above_pct * 1.3:
            primary_trap = "short"  # Shorts got trapped → expect bullish
        else:
            primary_trap = "neutral"

        result = WeekendTrapBox(
            box_high=box_high,
            box_low=box_low,
            box_range_pct=round(box_range_pct, 4),
            wick_high=wick_high,
            wick_low=wick_low,
            trap_above_pct=round(trap_above_pct, 2),
            trap_below_pct=round(trap_below_pct, 2),
            primary_trap_direction=primary_trap,
            box_start=friday_5pm,
            box_end=sunday_5pm,
            num_candles=len(weekend_candles),
            detected=box_range_pct >= MIN_BOX_RANGE_PCT,
        )

        logger.info(
            "weekend_trap_box_detected",
            box_high=box_high,
            box_low=box_low,
            range_pct=round(box_range_pct, 4),
            trap_above_pct=round(trap_above_pct, 2),
            trap_below_pct=round(trap_below_pct, 2),
            primary_trap=primary_trap,
            num_candles=len(weekend_candles),
        )

        return result

    def detect_fmwb(
        self,
        ohlc_1h: pd.DataFrame,
        current_time: datetime,
        trap_box: WeekendTrapBox | None = None,
    ) -> FMWBResult:
        """Detect the FMWB (False Move Week Beginning) after the weekend trap.

        Looks for an aggressive move in the first FMWB_WINDOW_HOURS hours
        after the weekly open (Sunday 5pm NY for crypto).

        Args:
            ohlc_1h: 1H OHLCV DataFrame with DatetimeIndex.
            current_time: Current wall-clock time (tz-aware).
            trap_box: Optional pre-computed trap box (will compute if None).

        Returns:
            FMWBResult with FMWB detection and direction.
        """
        if ohlc_1h is None or ohlc_1h.empty:
            return FMWBResult()

        ny_now = _to_ny(current_time)

        # Find most recent Sunday 5pm NY (weekly open)
        friday_5pm = _last_friday_5pm(ny_now)
        sunday_5pm = friday_5pm + timedelta(days=2)

        # Only look for FMWB if we're past the weekly open
        if ny_now < sunday_5pm:
            return FMWBResult()

        # FMWB window: Sunday 5pm → Sunday 5pm + FMWB_WINDOW_HOURS
        fmwb_end = sunday_5pm + timedelta(hours=FMWB_WINDOW_HOURS)

        # Don't try to detect if we're past the window and it should already be done
        # (allow detection if within window or up to 24h after)
        if ny_now > fmwb_end + timedelta(hours=24):
            # More than 24h past FMWB window — data may be stale, but still try
            pass

        # Convert to UTC for filtering
        sunday_utc = sunday_5pm.astimezone(ZoneInfo("UTC"))
        fmwb_end_utc = fmwb_end.astimezone(ZoneInfo("UTC"))

        idx = ohlc_1h.index
        if idx.tz is None:
            sunday_utc = sunday_utc.replace(tzinfo=None)
            fmwb_end_utc = fmwb_end_utc.replace(tzinfo=None)

        # Get the candle right before the weekly open (reference price)
        pre_open = ohlc_1h[idx <= sunday_utc]
        if pre_open.empty:
            return FMWBResult()
        ref_price = float(pre_open.iloc[-1]["close"])

        # Get candles in the FMWB window
        fmwb_candles = ohlc_1h[(idx > sunday_utc) & (idx <= fmwb_end_utc)]
        if fmwb_candles.empty:
            return FMWBResult()

        # Find the peak move in the FMWB window
        high_in_window = float(fmwb_candles["high"].max())
        low_in_window = float(fmwb_candles["low"].min())

        move_up = (high_in_window - ref_price) / ref_price * 100 if ref_price > 0 else 0
        move_down = (ref_price - low_in_window) / ref_price * 100 if ref_price > 0 else 0

        # The FMWB is the LARGER move (the aggressive false move)
        if move_up > move_down:
            fmwb_direction = "up"
            real_direction = "bearish"
            magnitude = move_up
            peak_price = high_in_window
            # Find when the peak occurred
            peak_idx = fmwb_candles["high"].idxmax()
        else:
            fmwb_direction = "down"
            real_direction = "bullish"
            magnitude = move_down
            peak_price = low_in_window
            peak_idx = fmwb_candles["low"].idxmin()

        # Check if FMWB broke outside the weekend trap box
        broke_box = False
        broke_side = ""
        if trap_box and trap_box.detected:
            if fmwb_direction == "up" and peak_price > trap_box.wick_high:
                broke_box = True
                broke_side = "above"
            elif fmwb_direction == "down" and peak_price < trap_box.wick_low:
                broke_box = True
                broke_side = "below"

        # Minimum magnitude check
        if trap_box and trap_box.detected and trap_box.box_range_pct > 0:
            # Relative to box range
            relative_move = magnitude / trap_box.box_range_pct
            is_significant = relative_move >= FMWB_MIN_MOVE_PCT
        else:
            # Absolute check: at least 0.3% move
            is_significant = magnitude >= 0.3

        if not is_significant:
            return FMWBResult()

        result = FMWBResult(
            detected=True,
            direction=fmwb_direction,
            real_direction=real_direction,
            magnitude_pct=round(magnitude, 4),
            peak_price=peak_price,
            started_at=sunday_5pm,
            ended_at=peak_idx if isinstance(peak_idx, datetime) else None,
            broke_box=broke_box,
            broke_box_side=broke_side,
        )

        logger.info(
            "fmwb_detected",
            direction=fmwb_direction,
            real_direction=real_direction,
            magnitude_pct=round(magnitude, 4),
            peak_price=peak_price,
            broke_box=broke_box,
        )

        return result

    def analyze(
        self,
        ohlc_1h: pd.DataFrame,
        current_time: datetime,
    ) -> WeekendAnalysis:
        """Run full weekend analysis: trap box + FMWB detection.

        Args:
            ohlc_1h: 1H OHLCV DataFrame with DatetimeIndex.
            current_time: Current wall-clock time (tz-aware).

        Returns:
            WeekendAnalysis with trap box, FMWB, and bias.
        """
        trap_box = self.detect_trap_box(ohlc_1h, current_time)
        fmwb = self.detect_fmwb(ohlc_1h, current_time, trap_box)

        # Determine overall bias
        bias = ""
        confidence = 0.0

        if fmwb.detected:
            bias = fmwb.real_direction
            confidence = 0.6
            # Higher confidence if FMWB broke the box
            if fmwb.broke_box:
                confidence = 0.75
            # Trap direction aligns with FMWB → even higher confidence
            if trap_box.detected and trap_box.primary_trap_direction:
                if (trap_box.primary_trap_direction == "long" and fmwb.direction == "up") or \
                   (trap_box.primary_trap_direction == "short" and fmwb.direction == "down"):
                    confidence = min(confidence + 0.15, 0.9)
        elif trap_box.detected and trap_box.primary_trap_direction in ("long", "short"):
            # Even without FMWB, the trap box gives directional bias
            if trap_box.primary_trap_direction == "long":
                bias = "bearish"
            else:
                bias = "bullish"
            confidence = 0.3

        return WeekendAnalysis(
            trap_box=trap_box,
            fmwb=fmwb,
            bias=bias,
            confidence=round(confidence, 2),
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _to_ny(dt: datetime) -> datetime:
    """Convert any datetime to New York timezone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(NY_TZ)


def _last_friday_5pm(ny_time: datetime) -> datetime:
    """Find the most recent Friday 5:00 PM NY at or before ny_time.

    If ny_time IS Friday and past 5pm, returns that Friday 5pm.
    Otherwise walks back to the previous Friday.
    """
    # Start from the current date in NY
    d = ny_time.date()
    day_of_week = d.weekday()  # Mon=0, Fri=4

    if day_of_week == 4:  # Friday
        friday_5pm = datetime(d.year, d.month, d.day, 17, 0, tzinfo=NY_TZ)
        if ny_time >= friday_5pm:
            return friday_5pm
        # Before 5pm Friday → go to previous Friday
        d -= timedelta(days=7)
    elif day_of_week == 5:  # Saturday
        d -= timedelta(days=1)  # Go back to Friday
    elif day_of_week == 6:  # Sunday
        d -= timedelta(days=2)  # Go back to Friday
    else:
        # Mon-Thu: go back to previous Friday
        d -= timedelta(days=day_of_week + 3)

    return datetime(d.year, d.month, d.day, 17, 0, tzinfo=NY_TZ)
