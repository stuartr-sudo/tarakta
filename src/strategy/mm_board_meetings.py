"""Board Meeting detection and Fibonacci retracement entries.

Board Meetings are consolidation periods between levels in the MM weekly cycle.
After each level completes, the Market Maker pauses to accumulate contracts
before the next push. These consolidation zones provide entry opportunities.

Two types of Board Meeting entries:

1. **Retracement Board Meeting**: Price retraces to Fibonacci levels
   (38.2%, 50%, 61.8%) of the prior level. Stagger orders across these
   levels. SL above the peak of the prior level.

2. **Sideways Board Meeting**: Price goes sideways, forming an M or W
   inside the consolidation. The stop hunt comes at the END of the board
   meeting (not the beginning). Reduced M/W criteria — just the shape
   is sufficient.

Fibonacci levels are also used for target setting in board meeting entries
and for identifying where retracements are likely to stall.

Key rules:
- Board Meeting after Level 1: expect retrace to 50 EMA
- Board Meeting after Level 2: expect retrace to 50 EMA again
- If retrace only reaches 10 EMA (not 50) and counter to macro trend → caution
- Stop hunt at END of board meeting signals the next level is about to begin
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fibonacci retracement levels (from course: 38.2%, 50%, 61.8%)
FIB_LEVELS = (0.382, 0.5, 0.618)

# How many candles of sideways to consider a board meeting
MIN_BOARD_MEETING_CANDLES = 4

# Maximum range (as % of prior level move) for sideways board meeting
MAX_SIDEWAYS_RANGE_PCT = 40.0

# Minimum candles for the prior level move
MIN_LEVEL_CANDLES = 3


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FibLevel:
    """A single Fibonacci retracement level."""
    ratio: float = 0.0     # 0.382, 0.5, or 0.618
    price: float = 0.0     # Actual price at this fib level
    label: str = ""        # Human label ("38.2%", "50%", "61.8%")
    touched: bool = False  # Has price reached this level?
    held: bool = False     # Did price bounce from this level?


@dataclass
class FibonacciRetracement:
    """Fibonacci retracement analysis for a move."""
    swing_high: float = 0.0
    swing_low: float = 0.0
    direction: str = ""        # "bullish" (low to high) or "bearish" (high to low)
    levels: list[FibLevel] = field(default_factory=list)
    deepest_retrace_pct: float = 0.0  # How deep did the retrace go (% of move)
    active_level: FibLevel | None = None  # Currently closest untouched level


@dataclass
class BoardMeetingEntry:
    """A potential entry from a board meeting consolidation.

    Contains the entry type, price levels, and Fibonacci analysis.
    """
    entry_type: str = ""          # "retracement" or "sideways"
    entry_price: float = 0.0      # Suggested entry price
    stop_loss: float = 0.0        # SL above prior level peak
    target: float = 0.0           # Target for next level
    risk_reward: float = 0.0      # R:R ratio
    fib: FibonacciRetracement | None = None  # Fibonacci analysis
    level_before: int = 0         # Which level preceded this BM (1, 2, or 3)
    confidence: float = 0.0       # 0-1 confidence


@dataclass
class BoardMeetingDetection:
    """Result of board meeting detection in price data."""
    detected: bool = False
    bm_type: str = ""             # "retracement" or "sideways"
    start_idx: int = -1           # Index in DataFrame where BM starts
    end_idx: int = -1             # Index where BM ends (-1 if ongoing)
    duration_candles: int = 0
    range_pct: float = 0.0       # Range of the BM as % of price
    # Stop hunt detection
    stop_hunt_detected: bool = False
    stop_hunt_direction: str = ""  # "up" or "down"
    stop_hunt_idx: int = -1       # Index of the stop hunt candle
    # Fibonacci
    fib: FibonacciRetracement | None = None
    # Entry suggestion
    entry: BoardMeetingEntry | None = None


# ---------------------------------------------------------------------------
# Fibonacci Calculator
# ---------------------------------------------------------------------------

class FibonacciCalculator:
    """Calculate and analyze Fibonacci retracement levels."""

    @staticmethod
    def calculate_retracement(
        swing_high: float,
        swing_low: float,
        direction: str = "bullish",
    ) -> FibonacciRetracement:
        """Calculate Fibonacci retracement levels for a swing.

        For a bullish swing (move up): retracement levels are below the high.
        For a bearish swing (move down): retracement levels are above the low.

        Args:
            swing_high: High of the swing.
            swing_low: Low of the swing.
            direction: "bullish" (retracing a move up) or "bearish" (retracing a move down).

        Returns:
            FibonacciRetracement with calculated levels.
        """
        move = swing_high - swing_low
        if move <= 0:
            return FibonacciRetracement(
                swing_high=swing_high,
                swing_low=swing_low,
                direction=direction,
            )

        levels = []
        for ratio in FIB_LEVELS:
            if direction == "bullish":
                # Retracing a move up: levels are below the high
                price = swing_high - (move * ratio)
            else:
                # Retracing a move down: levels are above the low
                price = swing_low + (move * ratio)

            levels.append(FibLevel(
                ratio=ratio,
                price=round(price, 8),
                label=f"{ratio * 100:.1f}%",
            ))

        return FibonacciRetracement(
            swing_high=swing_high,
            swing_low=swing_low,
            direction=direction,
            levels=levels,
        )

    @staticmethod
    def check_retracement_depth(
        fib: FibonacciRetracement,
        current_price: float,
    ) -> FibonacciRetracement:
        """Update Fibonacci levels with current price touch/hold status.

        Args:
            fib: Existing FibonacciRetracement to update.
            current_price: Current market price.

        Returns:
            Updated FibonacciRetracement.
        """
        if not fib.levels:
            return fib

        move = fib.swing_high - fib.swing_low
        if move <= 0:
            return fib

        # Calculate how deep the retrace has gone
        if fib.direction == "bullish":
            retrace = fib.swing_high - current_price
        else:
            retrace = current_price - fib.swing_low

        fib.deepest_retrace_pct = round((retrace / move) * 100, 2) if move > 0 else 0

        # Update touch status for each level
        active = None
        for level in fib.levels:
            if fib.direction == "bullish":
                level.touched = current_price <= level.price
            else:
                level.touched = current_price >= level.price

            # Track the closest untouched level as "active"
            if not level.touched and active is None:
                active = level

        fib.active_level = active
        return fib

    @staticmethod
    def get_entry_prices(
        fib: FibonacciRetracement,
    ) -> list[tuple[float, str]]:
        """Get suggested staggered entry prices from Fibonacci levels.

        Board Meeting entries should be staggered across the 3 fib levels
        (38.2%, 50%, 61.8%) with equal position size at each.

        Returns:
            List of (price, label) tuples for entry placement.
        """
        return [(level.price, level.label) for level in fib.levels]


# ---------------------------------------------------------------------------
# Board Meeting Detector
# ---------------------------------------------------------------------------

class BoardMeetingDetector:
    """Detects board meeting consolidation periods between levels.

    A board meeting is identified by:
    1. A significant move (level) followed by...
    2. A period of reduced volatility (consolidation) that is either:
       a. A retracement to Fibonacci levels, or
       b. A sideways range with an M/W shape
    """

    def __init__(self) -> None:
        self.fib_calc = FibonacciCalculator()

    def detect(
        self,
        ohlc: pd.DataFrame,
        level_direction: str = "bullish",
        level_start_idx: int = 0,
        level_end_idx: int = -1,
    ) -> BoardMeetingDetection:
        """Detect a board meeting following a completed level.

        Args:
            ohlc: OHLCV DataFrame (1H recommended).
            level_direction: Direction of the prior level ("bullish" or "bearish").
            level_start_idx: Index where the prior level started.
            level_end_idx: Index where the prior level ended. -1 = auto-detect.

        Returns:
            BoardMeetingDetection with type, Fibonacci analysis, and entry suggestion.
        """
        if ohlc is None or ohlc.empty or len(ohlc) < MIN_LEVEL_CANDLES + MIN_BOARD_MEETING_CANDLES:
            return BoardMeetingDetection()

        n = len(ohlc)

        # Auto-detect level end if not specified
        if level_end_idx < 0:
            level_end_idx = self._find_level_end(ohlc, level_direction, level_start_idx)
            if level_end_idx < 0:
                return BoardMeetingDetection()

        # Get the level move boundaries
        level_candles = ohlc.iloc[level_start_idx:level_end_idx + 1]
        if level_candles.empty:
            return BoardMeetingDetection()

        if level_direction == "bullish":
            swing_low = float(level_candles["low"].min())
            swing_high = float(level_candles["high"].max())
        else:
            swing_high = float(level_candles["high"].max())
            swing_low = float(level_candles["low"].min())

        level_move = swing_high - swing_low
        if level_move <= 0:
            return BoardMeetingDetection()

        # Get the post-level candles (board meeting zone)
        bm_start = level_end_idx + 1
        if bm_start >= n:
            return BoardMeetingDetection()

        bm_candles = ohlc.iloc[bm_start:]
        if len(bm_candles) < MIN_BOARD_MEETING_CANDLES:
            return BoardMeetingDetection()

        # Calculate Fibonacci retracement
        fib = self.fib_calc.calculate_retracement(swing_high, swing_low, level_direction)

        # Check current price against fib levels
        current_price = float(bm_candles.iloc[-1]["close"])
        fib = self.fib_calc.check_retracement_depth(fib, current_price)

        # Determine board meeting type
        bm_high = float(bm_candles["high"].max())
        bm_low = float(bm_candles["low"].min())
        bm_range = bm_high - bm_low
        bm_mid = (bm_high + bm_low) / 2
        bm_range_pct = (bm_range / bm_mid * 100) if bm_mid > 0 else 0

        # Is this a retracement or sideways?
        if fib.deepest_retrace_pct >= 25:
            bm_type = "retracement"
        else:
            bm_type = "sideways"

        # Detect stop hunt at END of board meeting
        stop_hunt = self._detect_bm_stop_hunt(bm_candles, level_direction, bm_high, bm_low)

        # Build entry suggestion
        entry = self._build_entry(
            bm_type=bm_type,
            fib=fib,
            level_direction=level_direction,
            swing_high=swing_high,
            swing_low=swing_low,
            current_price=current_price,
            level_move=level_move,
        )

        result = BoardMeetingDetection(
            detected=True,
            bm_type=bm_type,
            start_idx=bm_start,
            end_idx=-1,  # Ongoing until next level starts
            duration_candles=len(bm_candles),
            range_pct=round(bm_range_pct, 4),
            stop_hunt_detected=stop_hunt[0],
            stop_hunt_direction=stop_hunt[1],
            stop_hunt_idx=stop_hunt[2],
            fib=fib,
            entry=entry,
        )

        logger.info(
            "board_meeting_detected",
            bm_type=bm_type,
            range_pct=round(bm_range_pct, 4),
            retrace_depth=fib.deepest_retrace_pct,
            stop_hunt=stop_hunt[0],
            fib_levels=[f"{l.label}={l.price:.4f}" for l in fib.levels],
        )

        return result

    def _find_level_end(
        self,
        ohlc: pd.DataFrame,
        direction: str,
        start_idx: int,
    ) -> int:
        """Auto-detect where a level move ends by finding the extreme.

        For bullish: find the highest high after start_idx.
        For bearish: find the lowest low after start_idx.
        """
        if start_idx >= len(ohlc):
            return -1

        subset = ohlc.iloc[start_idx:]
        if direction == "bullish":
            return int(subset["high"].idxmax()) if hasattr(subset["high"].idxmax(), '__int__') else start_idx + int(subset["high"].values.argmax())
        else:
            return int(subset["low"].idxmin()) if hasattr(subset["low"].idxmin(), '__int__') else start_idx + int(subset["low"].values.argmin())

    def _detect_bm_stop_hunt(
        self,
        bm_candles: pd.DataFrame,
        level_direction: str,
        bm_high: float,
        bm_low: float,
    ) -> tuple[bool, str, int]:
        """Detect a stop hunt at the end of a board meeting.

        The stop hunt comes at the END of the BM (not beginning) and signals
        the next level is about to start.

        Returns:
            (detected, direction, candle_index)
        """
        if len(bm_candles) < 3:
            return (False, "", -1)

        # Look at the last few candles for a spike-and-reject pattern
        last_candles = bm_candles.iloc[-3:]
        bm_range = bm_high - bm_low
        if bm_range <= 0:
            return (False, "", -1)

        for i in range(len(last_candles)):
            candle = last_candles.iloc[i]
            body = abs(float(candle["close"]) - float(candle["open"]))
            high = float(candle["high"])
            low = float(candle["low"])
            full_range = high - low

            if full_range <= 0:
                continue

            body_ratio = body / full_range

            # Stop hunt = small body with large wick extending outside the BM range
            if body_ratio < 0.35:
                # Check for wick extension
                if level_direction == "bullish":
                    # After bullish level, BM stop hunt goes DOWN (traps shorts)
                    wick_below = float(candle["open" if candle["close"] > candle["open"] else "close"]) - low
                    if wick_below > bm_range * 0.5:
                        idx = bm_candles.index[-(3 - i)] if hasattr(bm_candles.index, '__getitem__') else -1
                        return (True, "down", int(idx) if isinstance(idx, (int, np.integer)) else -1)
                else:
                    # After bearish level, BM stop hunt goes UP (traps longs)
                    wick_above = high - float(candle["open" if candle["close"] < candle["open"] else "close"])
                    if wick_above > bm_range * 0.5:
                        idx = bm_candles.index[-(3 - i)] if hasattr(bm_candles.index, '__getitem__') else -1
                        return (True, "up", int(idx) if isinstance(idx, (int, np.integer)) else -1)

        return (False, "", -1)

    def _build_entry(
        self,
        bm_type: str,
        fib: FibonacciRetracement,
        level_direction: str,
        swing_high: float,
        swing_low: float,
        current_price: float,
        level_move: float,
    ) -> BoardMeetingEntry:
        """Build an entry suggestion from the board meeting analysis.

        Retracement BM: stagger entries at fib levels.
        Sideways BM: entry on M/W shape (or at current price after stop hunt).
        """
        if bm_type == "retracement":
            # Use the 50% fib level as the primary entry
            fib_50 = next((l for l in fib.levels if l.ratio == 0.5), None)
            entry_price = fib_50.price if fib_50 else current_price
        else:
            # Sideways: enter at current price (ideally after stop hunt)
            entry_price = current_price

        # SL: above/below the peak of the prior level
        # Add a small buffer (0.1% of the move)
        buffer = level_move * 0.001
        if level_direction == "bullish":
            # Bullish level went up, BM retraces down, next level goes up again
            stop_loss = swing_low - buffer  # Below the level's start
            target = swing_high + level_move  # Project next level as same size
        else:
            # Bearish level went down, BM retraces up, next level goes down again
            stop_loss = swing_high + buffer  # Above the level's start
            target = swing_low - level_move  # Project next level

        # Calculate R:R
        risk = abs(entry_price - stop_loss)
        reward = abs(target - entry_price)
        rr = round(reward / risk, 2) if risk > 0 else 0.0

        # Confidence based on fib alignment and board meeting characteristics
        confidence = 0.4  # Base
        if fib.deepest_retrace_pct >= 35 and fib.deepest_retrace_pct <= 65:
            confidence += 0.2  # Healthy retrace to fib zone
        if rr >= 3.0:
            confidence += 0.15  # Good R:R
        if rr >= 5.0:
            confidence += 0.1  # Excellent R:R

        return BoardMeetingEntry(
            entry_type=bm_type,
            entry_price=round(entry_price, 8),
            stop_loss=round(stop_loss, 8),
            target=round(target, 8),
            risk_reward=rr,
            fib=fib,
            confidence=min(round(confidence, 2), 1.0),
        )

    def get_staggered_entries(
        self,
        fib: FibonacciRetracement,
        total_position_size: float,
    ) -> list[dict]:
        """Calculate staggered entry orders across Fibonacci levels.

        Splits the total position size equally across the 3 fib levels.

        Args:
            fib: Fibonacci retracement analysis.
            total_position_size: Total intended position size in USD.

        Returns:
            List of dicts with price, size, and label for each order.
        """
        if not fib.levels:
            return []

        per_level_size = total_position_size / len(fib.levels)
        orders = []
        for level in fib.levels:
            orders.append({
                "price": level.price,
                "size_usd": round(per_level_size, 2),
                "label": f"Fib {level.label}",
                "ratio": level.ratio,
            })

        return orders
