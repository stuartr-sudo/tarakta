"""Adaptive confidence threshold based on recent trading performance.

Dynamically adjusts the entry threshold:
- Raises it during losing streaks (become more selective)
- Lowers it during winning streaks (capitalize on momentum)
- Stays within configurable bounds to prevent runaway drift
"""
from __future__ import annotations

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Bounds — never go below or above these
MIN_THRESHOLD = 55.0
MAX_THRESHOLD = 85.0

# How many recent trades to consider
LOOKBACK_WINDOW = 20

# How much to adjust per recalculation
STEP_UP = 2.0    # Raise threshold by this when losing
STEP_DOWN = 1.5  # Lower threshold by this when winning

# Win rate targets
TARGET_WIN_RATE = 0.50  # 50% — at this rate, threshold stays unchanged
WINNING_THRESHOLD = 0.55  # Above this, lower threshold (we're doing well)
LOSING_THRESHOLD = 0.40   # Below this, raise threshold (too many losers)


class AdaptiveThreshold:
    """Dynamically adjusts entry threshold based on recent win rate."""

    def __init__(self, base_threshold: float) -> None:
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold
        self._recent_outcomes: list[bool] = []  # True=win, False=loss

    @property
    def threshold(self) -> float:
        return self.current_threshold

    def record_outcome(self, is_win: bool) -> None:
        """Record a trade outcome and recalculate threshold."""
        self._recent_outcomes.append(is_win)

        # Keep only last N
        if len(self._recent_outcomes) > LOOKBACK_WINDOW:
            self._recent_outcomes = self._recent_outcomes[-LOOKBACK_WINDOW:]

        self._recalculate()

    def _recalculate(self) -> None:
        """Adjust threshold based on recent win rate."""
        if len(self._recent_outcomes) < 5:
            # Not enough data yet
            return

        wins = sum(1 for o in self._recent_outcomes if o)
        win_rate = wins / len(self._recent_outcomes)
        old_threshold = self.current_threshold

        if win_rate < LOSING_THRESHOLD:
            # Losing too much — become more selective
            self.current_threshold = min(
                self.current_threshold + STEP_UP,
                MAX_THRESHOLD,
            )
        elif win_rate > WINNING_THRESHOLD:
            # Winning well — can afford to be less selective
            self.current_threshold = max(
                self.current_threshold - STEP_DOWN,
                MIN_THRESHOLD,
            )
        # else: in the neutral zone, no change

        # Clamp
        self.current_threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, self.current_threshold))

        if abs(old_threshold - self.current_threshold) > 0.01:
            logger.info(
                "threshold_adjusted",
                old=old_threshold,
                new=self.current_threshold,
                win_rate=round(win_rate, 3),
                recent_trades=len(self._recent_outcomes),
                wins=wins,
            )

    def load_outcomes(self, outcomes: list[bool]) -> None:
        """Bulk-load historical outcomes (for startup recovery)."""
        self._recent_outcomes = outcomes[-LOOKBACK_WINDOW:]
        if self._recent_outcomes:
            self._recalculate()
            logger.info(
                "adaptive_threshold_loaded",
                threshold=self.current_threshold,
                outcomes_loaded=len(self._recent_outcomes),
            )

    def get_status(self) -> dict:
        """Return current threshold status for dashboard/logging."""
        total = len(self._recent_outcomes)
        wins = sum(1 for o in self._recent_outcomes if o)
        return {
            "current_threshold": self.current_threshold,
            "base_threshold": self.base_threshold,
            "recent_trades": total,
            "recent_wins": wins,
            "recent_win_rate": round(wins / total, 3) if total > 0 else 0,
        }
