"""Weekly cycle analysis — Fake Move Monday & Mid-Week Reversal.

ICT / Smart Money concept: the trading week follows a predictable pattern
driven by institutional order flow and market-maker manipulation:

  Monday    — "Fake Move Monday": market makers push price in a deceptive
              direction to trap early-week positions.  Avoid trading during
              the first N hours of the week.

  Tue       — Continuation or early signs of reversal.

  Wed / Thu — "Mid-Week Reversal": the REAL direction emerges, typically
              the opposite of Monday's fake move.  Counter-trend signals
              on these days have higher conviction.

  Fri       — Distribution / position squaring before the weekend.

Weekly open = Monday 00:00 UTC on most crypto exchanges (Binance, etc.).
In NZT (UTC+12/+13) this is approximately Monday 12:00–13:00.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WeeklyCycleResult:
    """Result of weekly cycle analysis for a single coin."""

    day_of_week: int = 0  # 0=Monday … 6=Sunday
    day_name: str = "Monday"
    hours_since_weekly_open: float = 0.0

    # Monday manipulation
    in_monday_manipulation: bool = False

    # Mid-week reversal
    in_midweek_reversal_window: bool = False
    early_week_direction: str | None = None  # "bullish" / "bearish"
    signal_aligns_with_reversal: bool = False

    # Net score adjustment to apply
    score_adjustment: float = 0.0
    reasons: list[str] = field(default_factory=list)


_DAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


class WeeklyCycleAnalyzer:
    """Analyzes weekly cycle for Fake Move Monday & Mid-Week Reversal.

    Parameters are injected from Settings so they can be tuned via .env /
    Fly.io secrets without code changes.
    """

    def __init__(
        self,
        monday_penalty_pts: float = 15.0,
        monday_manipulation_hours: float = 8.0,
        midweek_reversal_bonus_pts: float = 10.0,
    ) -> None:
        self.monday_penalty = monday_penalty_pts
        self.monday_hours = monday_manipulation_hours
        self.midweek_bonus = midweek_reversal_bonus_pts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        candles_1d: pd.DataFrame,
        signal_direction: str | None = None,
        now: datetime | None = None,
    ) -> WeeklyCycleResult:
        """Evaluate weekly cycle position and return score adjustment.

        Args:
            candles_1d: Daily OHLCV candles (DatetimeIndex, UTC) for the
                        coin being analyzed.  Used to derive the early-week
                        direction for mid-week reversal checks.
            signal_direction: Direction of the signal being scored
                              ("bullish" / "bearish").  Required for
                              mid-week reversal alignment bonus.
            now: Current UTC time.  Defaults to ``datetime.now(UTC)``.

        Returns:
            WeeklyCycleResult with ``score_adjustment`` (negative on Monday
            manipulation window, positive on mid-week reversal alignment).
        """
        if now is None:
            now = datetime.now(timezone.utc)

        day = now.weekday()  # 0=Mon … 6=Sun
        day_name = _DAY_NAMES[day]

        # Hours since Monday 00:00 UTC (the crypto weekly open)
        days_since_monday = day  # 0 on Monday, 1 on Tuesday, …
        hours_since_open = days_since_monday * 24 + now.hour + now.minute / 60

        result = WeeklyCycleResult(
            day_of_week=day,
            day_name=day_name,
            hours_since_weekly_open=hours_since_open,
        )

        # ── Monday Manipulation ──────────────────────────────────────
        if hours_since_open < self.monday_hours:
            result.in_monday_manipulation = True
            result.score_adjustment -= self.monday_penalty
            result.reasons.append(
                f"Monday manipulation window "
                f"({hours_since_open:.1f}h into week, "
                f"penalty −{self.monday_penalty:.0f} pts)"
            )
            logger.info(
                "weekly_cycle_monday_manipulation",
                hours_into_week=f"{hours_since_open:.1f}",
                penalty=self.monday_penalty,
            )

        # ── Mid-Week Reversal (Wednesday & Thursday) ─────────────────
        if day in (2, 3):  # Wednesday=2, Thursday=3
            early_dir = self._get_early_week_direction(candles_1d, now)
            result.early_week_direction = early_dir

            if early_dir and signal_direction:
                if signal_direction != early_dir:
                    # Signal opposes the Monday move → mid-week reversal
                    result.in_midweek_reversal_window = True
                    result.signal_aligns_with_reversal = True
                    result.score_adjustment += self.midweek_bonus
                    result.reasons.append(
                        f"Mid-week reversal: early week was {early_dir}, "
                        f"signal is {signal_direction} → "
                        f"+{self.midweek_bonus:.0f} pts bonus"
                    )
                    logger.info(
                        "weekly_cycle_midweek_reversal",
                        day=day_name,
                        early_week=early_dir,
                        signal=signal_direction,
                        bonus=self.midweek_bonus,
                    )
                else:
                    # Signal continues the Monday direction — no bonus,
                    # but no penalty either (could be genuine continuation)
                    result.in_midweek_reversal_window = True
                    result.signal_aligns_with_reversal = False
                    result.reasons.append(
                        f"Mid-week: signal continues early-week {early_dir} "
                        f"(no reversal bonus)"
                    )
            elif not early_dir:
                result.reasons.append(
                    "Mid-week reversal: could not determine early-week direction"
                )

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_early_week_direction(
        self,
        candles_1d: pd.DataFrame,
        now: datetime,
    ) -> str | None:
        """Determine the direction of the early-week move for this coin.

        Compares Monday's daily open price to Monday's close (or Tuesday's
        close if available).  A clear move up = "bullish", down = "bearish".

        Returns None if daily candles don't contain Monday data for this
        week, or if the move is too small to classify (<0.3%).
        """
        if candles_1d is None or candles_1d.empty:
            return None

        idx = candles_1d.index
        if not isinstance(idx, pd.DatetimeIndex):
            return None

        # Ensure UTC
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
            candles_1d = candles_1d.copy()
            candles_1d.index = idx

        # Find this week's Monday (go back to the most recent Monday)
        today = now.date()
        days_back = today.weekday()  # 0 if today is Monday
        monday_date = today - pd.Timedelta(days=days_back)

        monday_start = pd.Timestamp(
            year=monday_date.year,
            month=monday_date.month,
            day=monday_date.day,
            tz="UTC",
        )
        tuesday_end = monday_start + pd.Timedelta(days=2)

        # Get Monday + Tuesday candles
        early_week = candles_1d[
            (candles_1d.index >= monday_start) & (candles_1d.index < tuesday_end)
        ]

        if early_week.empty:
            return None

        # Monday open = first candle's open
        week_open = float(early_week["open"].iloc[0])

        # Use the latest available close (Tuesday close if we have it,
        # otherwise Monday close)
        week_early_close = float(early_week["close"].iloc[-1])

        if week_open <= 0:
            return None

        # Calculate move percentage
        move_pct = (week_early_close - week_open) / week_open

        # Require at least 0.3% move to classify a direction
        # (avoids noise from flat / doji days)
        MIN_MOVE_PCT = 0.003

        if move_pct > MIN_MOVE_PCT:
            return "bullish"
        elif move_pct < -MIN_MOVE_PCT:
            return "bearish"

        return None  # Too small to classify
