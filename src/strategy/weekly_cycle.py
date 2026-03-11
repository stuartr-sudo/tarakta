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
        midweek_delay_hours: float = 4.0,
    ) -> None:
        self.monday_penalty = monday_penalty_pts
        self.monday_hours = monday_manipulation_hours
        self.midweek_bonus = midweek_reversal_bonus_pts
        self.midweek_delay_hours = midweek_delay_hours

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

        # ── Monday Manipulation (tapering penalty) ──────────────────
        if hours_since_open < self.monday_hours:
            result.in_monday_manipulation = True
            # Linear taper: full penalty at hour 0, zero at window boundary
            taper = 1.0 - (hours_since_open / self.monday_hours)
            penalty = round(self.monday_penalty * taper, 1)
            result.score_adjustment -= penalty
            result.reasons.append(
                f"Monday manipulation window "
                f"({hours_since_open:.1f}h into week, "
                f"penalty −{penalty:.1f} pts, tapers to 0 at {self.monday_hours:.0f}h)"
            )
            logger.info(
                "weekly_cycle_monday_manipulation",
                hours_into_week=f"{hours_since_open:.1f}",
                base_penalty=self.monday_penalty,
                tapered_penalty=penalty,
                taper_pct=f"{taper:.0%}",
            )

        # ── Mid-Week Reversal (Wednesday & Thursday) ─────────────────
        # Delay: only apply bonus after N hours into the day (confirmation)
        # Strength: scale bonus by how strong the early-week fake move was
        if day in (2, 3):  # Wednesday=2, Thursday=3
            early_dir, move_pct = self._get_early_week_direction_with_magnitude(
                candles_1d, now
            )
            result.early_week_direction = early_dir

            # Hours into the current day (Wed or Thu)
            hours_into_day = now.hour + now.minute / 60

            if early_dir and signal_direction:
                if signal_direction != early_dir:
                    # Signal opposes the Monday move → potential mid-week reversal
                    result.in_midweek_reversal_window = True

                    # Delay gate: wait N hours into the day for confirmation
                    if hours_into_day < self.midweek_delay_hours:
                        result.signal_aligns_with_reversal = False
                        result.reasons.append(
                            f"Mid-week reversal detected but too early "
                            f"({hours_into_day:.1f}h into {day_name}, "
                            f"need {self.midweek_delay_hours:.0f}h confirmation)"
                        )
                        logger.info(
                            "weekly_cycle_midweek_reversal_delayed",
                            day=day_name,
                            hours_into_day=f"{hours_into_day:.1f}",
                            delay_required=self.midweek_delay_hours,
                            early_week=early_dir,
                            signal=signal_direction,
                        )
                    else:
                        # Delay passed — apply strength-scaled bonus
                        strength_mult = self._strength_multiplier(move_pct)
                        bonus = round(self.midweek_bonus * strength_mult, 1)

                        if bonus > 0:
                            result.signal_aligns_with_reversal = True
                            result.score_adjustment += bonus
                            result.reasons.append(
                                f"Mid-week reversal: early week was {early_dir} "
                                f"({abs(move_pct):.2%} move, "
                                f"{strength_mult:.1f}x strength), "
                                f"signal is {signal_direction} → "
                                f"+{bonus:.1f} pts bonus"
                            )
                            logger.info(
                                "weekly_cycle_midweek_reversal",
                                day=day_name,
                                early_week=early_dir,
                                signal=signal_direction,
                                move_pct=f"{move_pct:.4f}",
                                strength_mult=strength_mult,
                                bonus=bonus,
                            )
                        else:
                            result.signal_aligns_with_reversal = False
                            result.reasons.append(
                                f"Mid-week reversal: early-week move too small "
                                f"({abs(move_pct):.2%}) for bonus"
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

    @staticmethod
    def _strength_multiplier(move_pct: float) -> float:
        """Scale mid-week reversal bonus by early-week move magnitude.

        Bigger fake move = more conviction the reversal is real.

        Returns:
            0.0  if move < 0.3% (too small to classify)
            0.5  if 0.3% – 1%  (weak move → half bonus)
            1.0  if 1% – 2%    (moderate move → full bonus)
            1.5  if > 2%       (strong fake move → 1.5× bonus)
        """
        abs_move = abs(move_pct)
        if abs_move < 0.003:
            return 0.0
        elif abs_move < 0.01:
            return 0.5
        elif abs_move < 0.02:
            return 1.0
        else:
            return 1.5

    def _get_early_week_direction_with_magnitude(
        self,
        candles_1d: pd.DataFrame,
        now: datetime,
    ) -> tuple[str | None, float]:
        """Determine direction AND magnitude of the early-week move.

        Compares Monday's daily open price to Monday's close (or Tuesday's
        close if available).  A clear move up = "bullish", down = "bearish".

        Returns:
            (direction, move_pct) where direction is "bullish"/"bearish"/None
            and move_pct is the raw signed percentage change.
        """
        if candles_1d is None or candles_1d.empty:
            return None, 0.0

        idx = candles_1d.index
        if not isinstance(idx, pd.DatetimeIndex):
            return None, 0.0

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
            return None, 0.0

        # Monday open = first candle's open
        week_open = float(early_week["open"].iloc[0])

        # Use the latest available close (Tuesday close if we have it,
        # otherwise Monday close)
        week_early_close = float(early_week["close"].iloc[-1])

        if week_open <= 0:
            return None, 0.0

        # Calculate move percentage
        move_pct = (week_early_close - week_open) / week_open

        # Require at least 0.3% move to classify a direction
        # (avoids noise from flat / doji days)
        MIN_MOVE_PCT = 0.003

        if move_pct > MIN_MOVE_PCT:
            return "bullish", move_pct
        elif move_pct < -MIN_MOVE_PCT:
            return "bearish", move_pct

        return None, move_pct  # Too small to classify
