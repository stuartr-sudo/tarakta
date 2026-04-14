"""Linda Trade — multi-timeframe level cascade (course lesson 55).

The course teaches that a 3-level swing on a lower timeframe often IS a
single level on the next timeframe up:

    15m 3-level swing  =  1H  Level 1
    1H  3-level swing  =  4H  Level 1   (~1 week)
    4H  3-level swing  =  Daily Level 1 (~1 month)
    Daily 3-level swing = Weekly Level 1 (~3 months)

A "Linda Trade" is an entry taken on the lowest TF, held as each TF's
3-level swing completes and cascades up. The reward is ENORMOUS (multi-week
to multi-month moves) but identification requires tracking when each TF's
3-level cycle has completed and started the next one up.

This module provides:
  - ``TFLevelState``: per-(symbol, timeframe) running level count
  - ``LindaTracker``: maintains per-symbol dictionaries of TFLevelState
    keyed by TF, detects cascades, and emits events

Usage from ``MMEngine`` (wired into `_analyze_pair` / `_manage_position`):

    linda = LindaTracker()
    linda.record(symbol="BTC/USDT", timeframe="15m", level=3)
    if linda.cascade_detected(symbol, from_tf="15m", to_tf="1h"):
        # A 15m 3-level swing just completed — treat as 1H L1.
        ...

The course (lesson 55) also states:
  "HTF retracements give 1-2 rises (not 3)"

So the cascade from 4H up to Daily can happen on only 1-2 levels instead
of a full 3, particularly if it lines up with news / FMWB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


# Ordered TF hierarchy used for cascade detection.
TF_LADDER: list[str] = ["15m", "1h", "4h", "1d", "1w"]


def _next_tf_up(tf: str) -> str | None:
    """Return the next timeframe above the given one, or None at the top."""
    try:
        i = TF_LADDER.index(tf)
    except ValueError:
        return None
    return TF_LADDER[i + 1] if i + 1 < len(TF_LADDER) else None


@dataclass
class TFLevelState:
    """Running state of a symbol's 3-level cycle on a single timeframe."""

    symbol: str
    timeframe: str
    current_level: int = 0                   # 0..3
    cycles_completed: int = 0                # Number of full 3-level cycles observed
    last_level_completed_at: datetime | None = None
    direction: Literal["bullish", "bearish", "unknown"] = "unknown"

    # HTF retracement flag — lesson 55 notes HTF cycles only give 1-2 levels
    partial_cycle_allowed: bool = False


@dataclass
class CascadeEvent:
    """A "Linda" cascade: lower TF just completed a 3-level cycle which
    registers as a new level ticking on the next TF up."""

    symbol: str
    from_tf: str
    to_tf: str
    to_tf_new_level: int
    ts: datetime


@dataclass
class LindaTracker:
    """Per-symbol multi-TF level cascade bookkeeping.

    Not thread-safe by itself — the MMEngine calls this serialized from the
    scan loop. Reset-on-new-week happens through the weekly cycle tracker,
    which calls :meth:`reset_weekly`.
    """

    _states: dict[tuple[str, str], TFLevelState] = field(default_factory=dict)
    _events: list[CascadeEvent] = field(default_factory=list)

    def _key(self, symbol: str, tf: str) -> tuple[str, str]:
        return (symbol, tf)

    def get(self, symbol: str, timeframe: str) -> TFLevelState:
        k = self._key(symbol, timeframe)
        st = self._states.get(k)
        if st is None:
            st = TFLevelState(symbol=symbol, timeframe=timeframe)
            self._states[k] = st
        return st

    def record(
        self,
        symbol: str,
        timeframe: str,
        level: int,
        direction: Literal["bullish", "bearish", "unknown"] = "unknown",
        now: datetime | None = None,
    ) -> list[CascadeEvent]:
        """Record the current level observed on a TF. Returns any cascade
        events this observation fires off.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        st = self.get(symbol, timeframe)
        events: list[CascadeEvent] = []
        # Level transition — only act on strictly-increasing levels.
        if level > st.current_level:
            st.current_level = min(level, 3)
            st.direction = direction
            st.last_level_completed_at = now
            events.extend(self._maybe_cascade(symbol, timeframe, direction, now))
        return events

    def _maybe_cascade(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        now: datetime,
    ) -> list[CascadeEvent]:
        """If the TF has completed a (full or partial) cycle, cascade upward.

        A cycle is complete when:
          - current_level == 3, OR
          - current_level >= 2 AND partial_cycle_allowed (lesson 55 HTF rule).

        Emits a CascadeEvent, resets this TF to level 0, and recursively
        checks the next TF up.
        """
        events: list[CascadeEvent] = []
        st = self.get(symbol, timeframe)
        cycle_done = (
            st.current_level >= 3
            or (st.partial_cycle_allowed and st.current_level >= 2)
        )
        if not cycle_done:
            return events
        st.cycles_completed += 1
        st.current_level = 0
        upper = _next_tf_up(timeframe)
        if upper is None:
            return events
        upper_state = self.get(symbol, upper)
        new_upper_level = min(upper_state.current_level + 1, 3)
        upper_state.current_level = new_upper_level
        upper_state.direction = direction
        upper_state.last_level_completed_at = now
        ev = CascadeEvent(
            symbol=symbol,
            from_tf=timeframe,
            to_tf=upper,
            to_tf_new_level=new_upper_level,
            ts=now,
        )
        events.append(ev)
        self._events.append(ev)
        # Recursively cascade if the upper TF itself has now completed a cycle.
        events.extend(self._maybe_cascade(symbol, upper, direction, now))
        return events

    def cascade_detected(self, symbol: str, from_tf: str, to_tf: str) -> bool:
        """Return True if the most recent cascade event for this symbol was
        the requested from_tf → to_tf transition (useful for the engine to
        branch on "lower TF cycle just completed, treat next TF as L1")."""
        for ev in reversed(self._events):
            if ev.symbol == symbol:
                return ev.from_tf == from_tf and ev.to_tf == to_tf
        return False

    def reset_weekly(self, symbol: str) -> None:
        """Reset all per-symbol state at the start of a new week (Sun 5pm NY).

        Course: the 3-day / 3-level swing cycle restarts each week.
        """
        for k in list(self._states.keys()):
            if k[0] == symbol:
                st = self._states[k]
                st.current_level = 0
                st.direction = "unknown"
                st.last_level_completed_at = None
        # Keep cycles_completed running for telemetry; drop event history.
        self._events = [e for e in self._events if e.symbol != symbol]

    def snapshot(self, symbol: str) -> dict:
        """Return a dict snapshot for dashboard/telemetry."""
        out = {"symbol": symbol, "tfs": {}, "recent_cascades": []}
        for (s, tf), st in self._states.items():
            if s == symbol:
                out["tfs"][tf] = {
                    "current_level": st.current_level,
                    "cycles_completed": st.cycles_completed,
                    "direction": st.direction,
                    "last_level_completed_at": (
                        st.last_level_completed_at.isoformat()
                        if st.last_level_completed_at else None
                    ),
                }
        out["recent_cascades"] = [
            {
                "from_tf": e.from_tf,
                "to_tf": e.to_tf,
                "to_tf_new_level": e.to_tf_new_level,
                "ts": e.ts.isoformat(),
            }
            for e in self._events if e.symbol == symbol
        ][-10:]
        return out
