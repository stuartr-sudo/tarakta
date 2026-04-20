"""Unit tests for the P&L simulation in scripts/replay_scan.py.

Covers the deterministic path of simulate_signal: SL hit, TP ladder,
breakeven-after-TP1, timeout, pessimistic same-bar SL-vs-TP tie-break,
and short-side symmetry.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from replay_scan import simulate_signal  # noqa: E402


def _candles(*bars: tuple[float, float, float, float]) -> pd.DataFrame:
    """Build a 1H forward-candle DF from (open, high, low, close) tuples.

    Timestamps start at 2026-01-01 00:00 UTC, one hour apart.
    """
    base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    idx = []
    for i, (o, h, l, c) in enumerate(bars):
        idx.append(base + timedelta(hours=i + 1))  # first forward bar is T+1h
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": 100.0})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, tz="UTC"))


SIGNAL_TS = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


class TestLongSLHit:
    def test_immediate_sl_is_full_loss(self):
        # Long, entry 100, SL 99. First bar goes straight to 98. Full -1R.
        result = simulate_signal(
            signal_ts=SIGNAL_TS, direction="long",
            entry_price=100.0, sl=99.0, tp1=102.0, tp2=103.0, tp3=105.0,
            forward_candles=_candles((99.5, 99.5, 98.0, 98.5)),
        )
        assert result.sl_hit is True
        assert result.tp1_hit is False
        assert result.exit_reason == "sl"
        # ~-1R minus fees
        assert -1.10 <= result.r_multiple <= -0.95


class TestLongTPLadder:
    def test_tp1_hit_moves_sl_to_breakeven(self):
        # Long entry 100 SL 99 TP1 102. Bar 1 hits TP1 (high=102.5).
        # Then bar 2 dips to 99.5 — with SL now at breakeven (~100) that
        # trips a BE stop for the remainder. Net should be small POSITIVE
        # (banked 30% at TP1 = +2R * 0.3 = +0.6R minus tiny breakeven
        # clip + fees).
        result = simulate_signal(
            signal_ts=SIGNAL_TS, direction="long",
            entry_price=100.0, sl=99.0, tp1=102.0, tp2=103.0, tp3=105.0,
            forward_candles=_candles(
                (100.0, 102.5, 99.8, 101.5),   # TP1 hit
                (101.0, 101.5, 99.5, 99.8),    # SL @ BE triggers on remainder
            ),
        )
        assert result.tp1_hit is True
        assert result.tp2_hit is False
        assert result.sl_hit is True  # the BE stop
        # Banked 30% at +2R plus remainder stopped at ~breakeven
        # ≈ 0.3 × 2 - 0.7 × 0 = +0.6R gross, -fees ≈ +0.55R or similar
        assert 0.40 <= result.r_multiple <= 0.70

    def test_tp3_full_win(self):
        # Long, clean ride all the way to TP3 without stopping.
        result = simulate_signal(
            signal_ts=SIGNAL_TS, direction="long",
            entry_price=100.0, sl=99.0, tp1=102.0, tp2=103.0, tp3=105.0,
            forward_candles=_candles(
                (100.0, 102.5, 100.0, 102.2),
                (102.0, 103.5, 101.5, 103.2),
                (103.0, 105.5, 102.5, 104.5),  # TP2 and TP3 in same bar
            ),
        )
        assert result.tp1_hit is True
        assert result.tp2_hit is True
        assert result.tp3_hit is True
        assert result.exit_reason == "tp3"
        # 30% @ +2R + 40% @ +3R + 30% @ +5R = 0.6 + 1.2 + 1.5 = +3.3R
        # fees ≈ 0.08% of notional → small
        assert 3.0 <= result.r_multiple <= 3.4


class TestShortSymmetry:
    def test_short_sl_hit(self):
        # Short, entry 100, SL 101 (above). Bar spikes to 102 = stopped.
        result = simulate_signal(
            signal_ts=SIGNAL_TS, direction="short",
            entry_price=100.0, sl=101.0, tp1=98.0, tp2=97.0, tp3=95.0,
            forward_candles=_candles((100.5, 102.0, 100.0, 101.5)),
        )
        assert result.sl_hit is True
        assert result.exit_reason == "sl"
        assert -1.10 <= result.r_multiple <= -0.95

    def test_short_tp3_full_win(self):
        # Short, clean ride to TP3 at 95.
        result = simulate_signal(
            signal_ts=SIGNAL_TS, direction="short",
            entry_price=100.0, sl=101.0, tp1=98.0, tp2=97.0, tp3=95.0,
            forward_candles=_candles(
                (100.0, 100.2, 97.8, 98.5),
                (98.0, 98.0, 96.8, 97.5),
                (97.0, 97.0, 94.8, 95.2),  # TP2 and TP3
            ),
        )
        assert result.tp1_hit
        assert result.tp2_hit
        assert result.tp3_hit
        assert 3.0 <= result.r_multiple <= 3.4


class TestSameBarPessimism:
    def test_same_bar_sl_and_tp1_both_touched_sl_wins(self):
        # Long. Bar has both SL and TP1 touched. Pessimistic: SL wins.
        result = simulate_signal(
            signal_ts=SIGNAL_TS, direction="long",
            entry_price=100.0, sl=99.0, tp1=102.0, tp2=103.0, tp3=105.0,
            forward_candles=_candles((100.0, 102.5, 98.5, 100.0)),
        )
        assert result.sl_hit is True
        assert result.tp1_hit is False  # pessimistic rule
        assert result.exit_reason == "sl"


class TestTimeout:
    def test_timeout_closes_at_current_close(self):
        # Long that drifts sideways forever — no SL, no TP.
        # Build 200 hours of flat candles at 100.5 (below TP1 @ 102).
        # max_hold default is 168h (7d); we use 200 bars to go past that.
        bars = [(100.0, 100.8, 99.8, 100.5)] * 200
        result = simulate_signal(
            signal_ts=SIGNAL_TS, direction="long",
            entry_price=100.0, sl=99.0, tp1=102.0, tp2=103.0, tp3=105.0,
            forward_candles=_candles(*bars),
            max_hold_hours=24 * 7,
        )
        assert result.timed_out is True
        assert result.exit_reason == "timeout"
        # +0.5% on 100% position → ~+0.5R on a 1% SL
        assert 0.3 <= result.r_multiple <= 0.6


class TestEdgeCases:
    def test_zero_sl_distance_returns_invalid(self):
        result = simulate_signal(
            signal_ts=SIGNAL_TS, direction="long",
            entry_price=100.0, sl=100.0,  # same as entry — invalid
            tp1=102.0, tp2=103.0, tp3=105.0,
            forward_candles=_candles((100.0, 101.0, 99.5, 100.5)),
        )
        assert result.exit_reason == "invalid_risk"
        assert result.r_multiple == 0.0

    def test_empty_forward_candles_returns_incomplete(self):
        # No bars to walk — should fall through to the "window_end"
        # catch but with no data, nothing to close against.
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
        result = simulate_signal(
            signal_ts=SIGNAL_TS, direction="long",
            entry_price=100.0, sl=99.0, tp1=102.0, tp2=103.0, tp3=105.0,
            forward_candles=empty,
        )
        # Remainder couldn't be closed — exit_reason stays empty string,
        # normalised to "incomplete" by simulate_signal
        assert result.exit_reason == "incomplete"

    def test_window_end_closes_at_last_close(self):
        # 10 bars drifting to 101 but not triggering TP1 (102). Window
        # shorter than timeout → close at last bar's close.
        bars = [(100.0, 100.8, 99.8, 100.5 + i * 0.05) for i in range(10)]
        result = simulate_signal(
            signal_ts=SIGNAL_TS, direction="long",
            entry_price=100.0, sl=99.0, tp1=102.0, tp2=103.0, tp3=105.0,
            forward_candles=_candles(*bars),
        )
        assert result.exit_reason == "window_end"
        # final close ~ 100.5 + 9*0.05 = 100.95 → +0.95% on 100% → ~+0.9R
        assert 0.6 <= result.r_multiple <= 1.1
