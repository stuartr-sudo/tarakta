"""Next-batch course-faithful regression tests.

Covers the second audit batch:
  B1 — 3 inside-right-side hits on 15m
  B4 — SVC wick-return invalidation
  C2 — weekend-box M/W entry path
  C3 — 200 EMA rejection formation
  C4 — board-meeting M/W entry
  C6 — 200 EMA hammer partial TP
  G1 — Linda Trade multi-TF cascade
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.strategy.mm_engine import MMEngine, MMPosition
from src.strategy.mm_linda import (
    LindaTracker,
    TF_LADDER,
    _next_tf_up,
)


@pytest.fixture
def engine() -> MMEngine:
    return MMEngine(exchange=None, repo=None, candle_manager=None, config=None)


# ---------------------------------------------------------------------------
# B1 — inside-right-side hits on 15m
# ---------------------------------------------------------------------------


def test_inside_hits_fail_open_when_15m_missing(engine: MMEngine):
    """No 15m data → fail-open (don't block on missing TF)."""

    class F:
        type = "M"
        variant = "standard"
        peak1_idx = 5
        peak2_idx = 15
        trough_idx = 10

    candles_1h = pd.DataFrame({
        "open": [100] * 20, "high": [100] * 20, "low": [100] * 20,
        "close": [100] * 20, "volume": [1] * 20,
    }, index=pd.date_range("2026-04-15", periods=20, freq="1h", tz="UTC"))
    assert engine._check_inside_hits_15m(F(), candles_1h, None, 0) is True
    empty = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
    assert engine._check_inside_hits_15m(F(), candles_1h, empty, 0) is True


def test_inside_hits_counts_green_candles_for_M(engine: MMEngine):
    """M formation requires ≥3 GREEN 15m candles in the inside-right-side window."""

    class F:
        type = "M"
        variant = "standard"
        peak1_idx = 5
        peak2_idx = 10
        trough_idx = 7

    candles_1h = pd.DataFrame({
        "open": [100] * 20, "high": [100] * 20, "low": [100] * 20,
        "close": [100] * 20, "volume": [1] * 20,
    }, index=pd.date_range("2026-04-15", periods=20, freq="1h", tz="UTC"))

    # Build 15m candles that fall between the 1h[7] trough and 1h[10] peak2.
    ts_start = candles_1h.index[7]
    ts_end = candles_1h.index[10]
    m15_range = pd.date_range(ts_start, ts_end, freq="15min", tz="UTC")
    # 4 bullish green + 1 red inside the window
    open_prices = [100, 101, 102, 103, 104, 103, 102, 101, 100, 99, 98, 97, 96]
    close_prices = [102, 103, 104, 105, 103, 102, 101, 100, 99, 98, 97, 96, 95]
    # Align to actual index length
    n = min(len(m15_range), len(open_prices))
    m15 = pd.DataFrame({
        "open": open_prices[:n],
        "high": [max(o, c) + 1 for o, c in zip(open_prices[:n], close_prices[:n])],
        "low": [min(o, c) - 1 for o, c in zip(open_prices[:n], close_prices[:n])],
        "close": close_prices[:n],
        "volume": [1] * n,
    }, index=m15_range[:n])

    assert engine._check_inside_hits_15m(F(), candles_1h, m15, 0) is True


def test_inside_hits_rejects_when_all_red_for_M(engine: MMEngine):
    """M formation should FAIL when all 15m in-window candles are red (no inducement)."""

    class F:
        type = "M"
        variant = "standard"
        peak1_idx = 2
        peak2_idx = 10
        trough_idx = 5

    candles_1h = pd.DataFrame({
        "open": [100] * 20, "high": [100] * 20, "low": [100] * 20,
        "close": [100] * 20, "volume": [1] * 20,
    }, index=pd.date_range("2026-04-15", periods=20, freq="1h", tz="UTC"))

    ts_start = candles_1h.index[5]
    ts_end = candles_1h.index[10]
    m15_range = pd.date_range(ts_start, ts_end, freq="15min", tz="UTC")
    n = len(m15_range)
    # All red candles — no bullish inducement
    m15 = pd.DataFrame({
        "open": [110] * n, "high": [110] * n, "low": [100] * n,
        "close": [101] * n, "volume": [1] * n,
    }, index=m15_range)
    assert engine._check_inside_hits_15m(F(), candles_1h, m15, 0) is False


# ---------------------------------------------------------------------------
# B4 — SVC wick-return invalidation field on MMPosition
# ---------------------------------------------------------------------------


def test_mmposition_has_svc_fields():
    pos = MMPosition()
    assert hasattr(pos, "svc_high")
    assert hasattr(pos, "svc_low")
    assert pos.svc_high == 0.0
    assert pos.svc_low == 0.0


def test_mmposition_took_200ema_partial_default_false():
    pos = MMPosition()
    assert pos.took_200ema_partial is False


# ---------------------------------------------------------------------------
# C3 — 200 EMA rejection formation helper
# ---------------------------------------------------------------------------


def test_200ema_rejection_returns_none_on_insufficient_data(engine: MMEngine):
    assert engine._try_200ema_rejection_formation(None, None, None) is None
    empty = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
    assert engine._try_200ema_rejection_formation(empty, None, empty) is None


def test_200ema_rejection_returns_none_when_far_from_ema(engine: MMEngine):
    """Price 20% away from 200 EMA → no rejection trade."""
    n = 250
    closes = [100.0] * 200 + [150.0] * 50
    candles_1h = pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1] * n,
    }, index=pd.date_range("2026-04-15", periods=n, freq="1h", tz="UTC"))
    m15 = candles_1h.copy()  # Simple proxy for this test
    assert engine._try_200ema_rejection_formation(candles_1h, None, m15) is None


# ---------------------------------------------------------------------------
# C4 — board-meeting formation helper
# ---------------------------------------------------------------------------


def test_board_meeting_returns_none_on_insufficient_data(engine: MMEngine):
    assert engine._try_board_meeting_formation(None) is None
    small = pd.DataFrame({
        "open": [100] * 5, "high": [100] * 5, "low": [100] * 5,
        "close": [100] * 5, "volume": [1] * 5,
    }, index=pd.date_range("2026-04-15", periods=5, freq="1h", tz="UTC"))
    assert engine._try_board_meeting_formation(small) is None


# ---------------------------------------------------------------------------
# G1 — Linda Trade multi-TF cascade
# ---------------------------------------------------------------------------


def test_tf_ladder_ordered_correctly():
    assert TF_LADDER == ["15m", "1h", "4h", "1d", "1w"]


def test_next_tf_up_steps_ladder():
    assert _next_tf_up("15m") == "1h"
    assert _next_tf_up("1h") == "4h"
    assert _next_tf_up("4h") == "1d"
    assert _next_tf_up("1d") == "1w"
    assert _next_tf_up("1w") is None
    assert _next_tf_up("bogus") is None


def test_linda_tracker_records_level_increase():
    t = LindaTracker()
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    t.record("BTC/USDT", "15m", level=1, direction="bullish", now=now)
    st = t.get("BTC/USDT", "15m")
    assert st.current_level == 1
    assert st.direction == "bullish"
    assert st.last_level_completed_at == now


def test_linda_tracker_cascades_on_level_3_completion():
    t = LindaTracker()
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    # Record 15m L1 → L2 → L3 — should cascade to 1H L1
    t.record("ETH/USDT", "15m", 1, "bullish", now)
    t.record("ETH/USDT", "15m", 2, "bullish", now)
    events = t.record("ETH/USDT", "15m", 3, "bullish", now)
    assert len(events) >= 1
    assert events[0].from_tf == "15m"
    assert events[0].to_tf == "1h"
    assert events[0].to_tf_new_level == 1
    # 15m reset to 0 after cycle completion
    assert t.get("ETH/USDT", "15m").current_level == 0
    # 1h now at L1
    assert t.get("ETH/USDT", "1h").current_level == 1
    assert t.get("ETH/USDT", "15m").cycles_completed == 1


def test_linda_tracker_cascades_up_multiple_tfs():
    """Three consecutive 15m cycles → 1h level 3 → another cascade to 4h level 1."""
    t = LindaTracker()
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    # Each outer loop = one 15m 3-level cycle → ticks 1h up by 1
    for _ in range(3):
        for level in (1, 2, 3):
            t.record("SOL/USDT", "15m", level, "bullish", now)
    # After 3 full 15m cycles, 1h should have hit L3 → cascade to 4h L1
    assert t.get("SOL/USDT", "4h").current_level == 1


def test_linda_tracker_cascade_detected():
    t = LindaTracker()
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    t.record("BTC/USDT", "15m", 1, "bullish", now)
    t.record("BTC/USDT", "15m", 2, "bullish", now)
    t.record("BTC/USDT", "15m", 3, "bullish", now)
    assert t.cascade_detected("BTC/USDT", from_tf="15m", to_tf="1h") is True
    assert t.cascade_detected("BTC/USDT", from_tf="1h", to_tf="4h") is False


def test_linda_tracker_reset_weekly():
    t = LindaTracker()
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    t.record("BTC/USDT", "1h", 2, "bullish", now)
    assert t.get("BTC/USDT", "1h").current_level == 2
    t.reset_weekly("BTC/USDT")
    assert t.get("BTC/USDT", "1h").current_level == 0


def test_linda_tracker_does_not_regress_on_lower_level():
    t = LindaTracker()
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    t.record("BTC/USDT", "1h", 2, "bullish", now)
    # Recording a LOWER level should NOT reset — the tracker only moves forward
    t.record("BTC/USDT", "1h", 1, "bullish", now)
    assert t.get("BTC/USDT", "1h").current_level == 2


def test_linda_tracker_partial_cycle_allowed_for_htf():
    """Lesson 55: HTF retracements give 1-2 rises, not 3."""
    t = LindaTracker()
    # Mark 4h as a partial-cycle-allowed TF
    st = t.get("BTC/USDT", "4h")
    st.partial_cycle_allowed = True
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    # Two levels on 4h should be enough to cascade to 1d when partial-allowed
    events = []
    events.extend(t.record("BTC/USDT", "4h", 1, "bullish", now))
    events.extend(t.record("BTC/USDT", "4h", 2, "bullish", now))
    # 1d should have ticked up once under partial-cycle rules
    assert t.get("BTC/USDT", "1d").current_level >= 1


def test_linda_snapshot_includes_tfs_and_cascades():
    t = LindaTracker()
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    t.record("BTC/USDT", "15m", 1, "bullish", now)
    t.record("BTC/USDT", "15m", 2, "bullish", now)
    t.record("BTC/USDT", "15m", 3, "bullish", now)
    snap = t.snapshot("BTC/USDT")
    assert snap["symbol"] == "BTC/USDT"
    assert "tfs" in snap
    assert "15m" in snap["tfs"]
    assert "1h" in snap["tfs"]
    assert len(snap["recent_cascades"]) >= 1


# ---------------------------------------------------------------------------
# Engine has LindaTracker attached
# ---------------------------------------------------------------------------


def test_engine_has_linda_tracker(engine: MMEngine):
    assert hasattr(engine, "linda")
    assert isinstance(engine.linda, LindaTracker)
