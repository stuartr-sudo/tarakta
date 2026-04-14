"""Tests for MM engine rejection funnel logging (src.strategy.mm_engine).

These tests exercise the `_reject` helper and the `_scan_reject_counts`
counter directly, without booting the full engine (no exchange/DB/candles).
The goal is to prove that:

1. `_reject(reason, symbol, **kwargs)` returns ``None`` (so ``return self._reject(...)``
   preserves the early-exit contract).
2. Each call increments the per-reason counter.
3. The counter aggregates across many calls (mixed reasons).
4. Resetting the dict (as the scan loop does at cycle start) clears state.
5. The helper emits a ``mm_reject_{reason}`` structlog event with the kwargs.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.strategy.mm_engine import MMEngine


@pytest.fixture
def engine() -> MMEngine:
    """Build an MMEngine without triggering exchange/DB side effects.

    ``__init__`` only attaches collaborators and sets defaults — no I/O.
    Passing ``None`` for exchange/repo/candle_manager is safe because the
    tests below never call methods that touch them.
    """
    return MMEngine(
        exchange=None,
        repo=None,
        candle_manager=None,
        config=None,
    )


def test_reject_returns_none(engine: MMEngine) -> None:
    """The helper must return None so `return self._reject(...)` still exits the scan."""
    assert engine._reject("no_formation", "BTC/USDT:USDT") is None


def test_reject_increments_counter(engine: MMEngine) -> None:
    """Single rejection bumps the matching reason to 1."""
    engine._scan_reject_counts = {}
    engine._reject("no_formation", "BTC/USDT:USDT")
    assert engine._scan_reject_counts == {"no_formation": 1}


def test_reject_aggregates_same_reason(engine: MMEngine) -> None:
    """Multiple calls with the same reason accumulate."""
    engine._scan_reject_counts = {}
    for sym in ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"):
        engine._reject("sl_too_wide", sym, sl_distance_pct=7.2, max=5.0)
    assert engine._scan_reject_counts == {"sl_too_wide": 3}


def test_reject_aggregates_mixed_reasons(engine: MMEngine) -> None:
    """Counter shows a full funnel across many reasons in one cycle."""
    engine._scan_reject_counts = {}
    # Simulate one scan cycle
    for _ in range(10):
        engine._reject("no_formation", "X")
    for _ in range(6):
        engine._reject("against_weekly_bias", "Y")
    for _ in range(4):
        engine._reject("no_l1_target", "Z")
    for _ in range(2):
        engine._reject("sl_too_wide", "W")

    assert engine._scan_reject_counts == {
        "no_formation": 10,
        "against_weekly_bias": 6,
        "no_l1_target": 4,
        "sl_too_wide": 2,
    }
    assert sum(engine._scan_reject_counts.values()) == 22


def test_reject_reset_clears_state(engine: MMEngine) -> None:
    """The scan loop resets the counter at cycle start — verify that contract."""
    engine._reject("no_formation", "BTC/USDT:USDT")
    engine._reject("low_rr", "ETH/USDT:USDT", rr=0.8, min_required=1.0)
    assert sum(engine._scan_reject_counts.values()) == 2

    # Scan loop does: self._scan_reject_counts = {}
    engine._scan_reject_counts = {}
    assert engine._scan_reject_counts == {}

    engine._reject("no_formation", "SOL/USDT:USDT")
    assert engine._scan_reject_counts == {"no_formation": 1}


def test_reject_emits_log_event_with_prefix(engine: MMEngine) -> None:
    """Each rejection emits an `mm_reject_{reason}` structlog event with the payload."""
    engine._scan_reject_counts = {}
    with patch("src.strategy.mm_engine.logger") as mock_logger:
        engine._reject(
            "low_rr",
            "BTC/USDT:USDT",
            rr=0.75,
            min_required=1.0,
            entry=50_000.0,
            sl=50_500.0,
        )

    mock_logger.info.assert_called_once_with(
        "mm_reject_low_rr",
        symbol="BTC/USDT:USDT",
        rr=0.75,
        min_required=1.0,
        entry=50_000.0,
        sl=50_500.0,
    )


def test_reject_all_known_reasons_count_independently(engine: MMEngine) -> None:
    """Smoke-test every rejection reason used in _analyze_pair is countable."""
    reasons = [
        "candle_fetch", "insufficient_candles", "no_formation",
        "low_formation_quality", "level_too_advanced", "fmwb_phase",
        "friday_trap", "wrong_phase", "against_weekly_bias",
        "sl_too_wide", "no_l1_target", "zero_risk",
        "low_rr", "low_confluence", "low_retest",
    ]
    engine._scan_reject_counts = {}
    for r in reasons:
        engine._reject(r, "TEST/USDT:USDT")
    assert engine._scan_reject_counts == {r: 1 for r in reasons}
    assert len(engine._scan_reject_counts) == 15


def test_funnel_counter_initialized_on_construction(engine: MMEngine) -> None:
    """Freshly constructed engine has an empty reject counter."""
    assert hasattr(engine, "_scan_reject_counts")
    assert engine._scan_reject_counts == {}


def test_engine_has_reject_helper_bound(engine: MMEngine) -> None:
    """The helper is a bound method (guards against refactor regressions)."""
    assert callable(engine._reject)
    # Verify the method takes the signature we expect
    import inspect
    sig = inspect.signature(engine._reject)
    params = list(sig.parameters)
    assert params[0] == "reason"
    assert params[1] == "symbol"


def test_mock_scan_cycle_funnel_totals() -> None:
    """End-to-end: simulate a cycle, compute rejected_total like the scan loop does."""
    engine = MMEngine(exchange=None, repo=None, candle_manager=None, config=None)
    engine._scan_reject_counts = {}

    # Cycle 1: 5 no_formation + 3 against_weekly_bias + 1 low_rr
    for _ in range(5):
        engine._reject("no_formation", "X")
    for _ in range(3):
        engine._reject("against_weekly_bias", "Y", trade_dir="long", fmwb_dir="up", real_dir="short")
    engine._reject("low_rr", "Z", rr=0.8, min_required=1.0, entry=1.0, sl=1.1, t1=1.05)

    rejected_total = sum(engine._scan_reject_counts.values())
    assert rejected_total == 9

    # This is the shape the funnel log will emit
    sorted_rejects = dict(sorted(engine._scan_reject_counts.items(), key=lambda kv: -kv[1]))
    assert list(sorted_rejects.keys()) == ["no_formation", "against_weekly_bias", "low_rr"]
    assert list(sorted_rejects.values()) == [5, 3, 1]
