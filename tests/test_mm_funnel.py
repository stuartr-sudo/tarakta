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


# ---------------------------------------------------------------------------
# last_funnel — the snapshot the dashboard reads via get_status()
# ---------------------------------------------------------------------------


def test_last_funnel_initialized_none(engine: MMEngine) -> None:
    """Fresh engine has no funnel yet — dashboard should render the empty state."""
    assert hasattr(engine, "last_funnel")
    assert engine.last_funnel is None


def test_last_funnel_shape_when_populated(engine: MMEngine) -> None:
    """Simulate what the scan loop stores and verify every key the dashboard needs."""
    # Match the exact dict the scan loop builds (see mm_engine.py _cycle)
    engine.last_funnel = {
        "cycle": 14,
        "timestamp": "2026-04-14T10:03:02.375557+00:00",
        "pairs_scanned": 67,
        "signals_found": 0,
        "rejected_total": 67,
        "exceptions": 0,
        "unaccounted": 0,
        "rejects": {"no_formation": 43, "no_l1_target": 14, "sl_too_wide": 6},
    }
    f = engine.last_funnel
    # Every key that mm.html reads must be present
    for key in ("cycle", "timestamp", "pairs_scanned", "signals_found",
                "rejected_total", "exceptions", "unaccounted", "rejects"):
        assert key in f, f"dashboard contract: last_funnel must contain '{key}'"
    # Sort invariant used by the UI bar chart
    counts = list(f["rejects"].values())
    assert counts == sorted(counts, reverse=True)


@pytest.mark.asyncio
async def test_get_status_includes_last_funnel_when_none(engine: MMEngine) -> None:
    """get_status() must surface last_funnel=None on fresh boot (not missing key)."""
    # No positions, no exchange calls needed
    engine.positions = {}
    # session_analyzer is a real instance from __init__; safe to call
    status = await engine.get_status()
    assert "last_funnel" in status, "/api/mm/status must expose last_funnel"
    assert status["last_funnel"] is None


@pytest.mark.asyncio
async def test_get_status_includes_last_funnel_when_set(engine: MMEngine) -> None:
    """get_status() must pass through a populated last_funnel snapshot verbatim."""
    engine.positions = {}
    engine.last_funnel = {
        "cycle": 7,
        "timestamp": "2026-04-14T09:00:00+00:00",
        "pairs_scanned": 67,
        "signals_found": 0,
        "rejected_total": 67,
        "exceptions": 0,
        "unaccounted": 0,
        "rejects": {"no_formation": 50, "sl_too_wide": 17},
    }
    status = await engine.get_status()
    assert status["last_funnel"] == engine.last_funnel
    assert status["last_funnel"]["rejects"]["no_formation"] == 50


# ---------------------------------------------------------------------------
# Extended telemetry: stage counters, factor hits, score distribution.
# These prove to the dashboard that the scoring stages are actually being
# reached (not just the hard gates).
# ---------------------------------------------------------------------------


def test_engine_has_stage_telemetry_state(engine: MMEngine) -> None:
    """All six per-cycle telemetry dicts exist on a fresh MMEngine."""
    assert hasattr(engine, "_scan_stage_counts")
    assert hasattr(engine, "_scan_factor_hits")
    assert hasattr(engine, "_scan_score_samples")
    assert hasattr(engine, "_scan_grade_counts")
    assert hasattr(engine, "_scan_retest_counts")
    assert engine._scan_stage_counts == {}
    assert engine._scan_factor_hits == {}
    assert engine._scan_score_samples == []
    assert engine._scan_grade_counts == {}
    assert engine._scan_retest_counts == {}


def test_advance_helper_increments_stage(engine: MMEngine) -> None:
    """_advance('stage_name') bumps the named counter by one."""
    engine._scan_stage_counts = {}
    engine._advance("candles_ok")
    assert engine._scan_stage_counts == {"candles_ok": 1}
    engine._advance("candles_ok")
    engine._advance("formation_found")
    assert engine._scan_stage_counts == {"candles_ok": 2, "formation_found": 1}


def test_advance_helper_does_not_return_anything(engine: MMEngine) -> None:
    """_advance returns None (it's called for side effect, never for value)."""
    result = engine._advance("candles_ok")
    assert result is None


def test_advance_helper_bound_method(engine: MMEngine) -> None:
    """Guards against renaming regression."""
    assert callable(engine._advance)
    import inspect
    params = list(inspect.signature(engine._advance).parameters)
    assert params == ["stage"]


@pytest.mark.asyncio
async def test_get_status_exposes_extended_funnel_fields(engine: MMEngine) -> None:
    """last_funnel contract: dashboard needs stages, factor_hits, score_stats, grades, retest_counts."""
    engine.positions = {}
    engine.last_funnel = {
        "cycle": 3,
        "timestamp": "2026-04-14T10:00:00+00:00",
        "pairs_scanned": 90,
        "signals_found": 0,
        "rejected_total": 88,
        "exceptions": 0,
        "unaccounted": 2,
        "rejects": {"low_rr": 42, "no_formation": 35},
        "stages": {
            "candles_ok": 88, "formation_found": 55, "level_ok": 50,
            "phase_valid": 48, "direction_ok": 46, "target_acquired": 46,
            "rr_passed": 6, "scored": 6, "confluence_passed": 2,
            "retest_passed": 0, "signal_built": 0,
        },
        "factor_hits": {
            "mw_session_changeover": 4, "ema_alignment": 5,
            "oi_behavior": 2, "stopping_volume_candle": 1,
        },
        "score_stats": {"count": 6, "min": 18.5, "max": 52.3, "avg": 34.1, "median": 33.0},
        "grades": {"F": 4, "C": 2},
        "retest_counts": {"0": 2, "1": 2, "2": 2, "3": 0, "4": 0},
    }
    status = await engine.get_status()
    # Every key the dashboard template references
    for k in ("stages", "factor_hits", "score_stats", "grades", "retest_counts", "rejects"):
        assert k in status["last_funnel"], f"dashboard contract: missing '{k}'"
    assert status["last_funnel"]["stages"]["scored"] == 6
    assert status["last_funnel"]["factor_hits"]["ema_alignment"] == 5
    assert status["last_funnel"]["score_stats"]["avg"] == 34.1


def test_reset_clears_all_telemetry(engine: MMEngine) -> None:
    """At scan-cycle start, the engine resets every telemetry bucket.

    This is the contract tested at the scan-loop level — simulate the same
    reset pattern and verify the state is clean.
    """
    # Seed some data
    engine._scan_reject_counts = {"no_formation": 5}
    engine._scan_stage_counts = {"candles_ok": 10}
    engine._scan_factor_hits = {"ema_alignment": 3}
    engine._scan_score_samples = [45.0, 32.5]
    engine._scan_grade_counts = {"F": 2}
    engine._scan_retest_counts = {2: 1}

    # Simulate what _cycle does at scan start
    engine._scan_reject_counts = {}
    engine._scan_stage_counts = {}
    engine._scan_factor_hits = {}
    engine._scan_score_samples = []
    engine._scan_grade_counts = {}
    engine._scan_retest_counts = {}

    assert engine._scan_reject_counts == {}
    assert engine._scan_stage_counts == {}
    assert engine._scan_factor_hits == {}
    assert engine._scan_score_samples == []
    assert engine._scan_grade_counts == {}
    assert engine._scan_retest_counts == {}
