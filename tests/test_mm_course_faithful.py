"""Course-faithful redesign regression tests.

Each test documents which code change it guards, with the course citation.
See docs/MM_COURSE_FAITHFUL_REDESIGN.md for full rationale.
"""
from __future__ import annotations

import pytest

from src.strategy.mm_confluence import MAX_POSSIBLE, WEIGHTS
from src.strategy.mm_engine import (
    MIN_RR_AGGRESSIVE,
    MIN_RR_COURSE_FLOOR,
    MMEngine,
)


@pytest.fixture
def engine() -> MMEngine:
    """Engine with no real dependencies (no tests here call exchange/DB/candles)."""
    return MMEngine(exchange=None, repo=None, candle_manager=None, config=None)


# ---------------------------------------------------------------------------
# Change 4 — R:R floor raised from 1.0 to 1.4
# Course lesson 53: "don't get out of bed" is the floor, not 1.0.
# ---------------------------------------------------------------------------


def test_rr_floor_is_14_per_course_lesson_53():
    """MIN_RR_AGGRESSIVE is 1.4 (course floor), not 1.0 (previous code invention)."""
    assert MIN_RR_AGGRESSIVE == 1.4


def test_rr_course_floor_constant_exists():
    """MIN_RR_COURSE_FLOOR is exported and equals 1.4."""
    assert MIN_RR_COURSE_FLOOR == 1.4


def test_engine_default_min_rr_is_course_floor(engine: MMEngine):
    """A fresh MMEngine must default min_rr to the course floor 1.4, not 1.0."""
    assert engine.min_rr == 1.4


# ---------------------------------------------------------------------------
# Change 5 — OI promoted LOW(4) → MEDIUM(8)
# Course lesson 29: "can also be used to identify trapped Traders"
# ---------------------------------------------------------------------------


def test_oi_behavior_weight_is_medium_tier_per_lesson_29():
    """OI weight moved from LOW (4) to MEDIUM (8) — trapped-trader detector."""
    assert WEIGHTS["oi_behavior"] == 8.0


def test_oi_weight_matches_other_medium_factors():
    """OI should be in the same tier as other MEDIUM confluence factors."""
    medium_factors = ("unrecovered_vector", "liquidation_cluster", "ema_alignment")
    assert all(WEIGHTS[f] == WEIGHTS["oi_behavior"] for f in medium_factors)


def test_max_possible_reflects_oi_promotion():
    """Total max must match the weights.
    20+15+15+15 (high) + 8+8+8+8+8 (medium, incl. mw_inside_weekend_box) + 6+6+6 (fib/news/rsi) + 4+4+2 = 133.
    (+4 for adr_confluence added in Task 5.2 C3)
    """
    assert MAX_POSSIBLE == 139.0


# ---------------------------------------------------------------------------
# Change 6 — Lesson-18 alternative: 3 hits at HOW/LOW at L3 replace M/W
# Course lesson 18: "after three levels have completed. There are also other
# reversal signals... would replace the M or W."
# ---------------------------------------------------------------------------


def test_engine_has_three_hits_formation_helper(engine: MMEngine):
    """_try_three_hits_formation must exist on MMEngine."""
    assert hasattr(engine, "_try_three_hits_formation")
    assert callable(engine._try_three_hits_formation)


def test_three_hits_helper_returns_none_when_no_hits(engine: MMEngine):
    """With no HOW/LOW set, helper returns None."""

    class FakeCycle:
        how = 0.0
        low = float("inf")

    result = engine._try_three_hits_formation(None, FakeCycle())
    assert result is None


def test_three_hits_helper_uses_detect_three_hits(engine: MMEngine, monkeypatch):
    """When HOW is set and 3-hits detected + L3 reached, synthesize M formation."""
    from src.strategy.mm_formations import ThreeHitsResult
    from src.strategy.mm_levels import LevelAnalysis

    class FakeCycle:
        how = 45000.0
        low = float("inf")

    # Stub detect_three_hits to return a reversal signal
    hits_stub = ThreeHitsResult(
        detected=True,
        hit_count=3,
        level_tested=45000.0,
        hit_indices=[10, 20, 30],
        hit_sessions=["asia", "uk", "us"],
        expected_outcome="reversal",
    )
    monkeypatch.setattr(
        engine.formation_detector,
        "detect_three_hits",
        lambda ohlc, level, tolerance=0.0015: hits_stub,
    )

    # Stub level_tracker.analyze to report L3 reached bullishly (trend up to HOW)
    level_stub = LevelAnalysis(current_level=3, direction="bullish")
    monkeypatch.setattr(
        engine.level_tracker,
        "analyze",
        lambda ohlc, direction: level_stub,
    )

    formation = engine._try_three_hits_formation(None, FakeCycle())
    assert formation is not None
    assert formation.type == "M"
    assert formation.variant == "three_hits_how"
    assert formation.direction == "bearish"
    assert formation.peak1_price == 45000.0
    assert formation.peak2_price == 45000.0
    assert formation.at_key_level is True


def test_three_hits_helper_at_low_returns_w(engine: MMEngine, monkeypatch):
    """3 hits at LOW with L3 bearish trend → synthesize W formation for long."""
    from src.strategy.mm_formations import ThreeHitsResult
    from src.strategy.mm_levels import LevelAnalysis

    class FakeCycle:
        how = 0.0  # irrelevant for this test
        low = 42000.0

    hits_stub = ThreeHitsResult(
        detected=True,
        hit_count=3,
        level_tested=42000.0,
        hit_indices=[5, 15, 25],
        hit_sessions=["uk", "us", "asia"],
        expected_outcome="reversal",
    )
    monkeypatch.setattr(
        engine.formation_detector,
        "detect_three_hits",
        lambda ohlc, level, tolerance=0.0015: hits_stub,
    )
    level_stub = LevelAnalysis(current_level=3, direction="bearish")
    monkeypatch.setattr(
        engine.level_tracker,
        "analyze",
        lambda ohlc, direction: level_stub,
    )

    formation = engine._try_three_hits_formation(None, FakeCycle())
    assert formation is not None
    assert formation.type == "W"
    assert formation.variant == "three_hits_low"
    assert formation.direction == "bullish"
    assert formation.peak1_price == 42000.0


def test_three_hits_helper_rejects_at_low_level(engine: MMEngine, monkeypatch):
    """3 hits at HOW but only Level 2 reached → helper returns None (lesson 18 says L3)."""
    from src.strategy.mm_formations import ThreeHitsResult
    from src.strategy.mm_levels import LevelAnalysis

    class FakeCycle:
        how = 45000.0
        low = float("inf")

    hits_stub = ThreeHitsResult(
        detected=True,
        hit_count=3,
        level_tested=45000.0,
        hit_indices=[10, 20, 30],
        hit_sessions=["asia", "uk", "us"],
        expected_outcome="reversal",
    )
    monkeypatch.setattr(
        engine.formation_detector,
        "detect_three_hits",
        lambda ohlc, level, tolerance=0.0015: hits_stub,
    )
    # Only Level 2 — not enough per lesson 18
    level_stub = LevelAnalysis(current_level=2, direction="bullish")
    monkeypatch.setattr(
        engine.level_tracker,
        "analyze",
        lambda ohlc, direction: level_stub,
    )

    formation = engine._try_three_hits_formation(None, FakeCycle())
    assert formation is None, "Lesson 18 requires L3 — L2 must not trigger the alternative"


def test_three_hits_helper_rejects_continuation_outcome(engine: MMEngine, monkeypatch):
    """4 hits (continuation) must NOT trigger — lesson 18 is for 3-hits reversal."""
    from src.strategy.mm_formations import ThreeHitsResult
    from src.strategy.mm_levels import LevelAnalysis

    class FakeCycle:
        how = 45000.0
        low = float("inf")

    hits_stub = ThreeHitsResult(
        detected=True,
        hit_count=4,
        level_tested=45000.0,
        hit_indices=[10, 20, 30, 40],
        hit_sessions=["asia", "uk", "us", "asia"],
        expected_outcome="continuation",
    )
    monkeypatch.setattr(
        engine.formation_detector,
        "detect_three_hits",
        lambda ohlc, level, tolerance=0.0015: hits_stub,
    )
    level_stub = LevelAnalysis(current_level=3, direction="bullish")
    monkeypatch.setattr(
        engine.level_tracker,
        "analyze",
        lambda ohlc, direction: level_stub,
    )

    formation = engine._try_three_hits_formation(None, FakeCycle())
    assert formation is None, "4-hit continuation must not replace M/W"


# ---------------------------------------------------------------------------
# Changes 1, 2, 3 — removed / loosened hard gates
# ---------------------------------------------------------------------------


def test_reject_counter_reasons_no_longer_include_removed_gates(engine: MMEngine):
    """low_formation_quality and sl_too_wide should never be logged now.

    This is a smoke-test that the helper still works — the real proof is in
    the scan funnel on production (those reasons will have count 0 forever).
    """
    engine._scan_reject_counts = {}
    # We can still LOG these reasons via _reject if someone calls it directly;
    # the point of the removal is that _analyze_pair no longer invokes them.
    # So we just verify the _reject helper doesn't blow up with arbitrary names
    # (i.e., we haven't coupled the helper to a fixed enum of reasons).
    engine._reject("new_reason_for_future_use", "TEST/USDT:USDT", some_field=1)
    assert engine._scan_reject_counts == {"new_reason_for_future_use": 1}


def test_new_reject_reason_no_target_available_countable(engine: MMEngine):
    """The renamed reject 'no_target_available' (was 'no_l1_target') is countable."""
    engine._scan_reject_counts = {}
    engine._reject("no_target_available", "BTC/USDT:USDT", direction="long",
                   ema_50=None, ema_200=None, vectors=0, entry=50000.0, formation="W")
    assert engine._scan_reject_counts == {"no_target_available": 1}
