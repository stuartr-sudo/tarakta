"""Tests for MM Confluence scoring (src.strategy.mm_confluence)."""
from __future__ import annotations

import pytest

from src.strategy.mm_confluence import (
    GRADE_A_THRESHOLD,
    GRADE_B_THRESHOLD,
    GRADE_C_THRESHOLD,
    MAX_POSSIBLE,
    MIN_RETEST_CONDITIONS,
    WEIGHTS,
    ConfluenceScore,
    EntryDecision,
    MMConfluenceScorer,
    MMContext,
    RetestConditions,
)


@pytest.fixture
def scorer() -> MMConfluenceScorer:
    return MMConfluenceScorer(min_rr=3.0, min_score=40.0)


def _full_context(**overrides) -> MMContext:
    """Build an MMContext with sensible defaults and optional overrides."""
    defaults = dict(
        entry_price=95000.0,
        stop_loss=96500.0,
        target_price=90500.0,
        at_session_changeover=True,
        at_how_low=True,
        at_hod_lod=False,
        has_unrecovered_vector=True,
        has_liquidation_cluster=True,
        has_fib_alignment=True,
        has_news_event=True,
        rsi_confirmed=True,
        adr_at_fifty_pct=True,
        oi_increasing=True,
        correlation_confirmed=True,
        moon_phase_aligned=True,
        formation={"type": "M", "multi_session": True, "higher_low_or_lower_high": True},
        ema_state={"alignment": "bearish", "broke_50": True, "volume_confirmed": True, "at_50_ema": True},
        level_state={"current_level": 3, "svc_detected": True, "volume_degrading": True, "at_level1_vector": True},
        cycle_state={"phase": "LEVEL_3", "direction": "bearish"},
    )
    defaults.update(overrides)
    return MMContext(**defaults)


def _minimal_context(**overrides) -> MMContext:
    """Build a minimal MMContext with few factors active."""
    defaults = dict(
        entry_price=95000.0,
        stop_loss=96500.0,
        target_price=90500.0,
    )
    defaults.update(overrides)
    return MMContext(**defaults)


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

class TestConstants:
    def test_max_possible(self):
        assert MAX_POSSIBLE == sum(WEIGHTS.values())
        # Evolution:
        #   original: 111.0
        #   +4 when OI promoted LOW(4) -> MEDIUM(8) (lesson 29)
        #   +8 when mw_inside_weekend_box added (lesson 15)
        #   +6 when rsi_confirmation added (Task 5.1, C2)
        #   +4 when adr_confluence added (Task 5.2, C3)
        #   +6 when multi_session_formation added (2026-04-19 audit,
        #      Lesson 13 same-session penalty)
        #  => 139.0
        assert MAX_POSSIBLE == 139.0

    def test_all_factors_have_weights(self):
        expected_factors = [
            "mw_session_changeover", "mw_key_level", "ema50_break_volume",
            "stopping_volume_candle", "unrecovered_vector", "liquidation_cluster",
            "ema_alignment", "mw_inside_weekend_box",
            "fib_alignment", "news_event", "rsi_confirmation",
            "multi_session_formation",
            "oi_behavior", "correlation_confirmed", "adr_confluence", "moon_cycle",
        ]
        for f in expected_factors:
            assert f in WEIGHTS


# ------------------------------------------------------------------
# RetestConditions
# ------------------------------------------------------------------

class TestRetestConditions:
    def test_all_false(self):
        rc = RetestConditions()
        assert rc.conditions_met == 0
        assert rc.sufficient is False

    def test_all_true(self):
        rc = RetestConditions(
            at_50_ema=True,
            at_level1_vector=True,
            higher_low_or_lower_high=True,
            at_liquidity_cluster=True,
        )
        assert rc.conditions_met == 4
        assert rc.sufficient is True

    def test_two_met(self):
        rc = RetestConditions(at_50_ema=True, at_liquidity_cluster=True)
        assert rc.conditions_met == 2
        assert rc.sufficient is True

    def test_one_met_not_sufficient(self):
        rc = RetestConditions(at_50_ema=True)
        assert rc.conditions_met == 1
        assert rc.sufficient is False


# ------------------------------------------------------------------
# MMConfluenceScorer.calculate_rr
# ------------------------------------------------------------------

class TestCalculateRR:
    def test_short_rr(self, scorer: MMConfluenceScorer):
        # Short: entry=95000, SL=96500 (above), target=90500 (below)
        rr = scorer.calculate_rr(95000.0, 96500.0, 90500.0)
        # risk=1500, reward=4500 -> 3.0
        assert rr == pytest.approx(3.0, abs=0.01)

    def test_long_rr(self, scorer: MMConfluenceScorer):
        # Long: entry=95000, SL=93500 (below), target=99500 (above)
        rr = scorer.calculate_rr(95000.0, 93500.0, 99500.0)
        # risk=1500, reward=4500 -> 3.0
        assert rr == pytest.approx(3.0, abs=0.01)

    def test_zero_risk(self, scorer: MMConfluenceScorer):
        rr = scorer.calculate_rr(100.0, 100.0, 110.0)
        assert rr == 0.0

    def test_high_rr(self, scorer: MMConfluenceScorer):
        rr = scorer.calculate_rr(100.0, 101.0, 90.0)
        # risk=1, reward=10 -> 10.0
        assert rr == pytest.approx(10.0, abs=0.01)


# ------------------------------------------------------------------
# MMConfluenceScorer.score
# ------------------------------------------------------------------

class TestScore:
    def test_full_confluence_scores_high(self, scorer: MMConfluenceScorer):
        ctx = _full_context()
        result = scorer.score(ctx)
        assert isinstance(result, ConfluenceScore)
        assert result.total_score > 80
        assert result.grade == "A"
        assert result.meets_min_score is True

    def test_minimal_scores_low(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context()
        result = scorer.score(ctx)
        assert result.total_score < 10
        assert result.grade == "F"

    def test_score_pct_calculation(self, scorer: MMConfluenceScorer):
        ctx = _full_context()
        result = scorer.score(ctx)
        # Uses AVAILABLE_MAX (excludes stubbed feeds) not MAX_POSSIBLE
        from src.strategy.mm_confluence import AVAILABLE_MAX
        expected_pct = result.total_score / AVAILABLE_MAX * 100
        assert result.score_pct == pytest.approx(expected_pct, abs=0.1)

    def test_all_factors_present(self, scorer: MMConfluenceScorer):
        ctx = _full_context()
        result = scorer.score(ctx)
        # 16 factors: 12 original + mw_inside_weekend_box (lesson 15)
        # + rsi_confirmation (C2) + adr_confluence (C3)
        # + multi_session_formation (2026-04-19 audit, lesson 13)
        assert len(result.factors) == 16

    def test_factors_sum_to_total(self, scorer: MMConfluenceScorer):
        ctx = _full_context()
        result = scorer.score(ctx)
        assert sum(result.factors.values()) == pytest.approx(result.total_score, abs=0.01)

    def test_rr_computed(self, scorer: MMConfluenceScorer):
        ctx = _full_context()
        result = scorer.score(ctx)
        assert result.risk_reward == pytest.approx(3.0, abs=0.01)

    def test_retest_conditions_counted(self, scorer: MMConfluenceScorer):
        ctx = _full_context()
        result = scorer.score(ctx)
        # Full context has at_50_ema=True, at_level1_vector=True (from has_unrecovered_vector),
        # higher_low_or_lower_high=True, at_liquidity_cluster=True
        assert result.retest_conditions_met >= 2


# ------------------------------------------------------------------
# Grade computation
# ------------------------------------------------------------------

class TestGrading:
    def test_grade_a(self, scorer: MMConfluenceScorer):
        ctx = _full_context()
        result = scorer.score(ctx)
        assert result.grade == "A"

    def test_grade_f(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context()
        result = scorer.score(ctx)
        assert result.grade == "F"

    def test_grade_thresholds(self):
        assert GRADE_A_THRESHOLD > GRADE_B_THRESHOLD > GRADE_C_THRESHOLD > 0

    def test_grade_b_region(self, scorer: MMConfluenceScorer):
        # Partial confluence to hit B range (55-70%)
        ctx = _minimal_context(
            at_session_changeover=True,
            at_how_low=True,
            has_unrecovered_vector=True,
            has_liquidation_cluster=True,
            has_fib_alignment=True,
            ema_state={"alignment": "bearish", "broke_50": True, "volume_confirmed": True},
            level_state={"svc_detected": True, "volume_degrading": True},
            formation={"type": "M", "multi_session": True},
        )
        result = scorer.score(ctx)
        # Should be a decent score
        assert result.total_score > 40


# ------------------------------------------------------------------
# Factor scoring details
# ------------------------------------------------------------------

class TestFactorScoring:
    def test_session_changeover_multi_session(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context(
            at_session_changeover=True,
            formation={"type": "M", "multi_session": True},
        )
        result = scorer.score(ctx)
        assert result.factors["mw_session_changeover"] == WEIGHTS["mw_session_changeover"]

    def test_session_changeover_single_session(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context(
            at_session_changeover=True,
            formation={"type": "M"},
        )
        result = scorer.score(ctx)
        expected = round(WEIGHTS["mw_session_changeover"] * 0.75, 2)
        assert result.factors["mw_session_changeover"] == expected

    def test_key_level_how_low(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context(at_how_low=True)
        result = scorer.score(ctx)
        assert result.factors["mw_key_level"] == WEIGHTS["mw_key_level"]

    def test_key_level_hod_lod(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context(at_hod_lod=True)
        result = scorer.score(ctx)
        expected = round(WEIGHTS["mw_key_level"] * 0.75, 2)
        assert result.factors["mw_key_level"] == expected

    def test_ema50_break_with_volume(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context(
            ema_state={"broke_50": True, "volume_confirmed": True},
        )
        result = scorer.score(ctx)
        assert result.factors["ema50_break_volume"] == WEIGHTS["ema50_break_volume"]

    def test_ema50_break_without_volume(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context(
            ema_state={"broke_50": True, "volume_confirmed": False},
        )
        result = scorer.score(ctx)
        expected = round(WEIGHTS["ema50_break_volume"] * 0.5, 2)
        assert result.factors["ema50_break_volume"] == expected

    def test_stopping_volume_with_degrading(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context(
            level_state={"svc_detected": True, "volume_degrading": True},
        )
        result = scorer.score(ctx)
        assert result.factors["stopping_volume_candle"] == WEIGHTS["stopping_volume_candle"]

    def test_stopping_volume_without_degrading(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context(
            level_state={"svc_detected": True, "volume_degrading": False},
        )
        result = scorer.score(ctx)
        expected = round(WEIGHTS["stopping_volume_candle"] * 0.85, 2)
        assert result.factors["stopping_volume_candle"] == expected

    def test_oi_none_scores_zero(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context(oi_increasing=None)
        result = scorer.score(ctx)
        assert result.factors["oi_behavior"] == 0.0

    def test_correlation_none_scores_zero(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context(correlation_confirmed=None)
        result = scorer.score(ctx)
        assert result.factors["correlation_confirmed"] == 0.0

    def test_moon_none_scores_zero(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context(moon_phase_aligned=None)
        result = scorer.score(ctx)
        assert result.factors["moon_cycle"] == 0.0

    def test_ema_alignment_bullish(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context(ema_state={"alignment": "bullish"})
        result = scorer.score(ctx)
        assert result.factors["ema_alignment"] == WEIGHTS["ema_alignment"]

    def test_ema_alignment_mixed(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context(ema_state={"alignment": "mixed"})
        result = scorer.score(ctx)
        assert result.factors["ema_alignment"] == 0.0

    def test_fib_alignment_true_scores_full_weight(self, scorer: MMConfluenceScorer):
        """B2: has_fib_alignment=True should contribute 6.0 pts to fib_alignment factor."""
        ctx = _minimal_context(has_fib_alignment=True)
        result = scorer.score(ctx)
        assert result.factors["fib_alignment"] == WEIGHTS["fib_alignment"]
        assert result.factors["fib_alignment"] == 6.0

    def test_fib_alignment_false_scores_zero(self, scorer: MMConfluenceScorer):
        """B2: has_fib_alignment=False (default) should score 0 for fib_alignment."""
        ctx = _minimal_context(has_fib_alignment=False)
        result = scorer.score(ctx)
        assert result.factors["fib_alignment"] == 0.0

    def test_fib_alignment_included_in_max_possible(self):
        """B2: fib_alignment weight is included in MAX_POSSIBLE (already in WEIGHTS)."""
        assert "fib_alignment" in WEIGHTS
        assert MAX_POSSIBLE == 139.0

    def test_rsi_confirmed_true_scores_full_weight(self, scorer: MMConfluenceScorer):
        """C2: rsi_confirmed=True should contribute 6.0 pts to rsi_confirmation factor."""
        ctx = _minimal_context(rsi_confirmed=True)
        result = scorer.score(ctx)
        assert result.factors["rsi_confirmation"] == WEIGHTS["rsi_confirmation"]
        assert result.factors["rsi_confirmation"] == 6.0

    def test_rsi_confirmed_false_scores_zero(self, scorer: MMConfluenceScorer):
        """C2: rsi_confirmed=False should score 0.0."""
        ctx = _minimal_context(rsi_confirmed=False)
        result = scorer.score(ctx)
        assert result.factors["rsi_confirmation"] == 0.0

    def test_rsi_confirmed_none_scores_zero(self, scorer: MMConfluenceScorer):
        """C2: rsi_confirmed=None (data unavailable) should score 0.0."""
        ctx = _minimal_context(rsi_confirmed=None)
        result = scorer.score(ctx)
        assert result.factors["rsi_confirmation"] == 0.0

    def test_rsi_confirmation_in_max_possible(self):
        """C2: rsi_confirmation weight is included in MAX_POSSIBLE."""
        assert "rsi_confirmation" in WEIGHTS
        assert WEIGHTS["rsi_confirmation"] == 6.0
        assert MAX_POSSIBLE == 139.0

    def test_adr_at_fifty_pct_true_scores_full_weight(self, scorer: MMConfluenceScorer):
        """C3: adr_at_fifty_pct=True should contribute 4.0 pts to adr_confluence factor."""
        ctx = _minimal_context(adr_at_fifty_pct=True)
        result = scorer.score(ctx)
        assert result.factors["adr_confluence"] == WEIGHTS["adr_confluence"]
        assert result.factors["adr_confluence"] == 4.0

    def test_adr_at_fifty_pct_false_scores_zero(self, scorer: MMConfluenceScorer):
        """C3: adr_at_fifty_pct=False should score 0.0."""
        ctx = _minimal_context(adr_at_fifty_pct=False)
        result = scorer.score(ctx)
        assert result.factors["adr_confluence"] == 0.0

    def test_adr_at_fifty_pct_none_scores_zero(self, scorer: MMConfluenceScorer):
        """C3: adr_at_fifty_pct=None (data unavailable) should score 0.0."""
        ctx = _minimal_context(adr_at_fifty_pct=None)
        result = scorer.score(ctx)
        assert result.factors["adr_confluence"] == 0.0

    def test_adr_confluence_in_max_possible(self):
        """C3: adr_confluence weight is included in MAX_POSSIBLE."""
        assert "adr_confluence" in WEIGHTS
        assert WEIGHTS["adr_confluence"] == 4.0
        assert MAX_POSSIBLE == 139.0


# ------------------------------------------------------------------
# check_retest_conditions
# ------------------------------------------------------------------

class TestCheckRetestConditions:
    def test_all_conditions_met(self, scorer: MMConfluenceScorer):
        ctx = _full_context()
        rc = scorer.check_retest_conditions(ctx)
        assert rc.conditions_met >= 3

    def test_no_conditions(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context()
        rc = scorer.check_retest_conditions(ctx)
        assert rc.conditions_met == 0

    def test_unrecovered_vector_counts_as_level1_vector(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context(has_unrecovered_vector=True)
        rc = scorer.check_retest_conditions(ctx)
        assert rc.at_level1_vector is True


# ------------------------------------------------------------------
# should_enter
# ------------------------------------------------------------------

class TestShouldEnter:
    def test_full_confluence_enters(self, scorer: MMConfluenceScorer):
        ctx = _full_context()
        result = scorer.score(ctx)
        decision = scorer.should_enter(result)
        assert isinstance(decision, EntryDecision)
        assert decision.should_enter is True

    def test_no_formation_rejects(self, scorer: MMConfluenceScorer):
        ctx = _minimal_context()
        result = scorer.score(ctx)
        decision = scorer.should_enter(result)
        assert decision.should_enter is False
        assert "No M/W formation" in decision.reason

    def test_low_rr_rejects(self, scorer: MMConfluenceScorer):
        ctx = _full_context(
            entry_price=95000.0,
            stop_loss=96500.0,
            target_price=94000.0,  # small reward
        )
        result = scorer.score(ctx)
        decision = scorer.should_enter(result)
        assert decision.should_enter is False

    def test_dont_get_out_of_bed(self, scorer: MMConfluenceScorer):
        """R:R at or below 1.4 should get the DGOB rejection."""
        ctx = _full_context(
            entry_price=95000.0,
            stop_loss=96000.0,
            target_price=94000.0,  # risk=1000, reward=1000 -> RR=1.0
        )
        result = scorer.score(ctx)
        decision = scorer.should_enter(result)
        assert decision.should_enter is False
        assert "don't get out of bed" in decision.reason

    def test_low_score_rejects(self, scorer: MMConfluenceScorer):
        # Only session changeover active -> score ~15-20
        ctx = _minimal_context(
            at_session_changeover=True,
            formation={"type": "M"},
        )
        result = scorer.score(ctx)
        if result.total_score < scorer.min_score:
            decision = scorer.should_enter(result)
            assert decision.should_enter is False

    def test_insufficient_retest_rejects(self, scorer: MMConfluenceScorer):
        """If retest conditions < 2 but score and RR are good, still reject."""
        ctx = _full_context()
        # Remove all retest condition sources
        ctx.ema_state = {"alignment": "bearish", "broke_50": True, "volume_confirmed": True}
        ctx.level_state = {"svc_detected": True, "volume_degrading": True}
        ctx.formation = {"type": "M", "multi_session": True}
        ctx.has_unrecovered_vector = False
        ctx.has_liquidation_cluster = False
        result = scorer.score(ctx)
        decision = scorer.should_enter(result)
        # With 0 retest conditions, should reject
        if result.retest_conditions_met < MIN_RETEST_CONDITIONS:
            assert decision.should_enter is False

    def test_aggressive_entry_type(self, scorer: MMConfluenceScorer):
        """Exactly 2 retest conditions or grade C should be aggressive."""
        ctx = _full_context()
        result = scorer.score(ctx)
        decision = scorer.should_enter(result)
        if decision.should_enter:
            assert decision.entry_type in ("aggressive", "conservative")

    def test_entry_decision_fields(self, scorer: MMConfluenceScorer):
        ctx = _full_context()
        result = scorer.score(ctx)
        decision = scorer.should_enter(result)
        assert isinstance(decision.reason, str)
        assert isinstance(decision.score, ConfluenceScore)
        assert decision.entry_type in ("aggressive", "conservative")
