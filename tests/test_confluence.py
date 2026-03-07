"""Tests for the PostSweepEngine (Trade Travel Chill confluence scoring)."""
import pytest

from src.exchange.models import MarketStructureResult, PullbackResult, SweepResult
from src.strategy.confluence import PostSweepEngine


@pytest.fixture
def engine():
    return PostSweepEngine(entry_threshold=70.0)


def _ms(trend="ranging"):
    return MarketStructureResult(
        trend=trend,
        key_levels={"swing_high": 110, "swing_low": 90},
        last_bos_direction=1 if trend == "bullish" else -1 if trend == "bearish" else None,
        last_choch_direction=None,
        structure_strength=0.5,
    )


def _no_sweep():
    return SweepResult(
        sweep_detected=False, sweep_direction=None, sweep_level=0.0,
        sweep_type=None, target_level=0.0, sweep_depth=0.0,
    )


def _bullish_sweep():
    return SweepResult(
        sweep_detected=True, sweep_direction="bullish",
        sweep_level=89.5, sweep_type="asian_low",
        target_level=110.0, sweep_depth=0.5,
    )


def _bearish_sweep():
    return SweepResult(
        sweep_detected=True, sweep_direction="bearish",
        sweep_level=110.5, sweep_type="asian_high",
        target_level=90.0, sweep_depth=0.5,
    )


def _valid_pullback(direction="bullish"):
    return PullbackResult(
        pullback_detected=True,
        retracement_pct=0.45,
        displacement_open=90.0 if direction == "bullish" else 110.0,
        thrust_extreme=115.0 if direction == "bullish" else 85.0,
        current_price=100.0,
        optimal_entry=100.0,
        pullback_status="optimal",
    )


def _waiting_pullback():
    return PullbackResult(
        pullback_detected=False,
        retracement_pct=0.10,
        displacement_open=90.0,
        thrust_extreme=115.0,
        current_price=112.0,
        optimal_entry=112.0,
        pullback_status="waiting",
    )


def _failed_pullback():
    return PullbackResult(
        pullback_detected=False,
        retracement_pct=0.85,
        displacement_open=90.0,
        thrust_extreme=115.0,
        current_price=93.75,
        optimal_entry=93.75,
        pullback_status="failed",
    )


class TestPostSweepEngine:
    def test_no_sweep_returns_zero(self, engine):
        """No sweep detected -> score=0, direction=None."""
        ms = {"4h": _ms("bullish"), "1d": _ms("bullish"), "1h": _ms("bullish")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=100.0,
            sweep_result=_no_sweep(),
            displacement_confirmed=True,
            displacement_direction="bullish",
            htf_direction="bullish",
            in_post_kill_zone=True,
            ms_results=ms,
            pullback_result=_valid_pullback(),
        )
        assert signal.score == 0
        assert signal.direction is None
        assert "No completed sweep detected" in signal.reasons

    def test_sweep_no_displacement_scores_all_components(self, engine):
        """Sweep without displacement still evaluates HTF/timing/pullback.

        Before: early return at 35.  Now: sweep(35) + pullback(10) + HTF(15) + timing(15) = 75.
        """
        ms = {"4h": _ms("bullish"), "1d": _ms("bullish"), "1h": _ms("bullish")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=100.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=False,
            displacement_direction=None,
            htf_direction="bullish",
            in_post_kill_zone=True,
            ms_results=ms,
            pullback_result=_valid_pullback(),
        )
        assert signal.score == 75  # 35 + 10 + 15 + 15
        assert signal.direction == "bullish"
        assert signal.components["sweep_detected"] == 35
        assert signal.components["displacement_confirmed"] == 0
        assert signal.components["pullback_confirmed"] == 10
        assert signal.components["htf_aligned"] == 15
        assert signal.components["timing_optimal"] == 15

    def test_sweep_no_displacement_no_bonus_stays_low(self, engine):
        """Sweep alone (no displacement, no HTF, no timing) = 35, below threshold."""
        ms = {"4h": _ms("ranging"), "1d": _ms("ranging"), "1h": _ms("ranging")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=100.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=False,
            displacement_direction=None,
            htf_direction=None,
            in_post_kill_zone=False,
            ms_results=ms,
            pullback_result=None,
        )
        assert signal.score == 35
        assert signal.direction == "bullish"
        assert signal.score < engine.entry_threshold

    def test_sweep_displacement_no_pullback_returns_60(self, engine):
        """Sweep + displacement but no pullback -> score=60, below threshold."""
        ms = {"4h": _ms("ranging"), "1d": _ms("ranging"), "1h": _ms("ranging")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=100.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=True,
            displacement_direction="bullish",
            htf_direction=None,
            in_post_kill_zone=False,
            ms_results=ms,
            pullback_result=None,
        )
        assert signal.score == 60
        assert signal.score < engine.entry_threshold

    def test_pullback_waiting_returns_60(self, engine):
        """Sweep + displacement + waiting pullback -> score=60, below threshold."""
        ms = {"4h": _ms("ranging"), "1d": _ms("ranging"), "1h": _ms("ranging")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=100.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=True,
            displacement_direction="bullish",
            htf_direction=None,
            in_post_kill_zone=False,
            ms_results=ms,
            pullback_result=_waiting_pullback(),
        )
        assert signal.score == 60
        assert "Pullback pending" in " ".join(signal.reasons)

    def test_pullback_failed_returns_60(self, engine):
        """Sweep + displacement + failed pullback -> score=60, below threshold."""
        ms = {"4h": _ms("ranging"), "1d": _ms("ranging"), "1h": _ms("ranging")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=100.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=True,
            displacement_direction="bullish",
            htf_direction=None,
            in_post_kill_zone=False,
            ms_results=ms,
            pullback_result=_failed_pullback(),
        )
        assert signal.score == 60
        assert "too deep" in " ".join(signal.reasons).lower()

    def test_sweep_displacement_pullback_hits_threshold(self, engine):
        """Sweep + displacement + pullback = 70 points, meets threshold."""
        ms = {"4h": _ms("ranging"), "1d": _ms("ranging"), "1h": _ms("ranging")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=105.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=True,
            displacement_direction="bullish",
            htf_direction=None,
            in_post_kill_zone=False,
            ms_results=ms,
            pullback_result=_valid_pullback(),
        )
        assert signal.score == 70
        assert signal.direction == "bullish"
        assert signal.score >= engine.entry_threshold
        # Entry price should be from pullback, not original
        assert signal.entry_price == 100.0

    def test_full_setup_all_bonuses(self, engine):
        """Sweep + displacement + pullback + HTF + post-KZ = 100 points."""
        ms = {"4h": _ms("bullish"), "1d": _ms("bullish"), "1h": _ms("bullish")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=105.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=True,
            displacement_direction="bullish",
            htf_direction="bullish",
            in_post_kill_zone=True,
            ms_results=ms,
            pullback_result=_valid_pullback(),
        )
        assert signal.score == 100
        assert signal.direction == "bullish"

    def test_displacement_wrong_direction_no_displacement_pts(self, engine):
        """Displacement exists but opposite to sweep direction -> no displacement pts.

        Still scores HTF/timing/pullback though (no early return).
        sweep(35) + pullback(10) + HTF(15) + timing(15) = 75.
        """
        ms = {"4h": _ms("bullish"), "1d": _ms("bullish"), "1h": _ms("bullish")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=100.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=True,
            displacement_direction="bearish",
            htf_direction="bullish",
            in_post_kill_zone=True,
            ms_results=ms,
            pullback_result=_valid_pullback(),
        )
        assert signal.score == 75  # No displacement pts, but everything else scores
        assert signal.components["displacement_confirmed"] == 0
        assert "mismatch" in " ".join(signal.reasons).lower()

    def test_bearish_full_setup(self, engine):
        """Full bearish setup works correctly."""
        ms = {"4h": _ms("bearish"), "1d": _ms("bearish"), "1h": _ms("bearish")}
        signal = engine.score_signal(
            symbol="ETH/USD", current_price=100.0,
            sweep_result=_bearish_sweep(),
            displacement_confirmed=True,
            displacement_direction="bearish",
            htf_direction="bearish",
            in_post_kill_zone=True,
            ms_results=ms,
            pullback_result=_valid_pullback(direction="bearish"),
        )
        assert signal.score == 100
        assert signal.direction == "bearish"

    def test_partial_htf_alignment(self, engine):
        """Only 4H aligned, not Daily -> partial HTF bonus."""
        ms = {"4h": _ms("bullish"), "1d": _ms("ranging"), "1h": _ms("bullish")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=105.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=True,
            displacement_direction="bullish",
            htf_direction="bullish",
            in_post_kill_zone=False,
            ms_results=ms,
            pullback_result=_valid_pullback(),
        )
        # Sweep(35) + Displacement(25) + Pullback(10) + partial HTF(~10) = ~80
        assert 70 < signal.score < 100
        assert signal.direction == "bullish"

    def test_key_levels_collected(self, engine):
        """Key levels from market structure are collected for SL/TP."""
        ms = {"4h": _ms("bullish"), "1d": _ms("bullish"), "1h": _ms("bullish")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=105.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=True,
            displacement_direction="bullish",
            htf_direction="bullish",
            in_post_kill_zone=True,
            ms_results=ms,
            pullback_result=_valid_pullback(),
        )
        assert "4h_swing_high" in signal.key_levels
        assert "1h_swing_low" in signal.key_levels
