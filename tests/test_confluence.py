"""Tests for the PostSweepEngine (Trade Travel Chill confluence scoring)."""
import pytest

from src.exchange.models import MarketStructureResult, SweepResult
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
        )
        assert signal.score == 0
        assert signal.direction is None
        assert "No completed sweep detected" in signal.reasons

    def test_sweep_no_displacement_returns_40(self, engine):
        """Sweep detected but no displacement -> score=40, below threshold."""
        ms = {"4h": _ms("bullish"), "1d": _ms("bullish"), "1h": _ms("bullish")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=100.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=False,
            displacement_direction=None,
            htf_direction="bullish",
            in_post_kill_zone=True,
            ms_results=ms,
        )
        assert signal.score == 40
        assert signal.direction == "bullish"
        assert signal.score < engine.entry_threshold

    def test_sweep_plus_displacement_hits_threshold(self, engine):
        """Sweep + displacement = 70 points, meets threshold."""
        ms = {"4h": _ms("ranging"), "1d": _ms("ranging"), "1h": _ms("ranging")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=100.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=True,
            displacement_direction="bullish",
            htf_direction=None,
            in_post_kill_zone=False,
            ms_results=ms,
        )
        assert signal.score == 70
        assert signal.direction == "bullish"
        assert signal.score >= engine.entry_threshold

    def test_full_setup_all_bonuses(self, engine):
        """Sweep + displacement + HTF + post-KZ = 100 points."""
        ms = {"4h": _ms("bullish"), "1d": _ms("bullish"), "1h": _ms("bullish")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=100.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=True,
            displacement_direction="bullish",
            htf_direction="bullish",
            in_post_kill_zone=True,
            ms_results=ms,
        )
        assert signal.score == 100
        assert signal.direction == "bullish"

    def test_displacement_wrong_direction_rejected(self, engine):
        """Displacement exists but opposite to sweep direction -> below threshold."""
        ms = {"4h": _ms("bullish"), "1d": _ms("bullish"), "1h": _ms("bullish")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=100.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=True,
            displacement_direction="bearish",  # Wrong direction!
            htf_direction="bullish",
            in_post_kill_zone=True,
            ms_results=ms,
        )
        # Only gets sweep points (40), not displacement (30)
        assert signal.score == 40
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
        )
        assert signal.score == 100
        assert signal.direction == "bearish"

    def test_partial_htf_alignment(self, engine):
        """Only 4H aligned, not Daily -> partial HTF bonus."""
        ms = {"4h": _ms("bullish"), "1d": _ms("ranging"), "1h": _ms("bullish")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=100.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=True,
            displacement_direction="bullish",
            htf_direction="bullish",
            in_post_kill_zone=False,
            ms_results=ms,
        )
        # Sweep(40) + Displacement(30) + partial HTF(~10) = ~80
        assert 70 < signal.score < 100
        assert signal.direction == "bullish"

    def test_key_levels_collected(self, engine):
        """Key levels from market structure are collected for SL/TP."""
        ms = {"4h": _ms("bullish"), "1d": _ms("bullish"), "1h": _ms("bullish")}
        signal = engine.score_signal(
            symbol="BTC/USD", current_price=100.0,
            sweep_result=_bullish_sweep(),
            displacement_confirmed=True,
            displacement_direction="bullish",
            htf_direction="bullish",
            in_post_kill_zone=True,
            ms_results=ms,
        )
        assert "4h_swing_high" in signal.key_levels
        assert "1h_swing_low" in signal.key_levels
