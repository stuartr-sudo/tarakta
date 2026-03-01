import pytest

from src.exchange.models import (
    FVGResult,
    FairValueGap,
    LiquidityResult,
    MarketStructureResult,
    OrderBlock,
    OrderBlockResult,
    SweepEvent,
)
from src.strategy.confluence import ConfluenceEngine


@pytest.fixture
def engine():
    return ConfluenceEngine(entry_threshold=65.0)


def _empty_ms(trend="ranging"):
    return MarketStructureResult(
        trend=trend,
        key_levels={"swing_high": 110, "swing_low": 90},
        last_bos_direction=1 if trend == "bullish" else -1 if trend == "bearish" else None,
        last_choch_direction=None,
        structure_strength=0.5,
    )


def _empty_liq():
    return LiquidityResult(
        active_pools=[], recent_sweeps=[], nearest_buy_liquidity=None,
        nearest_sell_liquidity=None, sweep_detected_recently=False,
    )


def _empty_ob():
    return OrderBlockResult(
        active_order_blocks=[], price_in_order_block=None,
        nearest_bullish_ob=None, nearest_bearish_ob=None,
    )


def _empty_fvg():
    return FVGResult(
        active_fvgs=[], price_in_fvg=None,
        nearest_bullish_fvg=None, nearest_bearish_fvg=None,
    )


class TestConfluenceEngine:
    def test_no_htf_trend_returns_zero(self, engine):
        ms = {tf: _empty_ms("ranging") for tf in ["15m", "1h", "4h", "1d"]}
        liq = {tf: _empty_liq() for tf in ["15m", "1h"]}
        ob = {tf: _empty_ob() for tf in ["15m", "1h"]}
        fvg = {tf: _empty_fvg() for tf in ["15m", "1h"]}

        signal = engine.score_signal("BTC/USD", 100.0, ms, liq, ob, fvg)
        assert signal.score == 0
        assert signal.direction is None

    def test_full_bullish_confluence(self, engine):
        """All signals aligned bullish — should get high score."""
        ms = {
            "15m": _empty_ms("bullish"),
            "1h": _empty_ms("bullish"),
            "4h": _empty_ms("bullish"),
            "1d": _empty_ms("bullish"),
        }
        ms["1h"].last_choch_direction = 1  # CHoCH confirms

        ob_in = OrderBlock(direction="bullish", top=101, bottom=99, volume=1000, strength=0.8, candle_idx=190)
        ob = {
            "15m": OrderBlockResult(
                active_order_blocks=[ob_in], price_in_order_block=ob_in,
                nearest_bullish_ob=ob_in, nearest_bearish_ob=None,
            ),
            "1h": _empty_ob(),
        }

        fvg_in = FairValueGap(direction="bullish", top=101, bottom=99, candle_idx=185, midpoint=100)
        fvg = {
            "15m": FVGResult(
                active_fvgs=[fvg_in], price_in_fvg=fvg_in,
                nearest_bullish_fvg=fvg_in, nearest_bearish_fvg=None,
            ),
            "1h": _empty_fvg(),
        }

        sweep = SweepEvent(level=98, direction="bullish_sweep", candle_idx=195)
        liq = {
            "15m": LiquidityResult(
                active_pools=[], recent_sweeps=[sweep],
                nearest_buy_liquidity=98, nearest_sell_liquidity=105,
                sweep_detected_recently=True,
            ),
            "1h": _empty_liq(),
        }

        signal = engine.score_signal("BTC/USD", 100.0, ms, liq, ob, fvg)
        assert signal.score >= 65  # should be well above threshold
        assert signal.direction == "bullish"
        assert len(signal.reasons) > 3

    def test_partial_confluence(self, engine):
        """Only HTF trend + market structure, no OB/FVG/liquidity."""
        ms = {
            "15m": _empty_ms("bullish"),
            "1h": _empty_ms("bullish"),
            "4h": _empty_ms("bullish"),
            "1d": _empty_ms("bullish"),
        }
        liq = {tf: _empty_liq() for tf in ["15m", "1h"]}
        ob = {tf: _empty_ob() for tf in ["15m", "1h"]}
        fvg = {tf: _empty_fvg() for tf in ["15m", "1h"]}

        signal = engine.score_signal("BTC/USD", 100.0, ms, liq, ob, fvg)
        # HTF (25) + MS (15) = 40, below threshold
        assert signal.score < 65
        assert signal.direction == "bullish"

    def test_mixed_htf(self, engine):
        """4H bullish, Daily ranging — should still get partial HTF score."""
        ms = {
            "15m": _empty_ms("bullish"),
            "1h": _empty_ms("bullish"),
            "4h": _empty_ms("bullish"),
            "1d": _empty_ms("ranging"),
        }
        liq = {tf: _empty_liq() for tf in ["15m", "1h"]}
        ob = {tf: _empty_ob() for tf in ["15m", "1h"]}
        fvg = {tf: _empty_fvg() for tf in ["15m", "1h"]}

        signal = engine.score_signal("BTC/USD", 100.0, ms, liq, ob, fvg)
        assert signal.score > 0
        assert signal.direction == "bullish"
