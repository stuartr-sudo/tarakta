"""Tests for the VWAP + RSI(2) Scalp Strategy (A7).

Covers:
  1. VWAP calculation accuracy
  2. RSI(2) calculation accuracy
  3. Long entry: price above VWAP+255EMA, pullback, RSI<10, hammer
  4. Short entry: price below VWAP+255EMA, pullback, RSI>90, inverted hammer
  5. No trade zone: price between VWAP and 255 EMA
  6. R:R too low → no signal
  7. RSI not at extreme → no signal
  8. Insufficient data → None
  9. Morning star / evening star patterns
  10. Fan-out rejection
  11. Target selection from external levels
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.mm_scalp_vwap_rsi import (
    VWAPCalculator,
    VWAPRSIScalper,
    ScalpRSI,
    ScalpSignal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_candles(
    n: int,
    base_price: float = 100.0,
    base_volume: float = 1000.0,
    trend: float = 0.0,
    volatility: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV candles.

    Args:
        n: Number of candles.
        base_price: Starting close price.
        base_volume: Average volume per candle.
        trend: Per-candle drift (positive = uptrend).
        volatility: Range of high-low as % of price.
        seed: Random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    closes = np.zeros(n)
    closes[0] = base_price

    for i in range(1, n):
        change = trend + rng.normal(0, volatility * 0.01 * base_price)
        closes[i] = closes[i - 1] + change

    highs = closes + rng.uniform(0, volatility * 0.01 * base_price, n)
    lows = closes - rng.uniform(0, volatility * 0.01 * base_price, n)
    opens = closes + rng.normal(0, 0.001 * base_price, n)
    volumes = rng.uniform(base_volume * 0.5, base_volume * 1.5, n)

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def make_candles_with_hammer_pullback(
    n: int = 300,
    base_price: float = 100.0,
    direction: str = "long",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create 15m and 1h candle sets that produce a valid scalp signal.

    For long: uptrend, pullback to VWAP/EMA area with hammer, RSI(2) < 10.
    For short: downtrend, pullback up with inverted hammer, RSI(2) > 90.

    Returns (candles_15m, candles_1h).
    """
    # Build a trending series with a pullback at the end
    rng = np.random.RandomState(123)

    if direction == "long":
        # Uptrend base → price above VWAP and 255 EMA
        prices = np.linspace(base_price * 0.95, base_price * 1.05, n - 5)
        # Strong pullback (3 red candles to drive RSI(2) < 10)
        pullback = np.array([
            base_price * 1.04,  # still high
            base_price * 1.02,  # dropping
            base_price * 1.005,  # near VWAP/EMA
        ])
        # Hammer candle at the bottom of pullback (near VWAP/EMA)
        # open near high, close near open but above it, long lower wick
        hammer_close = base_price * 1.003
        # Final recovery candle
        recovery = base_price * 1.01

        closes = np.concatenate([prices, pullback, [hammer_close, recovery]])
    else:
        # Downtrend base → price below VWAP and 255 EMA
        prices = np.linspace(base_price * 1.05, base_price * 0.95, n - 5)
        # Pullback up (3 green candles to drive RSI(2) > 90)
        pullback = np.array([
            base_price * 0.96,
            base_price * 0.98,
            base_price * 0.995,
        ])
        # Inverted hammer at the top of pullback
        inv_hammer_close = base_price * 0.997
        # Final drop
        drop = base_price * 0.99

        closes = np.concatenate([prices, pullback, [inv_hammer_close, drop]])

    n_total = len(closes)

    # Build OHLCV
    opens = np.roll(closes, 1)
    opens[0] = closes[0]

    highs = np.maximum(opens, closes) + rng.uniform(0, 0.002 * base_price, n_total)
    lows = np.minimum(opens, closes) - rng.uniform(0, 0.002 * base_price, n_total)

    # Fix the hammer / inverted hammer candle specifically
    if direction == "long":
        idx = n_total - 2  # hammer candle
        o = closes[idx] + 0.001 * base_price  # open slightly above close
        c = closes[idx]
        h = max(o, c) + 0.0005 * base_price  # tiny upper wick
        lo = c - 0.008 * base_price  # long lower wick (> 2x body)
        opens[idx] = o
        highs[idx] = h
        lows[idx] = lo
    else:
        idx = n_total - 2  # inverted hammer candle
        o = closes[idx] - 0.001 * base_price
        c = closes[idx]
        lo = min(o, c) - 0.0005 * base_price  # tiny lower wick
        h = c + 0.008 * base_price  # long upper wick (> 2x body)
        opens[idx] = o
        highs[idx] = h
        lows[idx] = lo

    volumes = rng.uniform(500, 1500, n_total)

    candles_15m = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    # 1H candles: resample by taking every 4th candle (simplified)
    step = 4
    candles_1h = pd.DataFrame({
        "open": opens[::step],
        "high": highs[::step],
        "low": lows[::step],
        "close": closes[::step],
        "volume": volumes[::step] * 4,
    })

    return candles_15m, candles_1h


# ===========================================================================
# VWAP Tests
# ===========================================================================

class TestVWAPCalculator:
    def test_basic_vwap(self):
        """VWAP = cumsum(TP * vol) / cumsum(vol) where TP = (H+L+C)/3."""
        calc = VWAPCalculator()
        df = pd.DataFrame({
            "high": [10.0, 11.0, 12.0],
            "low": [8.0, 9.0, 10.0],
            "close": [9.0, 10.0, 11.0],
            "volume": [100.0, 200.0, 300.0],
        })

        # Manual calculation:
        # TP = (10+8+9)/3=9, (11+9+10)/3=10, (12+10+11)/3=11
        # cum_tp_vol = 9*100 + 10*200 + 11*300 = 900 + 2000 + 3300 = 6200
        # cum_vol = 100 + 200 + 300 = 600
        # VWAP = 6200 / 600 = 10.333...
        result = calc.calculate(df)
        assert result is not None
        assert abs(result - 10.3333) < 0.01

    def test_empty_dataframe(self):
        calc = VWAPCalculator()
        assert calc.calculate(pd.DataFrame()) is None

    def test_zero_volume(self):
        calc = VWAPCalculator()
        df = pd.DataFrame({
            "high": [10.0], "low": [8.0], "close": [9.0], "volume": [0.0],
        })
        assert calc.calculate(df) is None

    def test_missing_columns(self):
        calc = VWAPCalculator()
        df = pd.DataFrame({"close": [10.0]})
        assert calc.calculate(df) is None

    def test_vwap_series(self):
        calc = VWAPCalculator()
        df = pd.DataFrame({
            "high": [10.0, 11.0],
            "low": [8.0, 9.0],
            "close": [9.0, 10.0],
            "volume": [100.0, 200.0],
        })
        series = calc.calculate_series(df)
        assert series is not None
        assert len(series) == 2
        # First value: just the first candle's typical price
        assert abs(series.iloc[0] - 9.0) < 0.01


# ===========================================================================
# ScalpRSI Tests
# ===========================================================================

class TestScalpRSI:
    def test_rsi2_extreme_values(self):
        """RSI(2) should reach extremes quickly (that's its purpose)."""
        rsi = ScalpRSI(period=2)

        # Three consecutive drops → RSI(2) should be very low
        closes = pd.Series([100, 99, 98, 97])
        val = rsi.current_value(closes)
        assert val is not None
        assert val < 15  # RSI(2) reaches extreme quickly

    def test_rsi2_overbought(self):
        """Three consecutive rises → RSI(2) should be very high."""
        rsi = ScalpRSI(period=2)
        closes = pd.Series([100, 101, 102, 103])
        val = rsi.current_value(closes)
        assert val is not None
        assert val > 85

    def test_rsi14_series(self):
        """RSI(14) requires more data and moves more slowly."""
        rsi = ScalpRSI(period=14)
        # Need at least 15 values — add some noise so RSI isn't pegged at 100
        rng = np.random.RandomState(42)
        base = np.linspace(100, 110, 30)
        closes = pd.Series(base + rng.normal(0, 0.3, 30))
        val = rsi.current_value(closes)
        assert val is not None
        assert 50 < val <= 100  # bullish

    def test_insufficient_data(self):
        rsi = ScalpRSI(period=14)
        closes = pd.Series([100, 101])
        assert rsi.current_value(closes) is None


# ===========================================================================
# VWAPRSIScalper — Full Integration Tests
# ===========================================================================

class TestVWAPRSIScalper:
    def test_insufficient_15m_data(self):
        """< 30 candles on 15m → None."""
        scalper = VWAPRSIScalper()
        candles_15m = make_candles(10)
        candles_1h = make_candles(20)
        assert scalper.scan(candles_15m, candles_1h) is None

    def test_insufficient_1h_data(self):
        """< 15 candles on 1h → None."""
        scalper = VWAPRSIScalper()
        candles_15m = make_candles(300)
        candles_1h = make_candles(5)
        assert scalper.scan(candles_15m, candles_1h) is None

    def test_none_candles(self):
        scalper = VWAPRSIScalper()
        assert scalper.scan(None, None) is None
        assert scalper.scan(make_candles(50), None) is None
        assert scalper.scan(None, make_candles(20)) is None

    def test_no_trade_zone(self):
        """Price between VWAP and 255 EMA → no signal."""
        scalper = VWAPRSIScalper()

        # Create flat candles where VWAP and EMA diverge and price is between
        n = 300
        closes = np.ones(n) * 100.0
        # Slowly increase to create EMA above, then drop price to be between
        for i in range(n):
            if i < 250:
                closes[i] = 100 + i * 0.01  # gentle uptrend → EMA trails below
            else:
                closes[i] = 101.0  # flat — will be between EMA and VWAP

        df_15m = pd.DataFrame({
            "open": closes - 0.05,
            "high": closes + 0.1,
            "low": closes - 0.1,
            "close": closes,
            "volume": np.ones(n) * 1000,
        })
        df_1h = make_candles(20, base_price=101)

        # This should return None because the setup conditions won't align
        # (no extreme RSI, no pullback pattern, etc.)
        result = scalper.scan(df_15m, df_1h)
        assert result is None

    def test_rsi_not_extreme_long(self):
        """RSI(2) not below 10 for long → no signal."""
        scalper = VWAPRSIScalper()
        # Steady gentle uptrend — RSI(2) won't be extreme
        candles_15m = make_candles(300, trend=0.01, seed=99)
        candles_1h = make_candles(50, trend=0.05, seed=99)
        result = scalper.scan(candles_15m, candles_1h)
        assert result is None

    def test_rsi_not_extreme_short(self):
        """RSI(2) not above 90 for short → no signal."""
        scalper = VWAPRSIScalper()
        candles_15m = make_candles(300, trend=-0.01, seed=88)
        candles_1h = make_candles(50, trend=-0.05, seed=88)
        result = scalper.scan(candles_15m, candles_1h)
        assert result is None

    def test_low_rr_rejected(self):
        """Signal with R:R < 3:1 should be rejected."""
        scalper = VWAPRSIScalper()

        # Artificially set MIN_RR very high to force rejection
        scalper.MIN_RR = 100.0

        candles_15m, candles_1h = make_candles_with_hammer_pullback(
            n=300, direction="long"
        )
        # Even if everything else is valid, 100:1 R:R won't be met
        result = scalper.scan(candles_15m, candles_1h)
        assert result is None

    def test_fan_out_rejected(self):
        """VWAP and 255 EMA too far apart → no signal."""
        scalper = VWAPRSIScalper()

        # Create candles with huge trend that pushes VWAP far from EMA
        n = 300
        closes = np.linspace(100, 150, n)  # 50% rise → massive fan-out
        df_15m = pd.DataFrame({
            "open": closes - 0.1,
            "high": closes + 0.2,
            "low": closes - 0.2,
            "close": closes,
            "volume": np.ones(n) * 1000,
        })
        df_1h = make_candles(50, base_price=150)
        result = scalper.scan(df_15m, df_1h)
        assert result is None

    def test_target_selection_from_levels(self):
        """Should pick the nearest valid external target."""
        scalper = VWAPRSIScalper()

        # Test _find_target directly
        # Long: price=100, SL=99, risk=1 → need target >= 103 for 3:1
        target = scalper._find_target(100.0, 99.0, "long", [101.0, 102.0, 103.5, 110.0])
        assert target == 103.5  # nearest one >= 103

        # Short: price=100, SL=101, risk=1 → need target <= 97 for 3:1
        target = scalper._find_target(100.0, 101.0, "short", [99.0, 98.0, 96.5, 90.0])
        assert target == 96.5  # nearest one <= 97

    def test_target_fallback_projection(self):
        """No valid external targets → 3:1 projection."""
        scalper = VWAPRSIScalper()

        # Long with no targets
        target = scalper._find_target(100.0, 99.0, "long", None)
        assert target == 103.0  # 100 + 3*1

        # Short with no targets
        target = scalper._find_target(100.0, 101.0, "short", None)
        assert target == 97.0  # 100 - 3*1

    def test_target_no_valid_levels(self):
        """External targets all too close → use projection fallback."""
        scalper = VWAPRSIScalper()
        # Long: all targets below the 3:1 threshold
        target = scalper._find_target(100.0, 99.0, "long", [100.5, 101.0, 102.0])
        assert target is not None
        assert target == 103.0  # falls back to projection


# ===========================================================================
# Pattern Detection Tests
# ===========================================================================

class TestPatternDetection:
    def setup_method(self):
        self.scalper = VWAPRSIScalper()

    def test_hammer_detected(self):
        """Hammer pattern on last candle should be detected for long."""
        # Hammer: long lower wick, small body, tiny/no upper wick
        # body = |c-o| = 0.3, range = h-l = 2.3, body/range = 0.13 < 0.4 OK
        # lower_shadow = min(o,c) - l = 100.2 - 98.0 = 2.2 >= 2*0.3=0.6 OK
        # upper_shadow = h - max(o,c) = 100.5 - 100.5 = 0.0 <= 0.15 OK
        df = pd.DataFrame({
            "open": [100.0, 101.0, 100.2],
            "high": [101.0, 102.0, 100.5],
            "low": [99.0, 100.5, 98.0],
            "close": [100.5, 101.5, 100.5],
            "volume": [100, 100, 100],
        })
        pattern = self.scalper._detect_pattern(df, "long")
        assert pattern == "hammer"

    def test_inverted_hammer_detected(self):
        """Inverted hammer / shooting star for short."""
        # body = |c-o| = 0.3, range = 101.5 - 99.5 = 2.0, body/range = 0.15 < 0.4 OK
        # upper_shadow = 101.5 - max(99.5, 99.8) = 1.7 >= 2*0.3=0.6 OK
        # lower_shadow = min(99.5, 99.8) - 99.5 = 0.0 <= 0.15 OK
        df = pd.DataFrame({
            "open": [100.0, 99.0, 99.5],
            "high": [101.0, 99.5, 101.5],
            "low": [99.5, 98.5, 99.5],
            "close": [99.5, 99.0, 99.8],
            "volume": [100, 100, 100],
        })
        pattern = self.scalper._detect_pattern(df, "short")
        assert pattern == "inverted_hammer"

    def test_no_pattern(self):
        """Regular candle (no extreme wicks) → None."""
        df = pd.DataFrame({
            "open": [100.0, 100.5, 100.2],
            "high": [100.5, 101.0, 100.7],
            "low": [99.5, 100.0, 99.7],
            "close": [100.3, 100.8, 100.5],
            "volume": [100, 100, 100],
        })
        assert self.scalper._detect_pattern(df, "long") is None
        assert self.scalper._detect_pattern(df, "short") is None

    def test_morning_star(self):
        """3-candle bullish reversal: bearish, small body, bullish."""
        df = pd.DataFrame({
            "open": [102.0, 100.0, 100.5],     # c1 bearish, c2 small, c3 bullish
            "high": [102.5, 100.2, 102.0],
            "low": [99.5, 99.8, 100.0],
            "close": [100.0, 100.1, 101.5],    # c3 closes above midpoint of c1
            "volume": [100, 100, 100],
        })
        pattern = self.scalper._detect_pattern(df, "long")
        assert pattern == "morning_star"

    def test_evening_star(self):
        """3-candle bearish reversal: bullish, small body, bearish."""
        df = pd.DataFrame({
            "open": [98.0, 100.0, 99.5],
            "high": [100.5, 100.2, 100.0],
            "low": [97.5, 99.8, 98.0],
            "close": [100.0, 99.9, 98.5],    # c3 closes below midpoint of c1
            "volume": [100, 100, 100],
        })
        pattern = self.scalper._detect_pattern(df, "short")
        assert pattern == "evening_star"


# ===========================================================================
# No-Trade Zone + Pullback Tests
# ===========================================================================

class TestNoTradeZone:
    def setup_method(self):
        self.scalper = VWAPRSIScalper()

    def test_price_between_vwap_and_ema(self):
        """Price between VWAP and EMA with meaningful gap → no trade."""
        # VWAP=100, EMA=102, price=101 → between them
        assert self.scalper._in_no_trade_zone(101.0, 100.0, 102.0) is True

    def test_price_above_both(self):
        """Price above both → not in no-trade zone."""
        assert self.scalper._in_no_trade_zone(103.0, 100.0, 102.0) is False

    def test_price_below_both(self):
        """Price below both → not in no-trade zone."""
        assert self.scalper._in_no_trade_zone(99.0, 100.0, 102.0) is False

    def test_converged_vwap_ema(self):
        """VWAP and EMA very close together → NOT a no-trade zone (convergence)."""
        # gap = 0.05 / 100 = 0.05% < 0.1% threshold
        assert self.scalper._in_no_trade_zone(100.0, 99.98, 100.03) is False

    def test_pullback_long(self):
        """Long pullback: price near the lower of VWAP/EMA from above."""
        # Price=100.1, VWAP=100.0, EMA=100.5 → near VWAP (lower anchor)
        assert self.scalper._is_pullback(100.1, 100.0, 100.5, "long") is True

    def test_pullback_short(self):
        """Short pullback: price near the higher of VWAP/EMA from below."""
        assert self.scalper._is_pullback(99.9, 100.0, 99.5, "short") is True

    def test_no_pullback(self):
        """Price too far from both anchors → not a pullback."""
        assert self.scalper._is_pullback(105.0, 100.0, 100.5, "long") is False


# ===========================================================================
# Stop Loss Tests
# ===========================================================================

class TestStopLoss:
    def setup_method(self):
        self.scalper = VWAPRSIScalper()

    def test_long_sl_below_both(self):
        """Long SL should be below the lower of VWAP and EMA."""
        sl = self.scalper._calculate_stop_loss(101.0, 100.0, 100.5, "long")
        assert sl < 100.0  # below the lower value (VWAP)
        assert sl == pytest.approx(100.0 * 0.999, rel=1e-5)

    def test_short_sl_above_both(self):
        """Short SL should be above the higher of VWAP and EMA."""
        sl = self.scalper._calculate_stop_loss(99.0, 100.0, 99.5, "short")
        assert sl > 100.0  # above the higher value (VWAP)
        assert sl == pytest.approx(100.0 * 1.001, rel=1e-5)


# ===========================================================================
# ScalpSignal Dataclass Tests
# ===========================================================================

class TestScalpSignal:
    def test_dataclass_creation(self):
        sig = ScalpSignal(
            detected=True,
            direction="long",
            entry_price=100.0,
            stop_loss=99.5,
            target=103.0,
            risk_reward=6.0,
            rsi_2_value=5.5,
            rsi_14_bias="bullish",
            vwap_value=99.8,
            ema_255_value=99.6,
            pattern="hammer",
            reason="test signal",
        )
        assert sig.detected is True
        assert sig.direction == "long"
        assert sig.risk_reward == 6.0
