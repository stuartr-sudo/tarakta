"""Tests for MM Engine core behaviour — B3: EMA fan-out detection at Level 3.

Course citation: lessons 12 and 18 — "EMA fan-out at Level 3 = imminent reversal."
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.strategy.mm_engine import MMEngine, MMPosition


@pytest.fixture
def engine() -> MMEngine:
    return MMEngine(exchange=None, repo=None, candle_manager=None, config=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fan_out_candles(n: int = 300) -> pd.DataFrame:
    """Build 1H candles where the 10 EMA has fanned far away from the 200 EMA.

    Strategy: start flat for 200 bars then rocket upward for the last 100 bars.
    This ensures:
      - prior_median (bars -200:-50) reflects a small spread,
      - current spread (bar -1) is many multiples larger,
      - current_pct > 0.02 (2% of price).
    """
    flat_n = 200
    rocket_n = n - flat_n
    base_price = 1000.0

    flat_prices = np.full(flat_n, base_price)
    # Strong rocket: ~0.5% per bar compounded → big fan-out
    rocket_prices = base_price * np.cumprod(np.full(rocket_n, 1.005))

    closes = np.concatenate([flat_prices, rocket_prices])
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": np.ones(n) * 1000,
        },
        index=idx,
    )


def _make_flat_candles(n: int = 300) -> pd.DataFrame:
    """Build 1H candles with no trend — EMAs are not fanned out."""
    closes = np.full(n, 1000.0)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": np.ones(n) * 1000,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# B3 — _detect_ema_fan_out unit tests
# ---------------------------------------------------------------------------


def test_detect_ema_fan_out_returns_false_on_none(engine: MMEngine):
    """_detect_ema_fan_out returns False when candles is None."""
    assert engine._detect_ema_fan_out(None) is False


def test_detect_ema_fan_out_returns_false_on_insufficient_data(engine: MMEngine):
    """_detect_ema_fan_out returns False when fewer than 250 candles."""
    small = _make_fan_out_candles(n=200)
    assert engine._detect_ema_fan_out(small) is False


def test_detect_ema_fan_out_returns_false_on_flat_market(engine: MMEngine):
    """Flat market → EMAs don't fan out → returns False."""
    flat = _make_flat_candles(n=300)
    result = engine._detect_ema_fan_out(flat)
    assert result is False


def test_detect_ema_fan_out_returns_true_on_strong_trend(engine: MMEngine):
    """Strong fan-out candles → _detect_ema_fan_out returns True."""
    candles = _make_fan_out_candles(n=300)
    result = engine._detect_ema_fan_out(candles)
    assert result is True


# ---------------------------------------------------------------------------
# B3 — EMA fan-out WARNING logged when position is at L3
# ---------------------------------------------------------------------------


def test_ema_fan_out_l3_warning_logged(engine: MMEngine):
    """B3 (lessons 12, 18): EMA_FAN_OUT_L3_WARNING must be logged when position
    is at Level 3 and _detect_ema_fan_out returns True.

    The warning is informational only — it does NOT close the trade;
    existing SVC / vol-degradation checks own that decision.
    """
    candles = _make_fan_out_candles(n=300)

    # Confirm our candles actually trigger fan-out detection
    assert engine._detect_ema_fan_out(candles) is True, (
        "Precondition failed: test candles did not trigger fan-out"
    )

    with patch("src.strategy.mm_engine.logger") as mock_logger:
        engine._maybe_log_ema_fan_out_warning(
            pos=MMPosition(symbol="BTC/USDT", current_level=3),
            candles_1h=candles,
        )

    calls = [str(call) for call in mock_logger.info.call_args_list]
    assert any("EMA_FAN_OUT_L3_WARNING" in c for c in calls), (
        f"Expected EMA_FAN_OUT_L3_WARNING in logger.info calls, got: {calls}"
    )


def test_ema_fan_out_warning_not_logged_below_l3(engine: MMEngine):
    """EMA_FAN_OUT_L3_WARNING must NOT be logged when position is below Level 3."""
    candles = _make_fan_out_candles(n=300)

    with patch("src.strategy.mm_engine.logger") as mock_logger:
        engine._maybe_log_ema_fan_out_warning(
            pos=MMPosition(symbol="BTC/USDT", current_level=2),
            candles_1h=candles,
        )

    calls = [str(call) for call in mock_logger.info.call_args_list]
    assert not any("EMA_FAN_OUT_L3_WARNING" in c for c in calls), (
        "EMA_FAN_OUT_L3_WARNING must not fire below Level 3"
    )


def test_ema_fan_out_warning_not_logged_when_no_fan_out(engine: MMEngine):
    """EMA_FAN_OUT_L3_WARNING must NOT be logged when EMAs are not fanned out,
    even if the position is at Level 3.
    """
    flat = _make_flat_candles(n=300)

    with patch("src.strategy.mm_engine.logger") as mock_logger:
        engine._maybe_log_ema_fan_out_warning(
            pos=MMPosition(symbol="BTC/USDT", current_level=3),
            candles_1h=flat,
        )

    calls = [str(call) for call in mock_logger.info.call_args_list]
    assert not any("EMA_FAN_OUT_L3_WARNING" in c for c in calls), (
        "EMA_FAN_OUT_L3_WARNING must not fire when EMAs are flat"
    )
