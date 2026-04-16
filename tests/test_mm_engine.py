"""Tests for MM Engine core behaviour.

B3: EMA fan-out detection at Level 3.
  Course citation: lessons 12 and 18 — "EMA fan-out at Level 3 = imminent reversal."

B1: 2-hour scratch rule.
  Course citation: "if you don't see movement within 2 hours, it's a scratch."
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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


# ---------------------------------------------------------------------------
# B1 — 2-hour scratch rule
# ---------------------------------------------------------------------------


def _make_stale_position(symbol: str = "BTC/USDT", current_level: int = 0, hours_old: float = 3.0) -> MMPosition:
    """Build an MMPosition that has been open for `hours_old` hours."""
    pos = MMPosition(
        symbol=symbol,
        direction="long",
        entry_price=50000.0,
        quantity=0.01,
        stop_loss=49000.0,
        current_level=current_level,
    )
    pos.entry_time = datetime.now(timezone.utc) - timedelta(hours=hours_old)
    return pos


@pytest.mark.asyncio
async def test_scratch_2h_triggers_close_when_no_level_hit(engine: MMEngine):
    """B1: position open >2h with current_level=0 must close with reason scratch_2h."""
    pos = _make_stale_position(current_level=0, hours_old=3.0)
    engine.positions["BTC/USDT"] = pos

    # Mock exchange.fetch_ticker so _manage_position can get a price
    engine.exchange = MagicMock()
    engine.exchange.fetch_ticker = AsyncMock(return_value={"last": 50100.0})

    close_calls: list[tuple] = []

    async def _fake_close(p, price, reason):
        close_calls.append((p, price, reason))
        engine.positions.pop(p.symbol, None)

    # Patch early-exit checks that run before the scratch rule
    with patch.object(engine, "_close_position", side_effect=_fake_close):
        # Also stub the SVC check that would otherwise need candle data
        engine.candle_manager = MagicMock()
        engine.candle_manager.get_candles = AsyncMock(return_value=None)
        await engine._manage_position("BTC/USDT")

    assert len(close_calls) == 1, f"Expected exactly one close call, got {close_calls}"
    _, _, reason = close_calls[0]
    assert reason == "scratch_2h", f"Expected reason='scratch_2h', got '{reason}'"


@pytest.mark.asyncio
async def test_scratch_2h_does_not_trigger_when_level_hit(engine: MMEngine):
    """B1: position open >2h but current_level >= 1 must NOT trigger the scratch rule."""
    pos = _make_stale_position(current_level=1, hours_old=3.0)
    engine.positions["BTC/USDT"] = pos

    engine.exchange = MagicMock()
    engine.exchange.fetch_ticker = AsyncMock(return_value={"last": 50100.0})

    scratch_calls: list[str] = []

    async def _spy_close(p, price, reason):
        if reason == "scratch_2h":
            scratch_calls.append(reason)

    with patch.object(engine, "_close_position", side_effect=_spy_close):
        engine.candle_manager = MagicMock()
        engine.candle_manager.get_candles = AsyncMock(return_value=None)
        # _is_stopped_out needs to return False so we don't close via stop_loss
        with patch.object(engine, "_is_stopped_out", return_value=False):
            await engine._manage_position("BTC/USDT")

    assert len(scratch_calls) == 0, "scratch_2h must not fire when current_level >= 1"


@pytest.mark.asyncio
async def test_scratch_2h_does_not_trigger_when_under_2h(engine: MMEngine):
    """B1: position open <2h with current_level=0 must NOT trigger the scratch rule."""
    pos = _make_stale_position(current_level=0, hours_old=1.5)
    engine.positions["BTC/USDT"] = pos

    engine.exchange = MagicMock()
    engine.exchange.fetch_ticker = AsyncMock(return_value={"last": 50100.0})

    scratch_calls: list[str] = []

    async def _spy_close(p, price, reason):
        if reason == "scratch_2h":
            scratch_calls.append(reason)

    with patch.object(engine, "_close_position", side_effect=_spy_close):
        engine.candle_manager = MagicMock()
        engine.candle_manager.get_candles = AsyncMock(return_value=None)
        with patch.object(engine, "_is_stopped_out", return_value=False):
            await engine._manage_position("BTC/USDT")

    assert len(scratch_calls) == 0, "scratch_2h must not fire when trade is under 2 hours old"
