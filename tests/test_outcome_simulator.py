from __future__ import annotations

import pandas as pd

from src.advisor.outcome_simulator import simulate_trade_outcome


def _make_candles(prices: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build OHLCV DataFrame from (open, high, low, close) tuples."""
    data = {
        "open": [p[0] for p in prices],
        "high": [p[1] for p in prices],
        "low": [p[2] for p in prices],
        "close": [p[3] for p in prices],
        "volume": [10000.0] * len(prices),
    }
    ts = pd.date_range("2026-03-20T12:00", periods=len(prices), freq="1h", tz="UTC")
    return pd.DataFrame(data, index=ts)


def test_simulate_long_winner():
    candles = _make_candles(
        [
            (100, 102, 99, 101),
            (101, 103, 100, 102),
            (102, 106, 101, 105),
            (105, 112, 104, 110),
        ]
    )
    result = simulate_trade_outcome(
        candles=candles,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        direction="long",
    )
    assert result["outcome"] == "tp_hit"
    assert result["pnl_pct"] > 0
    assert result["exit_price"] == 110.0


def test_simulate_long_loser():
    candles = _make_candles(
        [
            (100, 101, 99, 100),
            (100, 100, 96, 97),
            (97, 98, 94, 95),
        ]
    )
    result = simulate_trade_outcome(
        candles=candles,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        direction="long",
    )
    assert result["outcome"] == "sl_hit"
    assert result["pnl_pct"] < 0


def test_simulate_expired():
    candles = _make_candles(
        [
            (100, 102, 98, 101),
            (101, 103, 99, 100),
            (100, 104, 97, 102),
        ]
    )
    result = simulate_trade_outcome(
        candles=candles,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=115.0,
        direction="long",
    )
    assert result["outcome"] == "expired"


def test_simulate_short_winner():
    candles = _make_candles(
        [
            (100, 101, 98, 99),
            (99, 100, 93, 94),
            (94, 95, 89, 90),
        ]
    )
    result = simulate_trade_outcome(
        candles=candles,
        entry_price=100.0,
        stop_loss=105.0,
        take_profit=90.0,
        direction="short",
    )
    assert result["outcome"] == "tp_hit"
    assert result["pnl_pct"] > 0
