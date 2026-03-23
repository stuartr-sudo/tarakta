"""Simulate what would have happened if a missed signal had been traded."""

from __future__ import annotations

from typing import Any

import pandas as pd


def simulate_trade_outcome(
    candles: pd.DataFrame,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
    max_candles: int = 48,
) -> dict[str, Any]:
    """Walk forward through candles and check if SL or TP would have been hit.

    Args:
        candles: OHLCV DataFrame starting from entry time.
        entry_price: Simulated entry price.
        stop_loss: Stop loss price.
        take_profit: Take profit price.
        direction: "long" or "short".
        max_candles: Maximum candles to simulate (default 48 = 48 hours on 1H).

    Returns:
        Dict with outcome, exit_price, pnl_pct, candles_held, exit_candle_time.
    """
    is_long = direction == "long"

    for i, (ts, row) in enumerate(candles.iterrows()):
        if i >= max_candles:
            break

        high = float(row["high"])
        low = float(row["low"])

        # Check SL hit
        sl_hit = (low <= stop_loss) if is_long else (high >= stop_loss)
        # Check TP hit
        tp_hit = (high >= take_profit) if is_long else (low <= take_profit)

        if sl_hit and tp_hit:
            # Both hit in same candle — assume SL hit first (conservative)
            exit_price = stop_loss
            pnl_pct = (
                ((exit_price - entry_price) / entry_price)
                if is_long
                else ((entry_price - exit_price) / entry_price)
            )
            return {
                "outcome": "sl_hit",
                "exit_price": exit_price,
                "pnl_pct": round(pnl_pct * 100, 2),
                "candles_held": i + 1,
                "exit_candle_time": str(ts),
            }

        if sl_hit:
            exit_price = stop_loss
            pnl_pct = (
                ((exit_price - entry_price) / entry_price)
                if is_long
                else ((entry_price - exit_price) / entry_price)
            )
            return {
                "outcome": "sl_hit",
                "exit_price": exit_price,
                "pnl_pct": round(pnl_pct * 100, 2),
                "candles_held": i + 1,
                "exit_candle_time": str(ts),
            }

        if tp_hit:
            exit_price = take_profit
            pnl_pct = (
                ((exit_price - entry_price) / entry_price)
                if is_long
                else ((entry_price - exit_price) / entry_price)
            )
            return {
                "outcome": "tp_hit",
                "exit_price": exit_price,
                "pnl_pct": round(pnl_pct * 100, 2),
                "candles_held": i + 1,
                "exit_candle_time": str(ts),
            }

    # Neither hit — trade would have expired
    last_close = float(candles.iloc[min(len(candles) - 1, max_candles - 1)]["close"])
    pnl_pct = (
        ((last_close - entry_price) / entry_price)
        if is_long
        else ((entry_price - last_close) / entry_price)
    )
    return {
        "outcome": "expired",
        "exit_price": last_close,
        "pnl_pct": round(pnl_pct * 100, 2),
        "candles_held": min(len(candles), max_candles),
        "exit_candle_time": str(candles.index[min(len(candles) - 1, max_candles - 1)]),
    }
