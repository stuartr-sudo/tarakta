"""MCP tools that give the trade advisor access to Tarakta data."""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import tool, create_sdk_mcp_server

from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_advisor_tools(db, instance_id: str = "main"):
    """Create MCP tools with a captured DB connection.

    Returns an MCP server with tools for:
    - get_missed_signals: Fetch signals that were detected but never traded
    - simulate_outcome: Simulate what would have happened if a signal had been traded
    """

    @tool(
        "get_missed_signals",
        "Fetch trading signals that were detected but never acted on. "
        "Returns signals with score, symbol, direction, reasons, and price at detection.",
        {"min_score": float, "days_back": int, "limit": int},
    )
    async def get_missed_signals(args: dict[str, Any]) -> dict[str, Any]:
        from src.advisor.missed_signals import fetch_missed_signals

        signals = await fetch_missed_signals(
            db,
            instance_id=instance_id,
            min_score=args.get("min_score", 55.0),
            limit=args.get("limit", 30),
            days_back=args.get("days_back", 7),
        )
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(signals, default=str, indent=2),
                }
            ]
        }

    @tool(
        "simulate_outcome",
        "Simulate what would have happened if a missed signal had been traded. "
        "Provide entry_price, stop_loss, take_profit, direction, and the symbol/timeframe "
        "to fetch candles from. Returns outcome (tp_hit, sl_hit, expired), pnl_pct, candles_held.",
        {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair e.g. ETH/USDT:USDT",
                },
                "entry_price": {"type": "number"},
                "stop_loss": {"type": "number"},
                "stop_loss_pct": {
                    "type": "number",
                    "description": "SL as percentage from entry (e.g. 2.0 for 2%). Used if stop_loss not given.",
                },
                "take_profit": {"type": "number"},
                "take_profit_pct": {
                    "type": "number",
                    "description": "TP as percentage from entry (e.g. 4.0 for 4%). Used if take_profit not given.",
                },
                "direction": {"type": "string", "enum": ["long", "short"]},
                "signal_time": {
                    "type": "string",
                    "description": "ISO timestamp of when signal was detected",
                },
            },
            "required": ["symbol", "entry_price", "direction", "signal_time"],
        },
    )
    async def simulate_outcome(args: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        import pandas as pd

        from src.advisor.outcome_simulator import simulate_trade_outcome

        entry = args["entry_price"]
        direction = args["direction"]
        is_long = direction == "long"

        # Derive SL/TP from percentages if absolute values not given
        sl = args.get("stop_loss")
        if sl is None:
            sl_pct = args.get("stop_loss_pct", 2.0) / 100
            sl = entry * (1 - sl_pct) if is_long else entry * (1 + sl_pct)

        tp = args.get("take_profit")
        if tp is None:
            tp_pct = args.get("take_profit_pct", 4.0) / 100
            tp = entry * (1 + tp_pct) if is_long else entry * (1 - tp_pct)

        # Fetch candles from DB cache (we read from candle_cache table)
        signal_time = args["signal_time"]
        symbol = args["symbol"]

        def _exec(q):
            return q.execute()

        result = await asyncio.to_thread(
            _exec,
            db.table("candle_cache")
            .select("*")
            .eq("symbol", symbol)
            .eq("timeframe", "1h")
            .gte("timestamp", signal_time)
            .order("timestamp", desc=False)
            .limit(96),
        )
        rows = result.data or []

        if len(rows) < 5:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "error": f"Not enough cached candles for {symbol} after {signal_time}. "
                                f"Only {len(rows)} found. The bot may not have cached this pair yet."
                            }
                        ),
                    }
                ]
            }

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        outcome = simulate_trade_outcome(
            candles=df,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            direction=direction,
        )
        outcome["stop_loss_used"] = sl
        outcome["take_profit_used"] = tp

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(outcome, default=str, indent=2),
                }
            ]
        }

    server = create_sdk_mcp_server(
        name="tarakta_advisor",
        version="1.0.0",
        tools=[get_missed_signals, simulate_outcome],
    )
    return server
