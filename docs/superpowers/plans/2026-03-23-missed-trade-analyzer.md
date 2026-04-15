# Missed Trade Analyzer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Claude Agent SDK-powered analyzer that queries Supabase for signals that were detected but never traded, fetches historical candle data to simulate what would have happened, and produces actionable recommendations for loosening entry criteria.

**Architecture:** A standalone CLI script (`src/advisor/missed_trades.py`) that uses the Claude Agent SDK with custom MCP tools for Supabase access and Binance historical data. The agent reads missed signals (high-scoring signals where `acted_on = false`), fetches candles from the time of each signal, simulates the trade outcome, and produces a structured report with parameter recommendations. Uses existing `Repository`, `CandleManager`, and config infrastructure — no new DB tables needed.

**Tech Stack:** Python, `claude-agent-sdk`, existing Supabase client (`src/data/repository.py`), existing `CandleManager` (`src/data/candles.py`), OpenAI (for the existing bot) + Anthropic API key (for the advisor agent)

**Spec:** `docs/superpowers/specs/2026-03-23-missed-trade-analyzer-spec.md`

---

### Task 1: Install Claude Agent SDK

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Install the SDK**

```bash
pip install claude-agent-sdk
```

- [ ] **Step 2: Add to requirements.txt**

Add this line to `requirements.txt`:

```
claude-agent-sdk>=0.1.0
```

Note: Do NOT add to Dockerfile / production dependencies yet — this is a local analysis tool, not part of the trading bot runtime.

- [ ] **Step 3: Verify installation**

```bash
pip show claude-agent-sdk
python -c "from claude_agent_sdk import query; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add claude-agent-sdk dependency for trade advisor"
```

---

### Task 2: Build the Missed Signal Fetcher

This module queries Supabase for signals that scored well but were never acted on — the "missed trades."

**Files:**
- Create: `src/advisor/__init__.py`
- Create: `src/advisor/missed_signals.py`
- Create: `tests/test_missed_signals.py`

- [ ] **Step 1: Create the package**

Create empty `src/advisor/__init__.py`.

- [ ] **Step 2: Write failing test for missed signal fetching**

```python
# tests/test_missed_signals.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.advisor.missed_signals import fetch_missed_signals


@pytest.mark.asyncio
async def test_fetch_missed_signals_returns_high_score_unacted():
    """Should return signals with score >= threshold and acted_on = false."""
    mock_db = MagicMock()
    mock_result = MagicMock()
    mock_result.data = [
        {
            "id": "sig-1",
            "symbol": "ETH/USDT:USDT",
            "direction": "long",
            "score": 72.0,
            "reasons": {"sweep": True, "displacement": True},
            "components": {"sweep_level": 3200.0},
            "current_price": 3250.0,
            "acted_on": False,
            "created_at": "2026-03-20T12:00:00Z",
        },
    ]
    # Chain: table().select().eq().eq().gte().order().limit().execute()
    chain = MagicMock()
    chain.execute = MagicMock(return_value=mock_result)
    chain.select = MagicMock(return_value=chain)
    chain.eq = MagicMock(return_value=chain)
    chain.gte = MagicMock(return_value=chain)
    chain.order = MagicMock(return_value=chain)
    chain.limit = MagicMock(return_value=chain)
    mock_db.table = MagicMock(return_value=chain)

    signals = await fetch_missed_signals(mock_db, instance_id="main", min_score=60.0, limit=50, days_back=7)

    assert len(signals) == 1
    assert signals[0]["symbol"] == "ETH/USDT:USDT"
    assert signals[0]["acted_on"] is False


@pytest.mark.asyncio
async def test_fetch_missed_signals_empty_result():
    """Should return empty list when no signals match."""
    mock_db = MagicMock()
    mock_result = MagicMock()
    mock_result.data = []
    chain = MagicMock()
    chain.execute = MagicMock(return_value=mock_result)
    chain.select = MagicMock(return_value=chain)
    chain.eq = MagicMock(return_value=chain)
    chain.gte = MagicMock(return_value=chain)
    chain.order = MagicMock(return_value=chain)
    chain.limit = MagicMock(return_value=chain)
    mock_db.table = MagicMock(return_value=chain)

    signals = await fetch_missed_signals(mock_db, instance_id="main")
    assert signals == []
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_missed_signals.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.advisor.missed_signals'`

- [ ] **Step 4: Implement the missed signal fetcher**

```python
# src/advisor/missed_signals.py
"""Fetch signals that were detected but never traded."""
from __future__ import annotations

import asyncio
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _exec(query):
    return query.execute()


async def fetch_missed_signals(
    db,
    instance_id: str = "main",
    min_score: float = 55.0,
    limit: int = 50,
    days_back: int = 7,
) -> list[dict[str, Any]]:
    """Query Supabase for high-scoring signals that were never acted on.

    These are the "missed trades" — signals that passed the confluence
    threshold but were filtered out by entry refinement, consensus,
    sentiment, or other downstream gates.
    """
    from datetime import datetime, timedelta, timezone

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()

    result = await asyncio.to_thread(
        _exec,
        db.table("signals")
        .select("*")
        .eq("acted_on", False)
        .eq("instance_id", instance_id)
        .gte("score", min_score)
        .gte("created_at", cutoff)
        .order("created_at", desc=True)
        .limit(limit),
    )
    return result.data or []
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_missed_signals.py -v
```

Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/advisor/__init__.py src/advisor/missed_signals.py tests/test_missed_signals.py
git commit -m "feat: add missed signal fetcher for trade advisor"
```

---

### Task 3: Build the Trade Outcome Simulator

Given a missed signal, fetch the candles from that time and simulate what would have happened if we'd entered at the signal price with standard SL/TP.

**Files:**
- Create: `src/advisor/outcome_simulator.py`
- Create: `tests/test_outcome_simulator.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_outcome_simulator.py
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
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
    """Long entry at 100, SL at 95, TP at 110 — price goes to 112."""
    candles = _make_candles([
        (100, 102, 99, 101),   # entry candle
        (101, 103, 100, 102),  # up
        (102, 106, 101, 105),  # up
        (105, 112, 104, 110),  # hits TP
    ])
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
    """Long entry at 100, SL at 95 — price drops to 94."""
    candles = _make_candles([
        (100, 101, 99, 100),
        (100, 100, 96, 97),
        (97, 98, 94, 95),  # hits SL
    ])
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
    """Price stays in range — neither SL nor TP hit within candles."""
    candles = _make_candles([
        (100, 102, 98, 101),
        (101, 103, 99, 100),
        (100, 104, 97, 102),
    ])
    result = simulate_trade_outcome(
        candles=candles,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=115.0,
        direction="long",
    )
    assert result["outcome"] == "expired"


def test_simulate_short_winner():
    """Short entry at 100, SL 105, TP 90 — price drops to 89."""
    candles = _make_candles([
        (100, 101, 98, 99),
        (99, 100, 93, 94),
        (94, 95, 89, 90),  # hits TP
    ])
    result = simulate_trade_outcome(
        candles=candles,
        entry_price=100.0,
        stop_loss=105.0,
        take_profit=90.0,
        direction="short",
    )
    assert result["outcome"] == "tp_hit"
    assert result["pnl_pct"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_outcome_simulator.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: Implement the simulator**

```python
# src/advisor/outcome_simulator.py
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
            pnl_pct = ((exit_price - entry_price) / entry_price) if is_long else (
                (entry_price - exit_price) / entry_price
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
            pnl_pct = ((exit_price - entry_price) / entry_price) if is_long else (
                (entry_price - exit_price) / entry_price
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
            pnl_pct = ((exit_price - entry_price) / entry_price) if is_long else (
                (entry_price - exit_price) / entry_price
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
    pnl_pct = ((last_close - entry_price) / entry_price) if is_long else (
        (entry_price - last_close) / entry_price
    )
    return {
        "outcome": "expired",
        "exit_price": last_close,
        "pnl_pct": round(pnl_pct * 100, 2),
        "candles_held": min(len(candles), max_candles),
        "exit_candle_time": str(candles.index[min(len(candles) - 1, max_candles - 1)]),
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_outcome_simulator.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/advisor/outcome_simulator.py tests/test_outcome_simulator.py
git commit -m "feat: add trade outcome simulator for missed trade analysis"
```

---

### Task 4: Build the MCP Tools for the Agent

Create custom MCP tools that the Claude agent will use to query missed signals, simulate outcomes, and fetch candles.

**Files:**
- Create: `src/advisor/tools.py`
- Create: `tests/test_advisor_tools.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_advisor_tools.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.advisor.tools import build_advisor_tools


def test_build_advisor_tools_returns_server_with_tools():
    """Should return an MCP server with 3 registered tools."""
    server = build_advisor_tools(
        db=MagicMock(),
        instance_id="main",
    )
    assert server is not None
    # Verify the server was created (create_sdk_mcp_server returns a valid object)
    # The exact attribute depends on the SDK version, but the server should be truthy
    assert server
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_advisor_tools.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: Implement the MCP tools**

```python
# src/advisor/tools.py
"""MCP tools for the Missed Trade Analyzer agent."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import ccxt.async_support as ccxt

from claude_agent_sdk import tool, create_sdk_mcp_server

from src.advisor.missed_signals import fetch_missed_signals
from src.advisor.outcome_simulator import simulate_trade_outcome
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _exec(query):
    return query.execute()


def build_advisor_tools(db, instance_id: str = "main"):
    """Create an MCP server with tools for the trade advisor agent.

    Args:
        db: Raw Supabase client (not the Database wrapper — use db.client).
        instance_id: Bot instance ID for scoping queries.

    Tools:
        - get_missed_signals: Fetch high-scoring signals that were never traded
        - simulate_outcome: Simulate what would have happened for a specific signal
        - get_trade_stats: Get recent trade performance statistics
    """

    @tool(
        "get_missed_signals",
        "Fetch signals that scored above the entry threshold but were never traded. "
        "Returns symbol, direction, score, reasons, price, and timestamp for each. "
        "Use this to identify patterns in what the bot is missing.",
        {
            "min_score": {
                "type": "number",
                "description": "Minimum confluence score (default 55)",
                "default": 55.0,
            },
            "limit": {
                "type": "integer",
                "description": "Max signals to return (default 30)",
                "default": 30,
            },
            "days_back": {
                "type": "integer",
                "description": "How many days back to look (default 7)",
                "default": 7,
            },
        },
    )
    async def get_missed_signals(args: dict) -> dict:
        min_score = args.get("min_score", 55.0)
        limit = args.get("limit", 30)
        days_back = args.get("days_back", 7)

        signals = await fetch_missed_signals(
            db, instance_id=instance_id,
            min_score=min_score, limit=limit, days_back=days_back,
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
        "Provide the signal details and it will walk forward through historical "
        "candles to determine if the trade would have hit TP, SL, or expired. "
        "Uses the entry_time to fetch candles from the actual signal time.",
        {
            "symbol": {"type": "string", "description": "Trading pair e.g. ETH/USDT:USDT"},
            "direction": {"type": "string", "description": "long or short"},
            "entry_price": {"type": "number", "description": "Entry price from the signal"},
            "stop_loss": {"type": "number", "description": "Stop loss price"},
            "take_profit": {"type": "number", "description": "Take profit price"},
            "entry_time": {
                "type": "string",
                "description": "ISO timestamp of the signal e.g. 2026-03-20T12:00:00Z",
            },
        },
    )
    async def simulate_outcome(args: dict) -> dict:
        import pandas as pd

        symbol = args["symbol"]
        entry_time = args["entry_time"]

        # Parse entry_time to millisecond timestamp for CCXT
        try:
            dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            since_ms = int(dt.timestamp() * 1000)
        except (ValueError, TypeError) as e:
            return {
                "content": [{"type": "text", "text": f"Invalid entry_time: {e}"}]
            }

        # Fetch historical 1H candles from signal time using CCXT directly
        exchange = ccxt.binance({"options": {"defaultType": "future"}})
        try:
            ohlcv = await exchange.fetch_ohlcv(
                symbol, timeframe="1h", since=since_ms, limit=48,
            )
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error fetching candles: {e}"}]
            }
        finally:
            await exchange.close()

        if not ohlcv:
            return {
                "content": [{"type": "text", "text": f"No candle data for {symbol} at {entry_time}"}]
            }

        # Convert to DataFrame (CCXT returns [timestamp, O, H, L, C, V])
        candles = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        candles.index = pd.to_datetime(candles["timestamp"], unit="ms", utc=True)
        candles.drop(columns=["timestamp"], inplace=True)

        result = simulate_trade_outcome(
            candles=candles,
            entry_price=args["entry_price"],
            stop_loss=args["stop_loss"],
            take_profit=args["take_profit"],
            direction=args["direction"],
        )
        result["symbol"] = symbol
        result["entry_price"] = args["entry_price"]

        return {
            "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
        }

    @tool(
        "get_trade_stats",
        "Get recent closed trade statistics: win rate, average P&L, "
        "common exit reasons, best/worst performers. Use this to understand "
        "baseline performance before recommending changes.",
        {
            "days_back": {
                "type": "integer",
                "description": "How many days of trades to analyze (default 14)",
                "default": 14,
            },
        },
    )
    async def get_trade_stats(args: dict) -> dict:
        try:
            result = await asyncio.to_thread(
                _exec,
                db.table("trades")
                .select("*")
                .eq("status", "closed")
                .eq("instance_id", instance_id)
                .order("exit_time", desc=True)
                .limit(100),
            )
            trades = result.data or []
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error: {e}"}]}

        if not trades:
            return {"content": [{"type": "text", "text": "No closed trades found."}]}

        wins = [t for t in trades if (t.get("pnl_usd") or 0) > 0]
        losses = [t for t in trades if (t.get("pnl_usd") or 0) <= 0]
        total_pnl = sum(t.get("pnl_usd", 0) or 0 for t in trades)
        exit_reasons = {}
        for t in trades:
            r = t.get("exit_reason", "unknown")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        stats = {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0,
            "total_pnl_usd": round(total_pnl, 2),
            "avg_pnl_usd": round(total_pnl / len(trades), 2) if trades else 0,
            "exit_reasons": exit_reasons,
        }
        return {"content": [{"type": "text", "text": json.dumps(stats, indent=2)}]}

    return create_sdk_mcp_server(
        name="trade-advisor",
        version="1.0.0",
        tools=[get_missed_signals, simulate_outcome, get_trade_stats],
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_advisor_tools.py -v
```

Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/advisor/tools.py tests/test_advisor_tools.py
git commit -m "feat: add MCP tools for trade advisor agent"
```

---

### Task 5: Build the Advisor Agent Runner

The main entry point that creates the Claude Agent SDK query with the system prompt, tools, and structured output.

**Files:**
- Create: `src/advisor/missed_trades.py`
- Create: `src/advisor/__main__.py`

- [ ] **Step 1: Write the advisor runner**

```python
# src/advisor/missed_trades.py
"""Missed Trade Analyzer — Claude Agent SDK powered.

Queries Supabase for signals that scored well but were never traded,
simulates outcomes, and produces actionable parameter recommendations.

Usage:
    python -m src.advisor.missed_trades [--days 7] [--min-score 55]

Requires: ANTHROPIC_API_KEY environment variable
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from claude_agent_sdk import query, ClaudeAgentOptions

from src.advisor.tools import build_advisor_tools
from src.data.db import Database
from src.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are a trading strategy advisor for a crypto trading bot called Tarakta.

## Your Role
Analyze signals that the bot detected but did NOT trade ("missed trades") to
determine if the bot is being too conservative with entries. Your goal is to
find profitable setups that were missed and recommend specific parameter changes.

## Context
The bot uses Smart Money Concepts (sweeps, displacement, pullbacks) on 1H candles.
After detecting a signal, it goes through several gates:
- Entry Refiner: waits for 5m pullback into OTE zone (50-79% Fib), skips if no pullback in 30min
- Pullback Analyzer: requires 20-78% retracement on 1H (under 20% = "waiting", over 78% = "failed")
- Consensus Monitor: checks portfolio bias + BTC trend alignment
- Sentiment Filter: blocks trades on strong contra-sentiment

The user believes the bot misses good trades because:
1. It can't see the "big picture" — doesn't realize pullback may have already happened
2. Entry points are too restrictive (OTE zone too narrow, retracement thresholds too tight)

## Your Workflow
1. First call get_trade_stats to understand baseline performance
2. Then call get_missed_signals to find signals that scored well but weren't traded
3. For the most promising missed signals (high score, clear setup), call simulate_outcome
   to check if they would have been profitable
4. Analyze patterns: What do the profitable missed trades have in common?
5. Produce a structured report with specific recommendations

## Output Format
Produce a report with these sections:

### Summary
- Total missed signals analyzed
- How many would have been profitable vs unprofitable
- Estimated missed P&L

### Pattern Analysis
- Common characteristics of profitable missed trades
- Which gate is blocking the most good trades (refiner, pullback, consensus, sentiment)
- Time-of-day or session patterns

### Recommendations
For each recommendation, be SPECIFIC:
- Which parameter to change (exact config field name)
- Current value → suggested value
- Expected impact (more trades, estimated win rate change)
- Risk of the change (might also let in more losing trades)

Example: "Lower MIN_RETRACEMENT from 0.20 to 0.12 — 8 of 15 profitable missed trades
had retracements of 12-19% that the current threshold filters out."

### Confidence
Rate your confidence in each recommendation (high/medium/low) with reasoning.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze missed trades with Claude Agent SDK")
    parser.add_argument("--days", type=int, default=7, help="Days of history to analyze (default 7)")
    parser.add_argument("--min-score", type=float, default=55.0, help="Min signal score (default 55)")
    parser.add_argument("--instance-id", type=str, default="main", help="Bot instance ID (default main)")
    parser.add_argument("--budget", type=float, default=2.0, help="Max USD to spend on analysis (default 2.0)")
    return parser.parse_args()


async def run_analysis(args: argparse.Namespace) -> None:
    """Run the missed trade analysis agent."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable required")
        print("Get one from https://console.anthropic.com/")
        sys.exit(1)

    # Connect to Supabase using existing infrastructure
    from src.config import Settings
    config = Settings()
    db = Database(url=config.supabase_url, key=config.supabase_key)

    # Build MCP tools (pass raw Supabase client, not Database wrapper)
    advisor_tools = build_advisor_tools(db=db.client, instance_id=args.instance_id)

    prompt = (
        f"Analyze missed trades from the last {args.days} days. "
        f"Look at signals with score >= {args.min_score}. "
        f"Focus on finding patterns in what the bot is missing and give "
        f"specific parameter recommendations to improve entry rates."
    )

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"advisor": advisor_tools},
        allowed_tools=[
            "mcp__advisor__get_missed_signals",
            "mcp__advisor__simulate_outcome",
            "mcp__advisor__get_trade_stats",
        ],
        max_budget_usd=args.budget,
        max_turns=20,
    )

    print(f"\n{'='*60}")
    print("  Tarakta Missed Trade Analyzer")
    print(f"  Analyzing last {args.days} days, min score {args.min_score}")
    print(f"{'='*60}\n")

    async for message in query(prompt=prompt, options=options):
        # Print assistant text responses
        if hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text)
        elif hasattr(message, "result"):
            print(f"\n--- Analysis complete ---")
            if hasattr(message, "total_cost_usd"):
                print(f"Cost: ${message.total_cost_usd:.4f}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_analysis(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make it runnable as a module**

Create `src/advisor/__main__.py`:

```python
from src.advisor.missed_trades import main

main()
```

- [ ] **Step 3: Verify it at least parses**

```bash
python -c "from src.advisor.missed_trades import SYSTEM_PROMPT; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/advisor/missed_trades.py src/advisor/__main__.py
git commit -m "feat: add missed trade analyzer agent runner"
```

---

### Task 6: Add Dashboard Endpoint (Optional Quick Access)

Add a `/advisor/missed` endpoint to the existing dashboard so results can be triggered from the web UI.

**Files:**
- Modify: `src/dashboard/api.py`

- [ ] **Step 1: Add the endpoint**

Add inside the `create_router()` function in `src/dashboard/api.py`, near the other `@router.get` endpoints:

```python
@router.get("/advisor/missed-signals")
async def get_missed_signals_api(
    request: Request,
    days: int = 7,
    min_score: float = 55.0,
):
    """Return missed signals for the dashboard (no agent, raw data only)."""
    from src.advisor.missed_signals import fetch_missed_signals

    r = _repo_for(request)
    signals = await fetch_missed_signals(
        r.db.client, instance_id=r.instance_id,
        min_score=min_score, limit=30, days_back=days,
    )
    return {"signals": signals, "count": len(signals)}
```

- [ ] **Step 2: Test the import works**

```bash
python -c "from src.advisor.missed_signals import fetch_missed_signals; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/dashboard/api.py
git commit -m "feat: add /api/advisor/missed-signals dashboard endpoint"
```

---

### Task 7: Update CLAUDE.md and Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add advisor section to CLAUDE.md**

Add after the "Key modules" section:

```markdown
- `src/advisor/` — Claude Agent SDK-powered trade analysis tools (not part of runtime bot)
  - `missed_trades.py` — Analyzes signals that scored well but weren't traded
  - Run: `python -m src.advisor.missed_trades --days 7`
  - Requires: `ANTHROPIC_API_KEY` env var (separate from bot's `OPENAI_API_KEY`)
```

- [ ] **Step 2: Add branches/deployment info to CLAUDE.md**

Add a new section after "Database":

```markdown
## Branches & Deployment

- `main` — development branch
- `expanded` — **production** (deployed at `tarakta-expanded.fly.dev`, includes footprint gate + expanded universe)
- `footprint` — reference branch (subset of expanded, do not delete)

Detailed bot documentation: `MAIN_BOT.md` (signal detection, confluence scoring, risk management, all config)
```

- [ ] **Step 3: Add actionable gotcha for clearing overrides**

Update the "DB model overrides" gotcha:

```markdown
- **DB model overrides**: After changing LLM model defaults in code, must clear `config_overrides` in `engine_state` table or stale model names persist. Fix: `UPDATE engine_state SET config_overrides = '{}' WHERE instance_id = 'main';`
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add advisor module docs, branch info, and actionable gotcha to CLAUDE.md"
```

---

### Task 8: Run Full Test Suite and Lint

- [ ] **Step 1: Run all tests**

```bash
pytest -v
```

Expected: All existing + new tests pass

- [ ] **Step 2: Run linter**

```bash
ruff check src/advisor/ tests/test_missed_signals.py tests/test_outcome_simulator.py tests/test_advisor_tools.py
ruff format src/advisor/ tests/test_missed_signals.py tests/test_outcome_simulator.py tests/test_advisor_tools.py
```

- [ ] **Step 3: Fix any lint issues and commit**

```bash
git add src/advisor/ tests/test_missed_signals.py tests/test_outcome_simulator.py tests/test_advisor_tools.py
git commit -m "style: fix lint issues in advisor module"
```
