"""Run the trade advisor using the Anthropic API directly.

This avoids the Claude Agent SDK which can't run as root in Docker.
Instead, we gather all data in Python and send it to Claude in a single API call.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

ADVISOR_SYSTEM_PROMPT = """\
You are Tarakta's trade advisor — an expert in Smart Money Concepts (SMC) crypto trading.

Your job is to analyze missed trading signals and determine:
1. Which missed signals would have been profitable trades
2. Why the bot's entry criteria might be too conservative
3. Specific, actionable recommendations to improve hit rate

## Output format

First, produce your analysis in plain text with:
- **Summary**: X missed signals analyzed, Y would have been winners (Z% win rate)
- **Top missed opportunities**: List the best trades that were missed
- **Pattern analysis**: What the winning signals had in common
- **Recommendations**: Specific changes to entry criteria

Then, at the END of your response, output a structured JSON block wrapped in \
```json fences with exactly these fields:

```json
{
  "signals_analyzed": <int>,
  "simulated_winners": <int>,
  "simulated_losers": <int>,
  "win_rate_pct": <float>,
  "top_missed": [
    {"symbol": "...", "direction": "...", "score": <float>, "pnl_pct": <float>}
  ],
  "patterns": {
    "common_components": "...",
    "avg_winner_score": <float>,
    "typical_direction": "..."
  },
  "recommendations": [
    "Specific recommendation 1",
    "Specific recommendation 2"
  ]
}
```

This JSON block is machine-parsed — be precise with the format.
"""


def _parse_structured_output(analysis: str) -> dict[str, Any] | None:
    """Extract the JSON block from the advisor's analysis text."""
    pattern = r"```json\s*\n(.*?)\n\s*```"
    matches = re.findall(pattern, analysis, re.DOTALL)
    if not matches:
        return None

    try:
        return json.loads(matches[-1])
    except (json.JSONDecodeError, TypeError):
        logger.warning("advisor_json_parse_failed", raw=matches[-1][:200])
        return None


async def _simulate_signal(db, signal: dict) -> dict[str, Any] | None:
    """Simulate outcome for a single missed signal."""
    from src.advisor.outcome_simulator import simulate_trade_outcome

    symbol = signal.get("symbol", "")
    signal_time = signal.get("created_at", "")
    entry_price = signal.get("current_price") or signal.get("price")
    direction_raw = signal.get("direction", "bullish")

    if not entry_price or not symbol:
        return None

    direction = "long" if "bull" in str(direction_raw).lower() else "short"
    is_long = direction == "long"
    sl_pct = 0.02
    tp_pct = 0.04
    sl = entry_price * (1 - sl_pct) if is_long else entry_price * (1 + sl_pct)
    tp = entry_price * (1 + tp_pct) if is_long else entry_price * (1 - tp_pct)

    def _exec(q):
        return q.execute()

    try:
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
    except Exception as e:
        logger.warning("advisor_candle_fetch_failed", symbol=symbol, error=str(e))
        return None

    if len(rows) < 5:
        return None

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    outcome = simulate_trade_outcome(
        candles=df,
        entry_price=entry_price,
        stop_loss=sl,
        take_profit=tp,
        direction=direction,
    )
    outcome["symbol"] = symbol
    outcome["direction"] = direction
    outcome["entry_price"] = entry_price
    outcome["score"] = signal.get("score", 0)
    outcome["signal_type"] = signal.get("signal_type", "")
    return outcome


async def run_advisor(
    db,
    instance_id: str = "main",
    prompt: str | None = None,
    store: bool = True,
) -> dict[str, Any]:
    """Run the trade advisor and return its analysis.

    Gathers missed signals and simulations in Python, then sends
    everything to Claude in a single API call (no Agent SDK needed).
    """
    import httpx

    from src.advisor.missed_signals import fetch_missed_signals

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"analysis": "Error: ANTHROPIC_API_KEY not set", "cost_usd": 0, "structured": {}}

    # Step 1: Fetch missed signals
    signals = await fetch_missed_signals(db, instance_id=instance_id, min_score=55.0, days_back=7)
    logger.info("advisor_signals_fetched", count=len(signals))

    if not signals:
        return {
            "analysis": "No missed signals found in the past 7 days.",
            "cost_usd": 0,
            "structured": {},
        }

    # Step 2: Simulate outcomes for high-scoring signals
    high_scoring = [s for s in signals if (s.get("score") or 0) >= 65][:15]
    simulations = []
    for sig in high_scoring:
        outcome = await _simulate_signal(db, sig)
        if outcome:
            simulations.append(outcome)

    logger.info("advisor_simulations_complete", simulated=len(simulations), total_signals=len(signals))

    # Step 3: Build context for Claude
    signal_summary = []
    for s in signals[:30]:
        signal_summary.append({
            "symbol": s.get("symbol"),
            "score": s.get("score"),
            "direction": s.get("direction"),
            "signal_type": s.get("signal_type"),
            "created_at": str(s.get("created_at", ""))[:19],
            "current_price": s.get("current_price"),
        })

    sim_summary = []
    for sim in simulations:
        sim_summary.append({
            "symbol": sim.get("symbol"),
            "direction": sim.get("direction"),
            "score": sim.get("score"),
            "outcome": sim.get("outcome"),
            "pnl_pct": sim.get("pnl_pct"),
            "candles_held": sim.get("candles_held"),
            "max_favorable": sim.get("max_favorable_pct"),
        })

    data_context = (
        f"## Missed Signals ({len(signals)} total, showing top 30)\n\n"
        f"```json\n{json.dumps(signal_summary, indent=2, default=str)}\n```\n\n"
        f"## Simulation Results ({len(simulations)} signals simulated with 2% SL / 4% TP)\n\n"
        f"```json\n{json.dumps(sim_summary, indent=2, default=str)}\n```"
    )

    user_message = prompt or (
        "Analyze these missed trading signals and simulation results. "
        "Identify patterns, calculate win rates, and provide recommendations "
        "to make the bot less conservative on entry.\n\n"
        + data_context
    )

    # Step 4: Call Claude API directly
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 4096,
                "system": ADVISOR_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_message}],
            },
        )

    if resp.status_code != 200:
        error_msg = f"Anthropic API error {resp.status_code}: {resp.text[:300]}"
        logger.error("advisor_api_error", status=resp.status_code, body=resp.text[:300])
        return {"analysis": error_msg, "cost_usd": 0, "structured": {}}

    data = resp.json()
    analysis = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            analysis += block["text"]

    # Estimate cost from usage
    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    # Claude Sonnet pricing: $3/M input, $15/M output
    cost_usd = (input_tokens * 3 / 1_000_000) + (output_tokens * 15 / 1_000_000)

    logger.info("advisor_complete", analysis_length=len(analysis), cost_usd=round(cost_usd, 4))

    # Parse structured output
    structured = _parse_structured_output(analysis) or {}

    # Store insights in DB
    if store and structured:
        try:
            from src.advisor.insights import store_insight

            insight_data = {**structured, "full_analysis": analysis, "cost_usd": cost_usd}
            await store_insight(db, instance_id, insight_data)
        except Exception as e:
            logger.error("advisor_store_failed", error=str(e))

    return {"analysis": analysis, "cost_usd": round(cost_usd, 4), "structured": structured}
