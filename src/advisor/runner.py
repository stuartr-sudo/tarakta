"""Run the trade advisor agent using Claude Agent SDK."""

from __future__ import annotations

import json
import re
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)

ADVISOR_SYSTEM_PROMPT = """\
You are Tarakta's trade advisor — an expert in Smart Money Concepts (SMC) crypto trading.

Your job is to analyze missed trading signals and determine:
1. Which missed signals would have been profitable trades
2. Why the bot's entry criteria might be too conservative
3. Specific, actionable recommendations to improve hit rate

## How to work

1. First, call get_missed_signals to fetch recent signals that were detected but never traded.
2. For each high-scoring signal (score >= 65), call simulate_outcome to see what would have happened.
   - Use the signal's current_price as entry_price
   - Use the signal's direction
   - Use the signal's created_at as signal_time
   - Use default SL/TP percentages (2% SL, 4% TP) unless the signal components suggest different levels
3. Analyze the pattern of results:
   - What percentage of missed signals would have been winners?
   - Are there common characteristics among the winners?
   - What does this tell us about the bot's entry criteria?

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
    # Find the last ```json ... ``` block
    pattern = r"```json\s*\n(.*?)\n\s*```"
    matches = re.findall(pattern, analysis, re.DOTALL)
    if not matches:
        return None

    try:
        return json.loads(matches[-1])
    except (json.JSONDecodeError, TypeError):
        logger.warning("advisor_json_parse_failed", raw=matches[-1][:200])
        return None


async def run_advisor(
    db,
    instance_id: str = "main",
    prompt: str | None = None,
    store: bool = True,
) -> dict[str, Any]:
    """Run the trade advisor and return its analysis.

    Args:
        db: Supabase database client.
        instance_id: Bot instance ID.
        prompt: Optional custom prompt (default analyzes all missed signals).
        store: Whether to store insights in the DB (default True).

    Returns:
        Dict with 'analysis' text, 'cost_usd', and 'structured' parsed data.
    """
    from src.advisor.tools import build_advisor_tools

    server = build_advisor_tools(db, instance_id=instance_id)

    default_prompt = (
        "Analyze the missed trading signals from the past 7 days. "
        "Fetch the signals, simulate outcomes for the high-scoring ones, "
        "and give me a full analysis with recommendations."
    )

    options = ClaudeAgentOptions(
        system_prompt=ADVISOR_SYSTEM_PROMPT,
        mcp_servers={"advisor": server},
        allowed_tools=[
            "mcp__advisor__get_missed_signals",
            "mcp__advisor__simulate_outcome",
        ],
        permission_mode="bypassPermissions",
        max_turns=15,
    )

    analysis_parts: list[str] = []
    cost_usd = 0.0

    async for message in query(prompt=prompt or default_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    analysis_parts.append(block.text)
        elif isinstance(message, ResultMessage):
            cost_usd = message.total_cost_usd or 0.0
            if message.is_error:
                logger.error("advisor_agent_error", cost=cost_usd)

    analysis = "\n".join(analysis_parts)
    logger.info("advisor_complete", analysis_length=len(analysis), cost_usd=cost_usd)

    # Parse structured output from the analysis
    structured = _parse_structured_output(analysis) or {}

    # Store insights in DB
    if store and structured:
        try:
            from src.advisor.insights import store_insight

            insight_data = {**structured, "full_analysis": analysis, "cost_usd": cost_usd}
            await store_insight(db, instance_id, insight_data)
        except Exception as e:
            logger.error("advisor_store_failed", error=str(e))

    return {"analysis": analysis, "cost_usd": cost_usd, "structured": structured}
