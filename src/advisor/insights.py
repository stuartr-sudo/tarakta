"""Store and retrieve advisor insights from Supabase."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _exec(query):
    return query.execute()


async def store_insight(db, instance_id: str, data: dict[str, Any]) -> None:
    """Upsert an advisor insight row for today."""
    today = date.today().isoformat()

    row = {
        "instance_id": instance_id,
        "run_date": today,
        "signals_analyzed": data.get("signals_analyzed", 0),
        "simulated_winners": data.get("simulated_winners", 0),
        "simulated_losers": data.get("simulated_losers", 0),
        "win_rate_pct": data.get("win_rate_pct"),
        "top_missed": json.dumps(data.get("top_missed", [])),
        "patterns": json.dumps(data.get("patterns", {})),
        "recommendations": json.dumps(data.get("recommendations", [])),
        "full_analysis": data.get("full_analysis", ""),
        "cost_usd": data.get("cost_usd", 0),
    }

    await asyncio.to_thread(
        _exec,
        db.table("advisor_insights").upsert(row, on_conflict="instance_id,run_date"),
    )
    logger.info("insight_stored", instance_id=instance_id, run_date=today)


async def get_recent_insights(db, instance_id: str, limit: int = 1) -> list[dict[str, Any]]:
    """Fetch the most recent advisor insights."""
    result = await asyncio.to_thread(
        _exec,
        db.table("advisor_insights")
        .select("*")
        .eq("instance_id", instance_id)
        .order("run_date", desc=True)
        .limit(limit),
    )
    rows = result.data or []

    # Parse JSON text fields back to Python objects
    for row in rows:
        for field in ("top_missed", "patterns", "recommendations"):
            if isinstance(row.get(field), str):
                try:
                    row[field] = json.loads(row[field])
                except (json.JSONDecodeError, TypeError):
                    pass

    return rows


def format_insights_for_agent(insights: list[dict[str, Any]]) -> str:
    """Format insights into a context string for Agent 1/2 prompts.

    Returns empty string if no insights available.
    """
    if not insights:
        return ""

    latest = insights[0]
    win_rate = latest.get("win_rate_pct")
    analyzed = latest.get("signals_analyzed", 0)
    winners = latest.get("simulated_winners", 0)
    run_date = latest.get("run_date", "unknown")

    if analyzed == 0:
        return ""

    parts = [
        f"\n## Advisor Insights (analysis date: {run_date})",
        f"- {analyzed} missed signals analyzed, {winners} would have been winners",
    ]

    if win_rate is not None:
        parts.append(f"- Simulated win rate of missed signals: {win_rate}%")

    patterns = latest.get("patterns")
    if patterns and isinstance(patterns, dict):
        pattern_strs = [f"  - {k}: {v}" for k, v in patterns.items()]
        if pattern_strs:
            parts.append("- Common traits of missed winners:")
            parts.extend(pattern_strs)

    recommendations = latest.get("recommendations")
    if recommendations and isinstance(recommendations, list):
        parts.append("- Recommendations:")
        for rec in recommendations[:3]:  # Top 3 only to keep context short
            if isinstance(rec, str):
                parts.append(f"  - {rec}")
            elif isinstance(rec, dict):
                parts.append(f"  - {rec.get('recommendation', rec.get('text', str(rec)))}")

    return "\n".join(parts)
