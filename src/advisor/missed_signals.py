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
    """Query Supabase for high-scoring signals that were never acted on."""
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
