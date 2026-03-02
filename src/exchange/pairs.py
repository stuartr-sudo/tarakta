from __future__ import annotations

from src.utils.logging import get_logger

logger = get_logger(__name__)


async def get_scannable_pairs(
    client,
    min_volume_usd: float = 50_000,
    quote_currencies: list[str] | None = None,
) -> list[str]:
    """Get all pairs eligible for scanning, sorted by volume descending."""
    pairs = await client.get_tradeable_pairs(
        min_volume_usd=min_volume_usd,
        quote_currencies=quote_currencies,
    )
    logger.info("scannable_pairs", count=len(pairs))
    return pairs
