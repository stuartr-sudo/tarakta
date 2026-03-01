"""Lightweight sentiment filter for crypto trades.

Uses CryptoCompare's free news API + keyword-based sentiment scoring.
No ML model needed — fits within 512MB Fly.io constraint.

Sentiment is used as a pre-trade filter: if strong negative sentiment
is detected for a long signal (or strong positive for a short signal),
the trade is skipped or score is penalized.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp

from src.utils.logging import get_logger

logger = get_logger(__name__)

# CryptoCompare news API (free, no key needed for basic access)
NEWS_API_URL = "https://min-api.cryptocompare.com/data/v2/news/"

# Cache TTL in seconds (avoid hammering the API)
CACHE_TTL = 900  # 15 minutes

# Sentiment keywords (weighted)
POSITIVE_KEYWORDS = {
    "bullish": 2, "surge": 2, "rally": 2, "breakout": 2, "soar": 2,
    "moon": 1, "gain": 1, "rise": 1, "up": 0.5, "high": 0.5,
    "adoption": 1.5, "partnership": 1.5, "launch": 1, "upgrade": 1,
    "buy": 1, "accumulate": 1.5, "institutional": 1.5, "etf": 2,
    "approval": 1.5, "growth": 1, "record": 1, "milestone": 1,
}

NEGATIVE_KEYWORDS = {
    "bearish": 2, "crash": 2, "plunge": 2, "dump": 2, "collapse": 2,
    "hack": 3, "exploit": 3, "vulnerability": 2, "scam": 3, "fraud": 3,
    "ban": 2, "regulate": 1, "fine": 1.5, "lawsuit": 2, "sec": 1.5,
    "sell": 1, "fear": 1, "panic": 1.5, "liquidation": 2, "delist": 2.5,
    "rug": 3, "ponzi": 3, "warning": 1, "risk": 0.5, "decline": 1,
}

# Score thresholds for filtering
STRONG_NEGATIVE = -3.0  # Block trade if sentiment below this
MODERATE_NEGATIVE = -1.5  # Penalize score if below this
STRONG_POSITIVE = 3.0


class SentimentFilter:
    """Keyword-based sentiment analysis from crypto news."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, float]] = {}  # symbol -> (score, timestamp)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self._session

    async def get_sentiment(self, symbol: str) -> float:
        """Get sentiment score for a symbol. Returns float in range ~[-10, +10].

        Positive = bullish sentiment, Negative = bearish sentiment.
        Returns 0.0 if no news found or on error.
        """
        # Extract base asset from pair (e.g., "BTC/USD" -> "BTC")
        base = symbol.split("/")[0].upper()

        # Check cache
        cached = self._cache.get(base)
        if cached:
            score, ts = cached
            if time.time() - ts < CACHE_TTL:
                return score

        try:
            score = await self._fetch_and_score(base)
            self._cache[base] = (score, time.time())
            return score
        except Exception as e:
            logger.warning("sentiment_fetch_failed", symbol=symbol, error=str(e))
            return 0.0

    async def _fetch_and_score(self, asset: str) -> float:
        """Fetch news from CryptoCompare and compute sentiment score."""
        session = await self._get_session()

        params: dict[str, Any] = {"categories": asset, "excludeCategories": "Sponsored"}

        try:
            async with session.get(NEWS_API_URL, params=params) as resp:
                if resp.status != 200:
                    return 0.0
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return 0.0

        articles = data.get("Data", [])
        if not articles:
            return 0.0

        # Score the most recent articles (up to 20)
        total_score = 0.0
        articles_scored = 0

        for article in articles[:20]:
            title = (article.get("title") or "").lower()
            body = (article.get("body") or "")[:500].lower()
            text = f"{title} {body}"

            article_score = 0.0
            for word, weight in POSITIVE_KEYWORDS.items():
                if word in text:
                    article_score += weight
            for word, weight in NEGATIVE_KEYWORDS.items():
                if word in text:
                    article_score -= weight

            total_score += article_score
            articles_scored += 1

        if articles_scored == 0:
            return 0.0

        # Normalize by number of articles
        normalized = total_score / articles_scored

        logger.debug(
            "sentiment_scored",
            asset=asset,
            articles=articles_scored,
            raw_score=round(total_score, 2),
            normalized=round(normalized, 2),
        )

        return normalized

    def should_block_trade(self, sentiment_score: float, direction: str) -> bool:
        """Check if sentiment is strongly against the proposed trade direction.

        Returns True if the trade should be blocked.
        """
        if direction == "long" and sentiment_score <= STRONG_NEGATIVE:
            return True
        if direction == "short" and sentiment_score >= STRONG_POSITIVE:
            return True
        return False

    def score_adjustment(self, sentiment_score: float, direction: str) -> float:
        """Return a confluence score adjustment based on sentiment alignment.

        Returns a value between -10 and +5 to add to the confluence score.
        """
        if direction == "long":
            if sentiment_score <= STRONG_NEGATIVE:
                return -10.0  # Strong penalty
            elif sentiment_score <= MODERATE_NEGATIVE:
                return -5.0  # Moderate penalty
            elif sentiment_score >= STRONG_POSITIVE:
                return 5.0  # Bonus
            elif sentiment_score > 0:
                return 2.0  # Mild bonus
            return 0.0
        elif direction == "short":
            if sentiment_score >= STRONG_POSITIVE:
                return -10.0  # Strong penalty (market bullish, shorting risky)
            elif sentiment_score > MODERATE_NEGATIVE:
                return -5.0 if sentiment_score > 0 else 0.0
            elif sentiment_score <= STRONG_NEGATIVE:
                return 5.0  # Bonus — market is bearish, short aligns
            return 0.0
        return 0.0

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
