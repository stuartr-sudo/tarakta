"""Sentiment filter for crypto trades using Hugging Face Inference API.

Primary: CryptoBERT (ElKulako/cryptobert) for crypto-native sentiment
classification. Outputs Bullish/Bearish/Neutral labels trained on 2M
StockTwits posts.
Secondary: Zero-shot classification (facebook/bart-large-mnli) for critical
event detection (hacks, rugs, delistings).
Fallback: Keyword-based scoring when HF API is unavailable.

All HF calls use aiohttp with strict timeouts so the async trading engine
is never blocked. If HF is slow or down, we fall back to keywords and
continue — never miss a stop-loss check.
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

# Hugging Face Inference API
HF_API_BASE = "https://router.huggingface.co/hf-inference/models"
CRYPTOBERT_MODEL = "ElKulako/cryptobert"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"

# Cache TTL in seconds
CACHE_TTL = 900  # 15 minutes

# HF API timeout — strict to avoid blocking the engine
HF_TIMEOUT = aiohttp.ClientTimeout(total=5)

# Critical event categories for zero-shot classification
CRITICAL_EVENTS = [
    "security breach or hack",
    "rug pull or exit scam",
    "delisting from exchange",
    "regulatory ban or enforcement action",
    "major partnership or adoption",
    "routine market news",
]

# Events that should trigger immediate trade blocking
NUKE_EVENTS = {"security breach or hack", "rug pull or exit scam", "delisting from exchange"}
NUKE_THRESHOLD = 0.70  # Minimum confidence to trigger event blocking

# Score thresholds for filtering
STRONG_NEGATIVE = -3.0
MODERATE_NEGATIVE = -1.5
STRONG_POSITIVE = 3.0

# --- Keyword fallback (used when HF API is unavailable) ---
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


class SentimentFilter:
    """CryptoBERT-powered sentiment analysis with keyword fallback."""

    def __init__(self, hf_api_token: str = "") -> None:
        self._cache: dict[str, tuple[float, list[str], float]] = {}  # symbol -> (score, events, timestamp)
        self._session: aiohttp.ClientSession | None = None
        self._hf_token = hf_api_token
        self._hf_available = bool(hf_api_token)
        self._hf_fail_count = 0
        self._hf_backoff_until = 0.0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self._session

    def _hf_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._hf_token}"}

    def _should_try_hf(self) -> bool:
        """Check if HF API should be attempted (not in backoff)."""
        if not self._hf_available:
            return False
        if time.time() < self._hf_backoff_until:
            return False
        return True

    def _record_hf_failure(self) -> None:
        """Track HF API failures with exponential backoff."""
        self._hf_fail_count += 1
        backoff = min(300, 30 * (2 ** (self._hf_fail_count - 1)))  # 30s, 60s, 120s, 300s max
        self._hf_backoff_until = time.time() + backoff
        logger.warning(
            "hf_api_backoff",
            fail_count=self._hf_fail_count,
            backoff_seconds=backoff,
        )

    def _record_hf_success(self) -> None:
        """Reset failure counter on success."""
        if self._hf_fail_count > 0:
            self._hf_fail_count = 0
            logger.info("hf_api_recovered")

    def _evict_stale_cache(self) -> None:
        """Remove expired entries from the sentiment cache."""
        now = time.time()
        stale = [k for k, (_, _, ts) in self._cache.items() if now - ts >= CACHE_TTL]
        for k in stale:
            del self._cache[k]

    async def get_sentiment(self, symbol: str) -> float:
        """Get sentiment score for a symbol. Returns float in range ~[-10, +10].

        Positive = bullish sentiment, Negative = bearish sentiment.
        Returns 0.0 if no news found or on error.
        """
        base = symbol.split("/")[0].upper()

        # Evict stale entries periodically
        if len(self._cache) > 50:
            self._evict_stale_cache()

        # Check cache
        cached = self._cache.get(base)
        if cached:
            score, _events, ts = cached
            if time.time() - ts < CACHE_TTL:
                return score

        try:
            score, events = await self._fetch_and_score(base)
            self._cache[base] = (score, events, time.time())
            return score
        except Exception as e:
            logger.warning("sentiment_fetch_failed", symbol=symbol, error=str(e))
            return 0.0

    def get_critical_events(self, symbol: str) -> list[str]:
        """Get cached critical events detected for a symbol."""
        base = symbol.split("/")[0].upper()
        cached = self._cache.get(base)
        if cached:
            _score, events, ts = cached
            if time.time() - ts < CACHE_TTL:
                return events
        return []

    async def _fetch_and_score(self, asset: str) -> tuple[float, list[str]]:
        """Fetch news and compute sentiment using CryptoBERT or keyword fallback."""
        session = await self._get_session()

        params: dict[str, Any] = {"categories": asset, "excludeCategories": "Sponsored"}

        try:
            async with session.get(NEWS_API_URL, params=params) as resp:
                if resp.status != 200:
                    return 0.0, []
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return 0.0, []

        articles = data.get("Data", [])
        if not articles:
            return 0.0, []

        # Extract titles and bodies from most recent articles
        texts = []
        for article in articles[:10]:  # Reduced from 20 to 10 for API efficiency
            title = (article.get("title") or "").strip()
            body = (article.get("body") or "")[:300].strip()
            if title:
                texts.append(f"{title}. {body}" if body else title)

        if not texts:
            return 0.0, []

        # Try CryptoBERT, fall back to keywords
        if self._should_try_hf():
            try:
                score = await self._finbert_score(texts)
                events = await self._detect_critical_events(texts)
                self._record_hf_success()

                logger.debug(
                    "sentiment_scored_cryptobert",
                    asset=asset,
                    articles=len(texts),
                    score=round(score, 2),
                    events=events,
                )
                return score, events
            except Exception as e:
                logger.warning("cryptobert_failed_using_keywords", asset=asset, error=str(e))
                self._record_hf_failure()

        # Keyword fallback
        score = self._keyword_score(texts)
        logger.debug(
            "sentiment_scored_keywords",
            asset=asset,
            articles=len(texts),
            score=round(score, 2),
        )
        return score, []

    async def _finbert_score(self, texts: list[str]) -> float:
        """Score texts using CryptoBERT via HF Inference API.

        CryptoBERT returns: Bullish, Bearish, Neutral for each text.
        We convert to a single score: bullish contributes +1, bearish -1,
        neutral 0, weighted by confidence. Then scale to [-10, +10] range.
        """
        session = await self._get_session()
        url = f"{HF_API_BASE}/{CRYPTOBERT_MODEL}"

        total_score = 0.0
        scored = 0

        # Send texts in a single batch request
        try:
            async with session.post(
                url,
                json={"inputs": texts, "parameters": {"top_k": None}},
                headers=self._hf_headers(),
                timeout=HF_TIMEOUT,
            ) as resp:
                if resp.status == 503:
                    # Model loading — raise to trigger fallback
                    raise RuntimeError("CryptoBERT model loading (503)")
                if resp.status != 200:
                    raise RuntimeError(f"CryptoBERT API returned {resp.status}")
                results = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise RuntimeError(f"CryptoBERT request failed: {e}") from e

        # results is list of list of {label, score} for each text
        for article_results in results:
            if not isinstance(article_results, list):
                continue
            # Handle extra nesting from HF router API
            if article_results and isinstance(article_results[0], list):
                article_results = article_results[0]
            # Find best label
            best = max(article_results, key=lambda x: x.get("score", 0))
            label = best.get("label", "neutral").lower()
            confidence = best.get("score", 0.5)

            if label == "bullish":
                total_score += confidence
            elif label == "bearish":
                total_score -= confidence
            # neutral contributes 0
            scored += 1

        if scored == 0:
            return 0.0

        # Normalize: raw is in [-1, +1] per article, averaged
        # Scale to [-10, +10] range to match existing thresholds
        normalized = (total_score / scored) * 10.0

        return normalized

    async def _detect_critical_events(self, texts: list[str]) -> list[str]:
        """Detect critical events using zero-shot classification.

        Only classifies the first 3 headlines to minimize API calls.
        Returns list of detected critical event types.
        """
        session = await self._get_session()
        url = f"{HF_API_BASE}/{ZERO_SHOT_MODEL}"

        detected_events: list[str] = []

        # Only check top 3 headlines for critical events
        for text in texts[:3]:
            try:
                async with session.post(
                    url,
                    json={
                        "inputs": text,
                        "parameters": {
                            "candidate_labels": CRITICAL_EVENTS,
                            "multi_label": False,
                        },
                    },
                    headers=self._hf_headers(),
                    timeout=HF_TIMEOUT,
                ) as resp:
                    if resp.status != 200:
                        continue
                    result = await resp.json()

                labels = result.get("labels", [])
                scores = result.get("scores", [])

                if labels and scores:
                    top_label = labels[0]
                    top_score = scores[0]

                    if top_label in NUKE_EVENTS and top_score >= NUKE_THRESHOLD:
                        detected_events.append(f"{top_label} ({top_score:.0%})")
                        logger.warning(
                            "critical_event_detected",
                            event=top_label,
                            confidence=round(top_score, 2),
                            headline=text[:100],
                        )

            except (aiohttp.ClientError, asyncio.TimeoutError):
                continue

        return detected_events

    def _keyword_score(self, texts: list[str]) -> float:
        """Fallback keyword-based scoring."""
        total_score = 0.0
        scored = 0

        for text in texts:
            text_lower = text.lower()
            article_score = 0.0
            for word, weight in POSITIVE_KEYWORDS.items():
                if word in text_lower:
                    article_score += weight
            for word, weight in NEGATIVE_KEYWORDS.items():
                if word in text_lower:
                    article_score -= weight
            total_score += article_score
            scored += 1

        if scored == 0:
            return 0.0

        return total_score / scored

    def should_block_trade(self, sentiment_score: float, direction: str) -> bool:
        """Check if sentiment is strongly against the proposed trade direction.

        Accepts both scanner labels (bullish/bearish) and position labels (long/short).
        Returns True if the trade should be blocked.
        """
        if direction in ("long", "bullish") and sentiment_score <= STRONG_NEGATIVE:
            return True
        if direction in ("short", "bearish") and sentiment_score >= STRONG_POSITIVE:
            return True
        return False

    def has_critical_event(self, symbol: str) -> str | None:
        """Check if a critical event was detected for this symbol.

        Returns the event description if found, None otherwise.
        """
        events = self.get_critical_events(symbol)
        if events:
            return "; ".join(events)
        return None

    def score_adjustment(self, sentiment_score: float, direction: str) -> float:
        """Return a confluence score adjustment based on sentiment alignment.

        Accepts both scanner labels (bullish/bearish) and position labels (long/short).
        Returns a value between -10 and +5 to add to the confluence score.
        """
        if direction in ("long", "bullish"):
            if sentiment_score <= STRONG_NEGATIVE:
                return -10.0
            elif sentiment_score <= MODERATE_NEGATIVE:
                return -5.0
            elif sentiment_score >= STRONG_POSITIVE:
                return 5.0
            elif sentiment_score > 0:
                return 2.0
            return 0.0
        elif direction in ("short", "bearish"):
            if sentiment_score >= STRONG_POSITIVE:
                return -10.0
            elif sentiment_score > MODERATE_NEGATIVE:
                return -5.0 if sentiment_score > 0 else 0.0
            elif sentiment_score <= STRONG_NEGATIVE:
                return 5.0
            return 0.0
        return 0.0

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
