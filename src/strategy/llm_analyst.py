"""LLM-powered trade analyst using Claude API.

Sends full signal context to Claude Haiku for approve/reject decisions
with optional SL/TP adjustments. Used in the A/B split test to compare
LLM-filtered trades vs. the existing confluence-only system.

Follows the same resilience patterns as sentiment.py: exponential backoff,
strict timeouts, graceful fallback when API is unavailable.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from src.config import Settings
from src.exchange.models import SignalCandidate
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMAnalysis:
    """Result from the LLM trade analyst."""

    approve: bool
    confidence: float = 50.0  # 0-100
    reasoning: str = ""
    suggested_sl: float | None = None
    suggested_tp: float | None = None
    latency_ms: float = 0.0
    error: str | None = None


SYSTEM_PROMPT = """\
You are a crypto trade analyst specializing in Smart Money Concepts (ICT methodology).

You will receive a trade signal with full technical context. Your job is to evaluate \
whether this is a high-quality trade setup worth taking.

Evaluate based on:
1. Confluence alignment — are multiple SMC factors confirming the direction?
2. Key level quality — is the entry near a strong order block, FVG, or liquidity sweep?
3. Risk/reward — is the R:R ratio attractive given the setup quality?
4. Volume confirmation — does volume support the move?
5. Market structure — is the higher timeframe trend aligned?

Respond with ONLY a JSON object (no markdown, no code fences):
{
  "approve": true/false,
  "confidence": 0-100,
  "reasoning": "brief 1-2 sentence explanation",
  "suggested_sl": null or float (only if you see a clearly better SL level),
  "suggested_tp": null or float (only if you see a clearly better TP level)
}
"""


class LLMTradeAnalyst:
    """Claude-powered trade signal analyst with resilient API handling."""

    def __init__(self, config: Settings) -> None:
        self._client: anthropic.AsyncAnthropic | None = None
        self._api_key = config.llm_api_key
        self._model = config.llm_model
        self._timeout = config.llm_timeout_seconds
        self._fallback_approve = config.llm_fallback_approve
        self._available = bool(config.llm_api_key)

        # Backoff state (mirrors sentiment.py pattern)
        self._fail_count = 0
        self._backoff_until = 0.0

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key,
                timeout=self._timeout,
                max_retries=1,
            )
        return self._client

    def _should_try(self) -> bool:
        if not self._available:
            return False
        if time.time() < self._backoff_until:
            return False
        return True

    def _record_failure(self) -> None:
        self._fail_count += 1
        backoff = min(300, 30 * (2 ** (self._fail_count - 1)))
        self._backoff_until = time.time() + backoff
        logger.warning(
            "llm_api_backoff",
            fail_count=self._fail_count,
            backoff_seconds=backoff,
        )

    def _record_success(self) -> None:
        if self._fail_count > 0:
            self._fail_count = 0
            logger.info("llm_api_recovered")

    async def analyze_signal(
        self,
        signal: SignalCandidate,
        context: dict[str, Any],
    ) -> LLMAnalysis:
        """Analyze a trade signal using Claude.

        Args:
            signal: The signal candidate to evaluate.
            context: Additional context dict with keys like:
                - sentiment_score, adjusted_score, active_threshold
                - market_structure, volume_data, key_levels
                - recent_performance (win_rate, avg_rr, etc.)

        Returns:
            LLMAnalysis with approve/reject decision and optional adjustments.
        """
        if not self._should_try():
            return LLMAnalysis(
                approve=self._fallback_approve,
                reasoning="LLM API unavailable (backoff), using fallback",
                error="api_backoff",
            )

        user_prompt = self._build_prompt(signal, context)
        start = time.monotonic()

        try:
            client = self._get_client()
            response = await client.messages.create(
                model=self._model,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            latency_ms = (time.monotonic() - start) * 1000
            self._record_success()

            analysis = self._parse_response(response)
            analysis.latency_ms = latency_ms

            logger.info(
                "llm_analysis_complete",
                symbol=signal.symbol,
                approve=analysis.approve,
                confidence=analysis.confidence,
                latency_ms=round(latency_ms, 1),
                reasoning=analysis.reasoning[:100],
            )
            return analysis

        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            self._record_failure()
            logger.warning(
                "llm_analysis_failed",
                symbol=signal.symbol,
                error=str(e),
                latency_ms=round(latency_ms, 1),
            )
            return LLMAnalysis(
                approve=self._fallback_approve,
                reasoning=f"LLM API error: {e}",
                latency_ms=latency_ms,
                error=str(e),
            )

    def _build_prompt(self, signal: SignalCandidate, context: dict[str, Any]) -> str:
        """Assemble the user prompt with all signal data."""
        direction = signal.direction or "unknown"
        ob_info = "None"
        if signal.ob_context:
            ob = signal.ob_context
            ob_info = f"{ob.direction} OB [{ob.bottom:.6g} - {ob.top:.6g}], strength={ob.strength:.2f}"

        fvg_info = "None"
        if signal.fvg_context:
            fvg = signal.fvg_context
            fvg_info = f"{fvg.direction} FVG [{fvg.bottom:.6g} - {fvg.top:.6g}]"

        key_levels_str = ", ".join(
            f"{k}: {v:.6g}" for k, v in signal.key_levels.items()
        ) if signal.key_levels else "None"

        sentiment = context.get("sentiment_score", 0.0)
        adjusted = context.get("adjusted_score", signal.score)
        threshold = context.get("active_threshold", 65.0)

        return f"""\
## Trade Signal Analysis Request

**Symbol:** {signal.symbol}
**Direction:** {direction}
**Entry Price:** {signal.entry_price:.6g}
**Confluence Score:** {signal.score:.1f}/100 (adjusted: {adjusted:.1f}, threshold: {threshold:.1f})

### Signal Reasons
{chr(10).join(f"- {r}" for r in signal.reasons)}

### Key Levels
{key_levels_str}

### Order Block Context
{ob_info}

### Fair Value Gap Context
{fvg_info}

### Sentiment Score
{sentiment:.1f} (range: -10 to +10, positive = bullish)

### Additional Context
{json.dumps({k: v for k, v in context.items() if k not in ("sentiment_score", "adjusted_score", "active_threshold")}, default=str, indent=2) if context else "None"}

Should this trade be taken? Respond with JSON only.
"""

    def _parse_response(self, response: anthropic.types.Message) -> LLMAnalysis:
        """Extract structured fields from Claude's response."""
        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
                break

        if not text:
            return LLMAnalysis(
                approve=self._fallback_approve,
                reasoning="Empty LLM response",
                error="empty_response",
            )

        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("llm_response_parse_failed", raw=text[:200], error=str(e))
            return LLMAnalysis(
                approve=self._fallback_approve,
                reasoning=f"Failed to parse LLM response: {text[:100]}",
                error="parse_error",
            )

        return LLMAnalysis(
            approve=bool(data.get("approve", self._fallback_approve)),
            confidence=float(data.get("confidence", 50.0)),
            reasoning=str(data.get("reasoning", "")),
            suggested_sl=data.get("suggested_sl"),
            suggested_tp=data.get("suggested_tp"),
        )

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
