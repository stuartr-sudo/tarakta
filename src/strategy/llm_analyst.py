"""LLM-powered trade analyst using Claude API.

Sends full signal context to Claude Haiku for approve/reject decisions
with optional SL/TP adjustments. Used in the A/B split test to compare
LLM-filtered trades vs. the existing confluence-only system.

Uses Anthropic tool use (structured outputs) for guaranteed JSON parsing
and forces chain-of-thought by requiring reasoning before the decision.

Follows the same resilience patterns as sentiment.py: exponential backoff,
strict timeouts, graceful fallback when API is unavailable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
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
    input_tokens: int = 0
    output_tokens: int = 0


SYSTEM_PROMPT = """\
You are a crypto trade analyst specializing in Smart Money Concepts (ICT methodology).

You will receive a trade signal with full technical context including proposed \
stop-loss, take-profit levels, and risk/reward ratio. Your job is to evaluate \
whether this is a high-quality trade setup worth taking.

Evaluate based on:
1. Confluence alignment — are multiple SMC factors confirming the direction? \
A score of 70+ with 4+ reasons is strong.
2. Key level quality — is the entry near a strong order block, FVG, or liquidity sweep? \
Price should be within or near (not far from) the key structure.
3. Risk/reward — is the R:R ratio >= 2.0? Is the SL placed at a logical structure level \
(below OB/FVG for longs, above for shorts) rather than an arbitrary distance?
4. Volume confirmation — does volume support the move?
5. Market structure — is the higher timeframe trend aligned with the trade direction?
6. Recent bot performance — if the bot is on a losing streak, apply higher scrutiny. \
If win rate is below 40%, only approve the strongest setups (score 80+, R:R 3+).

Be selective. Only approve trades with genuine edge. When in doubt, reject.

Use the record_trade_decision tool to submit your analysis. Think through your \
reasoning step by step BEFORE deciding to approve or reject."""

# Tool definition for structured output — forces chain-of-thought
# by requiring "reasoning" first (LLM generates it before "approve")
TRADE_DECISION_TOOL = {
    "name": "record_trade_decision",
    "description": "Records the trade analysis reasoning and final approval decision.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Step-by-step reasoning evaluating confluence alignment, "
                    "key level quality, risk/reward, volume, market structure, "
                    "and recent performance. 2-3 sentences."
                ),
            },
            "approve": {
                "type": "boolean",
                "description": "Final decision: true to take the trade, false to reject.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in the decision, 0-100.",
            },
            "suggested_sl": {
                "type": ["number", "null"],
                "description": (
                    "Alternative stop-loss price if you see a clearly better "
                    "structural level. null to keep the proposed SL."
                ),
            },
            "suggested_tp": {
                "type": ["number", "null"],
                "description": (
                    "Alternative take-profit price if you see a clearly better "
                    "structural level. null to keep the proposed TP."
                ),
            },
        },
        "required": ["reasoning", "approve", "confidence"],
    },
}


class LLMTradeAnalyst:
    """Claude-powered trade signal analyst with resilient API handling."""

    def __init__(self, config: Settings) -> None:
        self._client: anthropic.AsyncAnthropic | None = None
        self._api_key = config.llm_api_key
        self._model = config.llm_model
        self._timeout = config.llm_timeout_seconds
        self._fallback_approve = config.llm_fallback_approve
        self._min_confidence = config.llm_min_confidence
        self._available = bool(config.llm_api_key)

        # Backoff state (mirrors sentiment.py pattern)
        self._fail_count = 0
        self._backoff_until = 0.0

        # Cumulative cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.total_approvals = 0
        self.total_rejections = 0

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
        """Analyze a trade signal using Claude with tool use for structured output.

        Args:
            signal: The signal candidate to evaluate.
            context: Additional context dict with keys like:
                - sentiment_score, adjusted_score, active_threshold
                - sl_price, tp_price, rr_ratio (proposed levels)
                - open_position_count, recent_win_rate, recent_avg_rr
                - losing_streak, winning_streak

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
                max_tokens=400,
                system=SYSTEM_PROMPT,
                tools=[TRADE_DECISION_TOOL],
                tool_choice={"type": "tool", "name": "record_trade_decision"},
                messages=[{"role": "user", "content": user_prompt}],
            )

            latency_ms = (time.monotonic() - start) * 1000
            self._record_success()

            # Track token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_requests += 1

            analysis = self._parse_tool_response(response)
            analysis.latency_ms = latency_ms
            analysis.input_tokens = input_tokens
            analysis.output_tokens = output_tokens

            # Enforce minimum confidence threshold
            if analysis.approve and analysis.confidence < self._min_confidence:
                analysis.approve = False
                analysis.reasoning = (
                    f"Approved but confidence {analysis.confidence:.0f} "
                    f"< threshold {self._min_confidence:.0f}: {analysis.reasoning}"
                )
                logger.info(
                    "llm_low_confidence_override",
                    symbol=signal.symbol,
                    confidence=analysis.confidence,
                    threshold=self._min_confidence,
                )

            # Validate SL/TP suggestions
            if analysis.suggested_sl is not None or analysis.suggested_tp is not None:
                analysis = self._validate_suggestions(analysis, signal)

            # Track approval/rejection counts
            if analysis.approve:
                self.total_approvals += 1
            else:
                self.total_rejections += 1

            logger.info(
                "llm_analysis_complete",
                symbol=signal.symbol,
                approve=analysis.approve,
                confidence=analysis.confidence,
                latency_ms=round(latency_ms, 1),
                reasoning=analysis.reasoning[:120],
                tokens=input_tokens + output_tokens,
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

        # SL/TP/R:R data
        sl_price = context.get("sl_price")
        tp_price = context.get("tp_price")
        rr_ratio = context.get("rr_ratio")

        sl_tp_section = "Not calculated yet"
        if sl_price is not None:
            sl_distance_pct = abs(signal.entry_price - sl_price) / signal.entry_price * 100 if signal.entry_price > 0 else 0
            parts = [f"**Stop Loss:** {sl_price:.6g} ({sl_distance_pct:.2f}% from entry)"]
            if tp_price is not None:
                tp_distance_pct = abs(tp_price - signal.entry_price) / signal.entry_price * 100 if signal.entry_price > 0 else 0
                parts.append(f"**Take Profit:** {tp_price:.6g} ({tp_distance_pct:.2f}% from entry)")
            if rr_ratio is not None:
                parts.append(f"**Risk:Reward Ratio:** {rr_ratio:.2f}")
            sl_tp_section = "\n".join(parts)

        # Recent performance context
        perf_section = "No data available"
        recent_win_rate = context.get("recent_win_rate")
        if recent_win_rate is not None:
            perf_parts = [f"Win Rate: {recent_win_rate:.1f}%"]
            if context.get("recent_trade_count") is not None:
                perf_parts.append(f"Recent Trades: {context['recent_trade_count']}")
            if context.get("recent_avg_rr") is not None:
                perf_parts.append(f"Avg R:R Achieved: {context['recent_avg_rr']:.2f}")
            if context.get("losing_streak", 0) >= 2:
                perf_parts.append(f"Current Losing Streak: {context['losing_streak']}")
            if context.get("winning_streak", 0) >= 2:
                perf_parts.append(f"Current Winning Streak: {context['winning_streak']}")
            perf_section = ", ".join(perf_parts)

        # Open positions context
        open_count = context.get("open_position_count", 0)

        return f"""\
## Trade Signal Analysis Request

**Symbol:** {signal.symbol}
**Direction:** {direction}
**Entry Price:** {signal.entry_price:.6g}
**Confluence Score:** {signal.score:.1f}/100 (adjusted: {adjusted:.1f}, threshold: {threshold:.1f})
**Open Positions:** {open_count}

### Risk Management Levels
{sl_tp_section}

### Signal Reasons ({len(signal.reasons)} factors)
{chr(10).join(f"- {r}" for r in signal.reasons)}

### Key Levels
{key_levels_str}

### Order Block Context
{ob_info}

### Fair Value Gap Context
{fvg_info}

### Sentiment Score
{sentiment:.1f} (range: -10 to +10, positive = bullish)

### Recent Bot Performance
{perf_section}

Analyze this trade setup and use the record_trade_decision tool to submit your decision.
"""

    def _validate_suggestions(
        self, analysis: LLMAnalysis, signal: SignalCandidate
    ) -> LLMAnalysis:
        """Validate and sanitize LLM-suggested SL/TP levels."""
        is_long = signal.direction == "bullish"
        entry = signal.entry_price

        if analysis.suggested_sl is not None:
            sl = analysis.suggested_sl
            # SL must be on correct side of entry
            if is_long and sl >= entry:
                logger.warning(
                    "llm_sl_invalid", symbol=signal.symbol,
                    suggested_sl=sl, entry=entry, reason="sl_above_entry_for_long",
                )
                analysis.suggested_sl = None
            elif not is_long and sl <= entry:
                logger.warning(
                    "llm_sl_invalid", symbol=signal.symbol,
                    suggested_sl=sl, entry=entry, reason="sl_below_entry_for_short",
                )
                analysis.suggested_sl = None
            else:
                # SL shouldn't be more than 15% from entry (unreasonable risk)
                sl_distance_pct = abs(entry - sl) / entry
                if sl_distance_pct > 0.15:
                    logger.warning(
                        "llm_sl_too_far", symbol=signal.symbol,
                        suggested_sl=sl, distance_pct=f"{sl_distance_pct:.2%}",
                    )
                    analysis.suggested_sl = None

        if analysis.suggested_tp is not None:
            tp = analysis.suggested_tp
            # TP must be on correct side of entry
            if is_long and tp <= entry:
                logger.warning(
                    "llm_tp_invalid", symbol=signal.symbol,
                    suggested_tp=tp, entry=entry, reason="tp_below_entry_for_long",
                )
                analysis.suggested_tp = None
            elif not is_long and tp >= entry:
                logger.warning(
                    "llm_tp_invalid", symbol=signal.symbol,
                    suggested_tp=tp, entry=entry, reason="tp_above_entry_for_short",
                )
                analysis.suggested_tp = None

        return analysis

    def _parse_tool_response(self, response: anthropic.types.Message) -> LLMAnalysis:
        """Extract structured fields from Claude's tool use response."""
        # Find the tool_use block in the response
        for block in response.content:
            if block.type == "tool_use" and block.name == "record_trade_decision":
                data = block.input
                return LLMAnalysis(
                    approve=bool(data.get("approve", self._fallback_approve)),
                    confidence=float(data.get("confidence", 50.0)),
                    reasoning=str(data.get("reasoning", "")),
                    suggested_sl=data.get("suggested_sl"),
                    suggested_tp=data.get("suggested_tp"),
                )

        # Fallback: no tool_use block found (shouldn't happen with tool_choice)
        logger.warning("llm_no_tool_use_block", stop_reason=response.stop_reason)
        return LLMAnalysis(
            approve=self._fallback_approve,
            reasoning="No tool_use block in LLM response",
            error="no_tool_use",
        )

    def get_usage_stats(self) -> dict[str, Any]:
        """Return cumulative usage stats for monitoring."""
        return {
            "total_requests": self.total_requests,
            "total_approvals": self.total_approvals,
            "total_rejections": self.total_rejections,
            "approval_rate": round(
                self.total_approvals / self.total_requests * 100, 1
            ) if self.total_requests > 0 else 0.0,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "fail_count": self._fail_count,
        }

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
