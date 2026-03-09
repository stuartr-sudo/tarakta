"""AI-powered entry agent using OpenAI GPT models.

Unlike the legacy LLM analyst (binary approve/reject gate), this agent
makes the actual entry DECISION. Given a candidate that passed initial
formula screening (sweep detected, score >= 35), the agent reasons about
the full market context and returns one of:

  ENTER_NOW    — Take the trade at current price
  WAIT_PULLBACK — Wait for pullback to a specific level (feeds entry_refiner)
  SKIP         — Don't take this trade, the context is wrong

This gives us an edge because:
  - Most competitors use only formulas/indicators
  - The agent can reason about qualitative context (market regime, news,
    correlations, recent performance patterns) that formulas can't capture
  - The agent can adapt its reasoning without code changes

Uses OpenAI function calling (tool use) for structured output, mirroring
the resilience patterns from the Anthropic-based llm_analyst.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from src.config import Settings
from src.exchange.models import SignalCandidate
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentDecision:
    """Result from the AI entry agent."""

    action: str = "SKIP"  # ENTER_NOW, WAIT_PULLBACK, SKIP
    confidence: float = 0.0  # 0-100
    reasoning: str = ""
    suggested_entry: float | None = None  # For WAIT_PULLBACK: target price
    suggested_sl: float | None = None
    suggested_tp: float | None = None
    market_regime: str = ""  # trending/ranging/volatile/choppy
    risk_assessment: str = ""  # low/medium/high/extreme
    latency_ms: float = 0.0
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


SYSTEM_PROMPT = """\
You are an elite crypto futures trader specializing in Smart Money Concepts (ICT methodology). \
You manage a systematic trading bot and your job is to make the FINAL entry decision.

The bot's formula system has already detected a candidate signal. It found a completed liquidity \
sweep (price wicked through a key level and closed back), possibly confirmed by displacement \
(large-bodied candle with volume). Your job is to decide WHETHER to actually enter this trade \
and HOW — not just yes/no, but the optimal approach.

## Your Decision Options

1. **ENTER_NOW** — The setup is strong. Enter at the current market price immediately.
   Use when: sweep + displacement confirmed, HTF aligned, good timing, favorable context.

2. **WAIT_PULLBACK** — The setup has potential but entry price isn't optimal. Wait for price \
   to pull back to a specific level before entering.
   Use when: sweep confirmed but no pullback yet, or price ran too far from sweep level, \
   or you see a better entry zone on the structure.

3. **SKIP** — This trade isn't worth taking despite the formula signal.
   Use when: market regime is wrong, BTC context is dangerous, news risk is elevated, \
   the sweep looks like a fake/manufactured signal, R:R doesn't justify the risk, \
   or recent performance suggests the bot should be more selective.

## What Makes a GREAT Entry (approve aggressively)
- Sweep of Asian/London range high or low with large-body displacement candle
- HTF (4H + Daily) trend aligned with trade direction
- Post-kill-zone timing (manipulation is done, distribution begins)
- Volume spike on displacement (institutional participation)
- Clean R:R of 3:1 or better with structural SL placement
- Pullback into 50-79% Fibonacci OTE zone of the displacement move

## What Makes a BAD Entry (skip aggressively)
- Sweep without displacement (could be a fake sweep, price may continue)
- Counter-trend to 4H AND Daily (fighting the tide)
- Monday first 8 hours (likely fake move / manipulation)
- Sweep of a weak/arbitrary level (not session range, not swing high/low)
- Recent losing streak with similar setups (the pattern isn't working)
- Extreme funding rate in the same direction as trade (crowded)
- BTC in freefall or parabolic — altcoin setups become unreliable

## Risk Assessment
Rate the overall risk as: low, medium, high, extreme
- low: strong confluence, HTF aligned, clean structure
- medium: decent setup with 1-2 missing components
- high: marginal setup, proceed only if other factors are exceptional
- extreme: too many red flags, should almost always SKIP

## Market Regime
Classify the current market as: trending, ranging, volatile, choppy
This helps the bot calibrate its other parameters.

Be decisive. Don't hedge. If it's marginal, SKIP — there will always be another trade. \
The best traders make money by the trades they DON'T take.

Use the record_entry_decision tool to submit your analysis."""


ENTRY_DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": "record_entry_decision",
        "description": "Records the entry decision with reasoning and suggested parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Step-by-step analysis of the setup. Cover: sweep quality, "
                        "displacement strength, HTF alignment, timing, risk/reward, "
                        "market regime, and any red flags. 3-5 sentences."
                    ),
                },
                "action": {
                    "type": "string",
                    "enum": ["ENTER_NOW", "WAIT_PULLBACK", "SKIP"],
                    "description": (
                        "ENTER_NOW: take trade immediately. "
                        "WAIT_PULLBACK: wait for better price. "
                        "SKIP: don't take this trade."
                    ),
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in the decision, 0-100. 80+ = very confident.",
                },
                "suggested_entry": {
                    "type": ["number", "null"],
                    "description": (
                        "For WAIT_PULLBACK: the price level to wait for. "
                        "For ENTER_NOW: null (use current price). "
                        "For SKIP: null."
                    ),
                },
                "suggested_sl": {
                    "type": ["number", "null"],
                    "description": (
                        "Alternative stop-loss price if you see a better structural level. "
                        "null to keep the formula-calculated SL."
                    ),
                },
                "suggested_tp": {
                    "type": ["number", "null"],
                    "description": (
                        "Alternative take-profit if you see a better target. "
                        "null to keep the formula-calculated TP."
                    ),
                },
                "market_regime": {
                    "type": "string",
                    "enum": ["trending", "ranging", "volatile", "choppy"],
                    "description": "Current market regime classification.",
                },
                "risk_assessment": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "extreme"],
                    "description": "Overall risk level of this trade.",
                },
            },
            "required": [
                "reasoning", "action", "confidence",
                "market_regime", "risk_assessment",
            ],
        },
    },
}


class AgentEntryAnalyst:
    """OpenAI-powered entry agent with resilient API handling.

    Replaces the binary approve/reject LLM gate with an intelligent
    decision-maker that chooses between ENTER_NOW, WAIT_PULLBACK, and SKIP.
    """

    def __init__(self, config: Settings) -> None:
        self._client: AsyncOpenAI | None = None
        self._api_key = config.agent_api_key
        self._model = config.agent_model
        self._timeout = config.agent_timeout_seconds
        self._fallback_approve = config.agent_fallback_approve
        self._min_confidence = config.agent_min_confidence
        self._available = bool(config.agent_api_key)

        # Backoff state (mirrors sentiment.py / llm_analyst.py)
        self._fail_count = 0
        self._backoff_until = 0.0

        # Cumulative stats
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_enter_now = 0
        self.total_wait_pullback = 0
        self.total_skip = 0
        self.total_errors = 0

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
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
        self.total_errors += 1
        backoff = min(300, 30 * (2 ** (self._fail_count - 1)))
        self._backoff_until = time.time() + backoff
        logger.warning(
            "agent_api_backoff",
            fail_count=self._fail_count,
            backoff_seconds=backoff,
        )

    def _record_success(self) -> None:
        if self._fail_count > 0:
            self._fail_count = 0
            logger.info("agent_api_recovered")

    async def analyze_signal(
        self,
        signal: SignalCandidate,
        context: dict[str, Any],
    ) -> AgentDecision:
        """Analyze a trade signal and make an entry decision.

        Args:
            signal: The signal candidate from the formula system.
            context: Additional context dict with keys like:
                - sentiment_score, adjusted_score, active_threshold
                - sl_price, tp_price, rr_ratio
                - open_position_count, recent_win_rate, recent_avg_rr
                - losing_streak, winning_streak
                - btc_trend, btc_price_change_pct
                - recent_headlines

        Returns:
            AgentDecision with action, confidence, and reasoning.
        """
        if not self._should_try():
            action = "ENTER_NOW" if self._fallback_approve else "SKIP"
            return AgentDecision(
                action=action,
                reasoning="Agent API unavailable (backoff), using fallback",
                error="api_backoff",
            )

        user_prompt = self._build_prompt(signal, context)
        start = time.monotonic()

        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._model,
                max_tokens=600,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                tools=[ENTRY_DECISION_TOOL],
                tool_choice={"type": "function", "function": {"name": "record_entry_decision"}},
            )

            latency_ms = (time.monotonic() - start) * 1000
            self._record_success()

            # Track token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_requests += 1

            decision = self._parse_response(response)
            decision.latency_ms = latency_ms
            decision.input_tokens = input_tokens
            decision.output_tokens = output_tokens

            # Enforce minimum confidence
            if decision.action == "ENTER_NOW" and decision.confidence < self._min_confidence:
                decision.action = "SKIP"
                decision.reasoning = (
                    f"ENTER_NOW but confidence {decision.confidence:.0f} "
                    f"< threshold {self._min_confidence:.0f}: {decision.reasoning}"
                )
                logger.info(
                    "agent_low_confidence_override",
                    symbol=signal.symbol,
                    confidence=decision.confidence,
                    threshold=self._min_confidence,
                )

            # Validate SL/TP suggestions
            if decision.suggested_sl is not None or decision.suggested_tp is not None:
                decision = self._validate_suggestions(decision, signal)

            # Track action counts
            if decision.action == "ENTER_NOW":
                self.total_enter_now += 1
            elif decision.action == "WAIT_PULLBACK":
                self.total_wait_pullback += 1
            else:
                self.total_skip += 1

            logger.info(
                "agent_decision",
                symbol=signal.symbol,
                action=decision.action,
                confidence=decision.confidence,
                risk=decision.risk_assessment,
                regime=decision.market_regime,
                latency_ms=round(latency_ms, 1),
                reasoning=decision.reasoning[:150],
                tokens=input_tokens + output_tokens,
            )
            return decision

        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            self._record_failure()
            logger.warning(
                "agent_analysis_failed",
                symbol=signal.symbol,
                error=str(e),
                latency_ms=round(latency_ms, 1),
            )
            action = "ENTER_NOW" if self._fallback_approve else "SKIP"
            return AgentDecision(
                action=action,
                reasoning=f"Agent API error: {e}",
                latency_ms=latency_ms,
                error=str(e),
            )

    def _build_prompt(self, signal: SignalCandidate, context: dict[str, Any]) -> str:
        """Build the user prompt with all signal and market context."""
        direction = signal.direction or "unknown"

        # Score breakdown
        components_str = "\n".join(
            f"  - {k}: {v}" for k, v in signal.components.items()
        ) if signal.components else "  Not available"

        # Key levels
        key_levels_str = "\n".join(
            f"  - {k}: {v:.6g}" for k, v in signal.key_levels.items()
        ) if signal.key_levels else "  None"

        # Order block context
        ob_info = "None"
        if signal.ob_context:
            ob = signal.ob_context
            ob_info = f"{ob.direction} OB [{ob.bottom:.6g} - {ob.top:.6g}], strength={ob.strength:.2f}"

        # FVG context
        fvg_info = "None"
        if signal.fvg_context:
            fvg = signal.fvg_context
            fvg_info = f"{fvg.direction} FVG [{fvg.bottom:.6g} - {fvg.top:.6g}]"

        # Sweep details
        sweep_info = "None"
        if signal.sweep_result:
            sr = signal.sweep_result
            sweep_info = (
                f"Type: {sr.sweep_type}, Direction: {sr.sweep_direction}, "
                f"Depth: {sr.sweep_depth:.4f}, Level: {sr.sweep_level:.6g}"
            )
            if sr.htf_continuation:
                sweep_info += " (HTF continuation override)"

        # SL/TP/R:R
        sl_price = context.get("sl_price")
        tp_price = context.get("tp_price")
        rr_ratio = context.get("rr_ratio")
        sl_tp_section = "Not calculated yet"
        if sl_price is not None:
            sl_dist_pct = abs(signal.entry_price - sl_price) / signal.entry_price * 100 if signal.entry_price > 0 else 0
            parts = [f"Stop Loss: {sl_price:.6g} ({sl_dist_pct:.2f}% from entry)"]
            if tp_price is not None:
                tp_dist_pct = abs(tp_price - signal.entry_price) / signal.entry_price * 100 if signal.entry_price > 0 else 0
                parts.append(f"Take Profit: {tp_price:.6g} ({tp_dist_pct:.2f}% from entry)")
            if rr_ratio is not None:
                parts.append(f"Risk:Reward: {rr_ratio:.2f}")
            sl_tp_section = " | ".join(parts)

        # Sentiment
        sentiment = context.get("sentiment_score", 0.0)

        # Recent performance
        perf_parts = []
        if context.get("recent_win_rate") is not None:
            perf_parts.append(f"Win Rate: {context['recent_win_rate']:.1f}%")
        if context.get("recent_trade_count") is not None:
            perf_parts.append(f"Recent Trades: {context['recent_trade_count']}")
        if context.get("recent_avg_rr") is not None:
            perf_parts.append(f"Avg R:R: {context['recent_avg_rr']:.2f}")
        if context.get("losing_streak", 0) >= 2:
            perf_parts.append(f"LOSING STREAK: {context['losing_streak']} trades")
        if context.get("winning_streak", 0) >= 2:
            perf_parts.append(f"Winning Streak: {context['winning_streak']} trades")
        perf_section = " | ".join(perf_parts) if perf_parts else "No recent data"

        # Headlines
        headlines = context.get("recent_headlines", [])
        headlines_section = "\n".join(f"  - {h}" for h in headlines[:5]) if headlines else "  None"

        # BTC context
        btc_trend = context.get("btc_trend", "unknown")
        btc_change = context.get("btc_price_change_pct")
        btc_section = f"Trend: {btc_trend}"
        if btc_change is not None:
            btc_section += f", 24h Change: {btc_change:+.2f}%"

        # ML win probability
        ml_prob = context.get("ml_win_probability")
        ml_section = f"{ml_prob:.0f}%" if ml_prob is not None else "N/A"

        # Open positions
        open_count = context.get("open_position_count", 0)
        adjusted = context.get("adjusted_score", signal.score)
        threshold = context.get("active_threshold", 65.0)

        # Weekly cycle context (may not be set on all signals)
        weekly_info = ""
        weekly_factors = getattr(signal, "weekly_factors", None)
        if weekly_factors:
            if weekly_factors.get("is_monday_manipulation"):
                weekly_info = "WARNING: Monday manipulation window active"
            elif weekly_factors.get("is_midweek_reversal"):
                weekly_info = "Mid-week reversal window (Wed/Thu counter-trend bonus)"

        # Leverage/funding context (may not be set on all signals)
        leverage_info = ""
        leverage_bonus = getattr(signal, "leverage_bonus", None)
        if leverage_bonus is not None:
            leverage_info = f"Leverage alignment bonus: {leverage_bonus} pts"

        return f"""\
## Entry Decision Request

**Symbol:** {signal.symbol}
**Direction:** {direction}
**Current Price:** {signal.entry_price:.6g}
**Formula Score:** {signal.score:.1f}/100 (adjusted: {adjusted:.1f}, threshold: {threshold:.1f})
**Open Positions:** {open_count}

### Sweep Analysis
{sweep_info}

### Score Breakdown
{components_str}

### Risk Management
{sl_tp_section}

### Signal Reasons ({len(signal.reasons)} factors)
{chr(10).join(f"  - {r}" for r in signal.reasons)}

### Key Structural Levels
{key_levels_str}

### Order Block: {ob_info}
### Fair Value Gap: {fvg_info}

### Market Context
- BTC: {btc_section}
- Sentiment: {sentiment:.1f} (range: -10 to +10, positive = bullish)
- ML Win Probability: {ml_section}
{f"- {weekly_info}" if weekly_info else ""}\
{f"- {leverage_info}" if leverage_info else ""}

### Recent Headlines
{headlines_section}

### Bot Performance
{perf_section}

Analyze this setup and use record_entry_decision to submit your decision."""

    def _parse_response(self, response) -> AgentDecision:
        """Extract structured fields from OpenAI's function call response."""
        choice = response.choices[0]
        message = choice.message

        # Check for tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "record_entry_decision":
                    import json
                    data = json.loads(tool_call.function.arguments)
                    return AgentDecision(
                        action=data.get("action", "SKIP"),
                        confidence=float(data.get("confidence", 0)),
                        reasoning=str(data.get("reasoning", "")),
                        suggested_entry=data.get("suggested_entry"),
                        suggested_sl=data.get("suggested_sl"),
                        suggested_tp=data.get("suggested_tp"),
                        market_regime=data.get("market_regime", ""),
                        risk_assessment=data.get("risk_assessment", ""),
                    )

        # Fallback: no tool call found (shouldn't happen with tool_choice)
        logger.warning("agent_no_tool_call", finish_reason=choice.finish_reason)
        return AgentDecision(
            action="SKIP",
            reasoning="No tool call in agent response",
            error="no_tool_call",
        )

    def _validate_suggestions(
        self, decision: AgentDecision, signal: SignalCandidate
    ) -> AgentDecision:
        """Validate and sanitize agent-suggested SL/TP/entry levels."""
        is_long = signal.direction == "bullish"
        entry = signal.entry_price

        if decision.suggested_sl is not None:
            sl = decision.suggested_sl
            if is_long and sl >= entry:
                logger.warning("agent_sl_invalid", symbol=signal.symbol, reason="sl_above_entry_for_long")
                decision.suggested_sl = None
            elif not is_long and sl <= entry:
                logger.warning("agent_sl_invalid", symbol=signal.symbol, reason="sl_below_entry_for_short")
                decision.suggested_sl = None
            else:
                sl_distance_pct = abs(entry - sl) / entry
                if sl_distance_pct > 0.15:
                    logger.warning("agent_sl_too_far", symbol=signal.symbol, distance=f"{sl_distance_pct:.2%}")
                    decision.suggested_sl = None

        if decision.suggested_tp is not None:
            tp = decision.suggested_tp
            if is_long and tp <= entry:
                logger.warning("agent_tp_invalid", symbol=signal.symbol, reason="tp_below_entry_for_long")
                decision.suggested_tp = None
            elif not is_long and tp >= entry:
                logger.warning("agent_tp_invalid", symbol=signal.symbol, reason="tp_above_entry_for_short")
                decision.suggested_tp = None

        if decision.suggested_entry is not None and decision.action == "WAIT_PULLBACK":
            target = decision.suggested_entry
            # Entry target should be between current price and SL
            distance_pct = abs(target - entry) / entry
            if distance_pct > 0.10:
                logger.warning("agent_entry_too_far", symbol=signal.symbol, distance=f"{distance_pct:.2%}")
                decision.suggested_entry = None
            # For longs, pullback target should be below current price
            if is_long and target >= entry:
                logger.warning("agent_entry_above_current", symbol=signal.symbol)
                decision.suggested_entry = None
            # For shorts, pullback target should be above current price
            elif not is_long and target <= entry:
                logger.warning("agent_entry_below_current", symbol=signal.symbol)
                decision.suggested_entry = None

        return decision

    def get_usage_stats(self) -> dict[str, Any]:
        """Return cumulative usage stats for monitoring/dashboard."""
        return {
            "total_requests": self.total_requests,
            "total_enter_now": self.total_enter_now,
            "total_wait_pullback": self.total_wait_pullback,
            "total_skip": self.total_skip,
            "enter_rate": round(
                self.total_enter_now / self.total_requests * 100, 1
            ) if self.total_requests > 0 else 0.0,
            "skip_rate": round(
                self.total_skip / self.total_requests * 100, 1
            ) if self.total_requests > 0 else 0.0,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_errors": self.total_errors,
            "fail_count": self._fail_count,
            "model": self._model,
        }

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
