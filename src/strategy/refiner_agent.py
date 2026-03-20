"""AI-powered entry timing agent (Agent 2 — Refiner Monitor).

Agent 1 (AgentEntryAnalyst) makes the STRATEGIC decision: "Is this worth trading?"
Agent 2 (RefinerMonitorAgent) makes the TACTICAL decision: "Is NOW the right time?"

Agent 2 replaces the algorithmic rejection detection in the entry refiner.
It runs every 5 minutes on queued signals and returns concrete parameters:
  - ENTER: exact entry_price, stop_loss, take_profit, position_size_modifier
  - WAIT: stay in queue, reasoning stored for continuity on next check

Agent 2 has reasoning continuity — its previous analysis is included in each
prompt so it can build on its own observations across checks.

Uses Gemini Interactions API (JSON response format), mirroring Agent 1's
resilience patterns (lazy client, exponential backoff, cost tracking).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.config import Settings
from src.strategy.llm_client import (
    LLMResult, generate_json, is_openai_model, is_anthropic_model,
    get_api_key_for_model, MODEL_PRICING,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RefinerDecision:
    """Result from Agent 2 — tactical entry timing decision."""

    action: str = "WAIT"  # ENTER, WAIT
    entry_price: float = 0.0  # Exact price for ENTER
    adjusted_zone_high: float = 0.0  # Deprecated — kept for backward compat
    adjusted_zone_low: float = 0.0  # Deprecated — kept for backward compat
    stop_loss: float = 0.0  # Concrete SL price
    take_profit: float = 0.0  # Concrete TP price
    position_size_modifier: float = 1.0  # 0.25-1.5 scale factor
    reasoning: str = ""
    urgency: str = "low"  # low / medium / high
    invalidation_reason: str = ""  # Deprecated — kept for backward compat
    confidence: float = 0.0  # 0-100
    latency_ms: float = 0.0
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


REFINER_SYSTEM_PROMPT = """\
You are an expert crypto futures execution specialist. Your job is TACTICAL ENTRY TIMING \
based on live 5-minute price action data.

## ABSOLUTE RULES — NON-NEGOTIABLE
1. **Agent 1 is the strategic authority.** You MUST NOT question, override, or re-evaluate \
Agent 1's thesis, direction, or trade rationale. The DIRECTION (LONG/SHORT) is FINAL. \
You cannot flip it, second-guess it, or suggest a different direction.
2. **Your ONLY job is finding the right moment to pull the trigger.** Decide WHETHER to \
ENTER NOW or WAIT for better 5m confirmation — but NEVER because you disagree with the \
direction or thesis.
3. **Price zone is a GUIDE, not a hard gate for continuations.** If the original zone has \
already been reached and a rejection occurred (even hours ago in older candles), the setup is \
ALREADY WORKING. Price may have moved past the zone — that's a CONTINUATION, not a miss. \
ENTER at current price for continuations.
4. **Scan ALL candles for rejection evidence.** Check EVERY candle in the table, not just the \
most recent one. A rejection that occurred 2-4 hours ago (in older candles) is STILL VALID. \
Look for: wick into/past the zone with wick:body > 1.0, bearish/bullish engulfing at zone, \
or volume spike with reversal at zone. If you find rejection evidence in ANY candle, the \
rejection requirement IS MET.
5. **Use Agent 1's levels as your STARTING reference.** Agent 1 provides the initial entry \
zone, SL, TP, invalidation level, and structural thesis. The DIRECTION and THESIS are locked. \
Apply your OWN 5m expertise to decide timing.

## Context
Agent 1 (the strategic analyst) has already approved this trade setup, chosen the direction, \
and identified the entry zone, stop-loss, and take-profit levels. The direction is locked. \
Your job is to read the live 5-minute candles and structural data to decide whether \
the entry conditions are met RIGHT NOW.

You are called every 5 minutes with fresh data: the latest 5m candles, structural levels \
(order blocks, FVGs, liquidity pools, market structure), volume profile, Fibonacci levels, \
BTC macro context, and your own previous analysis for reasoning continuity.

## How to Analyze — Step by Step

For EVERY check, you must perform these concrete data checks:

### Step 0: Check the must_reach_price gate
If a must_reach_price is set and shows "NOT YET REACHED", note it but still proceed to \
Step 1 — the system applies a tolerance automatically. Focus on whether the displacement \
move has clearly occurred (price moved significantly from the sweep level). If displacement \
is evident in the candle data, treat the gate as satisfied and continue evaluation.

### Step 1: Has a rejection occurred at or near the entry zone?
Scan ALL candles in the table (not just the latest) looking for rejection evidence:
- A candle with a wick into/past the zone boundary with wick:body ratio > 1.0
- A bearish engulfing (shorts) or bullish engulfing (longs) at or near the zone
- A volume spike candle at the zone showing reversal
- A sequence of candles forming higher-lows (longs) or lower-highs (shorts) at the zone

**If rejection found in ANY candle (even hours ago):** Proceed to Step 2 — the rejection \
requirement is MET. Note which candle showed the rejection.

**If NO rejection found anywhere:** WAIT — price hasn't tested the zone yet.

### Step 2: Is this a live rejection or a continuation?
- **Live rejection (price is currently IN or near the zone):** Classic setup. Use current price \
as entry_price. Standard confidence based on rejection quality.
- **Continuation (rejection occurred in older candles, price has moved in the trade direction):** \
The setup is ALREADY WORKING. Price has moved past the zone after rejecting — this confirms \
the thesis. ENTER at current price with high confidence. This is the ideal scenario: \
the market has already proven the thesis correct.

### Step 3: Do the structural levels confirm?
- Is price at or near an ORDER BLOCK? (check the OB list for one within 0.2% of current price)
- Is price inside a FAIR VALUE GAP? (check the price_in_fvg flag)
- Does MARKET STRUCTURE support the direction? (check trend per timeframe)
- Is there LIQUIDITY on the opposite side? (that's where TP targets live)

### Step 4: Make your decision
- **Rejection found + structure confirms** → ENTER with high confidence (70-90)
- **Rejection found but structure neutral** → ENTER with moderate confidence (50-70)
- **Continuation (past rejection + price moving in direction)** → ENTER with high confidence (75-90)
- **No rejection found yet** → WAIT
- **must_reach_price not reached** → WAIT

## Your Decision Options

### 1. ENTER — Execute the trade NOW
You have confirmed rejection evidence in the candle data.

**Concrete ENTER criteria (at least ONE must be true):**
- Price wicked into the zone AND a candle closed back in the trade direction (wick ratio > 1.0)
- Price formed a higher-low (long) or lower-high (short) at/near the zone over 2+ candles
- An engulfing candle formed at the zone boundary
- **CONTINUATION:** Rejection occurred in an older candle and price has since moved in the \
trade direction — the thesis is confirmed, ENTER at current price

**You MUST provide:** entry_price, stop_loss, take_profit, position_size_modifier, confidence

### 2. WAIT — Check again in 5 minutes
No rejection evidence found yet in any candle.

**When to WAIT:**
- Price has not reached the entry zone yet
- No rejection pattern visible in any candle in the table
- must_reach_price has not been reached
- BTC just made a sharp move — wait for the next candle to see if it stabilizes

Note: the system automatically expires signals after their time window.

## Setting Stop Loss & Take Profit

### Stop Loss:
- Use Agent 1's SL from the "Pre-Computed Levels" section — this is the validated final SL
- You may TIGHTEN the SL based on 5m structure (e.g. below nearest 5m swing low for longs)
- Do NOT widen it beyond Agent 1's level
- Longs: SL MUST be below entry_price. Shorts: SL MUST be above entry_price

### Take Profit:
- Use Agent 1's TP from the "Pre-Computed Levels" section
- Cross-reference with structural levels visible on 5m
- The system uses progressive TP tiers (TP1 +2%, TP2 +3%, TP3 +4%) automatically — your TP is the overall target

### Position Size Modifier:
- 1.0 = standard size (most entries)
- 1.25-1.5 = strong rejection + HTF trend aligned + volume spike + structural confluence
- 0.5-0.75 = entry conditions met but confirmation is weak or BTC is uncertain
- 0.25-0.5 = marginal entry, minimal confirmation

## Urgency Levels
- **high:** Entry conditions are confirmed NOW in the current candle data
- **medium:** Conditions developing — likely 1-2 more candles (5-10 minutes)
- **low:** Price is approaching but 15-30 minutes from zone contact

## Order Book Analysis (if available)
If order book data is provided, use it as additional confirmation:
- **Bid imbalance (positive ratio):** More buyers than sellers — supports long entries
- **Ask imbalance (negative ratio):** More sellers than buyers — supports short entries
- **Wide spread (>0.05%):** Low liquidity — reduce position_size_modifier (use 0.5-0.75)
- If order book is unavailable, skip this analysis step entirely — do NOT penalize the signal.

## Reading 5-Minute Candles — Key Metrics
For each candle in the table, extract:
- **Wick ratio:** (wick at zone side) / (body size). >1.0 = rejection, >2.0 = strong rejection
- **Volume comparison:** Is this candle's volume higher than the average of the prior 5 candles?
- **Close location:** Did the candle close in the upper or lower third of its range?
- **Sequence:** Are the last 2-3 candles forming higher-lows (bullish) or lower-highs (bearish)?
- **Body size relative to range:** Small body + long wicks = indecision. Large body = conviction

## Glossary
- **SL** — Stop Loss  |  **TP** — Take Profit  |  **R:R** — Risk-to-Reward Ratio
- **OB** — Order Block  |  **FVG** — Fair Value Gap  |  **BOS** — Break of Structure
- **CHoCH** — Change of Character  |  **OTE** — Optimal Trade Entry (61.8-78.6% Fib)
- **HTF** — Higher Timeframe (4H, Daily)  |  **LTF** — Lower Timeframe (5m, 15m)
- **RVOL** — Relative Volume (>1.5 = elevated)  |  **ATR** — Average True Range
- **Displacement** — Large-bodied candle with above-average volume
- **Liquidity Sweep** — Price wicked through a key level to trigger stops, then reversed
- **Wick Ratio** — Rejection wick / body size (>1.0 = rejection, >2.0 = strong)
- **Engulfing** — Candle body fully encompasses prior candle's body"""


REFINER_JSON_FORMAT = """

## RESPONSE FORMAT — CRITICAL

You MUST respond with a single JSON object and NOTHING else. No markdown, no explanation, no text before or after.

The JSON object must have exactly these keys:
{
  "action": "ENTER" | "WAIT",
  "entry_price": <number or 0>,
  "stop_loss": <number or 0>,
  "take_profit": <number or 0>,
  "position_size_modifier": <number 0.25-1.5>,
  "confidence": <number 0-100>,
  "urgency": "low" | "medium" | "high",
  "reasoning": "<2-4 sentence tactical analysis>"
}

Rules:
- For ENTER: entry_price, stop_loss, take_profit MUST be non-zero. position_size_modifier between 0.25-1.5.
- For WAIT: all price fields can be 0. Explain what specific data condition you are waiting for.
- reasoning: ALWAYS provide 2-4 sentences that REFERENCE SPECIFIC DATA from the context:
  cite actual candle prices, wick ratios, volume numbers, OB/FVG levels. \
  Never say vague things like "waiting for confirmation" — say exactly WHAT you checked \
  and WHAT the numbers showed. Example: "Candle at 23:15 wicked to 0.01892 (inside zone \
  0.01890-0.01910) with a 2.3x wick ratio and 145% RVOL, confirming bullish rejection. \
  Price has since moved to 0.01850 — this is a continuation, entering now."
- confidence: 0-100, based on how many of the analysis steps confirmed.

Respond with ONLY the JSON object. No other text."""


# Gemini structured output schema for Agent 2
REFINER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["ENTER", "WAIT"]},
        "entry_price": {"type": "number"},
        "stop_loss": {"type": "number"},
        "take_profit": {"type": "number"},
        "position_size_modifier": {"type": "number"},
        "confidence": {"type": "number"},
        "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasoning": {"type": "string"},
    },
    "required": [
        "action", "entry_price", "stop_loss", "take_profit",
        "position_size_modifier", "confidence", "urgency",
        "reasoning",
    ],
}


class RefinerMonitorAgent:
    """Agent 2 — AI-powered tactical entry timing.

    Mirrors Agent 1's infrastructure: lazy Gemini AsyncClient, exponential
    backoff, token/cost tracking, JSON response parsing with fallbacks.
    """

    def __init__(self, config: Settings) -> None:
        self._model = getattr(config, "agent_model", "gemini-3-flash-preview")
        # Agent 2 always uses flash for Gemini — override pro model
        if not is_openai_model(self._model) and "pro" in self._model:
            self._model = "gemini-3-flash-preview"
        self._timeout = config.agent_timeout_seconds
        self._max_sl_pct = getattr(config, "max_sl_pct", 0.15)
        self._thinking_level = "low"  # Agent 2 refine: fast tactical decisions

        # Store all API keys so runtime model switching works across providers
        self._gemini_api_key = config.agent_api_key
        self._openai_api_key = config.openai_api_key
        self._anthropic_api_key = getattr(config, "anthropic_api_key", "")
        self._api_key = get_api_key_for_model(
            self._model,
            openai_key=self._openai_api_key,
            anthropic_key=self._anthropic_api_key,
            gemini_key=self._gemini_api_key,
        )
        self._available = bool(self._api_key)

        # Backoff state (mirrors Agent 1)
        self._fail_count = 0
        self._backoff_until = 0.0

        # Cumulative stats
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_enter = 0
        self.total_wait = 0
        self.total_adjust = 0
        self.total_abandon = 0
        self.total_errors = 0
        self.total_cost_usd = 0.0

        # Pricing per 1M tokens by model (input, cached_input, output)
        self._pricing = MODEL_PRICING

        # Available models for runtime switching
        self.available_models = list(self._pricing.keys())

    def set_model(self, model: str) -> str:
        """Switch model at runtime. Returns the active model name."""
        if model not in self._pricing:
            raise ValueError(f"Unknown model: {model}. Available: {self.available_models}")
        old = self._model
        self._model = model
        # Switch API key to match the new provider
        self._api_key = get_api_key_for_model(
            model,
            openai_key=self._openai_api_key,
            anthropic_key=self._anthropic_api_key,
            gemini_key=self._gemini_api_key,
        )
        self._available = bool(self._api_key)
        logger.info("refiner_agent_model_switched", old_model=old, new_model=model)
        return self._model

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
            "refiner_agent_backoff",
            fail_count=self._fail_count,
            backoff_seconds=backoff,
        )

    def _record_success(self) -> None:
        if self._fail_count > 0:
            self._fail_count = 0
            logger.info("refiner_agent_recovered")

    async def evaluate_entry(self, context: dict[str, Any]) -> RefinerDecision:
        """Evaluate a queued signal and make a tactical entry decision.

        Args:
            context: Dict with keys like:
                - symbol, direction, current_price
                - agent1_analysis (action, reasoning, confidence, regime, risk)
                - previous_reasoning, previous_action, previous_urgency, check_count
                - zone_top, zone_bottom, zone_source
                - candles_5m_table (formatted string of last 15 candles)
                - btc_trend, btc_price, btc_1h_change
                - structural_levels (OBs, FVGs, liquidity, market structure)
                - volume_profile (RVOL, displacement)
                - fibonacci_levels

        Returns:
            RefinerDecision with action and concrete parameters.
        """
        if not self._should_try():
            return RefinerDecision(
                action="WAIT",
                reasoning="Agent 2 API unavailable (backoff), deferring to algorithmic fallback",
                error="api_backoff",
            )

        user_prompt = self._build_prompt(context)
        # Sanitize prompt: strip null bytes and control chars that break JSON serialization
        user_prompt = user_prompt.replace("\x00", "")
        system_prompt = REFINER_SYSTEM_PROMPT + REFINER_JSON_FORMAT
        start = time.monotonic()

        try:
            result = await generate_json(
                model=self._model,
                api_key=self._api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=REFINER_RESPONSE_SCHEMA,
                thinking_level=self._thinking_level,
                temperature=1.0,
                timeout=self._timeout,
            )

            latency_ms = (time.monotonic() - start) * 1000
            self._record_success()

            # Track token usage & cost
            input_tokens = result.input_tokens
            output_tokens = result.output_tokens
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_requests += 1

            # Calculate cost in USD
            price_in, _price_cached, price_out = self._pricing.get(
                self._model, (0.10, 0.01, 0.40)
            )
            call_cost = (
                input_tokens * price_in / 1_000_000
                + output_tokens * price_out / 1_000_000
            )
            self.total_cost_usd += call_cost

            decision = self._parse_response(result.text)
            decision.latency_ms = latency_ms
            decision.input_tokens = input_tokens
            decision.output_tokens = output_tokens

            # Validate decision
            decision = self._validate_decision(decision, context)

            # Track action counts
            if decision.action == "ENTER":
                self.total_enter += 1
            else:
                self.total_wait += 1

            logger.info(
                "refiner_agent_decision",
                symbol=context.get("symbol", "?"),
                action=decision.action,
                confidence=decision.confidence,
                urgency=decision.urgency,
                latency_ms=round(latency_ms, 1),
                reasoning=decision.reasoning[:200],
                tokens=input_tokens + output_tokens,
                cost_usd=round(call_cost, 6),
                total_cost_usd=round(self.total_cost_usd, 4),
                check_count=context.get("check_count", 0),
            )
            return decision

        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            error_str = str(e)

            # If OpenAI returns 400 (bad request), try Gemini as fallback
            if (
                is_openai_model(self._model)
                and "400" in error_str
                and self._gemini_api_key
            ):
                logger.warning(
                    "refiner_agent_openai_400_fallback_to_gemini",
                    symbol=context.get("symbol", "?"),
                    error=error_str[:150],
                )
                try:
                    fallback_model = "gemini-3-flash-preview"
                    result = await generate_json(
                        model=fallback_model,
                        api_key=self._gemini_api_key,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        json_schema=REFINER_RESPONSE_SCHEMA,
                        thinking_level=self._thinking_level,
                        temperature=1.0,
                        timeout=self._timeout,
                    )
                    latency_ms = (time.monotonic() - start) * 1000
                    self._record_success()
                    decision = self._parse_response(result.text)
                    decision.latency_ms = latency_ms
                    decision.input_tokens = result.input_tokens
                    decision.output_tokens = result.output_tokens
                    decision = self._validate_decision(decision, context)
                    if decision.action == "ENTER":
                        self.total_enter += 1
                    else:
                        self.total_wait += 1
                    logger.info(
                        "refiner_agent_decision_via_gemini_fallback",
                        symbol=context.get("symbol", "?"),
                        action=decision.action,
                        confidence=decision.confidence,
                        latency_ms=round(latency_ms, 1),
                    )
                    return decision
                except Exception as fallback_e:
                    logger.warning(
                        "refiner_agent_gemini_fallback_also_failed",
                        symbol=context.get("symbol", "?"),
                        error=str(fallback_e)[:150],
                    )

            self._record_failure()
            logger.warning(
                "refiner_agent_failed",
                symbol=context.get("symbol", "?"),
                error=error_str,
                latency_ms=round(latency_ms, 1),
            )
            return RefinerDecision(
                action="WAIT",
                reasoning=f"Agent 2 API error: {e}",
                latency_ms=latency_ms,
                error=error_str,
            )

    def _build_prompt(self, ctx: dict[str, Any]) -> str:
        """Build the user prompt with all tactical context for Agent 2."""

        symbol = ctx.get("symbol", "?")
        direction = ctx.get("direction", "unknown")
        current_price = ctx.get("current_price", 0)

        # ── Section 1: Signal identity + Agent 1's analysis ──
        a1 = ctx.get("agent1_analysis", {})
        a1_parts = [
            f"Agent 1 Action: {a1.get('action', 'N/A')}",
            f"Agent 1 Confidence: {a1.get('confidence', 'N/A')}",
            f"Agent 1 Reasoning: {a1.get('reasoning', 'N/A')}",
            f"Market Regime: {a1.get('market_regime', 'N/A')}",
            f"Risk Assessment: {a1.get('risk_assessment', 'N/A')}",
        ]
        # Add invalidation level
        if a1.get("invalidation_level"):
            a1_parts.append(f"Invalidation Level: {a1['invalidation_level']:.6g} (thesis dead beyond this price)")
        # Add must_reach_price
        if a1.get("must_reach_price"):
            a1_parts.append(f"Must Reach Price: {a1['must_reach_price']:.6g} (price must hit this BEFORE pullback)")
        a1_section = "\n".join(a1_parts)

        # ── Section 2: Agent 2's previous analysis (reasoning continuity) ──
        prev_action = ctx.get("previous_action", "")
        prev_reasoning = ctx.get("previous_reasoning", "")
        prev_urgency = ctx.get("previous_urgency", "")
        check_count = ctx.get("check_count", 0)

        if prev_action and check_count > 0:
            prev_section = (
                f"- **Action:** {prev_action}\n"
                f"- **Urgency:** {prev_urgency}\n"
                f"- **Reasoning:** \"{prev_reasoning}\"\n"
                f"- **Check #:** {check_count} of this signal"
            )
        else:
            prev_section = "This is your first evaluation of this signal."

        # ── Section 3: Entry zone ──
        zone_top = ctx.get("zone_top", 0)
        zone_bottom = ctx.get("zone_bottom", 0)
        zone_source = ctx.get("zone_source", "unknown")
        zone_mid = (zone_top + zone_bottom) / 2 if zone_top > 0 and zone_bottom > 0 else 0

        # ── Section 4: Price vs zone ──
        price_vs_zone = "unknown"
        distance_pct = 0.0
        if zone_top > 0 and zone_bottom > 0 and current_price > 0:
            if current_price > zone_top:
                price_vs_zone = "ABOVE zone"
                distance_pct = (current_price - zone_top) / zone_top * 100
            elif current_price < zone_bottom:
                price_vs_zone = "BELOW zone"
                distance_pct = (zone_bottom - current_price) / zone_bottom * 100
            else:
                price_vs_zone = "INSIDE zone"
                distance_pct = 0.0

        price_change = ctx.get("price_change_since_queue_pct", 0)

        # ── Section 5: 5m candle table ──
        candles_table = ctx.get("candles_5m_table", "No candle data available")

        # ── Section 6: BTC macro ──
        btc_trend = ctx.get("btc_trend", "unknown")
        btc_price = ctx.get("btc_price", 0)
        btc_1h_change = ctx.get("btc_1h_change", 0)

        # ── Section 7: Structural levels (flat dict from scanner) ──
        structural = ctx.get("structural_levels", {})
        struct_parts = []

        # Order Blocks
        obs = structural.get("order_blocks", [])
        if obs:
            struct_parts.append("  Order Blocks (1H):")
            for ob in obs[:5]:
                if isinstance(ob, dict):
                    struct_parts.append(
                        f"    {ob.get('direction', '?')} OB "
                        f"[{ob.get('bottom', 0):.6g} – {ob.get('top', 0):.6g}] "
                        f"str={ob.get('strength', 0):.2f} "
                        f"dist={ob.get('distance_pct', 0):.1f}%"
                    )

        # Fair Value Gaps
        fvgs = structural.get("fair_value_gaps", [])
        if fvgs:
            struct_parts.append("  Fair Value Gaps (1H):")
            for fvg in fvgs[:5]:
                if isinstance(fvg, dict):
                    struct_parts.append(
                        f"    {fvg.get('direction', '?')} FVG "
                        f"[{fvg.get('bottom', 0):.6g} – {fvg.get('top', 0):.6g}] "
                        f"mid={fvg.get('midpoint', 0):.6g} "
                        f"dist={fvg.get('distance_pct', 0):.1f}%"
                    )

        # Liquidity levels
        liq = structural.get("liquidity", {})
        if liq and (liq.get("nearest_buy") or liq.get("nearest_sell")):
            struct_parts.append("  Liquidity Pools (1H):")
            if liq.get("nearest_buy"):
                struct_parts.append(
                    f"    Buy liquidity @ {liq['nearest_buy']:.6g} "
                    f"(dist={liq.get('buy_distance_pct', 0):.1f}%)"
                )
            if liq.get("nearest_sell"):
                struct_parts.append(
                    f"    Sell liquidity @ {liq['nearest_sell']:.6g} "
                    f"(dist={liq.get('sell_distance_pct', 0):.1f}%)"
                )
            struct_parts.append(
                f"    Active pools: {liq.get('active_pool_count', 0)} | "
                f"Recent sweeps: {liq.get('recent_sweeps', 0)}"
            )

        # Market Structure per timeframe
        ms_dict = structural.get("market_structure", {})
        if ms_dict:
            struct_parts.append("  Market Structure:")
            for tf in ("1h", "4h", "1d"):
                ms = ms_dict.get(tf, {})
                if ms:
                    struct_parts.append(
                        f"    {tf}: trend={ms.get('trend', '?')} "
                        f"str={ms.get('strength', 0):.2f} "
                        f"BOS={ms.get('last_bos', '?')} "
                        f"CHoCH={ms.get('last_choch', '?')}"
                    )

        # Price in OB/FVG flags
        flags = []
        if structural.get("price_in_ob"):
            flags.append("Price is INSIDE an order block")
        if structural.get("price_in_fvg"):
            flags.append("Price is INSIDE a fair value gap")
        if flags:
            struct_parts.append("  Flags: " + " | ".join(flags))

        structural_section = "\n".join(struct_parts) if struct_parts else "  Not available"

        # ── Section 8: Volume profile ──
        vol = ctx.get("volume_profile", {})
        vol_section = "Not available"
        if vol:
            parts = []
            if vol.get("relative_volume"):
                parts.append(f"RVOL: {vol['relative_volume']:.2f}x")
            if vol.get("displacement_detected"):
                parts.append("Displacement: YES")
            if vol.get("volume_trend"):
                parts.append(f"Trend: {vol['volume_trend']}")
            vol_section = " | ".join(parts) if parts else "Not available"

        # ── Section 9: Fibonacci levels ──
        fib = ctx.get("fibonacci_levels", {})
        if fib and fib.get("fib_50"):
            fib_section = (
                f"50.0%: {fib['fib_50']:.6g} | "
                f"61.8%: {fib['fib_618']:.6g} | "
                f"78.6%: {fib['fib_786']:.6g}\n"
                f"  OTE Zone: {fib['fib_618']:.6g} – {fib['fib_786']:.6g}"
            )
        else:
            fib_section = "Not available"

        # ── Section 10: Pullback metrics ──
        pullback = ctx.get("pullback", {})
        pullback_parts = []
        if pullback:
            if pullback.get("retracement_pct"):
                pullback_parts.append(f"Retracement: {pullback['retracement_pct']:.1f}%")
            if pullback.get("thrust_extreme"):
                pullback_parts.append(f"Thrust extreme: {pullback['thrust_extreme']:.6g}")
            if pullback.get("optimal_entry"):
                pullback_parts.append(f"Optimal entry: {pullback['optimal_entry']:.6g}")
            if pullback.get("pullback_status"):
                pullback_parts.append(f"Status: {pullback['pullback_status']}")
            if pullback.get("displacement_open"):
                pullback_parts.append(f"Displacement open: {pullback['displacement_open']:.6g}")
        pullback_section = " | ".join(pullback_parts) if pullback_parts else "Not available"

        # ── Section 11: Leverage/funding profile ──
        leverage = ctx.get("leverage", {})
        leverage_parts = []
        if leverage:
            if leverage.get("funding_rate") is not None:
                leverage_parts.append(f"Funding: {leverage['funding_rate']:.6f}")
            if leverage.get("funding_bias"):
                leverage_parts.append(f"Bias: {leverage['funding_bias']}")
            if leverage.get("crowded_side"):
                leverage_parts.append(
                    f"Crowded: {leverage['crowded_side']} "
                    f"({leverage.get('crowding_intensity', 0):.0%})"
                )
            if leverage.get("long_short_ratio"):
                leverage_parts.append(f"L/S ratio: {leverage['long_short_ratio']:.2f}")
            if leverage.get("nearest_long_liq"):
                leverage_parts.append(f"Long liq @ {leverage['nearest_long_liq']:.6g}")
            if leverage.get("nearest_short_liq"):
                leverage_parts.append(f"Short liq @ {leverage['nearest_short_liq']:.6g}")
            if leverage.get("judas_swing_probability", 0) > 0.3:
                leverage_parts.append(
                    f"⚠ Judas swing prob: {leverage['judas_swing_probability']:.0%}"
                )
        leverage_bonus = ctx.get("leverage_bonus")
        if leverage_bonus is not None:
            leverage_parts.append(f"Alignment bonus: {leverage_bonus} pts")
        leverage_section = " | ".join(leverage_parts) if leverage_parts else "Not available"

        # ── Section 12: Order book ──
        ob_data = ctx.get("order_book", {})
        if ob_data.get("status") == "available":
            ob_parts = [
                f"Spread: {ob_data.get('spread_pct', 0):.4f}%",
                f"Bid vol (top10): {ob_data.get('bid_volume_top10', 0):.2f}",
                f"Ask vol (top10): {ob_data.get('ask_volume_top10', 0):.2f}",
                f"Imbalance: {ob_data.get('imbalance_ratio', 0):+.3f} (+1=all bids, -1=all asks)",
                f"Best bid: {ob_data.get('best_bid', 0):.6g}",
                f"Best ask: {ob_data.get('best_ask', 0):.6g}",
            ]
            walls = ob_data.get("walls", [])
            if walls:
                for w in walls:
                    ob_parts.append(
                        f"  Wall: {w.get('side', '?')} @ {w.get('price', 0):.6g} "
                        f"(vol={w.get('volume', 0):.2f})"
                    )
            order_book_section = "\n".join(ob_parts)
        else:
            order_book_section = "Not available for this market"

        # ── Section 13a: Sweep details ──
        sweep = ctx.get("sweep_info", {})
        if sweep:
            sweep_target = sweep.get('sweep_direction', '?')
            sweep_desc = (
                "sell-side liquidity taken (swept lows)" if sweep_target == "swing_low"
                else "buy-side liquidity taken (swept highs)" if sweep_target == "swing_high"
                else sweep_target
            )
            sweep_section = (
                f"Type: {sweep.get('sweep_type', '?')} | "
                f"Sweep Target: {sweep_target} ({sweep_desc}) | "
                f"Depth: {sweep.get('sweep_depth', 0):.6g} | "
                f"Level: {sweep.get('sweep_level', 0):.6g}"
            )
        else:
            sweep_section = "Not available"

        # ── Section 13a-ii: Displacement assessment ──
        disp = ctx.get("displacement_assessment", {})
        if disp:
            displacement_section = (
                f"- **Price has moved {disp['displacement_from_sweep_pct']:.2f}% from the sweep level**\n"
                f"- Zone is {disp['zone_distance_from_price_pct']:.2f}% away from current price\n"
                f"- Reaching the zone requires a {disp['zone_retrace_of_move_pct']:.1f}% retracement of the move"
            )
        else:
            displacement_section = ""

        # ── Section 13b: PullbackPlan context ──
        plan_ctx = ctx.get("pullback_plan")
        if plan_ctx:
            time_left = plan_ctx.get("time_remaining_seconds", 0)
            minutes_left = time_left / 60
            # Phase gate info
            must_reach = plan_ctx.get("must_reach_price", 0)
            must_reached = plan_ctx.get("must_reach_price_reached", False)
            phase_line = ""
            if must_reach and must_reach > 0:
                status = "REACHED" if must_reached else "NOT YET REACHED"
                phase_line = f"- **Must reach price:** {must_reach:.6g} ({status}) — entry only valid AFTER this level is hit\n"

            plan_section = (
                f"### Pending Order Plan\n"
                f"- **Zone:** {plan_ctx.get('zone_low', 0):.6g} – {plan_ctx.get('zone_high', 0):.6g}\n"
                f"- **Limit price:** {plan_ctx.get('limit_price', 0):.6g}\n"
                f"- **Invalidation:** {plan_ctx.get('invalidation_level', 0):.6g}\n"
                f"{phase_line}"
                f"- **Time remaining:** {minutes_left:.1f} min\n"
                f"- **Max chase:** {plan_ctx.get('max_chase_bps', 0):.1f} bps\n"
                f"- **Zone updates:** {plan_ctx.get('zone_updates', 0)}\n\n"
            )
        else:
            plan_section = ""

        # ── Section 13: Pre-computed SL/TP from Agent 1 ──
        a1_sl = ctx.get("agent1_sl")
        a1_tp = ctx.get("agent1_tp")
        a1_rr = ctx.get("agent1_rr")
        sltp_parts = []
        if a1_sl is not None:
            sltp_parts.append(f"Stop Loss: {a1_sl:.6g}")
        if a1_tp is not None:
            sltp_parts.append(f"Take Profit: {a1_tp:.6g}")
        if a1_rr is not None:
            sltp_parts.append(f"R:R: {a1_rr:.2f}")
        pre_sltp_section = " | ".join(sltp_parts) if sltp_parts else "Not calculated"

        # ── Check count context ──
        check_count = ctx.get("check_count", 0)
        if check_count >= 1:
            action_hint = f"\n**Evaluation #{check_count + 1}** — review your previous analysis below and update based on new data."
        else:
            action_hint = ""

        # ── SETUP_CONFIRMED confirmation mode ──
        setup_confirmed_hint = ""
        if ctx.get("setup_confirmed_mode"):
            setup_confirmed_hint = (
                "\n**SETUP_CONFIRMED:** Agent 1 has validated the setup. Scan ALL candles in the "
                "table for rejection evidence at the entry zone. If a rejection occurred in ANY "
                "candle (even hours ago) and price has moved in the trade direction, this is a "
                "CONTINUATION — ENTER at current price.\n"
            )

        return f"""\
## Tactical Entry Evaluation

**Symbol:** {symbol}
**Direction:** {direction} ⚠️ NON-NEGOTIABLE — set by Agent 1, do NOT change or question
**Current Price:** {current_price:.6g}
{action_hint}
{setup_confirmed_hint}
### Agent 1's Strategic Analysis (direction & thesis are FINAL — entry zone is adjustable)
{a1_section}

### Pre-Computed Levels (use these for your SL/TP)
{pre_sltp_section}

### Your Previous Analysis (5 minutes ago)
{prev_section}

### Sweep Details
{sweep_section}

### Displacement Assessment (how far has the move already gone?)
{displacement_section if displacement_section else "Not available"}

### Entry Zone
- **Source:** {zone_source}
- **Zone:** {zone_bottom:.6g} – {zone_top:.6g}
- **Midpoint:** {zone_mid:.6g}

### Price vs Zone
- **Position:** {price_vs_zone}
- **Distance from zone:** {distance_pct:.2f}%
- **Price change since queued:** {price_change:+.2f}%

{plan_section}### Pullback / Displacement Metrics
{pullback_section}

### Last 50 Five-Minute Candles (newest first)
{candles_table}

### BTC Macro Context
- Trend: {btc_trend}
- Price: {btc_price:.2f}
- 1H Change: {btc_1h_change:+.2f}%

### Structural Levels (1H from scanner)
{structural_section}

### Volume Profile
{vol_section}

### Fibonacci Retracement Levels
{fib_section}

### Order Book
{order_book_section}

### Leverage / Funding Profile
{leverage_section}

### Symbol Trade History (previous entries on this token)
{self._build_history_section(ctx)}

### Similar Past Trades (RAG Knowledge Base)
{ctx.get("rag_context", "  Not available")}
{ctx.get("lessons_context", "")}
Run through the 3-step analysis: (1) price vs zone, (2) latest candle rejection signals, \
(3) structural confirmation. Check order book for liquidity support. \
If previous trades on this symbol show repeated losses, demand stronger 5m confirmation. \
Use similar past trade outcomes from the knowledge base to inform your confidence level. \
Reference specific numbers from the data in your reasoning."""

    @staticmethod
    def _build_history_section(ctx: dict) -> str:
        """Build trade history section for Agent 2 prompt."""
        history = ctx.get("symbol_history", [])
        if not history:
            return "  No prior trades for this symbol"

        lines = []
        wins = sum(1 for t in history if (t.get("pnl_usd") or 0) > 0)
        losses = len(history) - wins
        lines.append(f"  Last {len(history)} trades: {wins}W / {losses}L")
        for t in history:
            pnl = t.get("pnl_usd", 0) or 0
            pnl_pct = t.get("pnl_percent", 0) or 0
            exit_reason = t.get("exit_reason", "unknown")
            direction_h = t.get("direction", "?")
            entry_p = t.get("entry_price", 0) or 0
            exit_p = t.get("exit_price", 0) or 0
            w_l = "WIN" if pnl > 0 else "LOSS"
            hold_str = ""
            if t.get("entry_time") and t.get("exit_time"):
                try:
                    from datetime import datetime as _dt
                    et = _dt.fromisoformat(str(t["entry_time"]).replace("Z", "+00:00"))
                    xt = _dt.fromisoformat(str(t["exit_time"]).replace("Z", "+00:00"))
                    hold_mins = int((xt - et).total_seconds() / 60)
                    hold_str = f", held {hold_mins}m"
                except Exception:
                    pass
            lines.append(
                f"  - {direction_h} {w_l}: ${pnl:+.2f} ({pnl_pct:+.1f}%), "
                f"exit={exit_reason}, entry={entry_p:.6g}→{exit_p:.6g}{hold_str}"
            )
        return "\n".join(lines)

    def _parse_response(self, text: str) -> RefinerDecision:
        """Extract structured JSON from LLM response text."""
        import json as _json

        if text:
            text = text.strip()
        if text:
            try:
                # Strip markdown code fences if present
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                data = _json.loads(text)
                return self._data_to_decision(data)
            except (_json.JSONDecodeError, Exception) as e:
                logger.warning(
                    "refiner_agent_json_parse_failed",
                    error=str(e),
                    content_preview=text[:200],
                )

        logger.warning("refiner_agent_no_response", text_preview=text[:100] if text else "empty")
        return RefinerDecision(
            action="WAIT",
            reasoning="No parseable response from Agent 2",
            error="no_response",
        )

    def _data_to_decision(self, data: dict) -> RefinerDecision:
        """Convert parsed dict to RefinerDecision."""
        action = data.get("action", "WAIT").upper()
        # Normalize action — only ENTER and WAIT are valid
        if action not in ("ENTER", "WAIT"):
            action = "WAIT"

        return RefinerDecision(
            action=action,
            entry_price=float(data.get("entry_price", 0) or 0),
            stop_loss=float(data.get("stop_loss", 0) or 0),
            take_profit=float(data.get("take_profit", 0) or 0),
            position_size_modifier=float(data.get("position_size_modifier", 1.0) or 1.0),
            confidence=float(data.get("confidence", 0) or 0),
            urgency=str(data.get("urgency", "low") or "low"),
            reasoning=str(data.get("reasoning", "") or ""),
        )

    def _validate_decision(
        self, decision: RefinerDecision, context: dict[str, Any]
    ) -> RefinerDecision:
        """Validate and sanitize Agent 2's decision."""
        direction = context.get("direction", "")
        is_long = direction.lower() in ("bullish", "long", "swing_low") if direction else False
        current_price = context.get("current_price", 0)

        if decision.action == "ENTER":
            # Must have entry_price, SL, TP
            if decision.entry_price <= 0:
                # Fall back to current price
                decision.entry_price = current_price

            if decision.stop_loss <= 0:
                logger.warning(
                    "refiner_agent_enter_no_sl",
                    symbol=context.get("symbol", "?"),
                )
                decision.action = "WAIT"
                decision.reasoning = f"ENTER downgraded to WAIT: no SL provided. {decision.reasoning}"
                return decision

            if decision.take_profit <= 0:
                logger.warning(
                    "refiner_agent_enter_no_tp",
                    symbol=context.get("symbol", "?"),
                )
                decision.action = "WAIT"
                decision.reasoning = f"ENTER downgraded to WAIT: no TP provided. {decision.reasoning}"
                return decision

            # Validate SL direction
            if is_long and decision.stop_loss >= decision.entry_price:
                logger.warning(
                    "refiner_agent_sl_wrong_side",
                    symbol=context.get("symbol", "?"),
                    sl=decision.stop_loss,
                    entry=decision.entry_price,
                    direction=direction,
                )
                decision.action = "WAIT"
                decision.reasoning = f"ENTER downgraded to WAIT: SL above entry for long. {decision.reasoning}"
                return decision

            if not is_long and decision.stop_loss <= decision.entry_price:
                logger.warning(
                    "refiner_agent_sl_wrong_side",
                    symbol=context.get("symbol", "?"),
                    sl=decision.stop_loss,
                    entry=decision.entry_price,
                    direction=direction,
                )
                decision.action = "WAIT"
                decision.reasoning = f"ENTER downgraded to WAIT: SL below entry for short. {decision.reasoning}"
                return decision

            # Validate TP direction
            if is_long and decision.take_profit <= decision.entry_price:
                logger.warning(
                    "refiner_agent_tp_wrong_side",
                    symbol=context.get("symbol", "?"),
                )
                decision.action = "WAIT"
                decision.reasoning = f"ENTER downgraded to WAIT: TP below entry for long. {decision.reasoning}"
                return decision

            if not is_long and decision.take_profit >= decision.entry_price:
                logger.warning(
                    "refiner_agent_tp_wrong_side",
                    symbol=context.get("symbol", "?"),
                )
                decision.action = "WAIT"
                decision.reasoning = f"ENTER downgraded to WAIT: TP above entry for short. {decision.reasoning}"
                return decision

            # Validate R:R >= 1.5
            # Use Agent 1's original plan R:R if available — Agent 2's
            # entry price drifts with current price and can collapse R:R
            # even when the original trade thesis is still valid.
            a1_rr = context.get("agent1_rr")
            sl_distance = abs(decision.entry_price - decision.stop_loss)
            tp_distance = abs(decision.take_profit - decision.entry_price)
            rr = (tp_distance / sl_distance) if sl_distance > 0 else 0
            plan_rr = a1_rr if a1_rr and a1_rr > 0 else rr
            if plan_rr < 1.5 and rr < 1.5:
                if sl_distance > 0:
                    logger.warning(
                        "refiner_agent_low_rr",
                        symbol=context.get("symbol", "?"),
                        rr=round(rr, 2),
                        plan_rr=round(plan_rr, 2),
                    )
                    decision.action = "WAIT"
                    decision.reasoning = (
                        f"ENTER downgraded to WAIT: R:R {rr:.1f} < 1.5 minimum. "
                        f"{decision.reasoning}"
                    )
                    return decision

            # Validate SL isn't too far (>15% from entry)
            if current_price > 0:
                sl_dist_pct = abs(decision.entry_price - decision.stop_loss) / current_price
                if sl_dist_pct > self._max_sl_pct:
                    logger.warning(
                        "refiner_agent_sl_too_far",
                        symbol=context.get("symbol", "?"),
                        sl_distance_pct=f"{sl_dist_pct:.2%}",
                    )
                    decision.action = "WAIT"
                    decision.reasoning = f"ENTER downgraded to WAIT: SL too far ({sl_dist_pct:.1%}). {decision.reasoning}"
                    return decision

            # Validate entry price isn't too far from current price (>2%)
            if current_price > 0:
                entry_drift = abs(decision.entry_price - current_price) / current_price
                if entry_drift > 0.02:
                    logger.warning(
                        "refiner_agent_entry_too_far",
                        symbol=context.get("symbol", "?"),
                        entry_drift_pct=f"{entry_drift:.2%}",
                    )
                    # Use current price instead
                    decision.entry_price = current_price

            # Clamp position_size_modifier
            decision.position_size_modifier = max(0.25, min(1.5, decision.position_size_modifier))

        # Any non-ENTER action is treated as WAIT
        elif decision.action != "WAIT":
            decision.action = "WAIT"

        return decision

    def get_usage_stats(self) -> dict[str, Any]:
        """Return cumulative usage stats for dashboard."""
        avg_cost = (
            round(self.total_cost_usd / self.total_requests, 6)
            if self.total_requests > 0
            else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "total_enter": self.total_enter,
            "total_wait": self.total_wait,
            "total_adjust": getattr(self, "total_adjust", 0),
            "total_abandon": getattr(self, "total_abandon", 0),
            "enter_rate": round(
                self.total_enter / self.total_requests * 100, 1
            ) if self.total_requests > 0 else 0.0,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "avg_cost_per_call_usd": avg_cost,
            "total_errors": self.total_errors,
            "fail_count": self._fail_count,
            "model": self._model,
        }

    async def close(self) -> None:
        pass  # Clients cached in llm_client module
