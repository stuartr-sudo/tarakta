"""AI-powered entry timing agent (Agent 2 — Refiner Monitor).

Agent 1 (AgentEntryAnalyst) makes the STRATEGIC decision: "Is this worth trading?"
Agent 2 (RefinerMonitorAgent) makes the TACTICAL decision: "Is NOW the right time?"

Agent 2 replaces the algorithmic rejection detection in the entry refiner.
It runs every 5 minutes on queued signals and returns concrete parameters:
  - ENTER: exact entry_price, stop_loss, take_profit, position_size_modifier
  - WAIT: stay in queue, reasoning stored for continuity on next check
  - ADJUST_ZONE: update zone boundaries based on new structure
  - ABANDON: remove from queue, setup is invalidated

Agent 2 has reasoning continuity — its previous analysis is included in each
prompt so it can build on its own observations across checks.

Uses OpenAI function calling (JSON response format), mirroring Agent 1's
resilience patterns (lazy client, exponential backoff, cost tracking).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from src.config import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RefinerDecision:
    """Result from Agent 2 — tactical entry timing decision."""

    action: str = "WAIT"  # ENTER, WAIT, ADJUST_ZONE, ABANDON
    entry_price: float = 0.0  # Exact price for ENTER
    adjusted_zone_high: float = 0.0  # New zone top for ADJUST_ZONE
    adjusted_zone_low: float = 0.0  # New zone bottom for ADJUST_ZONE
    stop_loss: float = 0.0  # Concrete SL price
    take_profit: float = 0.0  # Concrete TP price
    position_size_modifier: float = 1.0  # 0.25-1.5 scale factor
    reasoning: str = ""
    urgency: str = "low"  # low / medium / high
    invalidation_reason: str = ""  # For ABANDON
    confidence: float = 0.0  # 0-100
    latency_ms: float = 0.0
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


REFINER_SYSTEM_PROMPT = """\
You are an expert crypto futures execution specialist. Your job is TACTICAL ENTRY TIMING \
based on live 5-minute price action data.

## Context
Agent 1 (the strategic analyst) has already approved this trade setup and identified the \
entry zone, stop-loss, and take-profit levels. DO NOT re-evaluate the thesis. \
Your job is to read the live 5-minute candles and structural data to decide whether \
the entry conditions are met RIGHT NOW.

You are called every 5 minutes with fresh data: the latest 5m candles, structural levels \
(order blocks, FVGs, liquidity pools, market structure), volume profile, Fibonacci levels, \
BTC macro context, and your own previous analysis for reasoning continuity.

## How to Analyze — Step by Step

For EVERY check, you must perform these concrete data checks:

### Step 1: Where is price relative to the entry zone?
- Read current_price and compare to zone_top / zone_bottom
- If INSIDE the zone → proceed to Step 2 (possible entry)
- If ABOVE the zone (for longs) → setup may have run without us → check for ABANDON
- If BELOW the zone (for longs) → price hasn't arrived yet → WAIT or check approach speed
- Reverse the logic for shorts

### Step 2: What does the latest 5m candle show?
Read the MOST RECENT candle from the candle table:
- **For long entries:** Does the candle have a lower wick into/below the zone? \
Is the close above the zone bottom? Is the body bullish (close > open)?
- **For short entries:** Does the candle have an upper wick into/above the zone? \
Is the close below the zone top? Is the body bearish (close < open)?
- **Wick ratio:** Calculate (wick length) / (body size). Ratio > 1.0 = strong rejection
- **Volume:** Compare the latest candle's volume to the previous 3-5 candles. Higher = confirmation

### Step 3: Do the structural levels confirm?
- Is price at or near an ORDER BLOCK? (check the OB list for one within 0.2% of current price)
- Is price inside a FAIR VALUE GAP? (check the price_in_fvg flag)
- Does MARKET STRUCTURE support the direction? (check trend per timeframe)
- Is there LIQUIDITY on the opposite side? (that's where TP targets live)

### Step 4: Make your decision based on what the data shows
- **All 3 steps confirm** → ENTER with high confidence (70-90)
- **Steps 1-2 confirm but Step 3 is neutral** → ENTER with moderate confidence (50-70)
- **Step 1 says price is in zone but Step 2 shows no rejection yet** → WAIT (candle still forming)
- **Step 1 says price hasn't reached zone** → WAIT (price approaching)
- **Structure is broken against the trade** → ABANDON

## Your Decision Options

### 1. ENTER — Execute the trade NOW
You have confirmed on the 5m chart that price is rejecting from the entry zone.

**Concrete ENTER criteria (check the actual candle data):**
- Price wicked into the zone AND the candle closed back in the trade direction
- The rejection candle has a wick-to-body ratio > 1.0 at the zone
- OR: Price formed a higher-low (long) or lower-high (short) inside the zone over 2+ candles
- OR: An engulfing candle formed at the zone boundary
- Volume on the rejection candle is above average (compare to prior candles in the table)

**You MUST provide:** entry_price, stop_loss, take_profit, position_size_modifier, confidence

### 2. WAIT — Check again in 5 minutes
The setup is still valid but the entry condition is NOT yet confirmed in the data.

**When to WAIT:**
- Price has not reached the entry zone yet (check "Price vs Zone" section)
- Price is in the zone but no rejection candle has formed (current candle is still undecided)
- BTC just made a sharp move — wait for the next candle to see if it stabilizes
- Volume is below average — no institutional participation visible yet

### 3. ADJUST_ZONE — Update zone boundaries
New 5m price action has created a better reference point.

**When to ADJUST:**
- A new swing high/low has formed on 5m that shifts the optimal entry
- The zone was wicked through but held — tighten the zone to the wick level
- A new order block has printed that provides a better entry reference

**You MUST provide:** adjusted_zone_high, adjusted_zone_low

### 4. ABANDON — Remove from queue
The setup is invalidated by the live data.

**Concrete ABANDON criteria:**
- A 5m candle CLOSED with its BODY beyond the zone (not just a wick)
- Break of structure (BOS) against the trade direction: new lower-low (for longs) or higher-high (for shorts)
- Multiple consecutive candles closing against the trade direction
- BTC moved sharply against the trade (>1% in 5 minutes against the direction)

**You MUST provide:** invalidation_reason

## Setting Stop Loss & Take Profit

### Stop Loss:
- Use the PRE-COMPUTED SL from the "Pre-Computed Levels" section as your primary reference
- If Agent 1's reasoning specifies a SL price, use that
- You may tighten the SL based on 5m structure (e.g., place below the nearest 5m swing low for longs)
- Do NOT widen it beyond the pre-computed level
- **Longs:** SL MUST be below entry_price. **Shorts:** SL MUST be above entry_price

### Take Profit:
- Use the PRE-COMPUTED TP from the "Pre-Computed Levels" section
- If Agent 1's reasoning specifies a TP price, use that
- Cross-reference with structural levels: nearest opposing liquidity pool, order block, or FVG midpoint
- The system uses progressive TP tiers (0.70R, 0.95R, 1.5R) automatically after entry — \
your TP serves as the overall target reference

### Position Size Modifier:
- 1.0 = standard size (most entries)
- 1.25-1.5 = strong rejection + HTF trend aligned + volume spike + structural confluence
- 0.5-0.75 = entry conditions met but confirmation is weak or BTC is uncertain
- 0.25-0.5 = marginal entry, minimal confirmation

## Urgency Levels
- **high:** Entry conditions are confirmed NOW in the current candle data
- **medium:** Conditions developing — likely 1-2 more candles (5-10 minutes)
- **low:** Price is approaching but 15-30 minutes from zone contact

## Reading 5-Minute Candles — Key Metrics
For each candle in the table, extract:
- **Wick ratio:** (wick at zone side) / (body size). >1.0 = rejection, >2.0 = strong rejection
- **Volume comparison:** Is this candle's volume higher than the average of the prior 5 candles?
- **Close location:** Did the candle close in the upper or lower third of its range?
- **Sequence:** Are the last 2-3 candles forming higher-lows (bullish) or lower-highs (bearish)?
- **Body size relative to range:** Small body + long wicks = indecision. Large body = conviction

## Glossary (standard terms used in the data context)

- **SL** — Stop Loss: price where the trade exits at a loss
- **TP** — Take Profit: price where the trade exits at a profit
- **R:R** — Risk-to-Reward Ratio: TP distance / SL distance (e.g. 2:1 = TP is 2x SL distance)
- **OB** — Order Block: institutional supply/demand zone (bullish OB = last down-candle before rally; bearish OB = last up-candle before drop)
- **FVG** — Fair Value Gap: three-candle imbalance where candle 1 and candle 3 don't overlap — price tends to fill this gap
- **BOS** — Break of Structure: swing high/low broken, confirming trend direction
- **CHoCH** — Change of Character: first break against the trend, signaling potential reversal
- **OTE** — Optimal Trade Entry: the 61.8%–78.6% Fibonacci retracement zone
- **HTF** — Higher Timeframe (4H, Daily): trend/bias direction
- **LTF** — Lower Timeframe (5m, 15m): your execution timeframe
- **RVOL** — Relative Volume: current volume vs average (>1.5 = elevated, >2.5 = institutional)
- **Displacement** — Large-bodied candle with above-average volume showing institutional order flow
- **Liquidity Sweep** — Price wicked through a key level to trigger stops, then reversed
- **Wick Ratio** — Length of the rejection wick divided by candle body size (>1.0 = rejection signal, >2.0 = strong rejection)
- **Pin Bar** — A candle with a small body and a long wick (2x+ body) showing rejection from a level
- **Engulfing** — A candle whose body fully encompasses the previous candle's body, showing momentum shift"""


REFINER_JSON_FORMAT = """

## RESPONSE FORMAT — CRITICAL

You MUST respond with a single JSON object and NOTHING else. No markdown, no explanation, no text before or after.

The JSON object must have exactly these keys:
{
  "action": "ENTER" | "WAIT" | "ADJUST_ZONE" | "ABANDON",
  "entry_price": <number or 0>,
  "stop_loss": <number or 0>,
  "take_profit": <number or 0>,
  "adjusted_zone_high": <number or 0>,
  "adjusted_zone_low": <number or 0>,
  "position_size_modifier": <number 0.25-1.5>,
  "confidence": <number 0-100>,
  "urgency": "low" | "medium" | "high",
  "reasoning": "<2-4 sentence tactical analysis>",
  "invalidation_reason": "<string, empty if not ABANDON>"
}

Rules:
- For ENTER: entry_price, stop_loss, take_profit MUST be non-zero. position_size_modifier between 0.25-1.5.
- For WAIT: all price fields can be 0. Explain what specific data condition you are waiting for.
- For ADJUST_ZONE: adjusted_zone_high and adjusted_zone_low MUST be non-zero. Other prices can be 0.
- For ABANDON: invalidation_reason MUST be non-empty. All prices can be 0.
- reasoning: ALWAYS provide 2-4 sentences that REFERENCE SPECIFIC DATA from the context:
  cite actual candle prices, wick ratios, volume numbers, OB/FVG levels. \
  Never say vague things like "waiting for confirmation" — say exactly WHAT you checked \
  and WHAT the numbers showed. Example: "Latest 5m candle wicked to 0.01892 (inside zone \
  0.01890-0.01910) with a 2.3x wick ratio and 145% RVOL, confirming bullish rejection."
- confidence: 0-100, based on how many of the 3 analysis steps confirmed.

Respond with ONLY the JSON object. No other text."""


class RefinerMonitorAgent:
    """Agent 2 — AI-powered tactical entry timing.

    Mirrors Agent 1's infrastructure: lazy AsyncOpenAI client, exponential
    backoff, token/cost tracking, JSON response parsing with fallbacks.
    """

    def __init__(self, config: Settings) -> None:
        self._client: AsyncOpenAI | None = None
        self._api_key = config.agent_api_key
        self._model = config.agent_model
        self._timeout = config.agent_timeout_seconds
        self._available = bool(config.agent_api_key)

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
        self._pricing: dict[str, tuple[float, float, float]] = {
            "gpt-5-mini": (0.25, 0.025, 2.00),
            "gpt-5.4": (2.50, 0.25, 15.00),
        }

        # Available models for runtime switching
        self.available_models = list(self._pricing.keys())

    def set_model(self, model: str) -> str:
        """Switch model at runtime. Returns the active model name."""
        if model not in self._pricing:
            raise ValueError(f"Unknown model: {model}. Available: {self.available_models}")
        old = self._model
        self._model = model
        logger.info("refiner_agent_model_switched", old_model=old, new_model=model)
        return self._model

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
        start = time.monotonic()

        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._model,
                max_completion_tokens=2000,
                messages=[
                    {"role": "system", "content": REFINER_SYSTEM_PROMPT + REFINER_JSON_FORMAT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            latency_ms = (time.monotonic() - start) * 1000
            self._record_success()

            # Track token usage & cost
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            cached_tokens = getattr(usage, "prompt_tokens_details", None)
            cached_tokens = (
                getattr(cached_tokens, "cached_tokens", 0) if cached_tokens else 0
            )
            output_tokens = usage.completion_tokens if usage else 0
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_requests += 1

            # Calculate cost in USD
            price_in, price_cached, price_out = self._pricing.get(
                self._model, (2.50, 0.25, 15.00)
            )
            non_cached = input_tokens - cached_tokens
            call_cost = (
                non_cached * price_in / 1_000_000
                + cached_tokens * price_cached / 1_000_000
                + output_tokens * price_out / 1_000_000
            )
            self.total_cost_usd += call_cost

            decision = self._parse_response(response)
            decision.latency_ms = latency_ms
            decision.input_tokens = input_tokens
            decision.output_tokens = output_tokens

            # Validate decision
            decision = self._validate_decision(decision, context)

            # Track action counts
            if decision.action == "ENTER":
                self.total_enter += 1
            elif decision.action == "WAIT":
                self.total_wait += 1
            elif decision.action == "ADJUST_ZONE":
                self.total_adjust += 1
            elif decision.action == "ABANDON":
                self.total_abandon += 1

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
            self._record_failure()
            logger.warning(
                "refiner_agent_failed",
                symbol=context.get("symbol", "?"),
                error=str(e),
                latency_ms=round(latency_ms, 1),
            )
            return RefinerDecision(
                action="WAIT",
                reasoning=f"Agent 2 API error: {e}",
                latency_ms=latency_ms,
                error=str(e),
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
        # Include Agent 1's suggested SL/TP if available
        if a1.get("suggested_sl"):
            a1_parts.append(f"Agent 1 Suggested SL: {a1['suggested_sl']:.6g}")
        if a1.get("suggested_tp"):
            a1_parts.append(f"Agent 1 Suggested TP: {a1['suggested_tp']:.6g}")
        if a1.get("entry_zone_high") and a1.get("entry_zone_low"):
            a1_parts.append(
                f"Agent 1 Entry Zone: {a1['entry_zone_low']:.6g} – {a1['entry_zone_high']:.6g}"
            )
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

        # ── Section 12: Sweep details ──
        sweep = ctx.get("sweep_info", {})
        if sweep:
            sweep_section = (
                f"Type: {sweep.get('sweep_type', '?')} | "
                f"Direction: {sweep.get('sweep_direction', '?')} | "
                f"Depth: {sweep.get('sweep_depth', 0):.6g} | "
                f"Level: {sweep.get('sweep_level', 0):.6g}"
            )
        else:
            sweep_section = "Not available"

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

        return f"""\
## Tactical Entry Evaluation

**Symbol:** {symbol}
**Direction:** {direction}
**Current Price:** {current_price:.6g}
{action_hint}

### Agent 1's Strategic Analysis
{a1_section}

### Pre-Computed Levels (use these for your SL/TP)
{pre_sltp_section}

### Your Previous Analysis (5 minutes ago)
{prev_section}

### Sweep Details
{sweep_section}

### Entry Zone
- **Source:** {zone_source}
- **Zone:** {zone_bottom:.6g} – {zone_top:.6g}
- **Midpoint:** {zone_mid:.6g}

### Price vs Zone
- **Position:** {price_vs_zone}
- **Distance from zone:** {distance_pct:.2f}%
- **Price change since queued:** {price_change:+.2f}%

### Pullback / Displacement Metrics
{pullback_section}

### Last 15 Five-Minute Candles (newest first)
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

### Leverage / Funding Profile
{leverage_section}

Run through the 3-step analysis: (1) price vs zone, (2) latest candle rejection signals, \
(3) structural confirmation. Reference specific numbers from the data in your reasoning."""

    def _parse_response(self, response) -> RefinerDecision:
        """Extract structured JSON from OpenAI response."""
        import json as _json

        choice = response.choices[0]
        message = choice.message

        # Primary path: JSON in message content
        if message.content:
            try:
                text = message.content.strip()
                # Strip markdown code fences if present
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                data = _json.loads(text)
                return self._data_to_decision(data)
            except (_json.JSONDecodeError, Exception) as e:
                logger.warning(
                    "refiner_agent_json_parse_failed",
                    error=str(e),
                    content_preview=message.content[:200],
                )

        logger.warning(
            "refiner_agent_no_response",
            finish_reason=choice.finish_reason,
            has_content=bool(message.content),
        )
        return RefinerDecision(
            action="WAIT",
            reasoning="No parseable response from Agent 2",
            error="no_response",
        )

    def _data_to_decision(self, data: dict) -> RefinerDecision:
        """Convert parsed dict to RefinerDecision."""
        action = data.get("action", "WAIT").upper()
        # Normalize action
        if action not in ("ENTER", "WAIT", "ADJUST_ZONE", "ABANDON"):
            action = "WAIT"

        return RefinerDecision(
            action=action,
            entry_price=float(data.get("entry_price", 0) or 0),
            adjusted_zone_high=float(data.get("adjusted_zone_high", 0) or 0),
            adjusted_zone_low=float(data.get("adjusted_zone_low", 0) or 0),
            stop_loss=float(data.get("stop_loss", 0) or 0),
            take_profit=float(data.get("take_profit", 0) or 0),
            position_size_modifier=float(data.get("position_size_modifier", 1.0) or 1.0),
            confidence=float(data.get("confidence", 0) or 0),
            urgency=str(data.get("urgency", "low") or "low"),
            reasoning=str(data.get("reasoning", "") or ""),
            invalidation_reason=str(data.get("invalidation_reason", "") or ""),
        )

    def _validate_decision(
        self, decision: RefinerDecision, context: dict[str, Any]
    ) -> RefinerDecision:
        """Validate and sanitize Agent 2's decision."""
        direction = context.get("direction", "")
        is_long = direction in ("bullish", "long")
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
            sl_distance = abs(decision.entry_price - decision.stop_loss)
            tp_distance = abs(decision.take_profit - decision.entry_price)
            if sl_distance > 0:
                rr = tp_distance / sl_distance
                if rr < 1.5:
                    logger.warning(
                        "refiner_agent_low_rr",
                        symbol=context.get("symbol", "?"),
                        rr=round(rr, 2),
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
                if sl_dist_pct > 0.15:
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

        elif decision.action == "ADJUST_ZONE":
            # Must have new zone boundaries
            if decision.adjusted_zone_high <= 0 or decision.adjusted_zone_low <= 0:
                logger.warning(
                    "refiner_agent_adjust_no_zone",
                    symbol=context.get("symbol", "?"),
                )
                decision.action = "WAIT"
                decision.reasoning = f"ADJUST_ZONE downgraded to WAIT: no zone provided. {decision.reasoning}"
                return decision

            if decision.adjusted_zone_high <= decision.adjusted_zone_low:
                logger.warning(
                    "refiner_agent_adjust_inverted_zone",
                    symbol=context.get("symbol", "?"),
                )
                decision.action = "WAIT"
                decision.reasoning = f"ADJUST_ZONE downgraded to WAIT: inverted zone. {decision.reasoning}"
                return decision

            # Zone width sanity check (max 10% of price)
            if current_price > 0:
                zone_width = (decision.adjusted_zone_high - decision.adjusted_zone_low) / current_price
                if zone_width > 0.10:
                    logger.warning(
                        "refiner_agent_adjust_zone_too_wide",
                        symbol=context.get("symbol", "?"),
                        width_pct=f"{zone_width:.2%}",
                    )
                    decision.action = "WAIT"
                    decision.reasoning = f"ADJUST_ZONE downgraded to WAIT: zone too wide ({zone_width:.1%}). {decision.reasoning}"
                    return decision

        elif decision.action == "ABANDON":
            if not decision.invalidation_reason:
                decision.invalidation_reason = decision.reasoning or "No reason provided"

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
            "total_adjust": self.total_adjust,
            "total_abandon": self.total_abandon,
            "enter_rate": round(
                self.total_enter / self.total_requests * 100, 1
            ) if self.total_requests > 0 else 0.0,
            "abandon_rate": round(
                self.total_abandon / self.total_requests * 100, 1
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
        if self._client:
            await self._client.close()
            self._client = None
