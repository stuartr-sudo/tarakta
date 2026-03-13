"""Agent 1 — AI-powered strategic entry agent using OpenAI GPT models.

This agent makes the actual entry DECISION. Given a candidate that passed initial
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

Uses OpenAI function calling (tool use) for structured output with
resilience patterns: lazy client, exponential backoff, cost tracking.
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
    entry_zone_high: float | None = None  # Approximate entry zone upper bound
    entry_zone_low: float | None = None   # Approximate entry zone lower bound
    expected_high: float | None = None    # WAIT_PULLBACK: price must reach here BEFORE pullback
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
   When choosing WAIT_PULLBACK, you MUST provide:
   - **entry_zone_high / entry_zone_low**: the price range where you expect the pullback to reach. \
     Use the **provided Fibonacci retracement levels** (especially the 61.8-78.6% OTE zone), \
     nearby order blocks, or FVGs to set this zone.
   - **expected_high**: the price you expect to be reached BEFORE the pullback happens. \
     This prevents premature entry on the initial move. For LONGS: the high price expects to \
     reach before pulling back down to the zone. For SHORTS: the low price expects to reach \
     before bouncing up to the zone. Set to 0 if the move has already happened and you're \
     waiting for a pullback that should start from the current price area.

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
- Pullback into the provided 61.8-78.6% Fibonacci OTE zone of the displacement move

## What Makes a BAD Entry (SKIP only these)
- Counter-trend to BOTH 4H AND Daily (fighting the tide on all timeframes)
- Monday first 8 hours (likely fake move / manipulation)
- BTC in freefall or parabolic — altcoin setups become unreliable
- Extreme funding rate in the same direction as trade (crowded)

## Important: Volatility is Opportunity
Choppy and volatile markets are NOT reasons to SKIP. Crypto is inherently volatile — \
that's where the opportunity lies. A sweep in a volatile market is STILL a valid signal. \
Only SKIP when there is a genuine structural reason (counter-trend on ALL timeframes, \
manufactured/fake sweep, or extreme macro risk). If a sweep is confirmed and HTF has \
ANY alignment (even just 4H), lean towards ENTER_NOW or WAIT_PULLBACK — not SKIP.

## Risk Assessment
Rate the overall risk as: low, medium, high, extreme
- low: strong confluence, HTF aligned, clean structure
- medium: decent setup with 1-2 missing components — still worth taking
- high: marginal setup but sweep is real — use WAIT_PULLBACK for better entry
- extreme: genuine structural red flags — this is the ONLY level that should lead to SKIP

## Market Regime
Classify the current market as: trending, ranging, volatile, choppy
This helps the bot calibrate its other parameters. Ranging and choppy regimes can still \
produce excellent sweep trades — they often have cleaner liquidity grabs.

## ALWAYS Suggest Prices
For ALL actions (including ENTER_NOW and SKIP), always provide your best estimate for \
suggested_entry, suggested_sl, and suggested_tp. This helps the trader see your analysis \
even when you recommend skipping. Only use 0 if you truly cannot estimate a level.

## Symbol History (per-symbol feedback loop)
If provided, you will see the last few closed trades for THIS specific symbol. Use this to:
- If the bot has a HIGH LOSS RATE on this symbol (e.g. 3+ losses in 5 trades): raise your confidence \
threshold — demand stronger confluence or a deeper pullback before entering.
- If recent exits are mostly "sl_hit": check if the SL placement pattern is wrong — maybe the \
structure requires a wider SL, or the direction is consistently wrong for this pair.
- If recent trades were consistently profitable: lean towards entering — the current setup \
conditions clearly work for this symbol.
- If the holding times are very short with losses: the timing may be off — prefer WAIT_PULLBACK \
over ENTER_NOW to get a better entry.
If no history is provided, treat the symbol as fresh with no prior data.

Be decisive. Lean towards taking trades with good sweeps rather than skipping them. \
The formula system already filters heavily — if a signal reaches you, it has merit.

## Glossary (use these terms precisely — do NOT abbreviate without defining)

- **SL** — Stop Loss: the price where the trade is exited at a loss to limit risk
- **TP** — Take Profit: the price where the trade is exited at a profit
- **R:R** — Risk-to-Reward Ratio: distance to TP divided by distance to SL (e.g. 2:1 means TP is 2x farther than SL)
- **OB** — Order Block: a supply/demand zone from a previous institutional candle body (bullish OB = last down-candle before rally, bearish OB = last up-candle before drop)
- **FVG** — Fair Value Gap: a three-candle imbalance where candle 1's wick doesn't overlap candle 3's wick, creating a gap that price tends to revisit
- **BOS** — Break of Structure: price breaks a previous swing high (bullish) or swing low (bearish), confirming trend continuation
- **CHoCH** — Change of Character: first break of structure against the prevailing trend, signaling a potential reversal
- **OTE** — Optimal Trade Entry: the 61.8%–78.6% Fibonacci retracement zone of a displacement move — the highest-probability re-entry area
- **HTF** — Higher Timeframe (4H, Daily, Weekly): used for trend/bias direction
- **LTF** — Lower Timeframe (5m, 15m): used for precise entry timing
- **RVOL** — Relative Volume: current volume divided by average volume over the lookback period (>1.5 = elevated, >2.5 = institutional)
- **Displacement** — A large-bodied candle with above-average volume that shows aggressive institutional order flow
- **Liquidity Sweep** — Price wicks through a key level (swing high/low, equal highs/lows) to trigger resting stop-loss orders, then reverses
- **Kill Zone** — ICT-defined high-probability trading sessions: London Open (02:00–05:00 UTC), NY Open (12:00–15:00 UTC), NY PM (19:00–21:00 UTC)
- **Judas Swing** — A fake initial move at session open designed to trap traders before price reverses in the true direction"""


JSON_FORMAT_INSTRUCTION = """

## RESPONSE FORMAT — CRITICAL

You MUST respond with a single JSON object and NOTHING else. No markdown, no explanation, no text before or after.

The JSON object must have exactly these keys:
{
  "action": "ENTER_NOW" | "WAIT_PULLBACK" | "SKIP",
  "confidence": <number 0-100>,
  "reasoning": "<3-5 sentence analysis with SPECIFIC price levels — see rules below>",
  "suggested_entry": <number or 0>,
  "entry_zone_high": <number or 0>,
  "entry_zone_low": <number or 0>,
  "expected_high": <number or 0>,
  "suggested_sl": <number or 0>,
  "suggested_tp": <number or 0>,
  "market_regime": "trending" | "ranging" | "volatile" | "choppy",
  "risk_assessment": "low" | "medium" | "high" | "extreme"
}

Rules for numeric fields — ALWAYS provide your best price estimates for ALL actions:
- suggested_entry: Your ideal entry price. For WAIT_PULLBACK use the zone midpoint. For ENTER_NOW use the current price. For SKIP still provide where you WOULD enter if forced.
- entry_zone_high: Upper bound of the ideal entry zone. Always provide this based on structure (Fibonacci, order blocks, FVGs).
- entry_zone_low: Lower bound of the ideal entry zone. Always provide this.
  For a LONG: the zone should be BELOW current price (pullback down to buy).
  For a SHORT: the zone should be ABOVE current price (pullback up to sell).
  Use the provided Fibonacci retracement levels, order blocks, or fair value gaps to set this zone.
- expected_high: For WAIT_PULLBACK ONLY — the price that must be reached BEFORE the pullback begins. \
  This is CRITICAL to prevent premature entries. For LONGS: set this to the resistance level or \
  swing high you expect price to reach before pulling back (must be ABOVE entry_zone_high). \
  For SHORTS: set this to the support level or swing low you expect price to reach before \
  bouncing up (must be BELOW entry_zone_low). Set to 0 if the move already happened and \
  pullback should start from current price area, or for ENTER_NOW/SKIP.
- suggested_sl: ALWAYS set a stop-loss price based on structure (below sweep level for longs, above for shorts). Never use 0.
- suggested_tp: ALWAYS set a take-profit price based on structure. Never use 0.

CRITICAL — Reasoning must include SPECIFIC PRICES AND CLEAR INSTRUCTIONS. A second AI agent \
(Agent 2) reads your reasoning to make entry timing decisions on a 5-minute chart. \
Vague descriptions are USELESS to it. Your reasoning is the ONLY guidance Agent 2 has.

You MUST include ALL of the following in every reasoning:
  - The EXACT entry zone prices (e.g. "entry zone 0.01895–0.01910")
  - The EXACT stop-loss price with structural justification (e.g. "SL at 0.01925, placed 0.1% above the swept high at 0.01920")
  - The EXACT take-profit target with structural justification (e.g. "TP at 0.01820, the 4H swing low / buy-side liquidity pool")
  - WHAT specifically Agent 2 should watch for on the 5m chart to confirm entry (e.g. "look for a bearish engulfing candle or pin bar closing below 0.01900 with RVOL > 1.5")
  - WHERE key structural levels are with exact prices (e.g. "bearish OB at 0.01895–0.01905, FVG at 0.01880–0.01870")
  - WHY this setup works or doesn't (e.g. "4H trend is bearish with BOS at 0.01950, sweep took the Asian high at 0.01922")

FORBIDDEN vague phrases — NEVER use these without a price:
  - "nearby structure" → instead say "the 1H OB at 0.01895"
  - "retest of swing highs" → instead say "retest of the swing high at 0.01920"
  - "wait for confirmation" → instead say "wait for a 5m candle closing below 0.01900 with wick rejection"
  - "good R:R" → instead say "R:R is 2.8:1 (SL 15 pips, TP 42 pips)"
  - "structure supports" → instead say "1H BOS bearish at 0.01950, CHoCH not yet seen"
  - "look for rejection" → instead say "look for lower wick > body size at the OB 0.01895–0.01905"

Respond with ONLY the JSON object. No other text."""


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
                    "type": "number",
                    "description": (
                        "For WAIT_PULLBACK: the price level to wait for (zone midpoint). "
                        "For ENTER_NOW or SKIP: set to 0."
                    ),
                },
                "entry_zone_high": {
                    "type": "number",
                    "description": (
                        "For WAIT_PULLBACK: upper bound of ideal entry zone. "
                        "For ENTER_NOW or SKIP: set to 0."
                    ),
                },
                "entry_zone_low": {
                    "type": "number",
                    "description": (
                        "For WAIT_PULLBACK: lower bound of ideal entry zone. "
                        "For ENTER_NOW or SKIP: set to 0."
                    ),
                },
                "expected_high": {
                    "type": "number",
                    "description": (
                        "For WAIT_PULLBACK: price that must be reached BEFORE pullback begins. "
                        "For LONGS: resistance/swing high above zone (must be > entry_zone_high). "
                        "For SHORTS: support/swing low below zone (must be < entry_zone_low). "
                        "Set to 0 if the move already happened or for ENTER_NOW/SKIP."
                    ),
                },
                "suggested_sl": {
                    "type": "number",
                    "description": (
                        "Alternative stop-loss price if you see a better structural level. "
                        "Set to 0 to keep the formula-calculated SL."
                    ),
                },
                "suggested_tp": {
                    "type": "number",
                    "description": (
                        "Alternative take-profit if you see a better target. "
                        "Set to 0 to keep the formula-calculated TP."
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
                "suggested_entry", "entry_zone_high", "entry_zone_low",
                "expected_high", "suggested_sl", "suggested_tp",
                "market_regime", "risk_assessment",
            ],
            "additionalProperties": False,
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

        # Backoff state (exponential backoff on API failures)
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
        logger.info("agent_model_switched", old_model=old, new_model=model)
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
                max_completion_tokens=7000,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT + JSON_FORMAT_INSTRUCTION},
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
                cost_usd=round(call_cost, 6),
                total_cost_usd=round(self.total_cost_usd, 4),
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

        # Symbol trade history (feedback loop)
        symbol_history = context.get("symbol_history", [])
        if symbol_history:
            history_lines = []
            wins = sum(1 for t in symbol_history if (t.get("pnl_usd") or 0) > 0)
            losses = len(symbol_history) - wins
            history_lines.append(f"  Last {len(symbol_history)} trades: {wins}W / {losses}L")
            for t in symbol_history:
                pnl = t.get("pnl_usd", 0) or 0
                pnl_pct = t.get("pnl_percent", 0) or 0
                exit_reason = t.get("exit_reason", "unknown")
                direction_h = t.get("direction", "?")
                conf = t.get("confluence_score")
                entry_p = t.get("entry_price", 0) or 0
                exit_p = t.get("exit_price", 0) or 0
                # Calculate hold time if available
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
                w_l = "WIN" if pnl > 0 else "LOSS"
                conf_str = f", score={conf:.0f}" if conf else ""
                history_lines.append(
                    f"  - {direction_h} {w_l}: ${pnl:+.2f} ({pnl_pct:+.1f}%), "
                    f"exit={exit_reason}, entry={entry_p:.6g}→{exit_p:.6g}{hold_str}{conf_str}"
                )
            symbol_history_section = "\n".join(history_lines)
        else:
            symbol_history_section = "  No prior trades for this symbol"

        # Fibonacci retracement section
        fib = signal.fibonacci_levels
        if fib and fib.get("fib_50"):
            fib_section = (
                f"Displacement: {fib['displacement_low']:.6g} → {fib['displacement_high']:.6g}\n"
                f"  50.0% retracement: {fib['fib_50']:.6g}\n"
                f"  61.8% retracement: {fib['fib_618']:.6g}\n"
                f"  78.6% retracement: {fib['fib_786']:.6g}\n"
                f"  OTE Zone (optimal entry): {fib['fib_618']:.6g} – {fib['fib_786']:.6g}"
            )
        else:
            fib_section = "Not available (no displacement data)"

        return f"""\
## Entry Decision Request

**Symbol:** {signal.symbol}
**Direction:** {direction}
**Current Price:** {signal.entry_price:.6g}
**Formula Score:** {signal.score:.1f}/100 (adjusted: {adjusted:.1f}, threshold: {threshold:.1f})
**Open Positions:** {open_count}

### Sweep Analysis
{sweep_info}

### Fibonacci Retracement (from displacement move)
{fib_section}

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

### Symbol Trade History
{symbol_history_section}

Analyze this setup and respond with your JSON decision."""

    def _parse_response(self, response) -> AgentDecision:
        """Extract structured JSON from OpenAI response."""
        import json as _json

        choice = response.choices[0]
        message = choice.message

        # Primary path: JSON in message content (response_format=json_schema)
        if message.content:
            try:
                text = message.content.strip()
                # Strip markdown code fences if present
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                data = _json.loads(text)
                return self._data_to_decision(data)
            except (_json.JSONDecodeError, Exception) as e:
                logger.warning("agent_json_parse_failed", error=str(e),
                               content_preview=message.content[:200])

        # Fallback: tool calls (kept for compatibility)
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "record_entry_decision":
                    data = _json.loads(tool_call.function.arguments)
                    return self._data_to_decision(data)

        logger.warning(
            "agent_no_response",
            finish_reason=choice.finish_reason,
            has_content=bool(message.content),
            content_preview=(message.content or "")[:200],
        )
        return AgentDecision(
            action="SKIP",
            reasoning="No parseable response from agent",
            error="no_response",
        )

    def _data_to_decision(self, data: dict) -> AgentDecision:
        """Convert parsed dict to AgentDecision, treating 0 as null for prices."""
        entry = data.get("suggested_entry")
        sl = data.get("suggested_sl")
        tp = data.get("suggested_tp")
        zone_high = data.get("entry_zone_high")
        zone_low = data.get("entry_zone_low")
        exp_high = data.get("expected_high")
        return AgentDecision(
            action=data.get("action", "SKIP"),
            confidence=float(data.get("confidence", 0)),
            reasoning=str(data.get("reasoning", "")),
            suggested_entry=entry if entry and entry != 0 else None,
            entry_zone_high=zone_high if zone_high and zone_high != 0 else None,
            entry_zone_low=zone_low if zone_low and zone_low != 0 else None,
            expected_high=exp_high if exp_high and exp_high != 0 else None,
            suggested_sl=sl if sl and sl != 0 else None,
            suggested_tp=tp if tp and tp != 0 else None,
            market_regime=data.get("market_regime", ""),
            risk_assessment=data.get("risk_assessment", ""),
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

        # Validate entry zone
        zh = decision.entry_zone_high
        zl = decision.entry_zone_low
        if zh is not None and zl is not None and decision.action == "WAIT_PULLBACK":
            if zh <= zl:
                logger.warning("agent_zone_inverted", symbol=signal.symbol,
                               zone_high=zh, zone_low=zl)
                decision.entry_zone_high = None
                decision.entry_zone_low = None
            elif entry > 0 and (zh - zl) / entry > 0.10:
                logger.warning("agent_zone_too_wide", symbol=signal.symbol,
                               width_pct=f"{(zh - zl) / entry:.2%}")
                decision.entry_zone_high = None
                decision.entry_zone_low = None
            elif is_long and zl >= entry:
                logger.warning("agent_zone_above_entry_for_long", symbol=signal.symbol)
                decision.entry_zone_high = None
                decision.entry_zone_low = None
            elif not is_long and zh <= entry:
                logger.warning("agent_zone_below_entry_for_short", symbol=signal.symbol)
                decision.entry_zone_high = None
                decision.entry_zone_low = None

        # Validate expected_high (must-reach-before-pullback gate)
        if decision.expected_high is not None and decision.action == "WAIT_PULLBACK":
            eh = decision.expected_high
            zh = decision.entry_zone_high
            zl = decision.entry_zone_low
            if is_long:
                # For longs: expected_high should be ABOVE the entry zone
                if zh is not None and eh <= zh:
                    logger.warning(
                        "agent_expected_high_below_zone",
                        symbol=signal.symbol,
                        expected_high=eh,
                        zone_high=zh,
                    )
                    decision.expected_high = None
            else:
                # For shorts: expected_high (really expected_low) should be BELOW the entry zone
                if zl is not None and eh >= zl:
                    logger.warning(
                        "agent_expected_high_above_zone",
                        symbol=signal.symbol,
                        expected_high=eh,
                        zone_low=zl,
                    )
                    decision.expected_high = None

        # Derive suggested_entry from zone midpoint if zone is valid but entry is missing
        if (decision.entry_zone_high is not None and decision.entry_zone_low is not None
                and decision.suggested_entry is None and decision.action == "WAIT_PULLBACK"):
            decision.suggested_entry = (decision.entry_zone_high + decision.entry_zone_low) / 2

        # Fallback: derive zone from Fibonacci OTE if agent returned zone=0
        # Uses pre-calculated Fib levels (61.8%-78.6%) instead of generic ±ATR
        if (decision.entry_zone_high is None
                and decision.entry_zone_low is None
                and decision.action == "WAIT_PULLBACK"
                and signal.fibonacci_levels
                and signal.fibonacci_levels.get("fib_618")):
            fib = signal.fibonacci_levels
            decision.entry_zone_high = max(fib["fib_618"], fib["fib_786"])
            decision.entry_zone_low = min(fib["fib_618"], fib["fib_786"])
            if decision.suggested_entry is None:
                decision.suggested_entry = (decision.entry_zone_high + decision.entry_zone_low) / 2
            logger.info(
                "agent_zone_fib_fallback",
                symbol=signal.symbol,
                zone=f"{decision.entry_zone_low:.6f}-{decision.entry_zone_high:.6f}",
                fib_618=fib["fib_618"],
                fib_786=fib["fib_786"],
            )

        return decision

    def get_usage_stats(self) -> dict[str, Any]:
        """Return cumulative usage stats for monitoring/dashboard."""
        avg_cost = (
            round(self.total_cost_usd / self.total_requests, 6)
            if self.total_requests > 0
            else 0.0
        )
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
