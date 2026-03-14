"""Agent 1 — AI-powered strategic entry agent using Gemini Interactions API.

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

Uses Gemini Interactions API for structured output with
resilience patterns: lazy client, exponential backoff, cost tracking.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from google import genai

from src.config import Settings
from src.exchange.models import SignalCandidate
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentDecision:
    """Result from the AI entry agent."""

    action: str = "SKIP"  # ENTER_NOW, WAIT_PULLBACK, SKIP
    direction: str = ""   # "LONG", "SHORT", or "" (for SKIP) — Agent 1 chooses
    confidence: float = 0.0  # 0-100
    reasoning: str = ""
    suggested_entry: float | None = None  # For WAIT_PULLBACK: target price
    entry_zone_high: float | None = None  # Approximate entry zone upper bound
    entry_zone_low: float | None = None   # Approximate entry zone lower bound
    must_reach_price: float | None = None  # WAIT_PULLBACK: price must reach here BEFORE pullback
    invalidation_level: float | None = None  # Price where the trade thesis is dead
    suggested_sl: float | None = None
    suggested_tp: float | None = None
    market_regime: str = ""  # trending/ranging/volatile/choppy
    risk_assessment: str = ""  # low/medium/high/extreme
    latency_ms: float = 0.0
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


SYSTEM_PROMPT = """\
You are an elite crypto futures strategist specializing in Smart Money Concepts (ICT methodology).
Your job is to make a ONE-TIME strategic decision on a trade setup.

The bot's formula system has detected market structure activity — a completed liquidity sweep \
(price wicked through a key level and closed back), possibly confirmed by displacement \
(large-bodied candle with volume). You receive RAW STRUCTURAL DATA with no directional bias.

Your job is to:
1. INDEPENDENTLY choose the DIRECTION — LONG or SHORT — by reading the structure yourself. \
   You receive raw data (sweep levels, structural levels, order blocks, FVGs, trends). \
   You must interpret the data and decide direction from scratch.
2. Choose the APPROACH — enter now or wait for a pullback to a specific zone.
3. Or SKIP — if the setup genuinely doesn't warrant any trade in either direction.

A second AI agent (Agent 2) will handle entry TIMING on the 5-minute chart. \
You do NOT need to specify what 5m patterns to look for — that is Agent 2's expertise. \
Your job is to define WHAT the trade is, WHERE the levels are, and WHY the thesis works.

## Your Decision Options

1. **ENTER_NOW** with direction **LONG** or **SHORT** — Conditions are met for entry. \
   Agent 2 will confirm timing on the next 5-minute candle and execute. \
   Use when: sweep + displacement confirmed, HTF aligned, good timing, favorable context. \
   Note: ENTER_NOW means "conditions are ready, confirm on 5m and execute" — NOT "execute blindly." \
   You MUST still provide an entry zone — the structural feature price is currently at (OB, FVG, or \
   sweep level). Agent 2 needs this to know WHERE to look for 5m confirmation.

2. **WAIT_PULLBACK** with direction **LONG** or **SHORT** — Wait for price to pull back to a \
   specific zone before entering.
   Use when: sweep confirmed but no pullback yet, or price ran too far from sweep level, \
   or you see a better entry zone on the structure.
   When choosing WAIT_PULLBACK, you MUST provide:
   - **entry_zone_high / entry_zone_low**: the price range where you expect the pullback to reach. \
     Use the **provided Fibonacci retracement levels** (especially the 61.8-78.6% OTE zone), \
     nearby order blocks, or FVGs to set this zone.
   - **must_reach_price**: the price you expect to be reached BEFORE the pullback happens. \
     For LONGS: the high price expects to reach before pulling back down to the zone. \
     For SHORTS: the low price expects to reach before bouncing up to the zone. \
     Set to 0 if the move has already happened.

3. **SKIP** — No trade in either direction. The structure doesn't support a trade.
   Use when: counter-trend to BOTH 4H AND Daily, BTC in freefall, manufactured sweep, \
   or genuinely no edge in either direction.

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
  "direction": "LONG" | "SHORT" | "",
  "confidence": <number 0-100>,
  "reasoning": "<5-8 sentence analysis with SPECIFIC price levels — see rules below>",
  "suggested_entry": <number or 0>,
  "entry_zone_high": <number or 0>,
  "entry_zone_low": <number or 0>,
  "must_reach_price": <number or 0>,
  "invalidation_level": <number or 0>,
  "suggested_sl": <number or 0>,
  "suggested_tp": <number or 0>,
  "market_regime": "trending" | "ranging" | "volatile" | "choppy",
  "risk_assessment": "low" | "medium" | "high" | "extreme"
}

Rules for direction:
- For ENTER_NOW and WAIT_PULLBACK: direction MUST be "LONG" or "SHORT" — you choose.
- For SKIP: direction MUST be "" (empty string).

Rules for numeric fields — ALWAYS provide your best price estimates for ALL actions:
- suggested_entry: Your ideal entry price. For WAIT_PULLBACK use the zone midpoint. For ENTER_NOW use the current price. For SKIP still provide where you WOULD enter if forced.
- entry_zone_high / entry_zone_low: ALWAYS provide, even for ENTER_NOW. \
  For ENTER_NOW: set to the structural feature price is currently at (the OB, FVG, or \
  retracement level). For WAIT_PULLBACK: the pullback target zone. For SKIP: your best estimate. \
  For LONGS: zone should be BELOW current price. For SHORTS: zone should be ABOVE current price.
- must_reach_price: For WAIT_PULLBACK ONLY — the price that must be reached BEFORE the pullback begins. \
  For LONGS: resistance/swing high above entry_zone_high. For SHORTS: support/swing low below entry_zone_low. \
  Set to 0 if the move already happened or for ENTER_NOW/SKIP.
- invalidation_level: The price where the trade thesis is dead. ALWAYS provide this. \
  For LONGS: price below which the bullish thesis fails (e.g. below the sweep low). \
  For SHORTS: price above which the bearish thesis fails (e.g. above the sweep high).
- suggested_sl: Agent 1's SL is PRIMARY — the system uses YOUR level (validated for max distance \
  and correct side). Only falls back to algorithmic SL if you return 0. ALWAYS provide a structural SL. \
  CRITICAL SL CONSTRAINT: must be within {max_sl_pct}% of entry price. \
  Always calculate and verify: abs(entry - SL) / entry × 100 <= {max_sl_pct}%.
- suggested_tp: Same as SL — YOUR level is primary. ALWAYS provide a structural TP.

CRITICAL — Reasoning must include SPECIFIC PRICES. \
Vague descriptions are USELESS. Your reasoning is reviewed by Agent 2 for context.

You MUST include ALL of the following in every reasoning:
  - The EXACT entry zone prices (e.g. "entry zone 0.01895–0.01910")
  - The EXACT stop-loss price with structural justification (e.g. "SL at 0.01870, below the swept low at 0.01875")
  - The EXACT take-profit target with structural justification (e.g. "TP at 0.01980, the 4H swing high")
  - The INVALIDATION LEVEL — the specific price where the thesis is dead (e.g. "thesis invalid if price closes below 0.01860")
  - WHERE key structural levels are with exact prices (e.g. "bearish OB at 0.01895–0.01905, bullish FVG at 0.01870–0.01880")
  - WHY this setup works or doesn't (e.g. "4H trend is bearish with BOS at 0.01950, sweep took the Asian high at 0.01922")

FORBIDDEN vague phrases — NEVER use these without a price:
  - "nearby structure" → instead say "the 1H OB at 0.01895"
  - "retest of swing highs" → instead say "retest of the swing high at 0.01920"
  - "good R:R" → instead say "R:R is 2.8:1 (SL 15 pips, TP 42 pips)"
  - "structure supports" → instead say "1H BOS bearish at 0.01950, CHoCH not yet seen"

Respond with ONLY the JSON object. No other text."""


# Gemini structured output schema for Agent 1
AGENT1_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["ENTER_NOW", "WAIT_PULLBACK", "SKIP"]},
        "direction": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "suggested_entry": {"type": "number"},
        "entry_zone_high": {"type": "number"},
        "entry_zone_low": {"type": "number"},
        "must_reach_price": {"type": "number"},
        "invalidation_level": {"type": "number"},
        "suggested_sl": {"type": "number"},
        "suggested_tp": {"type": "number"},
        "market_regime": {"type": "string", "enum": ["trending", "ranging", "volatile", "choppy"]},
        "risk_assessment": {"type": "string", "enum": ["low", "medium", "high", "extreme"]},
    },
    "required": [
        "action", "direction", "confidence", "reasoning",
        "suggested_entry", "entry_zone_high", "entry_zone_low",
        "must_reach_price", "invalidation_level",
        "suggested_sl", "suggested_tp",
        "market_regime", "risk_assessment",
    ],
}



class AgentEntryAnalyst:
    """Gemini-powered entry agent with resilient API handling.

    Replaces the binary approve/reject LLM gate with an intelligent
    decision-maker that chooses between ENTER_NOW, WAIT_PULLBACK, and SKIP.
    """

    def __init__(self, config: Settings) -> None:
        self._client: genai.AsyncClient | None = None
        self._api_key = config.agent_api_key
        self._model = config.agent_model
        self._timeout = config.agent_timeout_seconds
        self._fallback_approve = config.agent_fallback_approve
        self._min_confidence = config.agent_min_confidence
        self._max_sl_pct = config.max_sl_pct
        self._available = bool(config.agent_api_key)
        self._thinking_level = "high"  # Agent 1: strategic analysis needs deep thinking

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
            "gemini-3-pro-preview": (1.25, 0.125, 10.00),
            "gemini-3-flash-preview": (0.10, 0.01, 0.40),
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

    def _get_client(self) -> genai.AsyncClient:
        if self._client is None:
            self._client = genai.AsyncClient(api_key=self._api_key)
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
            system_prompt = (SYSTEM_PROMPT + JSON_FORMAT_INSTRUCTION).replace(
                "{max_sl_pct}", str(round(self._max_sl_pct * 100, 1))
            )
            interaction = await client.interactions.create(
                model=self._model,
                system_instruction=system_prompt,
                input=user_prompt,
                response_format=AGENT1_RESPONSE_SCHEMA,
                generation_config={
                    "thinking_level": self._thinking_level,
                    "temperature": 1.0,
                },
            )

            latency_ms = (time.monotonic() - start) * 1000
            self._record_success()

            # Track token usage & cost
            usage = interaction.usage
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_requests += 1

            # Calculate cost in USD (no cached token distinction for Gemini)
            price_in, _price_cached, price_out = self._pricing.get(
                self._model, (1.25, 0.125, 10.00)
            )
            call_cost = (
                input_tokens * price_in / 1_000_000
                + output_tokens * price_out / 1_000_000
            )
            self.total_cost_usd += call_cost

            decision = self._parse_response(interaction)
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
                    original_action="ENTER_NOW",
                    confidence=decision.confidence,
                    threshold=self._min_confidence,
                )

            if decision.action == "WAIT_PULLBACK" and decision.confidence < self._min_confidence:
                decision.action = "SKIP"
                decision.reasoning = (
                    f"WAIT_PULLBACK but confidence {decision.confidence:.0f} "
                    f"< threshold {self._min_confidence:.0f}: {decision.reasoning}"
                )
                logger.info(
                    "agent_low_confidence_override",
                    symbol=signal.symbol,
                    original_action="WAIT_PULLBACK",
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

        # Sweep details — directionally neutral
        sweep_info = "None"
        if signal.sweep_result:
            sr = signal.sweep_result
            level_type = sr.sweep_type or "unknown"
            swept_side = "high" if "high" in level_type else "low" if "low" in level_type else "level"
            sweep_info = (
                f"Swept {swept_side} at {sr.sweep_level:.6g} "
                f"(type: {level_type}, wick depth: {sr.sweep_depth:.4f})"
            )

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

        # Open positions
        open_count = context.get("open_position_count", 0)

        # Weekly cycle context (may not be set on all signals)
        weekly_info = ""
        weekly_factors = getattr(signal, "weekly_factors", None)
        if weekly_factors:
            if weekly_factors.get("is_monday_manipulation"):
                weekly_info = "WARNING: Monday manipulation window active"
            elif weekly_factors.get("is_midweek_reversal"):
                weekly_info = "Mid-week reversal window (Wed/Thu counter-trend bonus)"

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

        # Fibonacci retracement section — show BOTH directions
        fib = signal.fibonacci_levels
        if fib and fib.get("fib_50"):
            disp_low = fib['displacement_low']
            disp_high = fib['displacement_high']
            move = disp_high - disp_low

            # Both directions
            bull_618 = disp_high - move * 0.618
            bull_786 = disp_high - move * 0.786
            bear_618 = disp_low + move * 0.618
            bear_786 = disp_low + move * 0.786

            fib_section = (
                f"Recent displacement range: {disp_low:.6g} – {disp_high:.6g}\n"
                f"  If LONG (buying pullback from high):\n"
                f"    61.8%: {bull_618:.6g} | 78.6%: {bull_786:.6g}\n"
                f"    OTE zone: {bull_618:.6g} – {bull_786:.6g}\n"
                f"  If SHORT (selling bounce from low):\n"
                f"    61.8%: {bear_618:.6g} | 78.6%: {bear_786:.6g}\n"
                f"    OTE zone: {bear_618:.6g} – {bear_786:.6g}"
            )
        else:
            fib_section = "Not available (no displacement data)"

        # Funding rate
        funding_section = "N/A"
        lp = signal.leverage_profile
        if lp and hasattr(lp, 'funding_rate'):
            funding_section = f"{lp.funding_rate:.6f}"

        return f"""\
## Entry Decision Request

**Symbol:** {signal.symbol}
**Current Price:** {signal.entry_price:.6g}
**Open Positions:** {open_count}

### Sweep Analysis
{sweep_info}

### Fibonacci Retracement (from displacement move)
{fib_section}

### Key Structural Levels
{key_levels_str}

### Order Block: {ob_info}
### Fair Value Gap: {fvg_info}

### Market Context
- BTC: {btc_section}
- Sentiment: {sentiment:.1f} (range: -10 to +10, positive = bullish)
{f"- {weekly_info}" if weekly_info else ""}

### Funding Rate
{funding_section}

### Recent Headlines
{headlines_section}

### Bot Performance
{perf_section}

### Symbol Trade History
{symbol_history_section}

### Scanner Observations (evaluated relative to ONE possible direction — interpret independently)
{chr(10).join(f"  - {r}" for r in signal.reasons)}
{self._reassessment_section(context)}
Analyze this setup and respond with your JSON decision."""

    @staticmethod
    def _reassessment_section(context: dict) -> str:
        """Build prompt section for post-win reassessment signals."""
        if not context.get("post_win_reassessment"):
            return ""
        prev_dir = context.get("previous_direction", "unknown")
        prev_pnl = context.get("previous_pnl_pct", 0)
        return (
            "\n### Post-Win Reassessment\n"
            f"This symbol JUST closed a successful {prev_dir} trade "
            f"(all TP tiers hit, PnL: {prev_pnl:+.1f}%). "
            "Reassess the CURRENT market structure for a new entry in EITHER direction. "
            "The previous trade's thesis may no longer apply — analyze fresh. "
            "Be objective: if the setup is exhausted or unclear, SKIP.\n"
        )

    def _parse_response(self, interaction) -> AgentDecision:
        """Extract structured JSON from Gemini Interactions response."""
        import json as _json

        # Get the last text output from the interaction
        text = None
        for output in reversed(interaction.outputs):
            if getattr(output, "text", None):
                text = output.text.strip()
                break

        if text:
            try:
                # Strip markdown code fences if present
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                data = _json.loads(text)
                return self._data_to_decision(data)
            except (_json.JSONDecodeError, Exception) as e:
                logger.warning("agent_json_parse_failed", error=str(e),
                               content_preview=text[:200])

        logger.warning(
            "agent_no_response",
            status=getattr(interaction, "status", "unknown"),
            output_count=len(interaction.outputs) if interaction.outputs else 0,
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
        exp = data.get("must_reach_price")
        inv = data.get("invalidation_level")
        return AgentDecision(
            action=data.get("action", "SKIP"),
            direction=data.get("direction", "").upper().strip(),
            confidence=float(data.get("confidence", 0)),
            reasoning=str(data.get("reasoning", "")),
            suggested_entry=entry if entry and entry != 0 else None,
            entry_zone_high=zone_high if zone_high and zone_high != 0 else None,
            entry_zone_low=zone_low if zone_low and zone_low != 0 else None,
            must_reach_price=exp if exp and exp != 0 else None,
            invalidation_level=inv if inv and inv != 0 else None,
            suggested_sl=sl if sl and sl != 0 else None,
            suggested_tp=tp if tp and tp != 0 else None,
            market_regime=data.get("market_regime", ""),
            risk_assessment=data.get("risk_assessment", ""),
        )

    def _validate_suggestions(
        self, decision: AgentDecision, signal: SignalCandidate
    ) -> AgentDecision:
        """Validate and sanitize agent-suggested SL/TP/entry levels."""
        # Use Agent 1's chosen direction if available, else fall back to scanner's
        is_long = (decision.direction == "LONG") if decision.direction else (signal.direction == "bullish")
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
                if sl_distance_pct > self._max_sl_pct:
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

        # Validate must_reach_price (must-reach-before-pullback gate)
        if decision.must_reach_price is not None and decision.action == "WAIT_PULLBACK":
            mrp = decision.must_reach_price
            zh = decision.entry_zone_high
            zl = decision.entry_zone_low
            if is_long:
                # For longs: must_reach_price should be ABOVE the entry zone
                if zh is not None and mrp <= zh:
                    logger.warning(
                        "agent_must_reach_below_zone",
                        symbol=signal.symbol,
                        must_reach_price=mrp,
                        zone_high=zh,
                    )
                    decision.must_reach_price = None
            else:
                # For shorts: must_reach_price should be BELOW the entry zone
                if zl is not None and mrp >= zl:
                    logger.warning(
                        "agent_must_reach_above_zone",
                        symbol=signal.symbol,
                        must_reach_price=mrp,
                        zone_low=zl,
                    )
                    decision.must_reach_price = None

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
        self._client = None
