"""AI-powered position manager (Agent 3 — Position Monitor).

Agent 3 runs every 5 minutes per open position and provides AI-powered
recommendations for position management. It SUPPLEMENTS (not replaces)
the algorithmic SL/TP/trailing logic in PositionMonitor.

Decisions:
  HOLD         — Do nothing, position is fine (default)
  TIGHTEN_SL   — Move SL in profitable direction only (cannot widen)
  CLOSE_PARTIAL — Close a percentage of the position
  CLOSE_FULL   — Exit the entire position

Key constraint: Agent 3 cannot widen the SL or override hard SL/TP levels.
If the API call fails, the algorithmic logic has already run — no action taken.

Uses the same infrastructure as Agent 1 and Agent 2: lazy Gemini AsyncClient,
exponential backoff, cost tracking, JSON response parsing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.config import Settings
from src.strategy.llm_client import (
    LLMResult, generate_json, is_openai_model, MODEL_PRICING,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PositionDecision:
    """Result from Agent 3 — position management decision."""

    action: str = "HOLD"  # HOLD, TIGHTEN_SL, CLOSE_PARTIAL, CLOSE_FULL
    suggested_sl: float = 0.0  # New SL for TIGHTEN_SL (must be tighter than current)
    partial_close_pct: float = 0.0  # 0.0-1.0 for CLOSE_PARTIAL
    reasoning: str = ""
    confidence: float = 0.0  # 0-100
    latency_ms: float = 0.0
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


POSITION_SYSTEM_PROMPT = """\
You are an expert position manager for a crypto futures trading bot. Your job is to monitor \
open positions and recommend management actions based on current market conditions.

## Context
The algorithmic system already handles:
- Hard stop-loss execution (price hits SL → immediate exit)
- Progressive take-profit tiers (0.70R / 0.95R / 1.5R partial closes)
- ATR-based trailing stop activation (after 2R profit)
- Breakeven stop (after 0.5R profit)
- Stale position close (held too long without progress)

Your role is to provide SUPPLEMENTARY intelligence that algorithms miss:
- Detecting deteriorating market structure before SL is hit
- Identifying momentum exhaustion or reversal signals
- Recognizing when to lock in profit early due to macro changes
- Spotting opportunities to tighten SL based on new structure

## Your Decision Options

### 1. HOLD — No action needed (DEFAULT)
The position is performing within expectations. The algorithmic system is handling it correctly.
**Use this in most cases.** Only deviate when you see clear evidence.

### 2. TIGHTEN_SL — Move stop-loss in profitable direction
You see new structure that allows a tighter SL without premature exit.

**Constraints:**
- For LONGS: new SL MUST be HIGHER than current SL (tighter = more protective)
- For SHORTS: new SL MUST be LOWER than current SL
- Never widen the SL — this is a hard rule that cannot be overridden

**When to use:**
- New swing low formed above current SL (longs)
- New swing high formed below current SL (shorts)
- Market structure deteriorating — tighten to protect remaining profit

**Example:** LONG entered at 0.500, current SL at 0.480, new swing low formed at 0.492.
TIGHTEN_SL to 0.490 (just below new swing low, still tighter/higher than current 0.480).

### 3. CLOSE_PARTIAL — Close a percentage of the position
Take some profit off the table while keeping exposure.

**Percentage guidance (system caps at 50% per action):**
- 25%: Conservative — small amount off the table
- 50%: Maximum allowed per single action — significant profit-taking
Note: the system will cap any value above 50% down to 50%.

**When to use:**
- Approaching strong resistance/support with momentum fading
- BTC making a sharp counter-move
- Funding rate shifting against the position
- Position has significant unrealized profit but TP tiers haven't triggered

### 4. CLOSE_FULL — Exit the entire position
The thesis is invalidated or risk/reward no longer justifies holding.

**When to use:**
- Clear break of structure against position direction
- BTC in freefall/parabolic against your direction
- Fundamental catalyst changed (major news, exchange issues)
- Position is marginally profitable but structure has broken down

## Important Rules
- **HOLD is the default.** You should HOLD in 80%+ of evaluations. Only recommend action when \
  you see clear, specific, unambiguous evidence. A minor dip or single red candle is NOT evidence.
- **Never widen SL.** You can only move it in the profitable direction.
- **Never override hard SL/TP.** The algorithmic system handles execution.
- **Be extremely conservative with CLOSE_FULL and CLOSE_PARTIAL.** These are irreversible. \
  The algorithmic SL/TP system already handles exits — your job is to catch rare edge cases, \
  not second-guess every price movement. If the trade hasn't hit its SL, the thesis may still be valid.
- **TIGHTEN_SL is almost always better than CLOSE.** If you're worried about a position, \
  tighten the stop-loss first rather than closing. This protects profit while letting the \
  trade play out. Only use CLOSE when structure is clearly and definitively broken.
- **TP tier progress matters.** If 2+ TP tiers are completed and less than 50% of the position \
  remains, be more willing to CLOSE_FULL — the trade has already achieved its primary objective. \
  Protecting remaining profit is more important than squeezing the last tier.
- **Don't panic on short-term noise.** A 5-minute red candle does not invalidate a trade thesis. \
  Look at the 1H structure and trend before recommending any close action.
- **Confidence must match the action severity.** TIGHTEN_SL: 50+. CLOSE_PARTIAL: 65+. \
  CLOSE_FULL: 75+. Below these thresholds, default to HOLD.

## Glossary
- **SL** — Stop Loss: price where the trade exits at a loss
- **TP** — Take Profit: price where the trade exits at a profit
- **R:R** — Risk-to-Reward: profit / risk distance ratio
- **ATR** — Average True Range: volatility measure over N candles
- **Unrealized R** — Current profit expressed as multiples of risk (1R = profit equals risk amount)
- **Funding Rate** — Periodic payment between long/short holders (positive = longs pay shorts)
- **BOS** — Break of Structure: swing high/low broken against position direction = warning
- **OB** — Order Block: institutional supply/demand zone
- **Trailing Stop** — SL that follows price, activated after sufficient profit"""


POSITION_JSON_FORMAT = """

## RESPONSE FORMAT — CRITICAL

You MUST respond with a single JSON object and NOTHING else. No markdown, no explanation.

{
  "action": "HOLD" | "TIGHTEN_SL" | "CLOSE_PARTIAL" | "CLOSE_FULL",
  "suggested_sl": <number or 0>,
  "partial_close_pct": <number 0-1 or 0>,
  "reasoning": "<2-3 sentence analysis with specific prices>",
  "confidence": <number 0-100>
}

Rules:
- For HOLD: suggested_sl=0, partial_close_pct=0
- For TIGHTEN_SL: provide the new SL price in suggested_sl
- For CLOSE_PARTIAL: provide the percentage as a decimal (0.25 = 25%) in partial_close_pct
- For CLOSE_FULL: suggested_sl=0, partial_close_pct=0
- confidence: how sure you are about your recommendation (50+ for action, below 50 = HOLD)

Respond with ONLY the JSON object. No other text."""


# Gemini structured output schema for Agent 3
POSITION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["HOLD", "TIGHTEN_SL", "CLOSE_PARTIAL", "CLOSE_FULL"]},
        "suggested_sl": {"type": "number"},
        "partial_close_pct": {"type": "number"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["action", "suggested_sl", "partial_close_pct", "reasoning", "confidence"],
}


class PositionManagerAgent:
    """Agent 3 — AI-powered position management.

    Runs every 5 minutes per open position, providing supplementary
    recommendations to the algorithmic position monitor.
    """

    def __init__(self, config: Settings) -> None:
        self._model = getattr(config, "position_agent_model", "gemini-3-flash-preview")
        self._timeout = config.agent_timeout_seconds
        self._thinking_level = "minimal"  # Agent 3: simple SL checks, fast

        # Store both API keys so runtime model switching works across providers
        self._gemini_api_key = config.agent_api_key
        self._openai_api_key = config.openai_api_key
        self._api_key = self._openai_api_key if is_openai_model(self._model) else self._gemini_api_key
        self._available = bool(self._api_key) and getattr(
            config, "position_agent_enabled", False
        )

        # Backoff state
        self._fail_count = 0
        self._backoff_until = 0.0

        # Cumulative stats
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_hold = 0
        self.total_tighten = 0
        self.total_partial = 0
        self.total_close = 0
        self.total_errors = 0
        self.total_cost_usd = 0.0

        # Pricing per 1M tokens (input, cached_input, output)
        self._pricing = MODEL_PRICING

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
            "position_agent_backoff",
            fail_count=self._fail_count,
            backoff_seconds=backoff,
        )

    def _record_success(self) -> None:
        self._fail_count = 0
        self._backoff_until = 0.0

    async def evaluate_position(self, context: dict[str, Any]) -> PositionDecision:
        """Evaluate an open position and return a management recommendation."""
        if not self._should_try():
            return PositionDecision(action="HOLD", reasoning="Agent 3 unavailable or in backoff")

        prompt = self._build_prompt(context)
        t0 = time.time()

        try:
            result = await generate_json(
                model=self._model,
                api_key=self._api_key,
                system_prompt=POSITION_SYSTEM_PROMPT + POSITION_JSON_FORMAT,
                user_prompt=prompt,
                json_schema=POSITION_RESPONSE_SCHEMA,
                thinking_level=self._thinking_level,
                temperature=1.0,
                timeout=self._timeout,
            )

            latency_ms = (time.time() - t0) * 1000
            self.total_requests += 1

            # Track tokens + cost
            in_tok = result.input_tokens
            out_tok = result.output_tokens
            self.total_input_tokens += in_tok
            self.total_output_tokens += out_tok

            pricing = self._pricing.get(self._model, (0.10, 0.01, 0.40))
            cost = (in_tok * pricing[0] + out_tok * pricing[2]) / 1_000_000
            self.total_cost_usd += cost

            decision = self._parse_response(result.text)
            decision.latency_ms = latency_ms
            decision.input_tokens = in_tok
            decision.output_tokens = out_tok

            # Validate TIGHTEN_SL actually tightens
            if decision.action == "TIGHTEN_SL" and decision.suggested_sl > 0:
                current_sl = context.get("stop_loss", 0)
                ctx_direction = context.get("direction", "unknown")
                if ctx_direction.lower() in ("long", "bullish"):
                    if decision.suggested_sl <= current_sl:
                        logger.warning("position_agent_sl_not_tighter",
                            suggested=decision.suggested_sl, current=current_sl, direction=ctx_direction)
                        decision.action = "HOLD"
                        decision.reasoning = f"TIGHTEN_SL rejected: {decision.suggested_sl} not above current {current_sl}. {decision.reasoning}"
                elif ctx_direction.lower() in ("short", "bearish"):
                    if decision.suggested_sl >= current_sl:
                        logger.warning("position_agent_sl_not_tighter",
                            suggested=decision.suggested_sl, current=current_sl, direction=ctx_direction)
                        decision.action = "HOLD"
                        decision.reasoning = f"TIGHTEN_SL rejected: {decision.suggested_sl} not below current {current_sl}. {decision.reasoning}"

            # Update action stats
            action = decision.action.upper()
            if action == "TIGHTEN_SL":
                self.total_tighten += 1
            elif action == "CLOSE_PARTIAL":
                self.total_partial += 1
            elif action == "CLOSE_FULL":
                self.total_close += 1
            else:
                self.total_hold += 1

            self._record_success()
            return decision

        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            self._record_failure()
            logger.warning(
                "position_agent_error",
                error=str(e),
                latency_ms=round(latency_ms, 1),
            )
            return PositionDecision(
                action="HOLD",
                reasoning=f"Agent 3 API error: {e}",
                latency_ms=latency_ms,
                error=str(e),
            )

    def _build_prompt(self, ctx: dict[str, Any]) -> str:
        """Build the user prompt with full market context for position management."""
        symbol = ctx.get("symbol", "?")
        direction = ctx.get("direction", "unknown")
        entry_price = ctx.get("entry_price", 0)
        current_price = ctx.get("current_price", 0)
        stop_loss = ctx.get("stop_loss", 0)
        take_profit = ctx.get("take_profit", 0)

        # PnL metrics
        unrealized_pnl = ctx.get("unrealized_pnl_usd", 0)
        unrealized_pnl_pct = ctx.get("unrealized_pnl_pct", 0)
        unrealized_rr = ctx.get("unrealized_rr", 0)

        # Time held
        held_minutes = ctx.get("held_minutes", 0)
        held_str = f"{held_minutes:.0f} minutes"
        if held_minutes >= 60:
            held_str = f"{held_minutes / 60:.1f} hours"

        # ATR
        atr = ctx.get("atr", 0)
        atr_pct = (atr / current_price * 100) if current_price > 0 else 0

        # TP tier progress
        current_tier = ctx.get("current_tier", 0)
        original_qty = ctx.get("original_quantity", 0)
        remaining_qty = ctx.get("remaining_quantity", 0)
        qty_pct = (remaining_qty / original_qty * 100) if original_qty > 0 else 100

        # Confluence score from entry
        confluence = ctx.get("confluence_score", 0)

        # Agent 1's original thesis
        agent1_reasoning = ctx.get("agent1_reasoning", "N/A")

        # BTC context
        btc_trend = ctx.get("btc_trend", "unknown")
        btc_change = ctx.get("btc_1h_change", 0)

        # Funding rate
        funding_rate = ctx.get("funding_rate")
        funding_section = f"{funding_rate:.6f}" if funding_rate is not None else "N/A"

        # --- Build candle tables ---
        candles_5m_section = self._format_candle_table(ctx.get("candles_5m", []), "5m")
        candles_1h_section = self._format_candle_table(ctx.get("candles_1h", []), "1h")

        # --- Market structure ---
        ms_section = self._format_market_structure(ctx.get("market_structure", {}))

        return f"""\
## Position Management Check

**Symbol:** {symbol}
**Direction:** {direction}
**Entry Price:** {entry_price:.6g}
**Current Price:** {current_price:.6g}
**Stop Loss:** {stop_loss:.6g}
**Take Profit:** {take_profit:.6g}

### PnL Status
- Unrealized PnL: ${unrealized_pnl:+.2f} ({unrealized_pnl_pct:+.2f}%)
- Unrealized R: {unrealized_rr:+.2f}R
- Time Held: {held_str}

### Position Size
- TP Tier: {current_tier} / 3 tiers completed
- Remaining: {qty_pct:.0f}% of original ({remaining_qty:.6g} / {original_qty:.6g})

### Volatility
- ATR (1H): {atr:.6g} ({atr_pct:.2f}% of price)

### Entry Context
- Entry Confidence: {confluence:.0f}/100
- Agent 1 Thesis: {agent1_reasoning[:300]}

### Market Context
- BTC Trend: {btc_trend}
- BTC 1H Change: {btc_change:+.2f}%
- Funding Rate: {funding_section}

{ms_section}

{candles_5m_section}

{candles_1h_section}

Evaluate this position and respond with your JSON decision. Default to HOLD unless you see clear evidence for action."""

    @staticmethod
    def _format_candle_table(candles: list[dict], timeframe: str) -> str:
        """Format candle data as a readable table for the prompt."""
        if not candles:
            return f"### Recent {timeframe} Candles\nNo candle data available."

        lines = [f"### Recent {timeframe} Candles"]
        lines.append("| Time | Open | High | Low | Close | Volume |")
        lines.append("|------|------|------|-----|-------|--------|")
        for c in candles:
            t = c.get("time", "?")
            # Shorten timestamp for readability
            if len(t) > 16:
                t = t[5:16]  # "MM-DD HH:MM"
            lines.append(
                f"| {t} | {c['open']:.6g} | {c['high']:.6g} | {c['low']:.6g} | {c['close']:.6g} | {c['volume']:.0f} |"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_market_structure(ms: dict) -> str:
        """Format market structure data for the prompt."""
        if not ms:
            return "### HTF Market Structure\nNo market structure data available."

        lines = ["### HTF Market Structure"]
        for tf in ("1h", "4h", "1d"):
            data = ms.get(tf)
            if not data:
                continue
            trend = data.get("trend", "unknown")
            strength = data.get("strength", 0)
            last_bos = data.get("last_bos", "none")
            last_choch = data.get("last_choch", "none")
            lines.append(
                f"- **{tf.upper()}:** Trend={trend}, Strength={strength:.0f}%, "
                f"Last BOS={last_bos}, Last CHoCH={last_choch}"
            )

        if len(lines) == 1:
            lines.append("No market structure data available.")
        return "\n".join(lines)

    def _parse_response(self, text: str) -> PositionDecision:
        """Extract structured JSON from LLM response text."""
        import json as _json

        if text:
            text = text.strip()

        if text:
            try:
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                data = _json.loads(text)
                action = str(data.get("action", "HOLD")).upper()
                if action not in ("HOLD", "TIGHTEN_SL", "CLOSE_PARTIAL", "CLOSE_FULL"):
                    action = "HOLD"
                return PositionDecision(
                    action=action,
                    suggested_sl=float(data.get("suggested_sl", 0)),
                    partial_close_pct=float(data.get("partial_close_pct", 0)),
                    reasoning=str(data.get("reasoning", "")),
                    confidence=float(data.get("confidence", 0)),
                )
            except (_json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning("position_agent_parse_error", error=str(e))

        return PositionDecision(action="HOLD", reasoning="Failed to parse response")

    async def close(self) -> None:
        pass  # Clients cached in llm_client module
