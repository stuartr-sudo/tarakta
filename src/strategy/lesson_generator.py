"""AI-powered Trade Lesson Generator — the self-improving brain.

After every closed trade, this agent analyzes WHY the trade won or lost
and generates specific, actionable lessons. These lessons are:

1. Stored in the trade_lessons table
2. Fed back into Agent 1, 2, and 3 prompts
3. Tracked for effectiveness (did showing this lesson improve future trades?)

The system gets smarter with every trade because agents see an evolving
set of lessons tailored to the bot's actual performance patterns.
"""
from __future__ import annotations

import time
from typing import Any

from src.config import Settings
from src.strategy.llm_client import (
    generate_json, is_openai_model, MODEL_PRICING,
)
from src.data.repository import Repository
from src.utils.logging import get_logger

logger = get_logger(__name__)


LESSON_SYSTEM_PROMPT = """\
You are a trading post-mortem analyst. Your job is to analyze closed trades and extract \
specific, actionable lessons that will help AI trading agents make better decisions in the future.

## Your Role
After every closed trade (win or loss), you receive the full trade context and must:
1. Diagnose the ROOT CAUSE of why the trade won or lost
2. Generate 1-3 specific lessons that are actionable and concrete
3. Tag each lesson with which agent(s) should learn from it

## Lesson Quality Standards
- **Specific, not generic.** Bad: "Be careful with entries." Good: "When BTC is dropping >2% in 1H, \
skip long entries on altcoins — 7 of last 10 such trades hit SL within 2 hours."
- **Actionable.** Each lesson must describe a concrete condition + action. \
"IF [condition], THEN [do this instead]."
- **Evidence-based.** Reference specific prices, percentages, or timeframes from the trade.
- **Forward-looking.** Focus on what to do NEXT TIME, not what went wrong.

## Lesson Types
- **entry**: Was this the right trade to take? (Agent 1 learns from these)
- **timing**: Was the entry timed well? Too early? Too late? Chased? (Agent 2 learns)
- **exit**: Was the exit well-managed? SL too tight/wide? Closed too early/late? (Agent 3 learns)
- **position_management**: Was the position sized right? SL moved correctly? (Agent 3 learns)
- **risk**: Was the risk appropriate given market conditions? (All agents learn)

## Severity Levels
- **low**: Minor optimization, trade was mostly fine
- **medium**: Meaningful improvement possible, pattern worth noting
- **high**: Significant issue that likely cost real money
- **critical**: Fundamental mistake that must not be repeated

## Pattern Tags
Tag lessons with patterns so they can be retrieved for similar future setups:
- Entry patterns: chasing_entry, counter_trend, weak_structure, strong_confluence
- Timing patterns: premature_entry, late_entry, missed_pullback, good_timing
- Exit patterns: sl_too_tight, sl_too_wide, premature_exit, held_too_long, good_exit
- Market patterns: btc_divergence, low_volume, high_volatility, funding_crowded
- Outcome patterns: quick_win, slow_grind, instant_sl, partial_profit"""

LESSON_JSON_FORMAT = """

## RESPONSE FORMAT — CRITICAL

You MUST respond with a single JSON object and NOTHING else. No markdown, no explanation.

{
  "diagnosis": "<1-2 sentence root cause analysis>",
  "lessons": [
    {
      "lesson_type": "entry" | "timing" | "exit" | "position_management" | "risk",
      "lesson": "<Specific actionable lesson: IF [condition], THEN [action]>",
      "severity": "low" | "medium" | "high" | "critical",
      "applies_to": ["agent1", "agent2", "agent3"],
      "tags": ["pattern_tag_1", "pattern_tag_2"]
    }
  ]
}

Rules:
- Generate 1-3 lessons per trade (quality over quantity)
- For winning trades: identify what worked well so it can be repeated
- For losing trades: identify what went wrong and how to avoid it
- EVERY lesson must have an IF/THEN structure
- Tags should be from the pattern list above (can combine)

Respond with ONLY the JSON object. No other text."""

# Gemini structured output schema
LESSON_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "lesson_type": {"type": "string", "enum": ["entry", "timing", "exit", "position_management", "risk"]},
                    "lesson": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "applies_to": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["lesson_type", "lesson", "severity", "applies_to", "tags"],
            },
        },
    },
    "required": ["diagnosis", "lessons"],
}


class TradeLessonGenerator:
    """Generates AI-powered lessons from closed trades and stores them in the DB.

    This is the core self-improvement mechanism — every trade makes the system smarter.
    """

    def __init__(self, config: Settings, repo: Repository) -> None:
        self._model = getattr(config, "position_agent_model", "gemini-3-flash-preview")
        self._timeout = config.agent_timeout_seconds
        self._repo = repo

        # Route API key based on model provider
        if is_openai_model(self._model):
            self._api_key = config.openai_api_key
        else:
            self._api_key = config.agent_api_key
        self._available = bool(self._api_key)

        # Stats
        self.total_lessons_generated = 0
        self.total_errors = 0
        self.total_cost_usd = 0.0

        # Backoff
        self._fail_count = 0
        self._backoff_until = 0.0

        # Pricing per 1M tokens
        self._pricing = MODEL_PRICING

    def _should_try(self) -> bool:
        if not self._available:
            return False
        if time.time() < self._backoff_until:
            return False
        return True

    async def generate_lessons(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl_usd: float,
        pnl_percent: float,
        exit_reason: str,
        confluence_score: float,
        holding_seconds: float,
        stop_loss: float = 0,
        take_profit: float = 0,
        agent1_reasoning: str = "",
        agent2_reasoning: str = "",
        agent3_reasoning: str = "",
        btc_trend: str = "",
        signal_reasons: list[str] | None = None,
        current_tier: int = 0,
        recent_lessons: list[dict] | None = None,
    ) -> list[dict]:
        """Analyze a closed trade and generate actionable lessons.

        Returns list of lesson dicts that were stored in the DB.
        """
        if not self._should_try():
            return []

        outcome = "win" if pnl_usd > 0 else "loss"
        holding_hours = holding_seconds / 3600

        prompt = self._build_prompt(
            symbol=symbol, direction=direction, outcome=outcome,
            entry_price=entry_price, exit_price=exit_price,
            pnl_usd=pnl_usd, pnl_percent=pnl_percent,
            exit_reason=exit_reason, confluence_score=confluence_score,
            holding_hours=holding_hours, stop_loss=stop_loss,
            take_profit=take_profit, agent1_reasoning=agent1_reasoning,
            agent2_reasoning=agent2_reasoning, agent3_reasoning=agent3_reasoning,
            btc_trend=btc_trend, signal_reasons=signal_reasons or [],
            current_tier=current_tier, recent_lessons=recent_lessons or [],
        )

        t0 = time.time()
        try:
            result = await generate_json(
                model=self._model,
                api_key=self._api_key,
                system_prompt=LESSON_SYSTEM_PROMPT + LESSON_JSON_FORMAT,
                user_prompt=prompt,
                json_schema=LESSON_RESPONSE_SCHEMA,
                thinking_level="low",
                temperature=1.0,
                timeout=self._timeout,
            )

            latency_ms = (time.time() - t0) * 1000

            # Track cost
            in_tok = result.input_tokens
            out_tok = result.output_tokens
            pricing = self._pricing.get(self._model, (0.10, 0.01, 0.40))
            cost = (in_tok * pricing[0] + out_tok * pricing[2]) / 1_000_000
            self.total_cost_usd += cost

            # Parse
            parsed = self._parse_response(result.text)
            if not parsed:
                self._fail_count += 1
                self.total_errors += 1
                return []

            diagnosis = parsed.get("diagnosis", "")
            raw_lessons = parsed.get("lessons", [])

            stored_lessons = []
            for raw in raw_lessons:
                lesson_data = {
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "direction": direction,
                    "outcome": outcome,
                    "pnl_usd": round(pnl_usd, 4),
                    "pnl_percent": round(pnl_percent, 4),
                    "exit_reason": exit_reason,
                    "confluence_score": confluence_score,
                    "holding_hours": round(holding_hours, 2),
                    "lesson_type": raw.get("lesson_type", "entry"),
                    "lesson": raw.get("lesson", ""),
                    "severity": raw.get("severity", "medium"),
                    "applies_to": raw.get("applies_to", ["agent1"]),
                    "tags": raw.get("tags", []),
                }
                result = await self._repo.insert_lesson(lesson_data)
                if result:
                    stored_lessons.append(result)

            self.total_lessons_generated += len(stored_lessons)
            self._fail_count = 0
            self._backoff_until = 0.0

            logger.info(
                "trade_lessons_generated",
                symbol=symbol,
                outcome=outcome,
                pnl_usd=round(pnl_usd, 2),
                num_lessons=len(stored_lessons),
                diagnosis=diagnosis[:100],
                latency_ms=round(latency_ms, 0),
                cost_usd=round(cost, 4),
            )

            return stored_lessons

        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            self._fail_count += 1
            self.total_errors += 1
            backoff = min(300, 30 * (2 ** (self._fail_count - 1)))
            self._backoff_until = time.time() + backoff
            logger.warning(
                "lesson_generator_error",
                error=str(e),
                latency_ms=round(latency_ms, 1),
                backoff_seconds=backoff,
            )
            return []

    def _build_prompt(
        self, symbol: str, direction: str, outcome: str,
        entry_price: float, exit_price: float,
        pnl_usd: float, pnl_percent: float,
        exit_reason: str, confluence_score: float,
        holding_hours: float, stop_loss: float,
        take_profit: float, agent1_reasoning: str,
        agent2_reasoning: str, agent3_reasoning: str,
        btc_trend: str, signal_reasons: list[str],
        current_tier: int, recent_lessons: list[dict],
    ) -> str:
        held_str = f"{holding_hours:.1f} hours" if holding_hours >= 1 else f"{holding_hours * 60:.0f} minutes"

        # Format recent lessons so the AI doesn't repeat them
        lessons_context = ""
        if recent_lessons:
            lessons_context = "\n### Recent Lessons Already Generated (do NOT repeat these)\n"
            for i, l in enumerate(recent_lessons[:5], 1):
                lessons_context += f"{i}. [{l.get('lesson_type', '?')}] {l.get('lesson', '?')[:120]}\n"

        return f"""\
## Closed Trade Post-Mortem

**Symbol:** {symbol}
**Direction:** {direction}
**Outcome:** {outcome.upper()} (${pnl_usd:+.2f}, {pnl_percent:+.2f}%)
**Exit Reason:** {exit_reason}

### Trade Details
- Entry Price: {entry_price:.6g}
- Exit Price: {exit_price:.6g}
- Stop Loss: {stop_loss:.6g}
- Take Profit: {take_profit:.6g}
- Confluence Score: {confluence_score:.0f}/100
- Time Held: {held_str}
- TP Tiers Completed: {current_tier}/3
- BTC Trend: {btc_trend or 'unknown'}

### Signal Reasons
{chr(10).join(f'- {r}' for r in signal_reasons) if signal_reasons else '- No signal reasons recorded'}

### Agent Decisions
**Agent 1 (Entry):** {agent1_reasoning[:300] if agent1_reasoning else 'N/A'}
**Agent 2 (Timing):** {agent2_reasoning[:300] if agent2_reasoning else 'N/A'}
**Agent 3 (Position):** {agent3_reasoning[:300] if agent3_reasoning else 'N/A'}
{lessons_context}
Analyze this trade and generate specific, actionable lessons. Focus on what each agent should do differently next time."""

    def _parse_response(self, text: str) -> dict | None:
        import json as _json
        if not text:
            return None
        text = text.strip()
        try:
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return _json.loads(text)
        except (_json.JSONDecodeError, ValueError) as e:
            logger.warning("lesson_parse_error", error=str(e))
            return None

    async def close(self) -> None:
        pass  # Clients cached in llm_client module


async def format_lessons_for_prompt(
    repo: Repository,
    agent_name: str,
    symbol: str | None = None,
    max_lessons: int = 8,
) -> str:
    """Fetch and format recent lessons for inclusion in an agent prompt.

    This is the key function that makes agents smarter — it injects
    learned lessons into the agent's context so it doesn't repeat mistakes.

    Args:
        repo: Database repository.
        agent_name: Which agent is asking ('agent1', 'agent2', 'agent3').
        symbol: Optional symbol to include symbol-specific lessons.
        max_lessons: Max lessons to include.

    Returns:
        Formatted string to inject into the agent prompt.
    """
    lessons: list[dict] = []

    # Get general lessons for this agent (high+ severity first)
    high_lessons = await repo.get_recent_lessons(
        applies_to=agent_name, limit=max_lessons, min_severity="high"
    )
    lessons.extend(high_lessons)

    # Fill remaining with medium severity
    remaining = max_lessons - len(lessons)
    if remaining > 0:
        all_lessons = await repo.get_recent_lessons(
            applies_to=agent_name, limit=remaining + len(lessons)
        )
        # Deduplicate
        seen_ids = {l["id"] for l in lessons}
        for l in all_lessons:
            if l["id"] not in seen_ids and len(lessons) < max_lessons:
                lessons.append(l)
                seen_ids.add(l["id"])

    # Add symbol-specific lessons if provided
    if symbol:
        symbol_lessons = await repo.get_lessons_for_symbol(symbol, limit=3)
        seen_ids = {l["id"] for l in lessons}
        for l in symbol_lessons:
            if l["id"] not in seen_ids:
                lessons.append(l)
                seen_ids.add(l["id"])

    if not lessons:
        return ""

    # Track that these lessons were shown
    for l in lessons:
        await repo.increment_lesson_applied(l["id"])

    # Format for prompt injection
    lines = [
        "\n## LEARNED LESSONS (from past trade outcomes — apply these!)\n",
        "These lessons are generated from real trade post-mortems. Follow them to avoid repeating mistakes.\n",
    ]

    for i, l in enumerate(lessons, 1):
        outcome_emoji = "W" if l.get("outcome") == "win" else "L"
        severity = l.get("severity", "medium").upper()
        symbol_tag = f" [{l.get('symbol', '?')}]" if l.get("symbol") != symbol else ""
        lines.append(
            f"{i}. [{severity}] ({outcome_emoji}{symbol_tag}) {l.get('lesson', 'N/A')}"
        )

    return "\n".join(lines)
