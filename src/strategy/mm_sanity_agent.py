"""MM Sanity Agent (Agent 4) — LLM veto layer for MM Method trade entries.

Fires AFTER all deterministic rules have passed, BEFORE MMSignal is handed
to sizing/execution. The agent's job is to catch the judgement-call
failures rules cannot — e.g. "this three_hits_how counter-trend setup
looks valid, but the 4H is accelerating so it isn't actually at
exhaustion, so veto". See docs/MM_SANITY_AGENT_DESIGN.md for the full
design, rubric, and rollout plan.

This module is self-contained: no external LLM-client wrapper dependency,
just the raw `anthropic` SDK. Extended thinking is always on; system
prompt is cached with 1h TTL (March 2026 default regressed to 5m so we
request 1h explicitly). Graceful degradation is fail-open (APPROVE) so an
API outage never halts MM trading.

Cost / model selection:
- Default: claude-opus-4-7 with extended thinking.
- Budget-cap fallback: claude-sonnet-4-6 (also with thinking) when
  projected monthly spend exceeds 90% of the configured cap.
- See §6 of the design doc for full cost breakdown.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model pricing (per MTok, USD). Cache write = 2x input for 1h TTL;
# cache read = 0.1x input (90% discount). Published Anthropic prices.
# ---------------------------------------------------------------------------
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {
        "input": 15.00,
        "cache_write_1h": 30.00,   # 2x input
        "cache_read": 1.50,        # 0.1x input
        "output": 75.00,
    },
    "claude-sonnet-4-6": {
        "input": 3.00,
        "cache_write_1h": 6.00,
        "cache_read": 0.30,
        "output": 15.00,
    },
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "cache_write_1h": 2.00,
        "cache_read": 0.10,
        "output": 5.00,
    },
}


# ---------------------------------------------------------------------------
# Prompt version — bump when the rubric changes so decisions can be joined
# to their prompt version during post-hoc analysis.
# ---------------------------------------------------------------------------
PROMPT_VERSION = "prompt_v=1 rubric_v=1"


# ---------------------------------------------------------------------------
# System prompt (STATIC — cached with 1h TTL).
#
# The rubric, worked examples, and output schema all live here. Per-call
# user prompts are minimal (see _build_user_prompt). Keep this string
# deterministic byte-for-byte so the cache hit-rate stays high; any change
# invalidates the cache for every in-flight session.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the MM Method sanity reviewer for an automated crypto bot. A rule-based
engine has already passed this setup through a formation check, an HTF-trend
hard-veto, and confluence scoring. Your job is to catch the judgement-call
failures those rules cannot catch, grounded in the MMM Masterclasses course.

RUBRIC — reason through these in order before committing to a verdict:

1. HTF alignment (Lesson 12, Trend / EMAs):
   A short into a cleanly stacked bullish 4H (10>20>50>200) is counter-trend.
   A long into a stacked bearish 4H is counter-trend. Counter-trend is only
   acceptable at Level-3 exhaustion with explicit exemption variant
   (three_hits_how/low, final_damage, half_batman, nyc_reversal, 200ema_rejection,
   stophunt). If the 4H is accelerating (fan-out widening), even exemption
   variants are suspect — the move isn't exhausting.

2. Daily trend (Lesson 12 continued):
   1D direction should agree with 4H for A-grade setups. If 1D contradicts
   4H, this is a transition zone — tighter scrutiny, lower confidence.

3. Session timing (Lesson 04):
   Entries are cleanest at session changeover (Asia close, London open,
   NY open). Mid-session entries on reversal variants are weaker. Brinks
   setups are exception — they are session-dependent by design.

4. Asia spike bias (Lesson 15, London range):
   If Asia printed a wide spike in the opposite direction of the setup
   and we're in early UK session, that is a trap trigger — veto.

5. Formation composition (Lesson 07, M and W):
   Multi-session formations are premium; same-session formations are noise
   unless at HOW/LOW with strong retest (3+ of 4 conditions). A same-session
   formation with 2/4 retest conditions on a Grade C score is the weakest
   setup the engine allows — veto unless exceptional.

6. Weekly phase (Lesson 03, Weekly setup):
   Mon/Tue accumulation -> aligned trades only.
   Wed/Thu range -> counter-trend setups valid only at exhaustion.
   Fri trap -> veto counter-trend shorts regardless of formation; the Friday
   trap is a known MM move and we should not fight it from the wrong side.

7. Recent outcome context:
   If the last 3 trades on this symbol in this direction all lost, the
   current regime for this symbol is against us. Heavily weight VETO.

WORKED EXAMPLES:

EXAMPLE 1 - Counter-trend reversal into accelerating HTF (pattern seed: BNB 2026-04-17):
- direction=short, variant=three_hits_how, grade=F(37.8%), retest_met=2/4
- 4h_trend=bullish strength=0.72 accelerating=true
- 1d_trend=bullish
- Reasoning: Exemption list includes three_hits_how, BUT 4H is accelerating
  so exemption is void per Rubric 1. Grade F + aggressive entry already
  marginal. 2/4 retest.
- Verdict: VETO, confidence 0.92, concerns=[4h_alignment, accelerating_trend, low_grade]
- reason: "Three_hits_how exemption voided by accelerating 4H uptrend; Grade F
  signal does not justify fighting the trend."

EXAMPLE 2 - textbook long at LOW:
- direction=long, variant=multi_session, grade=A(72%), retest_met=3/4
- 4h_trend=bullish strength=0.61 accelerating=false
- 1d_trend=bullish
- session=london_open minutes_in=12
- Reasoning: HTF aligned, premium multi-session formation, strong retest,
  clean session timing. Textbook setup.
- Verdict: APPROVE, confidence 0.94, concerns=[]
- reason: "Multi-session W at LOW with 3/4 retest, HTF aligned, London open."

EXAMPLE 3 - Friday trap counter-trend short:
- direction=short, variant=standard, grade=B+(61%), retest_met=3/4
- 4h_trend=bullish strength=0.45
- weekly_phase=FRIDAY_TRAP dow=4
- Reasoning: Rubric 6 explicitly vetoes counter-trend shorts in Friday trap
  regardless of formation quality. Confluence is high but context wrong.
- Verdict: VETO, confidence 0.89, concerns=[friday_trap, wrong_phase]
- reason: "Friday trap phase - course rule explicitly forbids counter-trend
  shorts regardless of setup quality."

OUTPUT SCHEMA (strict, JSON only, no prose):
{
  "decision":    "APPROVE" | "VETO",
  "reason":      "<=30 words, cite the Rubric point number",
  "confidence":  0.0-1.0,
  "htf_trend_4h":"bullish" | "bearish" | "sideways",
  "htf_trend_1d":"bullish" | "bearish" | "sideways",
  "counter_trend":true | false,
  "concerns":    [<=4 tags from the vocabulary]
}

CONCERNS VOCABULARY (controlled - do not invent tags):
4h_alignment, daily_alignment, accelerating_trend, asia_spike, same_session,
wrong_phase, friday_trap, low_retest, low_grade, low_quality_formation,
recent_losses, mid_session

Output MUST be valid JSON matching the schema above. No surrounding prose,
no markdown fences, no commentary. Extended thinking is enabled - use it to
walk through each rubric point before committing to the final JSON.
"""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AgentVerdict:
    """Structured verdict returned by the sanity agent.

    ``decision`` is the binding call. ``None`` returned by
    ``MMSanityAgent.review`` (not this dataclass) means the agent failed
    and the engine should fail-open (approve).
    """

    decision: str  # "APPROVE" | "VETO"
    reason: str
    confidence: float
    htf_trend_4h: str
    htf_trend_1d: str
    counter_trend: bool
    concerns: list[str] = field(default_factory=list)

    # Observability metadata
    model: str = ""
    latency_ms: int = 0
    cost_usd: float = 0.0
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class MMSanityAgent:
    """Reviews MM setups with Opus 4.7 + extended thinking.

    Usage::

        agent = MMSanityAgent(config, repo)
        verdict = await agent.review(context)
        if verdict is None:
            # API failed; fail-open (APPROVE)
            ...
        elif verdict.decision == "VETO":
            # Reject the setup
            ...
    """

    def __init__(self, config: Any, repo: Any) -> None:
        self.config = config
        self.repo = repo
        self._client: Any = None  # lazily initialised (see _get_client)
        self._budget_exceeded_this_month: bool = False

    def _get_client(self) -> Any | None:
        """Lazily instantiate the Anthropic client.

        Returns None if the SDK is unavailable or the API key is missing.
        The caller treats that as "agent disabled" and falls open.
        """
        if self._client is not None:
            return self._client
        api_key = getattr(self.config, "anthropic_api_key", "") or ""
        if not api_key:
            return None
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            logger.error("mm_agent_anthropic_sdk_missing",
                         fix="pip install anthropic>=0.40.0")
            return None
        self._client = AsyncAnthropic(api_key=api_key)
        return self._client

    # -----------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------
    async def review(self, context: dict[str, Any]) -> AgentVerdict | None:
        """Review a setup; return an AgentVerdict or None on failure.

        None → fail-open (engine approves by default). This is deliberate:
        we do not want an API outage to halt all MM trading. A row with
        decision='ERROR' is still written to mm_agent_decisions for
        observability.
        """
        # Kill switch
        if not getattr(self.config, "mm_sanity_agent_enabled", True):
            return None

        client = self._get_client()
        if client is None:
            # No client = no key or no SDK. Log once per cycle.
            logger.warning("mm_agent_client_unavailable", symbol=context.get("symbol"))
            await self._log_decision(context, decision="ERROR",
                                     reason="client_unavailable",
                                     raw_response="", model="", latency_ms=0,
                                     cost_usd=0.0)
            return None

        model = await self._choose_model()
        timeout_s = float(getattr(self.config, "mm_sanity_agent_timeout_s", 20.0))
        # Effort controls adaptive-thinking depth on Opus 4.7 / Sonnet 4.6.
        # "high" is the default for this task — money-critical judgement,
        # not a classification.
        effort = str(getattr(self.config, "mm_sanity_agent_effort", "high")).lower()
        if effort not in {"low", "medium", "high", "max"}:
            effort = "high"

        user_prompt = self._build_user_prompt(context)
        started = time.perf_counter()

        try:
            raw_response, usage = await asyncio.wait_for(
                self._call_model(client, model, user_prompt, effort),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - started) * 1000)
            logger.warning("mm_agent_timeout", symbol=context.get("symbol"),
                           latency_ms=latency_ms, timeout_s=timeout_s)
            await self._log_decision(context, decision="ERROR",
                                     reason=f"timeout_{timeout_s}s",
                                     raw_response="", model=model,
                                     latency_ms=latency_ms, cost_usd=0.0)
            return None
        except Exception as e:
            latency_ms = int((time.perf_counter() - started) * 1000)
            logger.warning("mm_agent_api_error", symbol=context.get("symbol"),
                           error=str(e), latency_ms=latency_ms)
            await self._log_decision(context, decision="ERROR",
                                     reason=f"api_error:{e!s}",
                                     raw_response="", model=model,
                                     latency_ms=latency_ms, cost_usd=0.0)
            return None

        latency_ms = int((time.perf_counter() - started) * 1000)
        cost_usd = self._compute_cost(model, usage)

        verdict = self._parse_response(raw_response, model=model,
                                       latency_ms=latency_ms, cost_usd=cost_usd)
        if verdict is None:
            logger.warning("mm_agent_malformed_response", symbol=context.get("symbol"),
                           raw=raw_response[:400])
            await self._log_decision(context, decision="ERROR",
                                     reason="malformed_json",
                                     raw_response=raw_response, model=model,
                                     latency_ms=latency_ms, cost_usd=cost_usd)
            return None

        # Confidence threshold — if below min, downgrade VETO to a logged
        # concern and approve. With default 0.0 every VETO is honoured.
        min_conf = float(getattr(self.config, "mm_sanity_agent_min_confidence", 0.0))
        if verdict.decision == "VETO" and verdict.confidence < min_conf:
            logger.info("mm_agent_low_confidence_bypass",
                        symbol=context.get("symbol"),
                        confidence=verdict.confidence, min=min_conf)
            verdict = AgentVerdict(
                decision="APPROVE",
                reason=f"low_confidence_bypass({verdict.reason})",
                confidence=verdict.confidence,
                htf_trend_4h=verdict.htf_trend_4h,
                htf_trend_1d=verdict.htf_trend_1d,
                counter_trend=verdict.counter_trend,
                concerns=verdict.concerns,
                model=model, latency_ms=latency_ms, cost_usd=cost_usd,
                raw_response=raw_response,
            )

        await self._log_decision(
            context,
            decision=verdict.decision,
            reason=verdict.reason,
            confidence=verdict.confidence,
            htf_trend_4h=verdict.htf_trend_4h,
            htf_trend_1d=verdict.htf_trend_1d,
            counter_trend=verdict.counter_trend,
            concerns=verdict.concerns,
            raw_response=raw_response,
            model=model, latency_ms=latency_ms, cost_usd=cost_usd,
        )
        return verdict

    # -----------------------------------------------------------------
    # Model selection (budget-aware)
    # -----------------------------------------------------------------
    async def _choose_model(self) -> str:
        """Return the model to call for the next review.

        Auto-downgrades to the fallback model when projected monthly spend
        exceeds 90% of the configured cap. Cached once per scan cycle to
        avoid querying the DB on every call.
        """
        default = getattr(self.config, "mm_sanity_agent_model", "claude-opus-4-7")
        fallback = getattr(self.config, "mm_sanity_agent_fallback_model",
                           "claude-sonnet-4-6")
        cap = float(getattr(self.config, "mm_sanity_agent_monthly_budget_usd", 600.0))

        if cap <= 0:
            return default  # budget cap disabled
        if self._budget_exceeded_this_month:
            return fallback

        try:
            spent = await self.repo.get_mm_agent_month_cost()
        except Exception:
            return default  # DB hiccup shouldn't force downgrade

        # Project to full month assuming a proportional remainder. A
        # simple linear projection is good enough — we only need to catch
        # runaway spend, not predict it precisely.
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        days_so_far = max(1, now.day)
        days_in_month = 30
        projected = spent * (days_in_month / days_so_far)
        if projected > cap * 0.9:
            logger.warning("mm_agent_budget_downgraded",
                           spent=round(spent, 2), projected=round(projected, 2),
                           cap=cap, fallback_model=fallback)
            self._budget_exceeded_this_month = True
            return fallback
        return default

    # -----------------------------------------------------------------
    # API call
    # -----------------------------------------------------------------
    async def _call_model(
        self,
        client: Any,
        model: str,
        user_prompt: str,
        effort: str,
    ) -> tuple[str, dict]:
        """Send the request and return (raw_text, usage_dict).

        - System prompt is sent with cache_control ttl=1h so the ~10K
          tokens of rubric + fixtures are charged at 10% of input price
          on cache hits (virtually all calls after the first in a 1-hour
          window).
        - Adaptive thinking + output_config.effort is the ONLY supported
          thinking mode on Opus 4.7 — the legacy
          `thinking={"type": "enabled", "budget_tokens": N}` shape is
          rejected with a 400 ("thinking.type.enabled is not supported
          for this model"). Sonnet 4.6 also supports adaptive+effort, so
          the same call shape works for both our default and fallback
          models. See: https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
        - Response format: plain text output. We parse JSON out of it via
          _parse_response — belt + braces vs `response_format`.
        """
        # Build the message body. Note: Anthropic SDK accepts "system" at
        # top level OR as a list of content blocks; we need the list form
        # to attach cache_control.
        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
        )

        # Extract the JSON-bearing text block. With thinking enabled there
        # can be multiple content blocks — a `thinking` block first, then
        # a `text` block. Take the last `text` block.
        text_parts = [
            getattr(b, "text", "") for b in response.content
            if getattr(b, "type", "") == "text"
        ]
        raw = text_parts[-1] if text_parts else ""

        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", 0),
            "output_tokens": getattr(response.usage, "output_tokens", 0),
            "cache_creation_input_tokens":
                getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens":
                getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        }
        return raw, usage

    # -----------------------------------------------------------------
    # Response parsing
    # -----------------------------------------------------------------
    def _parse_response(
        self,
        raw: str,
        *,
        model: str,
        latency_ms: int,
        cost_usd: float,
    ) -> AgentVerdict | None:
        """Parse the model output into an AgentVerdict.

        Tolerates leading/trailing whitespace and ```json fences. Returns
        None if the response is malformed or missing required fields;
        caller falls open on None.
        """
        if not raw:
            return None
        text = raw.strip()
        # Strip markdown fences if the model ignored the "no fences" rule
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        # Find the first {...} block to be robust to any prose
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start: end + 1])
        except json.JSONDecodeError:
            return None

        decision = str(data.get("decision", "")).upper()
        if decision not in ("APPROVE", "VETO"):
            return None

        try:
            return AgentVerdict(
                decision=decision,
                reason=str(data.get("reason", "")),
                confidence=float(data.get("confidence", 0.0)),
                htf_trend_4h=str(data.get("htf_trend_4h", "sideways")),
                htf_trend_1d=str(data.get("htf_trend_1d", "sideways")),
                counter_trend=bool(data.get("counter_trend", False)),
                concerns=list(data.get("concerns", []))[:6],
                model=model, latency_ms=latency_ms, cost_usd=cost_usd,
                raw_response=raw,
            )
        except (TypeError, ValueError):
            return None

    # -----------------------------------------------------------------
    # Cost computation
    # -----------------------------------------------------------------
    def _compute_cost(self, model: str, usage: dict) -> float:
        """Compute the USD cost of a single call from the token usage dict."""
        pricing = MODEL_PRICING.get(model)
        if not pricing:
            # Unknown model — log and treat as zero (don't block on this)
            logger.warning("mm_agent_unknown_model_pricing", model=model)
            return 0.0
        # input_tokens on the response excludes cached-read tokens — it
        # represents "fresh" (uncached) input only. cache_creation is a
        # separate bucket billed at the cache-write rate.
        fresh_input = usage.get("input_tokens", 0) or 0
        cache_read = usage.get("cache_read_input_tokens", 0) or 0
        cache_write = usage.get("cache_creation_input_tokens", 0) or 0
        output = usage.get("output_tokens", 0) or 0

        total_mtok = (
            fresh_input * pricing["input"]
            + cache_read * pricing["cache_read"]
            + cache_write * pricing["cache_write_1h"]
            + output * pricing["output"]
        ) / 1_000_000
        return round(total_mtok, 6)

    # -----------------------------------------------------------------
    # User-prompt builder
    # -----------------------------------------------------------------
    def _build_user_prompt(self, ctx: dict[str, Any]) -> str:
        """Render the per-call user prompt. Keep it minimal — every byte
        is paid at the non-cached rate.
        """
        def _fmt_closes(key: str) -> str:
            closes = ctx.get(key) or []
            return ", ".join(f"{float(c):.4f}" for c in closes)

        recent = ctx.get("recent_trades") or []
        if not recent:
            recent_str = "(no recent trades on this symbol)"
        else:
            recent_str = "\n".join(str(r) for r in recent)

        return f"""# {PROMPT_VERSION}

SETUP
symbol={ctx.get('symbol')} direction={ctx.get('direction')} \
formation={ctx.get('formation_type')}/{ctx.get('formation_variant')}
grade={ctx.get('grade')} confluence={ctx.get('score_pct')}% \
retest_met={ctx.get('retest_met')}/4
entry={ctx.get('entry_price')} sl_ref={ctx.get('sl_ref')}
formation_quality={ctx.get('formation_quality')} \
at_key_level={ctx.get('at_key_level')}

HTF (pre-computed by engine)
4h_trend={ctx.get('htf_trend_4h')} strength={ctx.get('htf_4h_strength')} \
accelerating={ctx.get('htf_4h_accel')}
1d_trend={ctx.get('htf_trend_1d')}
price_vs_50ema_pct={ctx.get('price_vs_50ema_pct')} \
price_vs_200ema_pct={ctx.get('price_vs_200ema_pct')}
counter_trend={ctx.get('counter_trend')}

SESSION & CYCLE
session={ctx.get('session_name')} min_in={ctx.get('minutes_in')}
asia_range_pct={ctx.get('asia_range_pct')} \
asia_spike_dir={ctx.get('asia_spike_dir')}
weekly_phase={ctx.get('weekly_phase')} dow={ctx.get('dow')}
multi_session_formation={ctx.get('multi_session')}

RECENT (last 5 closed {ctx.get('symbol')} MM trades)
{recent_str}

REGIME (last 10 closes per TF, oldest-first; for context only)
4h: {_fmt_closes('c4h_closes')}
1h: {_fmt_closes('c1h_closes')}
15m: {_fmt_closes('c15m_closes')}

Return JSON per the schema in your system prompt.
"""

    # -----------------------------------------------------------------
    # Persistence — fire-and-forget write to mm_agent_decisions
    # -----------------------------------------------------------------
    async def _log_decision(self, ctx: dict[str, Any], **overrides: Any) -> None:
        """Write a mm_agent_decisions row. Never raises."""
        row: dict[str, Any] = {
            "symbol": ctx.get("symbol"),
            "cycle_count": ctx.get("cycle_count"),
            "formation_type": ctx.get("formation_type"),
            "formation_variant": ctx.get("formation_variant"),
            "confluence_grade": ctx.get("grade"),
            "confluence_pct": ctx.get("score_pct"),
            "direction": ctx.get("direction"),
            "prompt_version": PROMPT_VERSION,
            # agent-output overrides below
            "input_context": _jsonable(ctx),
        }
        row.update({k: v for k, v in overrides.items() if v is not None})
        try:
            await self.repo.insert_mm_agent_decision(row)
        except Exception as e:
            logger.warning("mm_agent_log_decision_failed",
                           symbol=ctx.get("symbol"), error=str(e))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-safe primitives.

    The input_context column is jsonb so Supabase will serialise whatever
    we hand it, but pandas Timestamps and numpy floats need coercion.
    """
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj
    return str(obj)


def build_context(
    *,
    symbol: str,
    trade_direction: str,
    best_formation: Any,
    confluence_result: Any,
    entry_price: float,
    sl_ref: float,
    trend_state_4h: Any,
    trend_state_1d: Any,
    ema_state: Any,
    ema_values: dict,
    session: Any,
    cycle_state: Any,
    candles_4h: pd.DataFrame | None,
    candles_1h: pd.DataFrame | None,
    candles_15m: pd.DataFrame | None,
    asia_range_pct: float | None,
    asia_spike_dir: str | None,
    recent_trades: list[dict] | None,
    cycle_count: int | None,
    now: Any,
) -> dict[str, Any]:
    """Assemble the pre-computed feature dict sent to the agent.

    The engine does the computation; the model does the reasoning. Every
    derived feature we hand-compute here is one less judgement error from
    the LLM. See design doc §4 for the contract.
    """
    def _recent_close(df: pd.DataFrame | None) -> float | None:
        if df is None or df.empty:
            return None
        return float(df["close"].iloc[-1])

    def _closes_last_n(df: pd.DataFrame | None, n: int) -> list[float]:
        if df is None or df.empty:
            return []
        return [float(c) for c in df["close"].tail(n).tolist()]

    def _pct_vs(price: float | None, ref: float | None) -> float | None:
        if price is None or ref is None or ref == 0:
            return None
        return round((price - ref) / ref * 100, 3)

    close_1h = _recent_close(candles_1h)
    ema50 = ema_values.get(50) if isinstance(ema_values, dict) else None
    ema200 = ema_values.get(200) if isinstance(ema_values, dict) else None

    dow = None
    try:
        dow = int(now.weekday())
    except Exception:
        pass

    # Formation multi-session heuristic: peaks more than ~16h apart (4
    # sessions of 4h) is multi-session for our purposes.
    multi_session = False
    try:
        p1 = getattr(best_formation, "peak1_idx", 0)
        p2 = getattr(best_formation, "peak2_idx", 0)
        multi_session = abs(p2 - p1) >= 16  # 16x 1H bars
    except Exception:
        pass

    recent_lines = []
    for t in (recent_trades or [])[:5]:
        try:
            recent_lines.append(
                f"dir={t.get('direction')} grade={t.get('mm_confluence_grade')} "
                f"pnl_pct={t.get('pnl_percent') or 0:.2f} "
                f"exit={t.get('exit_reason')} variant={t.get('mm_formation')}"
            )
        except Exception:
            continue

    return {
        "symbol": symbol,
        "direction": trade_direction,
        "formation_type": getattr(best_formation, "type", ""),
        "formation_variant": getattr(best_formation, "variant", ""),
        "formation_quality": round(float(getattr(best_formation, "quality_score", 0.0)), 3),
        "at_key_level": bool(getattr(best_formation, "at_key_level", False)),
        "multi_session": multi_session,
        "grade": getattr(confluence_result, "grade", ""),
        "score_pct": round(float(getattr(confluence_result, "score_pct", 0.0)), 1),
        "retest_met": int(getattr(confluence_result, "retest_conditions_met", 0)),
        "entry_price": round(float(entry_price), 6),
        "sl_ref": round(float(sl_ref), 6),
        "htf_trend_4h": getattr(trend_state_4h, "direction", "unknown") if trend_state_4h else "unknown",
        "htf_4h_strength": round(float(getattr(trend_state_4h, "strength", 0.0) or 0), 3),
        "htf_4h_accel": bool(getattr(trend_state_4h, "is_accelerating", False)),
        "htf_trend_1d": getattr(trend_state_1d, "direction", "unknown") if trend_state_1d else "unknown",
        "price_vs_50ema_pct": _pct_vs(close_1h, ema50),
        "price_vs_200ema_pct": _pct_vs(close_1h, ema200),
        "counter_trend": (
            trend_state_4h is not None
            and getattr(trend_state_4h, "direction", "sideways") != "sideways"
            and (
                (getattr(trend_state_4h, "direction") == "bullish"
                 and trade_direction == "short")
                or (getattr(trend_state_4h, "direction") == "bearish"
                    and trade_direction == "long")
            )
        ),
        "session_name": getattr(session, "session_name", ""),
        "minutes_in": int(getattr(session, "minutes_into_session", 0) or 0),
        "asia_range_pct": (
            round(float(asia_range_pct), 3) if asia_range_pct is not None else None
        ),
        "asia_spike_dir": asia_spike_dir or "none",
        "weekly_phase": getattr(cycle_state, "phase", ""),
        "dow": dow,
        "recent_trades": recent_lines,
        "c4h_closes": _closes_last_n(candles_4h, 10),
        "c1h_closes": _closes_last_n(candles_1h, 10),
        "c15m_closes": _closes_last_n(candles_15m, 10),
        "cycle_count": cycle_count,
    }
