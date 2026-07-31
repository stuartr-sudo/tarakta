"""MM Agent Committee.

Drop-in replacement for ``MMSanityAgent``. The public contract stays:
``async review(context) -> AgentVerdict | None``. The engine only ever sees
APPROVE/VETO; ERROR is persisted as an observability classification on
``mm_agent_decisions``.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.strategy.mm_claude_cli import ClaudeCLIClient, find_claude_cli
from src.strategy.mm_sanity_agent import AgentVerdict, MODEL_PRICING, _jsonable
from src.utils.logging import get_logger

logger = get_logger(__name__)


COMMITTEE_PROMPT_VERSION = "committee_prompt_v=1"
SKILL_DIR = Path(__file__).resolve().parents[2] / "docs" / "agent-committee" / "skills"


SPECIALISTS: dict[str, str] = {
    "structure": "structure_specialist.md",
    "flow_data": "flow_data_specialist.md",
    "cycle": "cycle_specialist.md",
    "htf": "htf_specialist.md",
    "risk": "risk_specialist.md",
}


HEAD_TRADER_SKILL = "head_trader.md"


@dataclass
class SpecialistVerdict:
    name: str
    alignment: int = 0       # -2 strong veto, -1 caution, 0 neutral, 1 approve, 2 strong approve
    decision: str = "NEUTRAL"
    reason: str = ""
    concerns: list[str] = field(default_factory=list)
    raw_response: str = ""
    model: str = ""
    latency_ms: int = 0
    cost_usd: float = 0.0


class MMCommittee:
    """Five Haiku specialists plus a Sonnet head trader.

    The committee is disabled by default via ``mm_committee_enabled=False``.
    In ``shadow`` mode, the actual committee verdict is logged but the engine
    receives APPROVE. In ``veto`` mode, a VETO is binding.
    """

    def __init__(self, config: Any, repo: Any) -> None:
        self.config = config
        self.repo = repo
        self._client: Any = None
        self._skill_cache: dict[str, str] = {}
        self._decision_cache: dict[tuple, tuple[AgentVerdict, dict, float, float]] = {}

    def _get_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        api_key = getattr(self.config, "anthropic_api_key", "") or ""
        if not api_key:
            # No API key → fall back to the Claude Code CLI, which uses the
            # user's subscription login (Keychain OAuth) instead of metered
            # API billing. See src/strategy/mm_claude_cli.py.
            if bool(getattr(self.config, "mm_committee_cli_enabled", True)):
                cli_path = find_claude_cli(
                    str(getattr(self.config, "mm_committee_cli_path", "") or "")
                )
                if cli_path:
                    self._client = ClaudeCLIClient(
                        cli_path,
                        timeout_s=float(
                            getattr(self.config, "mm_committee_cli_timeout_s", 120.0)
                        ),
                        oauth_token=str(
                            getattr(self.config, "claude_code_oauth_token", "") or ""
                        ),
                    )
                    logger.info("mm_committee_backend",
                                backend="claude_cli", cli_path=cli_path)
                    return self._client
                logger.warning("mm_committee_cli_not_found",
                               hint="no ANTHROPIC_API_KEY and no claude binary")
            return None
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            logger.error("mm_committee_anthropic_sdk_missing",
                         fix="pip install anthropic>=0.40.0")
            return None
        self._client = AsyncAnthropic(api_key=api_key)
        return self._client

    def _total_timeout_s(self, client: Any) -> float:
        """Committee-wide deadline. CLI calls are much slower than SDK calls
        (subprocess startup + no streaming), and the committee is two
        sequential stages (specialists, then head trader) — so give the CLI
        backend two per-call budgets plus slack instead of the SDK default."""
        base = float(getattr(self.config, "mm_committee_timeout_s", 30.0))
        if isinstance(client, ClaudeCLIClient):
            return max(base, client.timeout_s * 2 + 30.0)
        return base

    def _mode(self) -> str:
        mode = str(getattr(self.config, "mm_committee_mode", "shadow") or "shadow").lower()
        return mode if mode in {"shadow", "veto"} else "shadow"

    def _cache_key(self, ctx: dict[str, Any], mode: str) -> tuple | None:
        sym = ctx.get("symbol")
        direction = ctx.get("direction")
        variant = ctx.get("formation_variant")
        entry = ctx.get("entry_price")
        if not sym or not direction or not variant or not entry:
            return None
        try:
            entry_rounded = round(float(entry), 4)
        except (TypeError, ValueError):
            return None
        if entry_rounded <= 0:
            return None
        return (mode, str(sym), str(direction), str(variant), entry_rounded)

    async def review(self, context: dict[str, Any]) -> AgentVerdict | None:
        if not getattr(self.config, "mm_committee_enabled", False):
            return None

        mode = self._mode()
        cache_ttl_s = float(getattr(self.config, "mm_sanity_agent_cache_ttl_seconds", 1800.0))
        cache_price_drift_pct = float(getattr(self.config, "mm_sanity_agent_cache_price_drift_pct", 0.5))
        cache_key = self._cache_key(context, mode)
        if cache_ttl_s > 0 and cache_key is not None:
            hit = self._decision_cache.get(cache_key)
            if hit is not None:
                cached_verdict, committee, cached_at, cached_entry = hit
                age_s = time.time() - cached_at
                current_entry = float(context.get("entry_price") or 0)
                drift_pct = abs(current_entry - cached_entry) / cached_entry * 100 if cached_entry > 0 else 100.0
                if age_s <= cache_ttl_s and drift_pct <= cache_price_drift_pct:
                    verdict = self._copy_verdict(cached_verdict, reason_prefix="[cached] ", cost_usd=0.0)
                    logger.info(
                        "mm_committee_cache_hit",
                        symbol=context.get("symbol"),
                        mode=mode,
                        decision=verdict.decision,
                        age_seconds=int(age_s),
                        drift_pct=round(drift_pct, 3),
                    )
                    return self._engine_verdict(verdict, mode)
                self._decision_cache.pop(cache_key, None)

        client = self._get_client()
        if client is None:
            actual = self._error_verdict("client_unavailable", context, model="")
            committee = {"status": "error", "reason": "client_unavailable"}
            await self._log_decision(context, actual, committee, decision="ERROR")
            return self._engine_verdict(actual, mode)

        started = time.perf_counter()
        try:
            actual, committee = await asyncio.wait_for(
                self._run_committee(client, context),
                timeout=self._total_timeout_s(client),
            )
        except asyncio.TimeoutError:
            actual = self._error_verdict("timeout", context, model="")
            committee = {"status": "error", "reason": "timeout"}
            await self._log_decision(context, actual, committee, decision="ERROR")
            return self._engine_verdict(actual, mode)
        except Exception as e:
            actual = self._error_verdict(f"api_error:{e!s}", context, model="")
            committee = {"status": "error", "reason": f"api_error:{e!s}"}
            await self._log_decision(context, actual, committee, decision="ERROR")
            return self._engine_verdict(actual, mode)

        if actual.latency_ms <= 0:
            actual.latency_ms = int((time.perf_counter() - started) * 1000)
        await self._log_decision(context, actual, committee, decision=actual.decision)

        if cache_ttl_s > 0 and cache_key is not None:
            try:
                cached_entry = float(context.get("entry_price") or 0)
            except (TypeError, ValueError):
                cached_entry = 0.0
            if cached_entry > 0:
                self._decision_cache[cache_key] = (actual, committee, time.time(), cached_entry)

        return self._engine_verdict(actual, mode)

    async def _run_committee(self, client: Any, context: dict[str, Any]) -> tuple[AgentVerdict, dict]:
        canary = self._bnb_canary_verdict(context)
        if canary is not None:
            committee = {
                "status": "deterministic_canary",
                "specialists": [],
                "head_trader": {"reason": canary.reason},
            }
            return canary, committee

        specialist_model, head_model, escalation_allowed = await self._choose_models()
        specialists = await asyncio.gather(
            *[
                self._call_specialist(client, name, file_name, specialist_model, context)
                for name, file_name in SPECIALISTS.items()
            ]
        )
        contested = self._is_contested(specialists)
        if contested and escalation_allowed:
            # Contested bench + budget headroom (<75% of cap): the final call
            # is worth the strongest model. Previously the escalation model
            # was configured but never consumed — the boolean was only
            # rendered into the prompt.
            head_model = str(getattr(
                self.config, "mm_committee_escalation_model", head_model,
            ))
        head_verdict, head_raw, head_usage, head_latency_ms = await self._call_head_trader(
            client=client,
            model=head_model,
            context=context,
            specialists=specialists,
            contested=contested,
            escalation_allowed=escalation_allowed,
        )
        head_cost = self._compute_cost(head_model, head_usage)
        total_cost = head_cost + sum(s.cost_usd for s in specialists)
        head_verdict.cost_usd = round(total_cost, 6)
        head_verdict.latency_ms = max(head_verdict.latency_ms, head_latency_ms)
        head_verdict.raw_response = head_raw

        committee = {
            "status": "ok",
            "mode": self._mode(),
            "contested": contested,
            "specialist_model": specialist_model,
            "head_trader_model": head_model,
            "escalation_allowed": escalation_allowed,
            "specialists": [_jsonable(s.__dict__) for s in specialists],
            "head_trader": {
                "decision": head_verdict.decision,
                "reason": head_verdict.reason,
                "confidence": head_verdict.confidence,
                "concerns": head_verdict.concerns,
                "model": head_model,
            },
        }
        return head_verdict, committee

    def _bnb_canary_verdict(self, context: dict[str, Any]) -> AgentVerdict | None:
        """BNB 2026-04-17 canary: never approve three-hits-HOW short into accelerating 4H uptrend."""
        if (
            str(context.get("symbol", "")).upper().startswith("BNB/")
            and str(context.get("direction")) == "short"
            and str(context.get("formation_variant")) == "three_hits_how"
            and str(context.get("htf_trend_4h")) == "bullish"
            and bool(context.get("htf_4h_accel"))
            and str(context.get("grade", "")).upper() == "F"
        ):
            return AgentVerdict(
                decision="VETO",
                reason="BNB canary: three_hits_how short into accelerating bullish 4H F-grade setup.",
                confidence=1.0,
                htf_trend_4h="bullish",
                htf_trend_1d=str(context.get("htf_trend_1d") or "unknown"),
                counter_trend=True,
                concerns=["accelerating_trend", "4h_alignment", "low_grade"],
                model="deterministic_canary",
            )
        return None

    async def _call_specialist(
        self,
        client: Any,
        name: str,
        file_name: str,
        model: str,
        context: dict[str, Any],
    ) -> SpecialistVerdict:
        system = self._load_skill(file_name)
        user_prompt = self._render_specialist_prompt(name, context)
        started = time.perf_counter()
        raw, usage = await self._call_model(client, model, system, user_prompt, max_tokens=1024)
        latency_ms = int((time.perf_counter() - started) * 1000)
        parsed = self._parse_specialist(raw, name)
        parsed.raw_response = raw
        parsed.model = model
        parsed.latency_ms = latency_ms
        parsed.cost_usd = self._compute_cost(model, usage)
        return parsed

    async def _call_head_trader(
        self,
        *,
        client: Any,
        model: str,
        context: dict[str, Any],
        specialists: list[SpecialistVerdict],
        contested: bool,
        escalation_allowed: bool,
    ) -> tuple[AgentVerdict, str, dict, int]:
        system = self._load_skill(HEAD_TRADER_SKILL)
        user_prompt = self._render_head_prompt(context, specialists, contested, escalation_allowed)
        started = time.perf_counter()
        raw, usage = await self._call_model(client, model, system, user_prompt, max_tokens=1536)
        latency_ms = int((time.perf_counter() - started) * 1000)
        verdict = self._parse_head(raw, model=model, latency_ms=latency_ms, cost_usd=0.0)
        if verdict is None:
            raise ValueError("malformed_head_trader_response")
        return verdict, raw, usage, latency_ms

    async def _call_model(
        self,
        client: Any,
        model: str,
        system: str,
        user_prompt: str,
        *,
        max_tokens: int,
    ) -> tuple[str, dict]:
        if isinstance(client, ClaudeCLIClient):
            return await client.call(
                model, system, user_prompt, max_tokens=max_tokens
            )
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
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

    def _load_skill(self, file_name: str) -> str:
        if file_name not in self._skill_cache:
            path = SKILL_DIR / file_name
            self._skill_cache[file_name] = path.read_text(encoding="utf-8")
        return self._skill_cache[file_name]

    def _render_specialist_prompt(self, name: str, context: dict[str, Any]) -> str:
        return (
            f"# {COMMITTEE_PROMPT_VERSION}\n"
            f"specialist={name}\n"
            "Return strict JSON only.\n\n"
            f"SETUP_CONTEXT:\n{json.dumps(_jsonable(context), sort_keys=True)}"
        )

    def _render_head_prompt(
        self,
        context: dict[str, Any],
        specialists: list[SpecialistVerdict],
        contested: bool,
        escalation_allowed: bool,
    ) -> str:
        return (
            f"# {COMMITTEE_PROMPT_VERSION}\n"
            "Return strict JSON only.\n"
            f"contested={contested} escalation_allowed={escalation_allowed}\n\n"
            f"SETUP_CONTEXT:\n{json.dumps(_jsonable(context), sort_keys=True)}\n\n"
            f"SPECIALISTS:\n{json.dumps([_jsonable(s.__dict__) for s in specialists], sort_keys=True)}"
        )

    def _parse_specialist(self, raw: str, name: str) -> SpecialistVerdict:
        data = self._extract_json(raw)
        if not data:
            raise ValueError(f"malformed_specialist_response:{name}")
        decision = str(data.get("decision", "NEUTRAL")).upper()
        if decision not in {"APPROVE", "VETO", "NEUTRAL"}:
            decision = "NEUTRAL"
        alignment = int(max(-2, min(2, int(data.get("alignment", 0) or 0))))
        return SpecialistVerdict(
            name=name,
            alignment=alignment,
            decision=decision,
            reason=str(data.get("reason", "")),
            concerns=list(data.get("concerns", []) or [])[:6],
        )

    def _parse_head(
        self,
        raw: str,
        *,
        model: str,
        latency_ms: int,
        cost_usd: float,
    ) -> AgentVerdict | None:
        data = self._extract_json(raw)
        if not data:
            return None
        decision = str(data.get("decision", "")).upper()
        if decision not in {"APPROVE", "VETO"}:
            return None
        try:
            return AgentVerdict(
                decision=decision,
                reason=str(data.get("reason", "")),
                confidence=float(data.get("confidence", 0.0)),
                htf_trend_4h=str(data.get("htf_trend_4h", "unknown")),
                htf_trend_1d=str(data.get("htf_trend_1d", "unknown")),
                counter_trend=bool(data.get("counter_trend", False)),
                concerns=list(data.get("concerns", []) or [])[:6],
                model=model,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                raw_response=raw,
            )
        except (TypeError, ValueError):
            return None

    def _extract_json(self, raw: str) -> dict | None:
        """Extract the first JSON object from a model reply.

        Models (the CLI backend especially) wrap JSON in ```fences and append
        prose after it — observed live 2026-07-18: the risk specialist emitted
        valid fenced JSON followed by "To complete evaluation, provide: ...".
        A first-{-to-last-} slice breaks on any trailing brace, so decode
        incrementally from each candidate '{' and take the first valid object.
        """
        if not raw:
            return None
        text = raw.strip()
        decoder = json.JSONDecoder()
        idx = text.find("{")
        while idx != -1:
            try:
                data, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                idx = text.find("{", idx + 1)
                continue
            return data if isinstance(data, dict) else None
        return None

    def _is_contested(self, specialists: list[SpecialistVerdict]) -> bool:
        if not specialists:
            return False
        alignments = [s.alignment for s in specialists]
        return min(alignments) <= -2 and max(alignments) >= 1

    async def _choose_models(self) -> tuple[str, str, bool]:
        specialist = str(getattr(
            self.config,
            "mm_committee_specialist_model",
            "claude-sonnet-5",
        ))
        head = str(getattr(self.config, "mm_committee_head_trader_model", "claude-opus-5"))
        cap = float(getattr(self.config, "mm_committee_monthly_budget_usd", 600.0))
        if cap <= 0:
            return specialist, head, True
        try:
            spent = float(await self.repo.get_mm_agent_month_cost())
        except Exception:
            spent = 0.0
        if spent >= cap * 0.9:
            head = specialist
        escalation_allowed = spent < cap * 0.75
        return specialist, head, escalation_allowed

    def _compute_cost(self, model: str, usage: dict) -> float:
        if usage.get("backend") == "claude_cli":
            # Subscription-billed run: the CLI reports the authoritative cost
            # (0 for subscription usage). No pricing-table lookup or warning.
            return round(float(usage.get("total_cost_usd") or 0.0), 6)
        pricing = MODEL_PRICING.get(model)
        if pricing is None and model == "claude-haiku-4-5":
            pricing = MODEL_PRICING.get("claude-haiku-4-5-20251001")
        if not pricing:
            logger.warning("mm_committee_unknown_model_pricing", model=model)
            return 0.0
        fresh_input = usage.get("input_tokens", 0) or 0
        cache_read = usage.get("cache_read_input_tokens", 0) or 0
        cache_write = usage.get("cache_creation_input_tokens", 0) or 0
        output = usage.get("output_tokens", 0) or 0
        total = (
            fresh_input * pricing["input"]
            + cache_read * pricing["cache_read"]
            + cache_write * pricing["cache_write_1h"]
            + output * pricing["output"]
        ) / 1_000_000
        return round(total, 6)

    def _error_verdict(self, reason: str, context: dict[str, Any], model: str) -> AgentVerdict:
        return AgentVerdict(
            decision="VETO",
            reason=f"committee_error:{reason}",
            confidence=1.0,
            htf_trend_4h=str(context.get("htf_trend_4h") or "unknown"),
            htf_trend_1d=str(context.get("htf_trend_1d") or "unknown"),
            counter_trend=bool(context.get("counter_trend", False)),
            concerns=["committee_error"],
            model=model,
            latency_ms=0,
            cost_usd=0.0,
            raw_response="",
        )

    def _engine_verdict(self, actual: AgentVerdict, mode: str) -> AgentVerdict:
        if mode != "shadow":
            return actual
        if actual.decision == "APPROVE":
            return actual
        return AgentVerdict(
            decision="APPROVE",
            reason=f"committee_shadow({actual.decision}): {actual.reason}",
            confidence=actual.confidence,
            htf_trend_4h=actual.htf_trend_4h,
            htf_trend_1d=actual.htf_trend_1d,
            counter_trend=actual.counter_trend,
            concerns=list(actual.concerns),
            model=actual.model,
            latency_ms=actual.latency_ms,
            cost_usd=actual.cost_usd,
            raw_response=actual.raw_response,
        )

    def _copy_verdict(
        self,
        verdict: AgentVerdict,
        *,
        reason_prefix: str = "",
        cost_usd: float | None = None,
    ) -> AgentVerdict:
        return AgentVerdict(
            decision=verdict.decision,
            reason=f"{reason_prefix}{verdict.reason}",
            confidence=verdict.confidence,
            htf_trend_4h=verdict.htf_trend_4h,
            htf_trend_1d=verdict.htf_trend_1d,
            counter_trend=verdict.counter_trend,
            concerns=list(verdict.concerns),
            model=verdict.model,
            latency_ms=0,
            cost_usd=verdict.cost_usd if cost_usd is None else cost_usd,
            raw_response=verdict.raw_response,
        )

    async def _log_decision(
        self,
        ctx: dict[str, Any],
        verdict: AgentVerdict,
        committee: dict,
        *,
        decision: str,
    ) -> None:
        row = {
            "symbol": ctx.get("symbol"),
            "cycle_count": ctx.get("cycle_count"),
            "formation_type": ctx.get("formation_type"),
            "formation_variant": ctx.get("formation_variant"),
            "confluence_grade": ctx.get("grade"),
            "confluence_pct": ctx.get("score_pct"),
            "direction": ctx.get("direction"),
            "decision": decision,
            "reason": verdict.reason,
            "confidence": verdict.confidence,
            "htf_trend_4h": verdict.htf_trend_4h,
            "htf_trend_1d": verdict.htf_trend_1d,
            "counter_trend": verdict.counter_trend,
            "concerns": verdict.concerns,
            "input_context": _jsonable(ctx),
            "raw_response": verdict.raw_response,
            "model": verdict.model,
            "prompt_version": COMMITTEE_PROMPT_VERSION,
            "latency_ms": verdict.latency_ms,
            "cost_usd": verdict.cost_usd,
            "committee": _jsonable(committee),
        }
        try:
            await self.repo.insert_mm_agent_decision(row)
        except Exception as e:
            logger.warning("mm_committee_log_decision_failed",
                           symbol=ctx.get("symbol"), error=str(e))
