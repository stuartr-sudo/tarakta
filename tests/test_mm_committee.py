from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.strategy.mm_committee import MMCommittee
from src.strategy.mm_sanity_agent import AgentVerdict


def _config(mode: str = "shadow") -> SimpleNamespace:
    return SimpleNamespace(
        anthropic_api_key="sk-test",
        mm_committee_enabled=True,
        mm_committee_mode=mode,
        mm_committee_timeout_s=5.0,
        mm_sanity_agent_cache_ttl_seconds=1800.0,
        mm_sanity_agent_cache_price_drift_pct=0.5,
        mm_committee_specialist_model="claude-haiku-4-5-20251001",
        mm_committee_head_trader_model="claude-sonnet-4-6",
        mm_committee_monthly_budget_usd=600.0,
    )


def _ctx() -> dict:
    return {
        "symbol": "BTC/USDT",
        "direction": "long",
        "formation_type": "W",
        "formation_variant": "multi_session",
        "formation_timeframe": "1h",
        "entry_price": 100.0,
        "grade": "B",
        "score_pct": 62.0,
        "htf_trend_4h": "bullish",
        "htf_trend_1d": "bullish",
        "counter_trend": False,
    }


def _veto() -> AgentVerdict:
    return AgentVerdict(
        decision="VETO",
        reason="risk specialist veto",
        confidence=0.9,
        htf_trend_4h="bullish",
        htf_trend_1d="bullish",
        counter_trend=False,
        concerns=["risk_reward"],
        model="claude-sonnet-4-6",
    )


@pytest.mark.asyncio
async def test_shadow_mode_logs_veto_but_returns_approve():
    repo = AsyncMock()
    agent = MMCommittee(config=_config("shadow"), repo=repo)
    agent._get_client = MagicMock(return_value=object())
    agent._run_committee = AsyncMock(return_value=(_veto(), {"status": "ok"}))

    verdict = await agent.review(_ctx())

    assert verdict is not None
    assert verdict.decision == "APPROVE"
    assert "committee_shadow(VETO)" in verdict.reason
    row = repo.insert_mm_agent_decision.call_args.args[0]
    assert row["decision"] == "VETO"
    assert row["committee"]["status"] == "ok"


@pytest.mark.asyncio
async def test_cache_key_includes_mode_shadow_cache_does_not_feed_veto_mode():
    cfg = _config("shadow")
    repo = AsyncMock()
    agent = MMCommittee(config=cfg, repo=repo)
    agent._get_client = MagicMock(return_value=object())
    agent._run_committee = AsyncMock(return_value=(_veto(), {"status": "ok"}))

    first = await agent.review(_ctx())
    second = await agent.review(_ctx())
    assert first.decision == "APPROVE"
    assert second.decision == "APPROVE"
    assert agent._run_committee.await_count == 1

    cfg.mm_committee_mode = "veto"
    third = await agent.review(_ctx())
    assert third.decision == "VETO"
    assert agent._run_committee.await_count == 2


@pytest.mark.asyncio
async def test_api_error_returns_veto_in_veto_mode_and_logs_error_row_only():
    repo = AsyncMock()
    agent = MMCommittee(config=_config("veto"), repo=repo)
    agent._get_client = MagicMock(return_value=object())
    agent._run_committee = AsyncMock(side_effect=RuntimeError("boom"))

    verdict = await agent.review(_ctx())

    assert verdict is not None
    assert verdict.decision == "VETO"
    assert verdict.reason.startswith("committee_error:api_error")
    row = repo.insert_mm_agent_decision.call_args.args[0]
    assert row["decision"] == "ERROR"
    assert row["committee"]["status"] == "error"


@pytest.mark.asyncio
async def test_bnb_canary_is_binding_in_veto_mode():
    repo = AsyncMock()
    agent = MMCommittee(config=_config("veto"), repo=repo)
    agent._get_client = MagicMock(return_value=object())
    ctx = {
        **_ctx(),
        "symbol": "BNB/USDT",
        "direction": "short",
        "formation_variant": "three_hits_how",
        "grade": "F",
        "htf_trend_4h": "bullish",
        "htf_4h_accel": True,
        "counter_trend": True,
    }

    verdict = await agent.review(ctx)

    assert verdict is not None
    assert verdict.decision == "VETO"
    assert "BNB canary" in verdict.reason
    row = repo.insert_mm_agent_decision.call_args.args[0]
    assert row["decision"] == "VETO"
    assert row["committee"]["status"] == "deterministic_canary"
