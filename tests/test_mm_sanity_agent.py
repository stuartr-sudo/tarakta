"""Tests for the MM Sanity Agent (Agent 4).

These tests cover the parts we can test without a live LLM:
- Response parsing (tolerant JSON extraction, reject malformed)
- Cost computation (including cache-aware pricing)
- User-prompt assembly
- Graceful degradation when the SDK/API key is missing
- Kill-switch behaviour

The live-LLM fixture test against the BNB pattern lives in a separate
file and is gated on ``ANTHROPIC_LIVE_TEST=1`` so CI doesn't need a key.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.strategy.mm_sanity_agent import (
    MODEL_PRICING,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    AgentVerdict,
    MMSanityAgent,
    build_context,
)


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------
class TestParseResponse:
    def _agent(self):
        # Minimal config / repo stubs — _parse_response doesn't touch them
        return MMSanityAgent(config=SimpleNamespace(anthropic_api_key=""), repo=MagicMock())

    def test_clean_json_veto(self):
        raw = (
            '{"decision":"VETO","reason":"4H uptrend per Rubric 1",'
            '"confidence":0.9,"htf_trend_4h":"bullish","htf_trend_1d":"bullish",'
            '"counter_trend":true,"concerns":["4h_alignment"]}'
        )
        v = self._agent()._parse_response(raw, model="claude-opus-4-7",
                                          latency_ms=1200, cost_usd=0.05)
        assert v is not None
        assert v.decision == "VETO"
        assert v.confidence == 0.9
        assert v.htf_trend_4h == "bullish"
        assert v.counter_trend is True
        assert "4h_alignment" in v.concerns

    def test_clean_json_approve(self):
        raw = (
            '{"decision":"APPROVE","reason":"ok","confidence":0.8,'
            '"htf_trend_4h":"bullish","htf_trend_1d":"bullish",'
            '"counter_trend":false,"concerns":[]}'
        )
        v = self._agent()._parse_response(raw, model="x", latency_ms=0, cost_usd=0.0)
        assert v is not None
        assert v.decision == "APPROVE"
        assert v.concerns == []

    def test_strips_markdown_fences(self):
        raw = (
            '```json\n{"decision":"APPROVE","reason":"","confidence":0.5,'
            '"htf_trend_4h":"sideways","htf_trend_1d":"sideways",'
            '"counter_trend":false,"concerns":[]}\n```'
        )
        v = self._agent()._parse_response(raw, model="x", latency_ms=0, cost_usd=0.0)
        assert v is not None and v.decision == "APPROVE"

    def test_tolerates_prose_around_json(self):
        raw = (
            'Here is my verdict:\n'
            '{"decision":"VETO","reason":"x","confidence":0.7,'
            '"htf_trend_4h":"bearish","htf_trend_1d":"bearish",'
            '"counter_trend":true,"concerns":["asia_spike"]}\n'
            'hope this helps!'
        )
        v = self._agent()._parse_response(raw, model="x", latency_ms=0, cost_usd=0.0)
        assert v is not None and v.decision == "VETO"

    def test_returns_none_on_empty(self):
        v = self._agent()._parse_response("", model="x", latency_ms=0, cost_usd=0.0)
        assert v is None

    def test_returns_none_on_malformed_json(self):
        v = self._agent()._parse_response("{ decision: BAD }", model="x",
                                          latency_ms=0, cost_usd=0.0)
        assert v is None

    def test_returns_none_on_invalid_decision(self):
        raw = (
            '{"decision":"MAYBE","reason":"","confidence":0.5,'
            '"htf_trend_4h":"bullish","htf_trend_1d":"bullish",'
            '"counter_trend":false,"concerns":[]}'
        )
        v = self._agent()._parse_response(raw, model="x", latency_ms=0, cost_usd=0.0)
        assert v is None

    def test_concerns_capped(self):
        raw = (
            '{"decision":"VETO","reason":"x","confidence":0.9,'
            '"htf_trend_4h":"bullish","htf_trend_1d":"bullish",'
            '"counter_trend":true,'
            '"concerns":["a","b","c","d","e","f","g","h"]}'
        )
        v = self._agent()._parse_response(raw, model="x", latency_ms=0, cost_usd=0.0)
        assert v is not None
        # We cap at 6 to tolerate minor model over-tagging
        assert len(v.concerns) <= 6


# ---------------------------------------------------------------------------
# _compute_cost
# ---------------------------------------------------------------------------
class TestComputeCost:
    def _agent(self):
        return MMSanityAgent(config=SimpleNamespace(), repo=MagicMock())

    def test_opus_with_cache_hit(self):
        # 800 fresh input + 10K cached read + 1500 output on Opus 4.7
        usage = {
            "input_tokens": 800,
            "output_tokens": 1500,
            "cache_read_input_tokens": 10_000,
            "cache_creation_input_tokens": 0,
        }
        cost = self._agent()._compute_cost("claude-opus-4-7", usage)
        expected = (
            800 * 15.00
            + 10_000 * 1.50
            + 1500 * 75.00
        ) / 1_000_000
        assert abs(cost - round(expected, 6)) < 1e-6

    def test_opus_cache_write_first_call(self):
        # First call in a cache window: pays the write premium on 10K sys
        usage = {
            "input_tokens": 800,
            "output_tokens": 1500,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 10_000,
        }
        cost = self._agent()._compute_cost("claude-opus-4-7", usage)
        assert cost > 0

    def test_unknown_model_returns_zero(self):
        cost = self._agent()._compute_cost("claude-pluto-1", {"input_tokens": 100})
        assert cost == 0.0

    def test_all_models_priced(self):
        # Guard rail: every model we list in the config must have pricing
        for model in ("claude-opus-4-7", "claude-sonnet-4-6"):
            assert model in MODEL_PRICING
            pricing = MODEL_PRICING[model]
            for key in ("input", "cache_write_1h", "cache_read", "output"):
                assert pricing[key] > 0


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------
class TestBuildUserPrompt:
    def _agent(self):
        return MMSanityAgent(config=SimpleNamespace(), repo=MagicMock())

    def test_contains_prompt_version(self):
        p = self._agent()._build_user_prompt({"symbol": "BTC/USDT"})
        assert PROMPT_VERSION in p

    def test_renders_minimal_context(self):
        ctx = {
            "symbol": "BNB/USDT",
            "direction": "short",
            "formation_type": "M",
            "formation_variant": "three_hits_how",
            "grade": "F",
            "score_pct": 37.8,
            "retest_met": 2,
            "entry_price": 635.21,
            "sl_ref": 639.17,
            "htf_trend_4h": "bullish",
            "htf_4h_strength": 0.72,
            "htf_4h_accel": True,
            "htf_trend_1d": "bullish",
            "counter_trend": True,
            "weekly_phase": "BOARD_MEETING_2",
            "dow": 3,
            "c4h_closes": [1.0, 2.0, 3.0],
        }
        p = self._agent()._build_user_prompt(ctx)
        # Key values must appear in the rendered prompt
        assert "BNB/USDT" in p
        assert "three_hits_how" in p
        assert "bullish" in p
        assert "accelerating=True" in p
        assert "BOARD_MEETING_2" in p

    def test_handles_missing_keys(self):
        # build_user_prompt should not raise on sparse context
        p = self._agent()._build_user_prompt({"symbol": "X"})
        assert "symbol=X" in p


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------
class TestGracefulDegradation:
    async def test_kill_switch(self):
        config = SimpleNamespace(
            mm_sanity_agent_enabled=False,
            anthropic_api_key="sk-ant-xxx",
        )
        agent = MMSanityAgent(config=config, repo=AsyncMock())
        verdict = await agent.review({"symbol": "X"})
        assert verdict is None
        # With agent disabled we should NOT have called the repo either
        agent.repo.insert_mm_agent_decision.assert_not_called()

    async def test_missing_api_key_logs_error_row(self):
        config = SimpleNamespace(
            mm_sanity_agent_enabled=True,
            anthropic_api_key="",
        )
        repo = AsyncMock()
        agent = MMSanityAgent(config=config, repo=repo)
        verdict = await agent.review({"symbol": "X"})
        assert verdict is None
        # One ERROR row should be written for observability
        repo.insert_mm_agent_decision.assert_awaited_once()
        row = repo.insert_mm_agent_decision.call_args.args[0]
        assert row["decision"] == "ERROR"


# ---------------------------------------------------------------------------
# System prompt sanity
# ---------------------------------------------------------------------------
def test_system_prompt_contains_rubric():
    """If this ever changes, the prompt cache invalidates — intentional."""
    assert "RUBRIC" in SYSTEM_PROMPT
    # All 7 rubric points must be present
    for i in range(1, 8):
        assert f"\n{i}." in SYSTEM_PROMPT
    # The three worked examples must be present
    assert "EXAMPLE 1" in SYSTEM_PROMPT
    assert "EXAMPLE 2" in SYSTEM_PROMPT
    assert "EXAMPLE 3" in SYSTEM_PROMPT
    # The controlled vocabulary
    for tag in ("4h_alignment", "friday_trap", "accelerating_trend"):
        assert tag in SYSTEM_PROMPT


def test_system_prompt_stable_for_caching():
    """Two calls to the module should see the same object — it's a module
    constant. If someone refactors this to a function, the prompt cache
    invalidates on every call (bill goes 10x). This test guards that.
    """
    from src.strategy.mm_sanity_agent import SYSTEM_PROMPT as sp2
    assert SYSTEM_PROMPT is sp2


# ---------------------------------------------------------------------------
# build_context — the pre-computed features helper
# ---------------------------------------------------------------------------
def test_build_context_bnb_fixture():
    """Given the BNB 2026-04-17 setup, build_context should produce a
    payload that the system prompt's Example 1 would recognise as
    'counter-trend + accelerating + Grade F'. This is our core
    canary — if this shape ever changes silently, the agent starts
    misreading setups."""
    best_formation = SimpleNamespace(
        type="M", variant="three_hits_how",
        quality_score=0.4, at_key_level=True,
        peak1_idx=5, peak2_idx=10,
    )
    confluence = SimpleNamespace(grade="F", score_pct=37.8, retest_conditions_met=2)
    trend_4h = SimpleNamespace(direction="bullish", strength=0.72, is_accelerating=True)
    trend_1d = SimpleNamespace(direction="bullish", strength=0.6, is_accelerating=False)
    session = SimpleNamespace(session_name="uk", minutes_into_session=60)
    cycle = SimpleNamespace(phase="BOARD_MEETING_2")

    import pandas as pd
    candles_1h = pd.DataFrame({
        "open": [635]*10, "high": [640]*10, "low": [630]*10,
        "close": [638]*10, "volume": [100]*10,
    })

    from datetime import datetime, timezone
    now = datetime(2026, 4, 17, 13, 1, tzinfo=timezone.utc)  # Friday 13:01 UTC

    ctx = build_context(
        symbol="BNB/USDT",
        trade_direction="short",
        best_formation=best_formation,
        confluence_result=confluence,
        entry_price=635.21,
        sl_ref=639.17,
        trend_state_4h=trend_4h,
        trend_state_1d=trend_1d,
        ema_state=None,
        ema_values={50: 630.0, 200: 610.0},
        session=session,
        cycle_state=cycle,
        candles_4h=candles_1h,
        candles_1h=candles_1h,
        candles_15m=candles_1h,
        asia_range_pct=0.8,
        asia_spike_dir="up",
        recent_trades=[],
        cycle_count=5,
        now=now,
    )

    # The critical features that must make it into the agent context
    assert ctx["symbol"] == "BNB/USDT"
    assert ctx["direction"] == "short"
    assert ctx["formation_variant"] == "three_hits_how"
    assert ctx["grade"] == "F"
    assert ctx["htf_trend_4h"] == "bullish"
    assert ctx["htf_4h_accel"] is True
    assert ctx["counter_trend"] is True  # short into bullish 4H
    assert ctx["weekly_phase"] == "BOARD_MEETING_2"
    # 4h candles present and renderable
    assert len(ctx["c4h_closes"]) > 0


def test_build_context_counter_trend_detection():
    """The engine pre-computes counter_trend so the LLM doesn't have to.
    Verify both directions + sideways returns False."""
    bf = SimpleNamespace(type="W", variant="standard", quality_score=0.5,
                         at_key_level=False, peak1_idx=0, peak2_idx=5)
    cf = SimpleNamespace(grade="B", score_pct=60, retest_conditions_met=3)
    session = SimpleNamespace(session_name="uk", minutes_into_session=0)
    cycle = SimpleNamespace(phase="BOARD_MEETING_1")
    import pandas as pd
    from datetime import datetime, timezone
    cs = pd.DataFrame({"open":[1]*3,"high":[1]*3,"low":[1]*3,"close":[1]*3,"volume":[1]*3})
    now = datetime(2026, 4, 17, tzinfo=timezone.utc)

    # Long into bearish 4H = counter-trend
    ctx = build_context(
        symbol="X", trade_direction="long", best_formation=bf, confluence_result=cf,
        entry_price=1, sl_ref=1,
        trend_state_4h=SimpleNamespace(direction="bearish", strength=0.7,
                                       is_accelerating=False),
        trend_state_1d=None, ema_state=None, ema_values={},
        session=session, cycle_state=cycle,
        candles_4h=cs, candles_1h=cs, candles_15m=cs,
        asia_range_pct=None, asia_spike_dir=None,
        recent_trades=None, cycle_count=0, now=now,
    )
    assert ctx["counter_trend"] is True

    # Long into bullish 4H = aligned
    ctx = build_context(
        symbol="X", trade_direction="long", best_formation=bf, confluence_result=cf,
        entry_price=1, sl_ref=1,
        trend_state_4h=SimpleNamespace(direction="bullish", strength=0.7,
                                       is_accelerating=False),
        trend_state_1d=None, ema_state=None, ema_values={},
        session=session, cycle_state=cycle,
        candles_4h=cs, candles_1h=cs, candles_15m=cs,
        asia_range_pct=None, asia_spike_dir=None,
        recent_trades=None, cycle_count=0, now=now,
    )
    assert ctx["counter_trend"] is False

    # Sideways 4H = not counter-trend regardless of direction
    for direction in ("long", "short"):
        ctx = build_context(
            symbol="X", trade_direction=direction, best_formation=bf,
            confluence_result=cf, entry_price=1, sl_ref=1,
            trend_state_4h=SimpleNamespace(direction="sideways", strength=0.1,
                                           is_accelerating=False),
            trend_state_1d=None, ema_state=None, ema_values={},
            session=session, cycle_state=cycle,
            candles_4h=cs, candles_1h=cs, candles_15m=cs,
            asia_range_pct=None, asia_spike_dir=None,
            recent_trades=None, cycle_count=0, now=now,
        )
        assert ctx["counter_trend"] is False


# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------
def test_agent_verdict_defaults():
    v = AgentVerdict(
        decision="APPROVE", reason="", confidence=0.5,
        htf_trend_4h="sideways", htf_trend_1d="sideways",
        counter_trend=False,
    )
    assert v.concerns == []
    assert v.model == ""
    assert v.cost_usd == 0.0
