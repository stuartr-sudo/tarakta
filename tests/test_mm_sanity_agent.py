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

import time
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


# ---------------------------------------------------------------------------
# Tier 2 learning loop (2026-04-21)
# ---------------------------------------------------------------------------

class TestUserPromptOutcomeStats:
    """The user prompt must render the RECENT PROFILE OUTCOMES block
    when outcome stats are attached to context, and mark the THIS
    PROFILE row for the current (grade, htf_4h)."""

    def _agent(self):
        return MMSanityAgent(config=SimpleNamespace(), repo=MagicMock())

    def test_no_stats_renders_insufficient_data_placeholder(self):
        """Empty stats render the unified insufficient-data fallback
        (rubric_v=3 collapsed the old 'first pass' branch into this
        single message)."""
        ctx = {
            "symbol": "BNB/USDT",
            "grade": "C",
            "htf_trend_4h": "sideways",
            "_outcome_stats": {},
            "_outcome_lookback_days": 14,
        }
        p = self._agent()._build_user_prompt(ctx)
        assert "RECENT PROFILE OUTCOMES" in p
        assert "insufficient data" in p.lower()
        assert "skip Rubric 8" in p
        assert "THIS setup's profile: C|sideways" in p

    def test_stats_rendered_with_this_profile_marker(self):
        """With enough closed samples per bucket, both render and the
        current setup's profile is flagged ← THIS PROFILE. Sample sizes
        here are realistic for rubric_v=3 (min_n=20)."""
        stats = {
            "C|sideways": {
                "n": 24, "wins": 6, "losses": 14, "scratches": 2, "opens": 2,
                "net_pnl_usd": -163.61, "avg_pnl_usd": -7.44,
                "sample_trades": ["BNB/USDT", "NEAR/USDT", "AVAX/USDT", "DOGE/USDT"],
            },
            "F|sideways": {
                "n": 22, "wins": 8, "losses": 10, "scratches": 4, "opens": 0,
                "net_pnl_usd": -9.00, "avg_pnl_usd": -0.41,
                "sample_trades": ["BNB/USDT", "BTC/USDT"],
            },
        }
        ctx = {
            "symbol": "DOGE/USDT",
            "grade": "C",
            "htf_trend_4h": "sideways",
            "_outcome_stats": stats,
            "_outcome_lookback_days": 14,
        }
        p = self._agent()._build_user_prompt(ctx)
        assert "C|sideways" in p
        assert "F|sideways" in p
        assert "net=$-163.61" in p
        assert "net=$-9.00" in p
        # The current setup's profile must be marked so the agent sees itself
        assert "C|sideways" in p and "← THIS PROFILE" in p
        # And the unrelated F profile must NOT have the marker
        lines = p.split("\n")
        c_line = next(line for line in lines if "C|sideways" in line and "net=" in line)
        f_line = next(line for line in lines if "F|sideways" in line and "net=" in line)
        assert "← THIS PROFILE" in c_line
        assert "← THIS PROFILE" not in f_line

    def test_prompt_version_bumped_to_rubric_v3(self):
        """rubric_v=3 added engine-side min-n filter + removed over-eager
        n>=3 threshold. Version must bump so agents don't silently reuse
        cached rubric_v=2."""
        assert "rubric_v=3" in PROMPT_VERSION
        assert "prompt_v=3" in PROMPT_VERSION


def test_system_prompt_has_rubric_8():
    """The outcome-aware rubric point must be present in the system
    prompt, otherwise the agent won't know what to do with the stats
    block in the user prompt."""
    assert "8. Your own track record" in SYSTEM_PROMPT
    # Must mention confidence floor for VETO on this rubric
    assert "0.85" in SYSTEM_PROMPT
    # And the concern tag we expect vetoes to use
    assert "recent_losses" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tier 2 min-n filter (2026-04-22, rubric_v=3): block buckets that don't
# carry statistical signal so the agent doesn't VETO on 0W/1L noise.
# See docs/MM_SANITY_AGENT_DESIGN.md; triggered by v44 regression.
# ---------------------------------------------------------------------------

class TestUserPromptOutcomeStatsFilter:
    """Rubric 8 only reaches the model for buckets with enough closed
    samples to distinguish regime signal from variance. Threshold is
    ``mm_sanity_agent_outcome_min_n`` on config (default 20)."""

    def _agent(self, **config_kwargs):
        return MMSanityAgent(
            config=SimpleNamespace(**config_kwargs), repo=MagicMock()
        )

    def test_bucket_below_min_n_is_filtered_out(self):
        stats = {
            "C|sideways": {  # 4 closed — below threshold, must be filtered
                "n": 5, "wins": 1, "losses": 2, "scratches": 1, "opens": 1,
                "net_pnl_usd": -40.0,
            },
            "C|bullish": {   # 25 closed — passes threshold
                "n": 25, "wins": 15, "losses": 8, "scratches": 2, "opens": 0,
                "net_pnl_usd": 125.50,
            },
        }
        ctx = {"symbol": "X", "grade": "C", "htf_trend_4h": "bullish",
               "_outcome_stats": stats, "_outcome_lookback_days": 14}
        p = self._agent(mm_sanity_agent_outcome_min_n=20)._build_user_prompt(ctx)
        assert "C|sideways" not in p, "below-threshold bucket leaked into prompt"
        assert "C|bullish" in p
        assert "net=$+125.50" in p

    def test_all_buckets_below_min_n_renders_insufficient_data(self):
        stats = {
            "C|sideways": {"n": 5, "wins": 1, "losses": 2, "scratches": 1,
                           "opens": 1, "net_pnl_usd": -40.0},
            "F|sideways": {"n": 1, "wins": 0, "losses": 1, "scratches": 0,
                           "opens": 0, "net_pnl_usd": -12.0},
        }
        ctx = {"symbol": "X", "grade": "C", "htf_trend_4h": "sideways",
               "_outcome_stats": stats, "_outcome_lookback_days": 14}
        p = self._agent(mm_sanity_agent_outcome_min_n=20)._build_user_prompt(ctx)
        # No bucket rows
        assert "net=$" not in p
        # Explicit fallback so the model knows to skip Rubric 8
        assert "insufficient data" in p.lower()
        assert "skip Rubric 8" in p
        # The active threshold surfaces so decisions are reproducible
        assert ">=20" in p

    def test_empty_stats_renders_insufficient_data(self):
        """Empty stats (first pass, before any approvals closed) takes the
        same path as all-filtered — a single unified fallback message."""
        ctx = {"symbol": "X", "grade": "C", "htf_trend_4h": "sideways",
               "_outcome_stats": {}, "_outcome_lookback_days": 14}
        p = self._agent(mm_sanity_agent_outcome_min_n=20)._build_user_prompt(ctx)
        assert "insufficient data" in p.lower()
        assert "skip Rubric 8" in p

    def test_min_n_is_configurable(self):
        stats = {
            "C|sideways": {"n": 10, "wins": 4, "losses": 4, "scratches": 2,
                           "opens": 0, "net_pnl_usd": 5.0},
        }
        ctx = {"symbol": "X", "grade": "C", "htf_trend_4h": "sideways",
               "_outcome_stats": stats, "_outcome_lookback_days": 14}
        # n_closed = 10; passes when min_n=5 → bucket row renders (has net=$)
        p_low = self._agent(mm_sanity_agent_outcome_min_n=5)._build_user_prompt(ctx)
        assert "net=$+5.00" in p_low
        # Same bucket filtered out when min_n=20 → no bucket rows, fallback shown
        p_high = self._agent(mm_sanity_agent_outcome_min_n=20)._build_user_prompt(ctx)
        assert "net=$" not in p_high
        assert "insufficient data" in p_high.lower()

    def test_opens_do_not_count_toward_min_n(self):
        """Open trades have no pnl yet — they cannot contribute to a
        profile's win/loss signal, so they must not satisfy the filter."""
        stats = {
            "C|bullish": {  # 3 closed + 20 open; only 3 count
                "n": 23, "wins": 1, "losses": 2, "scratches": 0, "opens": 20,
                "net_pnl_usd": -5.0,
            },
        }
        ctx = {"symbol": "X", "grade": "C", "htf_trend_4h": "bullish",
               "_outcome_stats": stats, "_outcome_lookback_days": 14}
        p = self._agent(mm_sanity_agent_outcome_min_n=20)._build_user_prompt(ctx)
        # No bucket row rendered — only 3 closed samples, below threshold
        assert "net=$" not in p
        assert "insufficient data" in p.lower()

    def test_default_min_n_when_config_missing(self):
        """If the config namespace lacks the attr, default 20 applies —
        this is the behaviour in production until a tuning knob is set."""
        stats = {
            "C|sideways": {"n": 6, "wins": 1, "losses": 4, "scratches": 0,
                           "opens": 1, "net_pnl_usd": -159.0},
        }
        ctx = {"symbol": "X", "grade": "C", "htf_trend_4h": "sideways",
               "_outcome_stats": stats, "_outcome_lookback_days": 14}
        # Bare SimpleNamespace — no min_n attr. n_closed=5, below default 20.
        agent = MMSanityAgent(config=SimpleNamespace(), repo=MagicMock())
        p = agent._build_user_prompt(ctx)
        # No bucket row; insufficient-data fallback with the default threshold
        assert "net=$" not in p
        assert "insufficient data" in p.lower()
        assert ">=20" in p


# ---------------------------------------------------------------------------
# Per-setup decision cache (P2 fix 2026-04-22)
# ---------------------------------------------------------------------------
# Context: the 1H formation detector re-generates the same setup every 5-min
# scan. Pre-fix, each rescan billed a full Opus 4.7 call for zero new info —
# 82 calls on the same DOGE long setup over 6h on 2026-04-21. The cache
# collapses this into one call per 30-min window (+ invalidation on price
# drift / setup change).
# ---------------------------------------------------------------------------

class TestCacheKey:
    """_cache_key must be None-safe for incomplete contexts and stable
    for complete ones. A None return means "don't cache this call" —
    safer than caching partial data that might collide with a later
    setup."""

    def _agent(self):
        return MMSanityAgent(config=SimpleNamespace(), repo=MagicMock())

    def test_complete_context_returns_tuple(self):
        k = self._agent()._cache_key({
            "symbol": "DOGE/USDT",
            "direction": "long",
            "formation_variant": "standard",
            "entry_price": 0.156789,
        })
        # entry rounded to 4dp so tick-noise doesn't fragment the cache
        assert k == ("DOGE/USDT", "long", "standard", 0.1568)

    def test_missing_symbol_returns_none(self):
        assert self._agent()._cache_key({
            "direction": "long",
            "formation_variant": "standard",
            "entry_price": 1.0,
        }) is None

    def test_missing_direction_returns_none(self):
        assert self._agent()._cache_key({
            "symbol": "X",
            "formation_variant": "standard",
            "entry_price": 1.0,
        }) is None

    def test_missing_variant_returns_none(self):
        assert self._agent()._cache_key({
            "symbol": "X",
            "direction": "long",
            "entry_price": 1.0,
        }) is None

    def test_zero_entry_returns_none(self):
        assert self._agent()._cache_key({
            "symbol": "X",
            "direction": "long",
            "formation_variant": "s",
            "entry_price": 0.0,
        }) is None

    def test_negative_entry_returns_none(self):
        assert self._agent()._cache_key({
            "symbol": "X",
            "direction": "long",
            "formation_variant": "s",
            "entry_price": -1.0,
        }) is None

    def test_invalid_entry_returns_none(self):
        assert self._agent()._cache_key({
            "symbol": "X",
            "direction": "long",
            "formation_variant": "s",
            "entry_price": "not-a-number",
        }) is None

    def test_4dp_rounding_collapses_tick_noise(self):
        """Two setups with entry_price differing only in the 5th decimal
        place must collapse to the same cache key — otherwise the
        detector's floating-point drift would fragment the cache."""
        agent = self._agent()
        k1 = agent._cache_key({
            "symbol": "X", "direction": "long",
            "formation_variant": "s", "entry_price": 0.150001,
        })
        k2 = agent._cache_key({
            "symbol": "X", "direction": "long",
            "formation_variant": "s", "entry_price": 0.150004,
        })
        assert k1 == k2


def _cache_agent(**cfg_over):
    """Build an MMSanityAgent wired for cache tests.

    Mocks out the real Anthropic client and the `_call_model` network
    call. Returns (agent, repo, call_counter) where call_counter lets
    tests assert whether the API was hit on a given review().
    """
    default_cfg = dict(
        mm_sanity_agent_enabled=True,
        anthropic_api_key="sk-test",
        mm_sanity_agent_model="claude-opus-4-7",
        mm_sanity_agent_fallback_model="claude-sonnet-4-6",
        # Disable the budget-cap auto-downgrade — it would call the DB
        mm_sanity_agent_monthly_budget_usd=0,
        mm_sanity_agent_timeout_s=20.0,
        mm_sanity_agent_effort="high",
        mm_sanity_agent_min_confidence=0.0,
        # Disable the learning-loop repo call — not under test here
        mm_sanity_agent_outcome_lookback_days=0,
        mm_sanity_agent_cache_ttl_seconds=1800.0,
        mm_sanity_agent_cache_price_drift_pct=0.5,
    )
    default_cfg.update(cfg_over)
    config = SimpleNamespace(**default_cfg)

    repo = AsyncMock()
    repo.get_mm_agent_month_cost = AsyncMock(return_value=0.0)
    repo.get_mm_agent_outcome_stats = AsyncMock(return_value={})
    repo.insert_mm_agent_decision = AsyncMock(return_value=None)

    agent = MMSanityAgent(config=config, repo=repo)
    # Bypass the real anthropic SDK entirely
    agent._client = object()

    call_counter = {"count": 0, "return": "veto"}

    async def stub_call_model(client, model, user_prompt, effort):
        call_counter["count"] += 1
        if call_counter["return"] == "malformed":
            raw = "not valid json at all"
        else:
            raw = (
                '{"decision":"VETO","reason":"counter-trend per rubric 1",'
                '"confidence":0.9,"htf_trend_4h":"bullish",'
                '"htf_trend_1d":"bullish","counter_trend":true,'
                '"concerns":["4h_alignment"]}'
            )
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 10_000,
        }
        return raw, usage

    agent._call_model = stub_call_model
    return agent, repo, call_counter


BASE_CTX = {
    "symbol": "DOGE/USDT",
    "direction": "long",
    "formation_variant": "standard",
    "formation_type": "W",
    "entry_price": 0.1500,
    "sl_ref": 0.1470,
    "grade": "C",
    "score_pct": 40.0,
    "retest_met": 3,
    "htf_trend_4h": "sideways",
    "htf_trend_1d": "sideways",
    "cycle_count": 1,
}


class TestDecisionCache:
    """End-to-end cache behaviour through MMSanityAgent.review().

    Each test asserts against `call_counter` — our proxy for
    'did we pay for an API call?'. A cache hit should mean zero
    additional calls.
    """

    async def test_first_call_hits_api_and_populates_cache(self):
        agent, _repo, counter = _cache_agent()
        v = await agent.review(dict(BASE_CTX))
        assert v is not None
        assert v.decision == "VETO"
        assert counter["count"] == 1
        assert len(agent._decision_cache) == 1

    async def test_second_call_same_setup_hits_cache(self):
        """The core P2 guarantee: identical setup within TTL must NOT
        trigger a second API call."""
        agent, _repo, counter = _cache_agent()
        v1 = await agent.review(dict(BASE_CTX))
        v2 = await agent.review(dict(BASE_CTX))
        assert counter["count"] == 1, "second call must be served from cache"
        assert v2 is not None
        assert v2.decision == v1.decision
        # The cached copy must mark itself clearly so downstream audit
        # rows don't look like a fresh decision.
        assert v2.reason.startswith("[cached]")
        # Cached verdicts are free and instant
        assert v2.latency_ms == 0
        assert v2.cost_usd == 0.0

    async def test_cache_disabled_by_zero_ttl(self):
        """A cache_ttl_seconds of 0 must disable caching entirely —
        every review pays the full API cost. This is the kill switch."""
        agent, _repo, counter = _cache_agent(
            mm_sanity_agent_cache_ttl_seconds=0,
        )
        await agent.review(dict(BASE_CTX))
        await agent.review(dict(BASE_CTX))
        await agent.review(dict(BASE_CTX))
        assert counter["count"] == 3
        assert len(agent._decision_cache) == 0

    async def test_expired_ttl_forces_fresh_call(self):
        """Entries older than TTL must be evicted and refetched.
        Simulated by back-dating the cached timestamp."""
        agent, _repo, counter = _cache_agent(
            mm_sanity_agent_cache_ttl_seconds=1800.0,
        )
        await agent.review(dict(BASE_CTX))
        # Back-date the entry to 31 minutes ago (> 1800s TTL)
        key = agent._cache_key(BASE_CTX)
        verdict, _ts, entry = agent._decision_cache[key]
        agent._decision_cache[key] = (verdict, time.time() - 1860, entry)
        await agent.review(dict(BASE_CTX))
        assert counter["count"] == 2, "expired cache must force fresh call"

    async def test_different_variant_is_new_key(self):
        """A formation_variant change yields a different cache key —
        the two setups are distinct decisions and must not share a
        cached verdict."""
        agent, _repo, counter = _cache_agent()
        ctx_a = dict(BASE_CTX, formation_variant="standard")
        ctx_b = dict(BASE_CTX, formation_variant="three_hits_how")
        await agent.review(ctx_a)
        await agent.review(ctx_b)
        assert counter["count"] == 2
        assert len(agent._decision_cache) == 2

    async def test_different_symbol_is_new_key(self):
        agent, _repo, counter = _cache_agent()
        await agent.review(dict(BASE_CTX, symbol="DOGE/USDT"))
        await agent.review(dict(BASE_CTX, symbol="NEAR/USDT"))
        assert counter["count"] == 2

    async def test_direction_flip_is_new_key(self):
        """Same symbol + variant + entry but opposite direction is a
        genuinely different trade — must not share a cache entry."""
        agent, _repo, counter = _cache_agent()
        await agent.review(dict(BASE_CTX, direction="long"))
        await agent.review(dict(BASE_CTX, direction="short"))
        assert counter["count"] == 2

    async def test_entry_price_drift_invalidates_cache(self):
        """If the cached entry_price has drifted >0.5% from the current
        entry_price, the cached verdict is stale and must be refetched.

        Simulated by directly mutating the cached entry value (since
        the cache key's 4dp rounding makes a natural drift scenario
        hard to construct without changing the key)."""
        agent, _repo, counter = _cache_agent(
            mm_sanity_agent_cache_price_drift_pct=0.5,
        )
        await agent.review(dict(BASE_CTX))
        key = agent._cache_key(BASE_CTX)
        verdict, ts, _entry = agent._decision_cache[key]
        # Seed a "stale" cached entry 5% below current — far above the
        # 0.5% threshold
        stale_entry = BASE_CTX["entry_price"] * 0.95
        agent._decision_cache[key] = (verdict, ts, stale_entry)
        await agent.review(dict(BASE_CTX))
        assert counter["count"] == 2, "5% drift must force fresh call"

    async def test_malformed_response_not_cached(self):
        """If the API returns unparseable JSON, review() returns None
        (fail-open). That failure must NOT be cached — otherwise a
        transient bad response poisons the cache for the whole TTL."""
        agent, _repo, counter = _cache_agent()
        counter["return"] = "malformed"
        v = await agent.review(dict(BASE_CTX))
        assert v is None
        assert len(agent._decision_cache) == 0
        # A retry must be allowed to hit the API
        counter["return"] = "veto"
        v2 = await agent.review(dict(BASE_CTX))
        assert counter["count"] == 2
        assert v2 is not None

    async def test_kill_switch_bypasses_cache(self):
        """The kill switch must short-circuit BEFORE the cache is
        consulted — you don't want to be serving cached verdicts after
        you've disabled the agent."""
        agent, _repo, counter = _cache_agent()
        # Warm the cache
        await agent.review(dict(BASE_CTX))
        assert counter["count"] == 1
        # Kill the agent
        agent.config.mm_sanity_agent_enabled = False
        v = await agent.review(dict(BASE_CTX))
        assert v is None  # kill switch wins
        assert counter["count"] == 1  # no new call, but also no cached replay


class TestCacheConfigDefaults:
    """The cache config flags must exist with sane defaults. A regression
    that drops them silently reverts to un-cached (expensive) behaviour."""

    def test_default_ttl_is_30_minutes(self):
        from src.config import Settings
        s = Settings()
        assert s.mm_sanity_agent_cache_ttl_seconds == 1800.0

    def test_default_drift_is_half_percent(self):
        from src.config import Settings
        s = Settings()
        assert s.mm_sanity_agent_cache_price_drift_pct == 0.5
