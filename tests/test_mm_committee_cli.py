"""Claude Code CLI backend for the committee — selection, parsing, failure paths."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.strategy import mm_committee as mm_committee_mod
from src.strategy.mm_claude_cli import ClaudeCLIClient, ClaudeCLIError
from src.strategy.mm_committee import MMCommittee


def _config(api_key: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        anthropic_api_key=api_key,
        mm_committee_enabled=True,
        mm_committee_mode="shadow",
        mm_committee_timeout_s=5.0,
        mm_committee_cli_enabled=True,
        mm_committee_cli_path="",
        mm_committee_cli_timeout_s=120.0,
        mm_sanity_agent_cache_ttl_seconds=0.0,
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
        "entry_price": 100.0,
        "grade": "B",
        "score_pct": 62.0,
        "htf_trend_4h": "bullish",
        "htf_trend_1d": "bullish",
        "counter_trend": False,
    }


def _cli_result_json(
    result: str,
    *,
    is_error: bool = False,
    subtype: str = "success",
) -> bytes:
    return json.dumps(
        {
            "type": "result",
            "subtype": subtype,
            "is_error": is_error,
            "result": result,
            "total_cost_usd": 0.0,
            "usage": {
                "input_tokens": 120,
                "output_tokens": 30,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
    ).encode()


class _FakeProc:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self, input=None):
        return self._stdout, self._stderr

    def kill(self) -> None:
        pass


def test_no_key_selects_cli_backend(monkeypatch):
    monkeypatch.setattr(
        mm_committee_mod, "find_claude_cli", lambda explicit="": "/fake/claude"
    )
    agent = MMCommittee(config=_config(api_key=""), repo=AsyncMock())
    client = agent._get_client()
    assert isinstance(client, ClaudeCLIClient)
    assert client.cli_path == "/fake/claude"
    assert client.timeout_s == 120.0


def test_cli_disabled_via_config_returns_none(monkeypatch):
    monkeypatch.setattr(
        mm_committee_mod, "find_claude_cli", lambda explicit="": "/fake/claude"
    )
    cfg = _config(api_key="")
    cfg.mm_committee_cli_enabled = False
    agent = MMCommittee(config=cfg, repo=AsyncMock())
    assert agent._get_client() is None


async def test_no_key_no_cli_logs_client_unavailable(monkeypatch):
    monkeypatch.setattr(
        mm_committee_mod, "find_claude_cli", lambda explicit="": None
    )
    repo = AsyncMock()
    agent = MMCommittee(config=_config(api_key=""), repo=repo)

    verdict = await agent.review(_ctx())

    assert verdict is not None
    assert verdict.decision == "APPROVE"  # shadow mode masks the error VETO
    row = repo.insert_mm_agent_decision.call_args.args[0]
    assert row["decision"] == "ERROR"
    assert "client_unavailable" in row["reason"]


async def test_cli_call_parses_success_json(monkeypatch):
    captured_argv: list[str] = []

    async def fake_exec(*argv, **kwargs):
        captured_argv.extend(argv)
        return _FakeProc(_cli_result_json('{"decision": "APPROVE"}'))

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    client = ClaudeCLIClient("/fake/claude", timeout_s=10.0)

    raw, usage = await client.call(
        "claude-haiku-4-5-20251001", "system prompt", "user prompt"
    )

    assert raw == '{"decision": "APPROVE"}'
    assert usage["backend"] == "claude_cli"
    assert usage["input_tokens"] == 120
    assert "--system-prompt" in captured_argv
    assert "--no-session-persistence" in captured_argv


async def test_cli_call_scrubs_harness_env(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/tmp/sandbox-ca.pem")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://harness.example")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "abc")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "ambient-token")
    captured_env: dict = {}

    async def fake_exec(*argv, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return _FakeProc(_cli_result_json("ok"))

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    client = ClaudeCLIClient("/fake/claude", timeout_s=10.0)

    await client.call("claude-haiku-4-5-20251001", "sys", "user")

    assert "SSL_CERT_FILE" not in captured_env
    assert "ANTHROPIC_BASE_URL" not in captured_env
    assert "CLAUDECODE" not in captured_env
    assert "CLAUDE_CODE_SESSION_ID" not in captured_env
    assert captured_env["CLAUDE_CODE_OAUTH_TOKEN"] == "ambient-token"  # kept
    assert "PATH" in captured_env  # rest of the environment is inherited


async def test_cli_call_injects_configured_oauth_token(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    captured_env: dict = {}

    async def fake_exec(*argv, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return _FakeProc(_cli_result_json("ok"))

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    client = ClaudeCLIClient(
        "/fake/claude", timeout_s=10.0, oauth_token="from-dotenv"
    )

    await client.call("claude-haiku-4-5-20251001", "sys", "user")

    assert captured_env["CLAUDE_CODE_OAUTH_TOKEN"] == "from-dotenv"


def test_get_client_passes_configured_token(monkeypatch):
    monkeypatch.setattr(
        mm_committee_mod, "find_claude_cli", lambda explicit="": "/fake/claude"
    )
    cfg = _config(api_key="")
    cfg.claude_code_oauth_token = "from-dotenv"
    agent = MMCommittee(config=cfg, repo=AsyncMock())
    client = agent._get_client()
    assert isinstance(client, ClaudeCLIClient)
    assert client.oauth_token == "from-dotenv"


async def test_cli_call_raises_on_is_error_flag(monkeypatch):
    async def fake_exec(*argv, **kwargs):
        # Observed live 2026-07-18: exit code 0, subtype "success", but
        # is_error=true with the failure message in `result`.
        return _FakeProc(_cli_result_json("API Error: something", is_error=True))

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    client = ClaudeCLIClient("/fake/claude", timeout_s=10.0)

    with pytest.raises(ClaudeCLIError, match="cli_result_error"):
        await client.call("claude-haiku-4-5-20251001", "sys", "user")


async def test_cli_call_raises_on_nonzero_exit(monkeypatch):
    async def fake_exec(*argv, **kwargs):
        return _FakeProc(b"", stderr=b"boom", returncode=1)

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    client = ClaudeCLIClient("/fake/claude", timeout_s=10.0)

    with pytest.raises(ClaudeCLIError, match="cli_exit_1"):
        await client.call("claude-haiku-4-5-20251001", "sys", "user")


async def test_cli_failure_in_review_logs_api_error(monkeypatch):
    repo = AsyncMock()
    agent = MMCommittee(config=_config(api_key=""), repo=repo)
    client = ClaudeCLIClient("/fake/claude", timeout_s=10.0)
    agent._client = client

    async def raise_cli_error(*args, **kwargs):
        raise ClaudeCLIError("cli_exit_1: boom")

    monkeypatch.setattr(client, "call", raise_cli_error)
    monkeypatch.setattr(agent, "_load_skill", lambda name: "skill text")

    verdict = await agent.review(_ctx())

    assert verdict is not None
    assert verdict.decision == "APPROVE"  # shadow mode masks the error VETO
    row = repo.insert_mm_agent_decision.call_args.args[0]
    assert row["decision"] == "ERROR"
    assert "api_error" in row["reason"]


def test_compute_cost_cli_backend_skips_pricing_table():
    agent = MMCommittee(config=_config(api_key=""), repo=AsyncMock())
    assert agent._compute_cost(
        "claude-haiku-4-5-20251001",
        {"backend": "claude_cli", "total_cost_usd": 0.0},
    ) == 0.0
    assert agent._compute_cost(
        "some-unknown-model",
        {"backend": "claude_cli", "total_cost_usd": 0.42},
    ) == 0.42


def test_total_timeout_scales_for_cli_backend():
    agent = MMCommittee(config=_config(api_key=""), repo=AsyncMock())
    cli = ClaudeCLIClient("/fake/claude", timeout_s=120.0)
    assert agent._total_timeout_s(cli) == 270.0
    assert agent._total_timeout_s(object()) == 5.0
