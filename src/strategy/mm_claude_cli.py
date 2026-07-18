"""Claude Code CLI backend for the MM Agent Committee.

Routes committee model calls through the locally-authenticated Claude Code
CLI (``claude -p``) so the LLM layer runs on the user's Claude subscription
instead of a metered ``ANTHROPIC_API_KEY``. Selected automatically by
``MMCommittee._get_client()`` when no API key is configured (see
docs/MM_AGENT_COMMITTEE_DESIGN.md — the committee was specified to run on
the Claude Code runtime from the start).

Contract: ``call(model, system, user_prompt, max_tokens)`` returns the same
``(raw_text, usage_dict)`` tuple that ``MMCommittee._call_model`` produces
with the SDK client. ``usage_dict["backend"] == "claude_cli"`` marks the
transport so cost accounting skips the per-token pricing table (subscription
runs report ``total_cost_usd`` directly, usually 0).
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Locations tried after PATH lookup fails. The bot may be launched from a
# context (launchd, cron) whose PATH lacks Homebrew.
_CLI_FALLBACK_PATHS = (
    "/opt/homebrew/bin/claude",
    "/usr/local/bin/claude",
)

# Stripped from the child environment. Observed 2026-07-18: a harness-injected
# SSL_CERT_FILE (pointing at a sandbox-proxy CA) made the child CLI fail TLS
# against api.anthropic.com ("SSL certificate verification failed") after a
# ~3-minute retry loop, and nested Claude Code sessions inject session markers
# (CLAUDECODE, CLAUDE_CODE_SDK_HAS_OAUTH_REFRESH, ANTHROPIC_BASE_URL, ...)
# that make a child CLI behave as a host-managed subprocess instead of a
# standalone subscription login. The bot's own environment never needs these.
_ENV_STRIP = ("SSL_CERT_FILE", "SSL_CERT_DIR", "ANTHROPIC_BASE_URL", "CLAUDECODE")
# CLAUDE_CODE_OAUTH_TOKEN is the supported headless auth method (from
# `claude setup-token`) and is deliberately preserved / injectable.
_ENV_STRIP_PREFIX = "CLAUDE_CODE_"
_ENV_KEEP = ("CLAUDE_CODE_OAUTH_TOKEN",)


class ClaudeCLIError(RuntimeError):
    """Raised when the CLI subprocess fails, times out, or reports an error."""


def find_claude_cli(explicit_path: str = "") -> str | None:
    """Resolve the ``claude`` binary. Returns None when not installed."""
    if explicit_path:
        return explicit_path if shutil.which(explicit_path) else None
    found = shutil.which("claude")
    if found:
        return found
    for candidate in _CLI_FALLBACK_PATHS:
        if shutil.which(candidate):
            return candidate
    return None


class ClaudeCLIClient:
    """Minimal async wrapper around ``claude -p --output-format json``.

    One subprocess per model call. ``--tools ""`` and ``--strict-mcp-config``
    make it a pure LLM call (no agentic tools, no MCP servers), and the
    neutral cwd keeps the repo's CLAUDE.md out of the context. Session
    persistence is disabled so committee calls don't accumulate session files.
    """

    backend = "claude_cli"

    def __init__(
        self,
        cli_path: str,
        timeout_s: float = 120.0,
        oauth_token: str = "",
    ) -> None:
        self.cli_path = cli_path
        self.timeout_s = float(timeout_s)
        # Long-lived token from `claude setup-token` (config
        # claude_code_oauth_token / env CLAUDE_CODE_OAUTH_TOKEN in .env).
        # pydantic loads .env into Settings, not os.environ, so the token
        # must be injected into the child env explicitly here.
        self.oauth_token = oauth_token or ""

    def _child_env(self) -> dict[str, str]:
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in _ENV_STRIP
            and (not k.startswith(_ENV_STRIP_PREFIX) or k in _ENV_KEEP)
        }
        if self.oauth_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self.oauth_token
        return env

    async def call(
        self,
        model: str,
        system: str,
        user_prompt: str,
        *,
        max_tokens: int = 1024,  # noqa: ARG002 — CLI has no max-tokens control
    ) -> tuple[str, dict]:
        argv = [
            self.cli_path,
            "-p",
            "--output-format", "json",
            "--model", model,
            "--system-prompt", system,
            "--tools", "",
            "--strict-mcp-config",
            "--no-session-persistence",
        ]
        env = self._child_env()
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tempfile.gettempdir(),
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(user_prompt.encode("utf-8")),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise ClaudeCLIError(f"cli_timeout_after_{self.timeout_s:.0f}s") from None

        if proc.returncode != 0:
            snippet = (stderr or stdout or b"")[:300].decode("utf-8", "replace")
            raise ClaudeCLIError(f"cli_exit_{proc.returncode}: {snippet}")

        try:
            payload = json.loads(stdout.decode("utf-8", "replace"))
        except json.JSONDecodeError as e:
            snippet = stdout[:300].decode("utf-8", "replace")
            raise ClaudeCLIError(f"cli_bad_json: {snippet}") from e

        # The CLI exits 0 even for API-level failures — the error is flagged
        # inside the JSON (observed live: is_error=true with subtype
        # "success" on an SSL failure). Check the flag, not the exit code.
        result_text = str(payload.get("result") or "")
        if payload.get("is_error") or payload.get("subtype") != "success":
            raise ClaudeCLIError(
                f"cli_result_error[{payload.get('subtype')}]: {result_text[:300]}"
            )

        raw_usage = payload.get("usage") or {}
        usage = {
            "input_tokens": int(raw_usage.get("input_tokens") or 0),
            "output_tokens": int(raw_usage.get("output_tokens") or 0),
            "cache_creation_input_tokens": int(
                raw_usage.get("cache_creation_input_tokens") or 0
            ),
            "cache_read_input_tokens": int(
                raw_usage.get("cache_read_input_tokens") or 0
            ),
            "backend": self.backend,
            "total_cost_usd": float(payload.get("total_cost_usd") or 0.0),
        }
        return result_text, usage
