"""Shared helper: call a vision-capable Claude model on the subscription CLI
with the Read tool enabled so it can open chart images."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, "/Users/stuarta/tarakta")

from src.config import Settings  # noqa: E402

_STRIP = ("SSL_CERT_FILE", "SSL_CERT_DIR", "ANTHROPIC_BASE_URL", "CLAUDECODE")


def _cli() -> str:
    return shutil.which("claude") or "/opt/homebrew/bin/claude"


def _env(token: str) -> dict:
    env = {
        k: v for k, v in os.environ.items()
        if k not in _STRIP
        and (not k.startswith("CLAUDE_CODE_") or k == "CLAUDE_CODE_OAUTH_TOKEN")
    }
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    return env


async def vision_call(
    system: str,
    prompt: str,
    *,
    model: str = "claude-haiku-4-5-20251001",
    timeout_s: float = 180.0,
    token: str | None = None,
) -> tuple[str, float]:
    """Returns (result_text, cli_reported_cost). Raises RuntimeError on error."""
    if token is None:
        token = Settings().claude_code_oauth_token
    argv = [
        _cli(), "-p", "--output-format", "json", "--model", model,
        "--tools", "Read", "--strict-mcp-config", "--no-session-persistence",
        "--system-prompt", system,
    ]
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=tempfile.gettempdir(),
        env=_env(token),
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(prompt.encode()), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"vision_call timeout after {timeout_s}s") from None
    if proc.returncode != 0:
        raise RuntimeError(f"cli exit {proc.returncode}: {(err or out)[:200]!r}")
    payload = json.loads(out.decode("utf-8", "replace"))
    if payload.get("is_error") or payload.get("subtype") != "success":
        raise RuntimeError(f"cli error: {str(payload.get('result'))[:200]}")
    return str(payload.get("result") or ""), float(payload.get("total_cost_usd") or 0)


def extract_json(raw: str) -> dict | None:
    text = raw.strip()
    dec = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            data, _ = dec.raw_decode(text[idx:])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
    return None
