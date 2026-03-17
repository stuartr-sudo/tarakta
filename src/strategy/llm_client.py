"""Unified LLM client — supports Gemini and OpenAI with a single interface.

All agents call `generate_json()` which routes to the correct provider
based on the model name. Client instances are cached per (provider, api_key).
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


def is_openai_model(model: str) -> bool:
    """Return True if model name belongs to OpenAI."""
    m = model.lower()
    return any(m.startswith(p) for p in _OPENAI_PREFIXES)


# ---------------------------------------------------------------------------
# Pricing per 1M tokens: (input, cached_input, output)
# ---------------------------------------------------------------------------
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    # Gemini
    "gemini-3-pro-preview": (1.25, 0.125, 10.00),
    "gemini-3-flash-preview": (0.10, 0.01, 0.40),
    "gemini-3.1-pro": (1.25, 0.125, 10.00),
    # OpenAI
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-5": (2.00, 0.50, 8.00),
    "gpt-5-mini": (0.40, 0.10, 1.60),
    "o3-mini": (1.10, 0.55, 4.40),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class LLMResult:
    """Normalized result from any LLM provider."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Client cache
# ---------------------------------------------------------------------------
_gemini_clients: dict[str, Any] = {}
_openai_clients: dict[str, Any] = {}


def _get_gemini_client(api_key: str):
    if api_key not in _gemini_clients:
        from google import genai
        _gemini_clients[api_key] = genai.Client(api_key=api_key)
    return _gemini_clients[api_key]


def _get_openai_client(api_key: str):
    if api_key not in _openai_clients:
        from openai import AsyncOpenAI
        _openai_clients[api_key] = AsyncOpenAI(
            api_key=api_key,
            timeout=180.0,  # 3-min hard cap — GPT-5 can be slow with thinking
        )
    return _openai_clients[api_key]


# ---------------------------------------------------------------------------
# Schema patching for OpenAI strict mode
# ---------------------------------------------------------------------------
def _patch_schema_for_openai(schema: dict) -> dict:
    """OpenAI strict mode requires additionalProperties: false on all objects."""
    schema = copy.deepcopy(schema)

    def _patch(node: dict) -> None:
        if node.get("type") == "object":
            node["additionalProperties"] = False
            for prop in node.get("properties", {}).values():
                if isinstance(prop, dict):
                    _patch(prop)
        if node.get("type") == "array" and isinstance(node.get("items"), dict):
            _patch(node["items"])

    _patch(schema)
    return schema


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def generate_json(
    *,
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: dict,
    thinking_level: str = "low",
    temperature: float = 1.0,
    timeout: float = 60.0,
) -> LLMResult:
    """Call an LLM and return structured JSON text.

    Automatically routes to Gemini or OpenAI based on model name.
    """
    if is_openai_model(model):
        return await _call_openai(
            model=model, api_key=api_key,
            system_prompt=system_prompt, user_prompt=user_prompt,
            json_schema=json_schema, temperature=temperature,
            timeout=timeout,
        )
    else:
        return await _call_gemini(
            model=model, api_key=api_key,
            system_prompt=system_prompt, user_prompt=user_prompt,
            json_schema=json_schema, thinking_level=thinking_level,
            temperature=temperature,
        )


# ---------------------------------------------------------------------------
# Gemini implementation
# ---------------------------------------------------------------------------
async def _call_gemini(
    *, model: str, api_key: str, system_prompt: str, user_prompt: str,
    json_schema: dict, thinking_level: str, temperature: float,
) -> LLMResult:
    import asyncio
    from google.genai import types as _gentypes

    client = _get_gemini_client(api_key)
    response = await client.aio.models.generate_content(
        model=model,
        contents=user_prompt,
        config=_gentypes.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=_gentypes.ThinkingConfig(thinking_level=thinking_level),
            response_mime_type="application/json",
            response_json_schema=json_schema,
            temperature=temperature,
        ),
    )

    usage = response.usage_metadata
    return LLMResult(
        text=getattr(response, "text", "") or "",
        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
    )


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------
async def _call_openai(
    *, model: str, api_key: str, system_prompt: str, user_prompt: str,
    json_schema: dict, temperature: float, timeout: float,
) -> LLMResult:
    client = _get_openai_client(api_key)
    patched = _patch_schema_for_openai(json_schema)

    # asyncio.wait_for enforces a HARD total-request timeout.
    # The SDK's httpx timeout only limits time-between-bytes, which
    # doesn't fire when GPT-5 sends keepalive bytes during thinking.
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": patched,
                },
            },
            temperature=temperature,
        ),
        timeout=timeout,
    )

    text = response.choices[0].message.content or ""
    usage = response.usage
    return LLMResult(
        text=text,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )
