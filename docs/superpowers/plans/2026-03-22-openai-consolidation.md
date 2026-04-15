# OpenAI Consolidation & Branch Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip Gemini/Anthropic LLM providers, consolidate to OpenAI-only with new model defaults (Agent 1: gpt-5.4-mini, Agent 2: gpt-5-mini, Agent 3: gpt-5.4-nano), fix Agent 2 model inheritance bug, and delete unused branches.

**Architecture:** Single LLM provider (OpenAI) with direct API key resolution. No provider routing. Lesson generator upgraded to gpt-5-mini for quality. All changes applied to `main` then cherry-picked to `expanded`.

**Tech Stack:** Python 3.12, OpenAI API, FastAPI, Jinja2, Supabase

**Spec:** `docs/superpowers/specs/2026-03-22-openai-consolidation-design.md`

---

### Task 1: Strip `llm_client.py` to OpenAI-Only

**Files:**
- Modify: `src/strategy/llm_client.py`

This is the foundation — everything else depends on this file being clean.

- [ ] **Step 1: Remove Gemini provider code**

Remove these functions and their imports entirely:
- `_ANTHROPIC_PREFIXES` (line 22)
- `is_anthropic_model()` (lines 31-34)
- `get_api_key_for_model()` (lines 37-44)
- `_gemini_clients` cache (line 87)
- `_get_gemini_client()` (lines 92-96)
- `_anthropic_clients` cache (line 89)
- `_get_anthropic_client()` (lines 109-116)
- `_call_gemini()` (lines 183-208)
- `_call_anthropic()` (lines 256-294)

Keep: `_OPENAI_PREFIXES`, `is_openai_model()` (still useful for validation), `_openai_clients`, `_get_openai_client()`, `_call_openai()`, `_patch_schema_for_openai()`, `LLMResult`

- [ ] **Step 2: Simplify `generate_json()` routing**

Replace the if/elif/else routing (lines 157-177) with direct call:

```python
async def generate_json(
    *,
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: dict,
    temperature: float = 1.0,
    timeout: float = 60.0,
) -> LLMResult:
    """Call OpenAI and return structured JSON text."""
    return await _call_openai(
        model=model, api_key=api_key,
        system_prompt=system_prompt, user_prompt=user_prompt,
        json_schema=json_schema, temperature=temperature,
        timeout=timeout,
    )
```

Note: Remove `thinking_level` parameter (Gemini-only concept).

- [ ] **Step 3: Update `MODEL_PRICING` to OpenAI-only**

Replace the entire `MODEL_PRICING` dict (lines 50-69) with:

```python
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
}
```

- [ ] **Step 4: Clean up imports**

The file should only import: `asyncio`, `copy`, `json`, `dataclass`, `Any`, and the logger. Remove any lazy imports of `google.genai` and `anthropic` that exist inside the deleted functions.

- [ ] **Step 5: Verify file compiles**

Run: `python -c "from src.strategy.llm_client import generate_json, MODEL_PRICING, LLMResult; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/strategy/llm_client.py
git commit -m "refactor: strip llm_client to OpenAI-only, remove Gemini/Anthropic providers"
```

---

### Task 2: Update `config.py` — New Defaults + Refiner Model Field

**Files:**
- Modify: `src/config.py`

- [ ] **Step 1: Remove Gemini/Anthropic key fields**

Remove these fields from the `Settings` class:
- `agent_api_key: str = ""` (line 139)
- `anthropic_api_key: str = ""` (line 137)

- [ ] **Step 2: Update model defaults**

Change:
- `agent_model: str = "gemini-3-pro-preview"` → `agent_model: str = "gpt-5.4-mini"` (line 140)
- `position_agent_model: str = "gemini-3-flash-preview"` → `position_agent_model: str = "gpt-5.4-nano"` (line 154)

- [ ] **Step 3: Add `refiner_agent_model` field**

Add after line 148 (`refiner_agent_enabled`):

```python
refiner_agent_model: str = "gpt-5-mini"  # Agent 2: tactical timing (separate from Agent 1)
```

- [ ] **Step 4: Update comments**

- Line 133-136: Replace the multi-provider routing comments with:
  ```python
  # AI Entry Agent (OpenAI models only)
  # All agents use OPENAI_API_KEY for authentication
  ```
- Line 147: Replace "Shares agent_model with Agent 1 (Gemini: auto-downgrades to flash)" with:
  ```python
  # Refiner Monitor Agent (Agent 2 — tactical entry timing on 5m candles)
  ```

- [ ] **Step 5: Verify config loads**

Run: `python -c "from src.config import Settings; s = Settings(); print(s.agent_model, s.refiner_agent_model, s.position_agent_model)"`
Expected: `gpt-5.4-mini gpt-5-mini gpt-5.4-nano`

- [ ] **Step 6: Commit**

```bash
git add src/config.py
git commit -m "refactor: add refiner_agent_model, update defaults to OpenAI models, remove Gemini/Anthropic keys"
```

---

### Task 3: Update Agent 1 (`agent_analyst.py`)

**Files:**
- Modify: `src/strategy/agent_analyst.py`

- [ ] **Step 1: Simplify imports**

Line 27-30: Replace with:
```python
from src.strategy.llm_client import (
    LLMResult, generate_json, MODEL_PRICING,
)
```

Remove `is_openai_model`, `is_anthropic_model`, `get_api_key_for_model` imports.

- [ ] **Step 2: Simplify `__init__` API key handling**

Lines 286-296: Replace multi-key storage with:
```python
self._api_key = config.openai_api_key
self._available = bool(self._api_key)
```

Remove: `self._gemini_api_key`, `self._openai_api_key`, `self._anthropic_api_key`, the `get_api_key_for_model()` call, and the `is_openai`/`is_anthropic`/`has_*_key` log fields.

- [ ] **Step 3: Simplify `set_model()`**

Lines 329-344: Replace the multi-provider key routing in `set_model()` with:
```python
def set_model(self, model: str) -> str:
    """Switch model at runtime. Returns the active model name."""
    if model not in self._pricing:
        raise ValueError(f"Unknown model: {model}. Available: {self.available_models}")
    old = self._model
    self._model = model
    logger.info("agent_model_switched", old_model=old, new_model=model)
    return self._model
```

- [ ] **Step 4: Remove `thinking_level` from `generate_json` call**

In `analyze_signal()` around line 407, the call to `generate_json()` passes `thinking_level=self._thinking_level`. Remove this parameter (it was Gemini-only and was removed from `generate_json` in Task 1).

- [ ] **Step 5: Update docstrings**

- Line 1: `"Agent 1 — AI-powered strategic entry agent using Gemini Interactions API."` → `"Agent 1 — AI-powered strategic entry agent."`
- Line 17: Remove `"Uses Gemini Interactions API for structured output with"` line
- Line 243: `"# Gemini structured output schema"` → `"# Structured output schema"`
- Line 273: `"Gemini-powered entry agent"` → `"AI-powered entry agent"`

- [ ] **Step 6: Commit**

```bash
git add src/strategy/agent_analyst.py
git commit -m "refactor: Agent 1 to OpenAI-only, remove multi-provider routing"
```

---

### Task 4: Fix Agent 2 (`refiner_agent.py`) — Model Inheritance Bug

**Files:**
- Modify: `src/strategy/refiner_agent.py`

This is the critical cost fix.

- [ ] **Step 1: Fix model assignment (the bug)**

Line 268: Replace:
```python
self._model = getattr(config, "agent_model", "gemini-3-flash-preview")
```
With:
```python
self._model = config.refiner_agent_model
```

- [ ] **Step 2: Remove Gemini auto-downgrade logic**

Lines 269-271: Delete entirely:
```python
# Agent 2 always uses flash for Gemini — override pro model
if not is_openai_model(self._model) and "pro" in self._model:
    self._model = "gemini-3-flash-preview"
```

- [ ] **Step 3: Simplify API key handling**

Lines 277-285: Replace with:
```python
self._api_key = config.openai_api_key
self._available = bool(self._api_key)
```

Remove: `self._gemini_api_key`, `self._openai_api_key`, `self._anthropic_api_key`, and the `get_api_key_for_model()` call.

- [ ] **Step 4: Simplify `set_model()` method**

Same pattern as Task 3 Step 3 — remove multi-provider key routing, just switch `self._model`.

- [ ] **Step 5: Remove Gemini fallback error handler**

Search for any error-handling block that falls back to `"gemini-3-flash-preview"` on OpenAI 400 errors. Remove the Gemini-specific fallback path entirely.

- [ ] **Step 6: Remove `thinking_level` from `generate_json` call**

Same as Task 3 Step 4.

- [ ] **Step 7: Simplify imports**

Remove `is_openai_model`, `get_api_key_for_model` from imports. Keep `generate_json`, `MODEL_PRICING`.

- [ ] **Step 8: Update docstrings**

- Line 14: Remove Gemini reference
- Line 263: `"lazy Gemini AsyncClient"` → remove
- Any other `"Gemini"` references in comments

- [ ] **Step 9: Commit**

```bash
git add src/strategy/refiner_agent.py
git commit -m "fix: Agent 2 gets own model field (refiner_agent_model), fixes cost inheritance bug"
```

---

### Task 5: Update Agent 3 (`position_agent.py`)

**Files:**
- Modify: `src/strategy/position_agent.py`

- [ ] **Step 1: Fix model assignment**

Line 184: Replace:
```python
self._model = getattr(config, "position_agent_model", "gemini-3-flash-preview")
```
With:
```python
self._model = config.position_agent_model
```

- [ ] **Step 2: Simplify API key handling**

Lines 188-197: Replace with:
```python
self._api_key = config.openai_api_key
self._available = bool(self._api_key) and getattr(
    config, "position_agent_enabled", False
)
```

Remove: `self._gemini_api_key`, `self._openai_api_key`, `self._anthropic_api_key`, and `get_api_key_for_model()` call.

- [ ] **Step 3: Simplify `set_model()` and imports**

Same pattern as Tasks 3 and 4.

- [ ] **Step 4: Remove `thinking_level` from `generate_json` call**

- [ ] **Step 5: Update docstrings**

- Line 16: Remove `"lazy Gemini AsyncClient"`
- Line 162: `"# Gemini structured output schema"` → `"# Structured output schema"`

- [ ] **Step 6: Commit**

```bash
git add src/strategy/position_agent.py
git commit -m "refactor: Agent 3 to OpenAI-only, simplify key resolution"
```

---

### Task 6: Update Lesson Generator

**Files:**
- Modify: `src/strategy/lesson_generator.py`

- [ ] **Step 1: Fix model and key**

Line 126: Replace:
```python
self._model = getattr(config, "position_agent_model", "gemini-3-flash-preview")
```
With:
```python
self._model = config.refiner_agent_model  # gpt-5-mini — lessons need quality
```

Lines 131-134: Replace provider routing with:
```python
self._api_key = config.openai_api_key
```

- [ ] **Step 2: Simplify imports**

Line 20: Remove `is_openai_model` from import. Keep `generate_json`, `MODEL_PRICING`.

- [ ] **Step 3: Remove `thinking_level` from `generate_json` call**

Line 208: Remove `thinking_level="low"` parameter.

- [ ] **Step 4: Update comment**

Line 95: `"# Gemini structured output schema"` → `"# Structured output schema"`

- [ ] **Step 5: Commit**

```bash
git add src/strategy/lesson_generator.py
git commit -m "refactor: lesson generator to OpenAI-only, upgrade model to gpt-5-mini"
```

---

### Task 7: Update Engine Core (`core.py`)

**Files:**
- Modify: `src/engine/core.py`

This is the most critical file — agent instantiation guards live here.

- [ ] **Step 1: Fix Agent 3 instantiation guard**

Line 67: Replace `if config.agent_api_key:` with `if config.openai_api_key:`

- [ ] **Step 2: Fix Agent 2 instantiation guard**

Line 130: Replace `if config.agent_api_key:` with `if config.openai_api_key:`

- [ ] **Step 3: Fix Agent 2 log message**

Line 134: Replace `model=config.agent_model` with `model=config.refiner_agent_model`

- [ ] **Step 4: Fix lesson generator instantiation guard**

Line 183: Replace `if config.agent_api_key:` with `if config.openai_api_key:`

- [ ] **Step 5: Fix Agent 1 instantiation guard**

Line 189: Replace `if config.agent_api_key:` with `if config.openai_api_key:`

- [ ] **Step 6: Fix Agent 2 default model log**

Line 359: Replace `model=self.config.agent_model` with `model=self.config.refiner_agent_model`

- [ ] **Step 7: Simplify Agent 3 model restoration**

Lines 361-368: Replace the `get_api_key_for_model` block with:
```python
if self.position_agent and agent_models.get("agent3"):
    self.position_agent._model = agent_models["agent3"]
    self.position_agent._api_key = self.config.openai_api_key
    logger.info("agent3_model_restored", model=agent_models["agent3"])
```

- [ ] **Step 8: Update comment**

Line 187: `"# AI entry agent (Gemini — intelligent decision-maker)"` → `"# AI entry agent"`

- [ ] **Step 9: Commit**

```bash
git add src/engine/core.py
git commit -m "refactor: core.py agent guards to openai_api_key, fix Agent 2 model reference"
```

---

### Task 8: Update Dashboard Routes & API

**Files:**
- Modify: `src/dashboard/routes.py`
- Modify: `src/dashboard/api.py`
- Modify: `src/dashboard/templates/settings.html`

- [ ] **Step 1: Fix `routes.py`**

Line 213: Replace `config.agent_api_key` with `config.openai_api_key`

Line 229: Replace:
```python
ctx["agent3_model"] = agent_models.get("agent3", getattr(config, "position_agent_model", "gemini-3-flash-preview"))
```
With:
```python
ctx["agent3_model"] = agent_models.get("agent3", config.position_agent_model)
```

Line 228: Replace:
```python
ctx["agent2_model"] = agent_models.get("agent2", config.agent_model)
```
With:
```python
ctx["agent2_model"] = agent_models.get("agent2", config.refiner_agent_model)
```

- [ ] **Step 2: Fix `api.py` model switch endpoint**

Lines 925-931: Replace the `get_api_key_for_model` block with:
```python
engine.position_agent._model = model
engine.position_agent._api_key = config.openai_api_key
```

Remove the `from src.strategy.llm_client import get_api_key_for_model` import at line 925.

Line 970: Replace Gemini fallback list:
```python
else ["gemini-3-pro-preview", "gemini-3-flash-preview"]
```
With:
```python
else list(MODEL_PRICING.keys())
```

Add `MODEL_PRICING` to the imports at top of file (or use a local import).

- [ ] **Step 3: Fix `settings.html` template**

Lines 127, 175, 229: Replace all three instances of:
```html
{% if not config.agent_api_key %}
```
With:
```html
{% if not config.openai_api_key %}
```

Lines 129, 177, 231: Replace all three instances of:
```
Set AGENT_API_KEY in environment to use AI agents.
```
With:
```
Set OPENAI_API_KEY in environment to use AI agents.
```

- [ ] **Step 4: Commit**

```bash
git add src/dashboard/routes.py src/dashboard/api.py src/dashboard/templates/settings.html
git commit -m "refactor: dashboard to OpenAI-only, fix Agent 2 model default display"
```

---

### Task 9: Remove Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Remove Gemini and Anthropic packages**

Remove these two lines:
```
anthropic>=0.40.0
google-genai>=1.55.0
```

Keep: `openai>=1.30.0`

- [ ] **Step 2: Verify no broken imports**

Run: `python -c "import src.main; print('All imports OK')"`

This will transitively import everything. If any file still tries to import `google.genai` or `anthropic` at module level, this will fail.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: remove anthropic and google-genai dependencies"
```

---

### Task 10: Update Documentation

**Files:**
- Modify: `MAIN_BOT.md`
- Modify: `CUSTOM_BOT.md`

- [ ] **Step 1: Update MAIN_BOT.md**

Find and replace references to:
- `AGENT_API_KEY` → `OPENAI_API_KEY` (in the context of agent configuration)
- `claude-haiku-4-5-20251001` and Anthropic references → remove or replace with OpenAI model names
- Any "LLM Trade Analyst (Anthropic — legacy)" section → update to reflect OpenAI-only

- [ ] **Step 2: Update CUSTOM_BOT.md**

Replace any "Claude" LLM provider reference with "OpenAI".

- [ ] **Step 3: Commit**

```bash
git add MAIN_BOT.md CUSTOM_BOT.md
git commit -m "docs: update documentation for OpenAI-only consolidation"
```

---

### Task 11: Delete Unused Branches

- [ ] **Step 1: Delete local branches**

```bash
git branch -d simple-algorithmic-entry
git branch -d wide-entry
git branch -D worktree-agent-af539414
```

(Use `-D` for worktree branch as it may not be fully merged)

- [ ] **Step 2: Delete remote branches**

```bash
git push origin --delete simple-algorithmic-entry
git push origin --delete wide-entry
git push origin --delete worktree-agent-af539414
```

- [ ] **Step 3: Verify remaining branches**

Run: `git branch -a`
Expected: `main`, `expanded`, `footprint` (local + remote), no others

- [ ] **Step 4: No commit needed** (branch operations are already persisted)

---

### Task 12: Cherry-pick to `expanded` and Deploy

- [ ] **Step 1: Switch to expanded branch**

```bash
git checkout expanded
```

- [ ] **Step 2: Cherry-pick all commits from main**

Cherry-pick the commits from Tasks 1-10 (in order):

```bash
git cherry-pick <commit-hash-task1> <commit-hash-task2> ... <commit-hash-task10>
```

If there are conflicts, resolve them manually — the changes are mechanical so conflicts should be minimal.

- [ ] **Step 3: Verify expanded compiles**

```bash
python -c "import src.main; print('OK')"
```

- [ ] **Step 4: Push expanded**

```bash
git push origin expanded
```

- [ ] **Step 5: Switch back to main**

```bash
git checkout main
```

- [ ] **Step 6: Verify Fly.io secrets**

Remind user to:
1. Verify `OPENAI_API_KEY` is set on tarakta-expanded Fly.io app
2. Remove `AGENT_API_KEY` and `ANTHROPIC_API_KEY` secrets after successful deployment
