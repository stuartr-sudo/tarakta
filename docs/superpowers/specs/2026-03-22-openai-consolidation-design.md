# Tarakta: OpenAI Consolidation & Branch Cleanup

**Date:** 2026-03-22
**Status:** Draft

## Goal

Consolidate Tarakta to a single OpenAI-only LLM provider, remove unused deployment variants, clean up dependencies, and fix the Agent 2 model inheritance bug — applied to both `main` and `expanded` branches.

## 1. Branch Cleanup

**Delete (local + remote):**
- `simple-algorithmic-entry`
- `wide-entry`
- `worktree-agent-af539414`

**Keep:**
- `main` (development/secondary)
- `expanded` (production — deployed at tarakta-expanded.fly.dev, includes footprint gate + expanded universe)
- `footprint` (kept as reference — subset of expanded, do not delete)

## 2. LLM Client: Strip to OpenAI-Only

### Bug Fix: Agent 2 Model Inheritance

Agent 2 (`refiner_agent.py` line 268) inherits Agent 1's model via `config.agent_model`. There's a Gemini-specific downgrade (`pro` → `flash`), but when running OpenAI models this doesn't fire — so Agent 2 runs on Agent 1's full-price model. With Agent 2 firing every 5 minutes per queued signal, this is a major cost driver.

**Fix:** Add dedicated `refiner_agent_model` config field so Agent 2 has its own model assignment.

### Model Assignment

| Agent | Role | Model | Input $/1M | Cached $/1M | Output $/1M |
|-------|------|-------|-----------|------------|------------|
| Agent 1 | Strategic entry | `gpt-5.4-mini` | $0.75 | $0.075 | $4.50 |
| Agent 2 | Tactical timing | `gpt-5-mini` | $0.25 | $0.025 | $2.00 |
| Agent 3 | Position mgmt | `gpt-5.4-nano` | $0.20 | $0.02 | $1.25 |

### All Files to Change

**`src/strategy/llm_client.py`:**
- Remove: `_ANTHROPIC_PREFIXES`, `is_anthropic_model()`, `_get_anthropic_client()`, `_call_anthropic()`
- Remove: `_get_gemini_client()`, `_call_gemini()`, Gemini-specific imports
- Remove: `get_api_key_for_model()` (single provider, no routing needed)
- Remove: `is_openai_model()` (everything is OpenAI now)
- Simplify: `generate_json()` calls `_call_openai()` directly (remove provider routing)
- Update: `MODEL_PRICING` to only contain `gpt-5.4-mini`, `gpt-5-mini`, `gpt-5.4-nano`
- Keep: `_patch_schema_for_openai()`, `LLMResult`, `_get_openai_client()`, `_call_openai()`

**`src/config.py`:**
- Remove: `agent_api_key` (was Gemini key), `anthropic_api_key`
- Keep: `openai_api_key` as the single LLM API key
- Change default: `agent_model` → `"gpt-5.4-mini"`
- Change default: `position_agent_model` → `"gpt-5.4-nano"`
- Add: `refiner_agent_model: str = "gpt-5-mini"` (fixes Agent 2 inheritance bug)
- Update comments to reflect OpenAI-only setup

**`src/strategy/agent_analyst.py`:**
- Replace `get_api_key_for_model()` calls with direct `config.openai_api_key`
- Remove stored Gemini/Anthropic key fields
- Update docstrings (remove Gemini references)

**`src/strategy/refiner_agent.py`:**
- Use `config.refiner_agent_model` instead of `config.agent_model`
- Remove Gemini auto-downgrade logic (lines 269-271)
- Remove Gemini fallback error handler (lines 445-492 area)
- Replace `get_api_key_for_model()` with direct `config.openai_api_key`
- Remove stored Gemini/Anthropic key fields
- Update docstrings

**`src/strategy/position_agent.py`:**
- Replace `get_api_key_for_model()` with direct `config.openai_api_key`
- Replace `getattr(config, "position_agent_model", "gemini-3-flash-preview")` fallback with direct `config.position_agent_model`
- Remove stored Gemini/Anthropic key fields
- Update `set_model()` method to use `config.openai_api_key` directly
- Update docstrings

**`src/strategy/lesson_generator.py`:**
- Replace `config.agent_api_key` with `config.openai_api_key`
- Remove `is_openai_model()` routing
- Use `config.refiner_agent_model` (gpt-5-mini) instead of `config.position_agent_model` — lessons need quality, not speed
- Replace `getattr` fallback with direct config access

**`src/engine/core.py`:**
- Replace all `config.agent_api_key` guards with `config.openai_api_key`
- Replace `get_api_key_for_model()` calls with direct `config.openai_api_key`
- Update Agent 2 instantiation to pass `refiner_agent_model`
- Update comments

**`src/dashboard/routes.py`:**
- Replace `config.agent_api_key` with `config.openai_api_key`
- Remove Gemini fallback model references

**`src/dashboard/api.py`:**
- Replace `get_api_key_for_model()` with direct `config.openai_api_key`
- Remove hardcoded Gemini model fallback list
- Update model-switch endpoint to work with OpenAI-only pricing

**`src/dashboard/templates/settings.html`:**
- Replace `config.agent_api_key` checks with `config.openai_api_key`
- Update "Set AGENT_API_KEY" warnings to reference OPENAI_API_KEY

**`MAIN_BOT.md`:**
- Remove references to Claude/Anthropic models and `AGENT_API_KEY`
- Update LLM section to reflect OpenAI-only setup

**`CUSTOM_BOT.md`:**
- Update LLM provider reference from Claude to OpenAI

**All files — docstring cleanup:**
- Replace all "Gemini Interactions API", "lazy Gemini AsyncClient", "Gemini structured output schema" references with provider-neutral or OpenAI-specific wording across all agent files

## 3. Dependency Cleanup

**`requirements.txt`:**
- Remove: `anthropic>=0.40.0`
- Remove: `google-genai>=1.55.0`
- Keep: `openai>=1.30.0`

**`pyproject.toml`:** No changes needed (LLM provider packages are only in requirements.txt).

## 4. Fly.io

No code changes needed. Both apps already run single machines:
- `tarakta` (main) — `shared-cpu-1x`, 1024MB, ams region
- `tarakta-expanded` (expanded) — same config

**Secrets cleanup after deployment:**
- Verify `OPENAI_API_KEY` is set as a Fly.io secret on both apps
- Remove `AGENT_API_KEY` and `ANTHROPIC_API_KEY` secrets from both apps after successful deployment

## 5. Apply to Both Branches

All changes apply to both `main` and `expanded` branches:
1. Implement on `main` first
2. Cherry-pick or merge into `expanded`
3. Deploy `expanded` to tarakta-expanded.fly.dev
4. Clean up Fly.io secrets (remove AGENT_API_KEY, ANTHROPIC_API_KEY)

## 6. What Stays Unchanged

- Instance isolation (`INSTANCE_ID` / `instance_id` in repository layer)
- RAG knowledge base (uses OpenAI embeddings — already on `openai_api_key`)
- HuggingFace sentiment (uses `hf_api_token` + inference API, not an LLM provider)
- All trading logic, risk management, footprint gate, dashboard
- Database schema and migrations
- Docker configuration

## 7. Risk Assessment

- **Low risk:** Branch deletion is reversible (git reflog, remote restore)
- **Low risk:** Model swap is config-driven — can revert by changing env vars
- **Medium risk:** Removing Gemini/Anthropic code paths is permanent — if provider switch is needed later, code must be re-added
- **Note:** Dashboard model-switch dropdown will show only 3 models instead of ~12. Acceptable.

## 8. Estimated Cost Savings

**Agent 2 inheritance bug was the biggest cost driver.** When running OpenAI models, Agent 2 used Agent 1's model at full price, firing every 5 minutes per queued signal.

Current (with bug): Agent 2 on same model as Agent 1
- If Agent 1 = gpt-5.4-mini: Agent 2 also at $0.75/$4.50 per 1M tokens

New (with fix): Agent 2 on gpt-5-mini
- Agent 2 at $0.25/$2.00 per 1M tokens (67% cheaper input, 56% cheaper output)

Overall across all agents: ~60-70% reduction in LLM costs.
