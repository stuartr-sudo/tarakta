# Design: Settings API Key, Model Expansion, Training Page

**Date:** 2026-04-02
**Status:** Reviewed

---

## Feature 1: OpenAI API Key on Settings Page

### Goal
Allow admins to view/update the OpenAI API key from the Settings page without restarting the app.

### Design

**Storage:** Store in `config_overrides.api_keys.openai` in the `engine_state` DB table (same pattern as `agent_models`). Key is encrypted at rest using Fernet symmetric encryption with a key derived from `SESSION_SECRET` env var before storing in the JSONB column.

**UI (Settings page, admin-only):** Add an "API Keys" section above the existing Agent sections. Contains:
- A masked text input showing `sk-...xxxx` (last 4 chars) if a key is set, or "Not configured" if empty
- An "Update" button that reveals a full text input + Save/Cancel
- A status indicator: green dot if key is set, red if missing
- A "Clear" button to remove the DB override and revert to the env var key

**Backend:**
- `POST /api/settings/api-key` (admin-only) — accepts `{ "provider": "openai", "key": "sk-..." }`. Validates key format (starts with `sk-`, length > 20). Encrypts and stores in `config_overrides.api_keys.openai`. After saving, updates `engine.config.openai_api_key` and re-sets `self._api_key` on all active agent instances (same pattern as `/api/agent-model` updates `engine.position_agent._model`). Clears stale entries from `_openai_clients` cache in `llm_client.py`.
- `DELETE /api/settings/api-key` (admin-only) — removes the DB override, reverts agents to env var key
- `GET /api/settings/api-key-status` (admin-only) — returns `{ "openai": { "set": true, "hint": "...Bf4x", "source": "db" } }` (never returns full key). `source` is "db", "env", or "none".

**Key propagation:** When key is saved, the API endpoint directly updates `engine.config.openai_api_key` and each agent's `self._api_key` attribute in-memory (same pattern as agent model switching in `api.py` line ~924). No per-call DB lookup needed. On engine startup, check `config_overrides.api_keys.openai` and decrypt it to override the env var value.

**Security:**
- Key encrypted at rest in DB using Fernet (derived from SESSION_SECRET)
- Full key never returned to frontend after save — only last 4 chars
- Admin-only access on all endpoints
- Audit: ensure `config_overrides` is never logged or passed raw to templates

### Files Changed
- `src/dashboard/api.py` — new endpoints + key propagation logic
- `src/dashboard/templates/settings.html` — new API Keys section
- `src/strategy/llm_client.py` — add `clear_client_cache()` helper, startup key loading
- `src/engine/core.py` — on startup, load encrypted key from config_overrides

---

## Feature 2: Add GPT-5.3 and GPT-5.4 Full Models

### Goal
Add `gpt-5.3` and `gpt-5.4` as selectable models for all three agents.

### Design

**Changes to `MODEL_PRICING` in `llm_client.py`:**
```python
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    "gpt-5.4": (15.00, 1.50, 60.00),        # Full GPT-5.4
    "gpt-5.3": (10.00, 1.00, 40.00),         # Full GPT-5.3
    "gpt-5.4-mini": (0.75, 0.075, 4.50),     # Existing
    "gpt-5-mini": (0.25, 0.025, 2.00),       # Existing
    "gpt-5.4-nano": (0.20, 0.02, 1.25),      # Existing
}
```

**MUST look up actual pricing from OpenAI's pricing page before implementing.** The values above are placeholders and must not be shipped as-is. Check https://openai.com/api/pricing/ for current rates.

**That's it.** The settings page model dropdowns are built from `MODEL_PRICING.keys()`, so the new models automatically appear. The agent model switching API (`POST /api/agent-model`) already handles arbitrary model names.

### Files Changed
- `src/strategy/llm_client.py` — add entries to `MODEL_PRICING`

---

## Feature 3: Training Page

### Goal
A dedicated `/training` page with interactive accordion sections that explain the entire system to a non-technical business partner.

### Design

**Route:** `GET /training` — login required (same as other dashboard pages), accessible to both admin and viewer roles.

**Navigation:** Add "Training" link to the nav bar in `base.html`, between "Settings" and the right-side controls.

**Template:** `training.html` — extends `base.html`. Pure HTML/CSS/JS, no backend data needed. All content is static.

**Accordion behavior:** Click a section header to expand/collapse. Only one section open at a time (clicking a new one closes the previous). Smooth CSS transitions. Each section has a numbered header with chevron indicator.

**Sections (10 total):**

1. **System Overview** — What Tarakta is, the core loop (scan → analyze → decide → execute → manage), paper vs live mode concept
2. **The Three AI Agents** — What each agent does, their roles, how they work together. Sub-sections for Agent 1 (Signal Analyst), Agent 2 (Entry Timer), Agent 3 (Position Manager)
3. **Signal Scanning & Smart Money Concepts** — How the scanner works, what SMC indicators are (FVGs, order blocks, liquidity sweeps, market structure), confluence scoring
4. **Risk Management & Safety** — Max risk per trade, circuit breaker, daily drawdown limit, max concurrent positions, max exposure, cooldown periods
5. **Position Management & Trade Lifecycle** — Entry to exit flow, SL/TP levels (TP1/TP2/TP3), trailing stops, how Agent 3 manages positions, what "tighten SL" and "extend TP3" mean
6. **Dashboard Guide** — What each page shows: Dashboard (overview), Trades (history), Signals (scan results), Analytics (performance), Usage (API costs), Chart (TradingView), Settings (configuration)
7. **Paper vs Live Trading** — What paper trading is, how it simulates real trading, when to switch to live, what changes
8. **Settings & Configuration** — What each setting does (leverage, margin, entry threshold, etc.), what the agent model selector does, how to change risk parameters
9. **Understanding the Analytics** — How to read win rate, P&L charts, R:R ratios, what good vs bad metrics look like
10. **Glossary & Key Terms** — Quick reference for SMC terms, trading terms, and Tarakta-specific concepts

**Visual style:** Matches existing dashboard dark theme. Numbered sections with purple accent color (#6C63FF or matching the existing theme). Expanded sections show clear, jargon-light explanations with occasional highlighted info boxes for key concepts.

### Files Changed
- `src/dashboard/templates/training.html` — new template (bulk of work)
- `src/dashboard/templates/base.html` — add nav link
- `src/dashboard/routes.py` — add `/training` route

---

## Implementation Order

1. Feature 2 (Models) — smallest change, 1 file
2. Feature 1 (API Key) — moderate, 4 files
3. Feature 3 (Training) — largest, 3 files but training.html is content-heavy
