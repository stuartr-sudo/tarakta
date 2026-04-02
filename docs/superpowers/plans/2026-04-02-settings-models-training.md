# Settings API Key, Model Expansion & Training Page — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenAI API key management to Settings, expand available models with gpt-5.3/gpt-5.4, and create an interactive Training page.

**Architecture:** Three independent features sharing the existing FastAPI + Jinja2 dashboard. Feature 1 (models) is a one-line dict addition. Feature 2 (API key) adds encrypted key storage in config_overrides with in-memory propagation to agents. Feature 3 (training) is a new static page with CSS accordion.

**Tech Stack:** Python/FastAPI, Jinja2 templates, Supabase (PostgreSQL), cryptography (Fernet), vanilla JS/CSS

**Spec:** `docs/superpowers/specs/2026-04-02-settings-models-training-design.md`

---

### Task 1: Add GPT-5.3 and GPT-5.4 to MODEL_PRICING

**Files:**
- Modify: `src/strategy/llm_client.py:35-39`

- [ ] **Step 1: Add model entries**

In `src/strategy/llm_client.py`, replace the `MODEL_PRICING` dict:

```python
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.3": (1.75, 0.175, 14.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
}
```

- [ ] **Step 2: Verify**

Run: `python3 -c "from src.strategy.llm_client import MODEL_PRICING; print(sorted(MODEL_PRICING.keys()))"`
Expected: `['gpt-5-mini', 'gpt-5.3', 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.4-nano']`

- [ ] **Step 3: Commit**

```bash
git add src/strategy/llm_client.py
git commit -m "feat: add gpt-5.3 and gpt-5.4 full models to MODEL_PRICING"
```

---

### Task 2: Add cryptography dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add cryptography to requirements.txt**

Add `cryptography>=42.0` to `requirements.txt`.

- [ ] **Step 2: Install locally**

Run: `pip3 install cryptography>=42.0`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add cryptography dependency for API key encryption"
```

---

### Task 3: API Key encryption helpers + backend endpoints

**Files:**
- Modify: `src/dashboard/api.py`
- Modify: `src/strategy/llm_client.py`

- [ ] **Step 1: Create shared encryption helpers in `src/utils/crypto.py`**

Create `src/utils/crypto.py`:

```python
"""Symmetric encryption helpers for API key storage."""
from __future__ import annotations

import base64
import hashlib
import os

from cryptography.fernet import Fernet


def _get_fernet() -> Fernet:
    """Derive a Fernet key from SESSION_SECRET env var."""
    secret = os.environ.get("SESSION_SECRET", "fallback-key-not-for-production")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def encrypt_key(plain: str) -> str:
    """Encrypt an API key for DB storage."""
    return _get_fernet().encrypt(plain.encode()).decode()


def decrypt_key(encrypted: str) -> str:
    """Decrypt an API key from DB storage."""
    return _get_fernet().decrypt(encrypted.encode()).decode()
```

Then in `src/dashboard/api.py`, add this import (after existing imports):

```python
from src.utils.crypto import encrypt_key, decrypt_key
```

- [ ] **Step 2: Add clear_client_cache to llm_client.py**

In `src/strategy/llm_client.py`, after the `_get_openai_client` function (after line ~79), add:

```python
def clear_client_cache():
    """Clear cached OpenAI clients (e.g. after API key change)."""
    _openai_clients.clear()
```

- [ ] **Step 3: Add POST /api/settings/api-key endpoint**

In `src/dashboard/api.py`, inside the `create_router` function (note: NOT `create_api_router`), after the `/agent-model` endpoint block (~line 972), add:

**IMPORTANT:** The `create_router` function receives `engine` as a parameter. There is no standalone `config` in scope — access config via `engine.config`. The `engine` variable may be `None` during startup.

```python
    @router.post("/settings/api-key")
    @admin_required
    async def set_api_key(request: Request):
        """Update OpenAI API key at runtime. Encrypted and stored in DB."""
        try:
            body = await request.json()
            key = body.get("key", "").strip()
            if not key:
                return JSONResponse({"error": "key is required"}, status_code=400)
            if not key.startswith("sk-") or len(key) < 20:
                return JSONResponse({"error": "Invalid API key format"}, status_code=400)

            # Encrypt and store in config_overrides
            encrypted = encrypt_key(key)
            state = await repo.get_engine_state()
            if not state:
                return JSONResponse({"error": "Engine state not found"}, status_code=500)
            overrides = state.get("config_overrides") or {}
            if not isinstance(overrides, dict):
                overrides = {}
            api_keys = overrides.get("api_keys") or {}
            api_keys["openai"] = encrypted
            api_keys["openai_hint"] = key[-4:]
            overrides["api_keys"] = api_keys
            state["config_overrides"] = overrides
            await repo.upsert_engine_state(state)

            # Propagate to running engine
            if engine:
                engine.config.openai_api_key = key
                if engine.agent_analyst:
                    engine.agent_analyst._api_key = key
                if engine.refiner_agent:
                    engine.refiner_agent._api_key = key
                if engine.position_agent:
                    engine.position_agent._api_key = key
                # Clear cached OpenAI clients so new key is used
                from src.strategy.llm_client import clear_client_cache
                clear_client_cache()

            logger.info("api_key_updated", provider="openai", hint=key[-4:])
            return {"status": "ok", "hint": key[-4:]}
        except Exception as e:
            logger.error("api_key_update_failed", error=str(e))
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.delete("/settings/api-key")
    @admin_required
    async def clear_api_key(request: Request):
        """Clear DB API key override, revert to env var."""
        try:
            state = await repo.get_engine_state()
            if state:
                overrides = state.get("config_overrides") or {}
                if isinstance(overrides, dict) and "api_keys" in overrides:
                    del overrides["api_keys"]
                    state["config_overrides"] = overrides
                    await repo.upsert_engine_state(state)

            # Revert to env var
            if engine:
                env_key = os.environ.get("OPENAI_API_KEY", "")
                engine.config.openai_api_key = env_key
                if engine.agent_analyst:
                    engine.agent_analyst._api_key = env_key
                if engine.refiner_agent:
                    engine.refiner_agent._api_key = env_key
                if engine.position_agent:
                    engine.position_agent._api_key = env_key
                from src.strategy.llm_client import clear_client_cache
                clear_client_cache()

            return {"status": "ok", "source": "env"}
        except Exception as e:
            logger.error("api_key_clear_failed", error=str(e))
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.get("/settings/api-key-status")
    @admin_required
    async def api_key_status(request: Request):
        """Return API key status without exposing the key."""
        try:
            state = await repo.get_engine_state()
            overrides = (state.get("config_overrides") or {}) if state else {}
            api_keys = overrides.get("api_keys") or {} if isinstance(overrides, dict) else {}

            if api_keys.get("openai"):
                return {
                    "openai": {
                        "set": True,
                        "hint": api_keys.get("openai_hint", "????"),
                        "source": "db",
                    }
                }
            elif engine and engine.config.openai_api_key:
                return {
                    "openai": {
                        "set": True,
                        "hint": engine.config.openai_api_key[-4:] if len(engine.config.openai_api_key) >= 4 else "****",
                        "source": "env",
                    }
                }
            else:
                return {"openai": {"set": False, "hint": "", "source": "none"}}
        except Exception as e:
            return {"openai": {"set": False, "hint": "", "source": "error"}}
```

- [ ] **Step 4: Verify endpoints parse correctly**

Run: `python3 -c "from src.dashboard.api import create_router; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/api.py src/strategy/llm_client.py
git commit -m "feat: add API key management endpoints with encryption"
```

---

### Task 4: Load encrypted API key on engine startup

**Files:**
- Modify: `src/engine/core.py`

- [ ] **Step 1: Find the engine startup section**

In `src/engine/core.py`, find where `config_overrides` is loaded on startup (search for `agent_models` loading from overrides). This is in the `_restore_state` or initialization section.

- [ ] **Step 2: Add API key loading after agent_models loading**

After the block that loads `agent_models` from `config_overrides`, add:

```python
            # Load API key override from DB (encrypted)
            api_keys = overrides.get("api_keys") or {}
            if api_keys.get("openai"):
                try:
                    from src.utils.crypto import decrypt_key
                    decrypted = decrypt_key(api_keys["openai"])
                    self.config.openai_api_key = decrypted
                    logger.info("api_key_loaded_from_db", provider="openai", hint=api_keys.get("openai_hint", "?"))
                except Exception as e:
                    logger.warning("api_key_decrypt_failed", error=str(e))
```

- [ ] **Step 3: Verify core.py still imports cleanly**

Run: `python3 -c "import src.engine.core; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/engine/core.py
git commit -m "feat: load encrypted API key from DB on engine startup"
```

---

### Task 5: Add API Keys section to Settings UI

**Files:**
- Modify: `src/dashboard/templates/settings.html`
- Modify: `src/dashboard/routes.py`

- [ ] **Step 1: Add context variables to settings route**

In `src/dashboard/routes.py`, inside the `settings_page` function, before the `return templates.TemplateResponse(...)` line (~274), add:

```python
        # API key status
        api_keys = overrides.get("api_keys") or {}
        if api_keys.get("openai"):
            ctx["api_key_set"] = True
            ctx["api_key_hint"] = api_keys.get("openai_hint", "????")
            ctx["api_key_source"] = "db"
        elif config.openai_api_key:
            ctx["api_key_set"] = True
            ctx["api_key_hint"] = config.openai_api_key[-4:] if len(config.openai_api_key) >= 4 else "****"
            ctx["api_key_source"] = "env"
        else:
            ctx["api_key_set"] = False
            ctx["api_key_hint"] = ""
            ctx["api_key_source"] = "none"
```

- [ ] **Step 2: Add API Keys panel to settings.html**

In `src/dashboard/templates/settings.html`, after the closing `</div>` of the Trading Mode `two-col` block (line ~105, just before `<!-- AI Agents -->`), insert:

```html
<!-- API Keys -->
{% if role == 'admin' %}
<div class="panel" style="margin-bottom: 1.5rem;">
    <div class="panel-header">
        <h2>API Keys</h2>
    </div>
    <div class="panel-body">
        <table class="config-table">
            <tr>
                <td style="width:140px;">OpenAI API Key</td>
                <td>
                    <div id="api-key-display" style="display:flex; align-items:center; gap:8px;">
                        {% if api_key_set %}
                        <span class="pulse-dot green"></span>
                        <code style="color:var(--text-muted);">sk-...{{ api_key_hint }}</code>
                        <span style="font-size:0.7rem; color:var(--text-muted);">({{ api_key_source }})</span>
                        {% else %}
                        <span class="pulse-dot" style="background:#ef4444;"></span>
                        <span style="color:#ef4444;">Not configured</span>
                        {% endif %}
                        <button class="btn small" onclick="showApiKeyInput()" style="font-size:0.7rem; padding:2px 8px;">Update</button>
                        {% if api_key_source == 'db' %}
                        <button class="btn small" onclick="clearApiKey()" style="font-size:0.7rem; padding:2px 8px; background:var(--bg-card); border:1px solid var(--border);">Clear</button>
                        {% endif %}
                    </div>
                    <div id="api-key-input" style="display:none; margin-top:8px;">
                        <div style="display:flex; align-items:center; gap:8px;">
                            <input type="password" id="api-key-field" placeholder="sk-..." style="font-size:0.8rem; padding:4px 8px; background:var(--bg-card); color:var(--text-primary); border:1px solid var(--border); border-radius:4px; width:300px;">
                            <button class="btn small primary" onclick="saveApiKey()" style="font-size:0.7rem; padding:2px 8px;">Save</button>
                            <button class="btn small" onclick="hideApiKeyInput()" style="font-size:0.7rem; padding:2px 8px; background:var(--bg-card); border:1px solid var(--border);">Cancel</button>
                        </div>
                        <span id="api-key-status" style="font-size:0.7rem; margin-top:4px; display:inline-block;"></span>
                    </div>
                </td>
            </tr>
        </table>
    </div>
</div>
{% endif %}
```

- [ ] **Step 3: Add API key JavaScript functions**

In `src/dashboard/templates/settings.html`, inside the existing `<script>` block (before the closing `</script>`), add:

```javascript
function showApiKeyInput() {
    document.getElementById('api-key-input').style.display = 'block';
    document.getElementById('api-key-field').focus();
}
function hideApiKeyInput() {
    document.getElementById('api-key-input').style.display = 'none';
    document.getElementById('api-key-field').value = '';
    document.getElementById('api-key-status').textContent = '';
}
async function saveApiKey() {
    const field = document.getElementById('api-key-field');
    const status = document.getElementById('api-key-status');
    const key = field.value.trim();
    if (!key) { status.textContent = 'Enter a key'; status.style.color = '#ef4444'; return; }
    status.textContent = 'Saving...'; status.style.color = '#f59e0b';
    try {
        const resp = await fetch(apiUrl('/api/settings/api-key'), {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({key: key}),
        });
        const data = await resp.json();
        if (data.error) {
            status.textContent = data.error; status.style.color = '#ef4444';
        } else {
            status.textContent = 'Saved! Reloading...'; status.style.color = '#22c55e';
            setTimeout(() => location.reload(), 1000);
        }
    } catch (e) {
        status.textContent = 'Failed'; status.style.color = '#ef4444';
    }
}
async function clearApiKey() {
    if (!confirm('Clear DB key and revert to environment variable?')) return;
    try {
        await fetch(apiUrl('/api/settings/api-key'), {method: 'DELETE'});
        location.reload();
    } catch (e) { alert('Failed to clear key'); }
}
```

- [ ] **Step 4: Verify template renders**

Run: `python3 -c "from jinja2 import Environment, FileSystemLoader; e=Environment(loader=FileSystemLoader('src/dashboard/templates')); t=e.get_template('settings.html'); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/templates/settings.html src/dashboard/routes.py
git commit -m "feat: add API key management UI to Settings page"
```

---

### Task 6: Add Training page route and nav link

**Files:**
- Modify: `src/dashboard/routes.py`
- Modify: `src/dashboard/templates/base.html`

- [ ] **Step 1: Add /training route**

In `src/dashboard/routes.py`, inside the `create_router` function, after the settings-related routes, add:

```python
    @router.get("/training", response_class=HTMLResponse)
    @login_required
    async def training_page(request: Request):
        ctx = await _base_context(request)
        return templates.TemplateResponse(request, "training.html", context=ctx)
```

- [ ] **Step 2: Add Training link to nav bar**

In `src/dashboard/templates/base.html`, find the Settings nav link (line ~62):

```html
            <a href="/settings{{ _iq }}" class="{{ 'active' if current_page == '/settings' else '' }}">Settings</a>
```

Add after it:

```html
            <a href="/training{{ _iq }}" class="{{ 'active' if current_page == '/training' else '' }}">Training</a>
```

- [ ] **Step 3: Commit**

```bash
git add src/dashboard/routes.py src/dashboard/templates/base.html
git commit -m "feat: add /training route and nav link"
```

---

### Task 7: Create Training page template

**Files:**
- Create: `src/dashboard/templates/training.html`

- [ ] **Step 1: Create the training.html template**

Create `src/dashboard/templates/training.html` with the full interactive accordion content. The template extends `base.html`, uses pure CSS for accordion animations, and covers all 10 sections with clear, jargon-light explanations.

Key implementation details:
- CSS: Each accordion item uses a hidden checkbox + label pattern (no JS needed for basic expand/collapse). Add smooth `max-height` transitions.
- Alternatively, use simple JS `onclick` toggling with a `data-open` attribute for cleaner markup.
- Match the existing dashboard dark theme colors from `style.css`.
- Each section gets a numbered header (01-10) with the purple accent color used in the dashboard.
- Inside sections, use info boxes (styled `div` with left border accent) for key concepts.
- Content should be written for a non-technical business partner — avoid jargon, explain every acronym on first use.

The 10 sections and their content:

**01 — System Overview:** What Tarakta is (automated crypto trading bot), the core pipeline (Scanner → Agent 1 → Agent 2 → Execute → Agent 3), paper vs live concept, the scan cycle.

**02 — The Three AI Agents:** Agent 1 (Signal Analyst) evaluates whether a trade setup is worth taking. Agent 2 (Entry Timer) decides the precise moment to enter. Agent 3 (Position Manager) monitors open trades and adjusts stop-loss/take-profit. Explain that agents use OpenAI's GPT models and can be individually enabled/disabled.

**03 — Signal Scanning & Smart Money Concepts:** What scanning means (checking markets on a timer). SMC basics: Fair Value Gaps (price imbalances), Order Blocks (institutional support/resistance), Liquidity Sweeps (stop hunts), Market Structure (higher highs/lower lows). Confluence scoring (signals need multiple indicators agreeing).

**04 — Risk Management & Safety:** Max risk per trade (% of balance), circuit breaker (stops trading after X% loss), daily drawdown limit, max concurrent positions, max portfolio exposure, cooldown after losses. Explain these are safety nets.

**05 — Position Management & Trade Lifecycle:** A trade from start to finish: signal found → agents approve → order placed → SL and three TP levels set (TP1, TP2, TP3) → Agent 3 monitors → partial profits taken at each TP → trade closes at final TP or SL. What "tighten SL" means (move stop-loss closer to protect profit). What "extend TP3" means (let winners run further).

**06 — Dashboard Guide:** Each page explained: Dashboard (live overview, balance, open positions), Trades (full history of all trades with P&L), Signals (every signal the scanner found), Analytics (win rate, profit charts, performance over time), Usage (API costs for running the AI agents), Chart (TradingView chart for manual analysis), Settings (configure everything).

**07 — Paper vs Live Trading:** Paper = simulated trades with real market data, no real money at risk. Live = real orders on the exchange. How to switch (Settings page toggle). What to look for before going live (consistent paper profits, understanding of risk settings).

**08 — Settings & Configuration:** Leverage (how much borrowing), Margin per trade (how much of balance per trade), Entry threshold (minimum signal quality score), the agent model selectors (choose between faster/cheaper or smarter/more expensive AI models), agent enable/disable toggles.

**09 — Understanding the Analytics:** Win rate (% of profitable trades), P&L (profit and loss in dollars), R:R ratio (reward-to-risk — how much you win vs how much you risk), what good metrics look like (>50% win rate, >2:1 average R:R), how to spot problems (declining win rate, large losses).

**10 — Glossary & Key Terms:** Quick-reference alphabetical list of terms: Circuit Breaker, Confluence, Drawdown, Entry Threshold, Fair Value Gap (FVG), Leverage, Liquidity Sweep, Margin, Market Structure, Order Block, Paper Trading, R:R Ratio, SL (Stop Loss), Smart Money Concepts (SMC), TP (Take Profit), and any other terms used in the dashboard.

- [ ] **Step 2: Verify template renders**

Run: `python3 -c "from jinja2 import Environment, FileSystemLoader; e=Environment(loader=FileSystemLoader('src/dashboard/templates')); t=e.get_template('training.html'); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/dashboard/templates/training.html
git commit -m "feat: add interactive Training page with 10 accordion sections"
```

---

### Task 8: Final verification and deploy

- [ ] **Step 1: Run linter**

Run: `ruff check src/dashboard/api.py src/dashboard/routes.py src/strategy/llm_client.py src/engine/core.py`
Fix any issues.

- [ ] **Step 2: Run tests**

Run: `pytest -x`
Fix any failures.

- [ ] **Step 3: Deploy to Fly.io**

Run: `fly deploy --depot=false --remote-only --app tarakta-expanded`

- [ ] **Step 4: Verify all three features work**

1. Visit `/settings` — check new models appear in agent dropdowns, API key section is visible
2. Visit `/training` — all 10 accordion sections expand/collapse properly
3. Test API key save/clear cycle on settings page
