# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Tarakta MM** — an automated crypto futures trading bot that implements the TTC "Market Maker Method" course on Binance. Single strategy. No SMC. No multi-agent pipelines. One deterministic scanner + one LLM sanity-reviewer as a veto layer.

Production app: **`tarakta-mm`** on Fly.io (Singapore, `sin` region, shared-cpu-1x 2GB). Paper-trading by default.

## Coding Principles (Karpathy Guidelines)

**1. Think Before Coding** — Don't assume. Don't hide confusion. Surface tradeoffs.
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

**2. Simplicity First** — Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

**3. Surgical Changes** — Touch only what you must. Clean up only your own mess.
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that YOUR changes made unused.

**4. Goal-Driven Execution** — Define success criteria. Loop until verified.
- Transform tasks into verifiable goals (e.g., "Fix the bug" → "Reproduce it, then verify the fix").
- For multi-step tasks, state a brief plan with verification at each step.
- Strong success criteria let you work independently. Weak criteria require clarification — ask for it.

## Required Reading

Before modifying any MM engine code (`src/strategy/mm_engine.py`, `src/strategy/mm_*.py`, MM-related routes/API/templates), you **MUST** read `docs/MM_ENGINE_INTEGRATION_GUIDE.md` first. It contains the DB column contract, persistence rules, and a checklist that prevents silent data loss.

Before modifying the sanity agent (`src/strategy/mm_sanity_agent.py`), read `docs/MM_SANITY_AGENT_DESIGN.md`. The rubric, prompt-caching contract, and cost-cap behaviour are specified there.

## Architecture

```
 Binance candles ─┐
                  ▼
       ┌──────────────────────┐
       │   MM Engine scan     │  src/strategy/mm_engine.py (~4000 lines)
       │  (5-min cycle, pure  │  no LLM in this path
       │   algorithmic rules) │
       └──────────┬───────────┘
                  ▼
       ┌──────────────────────┐
       │ Deterministic gates  │  formation → HTF veto → confluence →
       │ (fail fast, no LLM)  │  retest conditions → R:R → …
       └──────────┬───────────┘
                  ▼
       ┌──────────────────────┐
       │  MM Sanity Agent     │  src/strategy/mm_sanity_agent.py
       │  (Opus 4.7, veto)    │  fires only on survivors;  fail-open
       └──────────┬───────────┘
                  ▼
          MMSignal → sizing → PaperExchange / Binance live
```

**Everything the bot does flows through `mm_engine.py`.** Scanner, scoring, HTF gate, sanity-agent call, sizing, and position lifecycle all live in one class (`MMEngine`). Parallel per-cycle `scan_symbol(...)` calls across the majors universe.

### Key modules

| Module | What it owns |
|---|---|
| `src/config.py` | Pydantic Settings; everything from env vars. Runtime overrides stored in `engine_state.config_overrides`. |
| `src/main.py` | Boots the engine + dashboard. |
| `src/strategy/mm_engine.py` | Scanner, entry pipeline, position lifecycle. |
| `src/strategy/mm_sanity_agent.py` | LLM veto layer (Opus 4.7 + 1h prompt caching + extended thinking). |
| `src/strategy/mm_confluence.py` | 12-factor confluence scoring. Direction-aware as of migration 018. |
| `src/strategy/mm_ema_framework.py` | 4H EMA stack + trend state (wired into HTF veto as of f95c507). |
| `src/strategy/mm_formations.py` | M/W detector + variants (three_hits, final_damage, etc.). |
| `src/strategy/mm_levels.py`, `mm_linda.py` | Course-3-level cycle tracking on each TF. |
| `src/strategy/mm_weekly_cycle.py` | Weekly phase machine (accumulation / FMWB / Friday trap / etc.). |
| `src/strategy/mm_sessions.py`, `mm_brinks.py`, `mm_board_meetings.py` | Session timing + named course setups. |
| `src/strategy/mm_targets.py`, `mm_risk.py` | TP tiers + 1% risk sizing. |
| `src/strategy/mm_scalp_vwap_rsi.py`, `mm_scalp_ribbon.py` | Fallback 15m scalp paths when no MM formation found. |
| `src/strategy/mm_data_feeds.py` | Pluggable registry: Binance liquidation, yfinance correlation (both free). |
| `src/exchange/paper.py` | `PaperExchange` wrapping CCXT for paper trading. |
| `src/dashboard/` | FastAPI + Jinja2 UI (port 8080). |
| `src/data/repository.py` | All Supabase I/O. `_TRADE_COLUMNS` allowlist — see Gotchas. |

### Pipeline gates (in order)

Every setup runs through these in `MMEngine.scan_symbol`. Each is a stage counter in the funnel (`/mm-status`):

1. `candles_ok` — 1H/4H/1D/15m all fetched and non-empty
2. `formation_found` — M/W or fallback scalp detected
3. `htf_aligned` — 4H trend veto (rejects counter-4H-trend setups unless exempt reversal variant at non-accelerating trend)
4. `level_ok` — post-formation Level 1/2/3 analysis
5. `phase_valid` — inside an entry-eligible weekly phase
6. `direction_ok` — weekly bias gate + FMWB real-move check
7. `target_acquired` — TP tiers computed
8. `rr_passed` — R:R ≥ min (default 1.5)
9. `scored` — confluence computed
10. `confluence_passed` — score ≥ threshold (Grade C default)
11. `retest_passed` — 2+ of 4 retest conditions
12. `sanity_agent_passed` — **LLM veto layer** (only reaches here if agent APPROVEs, fails open, or is disabled)
13. `signal_built` — MMSignal produced; sizing and execution proceed

### Sanity Agent (Agent 4)

- **Model:** `claude-opus-4-7` with extended thinking. Auto-downgrades to `claude-sonnet-4-6` when projected monthly spend exceeds 90% of `mm_sanity_agent_monthly_budget_usd` (default $600).
- **Prompt caching:** system prompt sent with `ttl=1h` explicitly (Anthropic default regressed to 5m in March 2026, which is useless for our bursty cadence).
- **Fails open** on API error, timeout, missing SDK, or missing API key — trading is never halted by agent failure. An `ERROR` row is still written to `mm_agent_decisions` for observability.
- **Kill switch:** set `MM_SANITY_AGENT_ENABLED=false` to disable without redeploy.
- See `docs/MM_SANITY_AGENT_DESIGN.md` for the full rubric, rollout plan, and fixtures.

### Data flow into the agent

Engine computes every derived feature (4H trend direction + strength + acceleration, price-vs-EMA %, session name + minutes_in, weekly phase, counter-trend flag, last 5 trades on the symbol) and hands the model a pre-computed dict. The model reasons; the model does not compute. See `mm_sanity_agent.build_context(...)`.

## Commands

```bash
# Run locally
python3 -m src.main

# Tests — asyncio_mode = "auto" is configured
pytest                              # all tests (should be 641 passing, 1 skipped)
pytest tests/test_mm_sanity_agent.py # single file
pytest -x                           # stop on first failure

# Lint (7 pre-existing F821 errors in mm_engine.py are known — don't "fix" them)
ruff check src/ tests/
ruff format src/ tests/

# Docker
docker compose up --build

# Deploy — ALWAYS use these flags; Depot builder times out
fly deploy --depot=false --remote-only

# View production logs
fly logs --app tarakta-mm

# Set secrets (e.g. the Anthropic key for the sanity agent)
fly secrets set ANTHROPIC_API_KEY=sk-ant-... --app tarakta-mm
```

## Gotchas

- **MM Engine DB contract.** Adding a new `trades` column needs THREE changes: (1) migration file, (2) add name to `_TRADE_COLUMNS` in `src/data/repository.py`, (3) reference the exact column name in `mm_engine.py`. Miss any step and data silently drops. Same applies to `mm_agent_decisions` via `_MM_AGENT_DECISION_COLUMNS`. See `docs/MM_ENGINE_INTEGRATION_GUIDE.md`.
- **State persistence.** Every in-memory MMPosition change (SL tighten, level advance, partial close, SVC invalidation flag, breakeven move) must be followed by `repo.update_trade(...)`. Otherwise a restart loses progress and the engine can double-enter or re-close partials.
- **Dead-code pattern to avoid.** `_var = compute()  # noqa: F841 — kept for future use` is exactly what let the BNB 2026-04-17 short slip through — `trend_state_4h` was computed and discarded. If state is worth computing it's worth using or deleting. Don't add new `noqa: F841` comments silencing "unused" state in the engine.
- **Prompt caching TTL.** Anthropic changed the default from 1h → 5m silently in March 2026. The sanity agent requests `ttl=1h` explicitly in `mm_sanity_agent._call_model`. Don't remove that — 5m is useless for our cadence.
- **Direction-aware scoring.** `mm_confluence._score_ema_alignment` takes `trade_direction` from `MMContext` and scores 0 when EMA alignment opposes the trade. Before migration 018 it awarded full 8 pts either way, which inflated counter-trend setup scores. Keep it direction-aware.
- **Deploy flag.** Always `--depot=false --remote-only`. Depot builder on Fly times out on this image.
- **Branch policy.** Work on `main`. Feature branches encouraged for multi-commit work; fast-forward merge back to `main` before deploy. CLAUDE.md changes and DB migrations go directly on main.
- **Instance isolation.** `engine_state` and `trades` are keyed by `instance_id`. `tarakta-mm` uses the default `main` instance_id. Don't spin up a second instance without changing the env var.
- **OOM risk.** shared-cpu-1x with 2GB. Large parallel candle fetches across the full symbol universe can OOM. Monitor `fly logs --app tarakta-mm`.
- **python3 only.** No bare `python` on macOS.
- **pandas-ta.** If re-adding it, install with `--no-deps` (Dockerfile handles this). Don't put it in regular pip install chain — the build breaks.
- **`Any` imports.** `mm_engine.py` doesn't `from typing import Any`. If you need a typed None placeholder in the engine, use `= None  # type: ignore[assignment]` rather than adding the import (one is enough, the type checker isn't run in CI here).
- **7 pre-existing F821 lint errors.** Quoted forward references to dataclasses in other modules (`"CorrelationSignal | None"` etc.). They're intentional (would create circular imports) and have been there for months. Don't "fix" them unless you also restructure the imports.

## Environment

Required (see `.env.example`):

| Var | Purpose |
|---|---|
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Binance futures API (read-only credentials are fine for paper mode) |
| `SUPABASE_URL` / `SUPABASE_KEY` | Supabase SWSP — service-role key required (we do DDL-ish inserts that anon can't do) |
| `DASHBOARD_USERNAME` / `DASHBOARD_PASSWORD_HASH` | Admin auth for `/mm` and `/mm/settings` |
| `VIEWER_USERNAME` / `VIEWER_PASSWORD_HASH` | Read-only role for viewing without mutating |
| `SESSION_SECRET` | 32+ char random string for Starlette session cookie signing |
| `TRADING_MODE` | `paper` (default) or `live` |
| `INSTANCE_ID` | Defaults to `main`; change only if running a second instance sharing the same DB |
| `ANTHROPIC_API_KEY` | Sanity agent. If absent, agent fails open (engine behaves as if agent disabled). |

Sanity-agent-specific optional overrides (all have sensible defaults in `config.py`):

| Var | Default | Purpose |
|---|---|---|
| `MM_SANITY_AGENT_ENABLED` | `true` | Global kill switch |
| `MM_SANITY_AGENT_MODEL` | `claude-opus-4-7` | Primary model |
| `MM_SANITY_AGENT_FALLBACK_MODEL` | `claude-sonnet-4-6` | Used when budget cap hit |
| `MM_SANITY_AGENT_THINKING_BUDGET` | `4000` | Extended-thinking token budget |
| `MM_SANITY_AGENT_TIMEOUT_S` | `20.0` | Hard latency cap |
| `MM_SANITY_AGENT_MIN_CONFIDENCE` | `0.0` | VETOs below this confidence get downgraded to a log-and-approve; 0.0 = honour every VETO |
| `MM_SANITY_AGENT_MONTHLY_BUDGET_USD` | `600.0` | Auto-downgrade to Sonnet when projected spend exceeds 90% of this |

## Deployment

Fly.io app **`tarakta-mm`**, region `sin` (Singapore), shared-cpu-1x / 2GB RAM. Config in `fly.toml`. Dashboard at https://tarakta-mm.fly.dev. Always deploy with `--depot=false --remote-only`.

Fly machine auto-stop is off, min-running is 1 — the bot must be always-on for live scans.

## Database

Supabase project **SWSP** (ref: `uounrdaescblpgwkgbdq`, region `ap-southeast-1`). Shared with non-trading apps — **never run `DROP TABLE` or truncate without scoping to our tables**.

### Migrations

`migrations/` 001-019. Gaps at 008 and 011 are historical (SMC-era reservations that were never filled). Do not re-use those numbers.

Latest MM-specific migrations:
- **015** — MM method columns (`strategy`, `mm_formation`, `mm_cycle_phase`, `mm_confluence_grade`)
- **017** — MM lifecycle (`mm_entry_type`, `mm_peak2_wick_price`, `mm_svc_high/low`, SL-progression flags, `mm_took_200ema_partial`)
- **018** — HTF trend snapshot (`htf_trend_4h`, `htf_trend_1d`, `counter_trend`)
- **019** — Sanity agent (`mm_agent_decision/reason/confidence/model/concerns` on `trades`; new `mm_agent_decisions` table)

### Tables this bot uses

| Table | Purpose |
|---|---|
| `trades` | All MM trades (tagged `strategy='mm_method'`). Open positions restored from here on restart. |
| `signals` | Rejected + accepted signal metadata. Telemetry. |
| `engine_state` | `config_overrides` — runtime settings editable from `/mm/settings` without redeploy. Keyed by `instance_id`. |
| `mm_agent_decisions` | Every sanity agent call: APPROVE, VETO, and ERROR. Primary observability surface for the LLM layer. |
| `partial_exits` | TP1/TP2 scale-outs. |
| `reversals` | Logged when a position flips direction at Level 3. |
| `portfolio_snapshots` | Hourly balance snapshots for the dashboard equity curve. |
| `candle_cache` | Reduces Binance API load. |
| `api_usage` | Token + cost log for LLM calls (sanity agent, anything else using the Anthropic SDK). Fire-and-forget writes. |
| `trade_lessons` | Post-trade lessons (legacy-ish — not currently written by this bot but schema is retained for future use). |
| `review_requests` / `review_comments` | Dashboard review-tool integration. |

### Tables in SWSP but NOT used by this bot

SWSP is shared with unrelated apps. Tables that look tempting (e.g. `profiles`, `posts`, `messages`, `subscription_*`) belong to other systems — do not query or modify them from the bot.

## Dashboard

FastAPI + Jinja2, port 8080. Health check at `/health`.

| Route | Purpose |
|---|---|
| `/` → `/mm` | Main dashboard — open positions, recent trades, PnL, quick actions |
| `/mm-status` | Scan-funnel telemetry — per-cycle stage counts, reject reasons, factor hits |
| `/mm/settings` | Runtime tunables (risk %, leverage, min RR, min confluence, scan interval, cooldown, max SL%) |
| `/api/mm/status` | JSON status for polling |
| `/api/mm/begin` / `/api/mm/stop` | Toggle scanning active state |
| `/api/mm/settings` | POST settings form |
| `/reviews` + `/reviews/{id}` + `/reviews/tool-map` | Review-tool endpoints |
| `/login` / `/logout` | Auth |
| `/health` | Fly health check |

## Testing

`pyproject.toml` sets `asyncio_mode = "auto"` — don't `@pytest.mark.asyncio` async tests, it's redundant.

Full suite: **641 passing, 1 skipped** as of migration 019. CI runs `pytest -x`. Lint runs `ruff check src/ tests/`.

`tests/test_mm_sanity_agent.py` covers parse_response, compute_cost, build_user_prompt, graceful degradation, and the `build_context` canary for the BNB 2026-04-17 pattern (counter-trend + accelerating + Grade F must all be visible to the model). Do not mock away the BNB canary — it's the regression test for the class of failure this whole agent exists to catch.
