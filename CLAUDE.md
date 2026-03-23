# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run
python -m src.main

# Tests
pytest                          # all tests (asyncio_mode = "auto")
pytest tests/test_confluence.py # single file
pytest -x                      # stop on first failure

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Docker
docker compose up --build

# Deploy (expanded footprint bot)
fly deploy --depot=false --remote-only --app tarakta-expanded
```

## Architecture

Automated crypto trading bot using Smart Money Concepts with three OpenAI-powered LLM agents.

**Core loop** (`src/engine/core.py`, ~3500 lines): TradingEngine runs scan/trade/monitor cycles on a configurable interval. The scanner scores signals via confluence of SMC indicators, then agents decide whether to enter and how to manage positions.

**Three-agent system + lesson generator:**
- **Agent 1** (`strategy/agent_analyst.py`): Analyzes signals with tool-use, produces reasoning + suggested SL/TP
- **Agent 2** (`strategy/refiner_agent.py`): Makes entry decisions. Outputs ENTER only — never WAIT, ADJUST, or ABANDON
- **Agent 3** (`strategy/position_agent.py`): Manages open positions. Can ONLY tighten SL and extend TP3 — never closes trades
- **Lesson Generator** (`strategy/lesson_generator.py`): Post-trade analysis, generates reusable lessons stored in `trade_lessons` table

**Key modules:**
- `src/config.py` — Pydantic Settings, all config from env vars. Overrides can be stored in `engine_state.config_overrides` in DB
- `src/strategy/scanner.py` + `confluence.py` — Market scanning and weighted signal scoring
- `src/exchange/paper.py` — Paper trading wrapper around CCXT exchange client
- `src/dashboard/` — FastAPI + Jinja2 web dashboard (port 8080). Pages: `/` dashboard, `/trades`, `/signals`, `/analytics`, `/usage`, `/chart`, `/settings`. API routes under `/api/`. Health check at `/health`
- `src/data/repository.py` — Supabase data access layer
- `src/data/rag.py` — Hybrid RAG (dense + lexical + RRF) over trade history
- `src/risk/` — Risk manager, circuit breaker, portfolio tracker
- `src/execution/` — Order execution and position monitoring (`orders.py`, `monitor.py`)
- `src/engine/entry_refiner.py` — Pullback entry refinement via watchlist
- `src/advisor/` — Trade advisor using Claude Agent SDK. Fetches missed signals (`missed_signals.py`), simulates outcomes (`outcome_simulator.py`), stores findings in `advisor_insights` table (`insights.py`), and injects learnings into Agent 1/2 context. Runs daily on reset or manually via dashboard button / `POST /advisor/run`. Runner: `runner.py`, MCP tools: `tools.py`

**Usage tracking:** All `generate_json()` calls in `llm_client.py` log to `api_usage` table (fire-and-forget). Each caller passes `caller="agent1|agent2|agent3|lessons"` and optionally `repo`, `trade_id`, `signal_id`. Cost calculated via `MODEL_PRICING` dict. Dashboard at `/usage` shows spend charts, per-model/per-agent breakdowns, and configurable alert threshold (stored in `engine_state.config_overrides` as `usage_alert_threshold_usd`).

## Gotchas

- **DB model overrides**: After changing LLM model defaults in code, must clear `config_overrides` in `engine_state` table or stale model names persist
- **Deploy flag**: Always `--depot=false --remote-only` for Fly.io (Depot builder times out)
- **Branch policy**: Work on main branch for agent changes, not feature branches
- **pandas-ta / smartmoneyconcepts**: Must install with `--no-deps` (Dockerfile handles this, don't add to regular pip install)
- **Instance isolation**: `engine_state` is keyed by `instance_id` (not singleton) — multiple bot instances share one DB

## Environment

Required (see `.env.example`):
- `BINANCE_API_KEY` / `BINANCE_API_SECRET`
- `SUPABASE_URL` / `SUPABASE_KEY` (service role)
- `OPENAI_API_KEY` (all 3 agents)
- `TRADING_MODE` (paper|live)
- `DASHBOARD_USERNAME` / `DASHBOARD_PASSWORD_HASH` / `SESSION_SECRET`
- `ANTHROPIC_API_KEY` (trade advisor agent — optional, only needed for `/advisor/run`)

## Database

Supabase project **SWSP** (ref: `uounrdaescblpgwkgbdq`, region: ap-southeast-1). Migrations in `migrations/` numbered 001-014 (008 and 011 don't exist). Key tables: `trades`, `signals`, `engine_state`, `portfolio_snapshots`, `knowledge_sources`, `knowledge_chunks`, `candle_cache`, `advisor_insights`, `api_usage`.
