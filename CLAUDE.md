# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Required Reading

Before modifying any MM Engine code (`src/strategy/mm_engine.py`, `src/strategy/mm_*.py`, MM-related routes/API/templates), you **MUST** read `docs/MM_ENGINE_INTEGRATION_GUIDE.md` first. It contains the DB column contract, persistence rules, and a checklist that prevents silent data loss.

## Commands

```bash
# Run
python3 -m src.main

# Tests
pytest                          # all tests (asyncio_mode = "auto")
pytest tests/test_confluence.py # single file
pytest -x                      # stop on first failure

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Docker
docker compose up --build

# Deploy
fly deploy --depot=false --remote-only --app tarakta-expanded  # ACTIVE production app
# fly deploy --depot=false --remote-only --app tarakta         # inactive/legacy
```

## Architecture

Automated crypto trading bot using Smart Money Concepts with three OpenAI-powered LLM agents.

**Core loop** (`src/engine/core.py`, ~4000 lines): TradingEngine runs scan/trade/monitor cycles on a configurable interval. The scanner scores signals via confluence of SMC indicators, then agents decide whether to enter and how to manage positions.

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
- `src/engine/` — Core loop (`core.py`), consensus voting (`consensus.py`), scan scheduler (`scheduler.py`), engine state (`state.py`), pullback watchlist (`watchlist.py`, `entry_refiner.py`)
- `src/strategy/` indicators — `fair_value_gaps.py`, `order_blocks.py`, `liquidity.py`, `sweep_detector.py`, `market_structure.py`, `footprint.py`, `sentiment.py`, `volume.py`, `sessions.py`, `weekly_cycle.py`, `pullback.py`
- `src/advisor/` — Trade advisor using Claude Agent SDK. Fetches missed signals (`missed_signals.py`), simulates outcomes (`outcome_simulator.py`), stores findings in `advisor_insights` table (`insights.py`), and injects learnings into Agent 1/2 context. Runs daily on reset or manually via dashboard button / `POST /advisor/run`. Runner: `runner.py`, MCP tools: `tools.py`
- `src/strategy/mm_engine.py` — Standalone MM Method trading engine (no LLM). Runs parallel to SMC engine with its own paper balance. Tags trades with `strategy='mm_method'`. **See `docs/MM_ENGINE_INTEGRATION_GUIDE.md` for integration rules — DB columns, persistence, and common mistakes.**

**Data flow to agents:**
- Scanner → `signal.agent_context` (OBs, FVGs, liquidity, market structure, volume, leverage) → Agent 1
- `entry_refiner._build_agent_context()` assembles Agent 2's full context from signal.agent_context + live data (order book, footprint, candles)
- Footprint analyzer runs live in entry_refiner before Agent 2, then again as a hard gate in core.py before execution
- Advisor insights (daily) are injected into Agent 1 and Agent 2 prompts via `format_insights_for_agent()`
- Trade lessons (per-trade) are injected into all agent prompts from `trade_lessons` table

**Usage tracking:** All `generate_json()` calls in `llm_client.py` log to `api_usage` table (fire-and-forget). Each caller passes `caller="agent1|agent2|agent3|lessons"` and optionally `repo`, `trade_id`, `signal_id`. Cost calculated via `MODEL_PRICING` dict. Dashboard at `/usage` shows spend charts, per-model/per-agent breakdowns, and configurable alert threshold (stored in `engine_state.config_overrides` as `usage_alert_threshold_usd`).

## Gotchas

- **Agent context pipeline**: When adding new indicators/data, must wire through: (1) scanner's `_enrich_agent_context()` OR entry_refiner's `_build_agent_context()`, (2) the prompt template in the relevant agent file. Data in `signal_components` does NOT auto-propagate to agents.
- **Footprint is ephemeral**: Order flow data (trade tape, live OI) is real-time only — cannot be retroactively fetched. To persist for analysis, must store on signal at scan time.
- **DB model overrides**: After changing LLM model defaults in code, must clear `config_overrides` in `engine_state` table or stale model names persist
- **Deploy flag**: Always `--depot=false --remote-only` for Fly.io (Depot builder times out)
- **Branch policy**: Work on main branch for agent changes, not feature branches
- **pandas-ta / smartmoneyconcepts**: Must install with `--no-deps` (Dockerfile handles this, don't add to regular pip install)
- **Instance isolation**: `engine_state` is keyed by `instance_id` (not singleton) — multiple bot instances share one DB
- **OOM risk**: Bot runs on Fly.io shared-cpu-1x with 2GB RAM. Memory-heavy operations (large candle fetches, multiple concurrent agent calls) can OOM — monitor via `fly logs -a tarakta`
- **python3 only**: No bare `python` on macOS — always use `python3`
- **Data models location**: `FootprintResult`, `SignalCandidate`, and other dataclasses live in `src/exchange/models.py`, not in the strategy files that use them
- **MM Engine DB contract**: Adding a new MM trade column requires THREE changes: (1) DB migration, (2) add to `_TRADE_COLUMNS` in `repository.py`, (3) use exact DB column name in `mm_engine.py`. Missing any step = data silently dropped. See `docs/MM_ENGINE_INTEGRATION_GUIDE.md`
- **MM Engine state persistence**: Every in-memory position change (SL, level, partial close) must be followed by `repo.update_trade()`. Otherwise restarts lose progress and can double-enter or re-close partials

## Environment

Required (see `.env.example`):
- `BINANCE_API_KEY` / `BINANCE_API_SECRET`
- `SUPABASE_URL` / `SUPABASE_KEY` (service role)
- `OPENAI_API_KEY` (all 3 agents)
- `TRADING_MODE` (paper|live)
- `DASHBOARD_USERNAME` / `DASHBOARD_PASSWORD_HASH` / `SESSION_SECRET`
- `ANTHROPIC_API_KEY` (trade advisor agent — optional, only needed for `/advisor/run`)

## Deployment

Fly.io app `tarakta-expanded` (active production) — region `ams` (Amsterdam), shared-cpu-1x, 2GB RAM. Config in `fly.toml`. Always deploy with `--depot=false --remote-only`. Legacy app `tarakta` exists but is not the active deployment.

## Database

Supabase project **SWSP** (ref: `uounrdaescblpgwkgbdq`, region: ap-southeast-1). Migrations in `migrations/` numbered 001-014 (008 and 011 don't exist). Key tables: `trades`, `signals`, `engine_state`, `portfolio_snapshots`, `knowledge_sources`, `knowledge_chunks`, `candle_cache`, `advisor_insights`, `api_usage`, `trade_lessons`.
