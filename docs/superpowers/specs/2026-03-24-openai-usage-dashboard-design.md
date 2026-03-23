# OpenAI Usage Dashboard — Design Spec

## Overview

A new `/usage` page in the Tarakta dashboard that tracks and visualizes all OpenAI API usage from this app. Internal logging only — every `generate_json()` call in `llm_client.py` is recorded to a new `api_usage` Supabase table.

## Data Source

**Single source: internal instrumentation of `src/strategy/llm_client.py`.**

After each successful `generate_json()` call, write a row to `api_usage` with the model, tokens, calculated cost, and caller identity. Cost is calculated using the existing `MODEL_PRICING` dict (per-1M-token rates for input/cached/output).

No OpenAI billing API calls. No cross-app tracking. Tarakta's own usage only.

## Database

### New migration: `014_api_usage.sql`

```sql
CREATE TABLE IF NOT EXISTS api_usage (
    id BIGSERIAL PRIMARY KEY,
    instance_id TEXT NOT NULL DEFAULT 'main',
    caller TEXT NOT NULL,          -- 'agent1', 'agent2', 'agent3', 'rag', 'scanner'
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd NUMERIC(10, 6) NOT NULL DEFAULT 0,
    trade_id UUID,                 -- nullable FK to trades
    signal_id UUID,                -- nullable FK to signals
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_api_usage_instance_created ON api_usage (instance_id, created_at DESC);
CREATE INDEX idx_api_usage_caller ON api_usage (caller);
CREATE INDEX idx_api_usage_model ON api_usage (model);
```

No foreign key constraints (trades/signals may not exist yet when the call is logged). Indexes support the dashboard queries (time-range scans by instance, group-by caller/model).

### Alert threshold

Stored in `engine_state.config_overrides` as:
```json
{
  "usage_alert_threshold_usd": 50.0
}
```

Default: no threshold (null). Configurable from the usage page.

## Instrumentation

### `llm_client.py` changes

Add a `caller` parameter to `generate_json()`:

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
    caller: str = "unknown",       # NEW
    repo: Repository | None = None, # NEW — optional, for logging
    trade_id: str | None = None,    # NEW
    signal_id: str | None = None,   # NEW
) -> LLMResult:
```

After `_call_openai()` returns successfully, fire-and-forget a `repo.log_api_usage()` call. Failures are logged but never block the agent pipeline.

### Callers to tag

| Caller tag | Source file | Context |
|-----------|------------|---------|
| `agent1` | `strategy/agent_analyst.py` | Signal analysis |
| `agent2` | `strategy/refiner_agent.py` | Entry decision |
| `agent3` | `strategy/position_agent.py` | Position management |
| `rag` | `data/rag.py` | RAG query generation |

Each caller passes `caller="agent1"` etc. and optionally `trade_id`/`signal_id` when available.

### Cost calculation

Uses existing `MODEL_PRICING` dict. For unknown models, falls back to gpt-5.4-mini rates with a warning log.

```python
def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = MODEL_PRICING.get(model, MODEL_PRICING["gpt-5.4-mini"])
    input_rate, _, output_rate = rates  # cached_input not tracked yet
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
```

## Repository Methods

Add to `Repository`:

```python
async def log_api_usage(self, usage: dict) -> None:
    """Insert a row into api_usage. Fire-and-forget, never raises."""

async def get_usage_summary(self, days: int = 30) -> list[dict]:
    """Daily aggregates: date, total_cost, total_requests, total_input_tokens, total_output_tokens."""

async def get_usage_by_model(self, days: int = 30) -> list[dict]:
    """Group by model: model, total_cost, total_requests."""

async def get_usage_by_caller(self, days: int = 30) -> list[dict]:
    """Group by caller: caller, total_cost, total_requests, total_input_tokens, total_output_tokens."""

async def get_usage_totals(self, days: int = 30) -> dict:
    """Current period totals: total_cost, total_requests, avg_daily_cost, projected_month_end."""
```

All queries scoped by `instance_id` and date range.

## Dashboard Page

### Route

`GET /usage` — `@login_required`, added to `routes.py`. Nav link added to `base.html` between "Analytics" and "Chart".

### API Endpoints

```
GET /api/usage/summary?days=30    — daily time series
GET /api/usage/by-model?days=30   — per-model breakdown
GET /api/usage/by-caller?days=30  — per-agent breakdown
GET /api/usage/totals?days=30     — header stat numbers
POST /api/usage/threshold         — save alert threshold (admin only)
```

All support `?instance=` for multi-instance.

### Template: `usage.html`

Extends `base.html`. Sections top to bottom:

**1. Header stat cards (same style as dashboard.html)**
- **Month spend** — total cost this calendar month (green if under threshold, amber if >80%, red if exceeded)
- **Daily average** — avg cost/day over selected period
- **Projected month-end** — daily avg × days in month
- **Budget remaining** — threshold minus month spend (or "No limit set" if null)
- **Total requests** — count of API calls in period

**2. Spend over time chart**
- Chart.js bar chart, one bar per day, y-axis = USD
- Toggle button: "Daily" / "Weekly" (JS re-aggregates client-side)
- Cumulative spend line overlay (secondary y-axis)
- Threshold line drawn as horizontal dashed red line (if configured)
- Period selector: 7d / 30d / 90d buttons (re-fetches from API)

**3. Per-model breakdown**
- Chart.js horizontal bar chart
- Each model as a bar, sorted by cost descending
- Label shows model name + dollar amount

**4. Per-agent breakdown**
- Chart.js doughnut chart
- Segments: Agent 1 (Analyst), Agent 2 (Refiner), Agent 3 (Position), RAG
- Center text: total cost
- Legend below with request counts

**5. Token volume table**
- HTML table (same style as trades.html)
- Columns: Agent, Requests, Input Tokens, Output Tokens, Total Tokens, Cost
- One row per caller, sorted by cost descending
- Footer row with totals
- Clicking a row could expand to show daily breakdown (stretch goal, not MVP)

**6. Alert threshold config (admin only)**
- Card at bottom with:
  - Current threshold display
  - Number input + "Save" button
  - "Clear" button to remove threshold
- Viewer role sees threshold but no controls

### Alert Banner

When month spend exceeds threshold, a banner appears on ALL pages (added to `base.html`):
- **>80% of threshold**: Amber banner — "OpenAI spend at $X of $Y limit"
- **>100% of threshold**: Red pulsing banner — "OpenAI budget exceeded: $X / $Y"

This requires `_base_context()` in `routes.py` to query current month spend and threshold on every page load. Cache this for 5 minutes to avoid hammering the DB.

## Styling

All new elements use existing CSS classes from `style.css`:
- `.stat-card` for header stats
- `.panel` for chart containers
- `.data-table` for token volume table
- Color variables: `--green`, `--orange`, `--red` for threshold states
- Chart.js colors match the dark theme (card backgrounds, muted borders)

New CSS additions (minimal):
- `.alert-banner-amber` / `.alert-banner-red` for the threshold banners
- `.chart-toggle` button group for daily/weekly and period selectors
- `.threshold-config` card styling

## Error Handling

- Usage logging is fire-and-forget: failures are logged via structlog but never block trading
- If `api_usage` table doesn't exist (migration not run), logging silently fails
- Dashboard page shows "No usage data yet" empty state if table is empty
- Chart.js handles empty datasets gracefully (shows empty chart with axes)

## Performance

- Usage logging: single async insert, no blocking
- Dashboard queries: indexed by `(instance_id, created_at DESC)`, aggregations done in SQL
- Base context alert check: cached 5 minutes in `app.state` to avoid per-request DB hits
- No polling on the usage page itself (static load, user refreshes manually)

## Files Changed

| File | Change |
|------|--------|
| `migrations/014_api_usage.sql` | New table + indexes |
| `src/strategy/llm_client.py` | Add `caller`/`repo`/`trade_id`/`signal_id` params, fire-and-forget logging, `calc_cost()` |
| `src/data/repository.py` | Add `log_api_usage()`, `get_usage_*()` methods |
| `src/strategy/agent_analyst.py` | Pass `caller="agent1"`, `repo`, `signal_id` to `generate_json()` |
| `src/strategy/refiner_agent.py` | Pass `caller="agent2"`, `repo`, `signal_id` |
| `src/strategy/position_agent.py` | Pass `caller="agent3"`, `repo`, `trade_id` |
| `src/data/rag.py` | Pass `caller="rag"`, `repo` |
| `src/dashboard/routes.py` | Add `/usage` route |
| `src/dashboard/api.py` | Add `/api/usage/*` endpoints |
| `src/dashboard/templates/usage.html` | New page template |
| `src/dashboard/templates/base.html` | Add "Usage" nav link, alert banner |
| `src/dashboard/static/style.css` | Alert banner styles, chart toggle styles |
