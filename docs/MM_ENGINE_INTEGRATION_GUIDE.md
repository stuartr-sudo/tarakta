# MM Engine Integration Guide

How the MM Method engine fits into tarakta, and the rules for modifying it without breaking production.

## Architecture Overview

The MM Engine (`src/strategy/mm_engine.py`) runs as a **parallel async task** alongside the existing SMC TradingEngine. It shares the database and exchange connection but has its own:

- **Paper balance** — isolated $10K starting balance via its own `PaperExchange`
- **Scan cycle** — independent 5-minute loop, not tied to SMC scan interval
- **Decision logic** — purely algorithmic (no LLM/agent calls)
- **Position tracking** — in-memory `self.positions` dict, synced to DB

```
main.py
  ├── TradingEngine (SMC)  ── run_engine()
  ├── MMEngine             ── run_mm_engine()   # parallel asyncio task
  └── Dashboard (FastAPI)  ── serves /mm page + /api/mm/* endpoints
```

## Database Contract

All MM trades go into the **same `trades` table** as SMC trades. They are distinguished by `strategy = 'mm_method'`.

### Required columns (added by `migrations/015_mm_method_columns.sql`)

| Column | Type | Purpose |
|--------|------|---------|
| `strategy` | text | Always `'mm_method'` for MM trades |
| `entry_reason` | text | Human-readable reason (e.g. "W formation + Asia sweep") |
| `mm_formation` | text | `"M"` or `"W"` |
| `mm_cycle_phase` | text | Weekly cycle phase (e.g. "accumulation") |
| `mm_confluence_grade` | text | `"A+"`, `"A"`, `"B+"`, `"B"`, `"C"` |

### Column name mapping

The DB schema uses specific column names. The MM engine must match them exactly:

| MM Engine concept | DB column name | NOT this |
|-------------------|---------------|----------|
| Position size (qty) | `entry_quantity` | ~~quantity~~ |
| Cost in USD | `entry_cost_usd` | ~~cost_usd~~ |
| Trade ID | auto-generated `id` (capture from insert result) | ~~trade_id~~ |
| Original qty | `original_quantity` | |
| Remaining qty | `remaining_quantity` | |
| Level progress | `current_tier` | ~~current_level~~ |

### Repository allowlist

`src/data/repository.py` has a `_TRADE_COLUMNS` set that **silently drops** any column not in the list. If you add a new MM column:

1. Add the column to the DB (migration)
2. Add it to `_TRADE_COLUMNS` in `repository.py`
3. Use it in `mm_engine.py`

**All three steps are required.** Missing any one = data silently lost.

## Lifecycle & State Persistence

### What gets persisted to DB

| Event | What's written | Where |
|-------|---------------|-------|
| Trade entry | Full trade row (`insert_trade`) | `trades` table |
| Level advance | `stop_loss`, `current_tier` | `update_trade` |
| Partial exit | `remaining_quantity`, `current_tier` | `update_trade` |
| Full close | `exit_price`, `exit_quantity`, `pnl_usd`, `exit_time`, `exit_reason`, `remaining_quantity=0` | `update_trade` |
| Scan on/off | `scanning_active` | `engine_state.config_overrides.mm_engine_settings` |

### Position restoration on restart

On startup, `run()` queries all open trades with `strategy='mm_method'` and rebuilds `self.positions` from DB rows. This means:

- Partial close progress (`remaining_quantity`) is restored correctly
- Level progress (`current_tier`) is restored
- Stop loss changes are restored
- The engine does **not** double-enter positions after a restart

### What is NOT persisted (in-memory only)

- `partial_closed_pct` on `MMPosition` — **derived** on restore from `original_quantity` vs `remaining_quantity`
- `MMSignal` objects — ephemeral, regenerated each scan cycle
- `cycle_count` — resets to 0 on restart (cosmetic only)

## Dashboard Integration

### Routes

| Path | File | Purpose |
|------|------|---------|
| `/mm` | `routes.py` | Full MM Engine page (SSR) |
| `/api/mm/status` | `api.py` | JSON status (polled every 15s by JS) |
| `/api/mm/begin` | `api.py` | Start scanning (POST, admin only) |
| `/api/mm/stop` | `api.py` | Pause scanning (POST, admin only) |

### Template context variables (`mm.html`)

| Variable | Source | Content |
|----------|--------|---------|
| `mm_snapshot` | `routes.py` | `{balance_usd, total_pnl_usd, drawdown_pct}` |
| `mm_open_trades` | `routes.py` | List of open MM trades from DB |
| `mm_stats` | `routes.py` | `{win_rate, wins, losses, total_pnl, avg_pnl}` |
| `mm_trades` | `routes.py` | Last 50 closed MM trades |

### Status API response shape

```json
{
  "available": true,
  "scanning_active": true,
  "running": true,
  "cycle_count": 42,
  "open_positions": 2,
  "session": "uk",
  "is_weekend": false,
  "total_unrealized_pnl": 15.23,
  "positions": [
    {
      "symbol": "BTC/USDT",
      "direction": "long",
      "entry_price": 68500.0,
      "current_price": 69100.0,
      "stop_loss": 67800.0,
      "current_level": 1,
      "partial_closed_pct": 30,
      "trade_id": "abc-123",
      "unrealized_pnl": 12.50
    }
  ]
}
```

## Checklist: Adding a New Feature to MM Engine

Use this checklist any time you modify the MM engine:

### If adding a new trade field:
- [ ] Add column to DB via migration (`migrations/0XX_*.sql`)
- [ ] Run migration on Supabase
- [ ] Add column name to `_TRADE_COLUMNS` in `repository.py`
- [ ] Use the **exact DB column name** in `insert_trade()` / `update_trade()` calls
- [ ] Verify the field appears in the dashboard (route query + template)

### If modifying position state (SL, qty, level):
- [ ] Update in-memory `MMPosition` object
- [ ] Persist change to DB via `repo.update_trade(pos.trade_id, {...})`
- [ ] Ensure the field is restored in `run()` position restoration block
- [ ] Check that `get_status()` includes the field for the dashboard API

### If adding a new API endpoint:
- [ ] Add route in `api.py` with `@login_required` or `@admin_required`
- [ ] Access MM engine via `_get_mm_engine(request)` helper
- [ ] Handle the `mm is None` case (returns 503)
- [ ] If the endpoint calls an async method, use `await`

### If changing the exchange interface:
- [ ] Test with `PaperExchange` (what MM uses in paper mode)
- [ ] Ensure `place_market_order()` return value is checked for `result.status == "closed"`
- [ ] Handle exchange errors with try/except (don't crash the engine)

## Common Mistakes (and what goes wrong)

| Mistake | Symptom | How to avoid |
|---------|---------|-------------|
| Using `quantity` instead of `entry_quantity` | Trade inserted with NULL quantity, position sizing breaks | Always check `_TRADE_COLUMNS` for exact names |
| Forgetting `_TRADE_COLUMNS` update | Column silently dropped, data lost, no error | Search for `_TRADE_COLUMNS` before any migration |
| Not persisting SL/level to DB | Restart loses SL tightening, re-enters at wrong levels | Every in-memory state change needs a `update_trade` call |
| Sync `get_status()` with async fetch | Status endpoint returns stale/missing data | `get_status()` is async — use `await` in API handler |
| Not restoring positions on startup | Orphaned open trades in DB, double entries | Check `run()` restoration block handles new fields |
| Using local `trade_id` for DB updates | `update_trade` silently updates nothing (wrong ID) | Always capture `db_row["id"]` from `insert_trade` result |

## Testing

```bash
# Run all MM tests (192 tests)
pytest tests/ -k "mm" --ignore=tests/test_advisor_tools.py

# Run specific module tests
pytest tests/test_mm_engine.py
pytest tests/test_mm_formations.py
pytest tests/test_mm_confluence.py
```

## Config (env vars)

| Variable | Default | Purpose |
|----------|---------|---------|
| `MM_METHOD_ENABLED` | `false` | Master switch — must be `true` to start engine |
| `MM_SCAN_INTERVAL_MINUTES` | `5` | How often the engine runs a cycle |
| `MM_MAX_POSITIONS` | `3` | Max concurrent MM positions |
| `MM_RISK_PER_TRADE_PCT` | `1.0` | Risk per trade as % of balance |
| `MM_INITIAL_BALANCE` | `10000` | Starting paper balance (isolated from SMC) |
