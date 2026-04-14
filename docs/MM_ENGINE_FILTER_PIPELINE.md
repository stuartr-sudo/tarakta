# MM Engine Filter Pipeline

Every potential MM Method trade must pass **23 gates across 4 stages** before it
can be entered. This document enumerates every filter as it actually runs in
`src/strategy/mm_engine.py`, with the default thresholds and the log event each
one emits when it rejects a symbol.

The counts aggregated here are what the dashboard's **Scan Funnel** panel shows
per cycle (and what the `mm_scan_funnel` log event records).

---

## Stage 1 — Cycle gates (skip entire scan)

If either of these is true, the whole cycle is skipped — no per-pair scan runs.

| # | Gate | Default | Log event |
|---|------|---------|-----------|
| 1 | Weekend check | Sat/Sun skipped | `mm_engine_weekend_skip` |
| 2 | Dead-zone session | No active trading session | `mm_engine_dead_zone_skip` |

---

## Stage 2 — Pair-selection filters (before per-pair scan)

These run once per cycle inside `_get_pairs()` to decide which pairs are scanned.

| # | Gate | Default | Notes |
|---|------|---------|-------|
| 3 | Min 24h volume | $5M USDT | `_get_pairs(min_volume_usd=5_000_000)` |
| 4 | Exchange quality filter | on | Excludes stables, leveraged tokens, etc. |
| 5 | Already open position | — | Excluded from scan (no pyramiding) |
| 6 | On cooldown | 4h | `SYMBOL_COOLDOWN_HOURS = 4` — applies after any close |

---

## Stage 3 — Per-pair analysis (15 filters in `_analyze_pair`)

This is where the bulk of rejections happen. These are the reasons you see in
the dashboard **Scan Funnel** panel. All use the centralized `_reject()`
helper so the per-cycle counter stays accurate.

| # | Gate | Threshold | Reject event |
|---|------|-----------|--------------|
| 7 | Candle fetch | exchange call succeeded | `mm_reject_candle_fetch` |
| 8 | Enough candles | ≥ 50 1h bars | `mm_reject_insufficient_candles` |
| 9 | M/W formation present | any detected | `mm_reject_no_formation` |
| 10 | Formation quality | ≥ 0.4 | `mm_reject_low_formation_quality` |
| 11 | Level not exhausted | current level < 3 | `mm_reject_level_too_advanced` |
| 12 | Not in FMWB phase | — | `mm_reject_fmwb_phase` |
| 13 | Not in Friday trap | — | `mm_reject_friday_trap` |
| 14 | Valid weekly-cycle phase | one of `FORMATION_PENDING`, `LEVEL_1`, `LEVEL_2`, `LEVEL_3`, `BOARD_MEETING_1`, `BOARD_MEETING_2` | `mm_reject_wrong_phase` |
| 15 | Direction matches weekly bias | FMWB false-move → opposite direction | `mm_reject_against_weekly_bias` |
| 16 | SL distance | ≤ 5% from entry | `mm_reject_sl_too_wide` |
| 17 | Valid L1 target (50 EMA in position) | required | `mm_reject_no_l1_target` |
| 18 | Non-zero risk | SL ≠ entry | `mm_reject_zero_risk` |
| 19 | Risk:Reward | ≥ 1.0 (L1 first, falls back to L2) | `mm_reject_low_rr` |
| 20 | Confluence score | ≥ 40% | `mm_reject_low_confluence` |
| 21 | Retest conditions met | ≥ 2 of 4 | `mm_reject_low_retest` |

---

## Stage 4 — Entry-time gates (after signals are ranked)

Signals that pass all of stage 3 are sorted by confluence score and then
processed through these gates inside `_process_entries()`.

| # | Gate | Default | Reject event |
|---|------|---------|--------------|
| 22 | Max open positions | 6 | Silent `break` (won't attempt entry) |
| 23 | Account margin utilization | ≤ 60% of balance | `mm_reject_margin_limit` |

---

## Behind the scenes

### Retest conditions (gate 21)

At least **2 of these 4** must be true:

- Price retest of the formation neckline
- 50 EMA break confirmed with volume
- Stopping volume with degrading trend
- Unrecovered displacement vector present

### Confluence factors (feed gate 20)

The confluence scorer (`src/strategy/mm_confluence.py`) combines these:

- Session changeover (multi-session formation)
- At key level (HOW / LOW / HOD / LOD)
- EMA alignment
- 50 EMA break with volume
- Stopping volume + degrading trend
- Open-interest context
- Correlation context
- Moon-phase / lunar context

### Threshold constants (module-level defaults)

| Constant | Value | Used in gate |
|----------|-------|--------------|
| `MIN_CONFLUENCE_PCT` | 40.0 | 20 |
| `MIN_RR_AGGRESSIVE` | 1.0 | 19 (default `min_rr`) |
| `MIN_RR` | 3.0 | Confluence scorer internal |
| `MIN_FORMATION_QUALITY` | 0.4 | 10 |
| `MIN_RETEST_CONDITIONS` | 2 | 21 |
| `MAX_SL_DISTANCE_PCT` | 5.0 | 16 |
| `MAX_MM_POSITIONS` | 6 | 22 |
| `MAX_MARGIN_UTILIZATION` | 0.60 | 23 |
| `SYMBOL_COOLDOWN_HOURS` | 4 | 6 |

### Runtime-overridable via `/mm/settings`

Saved to `engine_state.config_overrides.mm_engine_settings` in Supabase:

- `mm_risk_pct`
- `mm_leverage`
- `mm_min_rr`
- `mm_min_confluence`
- `mm_min_formation_quality`
- `mm_max_sl_pct`
- `mm_max_positions`
- `mm_cooldown_hours`
- `mm_scan_interval`
- `scanning_active`

---

## Why 0 trades per cycle is normal

To produce a single trade, **every gate 7–21 must pass on the same scan for
the same symbol, AND gates 22–23 must not block at entry time**. The strategy
is designed to be high-conviction, not high-frequency.

Typical production funnel shape (from the dashboard):

```
90 pairs scanned  →  0 signals found
Rejected:
  no_formation        57  (63%)  ← gate 9
  no_l1_target        19  (21%)  ← gate 17
  sl_too_wide          6  (7%)   ← gate 16
  against_weekly_bias  4  (4%)   ← gate 15
  low_rr               2  (2%)   ← gate 19
  fmwb_phase           1  (1%)   ← gate 12
  low_confluence       1  (1%)   ← gate 20
```

Gates 9 and 17 alone eliminate ~85% of candidates every cycle. This matches
MM Method design — most symbols don't have a clean M/W formation with the
50 EMA positioned as the first target at any given moment.

---

## How to watch selectivity live

- **Dashboard**: "Scan Funnel" panel (auto-refreshes with `/api/mm/status`)
- **Logs**: `fly logs -a tarakta-mm | grep mm_scan_funnel`
- **Per-symbol logs**: `fly logs -a tarakta-mm | grep mm_reject_`

When a trade qualifies, `mm_signal_generated` fires in the logs, a row appears
in the **Open Positions** panel, and the funnel's `signals_found` count turns
green.
