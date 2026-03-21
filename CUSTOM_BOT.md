# Custom Bot (FlippedTrader)

Complete reference for the custom trading bot — an independent paper-trading engine that runs alongside the main bot with its own balance, positions, and configurable direction logic.

---

## Overview

The Custom Bot is a **fully simulated** shadow trader built on the `FlippedTrader` class. It uses the same Post-Sweep Displacement strategy as the main bot but with a simplified pipeline (no sentiment, no LLM, no adaptive threshold) and a unique **direction flip** capability. Its primary purpose is to trade the opposite side of liquidity sweeps — when market makers sweep longs, it goes short, and vice versa.

**Key characteristics:**
- Purely paper-traded — no real exchange orders, uses simulated fills at ticker prices
- Independent $10,000 paper balance with its own P&L tracking
- Configurable direction modes: always flip, smart flip, or normal
- 20-minute scan cycle (offset from main bot's 15-minute cycle)
- Starts **paused** — user must click "Begin" on the dashboard
- Full progressive TP and trailing stop management
- Includes Entry Refiner and Consensus Monitor

---

## How It Differs From the Main Bot

| Aspect | Main Bot | Custom Bot |
|--------|----------|------------|
| **Execution** | Real or paper exchange orders | Purely simulated (ticker prices) |
| **Balance** | From exchange or paper config | Independent $10,000 paper balance |
| **Scan interval** | 15 minutes | 20 minutes |
| **Strategy pipeline** | Full (sentiment, LLM, adaptive threshold, watchlist) | Simplified (sweep + displacement + pullback + HTF + timing only) |
| **Direction** | Direct signal direction | Configurable flip modes |
| **Sentiment filter** | Yes (CryptoBERT) | No |
| **LLM analysis** | Yes (OpenAI) | No |
| **Adaptive threshold** | Yes (55-85 range) | No (fixed 60) |
| **Watchlist monitor** | Yes (near-miss to 5m) | No |
| **Circuit breaker** | Yes | No |
| **Entry refiner** | Yes | Yes |
| **Consensus monitor** | Yes | Yes |
| **Fee simulation** | Exchange fees (0.04% futures) | Simulated 0.04% taker fee |
| **Start state** | Paused (user clicks Start) | Paused (user clicks Begin) |

---

## Architecture

```
Independent Async Loop (20-min scan / 60-sec monitor)
    |
    v
_run_scan()
    |-- exchange.get_tradeable_pairs() (quality filtered)
    |-- _analyze_pair() per symbol (batches of 16)
    |   |-- MarketStructureAnalyzer
    |   |-- SessionAnalyzer
    |   |-- VolumeAnalyzer (with sustainability check)
    |   |-- SweepDetector
    |   |-- PullbackAnalyzer
    |   |-- ATR computation
    |   |-- HTF direction resolution
    |   |-- Scoring (sweep=35, displacement=25, pullback=10, HTF=15, timing=15)
    |
    |-- Filter by FLIPPED_THRESHOLD (60)
    |-- Sort by score descending
    |
    |-- Per signal:
    |   |-- Position/cooldown/balance/exposure checks
    |   |-- EntryRefiner queue (if sweep detected)
    |   |-- ConsensusMonitor check (portfolio + BTC penalty)
    |   |-- _try_enter() (immediate entry)
    |
    v
_monitor_loop() (every 60 seconds)
    |-- monitor_positions() (SL/TP/trailing)
    |-- _process_refined_entries() (entry refiner graduates)
    |-- _process_consensus_entries() (consensus graduates)
```

---

## Direction Flip Modes

The core differentiator of the custom bot. Three modes available:

### Always Flip (Original)
Every signal direction is inverted:
- Bullish sweep signal becomes **Short**
- Bearish sweep signal becomes **Long**

The logic: if market makers just swept longs (bullish sweep), the "smart money" move may actually be to go short because the sweep was the reversal itself and price will continue down. This is contrarian to the main bot.

### Smart Flip (Default)
Uses the `LeverageAnalyzer` to compute a **sweep flip probability** based on 6 weighted factors:

| Factor | Weight | Logic |
|--------|--------|-------|
| Funding bias | 25% | Extreme funding in signal direction = more likely to flip |
| L/S ratio | 20% | Heavy positioning on signal side = more likely to flip |
| Crowding intensity | 20% | Higher crowding = more likely |
| Liquidation proximity | 15% | Liquidation clusters near price = fuel for reversal |
| Kill zone timing | 10% | Kill zones = higher manipulation probability |
| OI magnitude | 10% | Higher OI = more fuel (scaled to $500M) |

If the computed probability >= `flip_threshold` (default 0.50), the direction flips. Otherwise, the signal direction is followed as-is.

### Normal
Direct signal direction — behaves like the main bot but without sentiment/LLM/adaptive threshold.

---

## Analysis Pipeline

### Timeframes

| Timeframe | Used For |
|-----------|----------|
| **1H** | Sweep detection, displacement, pullback, ATR, session ranges |
| **4H** | HTF trend confirmation |

Note: 1D is dropped from the custom bot's scan for efficiency (the main bot uses 1H/4H/1D).

### Scoring (Max 100 Points)

| Component | Points | Condition |
|-----------|--------|-----------|
| **Sweep Detected** | 35 | **REQUIRED** — no sweep = score 0, analysis stops |
| **Displacement Confirmed** | 25 | Volume displacement aligned with sweep, not declining |
| **Pullback Confirmed** | 10 | Retracement 20-78% |
| **HTF Alignment** | 15 | 4H+1D aligned = 15, 4H only = 10 |
| **Timing** | 15 | In post-kill zone |

**Entry threshold: 60 points** (fixed, no adaptive adjustment).

### Volume Sustainability Check

After displacement is detected, the bot checks if volume is sustainable:
- If the volume trend is "decreasing" post-displacement, the displacement is invalidated
- This prevents entering on a single spike that is already fading

---

## Trade Entry

### Pre-Entry Checks

Each signal must pass these gates before entry:

| Check | Condition |
|-------|-----------|
| Max concurrent | 0 = unlimited (no cap) |
| Minimum balance | > $5.00 |
| Total exposure | Total margin < 100% of initial balance |
| Duplicate symbol | Not already in position |
| Cooldown | Not recently closed with SL (4-hour cooldown) |
| Not in refiner queue | Symbol not already queued |
| Price staleness | Live price within 3% of signal price |

### Stop Loss Calculation

1. **Primary:** Sweep target level +/- 3% buffer (`sl_buffer`)
2. **Minimum floor:** 2% of entry price (`min_sl_pct`)
3. **ATR fallback:** 2.5x ATR, or 4% of entry if no ATR data
4. **Validation:** SL must be on correct side of entry

### Take Profit Calculation

- Uses sweep level, key levels (swing highs/lows from 1H/4H/1D)
- Minimum TP distance = SL distance * 2.0 (`min_rr`)
- R:R validation: rejects if R:R < 2.0

### Leverage Safety

Rejects entry if SL distance > 80% of theoretical liquidation distance:
```
liq_distance = entry_price / leverage * 0.95
if sl_distance > liq_distance * 0.8: reject
```

### Position Sizing

```
risk_amount  = balance * max_risk_pct (15%)
quantity     = risk_amount / sl_distance
cost_usd     = quantity * entry_price
margin_used  = cost_usd / leverage (10x)
```

**Caps:**
- Position size: max 15% of balance (`max_position_pct`)
- Available balance check
- Minimum notional: `min_trade_usd * leverage`

### Liquidity Gate

Before entry, these market quality checks must pass:

| Check | Threshold |
|-------|-----------|
| Spread | < 0.2% |
| Order book depth (top 5 levels) | > $1,000 |
| 24H volume | > $20M |
| Position vs daily volume | < 0.1% |

### Progressive TP Tiers

| Tier | Target | Size | On Fill |
|------|--------|------|---------|
| **TP1** | 1R | 33% | Move SL to breakeven (+0.1% fee buffer) |
| **TP2** | 2R | 33% | Move SL to TP1 price |
| **TP3** | Trailing | 34% | ATR-based trailing stop |

### Fee Simulation

All entries and exits deduct a simulated taker fee:
```
SIM_FEE_RATE = 0.0004 (0.04%)
entry_fee = cost_usd * 0.0004
exit_fee  = exit_notional * 0.0004
```

---

## Position Monitoring

Every **60 seconds**, each open position is evaluated:

### Stop Loss Check
- Longs: `current_price <= stop_loss`
- Shorts: `current_price >= stop_loss`
- **Smart labelling:** If SL was moved from original (trailing or TP progression), exit labelled "trailing_stop" instead of "sl_hit"

### Progressive TP Tier Fills

Iterates through unfilled tiers, triggers `_partial_exit()` when price reaches tier target:
- Calculates partial P&L on tier quantity
- Deducts exit fee
- Reduces position quantity
- Moves SL progressively (breakeven after TP1, TP1 price after TP2)
- Returns freed margin + profit to balance
- Logs to DB

### Trailing Stop

- **Activation:** After TP1 hit OR 2R profit
- **Trail distance:** ATR * 1.5 (`trailing_atr_multiplier`). Fallback: 2% of high water mark
- **Tier-aware width:**
  - Before TP2: wider = `max(ATR_trail, SL_distance)` (room to reach TP2)
  - After TP2: tighter = ATR trail only (protect the runner)
- **Ratchet:** Only moves in favourable direction

### Cooldown After Exit

- **SL exits:** 4-hour cooldown on the symbol (prevents re-entering the same failing trade)
- **TP/trailing exits:** No cooldown (profitable exits allow immediate re-entry)

---

## Entry Refiner

When a signal has an associated sweep result, it enters the refiner queue instead of immediate entry:

### Flow

1. 1H scan detects sweep signal (score >= 60) with `sweep_result`
2. Signal queued in EntryRefiner (max queue: 5)
3. Every 60 seconds, `_process_refined_entries()` checks for graduates
4. Refiner monitors 5-minute candles for **sweep level reclaim:**
   - Bullish: 5m close ABOVE swept low level
   - Bearish: 5m close BELOW swept high level
5. Volume confirmation: RVOL >= 1.3x
6. Bonus: rejection wick (wick/body >= 0.5)

### Outcomes

| Outcome | Action |
|---------|--------|
| **Confirmed** (reclaim + volume) | Graduates with refined entry price |
| **Expired** (30 minutes) | Graduates with current price as fallback |

Signals are **never dropped** — they always enter, either at the refined price or the fallback.

---

## Consensus Monitor

Evaluates whether a new position aligns with the existing portfolio direction and BTC trend:

### How It Works

1. Signal passes the 60-point threshold
2. `compute_consensus()` runs:
   - **Portfolio bias:** Batch-fetches live prices for all open positions, calculates unrealized P&L for each, counts profitable longs vs profitable shorts. Determines bias as "long", "short", "neutral", or "insufficient"
   - **BTC/USDT trend:** Analyses BTC/USDT 4H candles via MarketStructureAnalyzer. Result cached for 2 minutes.
3. Penalty applied:

| Condition | Penalty |
|-----------|---------|
| Portfolio bias disagrees, BTC neutral/agrees | -10 points |
| Both portfolio AND BTC disagree | -15 points total |
| Only BTC disagrees | -10 points |
| Agreement or insufficient data (< 3 positions) | 0 points |

4. If adjusted score drops below 60: signal queued in Consensus Monitor

### Queue Behaviour

- Re-checked every 60 seconds
- **Consensus clears:** Penalty drops to 0, signal graduates with original score
- **Direction flip:** If the opposite direction now aligns with both portfolio and BTC, signal graduates with **flipped direction** (e.g., a long signal becomes short)
- **Expired:** Dropped after 30 minutes
- Max queue: 10

---

## State Persistence

The custom bot's state is persisted to the Supabase `engine_state` table after every significant event:

### Persisted State

| Field | Description |
|-------|-------------|
| `balance` | Current paper balance |
| `peak_balance` | Highest balance reached |
| `daily_start_balance` | Balance at start of day |
| `daily_pnl` | P&L today |
| `total_pnl` | Cumulative P&L |
| `daily_trade_count` | Trades taken today |
| `scan_count` | Total scans run |
| `last_scan_time` | When last scan completed |
| `positions` | Full position data including TP tiers |
| `flip_direction` | Current direction setting |
| `margin_pct` | Current margin percentage |
| `flip_mode` | Current flip mode |
| `flip_threshold` | Current smart-flip threshold |
| `scanning_active` | Whether scanning is running |
| `entry_refiner` | Entry refiner queue state |
| `consensus_monitor` | Consensus monitor queue state |

### Restoration on Startup

1. Restore from DB: `engine_state.config_overrides["custom_trader"]`
2. Apply shared settings: `config_overrides["custom_trader_settings"]` (user's dashboard choices)
3. Reconcile from DB: `_reconcile_from_db()` syncs in-memory positions with `trades` table
4. Purge incompatible: `_purge_incompatible_positions()` removes symbols that don't match the exchange

---

## Reset Mechanism

Triggered via dashboard "Reset" button:

1. Clears all positions
2. Resets balance to $10,000
3. Resets peak, daily start, daily P&L, total P&L, trade count, scan count
4. Clears last scan time
5. Clears all symbol cooldowns
6. Clears entry refiner queue
7. Clears consensus monitor queue
8. Deletes all `custom_paper` trades from DB
9. Resets DB state
10. Saves clean state

---

## Multi-Market Support

The custom bot runs across multiple markets simultaneously:

| Market | State Key | Exchange |
|--------|-----------|----------|
| Crypto | `custom_trader` | Binance Futures |
| US Stocks | `custom_trader_stocks` | yfinance |
| Commodities | `custom_trader_commodities` | yfinance |

Each market engine creates its own `FlippedTrader` instance with a unique `state_key`. Dashboard settings (flip_direction, margin_pct, flip_mode, flip_threshold, leverage) are **shared** across all markets.

Non-crypto markets respect trading hours:
- **US Stocks:** 9:30 AM - 4:00 PM ET, Monday-Friday
- **Commodities:** 8:20 AM - 1:30 PM ET, Monday-Friday
- Scans are skipped when markets are closed

---

## Dashboard Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/custom/unrealized-pnl` | GET | login | Live unrealized P&L for all open positions (multi-market) |
| `/custom/trades/open` | GET | login | Open trades (mode = `custom_paper`) |
| `/custom/stats` | GET | login | Trade statistics |
| `/custom/settings` | POST | admin | Update settings (see below) |
| `/custom/trigger-scan` | POST | admin | Trigger immediate scan on all markets |
| `/custom/begin` | POST | admin | Start scanning (all markets) |
| `/custom/stop` | POST | admin | Pause scanning (all markets) |
| `/custom/status` | GET | login | Running status + available markets count |
| `/refiner/custom` | GET | login | Entry refiner queue (all markets, with `market` field) |
| `/consensus/custom` | GET | login | Consensus monitor queue (all markets, with `market` field) |
| `/reset/custom` | POST | admin | Full reset: delete trades, reset state, all markets |

### Settings Endpoint (`POST /custom/settings`)

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `flip_direction` | boolean | — | Legacy toggle (used by `always_flip` and fallback) |
| `margin_pct` | float | 5-40% | Position size as % of balance |
| `flip_mode` | string | `always_flip`, `smart_flip`, `normal` | Direction determination method |
| `flip_threshold` | float | 0.0-1.0 | Smart-flip probability threshold |
| `leverage` | integer | 1-100 | Leverage multiplier |

Settings persist to DB and propagate to all market engines immediately.

---

## Configuration Reference

### Module Constants (flipped.py)

| Constant | Value | Description |
|----------|-------|-------------|
| `SIM_FEE_RATE` | 0.0004 | Simulated taker fee (0.04%) |
| `FLIPPED_THRESHOLD` | 60.0 | Entry score threshold |
| `FLIPPED_MAX_CONCURRENT` | 0 | Max positions (0 = unlimited) |
| `MAX_EXPOSURE_PCT` | 1.0 | Max margin as % of initial balance (100%) |
| `BATCH_SIZE` | 16 | Pairs per concurrent scan batch |
| `BATCH_DELAY` | 0.5 | Seconds between batches |
| `SCAN_TIMEFRAMES` | ["1h", "4h"] | Timeframes to analyse |
| `REENTRY_COOLDOWN_HOURS` | 4 | Cooldown after SL exit |

### Config Settings (config.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `custom_enabled` | True | Enable custom bot |
| `custom_initial_balance` | 10,000.0 | Starting paper balance |
| `custom_leverage` | 10 | Leverage multiplier |
| `custom_flip_direction` | True | Default flip toggle |
| `custom_flip_mode` | `"smart_flip"` | Direction mode |
| `custom_flip_threshold` | 0.50 | Smart-flip probability cutoff |
| `custom_margin_pct` | 0.15 | Position size (15% of balance) |
| `custom_scan_interval_minutes` | 20 | Scan cycle interval |

### Risk Parameters (from config)

| Setting | Default | Description |
|---------|---------|-------------|
| `max_risk_pct` (uses `custom_margin_pct`) | 0.15 | Risk per trade (15%) |
| `max_position_pct` (uses `custom_margin_pct`) | 0.15 | Max position size (15%) |
| `min_rr_ratio` | 2.0 | Minimum risk:reward |
| `sl_buffer` | 0.03 | SL buffer beyond sweep (3%) |
| `min_sl_pct` | 0.02 | Minimum SL distance (2%) |
| `trailing_activation_rr` | 1.0 | Trailing activates at this R:R |
| `trailing_atr_multiplier` | 1.5 | Trail width = ATR * this |
| `min_volume_usd` | 20,000,000 | Minimum 24H volume |

### Entry Refiner Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `entry_refiner_enabled` | True | Enable 5m refinement |
| `entry_refiner_check_interval_seconds` | 60 | Check frequency |
| `entry_refiner_expiry_minutes` | 30.0 | Queue expiry time |
| `entry_refiner_max_queue` | 5 | Max queued signals |

### Consensus Monitor Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `consensus_enabled` | True | Enable consensus check |
| `consensus_portfolio_penalty` | 10.0 | Penalty for portfolio disagreement |
| `consensus_btc_penalty` | 15.0 | Total penalty when BTC also disagrees |
| `consensus_monitor_expiry_minutes` | 30.0 | Queue expiry time |
| `consensus_min_positions` | 3 | Min positions before consensus applies |
| `consensus_profitable_threshold` | 0.0 | P&L threshold for "profitable" |
| `consensus_max_queue` | 10 | Max queued signals |
