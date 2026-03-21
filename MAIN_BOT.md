# Main Bot (TradingEngine)

Complete reference for the primary trading engine — the production bot that scans, scores, and executes trades on Binance.

---

## Overview

The Main Bot is the core trading engine built around a **Post-Sweep Displacement** strategy. It identifies liquidity sweeps (stop hunts) on crypto futures markets, confirms them with volume displacement, and enters positions in the reversal direction. It runs on a 15-minute scan cycle with 60-second position monitoring, and includes progressive take-profit management, ATR-based trailing stops, sentiment filtering, optional LLM analysis, and adaptive self-tuning.

**Key characteristics:**
- Executes real or paper orders on Binance Futures via CCXT
- Scans 100+ quality-filtered perpetual swap pairs
- Dual-path signal detection: Sweep path (primary) and Breakout path (fallback)
- 100-point confluence scoring system with a 60-point entry threshold
- Progressive 3-tier take-profit with trailing stop runner
- Fully async Python architecture

---

## Architecture

```
Scheduler (15m primary / 60s monitor ticks)
    |
    v
AltcoinScanner
    |-- MarketStructureAnalyzer (trend, BOS/CHoCH)
    |-- SessionAnalyzer (kill zones, session ranges)
    |-- VolumeAnalyzer (displacement, RVOL)
    |-- SweepDetector (Asian/London/NY/Swing levels)
    |-- BreakoutDetector (session breakouts)
    |-- PullbackAnalyzer (retracement quality)
    |-- PostSweepEngine (confluence scoring)
    |-- LeverageAnalyzer (OI, funding, L/S ratio)
    |
    v
Signal Pipeline
    |-- AdaptiveThreshold (dynamic entry bar)
    |-- SentimentFilter (CryptoBERT + news)
    |-- SplitTestManager (A/B assignment)
    |-- LLMTradeAnalyst (Claude approve/reject)
    |-- ConsensusMonitor (portfolio + BTC alignment)
    |-- EntryRefiner (5m sweep reclaim)
    |-- WatchlistMonitor (near-miss graduation)
    |
    v
OrderExecutor
    |-- RiskManager (position sizing, validation)
    |-- CircuitBreaker (drawdown protection)
    |-- PortfolioTracker (balance, equity, P&L)
    |
    v
PositionMonitor
    |-- Stop-loss checks
    |-- Progressive TP tier fills
    |-- ATR trailing stop
```

---

## Signal Detection

### Scan Cycle

Every **15 minutes** (aligned to candle close boundaries), the scanner evaluates all tradeable pairs:

1. Fetches the tradeable pair list from Binance Futures (quality-filtered)
2. Processes pairs in batches of **8** with **1.5s** delay between batches
3. Each pair goes through the full analysis pipeline
4. Qualifying signals are enriched with leverage data (OI, funding, L/S ratio)
5. Near-miss signals (score 35-59) are sent to the Watchlist Monitor

### Timeframes Analysed

| Timeframe | Used For |
|-----------|----------|
| **1H** | Primary: sweep detection, displacement, pullback, ATR, session ranges |
| **4H** | Higher-timeframe trend confirmation, key levels |
| **1D** | Higher-timeframe trend confirmation, key levels |
| **5M** | Entry refinement (post-queue), watchlist graduation |

### Candle Management

The Candle Manager uses a **database-first caching strategy**:
- Checks DB cache first; if >= 90% of candles exist and data is fresh, returns cached data
- Stale cache: fetches only new candles since the last cached timestamp (incremental, limit 50)
- Empty cache: full API fetch, then caches to DB
- On API failure: falls back to whatever cached data exists

---

## Analysis Pipeline

For each pair, the following analysis runs in sequence:

### 1. Market Structure (smartmoneyconcepts library)

Analyses Break of Structure (BOS) and Change of Character (CHoCH) patterns:
- **Trend determination:** CHoCH is a reversal signal, BOS is continuation; most recent event wins
- **Key levels:** Latest swing high and swing low extracted per timeframe
- **Structure strength:** 0.0-1.0 score based on BOS direction consistency (last 5 events)
- **Swing lengths:** 15m=5, 1h=10, 4h=10, 1d=15

### 2. Session Analysis

Identifies which trading session is active and extracts manipulation ranges:

| Session | Hours (UTC) | Kill Zone |
|---------|-------------|-----------|
| Asian | 00:00 - 08:00 | N/A |
| London | 07:00 - 16:00 | 07:00 - 10:00 |
| New York | 12:00 - 21:00 | 12:00 - 15:00 |

Post-kill zones: London 10:00-12:00, NY 15:00-17:00.

Session ranges extracted:
- **Asian range:** 00:00-08:00 high/low
- **London range:** 07:00-12:00 high/low
- **NY range:** 12:00-17:00 high/low

### 3. Volume / Displacement Analysis

- **RVOL:** Current volume relative to 20-period average
- **Volume trend:** "increasing" (>20% change), "decreasing" (<-20%), or "flat"
- **Displacement detection:** Body > 1.5x ATR AND volume > 1.5x average (checks last 8 candles)
- **Volume sustainability:** If volume trend is "decreasing" after displacement, the displacement is invalidated

### 4. Sweep Detection

Identifies liquidity sweeps where price wicks through a key level then closes back above/below it.

**Detection logic:**
- Bullish sweep: Low pierces below level, close recovers above it
- Bearish sweep: High pierces above level, close recovers below it

**Level priority:** Asian > London > NY > Swing. Level types: `asian_low`, `asian_high`, `london_low`, `london_high`, `ny_low`, `ny_high`, `swing_low`, `swing_high`.

If displacement was detected, the sweep detector prefers sweeps aligned with the displacement direction.

### 5. Breakout Detection (Fallback Path)

If no qualifying sweep is found, the scanner tries the breakout path:
- Requires price closing above/below a session level for **2+ candles**
- Requires volume confirmation (RVOL >= 1.5x)
- Requires minimum ATR distance of 0.3x ATR
- Priority: London/NY > Asian > Swing
- Prefers volume-confirmed breakouts, then candles-held as tiebreaker

### 6. Pullback Analysis

After displacement is confirmed, checks for a healthy retracement:
- **< 20% retracement:** "waiting" (still in thrust, no pullback yet)
- **20-78% retracement:** "optimal" (valid entry zone)
- **> 78% retracement:** "failed" (setup invalidated)

### 7. HTF Direction Resolution

Combines 4H and 1D trend analysis:
- **4H priority** over 1D (closer to trading timeframe)
- If both timeframes are "ranging," direction is "ranging"
- If only one timeframe has a trend, that trend is used

---

## Confluence Scoring

### Sweep Path (Primary) - Max 100 + 10 Bonus

| Component | Points | Condition |
|-----------|--------|-----------|
| **Sweep Detected** | 35 | **REQUIRED** — no sweep = score 0, analysis stops |
| **Displacement Confirmed** | 25 | Volume displacement in sweep direction, not declining |
| **Pullback Confirmed** | 10 | Retracement between 20-78% (status = "optimal") |
| **HTF Alignment** | 15 | 4H+1D both aligned = 15pts, 4H only = 10pts (70%), 1D only = 7.5pts (50%) |
| **Timing (Post-Kill Zone)** | 15 | Currently in a post-kill zone window |
| **Leverage Alignment** | 10 | Bonus scored by scanner (see below) |

**Entry threshold: 60 points.** The classic qualifying path is Sweep (35) + Displacement (25) = 60.

### Breakout Path (Fallback) - Max 90

| Component | Points | Condition |
|-----------|--------|-----------|
| **Breakout Confirmed** | 25 | **REQUIRED** — closes above/below level for 2+ candles |
| **Volume Confirmed** | 20 | **REQUIRED** — RVOL >= 1.5x |
| **HTF Alignment** | 15 | Same logic as sweep path |
| **Timing** | 10 | In post-kill zone |
| **Candles Held Bonus** | 10 | 3 pts per candle above the minimum of 2 (max 10) |
| **ATR Distance Bonus** | 10 | Scales from 0.3 ATR distance |

**Breakout threshold: 45 points.**

### Leverage Scoring (Bonus, Futures Only)

Applied after base scoring for qualifying signals:
- Sweep aligns with crowded side: **+5**
- Crowding intensity > 0.8: **+3** (or > 0.5: **+2**)
- Judas swing probability > 0.6: **+2**
- Maximum bonus: **+10**

---

## Leverage & Crowding Analysis

Analyses market positioning data from Binance Futures:

### Data Sources
- **Open Interest:** Total USD value of open contracts
- **Funding Rate:** 8-hourly payment between longs and shorts
- **Long/Short Ratio:** Proportion of accounts long vs short

### Crowding Detection

Weighted model: Funding (60% weight) + L/S ratio (40% weight).

| Metric | Extreme | Moderate |
|--------|---------|----------|
| Funding rate | >= 0.05% per 8h | >= 0.02% per 8h |
| L/S ratio | >= 2.0 | >= 1.3 |

Extreme funding overrides contradictory ratio data. Crowding intensity ranges 0.0-1.0.

### Sweep Alignment

When the crowd is positioned heavily on one side, sweeps that hunt that side's liquidity are more likely to reverse:
- Bullish sweep + longs crowded = **aligned** (market makers hunting long stops)
- Bearish sweep + shorts crowded = **aligned** (market makers hunting short stops)

### Judas Swing Probability

Probability of a deceptive move during kill zones that traps the crowded side:
- Kill zone + crowding > 0.4 = high probability (0.4 + intensity * 0.6)
- Post-kill zone = lower probability
- No session/no crowding = zero

### Liquidation Level Estimation

Estimates liquidation clusters at standard leverage tiers (5x, 10x, 25x, 50x, 100x) using 0.4% maintenance margin.

---

## Signal Processing Pipeline

After scanning produces qualifying signals, each signal passes through these gates:

### 1. Adaptive Threshold

The entry threshold (default 60) self-adjusts based on recent performance:

| Win Rate (last 20 trades) | Action | Step |
|---------------------------|--------|------|
| < 40% | Raise threshold | +2.0 points |
| 40-55% | No change | 0 |
| > 55% | Lower threshold | -1.5 points |

**Bounds:** minimum 55, maximum 85. Requires at least 5 closed trades before adjusting.

### 2. Sentiment Filter (CryptoBERT)

Fetches crypto news from CryptoCompare, analyses with CryptoBERT model:
- **Strong contra-sentiment blocks the trade:** Sentiment <= -3.0 blocks longs, >= +3.0 blocks shorts
- **Score adjustment:** -10 to +5 based on sentiment alignment
- **Critical event detection:** Zero-shot classification for hack/rug pull/delisting (>= 70% confidence)
- **Cache:** 15-minute TTL per symbol
- **Fallback:** Keyword-based scoring when HuggingFace API is unavailable

### 3. A/B Split Test

Deterministic SHA-256 hash assignment:
- Hash of `symbol:timestamp` determines group
- `llm_split_ratio` (default 0.5) splits between "control" and "llm" groups
- Control group: proceeds with confluence score only
- LLM group: additionally passes through Claude for approve/reject

### 4. LLM Trade Analyst (Claude Haiku)

Sends full signal context to Claude for evaluation:
- Symbol, direction, entry price, scores, SL/TP, R:R, signal reasons, key levels, order block/FVG context, sentiment, recent bot performance
- Returns: approve/reject, confidence (0-100), reasoning, optional SL/TP adjustments
- **Minimum confidence override:** If approved but confidence < 40 (configurable), overrides to rejection
- **SL/TP validation:** Suggested adjustments must be on correct side of entry, within 15% distance
- **Resilience:** Exponential backoff on failure; `llm_fallback_approve` controls fallback behaviour

### 5. Market Consensus Check

Evaluates whether a new position aligns with the portfolio and BTC:

**Portfolio Bias:** Counts profitable longs vs profitable shorts across all open positions (unrealized P&L > $0.00). Minimum 3 open positions required to activate.

**BTC/USDT Trend:** Analyses BTC/USDT 4H candles via MarketStructureAnalyzer (cached for 2 minutes).

**Weighted Penalty:**

| Condition | Penalty |
|-----------|---------|
| Portfolio bias alone disagrees with signal direction | -10 points |
| Portfolio AND BTC both disagree | -15 points total |
| Only BTC disagrees (portfolio neutral/agrees) | -10 points |
| Everything agrees or insufficient data | 0 points |

If the penalty drops the adjusted score below the threshold, the signal is **queued in the Consensus Monitor** (not hard-blocked). It is removed from all other queues (refiner, watchlist).

**Consensus Monitor Queue:**
- Re-checked every 60 seconds
- If consensus clears (penalty drops to 0): signal graduates with original score
- If opposite direction now aligns with both portfolio and BTC: signal graduates with **flipped direction**
- Expires after 30 minutes
- Max queue size: 10

### 6. Entry Refiner (5-Minute Precision)

When a 1H sweep signal has an associated `sweep_result`, it is routed to the Entry Refiner instead of immediate entry:

**Refinement process:**
- Monitors 5-minute candles for **sweep level reclaim** (price closing back above/below the swept level)
- Requires volume confirmation: RVOL >= 1.3x on the reclaim candle
- Bonus: rejection wick (wick/body >= 0.5) provides additional confirmation
- On confirmation: graduates with refined entry price (closer to true reversal)
- On expiry (30 minutes): graduates with current price as fallback (still enters, never dropped)
- Max queue: 5, check interval: 60 seconds

### 7. Watchlist Monitor (Near-Miss Graduation)

Signals scoring 35-59 (below threshold but showing potential) are added to the watchlist:

- Independent async loop checking every **2.5 minutes**
- Monitors 5-minute candles for displacement/pullback/volume confirmation
- On graduation: signal rejoins the main execution queue with updated score
- Expiry: **3 hours**
- Max size: **10 symbols**

---

## Risk Management

### Position Sizing

```
risk_amount     = balance * max_risk_pct (10%)
quantity        = risk_amount / SL_distance
notional        = quantity * entry_price
fee_multiplier  = 1.002 (futures: 2 * 0.04% fee + 0.1% buffer)
total_cost      = notional * fee_multiplier
margin_required = total_cost / leverage
```

**Caps applied in order:**
1. Max position size: 25% of balance (`max_position_pct`)
2. Minimum order: exchange minimum ($5 for futures)
3. Leverage safety: SL must be within 80% of liquidation distance

### Trade Validation

Before any entry, the Risk Manager validates:

| Check | Threshold | Behaviour |
|-------|-----------|-----------|
| Max concurrent positions | 0 (unlimited) | Skips check when set to 0 |
| Max exposure | 100% of balance | Margin-based for futures |
| Daily drawdown | 10% of daily start balance | Rejects new entries |
| Duplicate symbol | N/A | No duplicate positions |
| Symbol cooldown | 2 hours | After previous exit on same symbol |
| Minimum balance | $10 | Below = no trading |
| Shorting support | Market-dependent | Rejects bearish on spot accounts |

### Circuit Breaker

Two-tier drawdown protection:

| Level | Trigger | Severity | Recovery |
|-------|---------|----------|----------|
| **Daily** | >= 10% drawdown from day start | Warning | Auto-resume at midnight + 24h |
| **Total** | >= 15% drawdown from peak balance | Critical | Manual restart required |

---

## Order Execution

### Entry Orders

1. **Stop Loss calculation:** Behind sweep extreme (wick tip) + 3% buffer. Minimum floor: 2% of entry. ATR fallback: 2.5x ATR or 4% of entry.
2. **Take Profit calculation:** Opposite liquidity pool (swing highs/lows from 1H/4H/1D). Enforces minimum R:R distance (2.0).
3. **Liquidity gate (pre-entry):**
   - Spread: reject if > 0.2%
   - Order book depth (top 5 levels): reject if < $1,000
   - 24H volume: reject if < $20M
   - Position vs volume: reject if notional > 0.1% of daily volume
4. **Order type:** Limit order at best bid (longs) or best ask (shorts)
5. **Fill verification (live mode):** Polls order status (2s delay, 3 retries at 3s). Cancels unfilled orders. Falls back to market order for exits.

### Progressive Take-Profit Tiers

| Tier | Target | Size | On Fill |
|------|--------|------|---------|
| **TP1** | 1R (SL distance) | 33% | Move SL to breakeven |
| **TP2** | 2R | 33% | Move SL to TP1 price |
| **TP3** | Trailing | 34% | ATR-based trailing stop |

---

## Position Monitoring

Every **60 seconds**, each open position is evaluated:

### Stop Loss Check
- Longs: current price <= stop loss
- Shorts: current price >= stop loss
- **Smart labelling:** If SL was moved up from original (by trailing/TP progression), the exit is labelled "trailing_stop" instead of "sl_hit" (since it is a profitable exit)

### Trailing Stop

- **Activation:** After TP1 hit OR unrealized R:R reaches `trailing_activation_rr` (default 1.0)
- **Trail distance:** ATR * 1.5 (configurable). Fallback: high water mark * 2%
- **Tier-aware width:**
  - Before TP2: wider trail = `max(ATR_trail, SL_distance)` (gives room to reach TP2)
  - After TP2: tighter trail = ATR trail only (protects the runner)
- **Ratchet:** Trailing SL only moves in the favourable direction (never moves back)

---

## Symbol Universe

### Quality Filter (Futures)

The scanner only trades pairs from the **QUALITY_BASES** whitelist (100+ symbols across categories):

| Category | Symbols |
|----------|---------|
| **Layer 1s** | BTC, ETH, SOL, ADA, AVAX, DOT, ATOM, NEAR, SUI, APT, SEI, TIA, INJ, FTM, ALGO, EGLD, HBAR, ICP, FIL, TON, TRX, XLM, XRP, EOS, FLOW, MINA, CELO, ONE, KAVA, VET, THETA |
| **Layer 2s / Scaling** | MATIC, POL, ARB, OP, IMX, STRK, MNT, METIS, ZK, MANTA, BLAST |
| **DeFi** | UNI, AAVE, MKR, SNX, COMP, CRV, SUSHI, YFI, DYDX, GMX, PENDLE, JUP, RAY, JTO, PYTH, LDO, RPL, FXS, LQTY, BAL, 1INCH |
| **Infrastructure / Oracles** | LINK, GRT, API3, BAND, REN |
| **Storage / Compute / AI** | AR, RENDER, AKT, TAO, FET, AGIX |
| **Gaming / Metaverse** | AXS, SAND, MANA, GALA, ENJ, RONIN, PIXEL |
| **Exchange Tokens** | BNB, OKB, CRO |
| **Privacy** | XMR, ZEC |
| **Interoperability** | RUNE, ZRO, W, WOO, STX |
| **Other Established** | LTC, BCH, ETC, DOGE, SHIB, PEPE, BONK, WIF, FLOKI, ONDO, ENS, SSV, EIGEN, ETHFI, WLD, JASMY, CHZ, MASK, BLUR |

Additional filters: minimum 24H volume >= $20M, USDT quote currency, linear perpetual swaps only.

---

## Self-Tuning Systems

### Adaptive Threshold

Automatically adjusts the entry score threshold based on a rolling window of the last 20 trades. Raises the bar during losing streaks, lowers it during winning streaks. Bounded between 55 and 85.

### Dynamic Weight Optimiser

Adjusts confluence component weights based on per-component win rates:
- Components that contribute to winning trades get more weight
- Exponential decay (half-life 30 trades) so recent performance matters more
- Recalculates every 10 closed trades
- Max step of 2.0 per recalculation (prevents wild swings)
- Weights always normalise to sum to 95 (+ 5 fixed R:R bonus = 100)
- Currently **disabled** by default (`dynamic_weights_enabled: False`)

---

## Multi-Market Support

The engine supports multiple exchange connectors simultaneously:

| Market | Connector | Trading Hours | Shorting |
|--------|-----------|---------------|----------|
| **Crypto** | `binance_futures` | 24/7 | Yes |
| **US Stocks** | `yfinance_stocks` | 9:30-16:00 ET, Mon-Fri | No |
| **Commodities** | `yfinance_commodities` | 8:20-13:30 ET, Mon-Fri | No |

Each market gets its own `TradingEngine` instance. Non-crypto markets use `TradingHoursManager` to skip scans when markets are closed.

---

## State Persistence

The engine state is persisted to Supabase on every significant event:

| Component | Persistence |
|-----------|-------------|
| Portfolio (balance, positions, P&L) | `portfolio_snapshots` table |
| Trades | `trades` table with full lifecycle |
| Partial exits | `partial_exits` table |
| Adaptive threshold | `engine_state` table (config_overrides) |
| Dynamic weights | `engine_state` table (config_overrides) |
| Entry refiner queue | `engine_state` table (config_overrides) |
| Consensus monitor queue | `engine_state` table (config_overrides) |
| Scanning active state | `engine_state` table (config_overrides) |

On startup, all state is restored from the database.

---

## Dashboard Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/portfolio` | GET | login | Latest portfolio snapshot + 7-day history |
| `/trades/open` | GET | login | All open trades |
| `/stats` | GET | login | Trade statistics (wins, losses, P&L, exit reasons) |
| `/signals/recent` | GET | login | Last 20 detected signals |
| `/unrealized-pnl` | GET | login | Live prices + unrealized P&L per position |
| `/trades/{id}/detail` | GET | login | Full trade record + partial exit history |
| `/nuke` | POST | admin | Emergency close ALL positions at market |
| `/main/settings` | GET | login | Current main bot settings |
| `/main/settings` | POST | admin | Update margin_pct (5-40%), leverage (1-100) |
| `/main/begin` | POST | admin | Start scanning |
| `/main/stop` | POST | admin | Pause scanning |
| `/main/status` | GET | login | Scanning status |
| `/refiner/main` | GET | login | Entry refiner queue contents |
| `/consensus/main` | GET | login | Consensus monitor queue contents |
| `/reset/main` | POST | admin | Delete all trades, reset balance and state |
| `/split-test` | GET | login | A/B test comparison (control vs LLM) |
| `/backtests` | GET | login | List backtest result files |
| `/backtests/{file}` | GET | login | Get specific backtest result |

**Authentication:** Two roles — admin (full control) and viewer (read-only). bcrypt password verification.

---

## Configuration Reference

### Core Trading

| Setting | Default | Description |
|---------|---------|-------------|
| `trading_mode` | `"paper"` | `"paper"` or `"live"` |
| `entry_threshold` | 60.0 | Minimum confluence score for entry |
| `max_risk_pct` | 0.10 | Risk per trade (10% of balance) |
| `max_position_pct` | 0.25 | Max position size (25% of balance) |
| `max_exposure_pct` | 1.0 | Max total margin deployed (100%) |
| `max_concurrent` | 0 | Max open positions (0 = unlimited) |
| `max_daily_drawdown` | 0.10 | Daily drawdown limit (10%) |
| `circuit_breaker_pct` | 0.15 | Total drawdown from peak (15%) |
| `leverage` | 10 | Default leverage |
| `min_rr_ratio` | 2.0 | Minimum risk:reward ratio |
| `sl_buffer` | 0.03 | SL buffer beyond sweep level (3%) |
| `min_sl_pct` | 0.02 | Minimum SL distance (2%) |
| `cooldown_hours` | 2.0 | Cooldown after exit on same symbol |
| `max_daily_trades` | 15 | Maximum trades per day |
| `min_trade_usd` | 150.0 | Minimum trade notional |
| `initial_balance` | 10000.0 | Starting balance (paper mode) |

### Scanning

| Setting | Default | Description |
|---------|---------|-------------|
| `scan_interval_minutes` | 15 | Primary scan interval |
| `min_volume_usd` | 20,000,000 | Minimum 24H volume |
| `quality_filter` | True | Use QUALITY_BASES whitelist |
| `max_spread_pct` | 0.002 | Max spread (0.2%) |
| `min_ob_depth_usd` | 1000.0 | Min order book depth |
| `max_position_volume_pct` | 0.001 | Max position vs daily volume (0.1%) |

### Progressive Take-Profit

| Setting | Default | Description |
|---------|---------|-------------|
| `tp_tiers_enabled` | True | Enable 3-tier TP |
| `tp1_rr` | 1.0 | TP1 at 1R |
| `tp1_pct` | 0.33 | Close 33% at TP1 |
| `tp2_rr` | 2.0 | TP2 at 2R |
| `tp2_pct` | 0.33 | Close 33% at TP2 |
| `tp3_pct` | 0.34 | Trail 34% as runner |
| `move_sl_to_be_after_tp1` | True | Move SL to breakeven after TP1 |
| `trailing_activation_rr` | 1.0 | Trailing activates at this R:R |
| `trailing_atr_multiplier` | 1.5 | Trail width = ATR * this |

### Monitors

| Setting | Default | Description |
|---------|---------|-------------|
| `watchlist_enabled` | True | Enable near-miss watchlist |
| `watchlist_min_score` | 35 | Minimum score for watchlist |
| `entry_refiner_enabled` | True | Enable 5m entry refinement |
| `entry_refiner_max_queue` | 5 | Max signals in refiner |
| `entry_refiner_expiry_minutes` | 30.0 | Refiner expiry time |
| `consensus_enabled` | True | Enable market consensus check |
| `consensus_portfolio_penalty` | 10.0 | Penalty for portfolio disagreement |
| `consensus_btc_penalty` | 15.0 | Total penalty when BTC also disagrees |
| `consensus_min_positions` | 3 | Min positions before consensus applies |
| `consensus_max_queue` | 10 | Max signals in consensus queue |
| `consensus_monitor_expiry_minutes` | 30.0 | Consensus queue expiry |

### LLM & Sentiment

| Setting | Default | Description |
|---------|---------|-------------|
| `llm_enabled` | False | Enable AI analysis |
| `llm_model` | `"gpt-5.4-mini"` | OpenAI model |
| `llm_split_ratio` | 0.5 | Fraction of trades sent to LLM |
| `llm_min_confidence` | 40.0 | Minimum LLM confidence to approve |
| `llm_fallback_approve` | True | Approve on API failure |
| `llm_timeout_seconds` | 15.0 | API timeout |
