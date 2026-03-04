# Trade Travel Chill Strategy

## Philosophy

Standard SMC/ICT signals (CRT, Order Blocks, FVGs, BOS/CHoCH) are taught on every trading YouTube channel and used by millions of retail traders. Market makers see this clustered order flow and hunt it.

**The insight**: Our old signals correctly identified the levels but we entered BEFORE the manipulation — exactly where retail clusters. Market makers sweep those levels, stop everyone out, then price moves in the direction we originally predicted.

**The fix**: Instead of entering at SMC levels, we use them as a **liquidity map** showing where retail is positioned. We wait for market makers to sweep those levels, then enter AFTER the manipulation with displacement confirmation.

---

## How It Works: The Signal Pipeline

### Overview

```
Scan pairs (every 120 min)
  └─ For each pair:
     ├─ 1. Fetch 1H, 4H, 1D candles
     ├─ 2. Market structure analysis (trend + swing levels)
     ├─ 3. Session analysis (Asian range + timing)
     ├─ 4. SWEEP DETECTION on 1H ← core signal
     ├─ 5. DISPLACEMENT CHECK on 1H ← confirmation
     └─ 6. Score → enter if >= 70
```

### Step 4: Sweep Detection (`src/strategy/sweep_detector.py`)

A **completed sweep** is when a 1H candle:
- **Wicks through** a significant level (wick crosses the level)
- **Closes back** on the original side (candle body is back above/below the level)

This means market makers have grabbed the liquidity at that level and are done. They have no reason to revisit it.

**Levels checked (in priority order):**

| Level | Source | Why it matters |
|-------|--------|----------------|
| Asian Low | Session analysis (00:00-08:00 UTC) | Retail places stops below overnight lows |
| Asian High | Session analysis (00:00-08:00 UTC) | Retail places stops above overnight highs |
| 1H Swing Low | Market structure analysis | Structural support = stop-loss cluster |
| 1H Swing High | Market structure analysis | Structural resistance = stop-loss cluster |

**Bullish sweep** (expect price to go UP):
- Candle low < level (wicked below)
- Candle close > level (closed back above)
- MMs grabbed sell-side liquidity (stopped out longs), now they drive price up

**Bearish sweep** (expect price to go DOWN):
- Candle high > level (wicked above)
- Candle close < level (closed back below)
- MMs grabbed buy-side liquidity (stopped out shorts), now they drive price down

**Lookback**: Checks the last 3 completed 1H candles (indices -2, -3, -4; skips -1 as potentially incomplete).

### Step 5: Displacement Check (`src/strategy/volume.py`)

A **displacement candle** confirms institutional commitment after the sweep. Without it, the sweep might just be noise.

**Requirements (BOTH must be true):**
- Candle body > 1.5x ATR(14) — unusually large move
- Candle volume > 1.5x 20-period average volume — institutional participation

**Direction must match the sweep** — a bullish sweep needs a bullish displacement (close > open with big body + volume).

**Lookback**: Checks the last 3 candles in the 1H data.

### Step 6: Scoring (`src/strategy/confluence.py`)

Binary checklist — NOT a weighted probability system:

| Component | Points | Required? | What it checks |
|-----------|--------|-----------|----------------|
| Sweep Detected | 40 | YES | Completed liquidity sweep on 1H |
| Displacement Confirmed | 30 | YES | Large body + high volume in sweep direction |
| HTF Trend Aligned | 15 | No (bonus) | 4H and/or Daily trend matches signal direction |
| Post-Kill Zone Timing | 15 | No (bonus) | 10:00-12:00 or 15:00-17:00 UTC |

**Minimum threshold: 70** (requires sweep 40 + displacement 30 at minimum).

**Maximum score: 100** (sweep + displacement + HTF alignment + optimal timing).

---

## Entry Execution (`src/execution/orders.py`)

### Stop Loss: Behind the Sweep Wick

The sweep wick tip is the safest SL — market makers already grabbed that liquidity and have no reason to revisit it.

```
Long SL  = sweep_level * 0.995  (0.5% below the sweep wick low)
Short SL = sweep_level * 1.005  (0.5% above the sweep wick high)
```

**Fallback** (if no sweep data): ATR(14) * 2.0 from entry price. If no ATR: 3% from entry.

### Take Profit: Opposite Liquidity Pool

TP targets the other side of the range — where the remaining stop-losses sit.

- **Bullish trade**: TP at Asian high / swing high (where shorts have stops)
- **Bearish trade**: TP at Asian low / swing low (where longs have stops)

Falls back to structural swing levels (1H → 4H → 1D) if the primary target doesn't meet the minimum R:R.

**Minimum R:R: 3.0** — if no TP target achieves 3:1, the minimum distance (3x SL distance) is used.

### Position Sizing

- Risk per trade: `MAX_RISK_PCT` of balance (default 10% — this is the MAX, actual is usually less)
- Position capped by `MAX_POSITION_PCT` of balance (default 5%)

### Liquidity Gate

Before placing any order, checks:
- Spread < `MAX_SPREAD_PCT` (default 0.15%)
- Order book depth > `MIN_OB_DEPTH_USD` (default $500)

---

## Trade Management (`src/execution/monitor.py`)

### Trailing Stop (single exit — no partial TPs)

| Phase | Condition | Behavior |
|-------|-----------|----------|
| **Hold** | P&L < 2.0R | SL stays at original level |
| **Trail** | P&L >= 2.0R | SL trails at entry + (high_water_mark - entry) - 1.5x ATR |

No progressive TP tiers. One entry, one exit. Ride the full move.

---

## Risk Controls

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MAX_DAILY_TRADES` | 3 | Prevents overtrading |
| `COOLDOWN_HOURS` | 8.0 | Min time between trades on same symbol |
| `SCAN_INTERVAL_MINUTES` | 120 | How often we scan for signals |
| `MAX_DAILY_DRAWDOWN` | 10% | Circuit breaker for daily losses |
| `CIRCUIT_BREAKER_PCT` | 15% | Hard stop on total drawdown |
| `REVERSAL_ENABLED` | false | No flipping positions |
| `TP_TIERS_ENABLED` | false | No partial exits |

---

## Session Timing (UTC)

| Session | Hours (UTC) | Role |
|---------|-------------|------|
| Asian | 00:00 - 08:00 | Accumulation — defines the range |
| London Kill Zone | 07:00 - 10:00 | Manipulation — sweeps happen here |
| **Post-London KZ** | **10:00 - 12:00** | **Distribution — real move begins** |
| NY Kill Zone | 12:00 - 15:00 | Manipulation — second sweep window |
| **Post-NY KZ** | **15:00 - 17:00** | **Distribution — real move begins** |

The "Post-Kill Zone" windows are when manipulation is DONE. Entering here gets the +15 timing bonus.

---

## When a Trade WILL Fire

All of these must be true simultaneously:

1. A 1H candle swept through Asian range or swing level and closed back (within last 3 hours)
2. A 1H candle had body > 1.5x ATR AND volume > 1.5x average, in the same direction (within last 3 candles)
3. Score >= 70 (sweep + displacement = 70 minimum)
4. R:R >= 3.0 (TP distance / SL distance)
5. Daily trade count < 3
6. No same-symbol cooldown active (8 hours)
7. Spread is acceptable, order book has depth
8. Circuit breaker not triggered

## When a Trade Will NOT Fire

- Sweep without displacement (score = 40, below 70 threshold)
- Displacement without sweep (score = 0, sweep is checked first)
- Sweep + displacement in opposite directions (score = 40)
- R:R below 3.0 (TP too close or SL too far)
- Already 3 trades today
- Same symbol traded within last 8 hours
- Wide spread or thin order book

---

## Potential Concern: Is This Too Restrictive?

### The displacement gate is the strictest filter

The requirement at `volume.py:109` demands BOTH:
- Body > 1.5x ATR (large candle)
- Volume > 1.5x 20-period average (high volume)

On lower-liquidity altcoins, volume can be erratic. A genuine post-sweep move might have a big body but "normal" volume, failing the check. Crypto volume data from exchanges can also be unreliable (wash trading inflates baselines).

### Possible adjustments if no trades fire after 24-48 hours:

1. **Lower displacement thresholds** (`volume.py:109`): Change `1.5` to `1.2` for both body_ratio and vol_ratio
2. **Lower entry threshold** (`.env`): Change `ENTRY_THRESHOLD=70` to `60` (allows sweep-only entries without displacement)
3. **Lower R:R requirement** (`.env`): Change `MIN_RR_RATIO=3.0` to `2.5`
4. **Increase scan frequency** (`.env`): Change `SCAN_INTERVAL_MINUTES=120` to `60`
5. **Reduce cooldown** (`.env`): Change `COOLDOWN_HOURS=8.0` to `4.0`

### Recommended first adjustment (if needed):
Lower the displacement body threshold from 1.5x to 1.2x ATR. A candle that's 1.2x ATR with 1.5x volume is still a meaningful institutional move.

---

## File Reference

| File | Purpose |
|------|---------|
| `src/strategy/sweep_detector.py` | Core sweep detection logic |
| `src/strategy/confluence.py` | PostSweepEngine scoring (4-component checklist) |
| `src/strategy/scanner.py` | Full scan pipeline orchestration |
| `src/strategy/sessions.py` | Asian range + kill zone + post-KZ timing |
| `src/strategy/volume.py` | Displacement detection (body + volume check) |
| `src/strategy/market_structure.py` | Trend analysis + swing level extraction |
| `src/execution/orders.py` | SL/TP calculation + order placement |
| `src/execution/monitor.py` | Trailing stop management |
| `src/engine/core.py` | Main loop, daily trade limits, cooldowns |
| `src/config.py` | All configurable parameters |
