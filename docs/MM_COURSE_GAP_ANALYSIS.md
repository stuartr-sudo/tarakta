# MM Engine vs Course Content — Gap Analysis

Generated 2026-04-16. Cross-referenced ALL course transcripts (54 lessons + 2 quizzes across 5 courses) against all `src/strategy/mm_*.py` modules (14 files).

---

## PRIORITY A — Missing Trade Setups (NOT implemented at all)

### A1. Brinks Trade (Lesson 06)
**What the course teaches:**
- Highest R:R setup (6:1 to 18:1), easiest to identify
- ONLY at two 15-min candle closes: **3:30-3:45am** or **9:30-9:45am** NY
- Second leg of M/W must form at those exact times
- Must be at HOD (M) or LOD (W)
- Entry candle = hammer/inverted hammer at 3:45 or 9:45
- Also: railroad tracks within the 30-min window (3:15-3:45 or 9:15-9:45)
- Time between first and second peak: 30-90 minutes
- Scratch rule: not in profit within 2 hours → close
- Expect extended 3-level drop/rise following

**Current bot:** No `mm_brinks.py` exists. Zero Brinks detection anywhere. The TBD indicator has Brinks alerts but the bot doesn't replicate them.

**Impact:** Missing the single highest-probability, highest-R:R setup in the entire method.

### A2. NYC Reversal Trade (Lesson 10)
**What the course teaches:**
- Within first 3 hours of US session (9:30am-12:30pm NY)
- Price must be at Level 3
- HOD and LOD already formed
- Price pulled away from moving averages
- Candlestick reversal pattern (railroad, hammer, or M/W)
- Target: 50 EMA or middle of range (recovery of MM candle)
- Often triggered by news event

**Current bot:** No NYC Reversal detection. The engine scans all sessions equally — no special US-session-first-3-hours logic.

**Impact:** Missing a named setup specifically tied to Level 3 reversals during US session.

### A3. Half Batman Pattern (Lesson 15)
**What the course teaches:**
- After 3-level rise/drop, only ONE peak forms (no second peak for M/W)
- Very tight sideways consolidation (no stop hunts, equal highs/lows)
- MM already has all contracts needed (no need for 2nd peak)
- Difficult entry — no trigger signal
- Stop loss above/below the single peak, drops/rises straight into Level 2

**Current bot:** Not detected. The formation detector requires two peaks for M/W. Single-peak-with-flat-consolidation patterns are missed entirely.

### A4. Stop Hunt Entry at Level 3 (Lesson 15)
**What the course teaches:**
- At Level 3 in board meeting expecting reversal
- Enter on the stop hunt candle (must be vector candle with big wick)
- Entry 1-2 candles AFTER the stop hunt (verify wick is left alone)
- "Put your entries where the masses put their stops"
- 3 wicks to upside in board meeting → wait for stop hunt down → enter long

**Current bot:** Board meeting stop hunts are detected (`detect_stop_hunts` in `mm_levels.py`) but only for logging, not for generating entries.

### A5. 33 Trade (Lesson 12)
**What the course teaches:**
- Three rises over three days AND three hits to high on Day 3
- EMA fan-out on Day 3 (trend acceleration)
- Extended consolidation with wicks turning to top side
- Short off inverted hammer at Rise Level 3
- Target: 50 EMA first, then 200 EMA
- Pump and dump pattern — aggressive both ways

**Current bot:** The three-hits rule exists in `mm_formations.py` and the EMA fan-out detection exists in `mm_engine.py` — but they are **not connected**. `_detect_ema_fan_out` is dead code (never called). The combination of "3 hits + EMA fan-out + Level 3" is never checked.

### A6. Market Resets — Type 1, 2, 3 (Lesson 15)
**What the course teaches:**
- **Type 1:** W fails to break 50 EMA → continuation (not reversal)
- **Type 2:** Two consecutive days where Asia forms at same price → UK/US fakeouts → final stop hunt indicates direction
- **Type 3:** Full-day consolidation (Asia to Asia) → stop hunt → continuation

**Current bot:** No reset detection. The weekly cycle state machine treats failed formations as errors, not as reset patterns. No check for "W that fails to break 50 EMA" as a continuation signal.

### A7. VWAP + RSI Scalping Strategy (Scalp Course, Lessons 02-10)
**What the course teaches:**
- RSI length = **2** on 15-min chart (not the standard 14)
- RSI bands: upper **90**, lower **10**
- VWAP as dynamic support/resistance
- **255 EMA** determines trend direction
- Entry when RSI(2) hits 10/90 combined with VWAP + 255 EMA confluence

**Current bot:** No RSI indicator. No VWAP. No 255 EMA. No scalping strategy at all.

### A8. Ribbon Strategy (Scalp Course, Lesson 01)
**What the course teaches:**
- EMAs from 2 to 100 forming a ribbon
- Color flip = trend change
- Yellow EMAs = retest zone
- Entry on retest of yellow EMA with hammer after color flip
- Top 3 EMAs fanning away = exhaustion

**Current bot:** Not implemented. The 5-EMA system (10/20/50/200/800) doesn't include the ribbon.

---

## PRIORITY B — Partially Implemented (Logic exists but incomplete)

### B1. Scratch Rule / Time-Based Exit (Lesson 13)
**What the course teaches:**
- If not in substantial profit within **2 hours**, scratch the trade
- If stopped out, no new trade for **2 hours** minimum

**Current bot:** Symbol cooldown exists (4 hours after close) but there is NO 2-hour scratch rule on open positions. Positions are held indefinitely as long as SL isn't hit.

### B2. Fibonacci Alignment in Confluence (Lesson 13, Board Meetings)
**What the course teaches:**
- Price at 38.2/50/61.8% fib retracement = strong confluence
- Golden Pocket (61.8-65%) = best entry zone
- 78.6% only for ranging markets

**Current bot:** `mm_board_meetings.py` calculates fib levels for board meeting entries. But `has_fib_alignment` on MMContext is **never populated** — the confluence scorer's `fib_alignment` factor (6 pts) always scores 0.

### B3. EMA Fan-Out as Reversal Warning (Lessons 12, 18)
**What the course teaches:**
- EMAs fanning out at Level 3 = trend acceleration = imminent reversal
- 10/20 EMAs flattening = consolidation coming
- Top 3 EMAs separating from others = exhaustion signal

**Current bot:** `_detect_ema_fan_out` and `_detect_ema_flatten` methods exist in `mm_engine.py` but are **DEAD CODE** (never called). The `EMAState.fan_out_score` is calculated in `mm_ema_framework.py` but not used in entry or exit decisions.

### B4. Linda Cascade for Entry Boost (Lesson: TBD System Linda Trade)
**What the course teaches:**
- Multi-TF cascade (15m→1H→4H→daily) indicates macro trend strength
- Active cascade = higher conviction entry

**Current bot:** Linda cascade works for **exit suppression** only (prevents premature SVC/vol-degradation exits). It does NOT boost confluence score or affect entry decisions.

### B5. Wick Direction Changes (Lessons 08, 18)
**What the course teaches:**
- During 3-level rise: wicks at BOTTOM of candles
- At Level 3 top: wicks turn to TOP (exhaustion signal)
- Two wicks together after aggressive move = MM stopping
- This is a key early warning for reversal

**Current bot:** Formation detector checks wick behavior for quality scoring, but the engine does NOT track wick-direction changes during an open position's level progression. No "wicks turning to top at L3" exit signal.

### B6. Weekend Hold Decision (Lesson 13)
**What the course teaches:**
- If only 2 levels complete by Friday: can hold through weekend IF SL can move without ruining R:R
- Don't hold leveraged positions over weekend (funding fees)

**Current bot:** ALL positions are closed at Friday UK session regardless. No conditional weekend hold logic.

### B7. Unrecovered Vector Candle Targets (Lessons 05, 13)
**What the course teaches:**
- MM candle recovery: imbalance must get refilled
- If price recovers >50% of MM candle, likely recovers rest
- Always look left for nearest unrecovered vector candle
- These ARE take-profit targets

**Current bot:** `mm_targets.py` has vector candle scanning and uses them as targets. However, the **50% recovery rule** (if >50% recovered, expect full recovery) is not implemented. And vector targets don't have the "always look left" priority — they're mixed with EMA targets.

### B8. Board Meeting Re-Entry (Lessons 03, 13)
**What the course teaches:**
- Beginners should take profit at board meetings, re-enter after
- Board meeting gives opportunity to add to position or re-enter

**Current bot:** Board meeting detection runs during position management but only for logging. No re-entry or position-adding logic exists during board meetings.

---

## PRIORITY C — Missing Indicators / Data Sources

### C1. All 7 External Data Feeds Are Stubbed
**Not working:**
- **Hyblock** (liquidation clusters, delta) — confluence factor `liquidation_cluster` (8 pts) always scores 0
- **Trading Lite** (order flow heat map) — no order flow data
- **News calendar** (Forex Factory) — `news_event` (6 pts) always scores 0
- **Options expiry** (Max Pain, put-call) — not in any decision logic
- **Dominance** (BTC.D/ETH.D/USDT.D, alt season) — not used
- **Correlation** (DXY/NASDAQ) — `correlation_confirmed` (4 pts) always scores 0
- **Sentiment** (Fear & Greed) — not used

**Impact:** 18 of 119 confluence points (15%) are permanently zeroed out. The bot can never score above ~85% even with perfect alignment of implemented factors.

### C2. RSI Indicator (Multiple Courses)
**What the course teaches:**
- RSI(14) on 1H for trend bias (uptrend: 80-40 range, downtrend: 60-20)
- RSI(2) on 15m for scalp entries
- RSI divergence at M/W = additional confluence
- RSI crossing 50 = trend confirmation

**Current bot:** No RSI calculation anywhere in the MM engine.

### C3. ADR (Average Daily Range) — Lesson 14
**What the course teaches:**
- 14-day ADR
- 50% ADR line as cheap/expensive boundary
- Confluence with EMAs at 50% line

**Current bot:** Not implemented.

### C4. BBWP (Bollinger Band Width Percentile) — Trading Strategies Lesson 04
**What the course teaches:**
- BBWP at 95 = local top OR bottom found
- BBWP at 5 = breakout coming soon
- Signals WHEN moves are coming (not direction)

**Current bot:** Not implemented.

---

## PRIORITY D — Rule Refinements (Logic exists but course teaches different values/behavior)

### D1. Brinks Trade Window for iHOD/iLOD Confirmation
**What the course teaches:**
- iHOD/iLOD confirmed by hammer/inverted hammer followed by 30-90 minutes of sideways holding
- If price hits a zone 3 times without breaking = money trapped

**Current bot:** HOD/LOD tracked by weekly cycle module but the "30-90 min sideways confirmation" isn't checked.

### D2. Asia Extension Pattern (Lesson 09)
**What the course teaches:**
- 2:00-2:30am NY: potential extension in 3 small pushes
- Asia spike down at end → London higher low (W) → bullish
- Asia spike up at end → London lower high (M) → bearish
- This PREDICTS the London session direction

**Current bot:** Asia range percentage is checked (>2% = skip) but the directional hint from Asia's closing spike is not used to bias London entries.

### D3. Three London Pattern Types (Lesson 09)
**What the course teaches:**
- **Type 1:** Multi-session M/W at session changeover (highest probability)
- **Type 2:** Single-session M/W respecting Asia level
- **Type 3:** Squeeze M/W between iHOD and iLOD

**Current bot:** Session changeover gets a confluence boost, but the three distinct pattern types aren't classified or scored differently.

### D4. Entry Only on Closed Candles (Lesson 13)
**What the course teaches:**
- Only enter on CLOSED candles, preferably with multiple confirmations

**Current bot:** The engine trims in-progress bars before analysis (good), but the entry decision runs on the current scan cycle — it doesn't explicitly wait for the current candle to close before entering.

### D5. Stagger Entries (Lessons 05, 16)
**What the course teaches:**
- Stagger entries: 2-3 limit orders across price zone
- For fib levels: split across 38.2/50/61.8

**Current bot:** Single market order entry. No staggered/laddered entries.

### D6. MM Candle as Counter-Trend Warning (Lesson 07)
**What the course teaches:**
- Big green candle = MM SELLING (not buying)
- After extended rise, big green candle = bearish (MM distributing)
- Reframe: treat large-body candles as the OPPOSITE of what they look like

**Current bot:** PVSRA classifies volume but doesn't apply the "big green at L3 = bearish" reframing logic.

### D7. Session-Specific Behavior Rules (Lessons 04, 05)
**What the course teaches:**
- Asia: consolidate, set iHOD/iLOD, DON'T trend trade
- UK: slow trend, 6-8 hours
- US: typically reverses UK
- Dead zone (5-8pm NY): most manipulative in crypto, avoid

**Current bot:** Sessions are classified and dead zone is skipped. But the bot doesn't apply session-specific entry biases (e.g., "UK = trend-follow only, US = reversal bias").

### D8. Friday Behavior (Lessons 03, 13, 15)
**What the course teaches:**
- Friday: false move UK open → trend → extension end of UK → US reversal → trap
- If caught Friday, exit if stop hunt in opposite direction
- Friday UK session = all positions should close

**Current bot:** Friday UK exit is implemented. But the Friday trap pattern detection (false move → trend → extension → reversal) is not.

### D9. Correlation-Based Pre-Positioning (Lesson 19)
**What the course teaches:**
- DXY up = BTC down (inverse)
- S&P/NASDAQ generally correlated with BTC
- If DXY moving but BTC sitting still → position before BTC catches up
- Set up 6-chart layout for divergence detection

**Current bot:** Correlation provider is stubbed. Even when active, the course teaches using divergence for TIMING (BTC hasn't moved yet, it will), not just confirmation.

---

## SUMMARY TABLE

| Category | Count | Impact |
|----------|-------|--------|
| A: Missing trade setups | 8 | The bot can't detect Brinks, NYC Reversal, Half Batman, Stop Hunt entries, 33 Trade, Market Resets, VWAP+RSI scalps, or Ribbon trades |
| B: Partially implemented | 8 | Scratch rule, Fibonacci scoring, EMA fan-out, Linda entry boost, wick direction tracking, weekend hold, vector recovery rule, board meeting re-entry |
| C: Missing indicators | 4 categories | RSI, ADR, BBWP, and 7 stubbed data feeds (15% of confluence permanently at 0) |
| D: Rule refinements | 9 | Asia directional hints, London pattern types, stagger entries, session biases, Friday trap patterns, etc. |

**Total: 29 gaps identified.**

---

## RECOMMENDED IMPLEMENTATION ORDER

1. **Brinks Trade** (A1) — Highest R:R, most mechanical, easiest to code (time window + M/W check)
2. **Scratch Rule** (B1) — Simple timer, prevents losses from hanging trades
3. **EMA Fan-Out activation** (B3) — Already coded, just needs to be wired in
4. **Fibonacci alignment wiring** (B2) — Already calculated, needs to flow to confluence scorer
5. **NYC Reversal** (A2) — Level 3 + US session + reversal pattern
6. **RSI indicator** (C2) — Adds divergence confluence + trend bias
7. **Wick direction tracking** (B5) — Level 3 exit warning
8. **Market Resets** (A6) — Continuation detection (prevents false reversal entries)
9. **Asia directional hint** (D2) — Biases London entries
10. **Half Batman** (A3) — Niche but distinct pattern
