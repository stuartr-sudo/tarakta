# MM Method (Market Makers Method) — Strategy Specification

> Synthesized from 55 lessons of the Trade by Design course (tradetravelchill.club).
> This document is the blueprint for implementing the MM Method into the tarakta trading bot.

---

## Table of Contents

1. [Core Philosophy](#1-core-philosophy)
2. [The Weekly Cycle Template](#2-the-weekly-cycle-template)
3. [Session Timing](#3-session-timing)
4. [The Three Elements](#4-the-three-elements)
5. [M and W Formations](#5-m-and-w-formations)
6. [Entry Rules](#6-entry-rules)
7. [Stop Loss Rules](#7-stop-loss-rules)
8. [Take Profit and Targets](#8-take-profit-and-targets)
9. [EMA Framework](#9-ema-framework)
10. [Volume and Order Flow](#10-volume-and-order-flow)
11. [Multi-Timeframe Charting Process](#11-multi-timeframe-charting-process)
12. [Risk Management](#12-risk-management)
13. [Confluence Scoring](#13-confluence-scoring)
14. [The Linda Trade (Multi-TF Scaling)](#14-the-linda-trade)
15. [Tarakta Implementation Mapping](#15-tarakta-implementation-mapping)

---

## 1. Core Philosophy

The Market Makers Method is built on the premise that price is not random — it follows a repeatable business model operated by institutional Market Makers (Citadel, Jump Crypto, Virtue Financial, Cumberland) who work in 6-8 hour shifts across three global sessions.

**Key axioms:**
- Market Makers MUST hit stop losses and liquidation levels to fill their positions — they cannot use limit orders (visible walls) or market orders (moves price too much)
- Market Makers have deadlines set by the IMF/World Bank: weekly, monthly, and seasonal targets
- The MM can do anything midweek as long as the target is met by Friday 5pm NY
- Slow price action = the REAL move. Fast price action = the FALSE move (trap)
- The same pattern repeats on every timeframe — smaller patterns build into larger ones

**Expected performance:**
- ~2 trades per week per asset
- Each trade held ~3 days
- Risk-to-reward: 4:1 to 10:1 typical, up to 20:1
- At 4:1 R:R, only need 3 wins out of 10 to be profitable

---

## 2. The Weekly Cycle Template

The MM operates on a predictable weekly business model:

```
Sun 5pm NY ──────────────────────────────────────────── Fri 5pm NY
    │                                                        │
    ▼                                                        ▼
┌─────────┐  ┌──────┐  ┌────────────────────┐  ┌──────┐  ┌─────────┐
│ Weekend  │→ │ FMWB │→ │  3-Day/3-Level     │→ │ Mid- │→ │ Weekend │
│  Trap    │  │      │  │  Swing Trade       │  │ week │  │  Trap   │
│(sideways)│  │(false│  │                    │  │ Rev. │  │ (Fri UK)│
│          │  │ move)│  │ Level1→Level2→L3   │  │      │  │         │
└─────────┘  └──────┘  └────────────────────┘  └──────┘  └─────────┘
```

### Sequence:
1. **Weekend Trap** (Fri 5pm → Sun 5pm NY): Sideways consolidation, spike stop hunt near end
2. **False Move Week Beginning (FMWB)**: Aggressive trap move at/near Sun 5pm NY (crypto) or UK/US open Mon (forex). Purpose: stop out weekend traders, build MM positions
3. **M or W Formation**: Forms after FMWB — entry on 2nd peak
4. **3-Level Swing** (~3 days): Level 1 → Board Meeting → Level 2 → Board Meeting → Level 3
5. **Midweek Reversal** (Wed or Thu for crypto, Wed for forex): Lines up with news event. M or W forms at Level 3, signals direction change
6. **3-Level Swing in opposite direction**
7. **Friday Weekend Trap** (UK session): UK opens with false move, runs trend 4-6 hours, repeats initial level by end of UK session — this IS the trap

---

## 3. Session Timing

All times in New York (immutable regardless of DST):

| Session   | Gap (Handover) | Open     | Close    |
|-----------|---------------|----------|----------|
| Dead Zone | 5:00pm-8:00pm | —        | —        |
| Asia      | 8:00pm-8:30pm | 8:30pm   | 3:00am   |
| UK        | 3:00am-3:30am | 3:30am   | 9:00am   |
| US        | 9:00am-9:30am | 9:30am   | 5:00pm   |

**Session behaviors:**
- **Dead Zone** (5pm-8pm): All MMs off duty, low volume, manipulable by bots
- **Asia** (8:30pm-3am): Creates the daily range (initial High/Low of Day). Typically sideways. BTC range should be ≤2%. If >2%, skip the day — session shift likely
- **UK** (3:30am-9am): Runs the TRUE trend. Slow, steady price action
- **US** (9:30am-5pm): Continues OR reverses UK trend. Ends in consolidation (trap)

**Week boundaries:**
- Week begins: Sunday 5pm NY
- Week ends: Friday 5pm NY
- Day begins/ends: 5pm NY each day

**M/W formations must form during Gap times (session changeovers) to be reliable.** Finding them elsewhere is lower probability.

---

## 4. The Three Elements

### Element 1: The Pattern

The MM business model creates a repeating pattern:

1. **Trap moves** (FMWB) → clear weekend leverage, build contracts
2. **M or W formation** → M = bearish (High → Lower High), W = bullish (Low → Higher Low)
3. **Three Levels** of rises (after W) or drops (after M)
4. **Board Meetings** between levels = consolidation where MM accumulates contracts
5. **Wick behavior shift** at Level 3: wicks move from one side to the other → reversal hint
6. **Stopping Volume Candle** at Level 3 confirms trend exhaustion
7. **New M or W** at the other extreme → entry for the next swing

### Element 2: The Timing

- M/W must form at session changeovers (Gap times)
- Multi-session M/W (peaks in different sessions) = strongest setup
- Midweek reversal: Wed/Thu (crypto), Wed (forex)
- Friday UK session: take profit day — exit during peak formation

### Element 3: The Levels

- **Levels** = bursts/fast movements in price, identified by abnormally high volume
- Always count to 3 levels. A 4th level (Extended Rise/Drop) almost always brings correction
- **Previous week's High/Low** and **previous day's High/Low** replace traditional S/R
- Only care about positions opened within the last 3-5 days

**Liquidation by leverage at each level:**
| Level | Liquidates |
|-------|-----------|
| Level 1 | 100x leverage traders |
| Level 2 | 50x leverage traders |
| Level 3 | 25x leverage traders |

---

## 5. M and W Formations

### Standard M/W

**Three MM appearances confirm the formation:**

1. **Stopping Volume Candle (SVC)**: Small body, large wick in trend direction, very high volume at Level 3. Mark the zone — price should not fully return
2. **Inside Right Side** (drop one timeframe): Look for 3 trap candles (3 green for M, 3 red for W)
3. **50 EMA break with volume**: Final confirmation

**Key rules:**
- 2nd peak must NOT reach the 1st peak (MM keeps traders trapped)
- Spikes must pull away quickly (that's how the trap works)
- Wait for candle CLOSE as hammer/engulfing before entering
- M/W is NOT confirmed until it successfully retests

### Multi-Session M/W (Most Powerful Setup)

- 1st peak in one session, 2nd peak in another (e.g., Asia peak → UK open peak)
- Perfect spot: 1st peak in Asia, 2nd peak at start of UK → "almost guarantees" 3-level follow-through
- Found at HOW/LOW = weekly reversal signal with tight SL and massive R:R

### Final Damage M/W

- 2nd peak makes a LOWER low (W) or HIGHER high (M) than 1st peak
- Must be a hammer (W) or inverted hammer (M) on 15min
- Better R:R but higher probability of being wrong

### Board Meeting M/W

- Forms inside consolidation between levels
- Does NOT need full standard criteria (no SVC, no 3 inside hits required)
- Just the shape is sufficient

### Three Hits Rule

- 3 tests of HOW or LOW without breaking → reversal imminent (must be at Level 3)
- 4 hits → continuation/breakout likely
- Hits must be in DIFFERENT sessions (up to 2 per session)
- After each hit, price must move away then return

---

## 6. Entry Rules

### Two Entry Types

**Aggressive Entry** (on 2nd peak of M/W):
- Enter when candle CLOSES as hammer/engulfing on 2nd peak
- SL above 1st peak wick (M) or below 1st peak wick (W)
- Higher R:R but requires more skill
- Typical R:R: 4:1 to 7.8:1

**Conservative Entry** (after Level 1 confirmation):
- Wait for 50 EMA break with volume (Level 1)
- Enter on retest of the 50 EMA or board meeting entry
- SL above 2nd peak of M/W or just clearing a stop hunt
- Typical R:R: 2.7:1 to 9.8:1

### Four Retest Conditions (need ≥2 for entry)

After Level 1, price must retrace to a combination of at least 2:
1. The 50 EMA
2. The vector that created the Level 1 move
3. A higher low (after W) or lower high (after M)
4. A heat map / liquidation level cluster

### Board Meeting Entries

**Retracement Board Meeting:**
- Fibonacci from peak to trough: stagger orders at 38.2%, 50%, 61.8%
- SL above the peak of prior level

**Sideways Board Meeting:**
- Look for M/W shape inside consolidation (reduced criteria)
- Stop hunt comes at END of board meeting, not beginning

### When NOT to Enter

- R:R below 1.4:1 → "don't get out of bed"
- Friday after 5pm NY market close
- If setup doesn't show clear M/W and clear 3 levels → skip
- After only Level 1 drop — do NOT counter-trend trade from 200 EMA back
- Asia range >2% for BTC → skip the day

---

## 7. Stop Loss Rules

| Scenario | Stop Loss Placement |
|----------|-------------------|
| W formation | Below the low of the candle preceding the 1st spike, OR below LOD |
| M formation | Above the high of the candle preceding the 1st spike, OR above HOD |
| After stop hunt | Just clear of the stop hunt (MM won't return) |
| Conservative entry | Above 2nd peak of M/W, or clearing 50 EMA + wicks |
| Board meeting | Below 1st peak of W (or above for M) + clear the 50 EMA |

**Stop Loss Management:**
- NEVER tighten SL to improve R:R — SL goes where it needs to go
- Only move SL to breakeven AFTER Level 2 starts (not before)
- After Level 1: SL options are (a) above initial TP level, or (b) at entry. Nothing in between
- Once Level 2 running: can place SL just under 50 EMA
- NEVER increase SL during a trade

### The Refund Zone

- Only applies when entering on 2nd peak M/W
- If price CLOSES below the wick of the 2nd peak W (or above for M): formation invalidated → cut immediately
- Tiny loss, wait for real M/W to form

---

## 8. Take Profit and Targets

### Level Targets

| Level | Primary Target | Secondary Target |
|-------|---------------|-----------------|
| Level 1 | 50 EMA (counter-trend TP), then 200 EMA | First unrecovered Vector candle |
| Level 2 | 800 EMA | Previous unrecovered Vector candle |
| Level 3 | Higher TF EMA (200 or 800 on higher TF) | Previous HOW/LOW + unrecovered Vector |

### Target Identification

- Use **Fixed Range Volume Profile (FRVP)** to identify imbalance zones (crevices = targets)
- **Unrecovered Vector candles** = price targets (imbalance persists until consolidation fills the zone)
- Front-run liquidation levels (e.g., cluster at 25,000 → TP at 24,950)

### Take Profit Rules

- Initial target for R:R calculation must be Level 1 only — anything beyond is bonus
- At Level 3: look for Stopping Volume Candle → take at least some profit
- Friday UK session: exit when peak formation appears during UK close
- If only 2 levels by Friday: can hold through weekend only if SL can be moved without ruining R:R
- Every SL movement point = also a partial profit point (close % of position)

### Partial Profit Framework

- At Level 1 complete: consider taking 25-33% off
- At Level 2 complete: take 50% if unwilling to risk giving back profit
- At Level 3 SVC: take remaining or majority

---

## 9. EMA Framework

Five EMAs used on ALL timeframes:

| EMA | Role | Color (TBD) |
|-----|------|-------------|
| 10  | Fast — first cross on entry | Turquoise |
| 20  | Fast — confirmation | Red |
| 50  | Mid — Level 1 break target, retest level, trend confirmation | Blue |
| 200 | Slow — Level 2 target, reversal on hammer rejection | Yellow |
| 800 | Slow — Level 2/3 target | Purple |

### EMA Cycle Through Levels

1. **Entry**: Price crosses 10 and 20 EMA (usually on 2nd M/W peak)
2. **Level 1**: Breaks 50 EMA with volume (MUST have volume)
3. **Board Meeting**: Retraces to 50 EMA (if only to 10 EMA and counter to macro trend → caution)
4. **Level 2**: Runs to 200 EMA. Rejection at 200 EMA with hammer = take profit zone
5. **Level 2 Board Meeting**: Retests 50 EMA
6. **Level 3**: Trend acceleration — EMAs fan out, price extends far from EMAs. This traps retail

### Trend Ending Signals

- EMAs flatten (stop pointing in trend direction)
- Stopping Volume Candle appears
- Gap between price and EMAs = magnetic attraction → reversion coming

---

## 10. Volume and Order Flow

### PVSRA Volume System

| Candle Color | Meaning | Threshold |
|-------------|---------|-----------|
| Red (bearish) / Green (bullish) | Major volume spike | 200% of avg of prior 10 candles |
| Magenta (bearish) / Blue (bullish) | Significant volume spike | 150% of avg of prior 10 candles |

- Only MM can spike volume this much in one candle
- Volume degradation (green → blue) at Level 3 = guaranteed pause, likely reversal

### Stopping Volume Candle (SVC)

- Small body + large wick in prior trend direction + very high volume + at Level 3
- Price must FAIL to return into the wick for confirmation
- At Level 2, SVC only stops that level (price returns before Level 3 completes)

### Hyblock (Liquidation Levels)

- Shows open leveraged positions and estimated liquidation prices
- Use as TARGETS, not automatic reversal entries
- Delta threshold: ~$20B historically triggers reversals
- Total open positions: want >1,000 for meaningful data; <600 = thin books, potential fakes
- Level 1 takes 100x, Level 2 takes 50x, Level 3 takes 25x

### TradingLite (Order Book Heat Map)

- Shows limit orders (pending entries)
- Adjust sensitivity so only highest-level orders stand out
- When high limit orders + high liquidation cluster at same price = SVC location
- After 1st M/W peak, watch if order clusters shift price → predicts 2nd peak location

### Open Interest

| Price Action | OI Behavior | Interpretation |
|-------------|-------------|---------------|
| Consolidating | Increasing | Breakout coming, stop hunt NOT needed |
| Consolidating | Flat/Decreasing | Fakeout likely, stop hunt IS likely |
| Retracing | Increasing | Traders getting trapped in M/W before continuation |

### Correlation Analysis

- **BTC vs DXY**: Usually negatively correlated. DXY trend change is early warning for BTC
- **BTC vs NASDAQ**: Positively correlated with delay. NASDAQ leads trend changes
- **BTC.D rising + USDT.D falling + ETH.D rising** = Alt Season
- **TOTAL2 rising while TOTAL falling** = alt season signal
- **TOTAL3 rising** = degen season (high-risk alt appetite)

---

## 11. Multi-Timeframe Charting Process

**Daily process (~15-20 minutes):**

1. **Monthly** (check end/start of month): Mark unrecovered Vectors above/below price as Areas of Interest
2. **Weekly** (check end/start of week): Refine areas, mark 50% lines of Vectors, build bull AND bear case
3. **Daily** (check every day): Refine areas further, mark levels, build both cases
4. **12H/8H/6H**: Quick scan only — is macro trend changing?
5. **4H**: Check areas, vectors, heat map orders, calculate R:R for potential trades
6. **1H**: Mark vectors, Weekend Trap Box, FMWB, Peak Formation, level count, SVCs. This is the primary trading timeframe for the weekly setup
7. **15min**: Drop to this ONLY when entry is imminent on 1H, then return to 1H

**Critical process order**: Chart levels FIRST, then check liquidation/order flow data. Never look at Hyblock/TradingLite before marking your own levels.

### Weekend Trap Box Marking (1H)

1. Mark from US Friday close candle
2. Weekend = everything from next candle through to Dead Gap Zone (5pm NY Sunday)
3. Mark candle closes (not wicks) above and below within the box
4. Identify who got trapped (shorts induced early → stopped out late, etc.)

### Weekly Swing Checklist (1H)

1. Weekend Trap Box → identify fake move
2. Peak Formation → start level count
3. Count complete? → Look for reversal (M/W)
4. Count in progress? → Identify Level 3 target
5. If M/W formed: Did it break 50 EMA with volume? → Level 1 confirmed
6. Plan ahead: reversal location, news event, liquidation levels, heat map orders
7. Moon cycles: check full/new moon positions (14 days apart, ±3 day buffer)
8. News events: check 1-2 days ahead (Forex Factory, red/orange impact, USD filter)

---

## 12. Risk Management

### Position Sizing

- **Risk per trade**: 1% of total account (all exchanges combined)
- **Formula**: Position Size = Risk Amount / Stop Loss Distance %
- Example: $10K account, 1% risk = $100. SL at 1.33% distance → Position = $100/0.0133 = $7,519

### Leverage

- Leverage = capital freeing tool, NOT profit multiplier
- Dollar profit is IDENTICAL regardless of leverage (only ROI% changes)
- Probably don't need past 10x even with 2-3 simultaneous trades
- Higher leverage = closer liquidation + higher fees
- Large accounts: keep only 10% on exchange, use leverage to cover position size

### Win Rate / R:R Relationship

| Min R:R | Wins needed out of 10 |
|---------|----------------------|
| 1:1     | 6                    |
| 2:1     | 4                    |
| 3:1     | 3                    |
| 4:1     | 2.5                  |
| 5:1     | 2                    |

### Rules

- Judge in batches of 10 trades, not individually
- Never tighten SL to improve R:R
- If trade doesn't meet minimum R:R → don't take it
- R:R is calculated to Level 1 target only — beyond is bonus
- Same R:R produces same dollar profit on any timeframe (lower TFs need less market movement)

---

## 13. Confluence Scoring

The best entries have maximum confluence. Score each factor:

| Factor | Weight | Description |
|--------|--------|-------------|
| M/W at session changeover | HIGH | Gap time formation (especially multi-session) |
| M/W at HOW/LOW or HOD/LOD | HIGH | Key level alignment |
| 50 EMA break with volume | HIGH | Level 1 confirmation |
| Unrecovered Vector zone | MEDIUM | Price target / area of interest |
| Liquidation level cluster (Hyblock) | MEDIUM | Where MM needs to go |
| Limit order cluster (TradingLite) | MEDIUM | Where orders will absorb/trigger |
| EMA alignment (10/20/50/200/800) | MEDIUM | Dynamic S/R confirmation |
| Fibonacci level (38.2/50/61.8%) | MEDIUM | Retracement target in board meetings |
| News event timing | MEDIUM | Catalyst for midweek reversal |
| Moon cycle alignment | LOW | Full moon = local bottom, New moon = local top (±3 days) |
| Open Interest behavior | LOW | Breakout vs fakeout signal |
| Correlation confirmation (DXY, NASDAQ) | LOW | Early warning from correlated assets |
| Stopping Volume Candle present | HIGH | Level 3 completion confirmation |

**Minimum for entry**: M/W formation at correct timing + at least 2 of the 4 retest conditions + acceptable R:R

---

## 14. The Linda Trade

A trade that starts on a lower timeframe but feeds up through the hierarchy:

```
15min 3-level rise = 1H Level 1
1H 3-level rise    = 4H Level 1  (~1 week)
4H 3-level rise    = Daily Level 1 (~1 month)
Daily 3-level rise = Weekly Level 1 (~3 months)
```

**Key rule**: Retracements on higher TFs only give 1-2 rises/drops (NOT 3). If you see only 2 and price gets stuck at a key level + lines up with news/FMWB → expect continuation in the original direction.

**For tarakta**: This is advanced (Part 3 material). Implement the weekly setup first, then add Linda detection as an enhancement.

---

## 15. Tarakta Implementation Mapping

### New Modules Required

| Module | Purpose | Source Lessons |
|--------|---------|---------------|
| `src/strategy/mm_weekly_cycle.py` | Weekly cycle state machine (FMWB → M/W → 3 Levels → Reversal → Trap) | 2, 4, 5, 7, 8, 12 |
| `src/strategy/mm_formations.py` | M/W detection (standard, multi-session, final damage, board meeting) | 6, 7, 10, 20, 21 |
| `src/strategy/mm_levels.py` | Level counting (1-2-3), stopping volume candle detection, extended level detection | 9, 12, 23, 47, 48 |
| `src/strategy/mm_sessions.py` | Session time tracking, gap detection, session changeover identification | 8, 3 |
| `src/strategy/mm_ema_framework.py` | 5-EMA system (10/20/50/200/800), EMA break detection, trend state | 9, 24 |
| `src/strategy/mm_board_meetings.py` | Board meeting detection, Fibonacci retracement entries, stop hunt identification | 22 |
| `src/strategy/mm_weekend_trap.py` | Weekend trap box detection, FMWB identification | 5, 19, 44 |
| `src/strategy/mm_confluence.py` | Confluence scoring across all factors | 28 |
| `src/strategy/mm_targets.py` | Target identification (Vectors, FRVP, HOW/LOW, EMAs, liquidation) | 47, 48, 49 |
| `src/strategy/mm_risk.py` | Position sizing, leverage calc, partial profit framework | 52, 53, 54 |

### Modified Existing Modules (all ✅ COMPLETE)

| Module | Changes | Status |
|--------|---------|--------|
| `src/strategy/scanner.py` | MM analysis pipeline, scoring, agent context enrichment | ✅ |
| `src/strategy/confluence.py` | Added `mm_method: 15` weight | ✅ |
| `src/config.py` | Added `mm_method_enabled`, `mm_method_weight`, `mm_min_confluence_score`, `mm_min_rr` | ✅ |
| `src/strategy/agent_analyst.py` | MM prompt section + `_format_mm_method_context()` | ✅ |
| `src/strategy/refiner_agent.py` | MM entry rules prompt + `_format_mm_method_context()` | ✅ |
| `src/strategy/position_agent.py` | MM position rules prompt + `_format_mm_method_context()` | ✅ |
| `src/engine/entry_refiner.py` | Passes `mm_method` through to Agent 2 context | ✅ |

### Data Requirements

| Data | Source | Usage |
|------|--------|-------|
| 1H candles (primary) | CCXT/Binance | Weekly setup trading TF |
| 15min candles | CCXT/Binance | Entry refinement |
| 4H/Daily/Weekly candles | CCXT/Binance | Multi-TF charting process |
| HOW/LOW/HOD/LOD | Calculated from candles | Key levels |
| Liquidation data | Hyblock API (if available) | Target identification, confluence |
| Order book depth | CCXT/Binance | Limit order cluster detection |
| Open Interest | CCXT/Binance futures | Breakout vs fakeout signal |
| Volume (PVSRA-style) | Calculated from candle volume | Vector identification, SVC detection |
| News events | Forex Factory scrape or API | Midweek reversal timing |
| Moon phases | Astronomical calculation | Low-weight confluence |

### Implementation Priority

**Phase 1 — Core Weekly Setup (MVP): ✅ COMPLETE**
1. ✅ Session timing module (`mm_sessions.py` — 319 lines)
2. ✅ HOW/LOW/HOD/LOD calculation (in `mm_weekly_cycle.py`)
3. ✅ EMA framework (`mm_ema_framework.py` — 527 lines)
4. ✅ Level counting with PVSRA volume (`mm_levels.py` — 710 lines)
5. ✅ M/W formation detection (`mm_formations.py` — 1,290 lines)
6. ✅ Weekly cycle state machine (`mm_weekly_cycle.py` — 1,165 lines)
7. ✅ Confluence scoring (`mm_confluence.py` — 730 lines)
8. ✅ Entry/exit rules integration with agents (all 3 agent prompts + formatters)

**Phase 2 — Enhanced Detection: ✅ COMPLETE**
9. ✅ Multi-session M/W detection (in `mm_formations.py`)
10. ✅ Board meeting detection and entries (`mm_board_meetings.py` — 524 lines)
11. ✅ Stopping Volume Candle detection (in `mm_levels.py`)
12. ✅ Weekend trap box marking (`mm_weekend_trap.py` — 443 lines)
13. ✅ FMWB identification (in `mm_weekend_trap.py`)
14. ✅ Fibonacci retracement integration (in `mm_board_meetings.py`)
15. ✅ Target identification (`mm_targets.py` — 507 lines)
16. ✅ Risk management (`mm_risk.py` — 427 lines)

**Phase 3 — Advanced Features (TODO):**
17. Order flow integration (Hyblock, order book depth)
18. Open Interest analysis
19. Correlation analysis (DXY, NASDAQ, dominances)
20. News event integration
21. Moon cycle confluence
22. The Linda Trade (multi-TF scaling)

**Total: 10 modules, 6,642 lines of MM Method implementation**

---

*Document generated from 55 TBD course transcripts (9,970 cues) on 2026-04-05.*
*Phase 1-2 implementation completed 2026-04-05.*
*For the tarakta trading bot — github.com/stuarta/tarakta*
