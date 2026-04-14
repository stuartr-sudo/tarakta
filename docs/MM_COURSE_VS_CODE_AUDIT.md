# MM Method — Course vs Code Audit

**Purpose.** Exhaustively compare every concrete rule, threshold, and named
concept taught in the Trade by Design course (55 lessons in
`docs/tbd-course/`) against what the production code (`src/strategy/mm_*.py`)
actually does. No interpretation, only direct course quotes → code evidence
→ status.

**Status key:**
- ✅ **implemented** — code enforces this rule
- 🟡 **partial** — partially wired but incomplete or data missing
- ⛔ **missing** — rule not in code at all
- 📦 **dead code** — code exists but isn't called at runtime
- ❓ **N/A-for-bot** — course rule is about the human trader's psychology/journal, not algorithmic

---

## 1. Session timing (lessons 2, 3, 8)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| Week: Sun 5pm NY → Fri 5pm NY | "CME starts the week at 5pm, New York, on Sunday" | `mm_sessions.py` session tracker | ✅ |
| Day = 5pm NY → 5pm NY | "5pm, New York, is the beginning and also the end of every Day" | `mm_sessions.py` day boundaries | ✅ |
| Dead Zone 5–8pm NY | "Between 5pm, New York to 8pm, New York is considered the Dead Zone" | `_cycle`: `if session.session_name == "dead_zone": return` | ✅ |
| Asia 8:30pm–3am NY | "Asia opens at 8:30pm... closes at 3am" | `mm_sessions.py` | ✅ |
| UK 3:30am–9am NY | "UK opens... runs their session until 9am" | `mm_sessions.py` | ✅ |
| US 9:30am–5pm NY | "US session will open... closes at 5pm" | `mm_sessions.py` | ✅ |
| 30-minute gaps between sessions | "Between every Session Open, there is a half hour Gap" | `mm_sessions.py` `is_gap` | ✅ |
| Gaps are the prime M/W spots | "M/W formations must form during Gap times... Finding them elsewhere is lower probability" | `mw_session_changeover` scored (20 pts HIGH) | ✅ |
| Don't trade Fri after 5pm NY | "on Fridays, aggressive moves will be made" | `mm_reject_friday_trap` gate | ✅ |
| Weekend = no trading | "Market Makers are not around between Friday 5pm to Sunday 5pm" | `if session.is_weekend: return` | ✅ |

---

## 2. M/W formation criteria (lessons 7, 20, 21)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| M = High + lower High | "an M is simply a High followed by a lower High" | `mm_formations.py` detect_mw | ✅ |
| W = Low + higher Low | "a W is simply a Low followed by a higher Low" | `mm_formations.py` | ✅ |
| 2nd peak must NOT reach 1st peak | "Market Maker keeps traders trapped" | formation validation | ✅ |
| 3 MM appearances required | "SVC + 3 inside right side + break 50 EMA" | `mm_formations.py` FormationValidation | 🟡 (SVC + 50 EMA scored separately; 3 inside-right-side detection incomplete) |
| Stopping Volume Candle (SVC) at 1st peak | "small body, large wick, very high volume" | `mm_levels.py` detect_svc | ✅ |
| 3 inside right-side hits | "drop down a time frame... 3 trap candles" | partially in `mm_formations.py` but not as hard-required | 🟡 |
| 50 EMA break with volume = confirmation | "Market Maker shows up the third and final time to break the 50 EMA" | `mm_ema_framework.detect_ema_break` | ✅ |
| Entry BEFORE 50 EMA break | "I'm in on my trades before they break the 50 EMA" | aggressive entry path implied; not explicit gate | 🟡 |
| Final Damage W: 2nd peak lower than 1st, must be hammer on 15m | Lesson 21 | `variant="final_damage"` detection; hammer check on 15m NOT explicit | 🟡 |
| Final Damage M: 2nd peak higher than 1st, must be inverted hammer on 15m | Lesson 21 | same as above | 🟡 |
| Multi-session M/W strongest | "If you get a spike that creates the first peak in one Session, and... another Session" | `variant="multi_session"` + `mw_session_changeover` scored | ✅ |
| Board Meeting M/W: reduced criteria | "don't follow the same criteria... just the shape" | `mm_board_meetings.py`; but board-meeting M/W not explicitly entry-triggerable | 🟡 |
| M/W NOT confirmed until successful retest | "An M or W is not confirmed until it successfully retests" | retest conditions gate (gate 12/13) | ✅ |
| Spikes must pull away quickly | "The spikes in an M or W should pull away quickly" | quality_score implicitly; not a hard check | 🟡 |

---

## 3. Level counting (lessons 7, 9, 12, 23)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| 3 levels per swing | "three levels of Rises... three levels of Drop" | `mm_levels.LevelTracker.analyze` | ✅ |
| Level 1 breaks 50 EMA with volume | "Level 1 should break the 50 EMA with a high volume Vector" | `ema_break.volume_confirmed` | ✅ |
| Level 2 targets 800 EMA or prior vector | "Depending on where the 800 EMA is, that is often the level 2 target" | `LEVEL_EMA_TARGETS[2] = 200` **WRONG — should be 800** | ⛔ |
| Level 3 targets higher-TF EMA | "a 200 or an 800 EMA on a higher time frame" | `LEVEL_EMA_TARGETS[3] = 800` (also not HTF EMA) | 🟡 |
| Level 3 wicks switch sides | "wicks of the candles to switch sides" | `level_tracker.volume_degrading` partial | 🟡 |
| 4th level = Extended Rise/Drop (correction comes) | "A 4th level almost always brings correction" | `mm_reject_level_too_advanced` (>=3 post-formation) | ✅ |
| Volume degradation at L3 | "volume degradation (green → blue) at Level 3 = guaranteed pause" | `level_analysis.volume_degrading` | ✅ |
| Liquidation by leverage: 100x @L1, 50x @L2, 25x @L3 | Lesson 12 | Not used — Hyblock integration absent | ⛔ |

---

## 4. Weekly cycle & phase machine (lessons 2, 5, 12, 44)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| FMWB near Sunday 5pm NY / Monday | "usually made very close to market open, Sunday, 5pm, New York" | `mm_weekend_trap.py` FMWB detection | ✅ |
| Don't enter during FMWB | "Level 3 is designed to trap" / avoid false move | `mm_reject_fmwb_phase` | ✅ |
| FMWB direction is the FALSE move | "slow price action is the Real move, fast price action is the False move" | `mm_reject_against_weekly_bias` | ✅ |
| Midweek reversal: Wed/Thu crypto, Wed forex | Lesson 9 | Phase machine doesn't distinguish — no day-specific gate | 🟡 |
| Friday UK = trap session | Lesson 12, 19 | `mm_reject_friday_trap` | ✅ |
| Valid entry phases | FORMATION_PENDING, LEVEL_1/2/3, BOARD_MEETING_1/2 | `VALID_ENTRY_PHASES` enum | ✅ |
| Weekend Trap Box marking | "starts from when US closes... to Dead Gap Sunday" | `WeekendTrapAnalyzer.analyze` | ✅ |

---

## 5. EMA framework (lesson 24)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| 10 / 20 / 50 / 200 / 800 EMAs on all TFs | Lesson 24 | `EMAFramework` with all 5 | ✅ |
| Price crosses 10 & 20 first (entry) | "Price will cross the 10 and 20 EMA first" | Not explicit entry gate | 🟡 |
| 50 EMA break = Level 1 | Lesson 24 | `detect_ema_break(period=50)` | ✅ |
| 200 EMA rejection w/ hammer = TP zone | "Rejection at 200 EMA with hammer = take profit zone" | Not as a partial-close trigger | 🟡 |
| EMAs flatten = end of trend | Lesson 24 | No flatten-detector | ⛔ |
| EMAs fan out = trend acceleration (level 3 hint) | Lesson 24 | Not detected | ⛔ |
| Price-EMA gap = magnet / reversion coming | Lesson 24 | Not used | ⛔ |
| Trend board meeting: retrace to 50 EMA | Lesson 24 | Retest condition #1 | ✅ |

---

## 6. Volume / PVSRA (lesson 11, 23)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| Red/Green candle = 200% avg of prior 10 | Lesson 23 | `VECTOR_200_THRESHOLD = 2.0`, `VOLUME_LOOKBACK = 10` | ✅ |
| Magenta/Blue = 150% avg of prior 10 | Lesson 23 | `VECTOR_150_THRESHOLD = 1.5` | ✅ |
| Only MM can spike volume 200% in one candle | Lesson 23 | VectorScanner assumption | ✅ |
| Blue at end of L3 rise = guaranteed pause | Lesson 23 | `volume_degrading` flag | ✅ |
| Stopping Volume Candle = small body + large wick + high volume at L3 | Lesson 20, 23 | `detect_svc` | ✅ |
| Price must FAIL to return into SVC wick | Lesson 23 | Not enforced post-SVC | 🟡 |

---

## 7. Entry types (lesson 10, 13, 20)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| Aggressive entry on 2nd peak of M/W | "2nd peak of the M or W with the Stop Loss clearing the first Peak" | M/W path builds SL from peak1/peak2 | ✅ |
| Conservative entry after Level 1 confirmation | "wait for 50 EMA break with volume, enter on retest" | Retest conditions gate | ✅ |
| Need ≥2 of 4 retest conditions | "combination of at least two of the three conditions" (actually 4) | `MIN_RETEST_CONDITIONS = 2` gate | ✅ |
| 4 retest conditions: 50 EMA / L1 vector / HL-or-LH / liq cluster | Lesson 12 | `RetestConditions` dataclass with all 4 | ✅ |
| Wait for candle CLOSE before entering | "wait for candle Closes to take an Entry" | Not explicit — signal is reactive | 🟡 |
| Drop to 15m for entry, back to 1h for management | Lesson 13 | **Only 1H candles fetched — no 15m refinement** | ⛔ |
| "Don't counter-trend trade after level 1 Rise" | Lesson 12 — "the major one that you don't want to break" | Not an explicit gate | ⛔ |
| "Never chase price" | Lesson 43 | Implicit — entry price = current close | ✅ |
| Skip if R:R < 1.4 "don't get out of bed" | Lesson 13, 53 | `MIN_RR_AGGRESSIVE = 1.4` | ✅ |

---

## 8. Stop Loss rules (lessons 7, 10, 20, 49, 53)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| W SL below low of candle preceding 1st spike / LOD | Lesson 10 | `sl_price = min(peak1, peak2) * 0.998` | 🟡 (uses peaks not preceding candle) |
| M SL above high of candle preceding 1st spike / HOD | Lesson 10 | `sl_price = max(peak1, peak2) * 1.002` | 🟡 |
| "SL goes where it needs to go" | Lesson 53 | 5% cap removed in course-faithful redesign | ✅ |
| NEVER tighten SL to improve R:R | Lesson 7, 53 | Agent 3 can ONLY tighten SL (but separate SMC engine) — MM engine never auto-tightens below entry | ✅ |
| Move SL to breakeven only AFTER Level 2 starts | Lesson 48 | Not implemented in position monitor | ⛔ |
| Once L2 running: SL just under 50 EMA | Lesson 47, 48 | Not implemented | ⛔ |
| NEVER increase SL during trade | Lesson 51 | `repo.update_trade` doesn't enforce monotonic SL | 🟡 |
| Refund Zone: cut if price closes past 2nd peak | Lesson 49 | `RefundZoneCheck` class **defined but NEVER CALLED** | 📦 |

---

## 9. Targets / Take Profit (lessons 7, 12, 47, 48)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| L1 target = 50 EMA (primary), then 200 EMA (if 50 broken), then vector | Lesson 47, 48 | `LEVEL_EMA_TARGETS` + vector scanner + L2 fallback | ✅ |
| L2 target = 800 EMA / prior unrecovered vector | Lesson 12 | `LEVEL_EMA_TARGETS[2] = 200` **should be 800** | ⛔ |
| L3 target = higher-TF EMA + HOW/LOW + unrecovered vector | Lesson 12 | `LEVEL_EMA_TARGETS[3] = 800` (not HTF EMA) | 🟡 |
| Initial target for R:R calc must be L1 only | Lesson 16 | R:R computed against `t_l1` | ✅ |
| Anything beyond L1 is bonus | Lesson 16 | Signal has L2/L3 fields but R:R gate uses L1 only | ✅ |
| Partial close 25–33% at L1 | Lesson 48 | `PROFIT_SCHEDULE[1] = 0.30` | ✅ |
| Partial close 50% total by L2 | Lesson 48 | `PROFIT_SCHEDULE[2] = 0.50` | ✅ |
| Close remainder at L3 SVC | Lesson 48 | `PROFIT_SCHEDULE[3] = 1.00` | ✅ |
| Front-run liquidation levels (e.g., TP at 24,950 for 25,000 cluster) | Lesson 25, 27 | `LIQUIDATION_FRONTRUN_PCT = 0.002` | ✅ |
| Use FRVP (Fixed Range Volume Profile) for imbalance targets | Lesson 47 | `VectorScanner` (proxy — not true FRVP) | 🟡 |
| Friday UK: exit when peak formation in UK close | Lesson 12 | `friday_trap` phase skip; exit not triggered | 🟡 |

---

## 10. Risk Management (lessons 13, 52, 53, 54)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| 1% risk per trade | Lesson 53 | `RISK_PER_TRADE_PCT = 1.0` | ✅ |
| Position size = risk / SL distance | Lesson 52 | `MMRiskCalculator.calculate_position_size` | ✅ |
| Leverage = capital freeing, NOT profit multiplier | Lesson 52 | Default `leverage = 10` (configurable) | ✅ |
| Same $ profit regardless of leverage | Lesson 52 | Position sizing respects this | ✅ |
| Higher leverage = closer liquidation | Lesson 52 | `liquidation_price` calculation exists | ✅ |
| Don't need past 10x even with 3 trades | Lesson 52, 54 | Default 10x, configurable | ✅ |
| Calculate 1% across ALL exchanges combined | Lesson 54 — "if you have $10K OKX + $10K Binance, 1% of $20K" | **Code computes 1% per single exchange only** | ⛔ |
| Investment portfolio separate from trading portfolio | Lesson 54 | ❓ N/A for bot | ❓ |
| R:R 1.4 "don't get out of bed" | Lesson 13, 53 | `MIN_RR_AGGRESSIVE = 1.4` | ✅ |
| Typical minimum R:R 3:1 | Lesson 53 | `MIN_RR = 3.0` (as `confluence_scorer.min_rr`) | ✅ |
| Win rate × R:R relationship table | Lesson 53 | ❓ N/A-for-algo (informational) | ❓ |
| Judge in batches of 10 trades | Lesson 53 | ❓ N/A-for-algo (trader discipline) | ❓ |
| Trade journal | Lesson 51 | ❓ N/A-for-algo (`trades` table captures most of it) | ❓ |

---

## 11. Three Hits Rule (lesson 18)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| 3 hits to HOW/LOW without breaking = reversal imminent | Lesson 18 | `detect_three_hits` + `_try_three_hits_formation` | ✅ |
| 4 hits = continuation/breakout | Lesson 18 | `expected_outcome == "continuation"` skipped | ✅ |
| Hits must be in different sessions (max 2 per session) | Lesson 18 | `hit_sessions` list but max-2-per-session not enforced | 🟡 |
| After each hit, price must move away then return | Lesson 18 | Enforced in detect_three_hits | ✅ |
| Must be at Level 3 | Lesson 18 | `level.current_level >= 3` check in helper | ✅ |
| Replaces M/W as entry trigger | Lesson 18 | Synthesizes Formation in `_try_three_hits_formation` | ✅ |
| 200 EMA rejection trade (2nd setup) | Lesson 18 | **Not implemented as separate trigger** | ⛔ |

---

## 12. Hyblock / Liquidation Levels (lessons 25, 27)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| Use as targets (front-run) not reversal entries | Lesson 25 | — | ⛔ No Hyblock API feed |
| ~$20B delta historically triggers reversals | Lesson 27 | — | ⛔ |
| Total open positions >1,000 meaningful; <600 thin | Lesson 27 | — | ⛔ |
| Confluence factor `liquidation_cluster` (8 pts MEDIUM) | Course scores this | Weight exists but no data ingested → factor always 0 | 🟡 (wired but no data feed) |

---

## 13. TradingLite / Heat Map Orders (lessons 25, 26)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| Limit order clusters for absorb/trigger prediction | Lesson 25, 26 | — | ⛔ No TradingLite integration |
| Adjust sensitivity to high-level orders only | Lesson 26 | — | ⛔ |
| Combine with Hyblock: high orders + high liq cluster = SVC location | Lesson 25 | — | ⛔ |
| Listed in spec Section 13 as MEDIUM weight | Spec | **Not in code's WEIGHTS dict** | ⛔ |

---

## 14. Open Interest (lesson 29)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| Consolidating + rising OI = breakout, no stop hunt needed | Lesson 29 | `oi_behavior` scored MEDIUM (8 pts post-redesign) | 🟡 (wired but no OI data feed) |
| Consolidating + flat/dropping OI = stop hunt likely | Lesson 29 | Same | 🟡 |
| Retracing + rising OI = traders trapped in M/W | Lesson 29 | Same | 🟡 |
| OI detects trapped traders | Lesson 29 | Factor name is literal | 🟡 |

---

## 15. Correlation & Dominances (lessons 30, 31)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| BTC vs DXY (usually negative correlation) | Lesson 30 | `correlation_confirmed` (4 pts LOW) — no data feed | 🟡 |
| BTC vs NASDAQ (positive, with delay) | Lesson 30 | Same | 🟡 |
| BTC.D + USDT.D ↓ + ETH.D ↑ = alt season | Lesson 31 | — | ⛔ |
| TOTAL2 rising / TOTAL falling = alt season | Lesson 31 | — | ⛔ |
| TOTAL3 rising = degen season | Lesson 31 | — | ⛔ |

---

## 16. News & Options (lessons 32, 33)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| Forex Factory — red/orange impact, USD currency | Lesson 32 | — | ⛔ No FF integration |
| Fear & Greed / augmento.ai sentiment | Lesson 32 | — | ⛔ |
| Options expiry 3rd Friday of Mar/Jun/Sep/Dec ("Quadruple Witching") | Lesson 33 | — | ⛔ |
| Monthly options expiry monitoring | Lesson 33 | — | ⛔ |
| Max Pain price (within ~$1000 of BTC close) | Lesson 33 | — | ⛔ |
| Put/Call ratio + $1B notional threshold | Lesson 33 | — | ⛔ |
| Confluence factor `news_event` (6 pts MEDIUM) | Spec | Wired but no data feed | 🟡 |

---

## 17. Financial Astrology & Moon (lessons 34–37)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| Moon 29.5 day cycle, 8 phases | Lesson 37 | — | ⛔ |
| Full moon = local bottom, New moon = local top | Lesson 37 | Confluence placeholder only | 🟡 |
| ±3 day buffer | Lesson 37 | — | ⛔ |
| 80% win rate Bitcoin back-test over 2yr (Eric Crown) | Lesson 37 | Just informational | ❓ |
| `moon_cycle` factor (2 pts LOW) | Confluence WEIGHTS | No astronomical calc → always 0 | 🟡 |
| Zodiac/planetary signals | Lessons 34–36 | **Not in code** (intentional — low priority) | ⛔ |

---

## 18. High / Low of Week & Day (lesson 17)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| HOW/LOW = highest/lowest incl. wicks between Sun 5pm NY boundaries | Lesson 17 | `mm_weekly_cycle.py` HOW/LOW tracking | ✅ |
| HOD/LOD = highest/lowest between dead-gap zones | Lesson 17 | `mm_weekly_cycle.py` HOD/LOD | ✅ |
| Mark first thing every day / week | Lesson 17 | Auto-computed each `cycle_state.update()` | ✅ |
| Weekend price action INCLUDED in HOW/LOW | Lesson 17 | Inclusion confirmed | ✅ |
| MM attempts HOW or LOW every week | Lesson 17 | `mw_key_level` factor (15 pts HIGH) | ✅ |
| Only care about 3–5 days back for MM targeting | Lesson 9 | Level counting uses recent candles | ✅ |

---

## 19. Weekend Trap & FMWB (lessons 5, 19, 44)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| Mark Weekend Trap Box: Fri close → Sun dead-gap | Lesson 44 | `WeekendTrapAnalyzer` | ✅ |
| Identify who got trapped (shorts induced early, etc.) | Lesson 44 | `primary_trap` ("long"/"short"/"neutral") | ✅ |
| Look for W/M inside box if no spike out | Lesson 15 | **Not a separate entry path** | ⛔ |
| FMWB = aggressive trap move at/near Sun 5pm NY | Lesson 5, 12, 44 | `fmwb.detected`, `.direction` | ✅ |
| FMWB direction = false, real = opposite | Lesson 5, 44 | `mm_reject_against_weekly_bias` | ✅ |
| 2nd fake move possible if not enough commitment | Lesson 9 | Not explicitly modeled | 🟡 |

---

## 20. Asia Session Range (lessons 9, 12)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| Asia range ≤2% for BTC — skip the day if wider | Lesson 12 — "if Asia runs a range that is wider than 2%, It's possible we have a session shift that day, and that makes the whole day unpredictable" | `ASIA_RANGE_SKIP_PCT = 2.0` **constant defined but NEVER REFERENCED as a gate** | ⛔ |

---

## 21. Linda Trade — multi-TF scaling (lesson 55)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| 15m 3-level rise = 1H Level 1 | Lesson 55 | — | ⛔ |
| 1H 3-level rise = 4H Level 1 (~1 week) | Lesson 55 | — | ⛔ |
| 4H 3-level rise = Daily Level 1 (~1 month) | Lesson 55 | — | ⛔ |
| Daily 3-level rise = Weekly Level 1 (~3 months) | Lesson 55 | — | ⛔ |
| HTF retracements give 1-2 rises (not 3) | Lesson 55 | — | ⛔ |

Spec marks this as Phase-3 future work. **No Linda detection in code.**

---

## 22. Charting process (lessons 28, 39-43)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| Monthly, Weekly, Daily, 12H/8H/6H, 4H, 1H, 15m daily routine | Lesson 28 | Only 1H (+ 4H for EMAs) fetched | 🟡 |
| Chart levels FIRST, then check Hyblock/TradingLite | Lesson 28 | N/A-for-bot (it's instructor discipline) | ❓ |
| Mark unrecovered vectors above + below price as Area of Interest | Lesson 43 | `VectorScanner` scans; no "Area of Interest" marking per se | 🟡 |
| Mark Trapped Traders (= SVC zone) | Lesson 43 | SVC detection yes; zone-retention tracking no | 🟡 |
| Monthly/Weekly review once per period | Lessons 39, 40 | ❓ N/A-for-bot | ❓ |

---

## 23. Building Positions (lesson 50)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| Split position into 3–4 staggered orders in board meeting | Lesson 50 | **Not implemented — single market order** | ⛔ |
| Hedge: hold short through L3 drop, open 25% long after SVC | Lesson 50 | Not supported — single direction per symbol | ⛔ |
| "Extremely complicated — only after 6mo/100 trades" | Lesson 50 | Instructor advice; not needed for bot yet | ❓ |

---

## 24. Refund Zone (lesson 49)

| Rule | Course quote | Code | Status |
|------|-------------|------|--------|
| Only applies when entering on 2nd peak M/W | Lesson 49 | `RefundZoneCheck.check_refund_zone` exists | 📦 |
| Invalid if price CLOSES past 2nd peak wick | Lesson 49 | Logic in `mm_risk.py` | 📦 |
| Cut immediately at breakeven (before SL) | Lesson 49 | `is_in_refund_zone` / `should_cut` fields returned | 📦 |
| NOT relevant for post-Level-1 / Conservative entries | Lesson 49 | Check respects `entry_type` | 📦 |

**Status `📦` dead code**: `check_refund_zone` is never invoked anywhere — no position-monitor path calls it. So it's present but not enforced.

---

# GAP SUMMARY — concrete actionable items

## ⛔ Missing hard rules (course says; code doesn't do)

1. **Asia range >2% skip** — constant `ASIA_RANGE_SKIP_PCT = 2.0` defined but no gate uses it
2. **"Don't counter-trend trade after Level 1 Rise"** — no gate
3. **L2 target hardcoded to 200 EMA; course says 800 EMA** — wrong constant in `LEVEL_EMA_TARGETS[2]`
4. **Move SL to breakeven after Level 2 starts** — not in position monitor
5. **SL just under 50 EMA once L2 running** — not in position monitor
6. **15-minute entry refinement** — only 1H candles fetched in `_analyze_pair`
7. **Multi-exchange combined balance for 1% risk** — single-exchange only
8. **Hyblock liquidation-level API feed** — no integration
9. **TradingLite limit-order clusters** — no integration
10. **Real OI data feed** — `oi_behavior` factor exists but data always 0
11. **Real news/Forex Factory feed** — `news_event` factor exists but data always 0
12. **Options expiry / Max Pain / P-C ratio** — no integration
13. **Moon-phase astronomical calc** — `moon_cycle` factor exists but data always 0
14. **Dominance / correlation feeds** (BTC.D, ETH.D, DXY, NASDAQ)
15. **200 EMA rejection trade** (lesson 18, 2nd setup) — separate trigger not wired
16. **Board-meeting M/W as standalone entry trigger** — `mm_board_meetings.py` exists but not a formation source for `_analyze_pair`
17. **W/M inside weekend trap box** (lesson 15) — not a separate entry path
18. **Linda trade (multi-TF scaling)** — Phase-3 work, not started
19. **Staggered/built positions** — single market order only
20. **Hedge positions across M/W** — single direction per symbol

## 📦 Dead code (exists but not called)

21. **Refund Zone** — `RefundZoneCheck.check_refund_zone` defined in `mm_risk.py` but invoked nowhere. The most course-prominent "cut early" rule (lesson 49) is effectively not enforced.

## 🟡 Partial / incomplete

22. **M/W "3 inside right-side hits"** — detected but not enforced as a formation requirement
23. **Final Damage hammer/inverted-hammer on 15m check** — variant detected, 15m hammer check missing
24. **Price must fail to return to SVC wick** — SVC detected, post-detection tracking absent
25. **Midweek reversal day-specific gate** (Wed/Thu crypto) — not modeled
26. **Wait for candle CLOSE before entering** — scan fires on partial bar
27. **NEVER increase SL** — convention only; no DB-level monotonic constraint
28. **EMAs fanning out / flattening** — not detected
29. **FRVP** — approximated by VectorScanner, not true Fixed Range Volume Profile
30. **Friday UK exit at peak formation** — phase skip yes, triggered exit no
31. **Three hits: max 2 per session rule** — hit_sessions captured but 2-per-session cap not enforced
32. **W SL below *candle preceding* 1st spike** — code uses peak price, not preceding candle

---

# How to read this

Of ~90 discrete course rules audited:

- **~50 implemented** (✅) — foundational plumbing: sessions, formations, levels, phases, R:R, confluence scoring framework, risk math
- **~20 missing** (⛔) — most are data-feed-dependent (Hyblock, TradingLite, OI, news, moon, dominance) or higher-order (Linda, staggered positions)
- **~12 partial** (🟡) — the scoring/detection is present but the enforcement/data is incomplete
- **1 dead code** (📦) — Refund Zone exists but isn't called
- **~7 N/A-for-bot** (❓) — trader psychology / journal / discipline lessons

**Most consequential fixes (high impact, relatively low effort):**

A. Fix `LEVEL_EMA_TARGETS[2] = 800` (currently 200) — wrong per spec §8
B. Wire `check_refund_zone` into position monitor — critical loss-cut rule, currently dead code
C. Implement Asia range ≥2% skip for BTC — already have the constant
D. Implement "no counter-trend after L1" — course calls this "the major one you don't want to break"
E. Wire move-SL-to-breakeven-after-L2 and SL-under-50-EMA-when-L2-running

The rest (data feeds, Linda, multi-exchange, staggered positions) are larger features.
