# MM Engine Course-Faithful Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all 29 course-derived trading concepts missing from the MM engine, making the bot a complete implementation of the TTC course material.

**Architecture:** 8 phases, each producing working testable software. New modules follow existing dataclass + analyzer pattern. Integration via the formation cascade in `_analyze_pair()` and position checks in `_manage_position()`. Confluence scoring extended by adding factors to the WEIGHTS dict (MAX_POSSIBLE auto-calculates).

**Tech Stack:** Python 3.11, pandas, pytest (asyncio_mode=auto), existing MM module ecosystem

**Reference:** `docs/MM_COURSE_GAP_ANALYSIS.md` for full course-vs-bot cross-reference, `docs/MM_ENGINE_INTEGRATION_GUIDE.md` for DB column rules.

---

## Phase Dependency Graph

```
Phase 1 (Quick Wins) ─────┬──── Phase 5 (RSI + ADR) [parallel OK]
                           │         │
                           │         ├── Phase 6 (Session Intelligence)
                           │         │         │
                           │         │         ├── Phase 2 (New Trade Setups)
                           │         │         │
                           │         │         ├── Phase 3 (Pattern Detection)
                           │         │                   │
                           │         │                   ├── Phase 4 (Position Mgmt)
                           │         │                            │
                           │         │                            ├── Phase 7 (Advanced Rules)
                           │         │
                           │         └── Phase 8 (Scalping) [independent]
```

**Critical path:** Phase 1 → Phase 6 → Phase 2 → Phase 3 → Phase 4 → Phase 7

---

## PHASE 1: Quick Wins — Wire Up Dead Code + Simple Rules

**6 tasks, ~150-250 lines changed, Complexity: S**

### Task 1.1: EMA Fan-Out Activation (B3)

**Files:**
- Modify: `src/strategy/mm_engine.py:2140-2163` (position management, before SVC L3 check)
- Modify: `src/strategy/mm_engine.py:1058` (analyze_pair, after EMA calc)
- Test: `tests/test_mm_engine.py`

**Context:** `_detect_ema_fan_out()` at line 2363 and `_detect_ema_flatten()` at line 2344 are fully implemented but never called. The course teaches EMA fan-out at Level 3 = imminent reversal (Lessons 12, 18).

- [ ] **Step 1:** Write test — position at L3 with fan-out should trigger exit warning log
- [ ] **Step 2:** Run test, verify FAIL
- [ ] **Step 3:** In `_manage_position()` after Linda cascade check (~line 2150), add: if `pos.current_level >= 3`, call `_detect_ema_fan_out(candles_1h)`. If True, log `"EMA_FAN_OUT_L3_WARNING"`. Let existing SVC/vol-degradation checks handle actual close.
- [ ] **Step 4:** Run test, verify PASS
- [ ] **Step 5:** Commit: `feat(mm): activate EMA fan-out detection at Level 3 (B3)`

### Task 1.2: Fibonacci Alignment Wiring (B2)

**Files:**
- Modify: `src/strategy/mm_engine.py:1640-1684` (MMContext construction in _analyze_pair)
- Modify: `src/strategy/mm_confluence.py:126` (has_fib_alignment on MMContext)
- Test: `tests/test_mm_confluence.py`

**Context:** `has_fib_alignment` on MMContext (line 126) is always False. `BoardMeetingDetector.detect()` returns `BoardMeetingDetection.fib` with `FibonacciRetracement.levels` (list of `FibLevel` with `.price`). The `_score_fib_alignment()` at line 647 returns 6 pts when True. Currently scores 0 always.

- [ ] **Step 1:** Write test — MMContext with `has_fib_alignment=True` should produce 6.0 points from `_score_fib_alignment`
- [ ] **Step 2:** Run test, verify FAIL (currently always 0)
- [ ] **Step 3:** In `_analyze_pair()` before building `mm_ctx` (~line 1640): run `self.board_meeting_detector.detect(candles_1h, level_direction=direction)`, check if `entry_price` is within 0.3% of any `fib.levels[i].price`. Set `has_fib_alignment=True` on MMContext if match found.
- [ ] **Step 4:** Run test, verify PASS
- [ ] **Step 5:** Run `pytest tests/test_mm_confluence.py -v` — verify MAX_POSSIBLE assertion still passes (fib_alignment was already in WEIGHTS)
- [ ] **Step 6:** Commit: `feat(mm): wire fibonacci alignment into confluence scoring (B2)`

### Task 1.3: Scratch Rule — 2-Hour Exit (B1)

**Files:**
- Modify: `src/strategy/mm_engine.py:2047-2050` (position management, before stop loss check)
- Test: `tests/test_mm_engine.py`

**Context:** Course Lesson 13: "If not in substantial profit within 2 hours, scratch the trade." `MMPosition.entry_time` exists at line 186. `pos.current_level` tracks level progression — level 0 means no target hit yet.

- [ ] **Step 1:** Write test — position open >2h with current_level=0 should trigger close with reason "scratch_2h"
- [ ] **Step 2:** Run test, verify FAIL
- [ ] **Step 3:** In `_manage_position()` after refund zone check (~line 2047), before stop loss check: `if pos.current_level == 0 and (now - pos.entry_time).total_seconds() >= 7200: await self._close_position(symbol, current_price, "scratch_2h")`
- [ ] **Step 4:** Run test, verify PASS
- [ ] **Step 5:** Commit: `feat(mm): add 2-hour scratch rule for stalled trades (B1)`

### Task 1.4: Linda Cascade Entry Boost (B4)

**Files:**
- Modify: `src/strategy/mm_engine.py:1528-1555` (R:R check in _analyze_pair)
- Test: `tests/test_mm_engine.py`

**Context:** Linda cascade flags (`linda_cascade_15m_to_1h`, `linda_cascade_1h_to_4h`) are computed at lines 1229-1237. Currently only suppress exits. Course teaches active cascade = higher conviction. The R:R gate at line 1553 uses `self.min_rr` (default MIN_RR_AGGRESSIVE=1.4).

- [ ] **Step 1:** Write test — signal with Linda cascade active should use lower R:R floor
- [ ] **Step 2:** Run test, verify FAIL
- [ ] **Step 3:** Before R:R check, compute `effective_min_rr`: if `linda_cascade_1h_to_4h or linda_cascade_4h_to_1d`, use `MIN_RR_COURSE_FLOOR` (1.4). Otherwise use `self.min_rr`. Use `effective_min_rr` in the R:R gate.
- [ ] **Step 4:** Run test, verify PASS
- [ ] **Step 5:** Commit: `feat(mm): Linda cascade lowers min R:R threshold for entry (B4)`

### Task 1.5: Closed-Candle Entry Verification (D4)

**Files:**
- Modify: `src/strategy/mm_engine.py:988-1031` (bar trimming block)
- Test: `tests/test_mm_engine.py`

**Context:** Engine already trims in-progress bars. This is a verification + test task.

- [ ] **Step 1:** Write test — verify that a DataFrame with an in-progress final bar is trimmed before formation detection
- [ ] **Step 2:** Run test, verify PASS (already implemented)
- [ ] **Step 3:** Add comment at bar-trimming block: `# D4 course-faithful: only analyze closed candles`
- [ ] **Step 4:** Commit: `docs(mm): verify closed-candle entry compliance (D4)`

### Task 1.6: Weekend Hold Decision (B6)

**Files:**
- Modify: `src/strategy/mm_engine.py:2208-2212` (Friday UK exit block)
- Test: `tests/test_mm_engine.py`

**Context:** Currently line 2208: unconditional close on Friday UK session. Course Lesson 13: can hold through weekend if only L1-L2 complete AND SL at breakeven. `pos.sl_moved_to_breakeven` exists on MMPosition (line 208).

- [ ] **Step 1:** Write test — position at L2 with SL at breakeven should NOT close on Friday UK
- [ ] **Step 2:** Write test — position at L0 (no progress) should still close on Friday UK
- [ ] **Step 3:** Run tests, verify both FAIL
- [ ] **Step 4:** Modify Friday exit block: `if pos.current_level <= 2 and pos.sl_moved_to_breakeven: log("weekend_hold_allowed"); return` else close as before
- [ ] **Step 5:** Run tests, verify PASS
- [ ] **Step 6:** Commit: `feat(mm): conditional weekend hold for breakeven positions (B6)`

---

## PHASE 2: New Trade Setups — Brinks + NYC Reversal + Stop Hunt Entry

**3 tasks, ~300-500 lines new code, Complexity: L**
**Dependencies: Phase 1 (scratch rule), Phase 6 (session intelligence)**

### Task 2.1: Brinks Trade Detection (A1)

**Files:**
- Create: `src/strategy/mm_brinks.py`
- Modify: `src/strategy/mm_engine.py` (formation cascade + init)
- Test: `tests/test_mm_brinks.py`

**Context:** Course Lesson 06. Highest R:R setup (6:1-18:1). ONLY at two 15-min candle windows: 3:30-3:45am or 9:30-9:45am NY. Second leg of M/W at HOD/LOD. Entry = hammer/inverted at exact window close. Time between peaks: 30-90 min. Scratch if not profitable in 2h.

- [ ] **Step 1:** Write tests for `BrinksDetector`:
  - `test_brinks_window_345am` — 15m candle at 3:45am NY with hammer at LOD → detected
  - `test_brinks_window_945am` — same at 9:45am
  - `test_brinks_outside_window` — 15m candle at 4:00am → NOT detected
  - `test_brinks_no_hammer` — correct time but no hammer/inverted hammer → NOT detected
  - `test_brinks_not_at_hod_lod` — correct time + hammer but not at HOD/LOD → NOT detected
  - `test_brinks_peak_separation` — peaks 30-90 min apart → detected; <30 min → rejected
- [ ] **Step 2:** Run tests, verify all FAIL
- [ ] **Step 3:** Create `src/strategy/mm_brinks.py`:

```python
@dataclass
class BrinksResult:
    detected: bool
    window: str  # "uk_open" or "us_open"
    entry_price: float
    stop_loss: float
    direction: str  # "long" or "short"
    formation_type: str  # "W" or "M"
    peak1_time: datetime
    peak2_time: datetime
    r_r_estimate: float

class BrinksDetector:
    BRINKS_WINDOWS_NY = [
        (time(3, 30), time(3, 45), "uk_open"),
        (time(9, 30), time(9, 45), "us_open"),
    ]
    MIN_PEAK_SEP_MINUTES = 30
    MAX_PEAK_SEP_MINUTES = 90

    def detect(self, candles_15m: pd.DataFrame, hod: float, lod: float, now_ny: datetime) -> BrinksResult | None
```

  Core logic:
  1. Check if current 15m candle closes within a Brinks window
  2. Check if candle is at HOD (→ M/short) or LOD (→ W/long)
  3. Check for hammer (W) or inverted hammer (M) pattern
  4. Look back 30-90 min for first peak (SVC or prior extreme)
  5. Calculate R:R estimate (target = 3-level move)

- [ ] **Step 4:** Run tests, verify PASS
- [ ] **Step 5:** Wire into `mm_engine.py`:
  - Import `BrinksDetector` in `__init__`, store as `self.brinks_detector`
  - In `_analyze_pair()` after the board meeting formation check (~line 1117), add attempt #5: `self._try_brinks_formation(candles_15m, now, cycle_state)`. This calls `brinks_detector.detect()` and synthesizes a `Formation` if detected.
- [ ] **Step 6:** Run full test suite: `pytest tests/ -x`
- [ ] **Step 7:** Commit: `feat(mm): add Brinks trade detection (A1)`

### Task 2.2: NYC Reversal Trade (A2)

**Files:**
- Modify: `src/strategy/mm_formations.py` (add detect_nyc_reversal)
- Modify: `src/strategy/mm_engine.py` (formation cascade)
- Test: `tests/test_mm_formations.py`

**Context:** Course Lesson 10. Within first 3 hours of US session (9:30am-12:30pm NY). Price at Level 3. HOD/LOD already formed. Reversal candlestick pattern. Target: 50 EMA or MM candle recovery.

- [ ] **Step 1:** Write tests for `detect_nyc_reversal(candles_1h, session_info, level, hod, lod)`
- [ ] **Step 2:** Run tests, verify FAIL
- [ ] **Step 3:** Implement `detect_nyc_reversal()` in `mm_formations.py`
- [ ] **Step 4:** Wire into formation cascade in `_analyze_pair()` — must be inserted BEFORE the L3 rejection gate (line 1248) since this setup IS the L3 reversal
- [ ] **Step 5:** Run tests, verify PASS
- [ ] **Step 6:** Commit: `feat(mm): add NYC Reversal trade detection (A2)`

### Task 2.3: Stop Hunt Entry at Level 3 (A4)

**Files:**
- Modify: `src/strategy/mm_formations.py` (add detect_stophunt_entry)
- Modify: `src/strategy/mm_engine.py` (formation cascade)
- Test: `tests/test_mm_formations.py`

**Context:** Course Lesson 15. At L3 in board meeting. Vector candle with big wick (stop hunt). Entry 1-2 candles AFTER. Verify wick "left alone."

- [ ] **Step 1:** Write tests
- [ ] **Step 2:** Run, verify FAIL
- [ ] **Step 3:** Implement — reuse `LevelTracker.detect_stopping_volume()` for the vector candle identification and `BoardMeetingDetector.detect()` for board meeting confirmation
- [ ] **Step 4:** Wire into cascade before L3 gate
- [ ] **Step 5:** Run, verify PASS
- [ ] **Step 6:** Commit: `feat(mm): add stop hunt entry at Level 3 (A4)`

---

## PHASE 3: Pattern Detection — Half Batman + 33 Trade + Market Resets

**3 tasks, ~200-400 lines, Complexity: M-L**
**Dependencies: Phase 1 (EMA fan-out for 33 Trade)**

### Task 3.1: Half Batman Pattern (A3)

**Files:**
- Modify: `src/strategy/mm_formations.py`
- Modify: `src/strategy/mm_engine.py` (cascade)
- Test: `tests/test_mm_formations.py`

**Context:** Course Lesson 15. Single peak after 3-level cycle. Tight sideways consolidation (no stop hunts, equal highs/lows). Entry on range break. Different from Trapping Volume (smaller range).

- [ ] **Step 1-6:** TDD cycle: test → fail → implement `detect_half_batman(candles_1h, level_analysis)` → pass → wire cascade → commit

### Task 3.2: 33 Trade (A5)

**Files:**
- Modify: `src/strategy/mm_engine.py`
- Test: `tests/test_mm_engine.py`

**Context:** Course Lesson 12. 3 rises over 3 days + 3 hits to high on Day 3 + EMA fan-out = "33 Trade." All three components already exist individually: `level_analysis.current_level >= 3`, `three_hits_at_how`, `_detect_ema_fan_out()`. Just need to combine them and relax the L3 gate for this specific pattern.

- [ ] **Step 1-5:** TDD: test combination check → implement in `_analyze_pair` → commit

### Task 3.3: Market Resets — 3 Types (A6)

**Files:**
- Modify: `src/strategy/mm_weekly_cycle.py`
- Modify: `src/strategy/mm_engine.py`
- Test: `tests/test_mm_weekly_cycle.py`

**Context:** Course Lesson 15. Type 1: W fails 50 EMA break → continuation. Type 2: two consecutive Asia at same level. Type 3: full-day consolidation → stop hunt → continuation. Currently, failed formations cause the cycle to stall.

- [ ] **Step 1-6:** TDD: test each reset type → implement `detect_market_reset()` in WeeklyCycleTracker → wire into phase transitions → commit

---

## PHASE 4: Position Management Enhancements

**4 tasks, ~150-300 lines, Complexity: M**
**Dependencies: Phases 1-3**

### Task 4.1: Wick Direction Tracking (B5)

**Files:**
- Modify: `src/strategy/mm_engine.py` (_manage_position)
- Test: `tests/test_mm_engine.py`

**Context:** Course Lessons 08, 18. At L3: wicks turning from bottom to top during rise = exit warning. Track last N candles' wick bias.

- [ ] **Step 1-5:** TDD cycle → commit: `feat(mm): track wick direction changes at L3 (B5)`

### Task 4.2: Vector 50% Recovery Rule (B7)

**Files:**
- Modify: `src/strategy/mm_targets.py`
- Test: `tests/test_mm_targets.py`

**Context:** Course Lesson 13. If >50% of vector candle body recovered → expect full recovery. Boost target confidence.

- [ ] **Step 1-5:** TDD → commit

### Task 4.3: Board Meeting Re-Entry (B8)

**Files:**
- Modify: `src/strategy/mm_engine.py:2112-2123`
- Test: `tests/test_mm_engine.py`

**Context:** Course Lesson 13. When board meeting detected during open position + partial taken → opportunity to add back position size.

- [ ] **Step 1-5:** TDD → commit

### Task 4.4: Stagger Entries (D5)

**Files:**
- Modify: `src/strategy/mm_engine.py` (_enter_trade / _process_entries)
- Test: `tests/test_mm_engine.py`

**Context:** Course Lessons 05, 16. Instead of single market order, place 2-3 limit orders across entry zone using fib levels.

- [ ] **Step 1-5:** TDD → commit

---

## PHASE 5: New Indicators — RSI + ADR

**2 tasks, ~200-300 lines new, Complexity: M**
**Dependencies: None (can run parallel with Phase 1)**

### Task 5.1: RSI Indicator (C2)

**Files:**
- Create: `src/strategy/mm_rsi.py`
- Modify: `src/strategy/mm_engine.py` (init + _analyze_pair)
- Modify: `src/strategy/mm_confluence.py` (new factor)
- Test: `tests/test_mm_rsi.py`

**Context:** RSI(14) on 1H. Uptrend: 40-80 range. Downtrend: 20-60. Divergence at M/W = confluence boost. Crossing 50 = confirmation. New `rsi_confirmation` factor (6 pts) in confluence.

- [ ] **Step 1:** Write tests for `RSIAnalyzer.calculate()`: value accuracy, trend_bias classification, divergence detection
- [ ] **Step 2:** Run, verify FAIL
- [ ] **Step 3:** Implement `mm_rsi.py` with `RSIState` dataclass + `RSIAnalyzer` class
- [ ] **Step 4:** Run, verify PASS
- [ ] **Step 5:** Add `"rsi_confirmation": 6.0` to WEIGHTS in `mm_confluence.py`, add `rsi_confirmed: bool | None = None` to MMContext, add `_score_rsi_confirmation()` method
- [ ] **Step 6:** Wire into `_analyze_pair()`: instantiate in `__init__`, call after EMA, pass to MMContext
- [ ] **Step 7:** Update `tests/test_mm_confluence.py` — assert new MAX_POSSIBLE (119+6=125), test rsi factor scoring
- [ ] **Step 8:** Run full suite, commit: `feat(mm): add RSI indicator + confluence factor (C2)`

### Task 5.2: ADR (Average Daily Range) (C3)

**Files:**
- Create: `src/strategy/mm_adr.py`
- Modify: `src/strategy/mm_targets.py`
- Modify: `src/strategy/mm_confluence.py` (optional factor)
- Test: `tests/test_mm_adr.py`

**Context:** 14-day ADR. 50% line = cheap/expensive boundary. Confluence with EMAs.

- [ ] **Step 1-6:** TDD → implement `ADRAnalyzer` → wire into targets + optional confluence factor (`"adr_confluence": 4.0`) → commit

---

## PHASE 6: Session Intelligence

**3 tasks, ~150-250 lines, Complexity: M**
**Dependencies: Phase 1**

### Task 6.1: Asia Directional Hint (D2)

**Files:**
- Modify: `src/strategy/mm_sessions.py`
- Modify: `src/strategy/mm_engine.py`
- Test: `tests/test_mm_sessions.py`

**Context:** Course Lesson 09. Asia closing spike (2:00-2:30am NY) predicts London direction. Spike down → expect higher low W (bullish). Spike up → expect lower high M (bearish).

- [ ] **Step 1-5:** TDD → add `detect_asia_closing_spike()` to MMSessionAnalyzer → wire as soft directional filter in `_analyze_pair` → commit

### Task 6.2: London Pattern Classification (D3)

**Files:**
- Modify: `src/strategy/mm_formations.py`
- Modify: `src/strategy/mm_engine.py`
- Test: `tests/test_mm_formations.py`

**Context:** Course Lesson 09. Type 1: multi-session M/W at changeover (highest prob). Type 2: single-session. Type 3: squeeze between iHOD/iLOD.

- [ ] **Step 1-5:** TDD → add `classify_london_pattern()` → store on MMSignal → commit

### Task 6.3: Session-Specific Entry Biases (D7)

**Files:**
- Modify: `src/strategy/mm_engine.py` (_analyze_pair)
- Test: `tests/test_mm_engine.py`

**Context:** Course Lessons 04, 05. UK = trend-following bias. US first 3h = reversal bias. Don't allow counter-trend reversal entries during UK.

- [ ] **Step 1-5:** TDD → add session bias checks after formation detection → commit

---

## PHASE 7: Advanced Rules + Refinements

**5 tasks, ~200-350 lines, Complexity: M-L**
**Dependencies: Phases 1-6**

### Task 7.1: iHOD/iLOD Confirmation (D1)
Add 30-90 min sideways confirmation after hammer. 3 touches = trapped money.

### Task 7.2: MM Candle Reframing (D6)
Big green at L3 rise = bearish (MM distributing). Exit signal in `_manage_position`.

### Task 7.3: Friday Trap Pattern (D8)
Detect: false move UK open → trend → extension end of UK → US reversal.

### Task 7.4: Correlation Pre-Positioning (D9)
Define interface for DXY/NASDAQ divergence timing. Gated on CorrelationProvider.

### Task 7.5: BBWP Indicator (C4)
New `mm_bbwp.py`. BBWP 95 = top/bottom; BBWP 5 = breakout imminent. Timing signal.

---

## PHASE 8: Scalping Strategy (Optional/Separate)

**3 tasks, Complexity: L**
**Dependencies: Phase 5 (RSI module)**

### Task 8.1: VWAP + RSI Scalp (A7)
Separate engine mode. RSI(2) on 15m + VWAP + 255 EMA. Fundamentally different strategy.

### Task 8.2: Ribbon Strategy (A8)
Multi-EMA ribbon (2-100). Separate mode.

### Task 8.3: External Data Feed Implementations (C1)
Per-provider as APIs become available. Replace stubs in `mm_data_feeds.py`.

---

## Verification

After each phase:
1. `pytest tests/ -x` — all tests pass
2. `ruff check src/ tests/` — no lint errors
3. `ruff format src/ tests/` — formatted
4. Review confluence MAX_POSSIBLE matches expected value
5. Check `docs/MM_ENGINE_INTEGRATION_GUIDE.md` if any DB columns added

After all phases:
1. `python3 -m src.main` — bot starts and runs scan cycles without error
2. Review MM engine logs for new detection events (Brinks, NYC Reversal, etc.)
3. Verify confluence scoring produces non-zero scores for new factors
4. Paper trade for 24-48 hours monitoring for regressions

## Summary

| Phase | Tasks | New Files | Complexity | Key Deliverable |
|-------|-------|-----------|-----------|-----------------|
| 1 | 6 | 0 | S | Wire dead code, scratch rule, weekend hold |
| 2 | 3 | 1 (mm_brinks.py) | L | Brinks + NYC Reversal + Stop Hunt entries |
| 3 | 3 | 0 | M-L | Half Batman, 33 Trade, Market Resets |
| 4 | 4 | 0 | M | Wick tracking, vector recovery, re-entry, staggers |
| 5 | 2 | 2 (mm_rsi.py, mm_adr.py) | M | RSI + ADR indicators |
| 6 | 3 | 0 | M | Asia hints, London patterns, session biases |
| 7 | 5 | 1 (mm_bbwp.py) | M-L | iHOD confirm, MM reframe, Friday trap, BBWP |
| 8 | 3 | 0-1 | L | Scalping (separate), data feeds |

**Total: 29 gaps, 8 phases, 4 new files, ~1500-2500 lines of new/changed code.**
