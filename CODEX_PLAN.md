# Codex Execution Plan — Agent Committee Rebuild

**Date:** 2026-06-13  
**Bot:** tarakta-mm (Fly.io, paper mode, Supabase project `uounrdaescblpgwkgbdq`)  
**Problem:** Deterministic engine alone loses −$7,613 over 62 trades. A single LLM veto agent was the only profitable configuration but was disabled because its rubric became stale. Goal is to replace it with a multi-agent committee, each grounded in the scraped TTC course materials.

**DO NOT SKIP:** Read `CLAUDE.md` fully before touching any code. Read `docs/MM_ENGINE_INTEGRATION_GUIDE.md` before any DB column change. Every rule change must cite the exact course lesson + timestamp in the commit message.

**Sequencing rule: Phase 0 engine fixes block committee activation.** Complete Phase 0 (Tasks 0.1–0.4) and verify with replay before enabling the committee in any mode. A committee that inherits a broken scratch rule or stale feed context will re-create the over-veto problem. Phase 1 skill-file authoring (Task 1.1 + 1.2) can proceed in parallel with Phase 0 since it touches no engine code.

**Course materials are at:**
- `docs/tbd-course/` — **55 lessons, full TBD System Pt1** (primary structured course — use this first)
- `docs/courses/mmm-masterclasses/` — 22 MMM Masterclass lessons (live Q&A format, rich with verbatim rules)
- `docs/courses/scalp-trading-strategies/` — 12 lessons
- `docs/courses/trading-strategies/` — 6 lessons
- `docs/courses/ttc-indicators/` — 7 lessons

Key `docs/tbd-course/` lessons by agent domain:
- **Structure:** 07 (pattern), 09 (levels), 10 (multi-session M/W), 18 (three-hits setup), 20 (mechanics of M/W), 21 (final damage M/W), 45 (marking levels/zones)
- **HTF & Trend:** 24 (trend EMAs), 30 (correlation/trend phases), 39–43 (charting each timeframe)
- **Cycle & Session:** 08 (timing element), 12 (weekly 3-day swing), 19 (weekend trap), 22 (board meeting entries), 44 (marking weekend trap), 46 (weekly setup recap)
- **Flow Data:** 23 (vectors/stopping volume), 25–27 (Hyblock/TradingLite/liquidation levels), 29 (open interest), 31 (dominance/totals)
- **Risk & Rules:** 13–16 (entries and SLs weeks 1–4), 17 (high/low of week significance), 47–49 (target management L1/L2/L3, refund zone), 52–54 (leverage, risk, advanced risk)

---

## Phase 0 — Fix the engine bleeders (do this before any agent work)

These are the two reasons the agent committee will inherit a negative-expectancy stream if skipped.

### Task 0.1 — Fix the 2h scratch rule

**File:** `src/strategy/mm_engine.py`  
**Course basis:** `docs/courses/mmm-masterclasses/lesson-13-the-trading-zone-the-trading-rules.md` [44:00] [46:30] [102:30]

**What to change:**

Replace the current MFE-based scratch logic. The current code scratches at 2h unless `mfe_r >= scratch_mfe_threshold_r (0.3)`. This is wrong in two ways:

1. The threshold should be "price moved far enough to place stop at breakeven" (i.e. MFE ≥ the distance from entry to breakeven level), not a fixed 0.3R. The course says [44:00]: *"moved into enough profit … that allows you to move your stop to Breakeven."*

2. The 2h window is stated for daily (1H) setups. At [102:30] Annii says: *"I'd hold on longer is if I found something on a four hour, or a daily."* The window must scale with the formation timeframe.

**Exact changes:**

- Add config: `mm_scratch_be_distance_r: float = 0.2` (the R-multiple at which BE becomes possible — typically the distance from entry to the nearest wick/structure). Replaces `scratch_mfe_threshold_r`.
- Add config: `mm_scratch_window_4h_bars: int = 2` (for 4H formations: scratch after 2 closed 4H bars without reaching BE distance).
- In the scratch check: if formation timeframe is `1h` or `15m`, keep the 2h wall-clock check but use `mfe_r >= mm_scratch_be_distance_r`. If formation timeframe is `4h` or `1d`, count closed 4H bars elapsed instead of wall-clock hours.
- Formation timeframe is available in `MMSignal.formation_timeframe` — add this field if it does not exist.

**Success criteria:**
- The canonical AVAX 2026-04-28 scratch (entered $9.2328, scratched $9.252 after 2h, MFE=0.0R) must still be scratched (MFE=0 never reached BE distance).
- A trade that goes +0.4R within 1h then pulls back to entry must NOT be scratched (MFE exceeded BE distance, stop should already be at BE).
- `pytest tests/` still passes entirely.
- Commit message must cite `lesson-13 [44:00]` and `lesson-13 [102:30]`.

---

### Task 0.2 — Shift to 4H/Daily as primary formation timeframe

**Files:** `src/strategy/mm_formations.py`, `src/strategy/mm_engine.py`, `src/strategy/mm_levels.py`  
**Course basis:** `docs/courses/mmm-masterclasses/lesson-05-the-daily-setup.md`, `lesson-07-m-and-w.md`, `lesson-11-the-count.md`, `lesson-13-the-trading-zone-the-trading-rules.md` [84:00]

**What to change:**

The engine currently runs formation detection primarily on 1H candles. This produces noisy entries with SLs placed below 1H peak wicks. The course daily setup (Lesson 5) expects the M/W to form on the **daily or 4H** view, with the session (1H/15m) used only for precise retest entry timing.

1. Add a formation scan on 4H candles in `mm_formations.py`. The existing `detect_mw_formation(candles_1h, ...)` API must gain a `timeframe` parameter so it can run on 4H input. The formation detector logic itself (peak detection, confirmation bars) does not need a rewrite — just ensure it works correctly on 4H-spaced candles.
2. In `mm_engine.scan_symbol`, run formation detection on 4H first. If a confirmed 4H M/W is found, use 1H/15m only for retest-entry timing (existing `retest_passed` gate). If no 4H formation, fall through to the existing 1H path (keep it as a secondary path, not deleted).
3. SL placement for 4H formations: per Lesson 13 [84:00] *"below the Low of the Day, or above the High of the Day after your three levels."* For 4H formations use the LOD/HOD from `mm_levels` as the SL anchor, not peak2_wick.
4. Add `formation_timeframe: str` field to `MMSignal` (and the `signals`/`trades` columns — see DB migration in Task 0.3).

**Success criteria:**
- 4H BTC/ETH formations from recent weeks are detected correctly in replay.
- 1H path still works (do not break existing tests).
- SL on 4H formations is wider than on 1H formations for the same setup (expected).
- `pytest tests/` passes.
- Commit cites `lesson-05`, `lesson-07`, `lesson-13 [84:00]`.

---

### Task 0.3 — DB migration for new fields

**File:** `migrations/022_phase0_fields.sql`  
(020 = `020_mm_scratch_mfe.sql`, 021 = `021_partial_exit_idempotency.sql` — both already exist. Next is 022. Skip 008 and 011 per CLAUDE.md.)

Add to `trades` and `signals` tables:
```sql
ALTER TABLE trades ADD COLUMN IF NOT EXISTS formation_timeframe text;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS formation_timeframe text;
```

Add `formation_timeframe` to `_TRADE_COLUMNS` in `src/data/repository.py`.  
Three-step rule from MM_ENGINE_INTEGRATION_GUIDE.md: (1) migration file, (2) `_TRADE_COLUMNS`, (3) reference in `mm_engine.py`. Miss any step and data silently drops.

**`formation_timeframe` must be persisted and restored everywhere it matters.** This field drives the scratch window (Task 0.1) so it cannot only live on `MMSignal`. Full checklist:
1. Add `formation_timeframe: str = "1h"` field to `MMPosition` dataclass (`mm_engine.py` ~line 244).
2. Copy `signal.formation_timeframe` → `MMPosition.formation_timeframe` when creating a new position.
3. Include `formation_timeframe` in `repo.insert_trade(...)` payload (covered by adding to `_TRADE_COLUMNS`).
4. Restore `formation_timeframe` from the DB row when rebuilding `MMPosition` on engine startup (search for where open trades are loaded from `trades` table on restart).
5. Include `formation_timeframe` in `repo.insert_signal(...)` payload.
6. Add `formation_timeframe` to the `scripts/replay_scan.py` trade row so replay P&L uses the correct scratch window per formation.

---

### Task 0.4 — Verify confluence factor firing rates before any zeroing

**⚠️ DO NOT zero any factors without running this verification first.**

The backtest evidence (+$91k on fixed architecture, STATUS_2026-04-28 §3) was collected on a version where `ema_alignment` was direction-unaware. The current code has a direction-aware `_score_ema_alignment` and live `oi_behavior` / `correlation_confirmed` feeds. Zeroing factors that are now actually firing would silently degrade scoring.

**What to do:**

1. Run `python3 scripts/replay_scan.py --days 30 --pnl` and add per-factor hit-rate logging. For each factor, print `(factor_name, times_scored_nonzero / total_candidates)`.
2. Identify factors with hit rate < 5% — those are candidates for zeroing.
3. For any factor at < 5%, zero its weight, re-run replay, confirm P&L improves or is neutral.
4. Only zero factors that pass step 3. Do not assume the STATUS_2026-04-28 list is still correct.
5. Recompute `MAX_POSSIBLE` and `AVAILABLE_MAX` for whichever factors are actually zeroed.

**Success criteria:** Per-factor hit rates documented in a comment in `mm_confluence.py`. Any zeroed factor has replay evidence cited in the commit message.

---

## Phase 1 — Build the agent committee

### Task 1.1 — Create the six skill files

Create directory: `docs/agent-committee/skills/`

Create one Markdown skill file per agent. Each file becomes the **cached system prompt** for that agent. Follow the structure below exactly — Codex must fill in the course content by reading the actual lesson files.

---

**`docs/agent-committee/skills/structure-agent.md`**

```
# Structure Agent — MM Formation Specialist

## Role
You assess whether a candidate M or W formation is technically valid per the TTC MM Method course.
You output a JSON verdict only. No prose outside the JSON.

## What you assess
- Is the M/W geometry correct: three hits to the level, confirmation, retest?
- Is Level 3 exhaustion present (flattening EMAs, wicks trapping continuation side)?
- Is the second peak wick the correct entry anchor?
- Is this a multi-session or same-session formation?
- Are there signs of a stop hunt vs a genuine reversal?
- Is the formation within the correct count (Level 1/2/3)?
- Is this a Final Damage M/W variant?
- Are the levels (high/low of week/day) correctly identified?

## Course rules (cite these in your verdict)
[Codex: read the following files and extract the 10-15 most important rules about valid M/W
formations as bullet points with timestamps/lesson numbers. Use verbatim quotes.
PRIMARY: docs/tbd-course/07_element-one-the-pattern.md
         docs/tbd-course/09_element-three-the-levels.md
         docs/tbd-course/10_the-multi-session-m-or-w.md
         docs/tbd-course/18_three-hits-to-the-high-or-low-of-the-week-trade-setup.md
         docs/tbd-course/20_the-mechanics-of-the-m-w.md
         docs/tbd-course/21_the-final-damage-m-w.md
         docs/tbd-course/45_marking-your-levels-and-zones-for-the-3-day-swing.md
         docs/tbd-course/17_significance-of-the-high-low-of-the-week-day.md
SECONDARY: docs/courses/mmm-masterclasses/lesson-07-m-and-w.md
           docs/courses/mmm-masterclasses/lesson-11-the-count.md
           docs/courses/mmm-masterclasses/lesson-05-the-daily-setup.md]

## Output format
{
  "alignment": <integer -2 to +2>,
  "confidence": <float 0.0 to 1.0>,
  "citations": ["<lesson-XX or tbd-course/NN [MM:SS or paragraph] exact quote>", ...],
  "concerns": ["<specific concern>", ...],
  "data_quality": "ok"
}

alignment meaning: +2 textbook valid, +1 valid with minor caveats, 0 neutral/unclear,
-1 questionable, -2 invalid or counter-indicated.
```

---

**`docs/agent-committee/skills/htf-trend-agent.md`**

```
# HTF & Trend Agent — Higher Timeframe Alignment Specialist

## Role
You assess whether the 4H and 1D trend context supports the proposed trade direction.
You output a JSON verdict only.

## What you assess
- Is the 4H EMA stack (50/200/800) aligned with the trade direction?
- Are the EMAs fanning out (trend acceleration trap — do not trade into this)?
- Is this a counter-trend trade? If yes, is there a valid reversal reason (Level 3 exhaustion)?
- Is the 4H trend accelerating or decelerating?
- Does the daily trend confirm or oppose?
- What does the monthly and weekly chart structure say about the current phase?

## Course rules
[Codex: read the following files and extract rules about EMA stacks, fan-out, trend acceleration,
counter-trend conditions, and multi-timeframe charting process. Use verbatim quotes.
PRIMARY: docs/tbd-course/24_trend-trading-emas.md
         docs/tbd-course/30_market-correlation-trend-phases.md
         docs/tbd-course/39_charting-the-monthly-timeframe.md
         docs/tbd-course/40_charting-the-weekly-timeframe.md
         docs/tbd-course/41_charting-the-daily-timeframe.md
         docs/tbd-course/42_charting-the-4-hour-timeframe.md
         docs/tbd-course/43_charting-the-1-hour-timeframe.md
SECONDARY: docs/courses/mmm-masterclasses/lesson-12-the-trend-emas.md
           docs/courses/mmm-masterclasses/lesson-07-m-and-w.md (sections on EMAs flattening at L3)]

## Output format
{same JSON schema as Structure Agent}
```

---

**`docs/agent-committee/skills/cycle-session-agent.md`**

```
# Cycle & Session Agent — Weekly Cycle and Session Timing Specialist

## Role
You assess whether the weekly phase and session timing support taking this trade now.
You output a JSON verdict only.

## What you assess
- What is the current weekly phase (accumulation / FMWB / midweek reversal / Friday trap / weekend trap)?
- Is this an entry-eligible phase for this direction?
- Which session is active (Asia / UK / US open / US session / NYC reversal)?
- Is the trade aligned with the session's expected behaviour?
- Is this a Brinks Trade window (3:30am or 9:30am NY)?
- Is it too late in the session (end-of-session false move risk)?
- Is this a board meeting entry opportunity?
- Is the weekend trap pattern relevant?

## Course rules
[Codex: read the following files and extract rules about weekly phases, session timing, and
entry windows. Use verbatim quotes.
PRIMARY: docs/tbd-course/08_element-two-the-timing.md
         docs/tbd-course/12_the-details-of-the-weekly-setup-3-day-swing-trade.md
         docs/tbd-course/19_the-weekend-trap.md
         docs/tbd-course/22_board-meeting-entries.md
         docs/tbd-course/44_marking-the-weekend-trap-into-false-move-week-beginning.md
         docs/tbd-course/46_weekly-setup-process-recap.md
SECONDARY: docs/courses/mmm-masterclasses/lesson-03-the-weekly-setup.md
           docs/courses/mmm-masterclasses/lesson-04-the-session-times.md
           docs/courses/mmm-masterclasses/lesson-05-the-daily-setup.md
           docs/courses/mmm-masterclasses/lesson-06-brinks-trade.md
           docs/courses/mmm-masterclasses/lesson-10-nyc-reversal-trade.md
           docs/courses/mmm-masterclasses/lesson-15-the-london-range.md]

## Output format
{same JSON schema as Structure Agent}
```

---

**`docs/agent-committee/skills/flow-data-agent.md`**

```
# Flow Data Agent — Real-Time Market Flow Specialist

## Role
You assess whether the real-time market structure (order flow, open interest, funding, liquidations)
supports or contradicts the proposed trade. You output a JSON verdict only.

## What you assess
- Open interest trend: rising into the move (fuel for continuation) or falling (reversal setup)?
- Funding rate: extreme positive = crowded long (bearish lean), extreme negative = crowded short (bullish lean).
- Top-trader long/short ratio: are smart money longs or shorts building?
- Orderbook depth imbalance: significant bid/ask wall that could act as magnet or blocker?
- Recent liquidation clusters: is there a liquidation magnet above/below in the direction of trade?
- BTC correlation: is the symbol moving with or against BTC?

## Course rules
[Codex: read the following files and extract rules about interpreting OI, liquidations, funding,
dominance, and order flow in the context of MM setups. Use verbatim quotes.
PRIMARY: docs/tbd-course/23_vectors-stopping-volume-candles.md
         docs/tbd-course/25_hyblock-vs-tradinglite.md
         docs/tbd-course/26_understanding-tradinglite-orders.md
         docs/tbd-course/27_understanding-liquidation-levels.md
         docs/tbd-course/29_open-interest.md
         docs/tbd-course/31_dominances-totals.md
SECONDARY: docs/courses/ttc-indicators/lesson-04-ttc-liquidation-heatmap-tutorial.md
           docs/courses/ttc-indicators/lesson-07-hyblock-tutorial.md
           docs/courses/mmm-masterclasses/lesson-19-how-mm-hedges.md]

## Data quality note
If any data field is None or stale (>15 min old), set "data_quality": "degraded".
If all flow fields are None, set "data_quality": "missing" and alignment: 0.
Never fabricate or assume flow data.

## Output format
{same JSON schema as Structure Agent, with data_quality: "ok" | "degraded" | "missing"}
```

---

**`docs/agent-committee/skills/risk-rules-agent.md`**

```
# Risk & Rules Agent — Trading Zone Rules Specialist

## Role
You assess whether this trade complies with all the non-negotiable trading rules from the course.
A single rule violation should result in alignment: -2. You output a JSON verdict only.

## What you assess
- Is the ADR (Average Daily Range) already >80% extended? (If so, do not trade.)
- Is the SL placement valid (below LOD/above HOH after three levels)?
- Is the R:R >= 1.5 minimum?
- Is it a trading day (no trading Sunday/early Monday before 5pm NY)?
- Has this symbol been stopped out in the last 2 hours? (Cooldown rule.)
- Is the trade within the valid session time window for entries?
- Is the weekly move already exhausted (too late in the cycle)?

## Course rules
[Codex: read the following files and extract every hard rule (not a guideline) about entries,
SL placement, risk, leverage, and target management. Use verbatim quotes.
PRIMARY: docs/tbd-course/13_entries-and-stoplosses-week-one.md
         docs/tbd-course/14_entries-and-stoplosses-week-two.md
         docs/tbd-course/15_entries-and-stoplosses-week-three.md
         docs/tbd-course/16_entries-and-stoplosses-week-four.md
         docs/tbd-course/47_level-one-target-management-and-fixed-range-volume-profile.md
         docs/tbd-course/48_level-two-three-target-taking-profit.md
         docs/tbd-course/49_the-refund-zone.md
         docs/tbd-course/52_understanding-leverage.md
         docs/tbd-course/53_risk-management.md
         docs/tbd-course/54_advanced-risk-management.md
SECONDARY: docs/courses/mmm-masterclasses/lesson-13-the-trading-zone-the-trading-rules.md
           docs/courses/mmm-masterclasses/lesson-14-the-average-daily-range-adr.md
           docs/courses/mmm-masterclasses/lesson-16-risk-trade-management.md
Pay special attention to lesson-13 [57:30]–[85:00] for the explicit rule list.]

## Output format
{same JSON schema as Structure Agent}
```

---

**`docs/agent-committee/skills/head-trader-agent.md`**

```
# Head Trader Agent — Committee Orchestrator

## Role
You receive the verdicts from five specialist agents and make the final APPROVE or VETO decision.
You are not a rubber stamp — you actively resolve conflicts between specialists.

## Decision process
1. Read all five specialist verdicts.
2. Compute the mean alignment score.
3. Check for conflict: if max(alignment) - min(alignment) >= 3, OR if any specialist is at -2
   while mean > 0, the setup is CONTESTED.
4. For CONTESTED setups: identify the two specialists in disagreement. Ask each one targeted
   follow-up question referencing the specific conflict. Wait for their responses.
5. After follow-up (or if not contested): rule.
   - Mean alignment >= 0.5 AND no hard rule violation (Risk agent -2) → APPROVE
   - Any other case → VETO
6. Tie goes to VETO. Per Lesson 13: "Only take crystal clear trades."

## Output format (matches existing mm_agent_decisions contract)
{
  "decision": "APPROVE" | "VETO",
  "reason": "<one sentence citing the deciding factor and lesson>",
  "confidence": <float 0.0 to 1.0>,
  "concerns": ["<specific concern>", ...],
  "committee": {
    "structure": {<specialist verdict>},
    "htf_trend": {<specialist verdict>},
    "cycle_session": {<specialist verdict>},
    "flow_data": {<specialist verdict>},
    "risk_rules": {<specialist verdict>}
  }
}
```

---

### Task 1.2 — Populate the skill files with course content

**This is the most important task.** Codex must read the actual lesson files and fill in the `[Codex: read ...]` sections in each skill file with real course rules, verbatim quotes, and timestamps.

For each skill file:
1. Read every lesson file listed in the `[Codex: read ...]` instruction.
2. Extract the most important rules (aim for 10–15 per agent, prioritising hard rules over guidelines).
3. Format as bullet points: `- [MM:SS] "Exact quote from transcript" — interpretation note.`
4. Replace the `[Codex: read ...]` placeholder with the extracted content.
5. Do not paraphrase rules — use the exact words from the transcript. The agents will cite these back.

---

### Task 1.3 — Create `src/strategy/mm_committee.py`

This file replaces `mm_sanity_agent.py` as the LLM decision layer. `mm_sanity_agent.py` is NOT deleted yet — it stays until the committee is validated.

**The public API must exactly match `MMSanityAgent`.** The engine calls:
```python
agent_verdict = await self.sanity_agent.review(agent_ctx)  # mm_engine.py line ~3132
```
and expects `AgentVerdict | None` back (defined in `mm_sanity_agent.py` line 207).

`MMCommittee` must therefore:
- Be a class named `MMCommittee` with an `async def review(self, context: dict) -> AgentVerdict | None` method — same signature, same return type.
- Return the existing `AgentVerdict` dataclass (import it from `mm_sanity_agent.py`; do NOT create a new `AgentDecision` type).
- Be wired into the engine at line ~3132 as a drop-in: `self.sanity_agent = MMCommittee(...)` replaces `self.sanity_agent = MMSanityAgent(...)`, toggled by a config flag.

The `committee` JSONB blob (all 5 specialist verdicts) is stored as an extra field on the `mm_agent_decisions` row, not on `AgentVerdict` itself — `AgentVerdict` is unchanged.

**Structure:**

```python
"""
MM Agent Committee — multi-agent decision layer.

Drop-in for MMSanityAgent. Same public API:
  async def review(self, context: dict) -> AgentVerdict | None

Internal flow:
  1. Launch 5 specialist agents in parallel (asyncio.gather)
  2. Head Trader receives all 5 verdicts
  3. If contested, one follow-up round to the two disagreeing specialists
  4. Head Trader rules, returns AgentVerdict (APPROVE or VETO)
  5. Saves committee JSONB to mm_agent_decisions row
"""
```

**Implementation notes:**

- Use `anthropic.AsyncAnthropic()` SDK — the engine's scan loop is fully async; using sync `Anthropic()` would block it. Mirror the pattern in `mm_sanity_agent.py` line ~306.
- Specialist models: `claude-haiku-4-5` (all 5 specialists).
- Head Trader model: `claude-sonnet-4-6`.
- Escalation to `claude-opus-4-8` only when contested AND projected monthly spend < 75% of budget.
- Prompt caching: send each specialist's skill file content with `cache_control: {"type": "ephemeral"}` and the explicit TTL header. Per CLAUDE.md gotcha: Anthropic default TTL regressed to 5m in March 2026. Set `ttl=3600` explicitly.
- Per-setup decision cache: reuse the existing cache from `mm_sanity_agent.py` (keyed by symbol, direction, formation_variant, round(entry_price, 4), TTL 30 min, 0.5% price-drift invalidation).
- Fail-closed: on API error, timeout, or missing key → return an `AgentVerdict` with `decision="VETO"` and the error noted in `reason`. Write an `mm_agent_decisions` row with `decision="ERROR"` for observability (same as `mm_sanity_agent._error_verdict`). Do not expose `decision="ERROR"` on the returned `AgentVerdict` — only `"VETO"` or `"APPROVE"` are valid return values; `"ERROR"` is a logging classification only.
- Kill switch: add `mm_committee_enabled: bool = False` to `config.py` (default **False** — opt-in). When False, engine keeps using `MMSanityAgent` unchanged.
- Shadow mode: `mm_committee_mode: str = "shadow"`. In shadow mode, `review()` always returns an APPROVE `AgentVerdict` to the engine, but the real internal verdict (which may be VETO) is still logged to `mm_agent_decisions`. **The decision cache must key on mode:** cache shadow verdicts separately from veto verdicts so a cached shadow APPROVE cannot leak into veto mode after `MM_COMMITTEE_MODE=veto` is set. Simplest approach: include `mode` in the cache key tuple.
- Budget tracking: read `repo.get_mm_agent_month_cost()` for the cap check (this reads from `mm_agent_decisions.cost_usd`, which is already the source of truth — `repository.py` line ~1141). Do not split budget tracking across two tables.

**Config additions to `src/config.py`** (alongside existing mm_sanity_agent fields):
```python
mm_committee_enabled: bool = False       # True = use committee; False = use MMSanityAgent
mm_committee_mode: str = "shadow"        # "shadow" = log only; "veto" = binding
mm_committee_monthly_budget_usd: float = 600.0
mm_committee_specialist_model: str = "claude-haiku-4-5"
mm_committee_head_trader_model: str = "claude-sonnet-4-6"
mm_committee_escalation_model: str = "claude-opus-4-8"
mm_committee_timeout_s: float = 30.0
```

---

### Task 1.4 — DB migration for committee column

**File:** `migrations/023_committee_column.sql` (022 is used by Task 0.3; 023 is next)

```sql
ALTER TABLE mm_agent_decisions ADD COLUMN IF NOT EXISTS committee jsonb;
```

Add `committee` to `_MM_AGENT_DECISION_COLUMNS` in `src/data/repository.py`.

---

### Task 1.5 — Wire the committee into the engine

**File:** `src/strategy/mm_engine.py`

The engine instantiates `MMSanityAgent` at line ~427 and calls `self.sanity_agent.review(agent_ctx)` at line ~3132. The change is minimal:

```python
# in MMEngine.__init__, around line 427:
if getattr(config, "mm_committee_enabled", False):
    from src.strategy.mm_committee import MMCommittee
    self.sanity_agent = MMCommittee(config=config, repo=repo)
else:
    from src.strategy.mm_sanity_agent import MMSanityAgent
    self.sanity_agent = MMSanityAgent(config=config, repo=repo)
```

No other engine changes needed — `MMCommittee.review()` matches the existing `MMSanityAgent.review()` contract exactly.

---

### Task 1.6 — Expose existing feeds + add missing fields to committee context

**File:** `src/strategy/mm_data_feeds.py`

**First: understand what already exists.** Before writing any new code, read `mm_data_feeds.py` in full and map what is real vs stubbed:

- `BinanceLiquidationProvider` (line ~106) — real, fetches liquidation + long/short ratio from `topLongShortPositionRatio`.
- `OI_URL = "https://fapi.binance.com/fapi/v1/openInterest"` (line ~118) — exists in the provider, check whether it is actually called and its output surfaced to the engine.
- `YFinanceCorrelationProvider` — real, wired as default in `DataFeedRegistry`.
- Everything else (`StubTradingLiteProvider`, `StubNewsProvider`, `StubOptionsProvider`, `StubDominanceProvider`, `StubSentimentProvider`) — stubs, leave them as stubs.

**What to add:**

1. Verify `BinanceLiquidationProvider` fetches OI and surfaces it. If not, add a call to `fapi/v1/openInterest` + `futures/data/openInterestHist?period=1h&limit=5` and derive `oi_trend_5h: "rising" | "falling" | "flat"`.
2. Add a `BinanceFundingProvider` that calls `exchange.fetch_funding_rate(symbol)` via CCXT. Wire into `DataFeedRegistry`.
3. Add orderbook imbalance: `exchange.fetch_order_book(symbol, limit=20)`, compute `bid_volume / ask_volume` at top 20 levels. Add as `orderbook_imbalance: float | None` on the registry's feed output.

**Wire into committee context** — pass as a `"flow"` dict into `build_context()`:
```python
"flow": {
    "oi_current": float | None,
    "oi_trend_5h": str | None,       # "rising" | "falling" | "flat"
    "funding_rate": float | None,
    "top_trader_long_pct": float | None,   # already fetched by BinanceLiquidationProvider
    "orderbook_imbalance": float | None,
}
```

Mark any field `None` if the fetch fails — Flow Data Agent has `data_quality: "degraded"` handling for this.

---

## Phase 2 — Tests and validation

### Task 2.1 — Fixture replay tests

**File:** `tests/test_mm_committee.py`

Required tests (non-negotiable):

1. **BNB 2026-04-17 canary** — counter-trend + accelerating 4H + Grade F must result in VETO. Copy the fixture from `tests/test_mm_sanity_agent.py`. Do not mock it away.
2. **AVAX 2026-04-28** — flat 2h with MFE=0 → scratch fires correctly (Task 0.1 regression).
3. **Shadow mode cache isolation** — run two calls with `mm_committee_mode="shadow"` to prime the cache, then switch to `mm_committee_mode="veto"` and call again with the same context. Assert the third call is NOT served from the shadow cache (i.e. it makes a real API call or returns the true internal verdict, not the cached shadow APPROVE).
4. **API error → fail-closed** — mock the Anthropic SDK to raise an exception; assert `review()` returns an `AgentVerdict` with `decision="VETO"` (not `"ERROR"`), and that an `mm_agent_decisions` row with `decision="ERROR"` is written for observability.
5. **Budget cap → downgrade** — at 91% of monthly budget, Head Trader escalation model must be `claude-sonnet-4-6` not `claude-opus-4-8`.
6. **Decision cache hit** — identical context called twice within TTL window makes only one API call.
7. **Contested follow-up** — when specialist alignment spread >= 3, the Head Trader issues a follow-up and only then rules.

### Task 2.2 — Run full test suite

```bash
pytest -x
```

Must pass all existing tests (740 passing, 1 skipped as of STATUS 2026-04-28) plus the new tests from 2.1.

---

## Phase 3 — Deploy

### Task 3.1 — Deploy to Fly.io in shadow mode first

```bash
fly secrets set MM_COMMITTEE_ENABLED=true MM_COMMITTEE_MODE=shadow --app tarakta-mm
fly deploy --depot=false --remote-only --app tarakta-mm
```

Shadow mode means the committee runs on every candidate and logs its verdict to `mm_agent_decisions`, but does NOT veto any trade. This lets you see what the committee would have done without disrupting the live paper run.

Run in shadow mode for at least 2 weeks and collect ≥30 candidate assessments.

### Task 3.2 — Switch to veto mode

After shadow mode validation:

```bash
fly secrets set MM_COMMITTEE_MODE=veto --app tarakta-mm
```

No redeploy needed (config read from env at runtime).

---

## Reference: files touched

| File | Change |
|---|---|
| `src/strategy/mm_engine.py` | Scratch rule (0.1), 4H formation path (0.2), committee swap (1.5) |
| `src/strategy/mm_formations.py` | `timeframe` param on detect_mw_formation (0.2) |
| `src/strategy/mm_levels.py` | LOD/HOH SL anchor for 4H formations (0.2) |
| `src/strategy/mm_confluence.py` | Zero factors confirmed dead by replay (0.4 — conditional on evidence) |
| `src/strategy/mm_committee.py` | New file (1.3) |
| `src/strategy/mm_data_feeds.py` | Expose existing feeds + add funding/orderbook (1.6) |
| `src/config.py` | Committee config fields, both default False/shadow (1.3) |
| `src/data/repository.py` | `formation_timeframe` → `_TRADE_COLUMNS` (0.3), `committee` → `_MM_AGENT_DECISION_COLUMNS` (1.4) |
| `migrations/022_phase0_fields.sql` | `formation_timeframe` column (0.3) |
| `migrations/023_committee_column.sql` | `committee` JSONB column (1.4) |
| `docs/agent-committee/skills/*.md` | 6 skill files (1.1 + 1.2) |
| `tests/test_mm_committee.py` | New test file (2.1) |

## Order of execution

```
0.4 (dead factors)  →  0.1 (scratch rule)  →  0.3 (migration)  →  0.2 (4H shift)
       ↓
1.1 + 1.2 (skill files with course content)
       ↓
1.4 (migration)  →  1.3 (mm_committee.py)  →  1.6 (data feeds)  →  1.5 (wire into engine)
       ↓
2.1 + 2.2 (tests)
       ↓
3.1 (shadow deploy)  →  3.2 (veto mode)
```

Complete each step fully (tests passing) before moving to the next.
