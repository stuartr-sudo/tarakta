# MM Agent Committee — Design Document

**Status:** Draft v1, 2026-06-13. Replaces the single sanity agent (`mm_sanity_agent.py`) with a multi-agent committee built on the Claude Agent SDK, running inside the bot.

**Decisions locked with Stuart (2026-06-13):** runs inside the bot · approve/veto authority only · engine fixes (scratch rule, timeframe shift) in scope · ~$600/mo LLM budget.

---

## 1. Why — the evidence

Live paper results, `tarakta-mm`, 2026-04-20 → 2026-06-11 (from Supabase):

| Period | Trades | P&L | Win % | Notes |
|---|---:|---:|---:|---|
| Agent era (Apr 20–27, Opus 4.7 veto layer ON) | 10 | **+$307** | 30% | 282 VETO / 9 APPROVE (97% veto) |
| Agent off (Apr 28 → Jun 11) | 62 | **−$7,613** | 26% | Deterministic engine alone |

Exit-reason breakdown (full window): `scratch_2h` 35 trades −$3,069 (49% of all exits), `stop_loss` 24 trades −$6,815 (avg −$284), winners only 8 exits totalling +$2,730.

The inverse mirror bot (`tarakta-mm-inverse`, May 29 →) is **also negative** (−$279 / 12 trades). When a strategy and its inverse both lose, the candidate stream is near-random and costs + scratch timing eat both sides.

**Conclusions:**

1. The LLM veto layer was not the problem — it was the only profitable configuration. It "ruined everything" only in the sense that it correctly refused 97% of a negative-expectancy candidate stream.
2. A selection layer alone cannot fix this. If candidates average negative expectancy, any honest committee must veto ~everything. The committee must therefore ship **together with** engine fixes that raise candidate quality (§6).
3. The old agent reasoned from a thin rubric with a then-broken HTF feed. The committee gets richer inputs: real-time flow data (§5) and the full course corpus as per-agent skills.

---

## 2. Architecture

```
 MM Engine scan (deterministic, unchanged role: candidate generator)
        │  gates 1–8 (candles → formation → HTF → … → R:R)
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │              AGENT COMMITTEE  (Claude Agent SDK)         │
 │                                                          │
 │  Round 1 — parallel, independent:                        │
 │   • Structure Agent      (Haiku 4.5)  skills/structure   │
 │   • HTF & Trend Agent    (Haiku 4.5)  skills/htf-trend   │
 │   • Cycle & Session Agt  (Haiku 4.5)  skills/cycle-sess  │
 │   • Flow Data Agent      (Haiku 4.5)  skills/flow-data   │
 │   • Risk & Rules Agent   (Haiku 4.5)  skills/risk-rules  │
 │        each returns {alignment −2..+2, confidence,       │
 │                      citations[], concerns[]}            │
 │                                                          │
 │  Round 2 — Head Trader (Sonnet 4.6):                     │
 │   reads all 5 verdicts; if contested (see §4) may ask    │
 │   ONE targeted follow-up to ≤2 agents, then rules.       │
 │   Escalates to Opus 4.8 only when round-2 is still       │
 │   contested AND budget headroom exists.                  │
 │                                                          │
 │  Output: APPROVE | VETO + reason + lesson citations      │
 └──────────────────────────┬──────────────────────────────┘
                            ▼
                MMSignal → sizing → execution
```

Principles carried over from the v1 agent design (still correct):

- **Engine computes, agents reason.** Every derived feature is pre-computed and handed over; no agent does arithmetic on candles.
- **Pipeline position unchanged:** after `retest_passed`, before `MMSignal` build.
- **Fail-closed** when enabled (a skipped trade is cheaper than an unfiltered one, given the candidate stream's history). Kill switch: `MM_COMMITTEE_ENABLED=false`.
- **Per-setup decision cache** retained (30 min TTL, 0.5% price-drift invalidation) — this is what makes the budget work.
- **Prompt caching `ttl=1h`, set explicitly** (Anthropic default regressed to 5m, March 2026). Each agent's skill file + fixtures live in the cached system prompt.
- **BNB 2026-04-17 canary preserved:** counter-trend + accelerating + Grade F must be visible to (and vetoed by) the committee in tests. Do not mock it away.

### What "agents discussing findings" means here

Unbounded multi-turn debate is unaffordable and non-deterministic. The committee implements bounded cross-referencing:

1. Specialists assess independently (no anchoring on each other).
2. Head Trader detects conflicts — e.g. Structure says +2 ("textbook W, three hits to the low") while Flow says −2 ("OI rising into the move — trend continuation, not reversal").
3. One follow-up round maximum: Head Trader sends the specific conflict back to the two disagreeing agents ("Flow reports rising OI; Structure, does that change your read of peak 2?").
4. Head Trader rules, citing lessons. Disagreement unresolved → VETO (tie goes to no-trade; Lesson 13: "Only take crystal clear trades").

---

## 3. Agents and their skills

Skill files live in `docs/agent-committee/skills/` (move to the SDK's skill dir at implementation). Each skill = role + course citations + structured-output contract. Summary:

| Agent | Owns | Primary lessons |
|---|---|---|
| Structure | M/W validity, the count (levels 1–3), peak geometry, wick traps, stop-hunt vs reversal | L7, L11, L5 |
| HTF & Trend | 4H/1D EMA stack, fan-out (trend acceleration trap), counter-trend discipline | L12, L7 [07:30] |
| Cycle & Session | Weekly phase, session timing, Brinks windows, NYC reversal, London range, Friday behaviour | L3, L4, L5, L6, L10, L15 |
| Flow Data | Real-time: liquidations, OI, funding, long/short ratio, orderbook depth, BTC correlation | ttc-indicators L4/L7, L19 |
| Risk & Rules | Trading-zone rules, ADR exhaustion, SL placement sanity, R:R, leverage, time-of-week rules | L13, L14, L16, L17 |
| Head Trader | Synthesis, conflict resolution, final APPROVE/VETO | all (cites via specialists) |

Specialist output contract (JSON, enforced by schema):

```json
{
  "alignment": 1,            // −2 strong against … +2 strong for
  "confidence": 0.7,
  "citations": ["L7 [05:00] wicks at level 3 trap the continuation side"],
  "concerns": ["peak separation 22 bars is near the 24-bar cap"],
  "data_quality": "ok"       // "ok" | "degraded" | "missing" — Head Trader discounts degraded inputs
}
```

Head Trader output matches the existing `mm_agent_decisions` contract (APPROVE/VETO/reason/confidence/concerns) plus a `committee` JSONB blob with all five verdicts — one new column, one migration, **and** the column name added to `_MM_AGENT_DECISION_COLUMNS` in `repository.py` (see MM_ENGINE_INTEGRATION_GUIDE.md — three-step rule).

---

## 4. Budget (~$600/mo cap)

Per committee run (with prompt caching, post-cache pricing):

| Stage | Model | Est. tokens (in/out) | Est. cost |
|---|---|---|---:|
| 5 specialists | Haiku 4.5 | ~10k / 1k each | ~$0.075 |
| Head Trader | Sonnet 4.6 | ~20k / 2k | ~$0.09 |
| Follow-up round (assume 30% of runs) | Haiku ×2 + Sonnet | — | ~$0.05 amortised |
| **Total per run** | | | **~$0.20** |

At 30 cached runs/day → **~$180/mo**, leaving headroom for Opus 4.8 escalations (~$0.60/escalation; cap at ~10/day) and the daily review job. Budget cap is checked via `repo.get_mm_agent_month_cost()`, which reads `mm_agent_decisions.cost_usd` — the same source of truth used by `MMSanityAgent`. Auto-downgrade Head Trader Sonnet → Haiku at 90% of `MM_COMMITTEE_MONTHLY_BUDGET_USD`, disable Opus escalation entirely at 75%.

"Contested" definition for escalation: max(alignment) − min(alignment) ≥ 3, or any specialist at −2 while mean > 0.

---

## 5. Real-time data — unstubbing `mm_data_feeds.py`

The Flow Data Agent is only as good as its feeds. Current state: only Binance liquidations + yfinance correlation are real; Hyblock, TradingLite, news, options, dominance, sentiment are stubs.

Priority order (free, Binance-native, no new vendor risk):

1. **Open interest** — `fapi/v1/openInterest` + `futures/data/openInterestHist` (rising OI into formation = continuation fuel, falling = squeeze done)
2. **Funding rate** — already on CCXT; extreme funding = crowded side
3. **Top-trader long/short ratio** — `futures/data/topLongShortPositionRatio`
4. **Orderbook depth imbalance** — CCXT `fetch_order_book`, top-20 levels
5. **BTC/ETH correlation + BTC dominance** — yfinance (partially real already)

Defer (paid/noisy): Hyblock liquidation heatmaps, news, sentiment, options. The agent's `data_quality` field handles their absence honestly — no stub ever silently feeds the committee fake-neutral values (that is exactly the broken-HTF failure mode that mis-calibrated agent v1).

---

## 6. Engine fixes (in scope, course-cited)

### 6a. Scratch rule rework — the −$3,069 bleeder

Course basis, Lesson 13: the threshold is **"enough profit … that allows you to move your stop to Breakeven"** [44:00], and the 2h window is **"for most of the daily setups"** — **"I'd hold on longer … if I found something on a four hour, or a daily"** [102:30].

Changes:

1. Replace the fixed `scratch_mfe_threshold_r = 0.3` with "MFE reached breakeven-move distance" (i.e. price moved ≥ the distance the BE ladder requires) — measures what the course measures.
2. Scratch window scales with formation timeframe: 1H setups keep 2h; 4H/1D setups get evaluated on closed 4H bars (propose 2 closed 4H bars, flagged as interpretation — re-read L13 [102:30] before committing; cite in commit).
3. Keep the stopped-out cooldown (no re-entry for 2h, L13 [79:00]) — already course-faithful.

Test the rework against the canonical AVAX 2026-04-28 scratch and the 35 live scratch trades (replay them: how many recovered vs stopped out later — that's the homework Annii assigns at L13 [76:30], literally).

### 6b. Timeframe shift — 4H/daily as primary

Stuart's read (agreed): 1H pattern recognition is too noisy, poisoning entry and SL placement. Direction:

- Formation detection primary on **4H** with 1D context; 1H demoted to timing/retest confirmation only.
- SL per L13 [84:00]: "below the low of the day, or above the high of the day after your three levels" — derive SL from LOD/HOD structure, not formation-internal wicks that whipsaw.
- Swing-hold management per L13 [85:00]: partials at each boardroom, stops trail under boardrooms.
- Expect far fewer candidates (good: "Stop trading every day" — L13 [57:30]) and longer holds; position the bot as 1–3 trades/week, not 2/day.

This is the biggest change and needs its own replay validation before deploy. Note the known replay limitation (SWING_WINDOW + closed-bar visibility) shrinks but does not vanish on 4H.

### 6c. Known cleanups that ride along

- Dead-factor zero-out in `mm_confluence.py` (+$91k backtest evidence, still unshipped).
- Engine self-seeds `engine_state` row at startup if missing (STATUS 2026-04-28 §5).
- TBD System Pt1 is already scraped at `docs/tbd-course/` (55 lessons). The skill files in `CODEX_PLAN.md` reference it correctly. Nothing to scrape.

---

## 7. Validation gates (in order, each blocks the next)

1. **Fixture replay:** committee re-judges the 291 historical agent decisions (Apr 20–24) + the 72 closed trades with known outcomes. Must beat both baselines: agent-off P&L (−$7.6k) and single-agent-era selection quality.
2. **BNB canary:** committee VETOs the 2026-04-17 pattern. Hard test, never mocked.
3. **Shadow mode (Phase 1):** committee runs live, logs verdicts, vetoes nothing. ≥2 weeks. Compare its would-have P&L vs actual.
4. **Live veto (Phase 2):** enable. ≥30 closed trades on paper before any settings loosening.
5. Only then: revisit gate thresholds / Rubric-8-style outcome learning on clean data.

---

## 8. Implementation phases

**Sequencing rule: Phase 0 engine fixes must complete and pass replay before the committee is activated in any mode.** Skill-file authoring (Phase 1a) can proceed in parallel with Phase 0 since it touches no engine code.

| Phase | Work | Size |
|---|---|---|
| 0 | Engine fixes: scratch rework → 4H timeframe shift → formation_timeframe persistence → confluence factor audit. Migrations 022. Each step replay-validated before next. | ~4 sessions |
| 1a (parallel with 0) | Skill files authored from course content (`docs/agent-committee/skills/`). No code changes. | ~1 session |
| 1b (after 0) | `mm_committee.py`: `AsyncAnthropic`, `review() -> AgentVerdict`, shadow/veto mode, cache keyed by mode, budget via `get_mm_agent_month_cost()`. Migration 023. Wire into engine behind `mm_committee_enabled=False`. | ~3 sessions |
| 2 | Expose existing feeds + add funding/orderbook to committee context (1.6). Fixture replay tests including BNB canary, shadow cache isolation, error→VETO. Full test suite green. | ~2 sessions |
| 3 | Shadow deploy (≥2 weeks, ≥30 candidates). Then flip to veto. | ongoing |

---

## 9. Open questions

- Exact "closed 4H bars" scratch interpretation — needs L13 [100:00–105:00] re-read before commit.
- Whether `signals` table or new `mm_committee_decisions` table stores round-1 verdicts (leaning: JSONB on `mm_agent_decisions`, keep one table).
- Whether the daily Tier-3 review job (ROADMAP §2) becomes a Head Trader weekly self-critique that proposes skill-file edits (human-approved, never auto-applied).
