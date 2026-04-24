# MM Engine Changelog

**Append-only.** Every engine / strategy / risk / agent change lands here with:
- commit SHA
- one-line summary
- **course citation** when the fix is course-driven (quote + lesson reference)
- what was wrong before, what's right now

This is the canonical "what did we change and why" log. When a future Claude session (or human) asks "why is X the way it is?", this answers it.

Paired docs:
- `docs/STATUS_YYYY-MM-DD.md` — daily snapshot of current state (not append-only, can be regenerated)
- `CLAUDE.md` — project guide
- `docs/MM_SANITY_AGENT_DESIGN.md` — Agent 4 spec
- `docs/MM_ENGINE_INTEGRATION_GUIDE.md` — DB contract rules
- `docs/COURSE_AUDIT_2026-04-19.md` — background audit report

---

## 2026-04-23 — Day 6: Rubric 8 min-n filter (rubric_v=3) + three user-reported bugs

Multiple fixes shipped on the same day. Structured as:
1. `eda4193` — Rubric 8 min-n filter (agent over-vetoing from small samples)
2. `ca38f8d` — `sl_too_wide` gate re-added (late-retest entries, lesson 9)
3. `01038af` — entry-slippage gate from peak2 wick (P1/3, this batch)
4. `a2c00d1` — per-setup decision cache (P2/3, this batch)
5. `40a0860` — MFE-based 2h scratch rule (P3/3, this batch)

### `eda4193` — fix(mm-agent): engine-side min_n filter on Rubric 8 outcome stats

**Motivation — v44 regression discovered:**

48 hours after v44 shipped (2026-04-21 00:40 UTC), the user noted "0 trades in a long time." Diagnosis confirmed a silent Tier 2 regression:

| Day | APPROVE | VETO | Rate |
|---|---|---|---|
| 2026-04-20 (pre-v44) | 7 | 9 | 41% |
| 2026-04-21 (v44 day) | 1 | 133 | 0.7% |
| 2026-04-22 | 0 | 102 | 0% |

The 2026-04-22 STATUS doc had read the 0% approve rate as *"agent correctly refusing losing profiles"*. It wasn't. Three defects in `rubric_v=2`:

1. **Thresholds are statistical noise.** "n ≥ 3 prefer VETO" and "n ≥ 5 strongly prefer VETO" fire on samples where win-rate CI spans ~[3%, 66%]. Cannot distinguish regime signal from variance.
2. **Model wasn't enforcing the rubric's own gates.** Apr 22 VETOs cited `"F|sideways profile already 0W/1L"` as a reason — n=1, which the rubric says should give "no penalty." "Prefer VETO" language was too soft against a pattern the model was primed to see.
3. **Selection-bias doom loop.** `get_mm_agent_outcome_stats` aggregates APPROVE decisions only. The agent is shown the P&L of its own best past guesses. When those lose, the rubric teaches "approve less" — which shrinks future samples and ratchets deeper into veto-everything. No mechanism to recover.

**Fix shipped (rubric_v=3):**

- New config `MM_SANITY_AGENT_OUTCOME_MIN_N` (default 20 closed samples). `src/config.py`.
- `_build_user_prompt` pre-filters `outcome_stats` to buckets where `wins + losses + scratches ≥ min_n`. Opens don't count — outcome unknown.
- When no bucket passes, the block is replaced with a single unified message: *"(insufficient data — no profile has ≥20 closed samples yet; skip Rubric 8 and decide on Rubrics 1-7 only)"*. Collapses the old "first pass" and "all filtered" branches into one.
- Rubric 8 text rewritten: instructs model to SKIP this rubric entirely when THIS PROFILE is not in the (filtered) block. Explicit "never cite a profile that isn't in the block" — closes defect #2.
- `PROMPT_VERSION` bumped `prompt_v=2 rubric_v=2` → `prompt_v=3 rubric_v=3`. Prompt cache rebuilds on first call post-deploy (one-time ~$0.04 cache write).

**Statistical rationale for min_n=20:** n=20 is roughly the sample size at which a 75% vs 25% win rate becomes distinguishable from variance at 95% CI. Lower values trade statistical rigour for faster feedback-loop activation. Tunable without redeploy.

**Defect #1 (thresholds)** — resolved by engine-side filter.
**Defect #2 (model ignoring rubric gates)** — resolved by not emitting small-sample buckets at all.
**Defect #3 (selection-bias doom loop)** — mitigated: buckets must reach n=20 before they gate future approvals, giving the agent 2-4 weeks to accumulate samples before self-blocking. Not fully solved — the feedback loop is still one-directional. Logged in ROADMAP as a future enhancement (two-way signal: if a profile runs clearly positive, surface that as *bullish* prior).

**Tests:** 6 new in `TestUserPromptOutcomeStatsFilter`. 3 existing (`test_no_stats_…`, `test_stats_rendered_…`, `test_prompt_version_bumped_…`) updated for v3 semantics. Full suite: **699 passing, 1 skipped**.

**Not a course change.** Rubric 8 is about the agent calibrating from its own track record, not about MM course rules. No course lesson citation required.

Ref: `docs/STATUS_2026-04-23.md` for full session notes.

### User-reported bugs batch (P1 + P2 + P3)

User flagged three issues after the first week of live operation:

> "The 2h scratch is odd... it seems to always close out too early...
> if the initial trade goes in the wrong direction, but comes back in
> the direction we expect, it gets closed... before it can actually
> realized. The entry points are all wrong... a lot of trades are
> getting rejected for 'SL too wide'... but that is actually because
> the entry point is NOT being calculated correctly. You made a HUGE
> mistake... the Agent was only supposed to be called when a setup is
> confirmed... NOT all potential trades. THAT cost me too much in
> wasted API credit."

All three shipped as P1/P2/P3 on the same day.

### `01038af` — fix(mm): entry-slippage gate from peak2 wick (P1/3)

**Motivation:** BTC entry on 2026-04-20 was +4.39% above the course's
"retest level" (2nd-peak wick). The engine was using `current_price`
as the entry reference, which silently drifted from the course-correct
peak2 wick whenever the 1H formation candle was not the most recent.
This cascaded:
- entry too high → SL placement too far below → "SL too wide" reject
- symptoms misdiagnosed: the rejection reason was the entry, not the SL

**Course citation (Lesson 20 [08:00] / Lesson 47 [12:00]):**
> *"The retest of the 2nd peak is where we enter — that wick is the
> level, not wherever price is sitting when we scan."*

**Fix:** new `mm_max_entry_slippage_pct` config (default 1.0%). Before
setting `entry_price = current_price`, we compute the slippage from the
peak2 wick. Setups where current_price has moved more than 1% past the
wick are rejected with `rejection_reason='entry_slipped_from_retest'`.

**Tests:** 10 new — config + runtime overrides + BTC/NEAR/BNB
regression guards with specific computed numbers.

### `a2c00d1` — fix(mm-agent): per-setup decision cache (P2/3)

**Motivation:** 82 identical Opus 4.7 calls on the same DOGE long setup
over 6 hours (2026-04-21). The 1H formation detector re-emits the same
setup every 5-min scan; each re-emit billed a fresh agent call (~$0.05
each) for zero new information. Projected spend spiked from $6/mo →
$250/mo.

**Fix:** in-memory cache on `MMSanityAgent` keyed by
`(symbol, direction, formation_variant, round(entry_price, 4))`. Hits
within the configurable TTL (30 min default) + price-drift bound
(0.5% default) replay the cached verdict. Cached verdicts prefix reason
with `"[cached]"` and report zero cost/latency.

**Invalidation rules:**
- TTL expiry — prevents a stale verdict from outliving regime change.
- Price drift above threshold — refetch when the setup has shifted.
- Malformed API responses never populate the cache (no poison).
- Kill switch short-circuits BEFORE cache lookup (no cached verdicts
  after disabling the agent).

**Config:** `mm_sanity_agent_cache_ttl_seconds` (0 disables),
`mm_sanity_agent_cache_price_drift_pct`.

**Tests:** 14 new — 7 cache-key edge cases + 7 review() integration
tests (first call, repeat hit, TTL expiry, drift invalidation, variant
/ symbol / direction key changes, malformed non-caching, kill switch
bypass).

**Expected impact:** 82-call DOGE pattern → ~12 calls over 6h (85%
reduction), cost back toward the designed $6/mo.

### `40a0860` — fix(mm-engine): MFE-based 2h scratch rule (P3/3)

**Motivation:** user's "closes too early" report. The B1 rule measured
unrealized P&L at the 2h instant only, so a trade that had been +1R
mid-flight but pulled back to break-even by the check was being
scratched — even though the MM clearly had been "holding for a move"
earlier.

**Course citation (Lesson 13 [47:00]):**
> *"If you're not in substantial profit within two hours you scratch
> the trade. It means the Market Maker has a different plan."*

The word "within" is a window, not an instant — the prior
implementation was over-eager.

**Fix:** track Max Favorable Excursion (MFE) in R-multiples on every
`_manage_position` tick. `R = abs(entry - original_stop_loss)`. At
the 2h mark, scratch only if peak MFE never cleared
`mm_scratch_mfe_threshold_r` (default 0.3R). A trade that peaked at
+0.5R and is now flat survives; a trade that never showed any favourable
movement still scratches.

**Persistence (migration 020):** new
`trades.mm_max_favorable_excursion_r` column. Persisted when MFE
increases by ≥0.1R (caps DB writes at ~10/trade) and restored on
engine restart so mid-trade outages don't lose the "already cleared
the bar" fact.

**Behaviour delta:**
- Recovered trades no longer scratched (the user-reported bug).
- Very-wide-SL trades with tiny absolute profits (e.g. 0.5% gain on
  8% risk = 0.062R) now scratch — this is stricter in the right
  direction, matching the course's sense of "substantial".

**Config:** `mm_scratch_mfe_threshold_r`. 0 effectively disables the
rule.

**Tests:** 10 new + 2 existing helpers updated (MFE tracking helpers
needed `original_stop_loss` to be set to use the new rule). Full
suite after all of today's batches merged: 745+ tests passing.

---

## 2026-04-22 — Day 5: Tier 2 learning loop verified live

No code changes today — this is an operational milestone, captured
for the handoff record.

**Verified v44 is deployed and running the Tier 2 rubric:**
- `prompt_version = 'prompt_v=2 rubric_v=2'` on 82 decisions in the
  last 6 hours
- 81 VETOs, 0 APPROVEs, 1 ERROR
- Average VETO confidence: 0.82
- Representative VETO reason: *"Rubric 8: C|sideways profile is
  1W/4L/$-158.93 over 6 samples; no differentiator here to justify
  override."* with concerns `["recent_losses", "low_grade"]`

**Interpretation:** the agent is correctly reading its own 14-day
track record via `get_mm_agent_outcome_stats()` and refusing to take
the same losing pattern again. Before Tier 2 went live, the same
DOGE long setup would have been approved (Grade C + sideways HTF +
multi-session W + 3/4 retest = rubric-compliant) and lost. Tier 2
closes the loop.

**Operational note:** agent cost rose from a projected $6/mo to an
observed $250/mo (at current 82-calls/6h pace). Still under the $600
cap, but worth monitoring. Contributing factor: the 1H formation
detector re-generates the same DOGE long setup every 5-min scan, and
each re-evaluation bills an agent call. A per-setup decision cache
(keyed by symbol + formation_hash, ~30 min TTL) would bring cost back
under control without losing protection. Logged as follow-up in
`docs/ROADMAP.md` §2b.

**No red flags observed.** Pipeline healthy, agent reasoning coherent,
no unexplained approvals or error spikes.

Ref: `docs/STATUS_2026-04-22.md` for full snapshot.

---

## 2026-04-21 — Day 4: Agent learning loop (Tier 1 + Tier 2)

### `<pending>` — feat(mm-agent): outcome-aware rubric + review CLI

Motivation: user observed "the agent keeps approving setups that end up
scratching or losing." Post-hoc analysis of 48h (7 APPROVE, 9 VETO,
1 ERROR) confirmed the pattern — all 7 APPROVEs were on HTF=sideways
Grade C/F setups; 4 of them closed at a net loss, 0 clear winners. The
agent reasoned correctly per the existing rubric, but its rubric had
no feedback from its own track record.

**Tier 1 — scripts/agent_review.py (visibility):**

- One-command forensic report on the last N days of agent activity
- Breakdown by decision × grade × HTF × formation variant
- VETO-reason frequency (most common concerns)
- APPROVED-trade outcomes (win/scratch/loss) + realized P&L
- Profile-level P&L (grade × HTF) — surfaces "which profiles are net-negative"
- `--json` flag for machine-readable output (future: feed weekly cron)

First run revealed the actionable signal: **C|sideways profile is
-$163.61 across 4 trades, 0 wins**. That's the precise anti-edge the
agent wasn't aware of.

**Tier 2 — outcome-aware prompting (self-improvement):**

- New repo helper `get_mm_agent_outcome_stats(days=14)` — aggregates
  APPROVE-decision outcomes grouped by (grade, htf_4h) profile. Match
  logic: decision created_at within ±120s of trade entry_time.
- User prompt now includes a `RECENT PROFILE OUTCOMES` block showing
  each profile's n / wins / losses / scratches / net-pnl, with the
  current setup's profile marked "← THIS PROFILE". Kept in the per-call
  user prompt (NOT the cached system prompt) so each decision sees
  fresh data without invalidating the 1h cache.
- System prompt rubric gets a new point 8 — "Your own track record":
  > If the current setup's profile has n≥5 with net_pnl_usd<0 and
  > losses≥wins, strongly prefer VETO unless confidence≥0.85 AND you
  > can cite a concrete factor THIS setup has that the losing trades
  > lacked. n≥3 with clearly negative net → prefer VETO unless
  > confidence≥0.80+clear differentiator. Positive or no samples → no
  > penalty. When vetoing on this rubric, put `recent_losses` in
  > concerns + cite the specific profile stats in reason.
- `PROMPT_VERSION` bumped 1→2 to prevent silent cache reuse of the
  old rubric on the new user prompt shape.
- New config `mm_sanity_agent_outcome_lookback_days = 14` — set to 0
  to disable the learning loop (reverts to pre-Tier-2 behaviour).

This is the cheapest form of "learning" — no retraining, no new model,
no vector store. Just feed the agent its own win/loss history as part
of every decision it makes. When patterns are systematically losing,
the agent can VETO them. When patterns haven't been sampled, no
penalty.

Tests: 4 new — first-pass placeholder rendering, stats block with
THIS PROFILE marker, version-bump guard, system-prompt rubric-8
presence check. Full suite 693 passing (was 689), 1 skipped.

Scheduled-task integration (Tier 3 of the original design doc's §11):
not yet. The next natural step is a daily cron running
scripts/agent_review.py and posting the output somewhere visible.

Refs:
  - scripts/agent_review.py (new)
  - src/data/repository.py::get_mm_agent_outcome_stats
  - src/strategy/mm_sanity_agent.py::SYSTEM_PROMPT (rubric 8)
  - src/strategy/mm_sanity_agent.py::_build_user_prompt (outcome block)

---

## 2026-04-20 — Day 3: Course-faithful rewrites

### `<pending>` — fix(mm): target-timeframe hierarchy (A + B + C)

User raised: BTC trade 2026-04-20 10:04 UTC entered at $75,251 with TP1
at $92,319 (+22.68%) — "I'd be extremely surprised if Bitcoin gets to
92k." User's own hypothesis: *"it must have taken trades off the 15min
or 1 hour but is using the 4 hour or daily to set the tp?"*. Correct
diagnosis.

**Three compounding issues found:**

**A. Timeframe mismatch.** Formations detected on 1H but EMAs computed
on 4H (`ema_framework.calculate(candles_4h)`), then fed to the target
analyzer. For a 1H-entry long at $75,289, 4H 200 EMA at $71,400 was
below entry — rejected as target, cascade to historical vectors.
1H 200 EMA at $74,372 would have been similarly rejected for this
specific trade, but on most retest setups the 1H EMA is above entry
and a tight target. Fix: EMAs now computed on `candles_1h`. 4H
calculation reserved for `trend_state_4h` (HTF-alignment veto only).

**B. No multi-TF target hierarchy.** `htf_ema_values` was computed
from 4H even though it's supposed to be the HIGHER-TF target per
Lesson 12. For a 1H-entry trade, 4H is only one step up. True "higher
TF" is 1D. Fix: `htf_ema_values` now prefers 1D (when ≥200 bars),
falls back to 4H if 1D insufficient.

**C. Target-less trades.** When no EMA across 1H/4H/1D is above entry
for a long (or below for a short), the cascade lands on a vector that
can be multi-week away. Course doesn't explicitly forbid this but
Lesson 16's "collect profits often" implies close targets. Added an
engineering cap (NOT a course rule — marked clearly as such in code):
`mm_max_tp1_distance_pct = 10.0` rejects setups where TP1 > 10% away
from entry. Configurable; set to 0 to disable.

The live BTC trade that triggered the investigation would have been
rejected under C with the default 10% cap (TP1 was 22.68% away).

Proof the timeframe mismatch was real (BTC snapshot 2026-04-20 10:16 UTC):
```
1H 200 EMA: $74,372  (-1.22%)  ← what should feed targets
4H 200 EMA: $71,400  (-5.17%)  ← what the code was using
1D 200 EMA: $84,053  (+11.64%) ← should feed htf_ema_values
```

8 new tests: TP1-cap constant, config read, engine fallback defaults,
runtime settings override, 1H vs 4H EMA distinctness, 1D-preferred
htf_ema selection, 4H fallback, None case. Full suite 689 passing.

### `eb1f130` — fix(mm): scratch rule + L1 target — match course text verbatim

**Course citations:**

> Lesson 13 [47:00]: *"If you're not in substantial profit within two hours you scratch the trade. It means the Market Maker has a different plan. That's the rule. Market Maker only holds the consolidation level to get more contracts."*

> Lesson 16 [47:00]: *"A Rise or a Drop at level 1 is to break the 50 EMA and head for the 200. Then once it gets to the 200 EMA, one of two things happen. We either retrace and price heads back to the 50 EMA, or we go into a board meeting and the EMA catches up to the price."*

**Before:**
- Scratch rule checked `pos.current_level == 0` at 2h → scratched on level-tracker state, not profit (the actual course trigger).
- `LEVEL_EMA_TARGETS[1] = 50` — engine used the 50 EMA as the L1 TARGET, when the course treats it as the L1 *event* (the thing you break through). Real target is the 200 EMA.
- `LEVEL_EMA_FALLBACKS` dict defined but never referenced (dead code).

**After:**
- Scratch rule: at 2h, check unrealized gross vs round-trip fees. Not in substantial profit → scratch. Ignores level tracker entirely.
- `LEVEL_EMA_TARGETS[1] = 200`. The 50 EMA being near entry no longer causes the engine to silently cascade to unrecovered vectors (multi-week structural highs).
- Dead `LEVEL_EMA_FALLBACKS` removed; vector-based fallback was already wired via the scan path.
- 9 scratch tests + 3 L1-target tests rewritten to cite the course quotes they validate.

**Real-world impact:** the BNB long from 02:01 UTC on 2026-04-20 had TP1 at +15.5% because L1 was quietly falling back to a vector weeks away. Under `eb1f130`, new BNB-class longs get TP1 at the 200 EMA (typically 1-3% away on 1H), matching the course's intraday retest pattern.

---

### `2a04c2e` — fix(mm): dynamic scratch window by SL + board-meeting exemption *(SUPERSEDED by eb1f130)*

Shipped and reverted same day. My invention — not course-grounded.

**Why it was wrong:** scaled the 2-hour scratch window with SL distance and exempted board_meeting phases. Neither rule appears in the course. I misread the problem (BNB scratched at +/- $0 on a wide-SL setup, I assumed time needed to scale, but the real issue was the scratch rule measuring level-tracker instead of profit).

**Lesson:** when the engine produces an unexpected outcome, read the course text before inventing a rule. Replaced by `eb1f130`.

---

### `2695f42` — feat(replay+mm): P&L simulation + ADR tolerance fix (0.3% → 1.0%)

**Course-adjacent:** ADR 50% line confluence (C3 factor) fires when price is "near" the middle of the day's range. The course doesn't specify a tolerance, but the old 0.3% band was 10% of a typical 3% BTC ADR — price passes through it so briefly the factor almost never fires.

**Before:**
- `AT_FIFTY_PCT_TOLERANCE = 0.003` → 0% hit rate across 14 days × 3 majors in replay diagnostic.
- No forward-simulation of signals in replay — we could see "what fires" but not "what it would have earned."

**After:**
- Tolerance 1.0% → 17% hit rate (worth +4 pts on ~1 in 6 scans, directly lifts borderline Grade F toward C).
- `scripts/replay_scan.py --pnl` walks each signal forward through 1H candles, tracks the 30/40/30 TP ladder, breakeven-after-TP1, same-bar pessimism (SL wins ties), 7-day timeout, 0.08% round-trip fees. Aggregates: total R, win rate, avg win/loss R, exit breakdown.

---

### `22b0d32` — feat(replay): multi-symbol, config overrides, factor-rate diagnostics

Tooling only — no engine changes.

- `--symbols BTC,ETH,BNB` batch mode with cross-symbol aggregate
- `--min-confluence N / --max-sl-pct N / --min-rr N` config overrides for A/B testing rule changes before shipping
- `--factor-rates` diagnostic showing per-confluence-factor hit % across a window

Revealed the "why are grades always low?" answer: median confluence 14–16% across majors, several factors permanently at 0% due to replay limitations (OI, correlation, news) or structural rarity (SVC pattern, session-changeover M/W).

---

### `718b858` — feat(scripts): Tier 2 — replay_scan.py historical backtest

Fetches historical 1H/4H/1D/15m from Binance REST, steps through each 1H bar, runs `MMEngine._analyze_pair` as if scanning live at that moment. Agent is disabled (would cost real money on replay). Records which gate each potential setup died at.

Primary value: lets us answer "if we change rule X, how many past signals would shift?" before deploying.

---

### `74a8763` — feat(scripts): trade_audit.py

Forensic one-view CLI for any trade — formation, HTF alignment, prices (with warnings for SL>5%), size, SL-lifecycle flags, agent decision + reasoning, partial exits, outcome.

---

### `e27d176` — fix(mm): SVC wick-return requires clean breakout first

**Course:** Lesson 20 / 23 — SVC "Trapped Traders" zone. *"We always want to see price fail to get back to the wick of a stopping volume candle."*

**Before:** any 1H close inside `[svc_low, svc_high]` post-entry triggered invalidation. But W-retest entries often enter WITH entry price inside the SVC zone by design — so the first near-entry candle scratched the trade.

**After:** require price to first cleanly break above `svc_high × 1.002` (long) or below `svc_low × 0.998` (short), THEN invalidate on a return into the zone. Matches the course's "break then return = trap" reading.

**Real-world impact:** NEAR trade on 2026-04-20 closed +$553 via false SVC invalidation (would have hit TP1 with correct logic). 8 unit tests including the exact NEAR pattern as regression guard.

---

### `19efc43` — fix(mm): stagger TP1/TP2/TP3 when target analyzer collapses

When 50 EMA unavailable AND target analyzer picks the same underlying level for L2 and L3 (common with a single vector in the direction), all three TPs collapsed to one price. No staggered partials could fire.

**Fix:** when L1==L3 detected, synthesize L1/L2 from R-multiples of SL distance (Lesson 47 fallback). L1=2R, L2=3R, identified L3 stays. If L3 too close (<3R), full 2R/3R/5R ladder.

6 unit tests including the exact NEAR 2026-04-20 collapse pattern.

**Note:** superseded in spirit by `eb1f130`'s L1=200 EMA fix, which reduces the frequency of L1 fallbacks. But the collapse guard is still correct when fallbacks DO occur.

---

### `9db81ed` — feat(mm): aggregate-risk budget replaces 3-position human-attention cap

**Course:** Lesson 16 — "1% risk per trade."

The old `mm_max_positions = 3` was the human-attention limit the course implicitly assumed. A bot has no attention constraint but does have a drawdown constraint. Expressed the 1%-per-trade rule at portfolio level.

- New config `mm_max_aggregate_risk_pct = 5.0` (default)
- Raised `MAX_MM_POSITIONS = 6 → 20` (effectively a sanity backstop)
- `_aggregate_open_risk_usd()` — respects SL progression (positions with SL at/past breakeven contribute 0 to risk budget)
- New gate in `_enter_trade` rejects with `mm_reject_aggregate_risk` when projected aggregate > cap

10 unit tests.

---

### `428c20b` — chore(deps): drop 4 dead SMC-era dependencies

Not course-related. `smartmoneyconcepts`, `pandas-ta`, `xgboost`, `scikit-learn` — all in `pyproject.toml` but zero imports anywhere in `src/` or `tests/`. Left over from the SMC-era bot that no longer exists. Production Docker image was already clean (uses `requirements.txt`); this aligns `pyproject.toml` with reality.

---

### `1c6b24f` — fix(mm-agent): switch to adaptive thinking + effort for Opus 4.7

**Anthropic API breaking change:** Opus 4.7 rejects `thinking={"type":"enabled","budget_tokens":N}` with a 400. Must use `thinking={"type":"adaptive"}` + `output_config={"effort":"high"}`.

Hit live on 2026-04-20 00:12 UTC — the first real MM setup of the week (NEAR long) approved via fail-open because every agent call 400'd in 372ms. Trade entered without LLM review. Graceful degradation worked as designed but the agent was silently offline until fixed.

- New config `mm_sanity_agent_effort = "high"` replaces the old `thinking_budget`
- Same API shape works for Sonnet 4.6 (budget-cap fallback)

---

### `433f704` — feat(mm): close 2 known MEDIUM audit gaps + add course audit report

Two audit gaps from the 2026-04-19 course audit:

1. **Multi-session M/W bonus** (+6 pts confluence factor) — Course Lesson 13: *"I very rarely would trade Ms or Ws if they form in the same session... much higher probability multi-session."* Formation detector had tagged `variant=multi_session` for a while but scorer awarded same points regardless.
2. **Per-trade notional cap** (default 50% of account) — Course Lesson 16 says 1% risk; with a tight SL the resulting notional can balloon to 100% of account (BNB 2026-04-17 was $99,943 on $100K). Cap prevents a single position from becoming a blow-up risk.

Also added `docs/COURSE_AUDIT_2026-04-19.md` — background-agent walk through all 22 lessons and report on code gaps.

---

### `91a4eea` — docs(claude): rewrite CLAUDE.md for tarakta-mm reality

Old CLAUDE.md described a three-SMC-agent system on `tarakta-expanded` that hasn't existed since the repo was refactored. Rewritten to describe the actual MM-only bot on `tarakta-mm` (Singapore), with real env vars, dashboard routes, DB tables, and the 12-gate pipeline.

---

### `32970a0` — feat(mm): Agent 4 — Opus 4.7 sanity agent vetoes rubbish setups live

New LLM veto layer. See `docs/MM_SANITY_AGENT_DESIGN.md` for the full spec.

- `src/strategy/mm_sanity_agent.py` — Opus 4.7 + extended thinking + 1h prompt caching
- Hook at `mm_engine.py` after `retest_passed`, before `MMSignal` construction
- Fail-open on API error / timeout / missing SDK — never halts trading
- Budget cap: $600/mo, auto-downgrades to Sonnet 4.6 at 90% projected
- Migration 019: `mm_agent_decisions` table + `mm_agent_*` columns on `trades`
- 22 unit tests

---

### `5c8d2e1` + `19ead57` — docs(mm-agent): design v2 and v3

Iterations on the MM Sanity Agent design doc.
- v2: Opus 4.7 + thinking + 1h caching + async learning loop via Claude Code Routines
- v3: local scheduled tasks (not cloud routines — daily run cap too low), class-based fixtures

---

### `f95c507` — fix(mm): hard-veto counter-4H-trend entries + direction-aware EMA scoring

**The bug that started it all.** BNB-short-into-4H-uptrend disaster on 2026-04-17.

**Course citations:**
> Lesson 12: *"The trend is your friend... EMAs stacked in the direction you're trading."*

> Lessons 03, 10: HTF alignment.

**Before:**
- `_trend_state = self.ema_framework.get_trend_state(candles_4h)  # noqa: F841 — kept for future use`
  → 4H trend state computed, then thrown away. Literal dead code.
- `_score_ema_alignment` awarded full 8 pts for any clean bullish/bearish stack, regardless of trade direction — directly inflating counter-trend setup scores.

**After:**
- 4H trend state wired into a hard-veto gate (rejects counter-trend setups unless variant is a recognised reversal AND trend isn't accelerating).
- EMA alignment scoring takes `trade_direction` from `MMContext`; scores 0 when alignment opposes the trade.
- `htf_trend_4h`, `htf_trend_1d`, `counter_trend` persisted on every trade for post-mortems.
- Migration 018 + `_TRADE_COLUMNS` update.

---

## 2026-04-20 — DB operations (not code)

### Deleted 2 historical trades from ledger
User-requested wipe of pre-fix trades that were polluting current stats:
- BNB SHORT 2026-04-17 13:01 (id `f452c1dc-dcab-41b3-a844-3023efc53dd2`), -$683.76
- APT SHORT 2026-04-17 03:00 (id `5ea61683-a768-4c30-81f1-cafb91f5fac3`), +$218.44

Hard DELETE. Zero related rows in `partial_exits` or `mm_agent_decisions`.

---

## Anti-patterns we've paid for — for future self

### 1. `# noqa: F841 — kept for future use`

Literally how the BNB-short disaster happened. If you compute state and don't use it, delete the line or wire it in. The silencing comment is a ticking bomb.

### 2. Defining lookup dicts that nothing reads

`LEVEL_EMA_FALLBACKS` was course-correct in intent but never referenced in any code path. It fooled multiple reviewers (me included) into thinking the fallback was wired.

**Rule:** before trusting a lookup/config dict, grep to confirm it's actually used.

### 3. Inventing rules not in the course

My `2a04c2e` shipped a "dynamic-by-SL + board-meeting exemption" scratch rule. It was reasonable but nowhere in the course. When the engine produces unexpected behaviour, the first step is to re-read the course text, not invent a fix.

### 4. Measuring the wrong signal

Old scratch rule checked `current_level == 0` when the course says "substantial profit." Two different signals that rarely coincide. A rule is only course-faithful if it measures the thing the course describes.

### 5. Stale documentation

CLAUDE.md described a system that didn't exist anymore. Trusting stale docs cost time. Rewrite docs whenever architecture shifts meaningfully.

---

## Open items / follow-ups

- Correlation provider (`YFinanceCorrelationProvider`) fetches live data only; replay-mode hit rate is artificially 0%. Documented as limitation, not a bug.
- Cooldown (`_cooldowns`) is in-memory only — resets on every deploy. Noticed 2026-04-20 when BNB re-entered 22 min after close (next deploy cycle). **Not yet fixed.** Should persist to DB or query `trades` for recent closes on entry.
- NYC reversal isn't a named detector (works via generic session-changeover classification).
- No trailing-stop warning.
- `mm_agent_decisions.trade_id` foreign key not populated after entry (audit tool timestamp-matches instead).
