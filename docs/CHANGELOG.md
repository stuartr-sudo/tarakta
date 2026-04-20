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
