# Tarakta MM — Roadmap

**Purpose:** the living plan. What's next, in what order, and why.

Read alongside:
- [`docs/STATUS_YYYY-MM-DD.md`](.) — current state snapshot
- [`docs/CHANGELOG.md`](./CHANGELOG.md) — what's already shipped
- [`docs/MM_SANITY_AGENT_DESIGN.md`](./MM_SANITY_AGENT_DESIGN.md) — agent architecture

---

## ✅ Done

### 1. Deploy the 3 pending engine commits — DEPLOYED 2026-04-21 as v44

Verified 2026-04-22: `prompt_version='prompt_v=2 rubric_v=2'` on all recent rows. 82 VETOs in 6h, all citing Rubric 8 with correct profile stats. Tier 2 learning loop confirmed working.

### P1 — Entry slippage gate from peak2 wick — SHIPPED 2026-04-23 (`01038af`)

Symptoms of "SL too wide" rejects were actually entries drifting >1% above the course's retest level (2nd-peak wick per Lesson 20/47). New `mm_max_entry_slippage_pct` config rejects those setups explicitly with `entry_slipped_from_retest` reason. 10 new tests covering BTC/NEAR/BNB regression cases.

### P2 (= §2b below) — Per-setup decision cache — SHIPPED 2026-04-23 (`a2c00d1`)

In-memory cache keyed by (symbol, direction, formation_variant, round(entry_price,4)) with 30 min TTL + 0.5% price-drift invalidation. 14 new tests. Expected 85% call reduction on duplicate-setup scan storms.

### P3 — MFE-based 2h scratch rule — SHIPPED 2026-04-23 (`40a0860`)

Course Lesson 13 "substantial profit WITHIN two hours" is a window, not an instant. Track Max Favorable Excursion in R-multiples; scratch at 2h only if peak MFE never cleared 0.3R. Stops closing trades that recovered from an early dip. Migration 020 persists MFE so restarts don't lose the bar-cleared fact. 10 new tests.

---

## Now (operational + small, high-value)

### 2. Tier 3 — daily cron running agent_review.py

Already spec'd in [`MM_SANITY_AGENT_DESIGN.md §11`](./MM_SANITY_AGENT_DESIGN.md#11-async-learning-loop).

Goal: you wake up, there's a report in Slack / email / the dashboard saying "agent approved N trades, vetoed M, net P&L $X, biggest concern pattern Y."

**Options for scheduler:**
- (a) Desktop scheduled task on your local Mac via the `schedule` MCP — simplest, no cloud deps, breaks when laptop off
- (b) Fly Machine scheduled task — runs alongside the bot, always on, but needs a bit more Docker plumbing
- (c) Cloud Routine — future-proof but 5-25/day cap (overkill for 1 run/day)

**Recommendation:** (b). Same host as the bot, always on. Roughly 30 min of wiring.

### 2b. Per-setup decision cache — SHIPPED 2026-04-23 as P2 (see Done section above).

---

### 3. Backfill `mm_agent_decisions.trade_id`

The FK column exists but isn't populated. Currently `scripts/agent_review.py` and `scripts/trade_audit.py` join by timestamp proximity (±120s). Works but fragile.

Fix: right after `insert_trade()` returns in `mm_engine._enter_trade`, update the preceding agent-decision row with the new trade ID.

**Size:** 20 lines + a small test. Low risk.

### 4. Named NYC reversal detector

Course Lesson 10 describes NYC Reversal as a specific named setup — "if you can be awake for the first three hours into US, you're looking for an NYC reversal trade." Currently our engine just treats it as a generic session-changeover M/W. A named detector would:
- Specifically scan for reversal patterns at 13:30–16:30 UTC
- Tag the trade with `formation_variant = "nyc_reversal"`
- Enable agent + confluence scoring to recognise it as premium

**Size:** new `mm_nyc_reversal.py` (~150 lines), hook into `_analyze_pair`, tests.

---

## Near-term (this week)

### 5. Web UI for replay + agent review

New dashboard page `/mm-agent` that renders:
- Recent `mm_agent_decisions` with concerns tags
- Profile P&L (grade × HTF) — the Tier 1 table, served from DB
- Current rubric version + approve/veto ratio over last 7/14/30 days
- A simple form to kick off a replay (`--symbol X --days N`) and show results inline

**Size:** ~3–4 hours. Nice-to-have; CLI already works.

### 6. Scenario fixtures (Tier 3 from the original testing plan)

Growing library of `tests/fixtures/scenarios/*.json` — each fixture is a hand-crafted setup (BNB-counter-trend, NEAR-SVC, BTC-no-nearby-target, etc.) with assertions about expected engine behaviour. Protects against regressions on known-bad patterns.

**Size:** ~1 hour for initial scaffolding + 1 fixture per bug we've fixed. Grows over time.

### 7. Rubric iteration based on Tier 1 data

After 2 weeks of live running with Tier 2, re-run `scripts/agent_review.py --days 14` and look for:
- Profiles where the agent consistently approved losers → tighten rubric
- Profiles where the agent over-vetoed winners → loosen rubric (or identify the specific differentiator)
- Reasons that appear with low confidence but negative P&L → prompt clarity needed

Every rubric change bumps `PROMPT_VERSION` + cites exact course text + includes a test.

---

## Medium-term (next 1–2 weeks)

### 8. Trailing-stop warning

Course Lesson 16 explicitly forbids trailing stops — *"A trailing stop isn't your friend, it's your enemy."* Our engine doesn't set one, but if the operator manually adds one on the exchange UI, we should detect + warn. Requires polling exchange position details, not just local state.

### 9. EMA direction check on BBWP / LINDA

Flagged in the 2026-04-19 audit as MEDIUM-severity. Unverified whether BBWP and LINDA confluence factors respect trade direction. Same class as the pre-`f95c507` EMA alignment bug.

### 10. Live correlation provider fixes

yfinance `download(period="1d")` fetches current day's data. Live mode works during UK/NY session when DXY/NASDAQ are trading, but goes dormant during Asia hours and weekends. Add awareness of when the provider is "genuinely zero vs. stale data."

---

## Longer-term (uncertain)

### 11. Replay P&L → agent training pair dataset

Each replay signal + its simulated P&L outcome is a labelled training example. After accumulating 500+ we could:
- Fine-tune Sonnet 4.6 on the best-reasoned VETOs from production + what-would-have-been-wins from replay
- Or: train a small classifier (e.g. gradient-boosted) purely on features vs. outcome, use it as a second opinion alongside Opus 4.7

Not worth touching until volume supports it.

### 12. Multi-instance / strategy A/B

Run two instances of the engine with different rubric versions against the same candle stream, compare P&L after N weeks. Requires `instance_id` isolation (already in place) and a routing layer for trades to distinct paper accounts.

### 13. Live-mode activation

Currently `TRADING_MODE=paper`. Going live requires:
- Real Binance API keys with trading permissions
- Smaller starting balance (paper is $100K; real should probably start at $5–10K)
- One-week observation period with live-but-tiny sizes before scaling
- Clearly documented rollback plan
- Review of all SL placement logic under live slippage

Not before Tier 3 learning loop has shown 2+ weeks of agent self-calibration.

---

## Things we've DECIDED NOT to do (for now)

### ❌ Tier 4 RL-style agent fine-tuning
Not worth touching until 500+ labelled decisions exist. Currently 17. Revisit in ~2 months.

### ❌ Cloud Routines for the daily cron
Daily cap (5-25/day) is overkill for 1 run/day. Desktop or Fly Machine scheduled tasks are simpler.

### ❌ Lowering Grade C threshold from 40% → 35%
Considered on 2026-04-20. Decided against because:
- The ADR tolerance fix + Tier 2 learning loop address the underlying cause (grades being artificially low AND the agent blind to that fact)
- Changing a course-threshold to mask a scoring gap would have been working around the symptom

### ❌ Global cooldown between any two trades
Considered when user observed BNB closing immediately followed by NEAR opening. Decided against — per-symbol cooldown exists (4h); global cooldown isn't in the course; no signal-density problem observed live.

### ❌ Hardcoded max SL distance as a hard reject
Course doesn't give a numeric bound on SL distance. The `mm_wide_sl_warning` log is informational; the agent can factor it in qualitatively. Revisit only if data shows consistent losses on very-wide-SL trades.

---

## How decisions are made in this repo

Every rule change follows this flow:

1. **Observe** something in live trades (or via replay)
2. **Read the relevant course lesson** BEFORE proposing a fix. Cite lesson + timestamp.
3. **Test the hypothesis** with `scripts/replay_scan.py --min-confluence N` or a similar override
4. **Write a test** that would have failed with the old code and passes with the new
5. **Ship with a commit message that includes the verbatim course quote**
6. **Update `docs/CHANGELOG.md`** with the commit + citation

Anti-pattern: my own `2a04c2e` — I invented a "dynamic scratch by SL" rule without citing the course. Reverted same day by `eb1f130`. Don't do that.

---

## When you pick this up, start here

1. [`docs/STATUS_YYYY-MM-DD.md`](.) — what's running today
2. [`docs/CHANGELOG.md`](./CHANGELOG.md) — what already shipped (appendable)
3. This file — what's next
4. [`CLAUDE.md`](../CLAUDE.md) — project guide, required reading before touching engine/agent code

Before changing any rule, re-read the relevant course lesson. No invention.
