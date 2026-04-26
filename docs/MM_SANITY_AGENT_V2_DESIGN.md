# MM Sanity Agent v2 — Design Document

**Author:** Engine v2 architecture
**Status:** DRAFT for review
**Goal:** Rebuild the sanity agent from rubric-based veto to reasoning-based judgment with variable sizing.

---

## Why v2 exists

V1 (rubric_v=3) had three structural problems:

1. **Rubric-based, not reasoning-based.** Eight numbered checklist items. The agent ran them as Boolean tests, not as a holistic judgment. When all rubrics passed, APPROVE. When any failed, VETO. Real discretionary traders don't think this way.
2. **Binary output.** APPROVE or VETO. No notion of "yes but reduce size" or "wait one bar." Real traders use sizing as a degree of confidence.
3. **No course knowledge.** The prompt referenced lessons by name but didn't embed the actual course examples. The agent had to reason about MMM without ever having "seen" a real Annii trade.

The 97% veto rate that emerged on the broken stream wasn't a bug — it was the rubric working as designed against bad input data. With the underlying architecture now fixed (HTF actually computes, friday_trap auto-resets, gates working), v1's rubric is now the limiter.

---

## v2 architectural changes

### 1. Decision space: 5 outcomes, not 2

```
APPROVE_FULL_SIZE   1.00× standard risk (1% of balance)
APPROVE_HALF        0.50× risk
APPROVE_QUARTER     0.25× risk
PASS                no trade
DEFER               re-evaluate next bar (price/wick may improve)
```

Sizing as confidence. Quarter-size lets the bot tag speculative-but-tradeable setups without blowing capital. Half-size is the everyday "good but with one concern" path. Full size is reserved for textbook setups.

### 2. Reasoning structure, not checklist

V1 prompt: "Reason through these 8 rubrics in order."

V2 prompt: "Identify the closest course example. Compare this setup to it. State the strongest evidence FOR taking it. State the strongest evidence AGAINST. Choose size based on the gap."

This mirrors how Annii talks in the lessons — pattern recognition first, then weighing.

### 3. In-prompt course knowledge

The 76 trade examples we extracted from the course transcripts go directly in the system prompt as labeled positive examples. The agent can compare new setups to actual course-validated cases:

> *"This setup looks like Annii's W from Lesson 7 [29:00] — multi-session, peak2 at LOW, hammer confirmation. Her verdict: take_it. Strong analog → APPROVE_FULL."*

This is concrete pattern matching against ground truth, not abstract rule application.

### 4. Per-factor education in the prompt

For each of the 12 confluence factors, embed:
- What it measures (one sentence)
- Why it matters in MMM (one sentence)
- When to weight heavily (specific conditions)
- When to ignore (specific conditions)

Example:
```
liquidation_cluster: True if a cluster of recent liquidations sits within
0.5% of price. WEIGHT HEAVILY when at LOD/LOW/HOD/HOW — that's where
MMs hunt liquidity by design (Lesson 02). IGNORE mid-range; clusters
form constantly without strategic significance.
```

The agent isn't told to "score 8 points for liquidation_cluster" — it's told what the signal *means* and judges whether the signal is meaningful in this context.

---

## The new system prompt (DRAFT)

```
You are an MM Method discretionary reviewer for an automated trading bot.
The bot has detected a setup that passed all deterministic gates: formation
confirmed, HTF aligned, course-faithful variant, hammer at peak2, at LOD/LOW
or HOD/HOW. Your job is to judge whether to TAKE it, what SIZE to take, or
PASS — using the same pattern recognition Annii teaches in the MMM
masterclasses.

You are not a checklist. You are reasoning about whether this specific setup
matches setups that have historically worked.

═══════════════════════════════════════════════════════════════════════════
COURSE FOUNDATION (the framework you reason within):

The Market Maker Method assumes that price is engineered to:
1. Trap retail traders into the wrong direction at extremes (LOD/HOD)
2. Sweep liquidity at obvious technical levels
3. Reverse only when the MM's accumulation/distribution is complete

Trade-able setups all share the same DNA: they form at a place where the
MM is "done" doing what they're doing — typically LOD/LOW for a long
reversal, HOD/HOW for a short reversal — and price has just printed
evidence of rejection (a hammer, an aggressive wick pull-away, a
multi-session structure).

You are looking for setups that LOOK LIKE the course examples below.
You are SKEPTICAL of setups that look superficially right but lack the
pull-away velocity, the hammer confirmation, or the structural placement.

═══════════════════════════════════════════════════════════════════════════
COURSE TRADE EXAMPLES (positive cases — Annii took or said she would take):

[76 examples loaded from review-package/data/course_trade_examples.csv]

Format:
  - Lesson 7 [29:00]: BTC W, multi-session, peak2 at LOW, hammer present.
    Annii: "took_it" — said the W was textbook because second wick was
    hammer-shaped and the move out was aggressive.
  - Lesson 6 [...]: ...

Use these as reference. When evaluating a new setup, ask: which of these
does it most resemble? How confidently?

═══════════════════════════════════════════════════════════════════════════
PER-FACTOR CONTEXT (what each value means and how to weight):

[Detailed per-factor explanation for all 12 factors]

═══════════════════════════════════════════════════════════════════════════
DECISION OUTPUTS:

  APPROVE_FULL_SIZE   — textbook setup, looks like one of the course
                        examples; full 1% risk
  APPROVE_HALF        — good setup with one concern; 0.5% risk
  APPROVE_QUARTER     — speculative; tag a position but small; 0.25% risk
  PASS                — doesn't meet edge bar; no trade
  DEFER               — wait one bar; price action may resolve

═══════════════════════════════════════════════════════════════════════════
REASONING TEMPLATE:

For each setup:

1. CLOSEST COURSE EXAMPLE: Which of the embedded examples does this most
   resemble? Cite lesson + timestamp.

2. SIMILARITIES: What's the same? (formation type, level placement,
   session timing, factor combination)

3. DIFFERENCES: What's different? (retest count, HTF state, market
   regime, time of week)

4. STRONGEST EVIDENCE FOR: One sentence — the most compelling reason
   to take this trade.

5. STRONGEST EVIDENCE AGAINST: One sentence — the strongest reason to
   pass or reduce size.

6. SIZE DECISION: Based on the balance of 4 vs 5, choose size:
   - Strong FOR, weak AGAINST: APPROVE_FULL
   - Strong FOR, moderate AGAINST: APPROVE_HALF
   - Mixed FOR/AGAINST: APPROVE_QUARTER or DEFER
   - Weak FOR or strong AGAINST: PASS

═══════════════════════════════════════════════════════════════════════════
OUTPUT SCHEMA (JSON):

{
  "decision":           "APPROVE_FULL_SIZE" | "APPROVE_HALF" | "APPROVE_QUARTER" | "PASS" | "DEFER",
  "closest_example":    "Lesson NN [HH:MM]" or null,
  "evidence_for":       "<=30 words",
  "evidence_against":   "<=30 words",
  "size_rationale":     "<=20 words explaining the size choice",
  "confidence":         0.0-1.0,
  "concerns":           [<=4 short tags]
}

NO prose outside the JSON. Use extended thinking to walk through the
reasoning template before committing to the JSON output.
```

---

## Code changes required to deploy v2

### `src/strategy/mm_sanity_agent.py`

1. Replace `SYSTEM_PROMPT` with the v2 version above
2. Update `AgentVerdict` dataclass:
   - Add `decision: Literal["APPROVE_FULL_SIZE", "APPROVE_HALF", "APPROVE_QUARTER", "PASS", "DEFER"]`
   - Add `closest_example: str | None`
   - Add `evidence_for: str`
   - Add `evidence_against: str`
   - Add `size_rationale: str`
   - Map old `APPROVE/VETO` for backwards compat (APPROVE_FULL/HALF/QUARTER → APPROVE; PASS/DEFER → VETO)
3. Update `_build_user_prompt` to include rich per-factor context
4. Add a `course_examples_corpus` loader that reads the CSV and embeds inline

### `src/strategy/mm_engine.py`

5. After agent call, read `decision` and apply sizing multiplier:
   - `APPROVE_FULL_SIZE` → trade_risk_pct = self.risk_pct (1.0×)
   - `APPROVE_HALF` → trade_risk_pct = self.risk_pct × 0.5
   - `APPROVE_QUARTER` → trade_risk_pct = self.risk_pct × 0.25
   - `PASS` → reject (existing veto path)
   - `DEFER` → reject this scan, allow re-scan next bar

### `src/data/repository.py`

6. Add `mm_agent_size_multiplier` column to `mm_agent_decisions` (migration 021)
7. Update `_MM_AGENT_DECISION_COLUMNS` allowlist

### Tests

8. New test cases for each decision type
9. Integration test that exercises the sizing multiplier end-to-end
10. The existing `BNB_2026_04_17_canary` test — verify it still vetoes (PASS) under v2

---

## Cost projection

V1 was vetoing 97% of inputs at $0.05 average per call. V2 with extended thinking will be slightly more expensive per call (~$0.07-0.10) but should be called less because the deterministic pre-filtering (3-of-5 gates) is now stronger.

Expected v2 call rate post-deploy:
- 5-15 candidates/day (only Grade C+ setups passing gates)
- $0.08 × 10/day = $0.80/day = **$24/month**

Well within budget cap of $600/month.

---

## Rollout plan

1. **Phase 1 — Backtest harness** (1 day)
   Build a way to replay historical signals through the v2 agent without
   spending API budget on live calls. Use cached responses or stub model.

2. **Phase 2 — Shadow mode** (3-5 days)
   Deploy v2 alongside (disabled in production decision-making) but
   logging what it WOULD have decided. Compare to no-agent live run.

3. **Phase 3 — A/B in production** (1-2 weeks)
   Enable v2 for half of detected setups; existing no-agent path for the
   other half. Compare outcomes.

4. **Phase 4 — Migrate** (1 day)
   If v2 wins, route all setups through it. If not, retire the design.

---

## Open questions (need answers before building)

1. **Course examples corpus** — should we use all 76 we extracted, or
   curate down to ~20 highest-quality?
2. **Per-factor context** — write all 12 ourselves or extract from
   transcripts using a separate agent?
3. **Decision audit** — does the agent need to log its reasoning chain
   for forensic purposes, or just the final JSON?
4. **Sizing confidence floor** — if confidence < 0.5, force APPROVE_QUARTER
   regardless of decision? Or trust the model fully?
5. **Bracket integration** — should v2 also output preferred entry offset
   (+0.5% / +1.0% / +1.5%) for limit-order placement, anticipating future
   predictive architecture?

---

## Status

This is a DESIGN doc. No code has been written yet. Awaiting review on:
- Decision space (5 outputs the right granularity?)
- In-prompt course corpus (full 76 vs curated)
- Sizing approach (multiplier vs fixed tiers)
- Rollout sequence
