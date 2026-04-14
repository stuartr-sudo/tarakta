# MM Engine — Course-Faithful Filter Redesign

**Goal.** Align `_analyze_pair` with what the raw Trade by Design course
transcripts actually teach (not with my synthesized weight table in
`mm-method-strategy-spec.md`).

**Course evidence used.** Raw transcripts in `docs/tbd-course/*.md`, specifically:

- Lesson 20 — M/W mechanics: "the criteria for an M formation is a stopping
  volume candle… they turn up three times on the inside right side… then the
  3rd time is to break the 50 EMA. Now, **I'm in on my trades before they
  break the 50 EMA**."
- Lesson 43 — 1H charting: "I've got my **Trapped Traders** and my Area of
  Interest. They're the two things that I want to mark."
- Lesson 18 — HOW/LOW three-hits: "after three levels have completed. There
  are also other reversal signals that you could get, that **would replace
  the M or W**. If the Market Maker comes to test a weekly High or Low three
  times…"
- Lesson 29 — Open Interest: "can also be used to **identify trapped Traders**
  … these trapped Traders can often lead to additional…"
- Lesson 53 — Risk: "If the trade doesn't meet your minimum risk to reward,
  you don't take the trade."

---

## Divergences this redesign fixes

### 1. `mm_reject_low_formation_quality` (<0.4) — **remove**

The course never mentions a scored formation-quality number. An M/W either
satisfies its three MM appearances (SVC, 3 inside hits, 50 EMA break) or it's
not an M/W. The `quality_score` is a code-internal ranking used by
`FormationDetector` to pick the best candidate from several — it should
**not** be a hard reject threshold.

**Change:** drop this gate. `quality_score` still ranks formations, but a
weak-quality formation that passes all other course rules is allowed.

### 2. `mm_reject_sl_too_wide` (>5%) — **remove**

The course explicitly says (lesson 53, "Stop Loss Rules"):
> "NEVER tighten SL to improve R:R — SL goes where it needs to go."

Risk is capped via **position sizing** (1% of account / SL distance), not by
refusing trades with wide SLs. The 5% cap is a code invention.

**Change:** drop this gate entirely. The R:R check (gate `low_rr`) plus
position sizing already prevent bad risk exposure.

### 3. `mm_reject_no_l1_target` — **loosen**

The course (lesson 20) says "I'm in on my trades **before** they break the
50 EMA." The 50 EMA break is the **3rd MM appearance / confirmation**, not
an entry requirement — the 50 EMA serves as the L1 *target*, not a gate.

The course's Level 1 target hierarchy (spec section 8, course lesson 47):

1. 50 EMA (primary, counter-trend TP)
2. 200 EMA (secondary, if 50 EMA already broken/exhausted)
3. First unrecovered Vector candle

The current code only looks at priority-1 (50 EMA) and nearby vectors for
L1. If the 50 EMA isn't positioned correctly (e.g., already broken) it
returns `primary_l1 = None` and rejects.

**Change:** if `primary_l1` is missing, fall back to `primary_l2` (which
includes the 200 EMA and farther vectors). Only reject if **no target at
any level** is available. This matches the course's "50 EMA, then 200 EMA,
then vector" hierarchy.

### 4. R:R floor 1.0 → 1.4

Course lesson 53 is explicit: 1.4 is the "don't get out of bed" floor, not 1.0.

**Change:** default `self.min_rr = DONT_GET_OUT_OF_BED_RR` (1.4). Still
runtime-overridable via `/mm/settings` (users who want 3:1 minimum can set
it; 1.4 is the true lower bound per course).

### 5. Open Interest trapped-trader pattern — **score higher**

Course lesson 29 names OI as a trapped-trader detector. The current
confluence scorer has `oi_behavior` at LOW weight (4 pts). Given the course
framing of trapped traders as one of the **two things marked on every
chart** (lesson 43), OI-based trapped-trader evidence deserves more weight.

**Change:** promote `oi_behavior` weight from 4 → 8 (LOW → MEDIUM tier).
Still a scored factor, not a hard gate — courses treats it as confluence,
not a trigger.

Not adding a separate `trapped_traders_zone` scored factor because the
existing `stopping_volume_candle` factor (15 pts HIGH) already captures
"SVC is present" — the Trapped Traders zone per the course *is* the
SVC-marked zone. The weight is already at the top tier.

### 6. Lesson-18 alternative trigger — **add**

Course lesson 18 (direct quote):
> "after three levels have completed. There are also other reversal signals
> that you could get, that **would replace the M or W**. If the Market Maker
> comes to test a weekly High or Low three times, and they don't break it,
> it's likely a reversal is imminent."

The code already detects `three_hits_at_how` / `three_hits_at_low`, but only
uses them as a confluence *boost* (`three_hit_boost`). A valid 3-hit at
HOW/LOW with a completed Level 3 should be usable as a formation
**replacement** — enter anyway, in the reversal direction, even without an
explicit M/W.

**Change:** when `_analyze_pair` finds no formation BUT finds a 3-hit
reversal at HOW/LOW with `current_level >= 3` (on the whole chart, not
post-formation), synthesize a formation-like signal with SL at the
hit-level itself and continue through the rest of the pipeline.

---

## Remaining hard gates (kept as-is — all are course-faithful)

| # | Gate | Course source |
|---|------|---------------|
| 1 | candle_fetch | obvious |
| 2 | insufficient_candles (< 50 1h) | needs enough history |
| 3 | no_formation | M/W required OR lesson-18 alternative |
| 4 | level_too_advanced (>= 3 post-formation) | Lesson 9: "A 4th level almost always brings correction" |
| 5 | fmwb_phase | Lesson 19, 44: don't enter during false move |
| 6 | friday_trap | Lesson 46: "Friday after 5pm NY market close" |
| 7 | wrong_phase | Phase machine enforcement (explicit in spec) |
| 8 | against_weekly_bias | Lesson 44: FMWB direction defines real direction |
| 9 | zero_risk | obvious |
| 10 | low_rr | Lesson 53: hard rule |
| 11 | low_confluence | Lesson 43: "didn't meet my condition" |
| 12 | low_retest | Lesson 13 onwards: 2/4 retest conditions |

---

## Expected effect on the scan funnel

Current typical cycle (90 pairs):

```
no_formation        57
no_l1_target        19
sl_too_wide          6
against_weekly_bias  4
low_rr               2
fmwb_phase           1
low_confluence       1
```

Post-redesign expected shift:

- `sl_too_wide` → 0 (gate removed)
- `low_formation_quality` → 0 (gate removed; was already 0 in practice)
- `no_l1_target` → ~50% fewer (fallback to L2 catches setups where 50 EMA
  is out of position)
- `low_rr` → more (floor raised from 1.0 → 1.4)
- Total signals reaching entry stage: modest increase. Confluence/retest
  still gate quality.

---

## Test plan

| Behavior | Test |
|----------|------|
| Quality 0.2 formation passes (previously rejected) | `test_analyze_pair_accepts_low_quality_formation` |
| 8% SL distance passes (previously rejected) | `test_analyze_pair_accepts_wide_sl` |
| 50 EMA out-of-direction → falls back to 200 EMA L1 | `test_l1_fallback_to_l2_when_ema50_unusable` |
| R:R 1.2 rejected (previously accepted) | `test_rr_below_14_rejected` |
| R:R 1.5 accepted | `test_rr_above_14_accepted` |
| OI weight is 8 pts (was 4) | `test_oi_behavior_weight_is_medium` |
| 3-hit HOW at L3 triggers signal without M/W | `test_three_hits_how_at_l3_triggers_signal` |
| 3-hit HOW at L2 does NOT trigger (lesson 18: must be at L3) | `test_three_hits_how_not_at_l3_no_trigger` |

---

## Backward compatibility

- `mm_engine_settings` from `engine_state.config_overrides` still drives
  runtime overrides — users who set `mm_max_sl_pct` keep that value
  (but it's now advisory; ignored as a hard gate).
- `mm_min_formation_quality` override still stored but ignored.
- Existing funnel events / dashboard card unchanged (the set of reject
  reasons shrinks; no format change).
- `mm_signal` trade records still populated the same way.
