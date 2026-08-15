# Verified backtest sweep — 2026-08-15

Multi-agent run: 2× 90d replay generation (6 symbols, 4h step, `--pnl --show-rejects`),
4 hypothesis analysts, 1 fresh-eyes theory agent, 1 live-funding analyst, then one
adversarial verifier per finding (independent recompute + refutation attempt).
**All four findings survived verification.** Artifacts (scripts, CSVs, raw logs) in the
session scratchpad `backtests/` dir; key numbers reproduced below are the verified ones.

Standing caveats apply (CLAUDE.md): replay sees closed bars only + SWING_WINDOW=5 →
lower bound on live behaviour; external feeds score zero retroactively; May–Aug 2026 is
one regime.

## H1 — FMWB weekly-bias gate blocks profitable shorts — **SUPPORTED (verified)**

- 90d × 6 symbols: **359** `against_weekly_bias` rejections (176 short / 183 long).
  Mechanism confirmed in code + data: whenever the weekly first move is DOWN, the gate
  declares real-direction LONG and blocks every short **for the whole week** —
  `detect_fmwb` never checks that the move failed (`broke_box` is computed and ignored,
  no expiry). 100% of blocked shorts had `fmwb_dir=down`.
- Blocked shorts re-simmed under the reversal-test ruler (entry next 1h open,
  median-SL 0.695%, 2R/−1R/168h, both-touched=loss), overlap-suppressed:
  **n=102, +24.0R, 41% WR** — positive at every SL width 0.7–3%, in 3 of 4 months,
  5 of 6 symbols; survives dropping the best symbol (+12R without XRP).
- The 12 signals the pipeline actually took in the same window: **0-for-12, −12R** under
  the identical ruler (−9.74R under real engine management).
- Verifier softening: blocked LONGS at the tightest SL are also mildly positive (+10R),
  so read it as a strong directional asymmetry, not "gate right on longs"; fees ≈0.14R/trade
  would cut blocked shorts to ~+9R (still positive); ETH+XRP carry ~96% of the +24R.

## H2 — post-stop re-entry cooldown — **CONFIRMED (verified, high confidence)**

Using each dataset's own engine-managed R (no re-sim):

| Policy | 180d/38-signal set | 90d fresh set |
|---|---|---|
| baseline | −8.68R | −9.74R |
| 24h cooldown (same sym+dir after loss) | **+0.32R (Δ+9.00)** | −6.79R (Δ+2.95) |
| 48h cooldown | +1.75R | −5.61R |
| 48h one-per-idea dedup | +2.81R (WR 23.7→32.0%) | −5.61R |

- Winners killed at 24h across both sets: 3, worth **+0.63R total** (largest +0.45R).
  Losers avoided: 12 (−12.58R). Every meaningful winner is first-in-cluster and survives.
- 1h cooldown does nothing; 4h too short (Jul-21 BTC re-entries came 4–6h apart).
- **No policy creates positive expectancy** — cooldown removes martingale oversampling
  of losing ideas; it does not add edge. 180d turning positive rests on one +10.2R outlier.

## H3 — trade only WITH the 1D trend — **NOT SUPPORTED (verified)**

- All 12 taken signals sat in a 1D downtrend (EMA20/50). The **with-trend shorts were the
  worst bucket: 0/7, −0.97R avg** — worse per trade than the against-trend longs (−0.59R).
  Dropping against-trend improves the total only by trading less; every bucket loses.
  Flipping against-trend is worse than dropping.
- Read with H1: candidates blocked EARLY at the weekly gate perform fine under a crude
  ruler, while signals surviving the FULL gate stack lose regardless of direction →
  **the late pipeline (retest/entry construction) is anti-selecting entries**. Direction
  filters cannot fix an entry-quality problem. Consistent with MFE evidence (below).

## H4 — funding-rate direction gate — **NOT SUPPORTED (verified)**

- Replay (H4b): gate inert at ±0.01%/8h (fires 1/12; funding pinned near zero May–Aug);
  corr(crowding, R) = +0.003. Do not add.
- Live (H4a, 161 trades): the contrarian version is **inverted** — "funding-aligned"
  trades went 0/11 while "funding-opposed" was the only profitable bucket
  (shorts entered while funding<0: 16/27, 59% WR, +$2,191, p≈0.03; longs while
  funding<0: 0/10). Funding sign is a *regime* proxy (correlated with daily trend),
  candidate for shadow-scoring only, not a gate.

## Fresh-eyes theory ranking (independent agent, live-DB grounded)

1. **Shape-without-location entries have no edge (root cause).** MFE on losers: only
   12/107 ever reached +1R before losing — entries, not exits, are the binding constraint.
2. **Serial re-entry martingale** — 31/161 live trades were <24h post-loss re-entries,
   net −15.9R (matches H2).
3. **Grade-F admission** — 49 F-grade trades = −$8.1k of the −$10.5k lifetime loss;
   grades are monotonic (dead-factor zero-out / floor calibration matters).
4. **FMWB anti-trend lock** (matches H1).
5. Payoff asymmetry (+1.02R avg win vs −1.67R avg loss) — real but secondary.
6. Bot trades almost entirely in HTF chop (124/161 at 4H sideways) — trend gates veto,
   never require.
7. **Missing minimum-SL-distance floor**: two micro-stop trades (SL 0.007%/0.022% from
   entry) filled at −87.7R and −15.4R. Trivial dollar cost in paper, catastrophic under
   real 1%-risk sizing. Free safety fix.

Devil's advocate (recorded): effective independent n ≈ 60–90 ideas (3 instances double-trade
the same signals); committee has *no evidence yet of positive selection* (n≈7 scored, vetoed
both winners) — it is currently a capital-preservation brake; regime luck explains monthly
magnitude, not the sign of expectancy.

## Recommended build order (evidence-ranked)

1. **24h post-stop cooldown** (H2, verified, high confidence). Ship after finding the
   course's own re-entry guidance — cite lesson+timestamp per CLAUDE.md.
2. **Minimum SL-distance floor** (theory #7) — risk-engine sanity bound, no course rule
   invention involved; prevents −80R-class fills.
3. **FMWB gate fix** (H1, verified): only assert `real_direction` when the first move
   actually **broke out of the weekend box and failed** (that is what a trap is), with an
   expiry; otherwise let M/W direction stand (the code's own Lesson-15 comment already
   allows M/W through when no FMWB exists). Re-read C2 L15 first and cite.
4. **Do NOT build**: daily-trend hard filter (H3), funding gates (H4), reversal/flip
   (re-confirmed dead). Funding-sign may be shadow-scored as committee context only.
5. Unchanged strategic conclusion, now better-evidenced: items 1–3 stop the bleeding but
   none create positive expectancy from the current entry stream — the path to edge remains
   the **location-first nominator** (MFE data says entries are the constraint), with grade
   floor recalibration (theory #3 / CODEX_PLAN Task 0.4) as its companion.
