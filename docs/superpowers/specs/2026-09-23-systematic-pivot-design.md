# Systematic Pivot — Design Spec

**Date:** 2026-09-23 · **Status:** approved in brainstorming, pending spec review
**Decision owner:** Stuart · **Author:** Claude (Opus 5.5)

## 1. Why

After ~7 months and ~490 commits, the MM engine has never shown an edge.

Live record (Supabase, all instances, 174 closed `mm_method` trades, 2026-04 → 2026-09):

| Metric | Value |
|---|---|
| Net P&L | −$11,518 |
| Win rate | 33.3% |
| Avg win / avg loss | +1.16R / −1.64R |
| Expectancy | −0.71R per trade |
| Stop-loss exits | 71 trades, avg −2.41R (stops 0.02–0.5% from entry; noise + overshoot) |
| Committee, last 60 days | 258 VETO / 20 APPROVE / 71 ERROR |
| September 2026 | 7 trades, 0 wins, −$1.8k |

Every backtest is negative (180d × 5 coins: −8.7R; 90d × 6 coins: 0-for-12). The
reversal test is ~0R in both directions, i.e. the entry signal carries no information.
The March 2026 SMC-era bot "took trades" because gates were loose and risk was
10%/trade across up to 100 positions; its own 7-day backtest lost 4.6%.

Root causes:

1. **Discretionary course → rules translation.** The course's "location" judgement does
   not survive codification (4/43 historical candidates at a course-valid location).
2. **Undersized evidence.** Rules were tuned on 90–180 days, 5–10 coins, ~40 signals —
   far too few to separate edge from noise.
3. **Inverted payoff shape.** Tight stops, 2h scratches and early partials produce small
   wins and full losses.
4. **Unverifiable LLM veto.** The committee cannot be replayed over years without
   lookahead, so its value cannot be measured.

The pivot: stop tuning the MM engine. Build an evidence-first research harness on
multi-year data, run a pre-registered tournament of short-horizon strategies, and
promote only out-of-sample survivors to paper trading.

## 2. Constraints (decided with Stuart)

- **Max holding period: 48 hours.** Every strategy carries a hard 48h time stop.
- **Drawdown tolerance: 10–20%.** Pass bar is max DD ≤ 20% at the chosen sizing;
  default risk 0.5% of equity per trade.
- **Approach: research first.** Nothing new trades (even on paper) until it passes the
  holdout.
- MM course is guidance only — its ideas enter as individually testable hypotheses (S4),
  not as a gate stack.
- Paper mode only. Any move to real money is Stuart's decision, made after the paper
  ladder in §6.

## 3. Architecture

```
data.binance.vision ──► research/data.py ──► research/cache/*.parquet (gitignored)
                                                   │
                         src/systematic/strategies/*.py   (pure functions, shared)
                               │                       │
                     research/backtest.py      src/systematic/engine.py
                     research/run.py                   │
                               │               PaperExchange + Repository
                     research/reports/*.md             │
                     research/trials.csv        trades (strategy='sys_<name>',
                                                        instance_id='sys-paper')
```

- `research/` is a new top-level package. It imports from `src/systematic/` and never
  from the MM engine (`src/strategy/mm_*`).
- `src/systematic/strategies/` holds each strategy as a pure function: closed candles
  (+ funding) in, target orders out. The backtester and the live engine import the same
  function, so live and backtest logic cannot drift.
- Tests live in `tests/research/` and `tests/systematic/`.

## 4. Data layer (`research/data.py`)

- **Source:** monthly zip archives from `data.binance.vision`:
  - USDT-M futures 1h klines, 2020-01 → 2026-08. 4h and 1d are resampled from 1h.
  - 8h funding-rate history for the same symbols.
- **Cache:** parquet per symbol under `research/cache/` (add to `.gitignore`).
  Re-downloads are idempotent; archive checksums verified where Binance publishes them.
- **Universe:** re-ranked monthly as the top 30 USDT-M perps by trailing 30-day quote
  volume, computed only from data available at that time. Delisted symbols are
  included, so results do not carry survivorship bias.
- **Integrity checks:** no duplicate timestamps, monotonic index, gaps reported (not
  silently filled), no symbol enters the universe before it has 30 days of history.

## 5. Backtester (`research/backtest.py`)

- Portfolio-level, bar-by-bar simulation on 1h bars. Plain loops for correctness before
  speed (≈1.7M symbol-bars).
- **Timing:** signals computed on the close of bar *t*; entries fill at the open of
  bar *t+1*.
- **Costs:**
  - Taker fee 0.05% per side.
  - Slippage 0.02–0.08% per side, scaled by liquidity tier (trailing volume rank).
  - Funding paid/received every 8h on open notional, using the historical rate; longs
    pay positive funding, shorts receive it.
- **Fills:**
  - Stop fills at the stop price plus slippage; if a bar opens through the stop, the
    fill is at that open.
  - If one bar touches both stop and target, the stop is assumed hit first.
  - Hard 48h time stop, exits at the open of the first bar after 48h.
- **Sizing and caps:** risk 0.5% of equity per trade from stop distance; gross exposure
  ≤ 3× equity; a configurable max number of concurrent positions (default 10).
- **Outputs per run:** trade list, equity curve, and the metrics used by the pass bar.

### Validation protocol (fixed before any results are seen)

- **Development period:** 2020-01 → 2025-08, evaluated walk-forward.
- **Holdout:** 2025-09 → 2026-08. Locked; each finalist is run on it exactly once.
- **Pass bar — all must hold:**
  - Holdout net expectancy > 0 after costs, t-stat ≥ 2.
  - ≥ 300 trades in development, ≥ 100 in holdout.
  - Profit factor ≥ 1.2.
  - Max drawdown ≤ 20%.
  - Profitable on ≥ 60% of symbols traded and in ≥ ⅔ of calendar years.
  - Still profitable with every parameter moved ±25%.
- **Multiple-testing control:** every variant run is appended to `research/trials.csv`
  (strategy, params, period, metrics). The pass bar is haircut for the trial count
  (deflated Sharpe ratio).
- **Trial budget:** ~50 variants total across all strategies. Hitting the budget means
  stop and report, not keep tuning.

## 6. Candidate strategies (`src/systematic/strategies/`)

All trade long and short, all carry the 48h time stop, and each declares a parameter
grid of ≤ ~12 variants up front. "ATR" means ATR(14) on the strategy's signal
timeframe (4H for S1, 1D for S2, 1H for S3 and S4).

**S1 — Volatility breakout (trend).**
Entry on a 4H close beyond the prior N-bar high/low (N ∈ {20, 55}) with ATR expanding.
Stop 1.5–2.5 × ATR. Exit on a trailing ATR stop or the time stop. Variant: only in the
direction of the 1D EMA trend. Rationale: momentum is the best-documented crypto
effect; "lose small, win big" profile.

**S2 — Daily cross-sectional rotation.**
At 00:00 UTC rank the universe by volatility-adjusted trailing return (1, 3 or 7 days).
Long top 3–5, short bottom 3–5, hold 24h or 48h, catastrophe stop 3 × ATR. Tested in
both momentum and reversal form. Rationale: near market-neutral, so it can earn in
chop.

**S3 — Funding-extreme fade.**
When a symbol's funding rate is in the top/bottom 5% of its own trailing 90 days, trade
against the crowd (and collect the funding). ATR stop or time stop. Rationale: crowded
leverage unwinds; live data showed shorts entered while funding < 0 at 59% WR.

**S4 — Course sweep, tested systematically.**
A 1H bar wicks through the prior day's or week's high/low by ≥ x ATR (x ∈ {0.1, 0.25})
and closes back
inside; trade the reversal. Stop beyond the wick plus buffer; target the day/week
range midpoint. Variants with/without the course's London/NY session-open timing.
Rationale: the course's core stop-hunt idea, tested on thousands of events instead of
~40.

**Portfolio step:** if ≥ 2 strategies pass, test an equal-risk combination.

## 7. Promotion to paper (only for survivors)

- **Engine:** `SystematicEngine` in `src/systematic/engine.py`, selected by a new
  `ENGINE=systematic` config value in `main.py` (same pattern as the existing
  `MMEngine` / `InverseMirrorEngine` switch).
  - Runs on each 1H bar close and calls the shared strategy function on the same
    closed bars the backtest uses.
  - Orders go through the existing `PaperExchange`; the engine enforces the stop, the
    48h time stop and the portfolio caps; the existing websocket fast-stop loop is
    reused.
  - Trades are written to the existing `trades` table with
    `strategy='sys_<name>'`, using existing columns only (no migration). Every
    in-memory position change is followed by `repo.update_trade(...)`.
- **Isolation:** own process and own `instance_id='sys-paper'`, run locally under
  launchd first. MM bots are untouched.
- **Live-vs-backtest tracking:** weekly, re-run the backtest over the same live weeks
  and compare with paper fills. Live must sit inside the backtest's normal range;
  meaningful underperformance halts the ladder until explained.
- **Ladder:**
  1. Paper for ≥ 6 weeks and ≥ 50 trades, inside the backtest range.
  2. Then Stuart decides on real money, starting small.

## 8. Milestones and checkpoints

1. **Data** — downloader + cache + universe. Tests: no gaps/dupes, monotonic
   timestamps, point-in-time universe never uses future volume.
2. **Backtester** — accounting tests on synthetic candles with known answers (fees,
   slippage, funding sign long vs short, gap-through-stop, stop-first on both-touch,
   48h exit, sizing/leverage caps). No-lookahead test: scrambling bars after *t* must
   not change signals at *t*.
3. **Strategies S1–S4** — unit tests on hand-built candles, development-period runs,
   trial log.
   → **Checkpoint 1 with Stuart:** development results; choose holdout finalists.
4. **Holdout** — one run per finalist; report published as a private artifact page.
   → **Checkpoint 2 with Stuart:** go/no-go for paper; decide the MM bots' future.
5. **Paper engine** (only if something passed) — `SystematicEngine` + fake-exchange
   tests, launched on `sys-paper`.

## 9. What does not change

- MM bots (`main` local, `tarakta-fly` on Fly) keep running unchanged during research.
  Cost of leaving them: Fly committee API calls, ~$30 per 60 days.
- No DB migrations, no changes to `mm_engine.py`, the committee, or the dashboard.

## 10. Failure path

If nothing passes the holdout, the deliverable is a report stating that and why. The
harness stays as the tool for testing any future idea, and no more tuning happens
against the holdout.

## 11. Out of scope

- Real-money trading.
- Replacing or refactoring the MM engine.
- LLM involvement in trade decisions (revisit only once its effect is measurable).
- New paid data sources.
