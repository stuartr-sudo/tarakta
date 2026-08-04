# LAUNCH HANDOFF — read this first

**Last updated:** 2026-08-03 by Claude (Fable 5) · **Repo:** https://github.com/stuartr-sudo/tarakta · **Local:** `/Users/stuarta/tarakta`
**Branch:** `main` == `codex/tarakta-stabilization` (pushed, clean) · **Tests:** 797 passed / 1 skipped

> **Session log 2026-07-31→08-03 (this account is out of usage; a second
> Claude account continues on the SAME laptop/folder — memory dir is shared):**
> (1) Committee superseded to the **Claude 5 family** and the escalation model
> — previously configured but never consumed — was actually wired (`4dafbd4`,
> deployed to Fly `tarakta-mm2` AND local, both verified healthy after).
> (2) **180d × 5-coin deterministic backtest ran**: −$8,671 / −8.7R / 24% WR
> over 38 signals. (3) Stuart asked "what if we reversed every trade?" —
> tested empirically; raw reversal looks great (50% WR) but it's ENTIRELY a
> serial-re-entry artifact; per independent idea both directions are ~0R.
> **Read `docs/BACKTEST_180D_2026-08-03.md` — it reranks the failure causes**
> (re-entry martingale ≈ 8R of the damage) and is the basis for the new build
> queue item 1b below. No engine rule changes were made.

This file is the single entry point for a new agent or human session. It states
what is running, what is broken, what is blocked on the user, and what to do
next — with commands. Deeper detail: `docs/STATUS_2026-07-19.md` (full
narrative), `docs/CHANGELOG.md` (top 3 entries), then `CLAUDE.md` (rules).

---

## 1. One-paragraph state of the world

Tarakta MM is a paper-trading crypto futures bot implementing the TTC "Market
Maker Method". **Two instances exist.** The **local Mac bot** (instance `main`)
is the live experiment: it runs current code and a 6-agent LLM committee in
**shadow mode** on Stuart's Claude Code subscription (no API key). The **Fly
production app** (`tarakta-mm`) still runs the **2026-04-28 build**, has no
committee, and has been losing ~$500/day of paper money on junk-grade setups.
The strategy is **not yet profitable**: the deterministic engine nominates
setups that are not where the course says to look (evidence: only 4 of 43
historical candidates were at a course-valid location). The next real build is
the **location-first nominator**. Nothing here risks real money — everything is
paper mode.

## 2. What is running right now

| Thing | State | Control |
|---|---|---|
| **Local bot** (instance `main`) | ✅ Running, healthy, scanning 10 pairs / 5 min (verified 2026-07-31; the in-log cycle counter resets to 0 whenever launchd restarts the process — a low cycle number is NOT a problem) | launchd `com.tarakta.bot`; restart: `launchctl kickstart -k gui/501/com.tarakta.bot`; stop: `launchctl unload ~/Library/LaunchAgents/com.tarakta.bot.plist` (a plain `kill` just gets KeepAlive-restarted). Logs: `/tmp/tarakta-bot.launchd.out.log` |
| **Committee** (5 Sonnet-5 specialists + Opus-5 head trader; Fable 5 on contested escalation — Claude 5 supersede 2026-07-31) | ✅ Live in **shadow** (logs verdicts, blocks nothing) | `.env`: `MM_COMMITTEE_ENABLED=true`, `MM_COMMITTEE_MODE=shadow`. Auth = `CLAUDE_CODE_OAUTH_TOKEN` in `.env` (1-year token from `claude setup-token`). **Never print that value.** Fable-5-on-API needs org 30-day retention (CLI transport exempt). |
| **Fly NEW** `tarakta-mm2` (new account, instance `tarakta-fly`) | ✅ LIVE 2026-07-31: current main, region `sin` (Binance access verified from first scan cycle), 1 machine (HA second machine deliberately destroyed — two machines on one instance_id double-trade), scanning ACTIVE on a fresh $100k paper book, committee SHADOW | `fly deploy --app tarakta-mm2 --config fly.toml --depot=false --remote-only`. Committee needs `ANTHROPIC_API_KEY` secret (user one-liner in §3) — until set, decisions log ERROR client_unavailable (expected, shadow-safe) |
| **Fly OLD** `tarakta-mm` (dead account) | ⚠️ April build, still trading junk; Fly account unreachable | Only kill switch: its own dashboard Stop button https://tarakta-mm.fly.dev/mm (app login). User to press it. |
| **Inverse mirror** `tarakta-mm-inverse` | Idle; lost −$1.1k; recommended for retirement | Not yet disabled |

## 3. Fly status (UPDATED 2026-07-31 evening: DEPLOYED — two user actions left)

`tarakta-mm2` is live on the new account (see §2). Remaining user actions:

1. ~~**Anthropic key**~~ **DONE 2026-08-04**: `ANTHROPIC_API_KEY` is set as a
   Fly secret on `tarakta-mm2` (via clipboard→pbpaste, never printed) and
   mirrored in gitignored `env.local`. Machine restarted healthy; committee
   on Fly now has an SDK client (Claude 5 models, ~$0.10/run, ~$5–8/mo at
   current cadence, $600/mo cap in config). Watch `mm_agent_decisions` for
   the first APPROVE/VETO from instance `tarakta-fly` — ERROR
   `client_unavailable` should no longer appear.
2. **Press Stop on the OLD app**: https://tarakta-mm.fly.dev/mm (any time).
   Still outstanding — the April build answers /health as of 2026-08-04.

Tier-0 free data feeds (commit `92a3f4e`) are live on both bots: dominance
(CoinGecko), Fear & Greed, RSS news-event boolean, retail-crowd + taker
ratios. **The Binance websocket layer SHIPPED same evening** (commits
`25156a7` + `3e1b045`, verified streaming on BOTH bots): markPrice@1s
drives a 2s fast-stop loop (same SL predicate/close path — attacks the
measured −1.39R overshoot), !forceOrder liquidations feed the committee
context. Gotchas baked into the code comments: futures ws now REQUIRES the
routed `/market/stream` path (legacy unrouted URL hangs on upgrade), and
the Mac's framework Python has NO default CA path (pass certifi
explicitly). Pre-ship adversarial review caught a stale-reference
double-close hazard — identity guards now in _close_position /
_take_partial / _mark_fully_exited_after_partial. Kill switch:
MM_WS_ENABLED=false. Still open from the ws queue: aggTrade CVD and the
Bybit allLiquidation self-built heatmap.

<details><summary>Superseded plan (pre-2026-07-31, kept for context)</summary>

## OLD §3 — BLOCKED ON USER — the Fly path (fresh-account plan)

Plan changed: Stuart is creating a **new Fly.io account** (old account token
dead), and he now has a **funded Anthropic API key** — which means the
committee can run on Fly via the plain SDK (`anthropic>=0.40.0` is already in
the image; no Claude CLI, no extra RAM needed). Everything is scripted in
`scripts/fly_bootstrap_new_account.sh`; read its header before running.

**User steps (~3 min total):**
1. Create the account in a browser: https://fly.io/app/sign-up — agents must
   not create accounts.
2. `fly auth login`
3. Put `ANTHROPIC_API_KEY=sk-ant-...` in **`env.local`** at the repo root
   (gitignored; the local bot does NOT read it). **NOT in `.env`** — that
   would silently flip the LOCAL bot from the free subscription CLI to
   metered API billing.
4. Stop the OLD Fly app's scanning via its own dashboard (app login, no Fly
   account needed): https://tarakta-mm.fly.dev/mm → Stop. Two live bots on
   one instance_id fight over positions.

**Then (user or agent):**
```bash
./scripts/fly_bootstrap_new_account.sh          # tarakta-mm2, sin, INSTANCE_ID=tarakta-mm
```
The script: creates the app, pipes secrets from `.env` + `env.local` into Fly
(values never printed), sets paper mode + committee SHADOW, and deploys with
the mandatory `--depot=false --remote-only`. With INSTANCE_ID=tarakta-mm the
new app ADOPTS the old book: comes up **paused** (`scanning_active:false` in
engine_state, read at startup) and manages the 3 stale open positions (expect
NEAR — open since 2026-07-01, $1,368 risk — to realise its hidden loss). If
the old app can't be stopped, pass a fresh instance id as arg 3 instead.

The old account's app will keep running its April build until stopped via its
dashboard or until the old account is recovered/deleted — its dashboard Stop
button is the practical kill switch.

</details>

## 4. How to check status in 60 seconds

```bash
launchctl list | grep tarakta && pgrep -f src.main
tail -20 /tmp/tarakta-bot.launchd.out.log
```
Then in Supabase (project `uounrdaescblpgwkgbdq`, table `mm_agent_decisions`):
```sql
SELECT created_at, instance_id, symbol, decision, confluence_grade,
       left(reason,100) AS reason, latency_ms
FROM mm_agent_decisions ORDER BY created_at DESC LIMIT 10;
```
Healthy = recent rows with `decision` APPROVE/VETO (**not** ERROR) and
`latency_ms` ~10–15k. Any run of `ERROR` means the LLM layer is down again —
check `.env` token first (that exact failure ran silently for six weeks).

## 5. Shadow-mode scoreboard (as of 2026-07-22)

8 verdicts since the token landed. Shadow logs but never blocks, so all were
traded anyway:

| Verdict | Outcome | Note |
|---|---|---|
| 6 VETO | 2 winners (+$57), 3 losers (−$140), 1 still open | Vetoes blocked both winners |
| 1 APPROVE | loser (−$59) | The only Grade-B, at-key-level setup |
| 1 ERROR | pre-fix, 2026-07-18 | Fixed by `244912f` |

Binding veto mode would have turned −$146 into −$59. **n=7 is far too small to
conclude anything.** Gate before flipping `MM_COMMITTEE_MODE=veto`: **≥30
closed trades / ≥2 weeks** (~10 more days at ~2 verdicts/day), scored against
outcomes. Veto reasons repeatedly cite *"not at key level"* and *"sideways
4H/1D"* — the location problem, live.

## 6. Why it is not profitable (verified, ranked — UPDATED 2026-08-03)

**New evidence:** `docs/BACKTEST_180D_2026-08-03.md` (180d × ETH/BTC/SOL/DOGE/
ADA, 38 signals, −8.7R). Key reranking: **~8R of the −8.7R comes from serial
re-entry** — a stopped loser frees the position slot in 1–3h and the
still-present formation re-enters the same wrong idea 3–7× (BTC 2026-07-20/21:
7 consecutive shorts; the live committee vetoed all 7 in shadow). Per
independent idea, entries are ~0R in BOTH directions (reversal tested and
rejected — the live inverse bot losing −$1.1k confirms). Original causes below
still stand; the re-entry martingale sits on top of them:

1. **Wrong candidates.** The detector is a 40-bar swing-pair matcher (1H: sees
   40h of history; 4H path: ~6.7 days). The course demands setups **only** at
   the high/low of the day/week after a 3-level count. Vision backtest over 43
   historical trades: **4/43 at a valid location** — including the 9 April
   Opus-approved picks. This is the root cause.
2. **Deflated grading.** 16 confluence factors; 4 dead (~0% fire), 1 permanent
   stub; dead weight stays in the denominator → real ceiling ~74% → everything
   grades C. The +$91k dead-factor zero-out was never shipped (gate: verify
   per-factor fire rates first — `CODEX_PLAN.md` Task 0.4).
3. **Exit leaks ~$5–6k/quarter.** Stops fill at −1.39R avg (soft stop, 5-min
   poll); the breakeven ladder waits on a Level-2 advance that rarely fires
   (36/41 stops never armed BE despite MFE up to 5R); scratch closes at market
   (observed live 2026-07-21: a "scratch" at **−1.36R**).
4. Flow data (funding, OI, orderbook) is fetched but reaches **only** the
   committee context — zero deterministic decisions use it.

## 7. Build queue (agreed with Stuart, in order)

1b. **NEW (2026-08-03, smallest + highest-certainty): post-stop-loss re-entry
   cooldown** per symbol+setup — kills the ~8R martingale documented in
   `docs/BACKTEST_180D_2026-08-03.md`. **Before shipping: find the course's
   own re-entry guidance and cite lesson+timestamp in the commit** (CLAUDE.md
   rule — do not invent the rule shape). Suggested start: search course
   transcripts for re-entry / "stopped out" / "wait for" guidance.

1. **Vision judge calibration** — gate NOT passed (0/8 on positive controls;
   1 approval in 60+ charts = hanging judge). Plan in
   `docs/STATUS_2026-07-19.md` §4: itemized-scoring probe → make TRUE absolutes
   veto and qualities grade → judge at reversal-candle close → re-run
   `experiments/vision/positive_controls.py` + `vision_controls_v2.py`
   (gate: ≥2/8 approvals AND chop/downtrend still rejected).
2. **Location-first nominator** — the big one. Persist prior-week/multi-week
   levels (today only the current week exists and it resets each Sunday),
   implement the 3-level count (course TBD-45), and only trigger review when
   price is AT such a level. Inverts the pipeline to match the course.
3. Deterministic groundwork: dead-factor zero-out (after Task 0.4), scan the 1D
   candles (fetched but never scanned), feed flow into deterministic scoring.
4. Exit fixes (course-cited + replay-validated): stop-fill realism, level-advance
   audit, scratch bound.
5. Committee → veto mode after the §5 gate.
6. Datafeeds: full verified catalog in `docs/DATAFEEDS_2026-07-31.md` —
   Tier 0 is all free (Binance ratio endpoints + websockets [markPrice@1s SL
   checks attack the −1.39R overshoot], Coinalyze, CoinGecko+CMC dominance,
   Fear & Greed, CoinDesk/Cointelegraph RSS; CryptoPanic free tier is DEAD),
   Tier 1 is CoinAnk $30/mo (only sub-$50 liquidation-heatmap API; 7-day
   trial first). Feed work stays subordinate to the nominator.
7. ERROR-streak alert on `mm_agent_decisions`.

## 8. Gotchas that will bite you (beyond CLAUDE.md)

- **Use the framework interpreter:** `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`. Bare `python3` is Homebrew 3.14 with **no project deps**.
- **TLS:** `~/.zshrc` used to export an **empty** `SSL_CERT_FILE`, breaking TLS for all Node tools. Line is commented out. For one-off CLI probes use `env -u SSL_CERT_FILE …`.
- **The `claude` CLI can exit 0 with `is_error:true`** inside its JSON — check the flag, not the exit code. Models also append prose after fenced JSON; `MMCommittee._extract_json` uses incremental `raw_decode` — keep it.
- **Never print `CLAUDE_CODE_OAUTH_TOKEN`** or any `.env` value.
- **DB contract:** new `trades` column = migration + `_TRADE_COLUMNS` in `repository.py` + use in `mm_engine.py`. Miss one → silent data loss.
- `trades.pnl_usd` is **cumulative** (includes partials). Never add `partial_exits.pnl_usd` on top.
- Committee replay/backtest scripts write **no** DB rows; only `experiments/vision/e2e_committee.py` writes one real row.
- Course transcripts are the source of truth. Every rule change cites lesson + timestamp in the commit (CLAUDE.md).

## 9. Artifact map

| Path | What |
|---|---|
| `src/strategy/mm_claude_cli.py` | Subscription-CLI transport (live) |
| `src/strategy/mm_committee.py` | Committee, CLI fallback, robust JSON extraction |
| `src/strategy/mm_engine.py` | Everything else: scan → gates → sizing → lifecycle |
| `docs/STATUS_2026-07-19.md` | Full narrative handoff |
| `docs/agent-committee/skills/` | Committee skill files (`structure_vision_specialist_DRAFT.md` is NOT loaded — vision is ungated) |
| `experiments/vision/` | Chart renderer, vision helper, both backtests, control gates, results JSONL + README. **Wired into nothing.** |
| `CODEX_PLAN.md`, `docs/MM_AGENT_COMMITTEE_DESIGN.md` | Governing plans |
| `docs/BACKTEST_180D_2026-08-03.md` | 180d backtest + reversal analysis: full 38-signal table, cause reranking |
| `scripts/replay_reversal_test.py` | Diagnostic: re-sims a replay log's signals both directions under symmetric management |
