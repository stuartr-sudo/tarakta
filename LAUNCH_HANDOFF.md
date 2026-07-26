# LAUNCH HANDOFF — read this first

**Last updated:** 2026-07-22 by Claude (Fable 5) · **Repo:** https://github.com/stuartr-sudo/tarakta · **Local:** `/Users/stuarta/tarakta`
**Branch:** `main` == `codex/tarakta-stabilization` == `22ba0bd` (pushed, clean) · **Tests:** 779 passed / 1 skipped

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
| **Local bot** (instance `main`) | ✅ Running, healthy, ~cycle 766+, scanning 10 pairs / 5 min | launchd `com.tarakta.bot`; restart: `launchctl kickstart -k gui/501/com.tarakta.bot`; stop: `launchctl unload ~/Library/LaunchAgents/com.tarakta.bot.plist` (a plain `kill` just gets KeepAlive-restarted). Logs: `/tmp/tarakta-bot.launchd.out.log` |
| **Committee** (5 Haiku specialists + Sonnet head trader) | ✅ Live in **shadow** (logs verdicts, blocks nothing) | `.env`: `MM_COMMITTEE_ENABLED=true`, `MM_COMMITTEE_MODE=shadow`. Auth = `CLAUDE_CODE_OAUTH_TOKEN` in `.env` (1-year token from `claude setup-token`). **Never print that value.** |
| **Fly prod** `tarakta-mm` | ⚠️ April build, bleeding, 3 stale open positions | **BLOCKED on user** — see §3 |
| **Inverse mirror** `tarakta-mm-inverse` | Idle; lost −$1.1k; recommended for retirement | Not yet disabled |

## 3. BLOCKED ON USER — the one open action

Stuart asked for Fly prod to be **redeployed**. It cannot be done without him:
`~/.fly/config.yml` has a dead token, so every `flyctl` command fails.

**He must run (browser login, ~20s):**
```bash
fly auth login
```

**Then the agent runs:**
```bash
fly deploy --depot=false --remote-only --app tarakta-mm
```
(`--depot=false --remote-only` is mandatory — the Depot builder times out on
this image. See CLAUDE.md.)

**What the redeploy does, and why it is the fix:** the engine reads
`engine_state.config_overrides.mm_engine_settings` **only at startup**
(`mm_engine.run()`, ~line 1178). That row has held `scanning_active: false`
since 2026-06-03 but the machine never restarted, so it kept trading. A
redeploy restarts it → **it comes up paused**: no new entries, bleed stops, and
it finally manages its 3 stale positions (expect NEAR — open since 2026-07-01,
$1,368 risk — to realise a loss that is currently hiding as unrealised).

**Important:** the redeploy does **not** bring the committee to Fly (no `claude`
CLI, no token in the image). Correct end state: **deploy it and leave it
paused** as a healthy standby. Local Mac stays the live experiment. Making Fly
the real home later needs: node + Claude CLI in the image,
`CLAUDE_CODE_OAUTH_TOKEN` as a Fly secret, and probably 4GB RAM (6 CLI
subprocesses per committee run will OOM a 2GB shared machine).

Fly secrets (Binance, Supabase, dashboard auth) are **already set** from prior
deploys — a redeploy reuses them. Nothing to re-enter.

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

## 6. Why it is not profitable (verified, ranked)

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
6. Free connectors for stubs: CoinGecko (dominance), Alternative.me (sentiment),
   CryptoPanic/RSS (news), 2nd CCXT exchange for resilience.
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
