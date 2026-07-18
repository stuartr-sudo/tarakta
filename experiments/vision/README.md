# Vision / replay experiments — 2026-07-18/19

Prototype harnesses from the profitability-review session. **Nothing here is
wired into the live engine.** The vision judge did NOT pass its positive gate
(see `docs/STATUS_2026-07-19.md` §4) — calibrate before promoting anything.

All scripts run with the framework interpreter and clean TLS env:

```bash
env -u SSL_CERT_FILE /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 <script>
```

They read `/Users/stuarta/tarakta/.env` via `src.config.Settings` — the
`CLAUDE_CODE_OAUTH_TOKEN` there powers all LLM calls (subscription, no API
billing). Never print that value.

| File | What it does |
|---|---|
| `chart_render.py` | Candlestick PNG renderer (EMA50/200, volume subplot, level lines). Promote to `src/` only after gates pass. |
| `vision_call.py` | Subscription-CLI vision helper: `claude -p --tools Read` so the model can open chart images. |
| `e2e_committee.py` | One-shot live committee run (writes ONE real row to `mm_agent_decisions`). |
| `replay_committee.py` | Text-context committee replay over historical trades. Result 2026-07-18: 31/32 vetoed, sample −$8,726 → +$145. Writes `results/replay_results.jsonl`; resumable. |
| `vision_controls.py` / `vision_controls_v2.py` | Negative-control gates (naive prompt vs course skill). v2 outcome: judge reads charts accurately; synthetics keep failing on legitimate course grounds. |
| `vision_backtest.py` | THE key experiment: fetches Binance candles as-of each historical trade's entry, renders 4H+1H charts, vision-judges with the course skill. Result: only 4/43 candidates at a course-valid location — incl. April Opus-approved picks. Writes `results/vision_backtest_results.jsonl`. |
| `positive_controls.py` | Textbook-moment hunt over 1y BTC/ETH/SOL. Result: 0/8 approved → **gate NOT passed** (judge too strict — see STATUS §4 calibration plan). |
| `results/*.jsonl` | Raw verdicts from the runs above (committed as evidence). |

Skill under test: `docs/agent-committee/skills/structure_vision_specialist_DRAFT.md`
(26 verbatim course quotes; NOT in `mm_committee.SPECIALISTS`, so the live
committee never loads it).
