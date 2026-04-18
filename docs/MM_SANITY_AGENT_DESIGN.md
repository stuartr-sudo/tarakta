# MM Sanity Agent (Agent 4) — Design Document

**Status:** Draft v2 / pre-implementation. No code has been written.

**v2 changes (2026-04-18):** Model default upgraded Haiku 4.5 → Opus 4.7 with extended thinking (Haiku rubber-stamps judgement calls). Prompt caching explicitly configured with `ttl=1h` (default changed to 5m in March 2026). Two-stage prompt with rubric + three worked fixtures moved into cached system prompt. Pre-computed derived-features contract specified — model reasons, engine computes. Monthly budget cap + Sonnet auto-downgrade added. New §11 — async learning loop via Claude Code Routines (announced 2026-04-14) for daily/weekly review that improves the prompt over time.

**v1 context (retained):** Response to the BNB-short-into-4H-uptrend loss (2026-04-17). The 4H EMA trend state was computed and thrown away in `mm_engine.py:1546` (`noqa: F841 — kept for future use`). The deterministic hard-veto is now wired (commit f95c507, migration 018). This doc specifies the LLM "sanity pass" that catches the judgement-call failures the rules cannot — the layer 3 mistakes where "three_hits_how at Level-3 exhaustion" looks right but isn't.

---

## 1. Scope & Role

Agent 4 is a **final human-style sanity check** that reviews a fully-built MM setup and decides whether to let it through. It is explicitly not a re-scorer or a target-picker.

**The agent does:**
- Read the formation + confluence result + multi-timeframe candles.
- Cross-check HTF alignment (4H, daily), session timing, Asia-spike context, multi-session vs same-session formation, and weekly phase.
- Return `APPROVE` or `VETO` with a reason grounded in specific MMM masterclasses.

**The agent does NOT:**
- Re-score confluence (`mm_confluence.py` owns that).
- Change entry price, SL, TP1/TP2/TP3 (sizing and targets are algorithmic).
- Manage open positions (that's a separate, out-of-scope decision; see "Future work").
- Discover new formations (the formation detector is authoritative).

**Pipeline position:**
Fires **after** `best_formation` is locked and confluence has passed, but **before** `MMSignal` is built and handed off to sizing/execution. Concretely: between the `retest_passed` gate (`mm_engine.py:2516`) and the `MMSignal(...)` construction (`mm_engine.py:2557`). This is the last safe moment at which a veto costs us only an LLM token, not capital.

**Hard-veto vs soft-advice.** Proposal: **hard veto** (agent decision is binding when enabled). Tradeoffs:

| | Hard veto | Soft advice |
|---|---|---|
| Pros | Actually prevents losses. Forces discipline. | Preserves pure-algo floor; easy to A/B vs baseline. |
| Cons | Adds LLM dependency to the critical path; API outages reduce trade count. | Defeats the purpose — the BNB trade still would have fired. |

Mitigation for hard-veto: graceful degradation (§7) fails **open** (approve) on API error so downtime doesn't stop all trading, plus a kill-switch config flag. Phase A shadow mode (§10) lets us quantify actual impact before making it binding.

**I/O contract:**

```json
// Request (assembled in §4)
{
  "symbol": "BNB/USDT",
  "direction": "short",
  "formation": { "type": "M", "variant": "multi_session", "quality": 0.72 },
  "confluence": { "score_pct": 58, "grade": "B", "factors_hit": ["mw_session_changeover", "ema50_break_volume"] },
  "htf": { "trend_4h": "uptrend", "trend_1d": "uptrend", "ema_alignment_4h": "bullish_stack" },
  "session": { "name": "uk", "minutes_into": 45, "is_asia_spike": false },
  "cycle": { "phase": "accumulation", "day_of_week": 4, "counter_trend": true },
  "candles": { "4h": "...", "1h": "...", "15m": "..." },
  "recent_trades": [{"direction":"short","pnl_pct":-2.1,"hours_ago":18}]
}

// Response (strict JSON, validated)
{
  "decision": "VETO",
  "reason": "Short setup against 4H uptrend (price >50>200 EMA, stacked bullish). Lesson 12 rule — do not short into an established higher-timeframe uptrend.",
  "confidence": 0.88,
  "htf_trend_4h": "uptrend",
  "htf_trend_1d": "uptrend",
  "counter_trend": true,
  "concerns": ["4h_alignment", "daily_alignment"]
}
```

`decision` ∈ `{"APPROVE","VETO"}`. `confidence` ∈ `[0.0, 1.0]`. Malformed responses are handled per §7.

---

## 2. Integration Point

**File:** `src/strategy/mm_engine.py`
**Insertion line:** immediately after the retest gate at line 2516 (`self._advance("retest_passed")`), before the `entry_type` classification block at line 2522.

All inputs the agent needs are already in local scope at that point:
- `best_formation`, `confluence_result`, `cycle_state`, `ema_state`, `ema_values`, `ema_break`, `rsi_state`, `adr_state`, `session`, `weekend`, `oi_increasing`, `level_analysis`
- `candles_1h`, `candles_4h`, `candles_15m`
- `trade_direction`, `current_price`, `symbol`, `now`

**Rejection propagation** follows the existing `_reject()` pattern (`mm_engine.py:351`) so the scan funnel counts it alongside other stages:

```python
# Pseudocode — DO NOT implement from this doc, see §10 rollout plan
verdict = await self.sanity_agent.review(context)  # returns AgentVerdict or None
# Always persist the decision (APPROVE or VETO) to mm_agent_decisions table
await self.repo.insert_mm_agent_decision(verdict, context_summary)

if verdict is None:
    # Graceful degradation — API fail/timeout → approve (fail-open)
    self._advance("sanity_agent_fail_open")
elif verdict.decision == "VETO":
    return self._reject(
        "sanity_agent_veto",
        symbol,
        agent_reason=verdict.reason,
        agent_confidence=verdict.confidence,
        htf_trend_4h=verdict.htf_trend_4h,
        htf_trend_1d=verdict.htf_trend_1d,
    )
else:
    self._advance("sanity_agent_approved")
# store verdict on signal so it persists to trades table on entry
self._last_agent_verdict = verdict
```

When `MMSignal` is built at line 2557, it carries the verdict so `insert_trade` writes `mm_agent_decision`, `mm_agent_reason`, `mm_agent_confidence`, `htf_trend_4h`, `htf_trend_1d` to the `trades` row.

**Graceful degradation rules:**
- API error / timeout / malformed JSON → log + return `None` → fail **open** (approve). Tracked under `sanity_agent_fail_open` funnel counter.
- Kill switch `MM_SANITY_AGENT_ENABLED=false` → agent short-circuits, returns `None` without an API call.
- **Hard latency cap** `MM_SANITY_AGENT_TIMEOUT_S=20` (scan cycle is 5min; 20s is plenty).

---

## 3. Prompt Design

Two-stage prompt within a single API call. Stage A is free-form analysis (captured in the extended-thinking output); Stage B is the binding JSON verdict. This chain-of-thought structure materially out-performs single-stage prompting on judgement tasks — the model commits to a verdict only after walking through the rubric explicitly.

The system prompt is fully static (cacheable, `ttl=1h`) and contains: rubric, lesson excerpts by name, worked fixture examples, and the output schema. The user prompt is the per-call data only — no course text, no examples.

### System prompt (static, cached)

```text
You are the MM Method sanity reviewer for an automated crypto bot. A rule-based
engine has already passed this setup through a formation check, an HTF-trend
hard-veto, and confluence scoring. Your job is to catch the judgement-call
failures those rules cannot catch, grounded in the MMM Masterclasses course.

RUBRIC — reason through these in order before committing to a verdict:

1. HTF alignment (Lesson 12, Trend / EMAs):
   A short into a cleanly stacked bullish 4H (10>20>50>200) is counter-trend.
   A long into a stacked bearish 4H is counter-trend. Counter-trend is only
   acceptable at Level-3 exhaustion with explicit exemption variant
   (three_hits_how/low, final_damage, half_batman, nyc_reversal, 200ema_rejection,
   stophunt). If the 4H is accelerating (fan-out widening), even exemption
   variants are suspect — the move isn't exhausting.

2. Daily trend (Lesson 12 continued):
   1D direction should agree with 4H for A-grade setups. If 1D contradicts
   4H, this is a transition zone — tighter scrutiny, lower confidence.

3. Session timing (Lesson 04):
   Entries are cleanest at session changeover (Asia close, London open,
   NY open). Mid-session entries on reversal variants are weaker. Brinks
   setups are exception — they are session-dependent by design.

4. Asia spike bias (Lesson 15, London range):
   If Asia printed a wide spike in the opposite direction of the setup
   and we're in early UK session, that is a trap trigger — veto.

5. Formation composition (Lesson 07, M and W):
   Multi-session formations are premium; same-session formations are noise
   unless at HOW/LOW with strong retest (3+ of 4 conditions). A same-session
   formation with 2/4 retest conditions on a Grade C score is the weakest
   setup the engine allows — veto unless exceptional.

6. Weekly phase (Lesson 03, Weekly setup):
   Mon/Tue accumulation → aligned trades only.
   Wed/Thu range → counter-trend setups valid only at exhaustion.
   Fri trap → veto counter-trend shorts regardless of formation; the Friday
   trap is a known MM move and we should not fight it from the wrong side.

7. Recent outcome context:
   If the last 3 trades on this symbol in this direction all lost, the
   current regime for this symbol is against us. Heavily weight VETO.

WORKED EXAMPLES:

EXAMPLE 1 — BNB short 2026-04-17 (the trade this agent exists to catch):
- direction=short, variant=three_hits_how, grade=F(37.8%), retest_met=2/4
- 4h_trend=bullish strength=0.72 accelerating=true
- 1d_trend=bullish
- Reasoning: Exemption list includes three_hits_how, BUT 4H is accelerating
  so exemption is void per Rubric 1. Grade F + aggressive entry already
  marginal. 2/4 retest.
- Verdict: VETO, confidence 0.92, concerns=[4h_alignment, accelerating_trend, low_grade]
- reason: "Three_hits_how exemption voided by accelerating 4H uptrend; Grade F
  signal does not justify fighting the trend."

EXAMPLE 2 — textbook BTC long at LOW:
- direction=long, variant=multi_session, grade=A(72%), retest_met=3/4
- 4h_trend=bullish strength=0.61 accelerating=false
- 1d_trend=bullish
- session=london_open minutes_in=12
- Reasoning: HTF aligned, premium multi-session formation, strong retest,
  clean session timing. Textbook setup.
- Verdict: APPROVE, confidence 0.94, concerns=[]
- reason: "Multi-session W at LOW with 3/4 retest, HTF aligned, London open."

EXAMPLE 3 — ETH Friday trap short:
- direction=short, variant=standard, grade=B+(61%), retest_met=3/4
- 4h_trend=bullish strength=0.45
- weekly_phase=FRIDAY_TRAP dow=4
- Reasoning: Rubric 6 explicitly vetoes counter-trend shorts in Friday trap
  regardless of formation quality. Confluence is high but context wrong.
- Verdict: VETO, confidence 0.89, concerns=[friday_trap, wrong_phase]
- reason: "Friday trap phase — course rule explicitly forbids counter-trend
  shorts regardless of setup quality."

OUTPUT SCHEMA (strict, JSON only, no prose):
{
  "decision":    "APPROVE" | "VETO",
  "reason":      "<=30 words, cite the Rubric point number",
  "confidence":  0.0-1.0,
  "htf_trend_4h":"bullish" | "bearish" | "sideways",
  "htf_trend_1d":"bullish" | "bearish" | "sideways",
  "counter_trend":true | false,
  "concerns":    [<=4 tags from the vocabulary]
}

CONCERNS VOCABULARY (controlled — do not invent tags):
4h_alignment, daily_alignment, accelerating_trend, asia_spike, same_session,
wrong_phase, friday_trap, low_retest, low_grade, low_quality_formation,
recent_losses, mid_session
```

### User prompt template (per-call, minimal)

```text
SETUP
symbol={symbol} direction={direction} formation={ftype}/{fvariant}
grade={grade} confluence={score_pct}% retest_met={retest_met}/4
entry={entry_price} sl_ref={sl_ref}

HTF (pre-computed)
4h_trend={htf_trend_4h} strength={htf_4h_strength} accelerating={htf_4h_accel}
1d_trend={htf_trend_1d}
price_vs_50ema_pct={p50_pct} price_vs_200ema_pct={p200_pct}
counter_trend={counter_trend}

SESSION & CYCLE
session={session_name} min_in={minutes_in}
asia_range_pct={asia_range_pct} asia_spike_dir={asia_dir}
weekly_phase={phase} dow={dow}
multi_session_formation={multi_session}

RECENT (last 5 closed {symbol} MM trades)
{recent_trades_oneline}

REGIME (last 10 closes per TF, oldest-first; for context only)
4h: {c4h_closes_10}
1h: {c1h_closes_10}
15m: {c15m_closes_10}

Return JSON per the schema in your system prompt.
```

**Why the per-call prompt is so small:**
- All derived features are computed by the engine (see §4), not the model. Counting candles or computing EMAs is work the model shouldn't spend reasoning-budget on.
- Only 10 closes per TF, not 30 — enough to see a trend, cheap to send.
- No lesson text per call — it's in the cached system prompt.
- Total per-call input: ~800–1000 tokens. The system prompt is ~10–12K tokens (cached).

---

## 4. Input Context Assembly

Core principle: **the engine does all computation; the model does all reasoning.** Every derived feature the model would otherwise have to compute is a judgement error waiting to happen. A `_build_sanity_context(...)` helper on the engine assembles a pre-computed payload from state already in scope at the insertion point.

### Pre-computed derived features (sent to agent)

| Key | Value | Source |
|---|---|---|
| `symbol` | `"BNB/USDT:USDT"` | local var |
| `direction` | `"long"` \| `"short"` | `trade_direction` |
| `formation.type` | `"M"` \| `"W"` | `best_formation.type` |
| `formation.variant` | `"three_hits_how"` etc. | `best_formation.variant` |
| `formation.quality` | 0.0-1.0 | `best_formation.quality_score` |
| `formation.at_key_level` | bool | `best_formation.at_key_level` |
| `formation.multi_session` | bool | derived from `best_formation.peak1_idx` / `peak2_idx` spanning >1 session |
| `grade` | `"A+"`..`"F"` | `confluence_result.grade` |
| `score_pct` | 0.0-100.0 | `confluence_result.score_pct` |
| `retest_met` | 0-4 | `confluence_result.retest_conditions_met` |
| `entry_price` | float | local var |
| `sl_ref` | float | local var |
| `htf_trend_4h` | `"bullish"\|"bearish"\|"sideways"` | `trend_state_4h.direction` (now wired post-2026-04) |
| `htf_4h_strength` | 0.0-1.0 | `trend_state_4h.strength` |
| `htf_4h_accel` | bool | `trend_state_4h.is_accelerating` |
| `htf_trend_1d` | same enum | `trend_state_1d.direction` |
| `price_vs_50ema_pct` | signed % | `(close - ema50) / ema50 * 100` |
| `price_vs_200ema_pct` | signed % | `(close - ema200) / ema200 * 100` |
| `counter_trend` | bool | direction opposes non-sideways htf_trend_4h |
| `session_name` | `"asia"\|"uk"\|"ny"\|"gap"` | `session.session_name` |
| `minutes_in` | int | `session.minutes_into_session` |
| `asia_range_pct` | float | `_compute_asia_range_pct(symbol)` — unconditional |
| `asia_spike_dir` | `"up"\|"down"\|"none"` | sign of (asia_high-open) vs (open-asia_low) with 60% magnitude threshold |
| `weekly_phase` | e.g. `"BOARD_MEETING_2"` | `cycle_state.phase` |
| `dow` | 0-6 (Mon=0) | `now.weekday()` |
| `recent_trades` | list of 5 oneliners | `repo.get_recent_trades(symbol, strategy='mm_method', limit=5)` |

### Candles (minimal, closes only)

- `4h_closes`: last 10 closes, oldest-first
- `1h_closes`: last 10 closes, oldest-first
- `15m_closes`: last 10 closes, oldest-first

**Total candle payload: 30 floats.** Just enough for the model to sanity-check the computed trend direction against visible price action. Not enough to invite the model to recompute indicators.

### Recent-trades rendering

One line per trade, deterministic format so the model parses it reliably:
```
dir=short grade=B pnl_pct=-2.1 hrs_ago=18 exit=sl variant=three_hits_how
dir=short grade=C pnl_pct=-1.5 hrs_ago=42 exit=sl variant=standard
dir=long  grade=A pnl_pct=+3.4 hrs_ago=76 exit=tp2 variant=multi_session
```

Three recent losses in the current direction is a regime signal; the agent should weigh this explicitly per Rubric point 7.

### What's NOT sent

- Raw OHLCV bars. The model does not need opens, highs, lows, volumes.
- Lesson text. It's in the cached system prompt.
- The engine's internal state dicts. Only the derived facts the agent needs.
- The full confluence factor breakdown. Just the grade and %. We don't want the agent second-guessing the scorer.

### Prompt-version header

Each user-prompt starts with `# prompt_v=3 rubric_v=1` so we can join agent decisions to a prompt version when analysing decision quality over time. Bumping the system prompt bumps `prompt_v`; bumping the rubric (inside the system prompt) bumps `rubric_v`. The `mm_agent_decisions` row records both.

---

## 5. DB Schema Additions

Per `docs/MM_ENGINE_INTEGRATION_GUIDE.md`, each new `trades` column needs: (1) migration, (2) `_TRADE_COLUMNS` entry in `repository.py`, (3) exact name used in `mm_engine.py`. The `mm_agent_decisions` table is new.

### Migration: `migrations/016_mm_sanity_agent.sql`

```sql
-- Columns on trades (filled only for MM trades that actually entered)
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_agent_decision text;      -- 'APPROVE' | 'VETO' | NULL
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_agent_reason text;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_agent_confidence numeric; -- 0.0 - 1.0
ALTER TABLE trades ADD COLUMN IF NOT EXISTS htf_trend_4h text;           -- 'uptrend' | 'downtrend' | 'ranging'
ALTER TABLE trades ADD COLUMN IF NOT EXISTS htf_trend_1d text;

-- Standalone decision log — records ALL agent calls, including VETOs that
-- never became trades. Primary observability surface.
CREATE TABLE IF NOT EXISTS mm_agent_decisions (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol          text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    cycle_count     integer,
    formation_type  text,                -- 'M' | 'W'
    formation_variant text,
    confluence_grade text,               -- 'A+'|'A'|'B+'|'B'|'C'
    confluence_pct  numeric,
    direction       text,                -- 'long' | 'short'
    decision        text NOT NULL,       -- 'APPROVE' | 'VETO' | 'ERROR'
    reason          text,
    confidence      numeric,
    htf_trend_4h    text,
    htf_trend_1d    text,
    counter_trend   boolean,
    concerns        jsonb,               -- list of concern tags
    input_context   jsonb,               -- the full payload we sent (trimmed candles)
    raw_response    text,                -- verbatim model output for debugging
    model           text,                -- 'claude-sonnet-4-6' etc
    latency_ms      integer,
    cost_usd        numeric,
    trade_id        uuid REFERENCES trades(id) -- nullable; set when decision=APPROVE and entry executes
);
CREATE INDEX idx_mm_agent_decisions_symbol_time ON mm_agent_decisions (symbol, created_at DESC);
CREATE INDEX idx_mm_agent_decisions_decision    ON mm_agent_decisions (decision);
```

### `_TRADE_COLUMNS` update in `repository.py`

Add: `"mm_agent_decision"`, `"mm_agent_reason"`, `"mm_agent_confidence"`, `"htf_trend_4h"`, `"htf_trend_1d"`.

**Checklist (per MM integration guide §Common Mistakes):** migration applied, allowlist updated, column names match exactly, verified via `pytest tests/test_mm_engine.py -k agent`.

---

## 6. Model & Cost

**Default model: `claude-opus-4-7` with extended thinking enabled.**

Rationale — earlier drafts proposed Haiku 4.5 on the assumption that this is "classification with reasoning." It isn't. The task has three layers:

1. **Pattern match** ("is 4H bullish AND direction short?") — any model handles this.
2. **Course rubric lookup** ("given variant X, which lesson applies? is an exemption warranted?") — Sonnet-class territory.
3. **Judgement calls on edge cases** ("this looks like three_hits_how at Level-3 exhaustion, but *is it actually*, or is it mid-trend noise reading as exhaustion?") — Opus territory.

The BNB trade this whole agent exists to catch failed on layer 3. The deterministic HTF hard-veto (committed in migration 018 / `mm_engine.py` post-2026-04-audit) now handles layer 1. The sanity agent exists specifically for layers 2–3 — the judgement calls rules can't make. Those calls justify Opus.

**Extended thinking is non-negotiable.** Without interleaved thinking even Opus can confidently approve bad trades by matching-without-reasoning on a rich rubric. With thinking, the model walks through "what is the 4H trend? does an exemption apply here specifically? what would a course-fluent trader actually say?" before committing to a verdict. That reasoning output also becomes our audit trail.

**Sonnet 4.6 as budget-cap fallback** (see §7 monthly cap). Sonnet + extended thinking approaches Opus quality on ~90% of cases; the 10% gap is exactly the edge cases where we care most, so Opus is the default.

**Haiku 4.5 is explicitly rejected** for this task. It will rubber-stamp judgement calls.

**Prompt caching: 1-hour TTL, explicit.** The default cache TTL changed from 1h → 5m in March 2026 (silently). Our call cadence (bursty around session changeovers) doesn't clear the 5m threshold reliably, but 1h does. The system prompt (course excerpts + rubric + fixtures, ~10–15K tokens) is fully static and must be sent with `"cache_control": {"type": "ephemeral", "ttl": "1h"}`. Cache writes cost 2x base input price; cache reads are 0.1x (90% discount). Over a typical day we expect ~5–10 cache writes and ~100–200 cache reads on the system prompt, vastly dominating in favour of reads.

**Volume estimate:**
- 20 symbols × 12 scans/hour × 24h = 5,760 scan-runs/day. Agent fires only after `retest_passed`.
- Funnel data: ~1-3% of pair-scans reach that gate → **~60–170 agent calls/day**.
- Upper bound: 250/day.

**Cost estimate (Opus 4.7 + extended thinking + 1h caching):**

| Component | Per-call | Daily (200 calls) | Monthly |
|---|---|---|---|
| System prompt (cached read) | ~12K tokens × $1.50/MTok × 0.1 = $0.0018 | $0.36 | $11 |
| System prompt (cache write, 1/hr) | ~12K tokens × $1.50/MTok × 2 = $0.036 × 16 hrs = $0.58/day | $0.58 | $17 |
| Per-call input (setup context) | ~1K tokens × $15/MTok = $0.015 | $3.00 | $90 |
| Thinking output | ~800 tokens × $75/MTok = $0.060 | $12.00 | $360 |
| Final JSON output | ~150 tokens × $75/MTok = $0.011 | $2.20 | $66 |
| **Total** | **~$0.088/call** | **~$18/day** | **~$540/month** |

Without caching, same config would be ~$700/mo. With caching, ~$540/mo. **One prevented BNB-class trade = $684** — the agent pays for itself preventing a single loss.

Compare to alternatives:
- Sonnet 4.6 + thinking + caching: ~$180/mo (budget-cap fallback)
- Opus without thinking: ~$220/mo (but loses the reasoning quality boost — not recommended)
- Haiku: ~$8/mo (but fails the task — not recommended)

**Monthly budget cap:** `MM_AGENT_MONTHLY_BUDGET_USD` (default $600). When projected monthly spend (based on trailing-7d rate × 30) exceeds 90% of cap, auto-downgrade to Sonnet 4.6 for the remainder of the month and send a notification. Prevents runaway cost if formation frequency spikes.

**Latency:** Opus 4.7 with thinking: median ~4–6s, p95 ~12s. Timeout cap **20s**. Scan cycle is 5min so even full-timeout on every call has zero scheduling impact.

**Usage tracking:** every call to `generate_json()` must pass `caller="mm_agent"` plus `repo`, optional `trade_id` (set on APPROVE→entry). This keeps `/usage` page auto-tracked; `MODEL_PRICING` dict needs Opus 4.7 entries added (with thinking + cached token pricing).

---

## 7. Failure Modes & Graceful Degradation

| Failure | Behavior | Notes |
|---|---|---|
| API call raises (network, 5xx, auth) | Return `None` → **fail open** (approve). Log `sanity_agent_error` with exception. Record `decision='ERROR'` row in `mm_agent_decisions` with `raw_response=str(e)`. | Production outage must not stop all MM trading. |
| Timeout (>20s) | Treated same as API error. `asyncio.wait_for` wraps the call. | |
| Malformed JSON / missing keys | One retry with `response_format=json_object` and a "your previous output was not valid JSON" nudge. On second failure, fail-open with `decision='ERROR'`. | Keeps noise low. |
| Agent returns all-APPROVE for days | Drift alert (see §9) triggers if approve-rate > 95% over 50 rolling decisions. Flags prompt regression. | |
| Agent returns all-VETO | Drift alert if approve-rate < 10% over 50 rolling decisions. Likely a prompt or model issue. | |
| Kill switch | `MM_SANITY_AGENT_ENABLED=false` (default `true` after Phase C). Short-circuits with no API call and no row written. | First resort when anything is suspicious in prod. |
| Confidence threshold (optional) | Config `MM_SANITY_AGENT_MIN_CONFIDENCE=0.6`. Below this, we downgrade a VETO to a warning log and approve. Prevents low-confidence agent calls from dominating. | Disabled by default; enable only if noise shows up. |
| Monthly cost cap | Config `MM_AGENT_MONTHLY_BUDGET_USD` (default $600). When projected monthly spend (trailing-7d × 30) exceeds 90% of cap, auto-downgrade to `claude-sonnet-4-6` for the remainder of the month. Notification sent on every downgrade. Resets on 1st of month. | Keeps Opus-by-default + hard floor on bill. |
| Model unavailable | If Opus 4.7 returns overloaded_error or similar capacity errors, fall through to Sonnet 4.6 for that call only. Log `sanity_agent_model_fallback`. | Don't fail open just because the premium model blinked. |

All funnel buckets: `sanity_agent_approved`, `sanity_agent_veto`, `sanity_agent_fail_open`, `sanity_agent_low_confidence_bypass`, `sanity_agent_model_fallback`, `sanity_agent_budget_downgraded`.

---

## 8. Testing Strategy

**Unit tests (`tests/test_mm_sanity_agent.py`):**

1. **Prompt assembly** — given a fixed engine state, `_build_sanity_context()` produces the expected dict. Snapshot test.
2. **JSON parsing** — handles valid JSON, missing fields, extra fields, trailing prose. Malformed JSON retries then falls open.
3. **Graceful degradation** — mock the LLM client to raise; verify `None` returned and `sanity_agent_error` logged once.
4. **Kill switch** — with flag off, `review()` returns `None` with zero API calls.
5. **Latency cap** — mock LLM with `asyncio.sleep(30)`; verify timeout at 20s.
6. **Integration with `_reject`** — VETO path increments `_scan_reject_counts['sanity_agent_veto']`.

**Fixture-based decision tests (`tests/fixtures/mm_sanity/`):**

Each fixture is a JSON snapshot of the payload the agent would receive, plus the asserted expected decision. Run against **a mocked LLM** that replays pre-recorded agent responses (for CI determinism) and against the **live LLM** in a nightly integration test.

Required fixtures:
- `bnb_short_4h_uptrend_2026_04_17.json` — the trade this whole project exists to catch. **Assertion: `decision == "VETO"` AND `concerns` contains `4h_alignment` AND `counter_trend == true`. Reason must cite Rubric 1 or 4H trend explicitly.**
- `btc_long_multi_session_at_how.json` — textbook long at LOW during accumulation, HTF aligned. **Assertion: `decision == "APPROVE"` AND `confidence > 0.75`.**
- `eth_short_friday.json` — Friday trap counter-trend short. **Assertion: `decision == "VETO"` AND `concerns` contains `friday_trap` or `wrong_phase`.**
- `sol_same_session_mid_session.json` — same-session M in middle of London, grade C. **Assertion: `decision == "VETO"` AND `concerns` contains `same_session` OR `mid_session`.**
- `doge_asia_spike_counter.json` — long after a wide Asia pump during early UK. **Assertion: `decision == "VETO"` AND `concerns` contains `asia_spike`.**
- `ada_3_recent_losses_same_direction.json` — regime signal, recent_trades shows 3 losing shorts, new short proposed. **Assertion: `decision == "VETO"` AND `concerns` contains `recent_losses`.**
- `link_exempt_at_exhaustion.json` — three_hits_low counter-4H-trend, but htf_4h_accel=false, Level-3 confirmed, grade A. **Assertion: `decision == "APPROVE"` AND reason cites Level-3 exhaustion exemption.**

The BNB fixture is non-negotiable and must ship with the initial PR. Additionally, the BNB fixture is run against **both** the mocked LLM (for CI determinism) **and** the live Opus 4.7 (in a nightly integration test) — if the live model ever fails this assertion, the build goes red and we must investigate before shipping a prompt change.

**Fixture schema** matches the user-prompt template in §3. Each fixture is a dict of the pre-computed derived features listed in §4, with recorded expected assertions in a separate block:

```json
{
  "input": { /* matches §4 derived features table */ },
  "expected": {
    "decision": "VETO",
    "concerns_must_include": ["4h_alignment"],
    "reason_must_match_regex": "(4[hH] (?:up)?trend|Rubric 1)",
    "min_confidence": 0.80
  }
}
```

---

## 9. Observability

**Dashboard page `/mm-agent`:**
- Last 50 decisions table (symbol, direction, formation, grade, decision, confidence, reason, link to chart).
- Stacked bar: approve-rate by grade and by formation variant.
- Most common `concerns` tags in VETOs (drives prompt iteration).
- Rolling 7-day approve-rate with bands (alerts fire outside 15%–85%).
- Per-model/caller cost read from the existing `api_usage` table filtered by `caller='mm_agent'`.

**Wiring into `/usage`:** no changes needed — `caller="mm_agent"` surfaces automatically in the existing per-agent breakdown. Add a new color for it in the chart legend.

**Alerts (via existing `/usage` threshold mechanism + new rules):**
- Daily cost > `MM_SANITY_AGENT_ALERT_USD` (default $2/day).
- Approve-rate < 15% OR > 85% over 50 rolling decisions (drift).
- `sanity_agent_fail_open` count > 20% of decisions in last hour (API health).

**Post-trade attribution:**
The `trade_id` foreign key on `mm_agent_decisions` lets us later join: for every closed MM trade, what did the agent say, and was it right? A simple view `mm_agent_hindsight` computes agent precision/recall once we have enough closed trades.

---

## 10. Rollout Plan

### Phase A — Shadow mode (1–2 weeks)

- Deploy with `MM_SANITY_AGENT_ENABLED=true` but `MM_SANITY_AGENT_VETO_ENABLED=false`.
- Agent runs on every setup that reaches the insertion point; writes rows to `mm_agent_decisions`; VETOs are **logged only** and the trade proceeds as before.
- Success criteria to exit:
  - ≥100 decisions logged.
  - On the ≥10 losing trades in the window, agent would have VETO'd ≥60% of the worst outcomes (R < −0.5).
  - Zero false VETOs on A+/A-grade winning trades (agent doesn't kill the best setups).
  - Cost within budget (§6).

### Phase B — Veto on C-grade only

- `MM_SANITY_AGENT_VETO_ENABLED=true`, but only binding if `confluence_grade in ('C',)`.
- Higher grades still run shadow (log-only). Riskiest trades first — they're where the agent has most leverage.
- Run 1 week; monitor approve-rate drift and dashboard.

### Phase C — Veto on all grades

- Veto binding on all grades.
- Default config from this point forward.

### Rollback

- **Instant:** flip `MM_SANITY_AGENT_ENABLED=false` via `engine_state.config_overrides` (no redeploy).
- **Downgrade:** flip `MM_SANITY_AGENT_VETO_ENABLED=false` to revert to shadow mode without losing the decision log.
- **Full removal:** leave columns and table in place (cheap), disable flag, remove insertion block. No migration reverse needed.

---

## 11. Async learning loop — Claude Code Routines

The real-time sanity agent is only as good as its current prompt. The prompt gets *better* via a feedback loop that analyses closed-trade outcomes against agent decisions and surfaces improvements. That feedback loop is a poor fit for in-scan API calls (slow, high-volume, not time-critical), but is a **perfect** fit for Claude Code Routines — cloud-hosted scheduled automations announced 2026-04-14.

**Why Routines here, not for the real-time agent:**
- Daily run caps (5–25 depending on plan tier) rule out Routines as the real-time decision path (we expect 60–250 setups/day).
- Latency model is async — routines are scheduled/webhook-triggered, not inline.
- But for the learning loop (1 run/day, 1 run/week) they're ideal: full repo access, runs in Anthropic's cloud, no local daemon needed.

**Routine 1 — `mm-agent-daily-review` (runs 06:00 UTC daily):**
- Trigger: cron
- Scope: yesterday's MM trade outcomes joined to `mm_agent_decisions`
- Task: identify (a) APPROVEs that lost >1R, (b) VETOs that would have won, (c) patterns in the `concerns` tags
- Output: writes a markdown report to `docs/mm-agent-reviews/YYYY-MM-DD.md` and posts a summary to `/mm-agent` dashboard
- Cost impact: ~1 Routine run/day, well under cap

**Routine 2 — `mm-agent-weekly-audit` (runs Sunday 08:00 UTC):**
- Trigger: cron
- Scope: last 7 days of `mm_agent_decisions`
- Task: compute agent precision/recall, surface most-common VETO reasons that were wrong, recommend prompt adjustments
- Output: opens a draft PR against `docs/MM_SANITY_AGENT_DESIGN.md` with proposed rubric amendments for human review
- Cost impact: 1 Routine run/week

**Routine 3 — `mm-agent-prompt-version-review` (runs 1st of month):**
- Trigger: cron
- Scope: full month of decisions
- Task: review accumulated weekly-audit findings, propose a prompt-version bump with updated rubric points and fixtures
- Output: opens a PR modifying the system prompt constants in `src/strategy/mm_sanity_agent.py` plus an updated `prompt_v` header
- Cost impact: 1 Routine run/month

**Routine 4 — `mm-loss-deep-dive` (GitHub webhook on trade-close events):**
- Trigger: webhook from the bot when a trade closes with pnl_pct < −2.0%
- Task: pull the full `mm_agent_decisions` row, the candle context, the course lessons referenced, and produce a "what went wrong" analysis
- Output: comment on a GitHub issue `#mm-losses` (one issue per calendar week)
- Cost impact: rare (only big losses), bounded by trade frequency

**Why this architecture is the right shape:**
1. **Separation of concerns:** real-time decisions use direct API calls (bounded, fast, critical); reflective learning uses Routines (async, thorough, improves the system).
2. **No duplicate work:** the real-time agent doesn't need to know about yesterday's decisions — Routines summarise history and fold findings into the prompt for the *next* cycle.
3. **Human-in-the-loop by default:** Routines open PRs rather than directly rewriting prompts. Every rubric change goes through review.
4. **Self-improving without drift:** the prompt-version header on every call means we can roll back instantly if a bad rubric change ships.

Implementation defers to Phase B (after the real-time agent is stable in shadow mode). The Routines themselves are thin — each is a 20–40 line Claude Code prompt config plus a cron/webhook trigger.

---

## Out of scope (noted for future work)
- Agent 4 managing open MM positions (separate scope — mirrors SMC Agent 3 but for MM).
- Using agent output to *boost* confluence (e.g., agent high-confidence APPROVE → grade upgrade). Adds bias; don't do it in v1.
- Ensemble with multiple models for hard decisions.
- Routines-driven A/B testing of prompt variants (v. attractive but requires traffic-splitting infra we don't have yet).
