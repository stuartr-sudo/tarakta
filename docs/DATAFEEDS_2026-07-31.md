# Datafeed deep dive — 2026-07-31

Five research agents surveyed the crypto data-API landscape with **live checks
against each vendor's current pricing/docs pages on 2026-07-31** (several free
endpoints were also exercised with real requests from this machine). An
adversarial verification pass re-checked the headline pricing claims.
Question: what should feed `src/strategy/mm_data_feeds.py` next — filling the
permanent stubs (liquidation heatmap, news, dominance/totals, sentiment,
options) and hardening the real feeds — at minimum cost.

**Headline: everything the course actually needs except the liquidation
heatmap is available for $0**, and the heatmap has exactly one sub-$50 option.
The single highest-impact item is not a vendor at all — it is switching
position management from 5-minute REST polling to Binance's free websockets.

---

## Tier 0 — free, verified working, wire now

| Source | Fills | Verified free tier | Effort |
|---|---|---|---|
| **Binance extras** (already our venue) | liquidation events, crowd vs top-trader positioning, taker aggression | `topLongShortAccountRatio`, `globalLongShortAccountRatio`, `takerlongshortRatio` returned live data via curl today; `!forceOrder@arr` liquidation websocket connected (caveat: snapshot stream, max 1 order/symbol/sec — undercounts cascades) | S — copy-adjacent to the existing `topLongShortPositionRatio` call |
| **Binance websockets for position mgmt** | real-time SL checks, `markPrice@1s`, `aggTrade` → local CVD | free, keyless, push; no history (CVD accrues from deploy) | M — background task + rolling buffers. **Directly attacks the measured −1.39R stop overshoot** (soft stops at 5-min-poll price) and unlocks the course's "stopping volume"/delta-divergence read at M/W peak 2 |
| **Coinalyze** | cross-exchange aggregated liquidation history, OI, funding + predicted funding, L/S ratio | free API key, 40 req/min; intraday granularities keep ~1500–2000 points (daily kept forever) | S — one REST feed class |
| **CoinGecko `/global`** | BTC dominance, TOTAL; TOTAL2/3 by arithmetic (TOTAL − BTC − ETH) | Demo key: 10k calls/mo (docs state 100/min); data refresh ~10 min — one call per 10 min uses half the cap | S — fills the lesson-31 `DominanceProvider` stub outright |
| **CoinMarketCap free Basic** | historical dominance backfill (`btc_dominance`, `altcoin_market_cap` = TOTAL2) | 15k credits/mo, 50 req/min; historical ≈ 1 credit / 100 points | S — so dominance *trends* work from day one |
| **Alternative.me Fear & Greed** | the sentiment stub | free, no auth (returned value=25 "Extreme Fear" during the check); updates daily | S — one cached fetch/hour |
| **CoinDesk + Cointelegraph RSS** | the news stub's "major event now" boolean | free, both feeds fetched live and valid; poll politely every 5–15 min | S — 2-of-2 keyword hit within 30 min = event |
| **Bybit + OKX public via CCXT** | multi-venue OI/funding consensus (3 venues agreeing on OI direction ≫ 1) | keyless; verified live from this machine on ccxt 4.5.40 | S |
| **Bybit `allLiquidation` websocket** | FULL liquidation feed (unlike Binance's 1/sec snapshot) → the budget path to a **self-built** liquidation heatmap | free, no key | M — lands after the Binance WS layer exists |

Monthly cost of all of Tier 0: **$0**.

## Tier 1 — the one paid candidate ($30/mo, trial first)

**CoinAnk OpenAPI** — the only verified sub-$50 API serving actual
liquidation **heatmap/map** endpoints (Liquidation Map, Aggregated Map,
Heatmap in its catalog). Plan1 $30/mo (30 req/min, history restricted to
4h+ intervals); **7-day free trial** — use it to confirm Plan1 actually
serves the heatmap endpoints before paying (the pricing page doesn't gate
endpoints per plan explicitly). This fills the permanent Hyblock/TradingLite
stub with the course's "liquidity magnet" signal. Integration M: heatmap →
nearest-large-cluster-above/below-price reducer.

For contrast (why not the names you know): **Coinglass** gates its heatmap
behind Professional at **$699/mo** (the $29 Hobbyist tier mostly duplicates
free Coinalyze at 4h granularity); **Hyblock** gates API access behind
**$399/mo** ($69 tier is charts-only, no API).

## Skips, with reasons (all verified 2026-07-31)

- **CryptoPanic** — free Developer tier **discontinued 2026-04-01**; cheapest real plan $199/mo for 3k req/mo. RSS covers the need for $0.
- **LunarCrush** — free tier now excludes ALL social data; cheapest social tier $90/mo.
- **Messari** — News/Signals APIs are Enterprise (~$417/mo equivalent).
- **X (Twitter) API** — no free tier; pay-per-usage economics don't fit.
- **CryptoQuant** — $99/mo tier is 1-day resolution / 20 req/min: useless for a 5-min cycle; minute-resolution is $799/mo annual-only.
- **Glassnode** — $49 tier's "API Light" is 14-day history at daily resolution, 50 calls/day; real API $833/mo.
- **Santiment** — free/Pro tiers carry a **30-day lag** on the metrics that matter.
- **Arkham / Dune** — no live-loop fit (enterprise-priced attribution; batch query warehouse with seconds-to-minutes latency).
- **TradingView CRYPTOCAP** — no official API at any price, unofficial libs violate ToS, and every series is reconstructible from CoinGecko/CMC arithmetic anyway.
- **Velo ($199) / Tardis ($350)** — over budget; Tardis upgrades backtests, not live expectancy.
- **Whale Alert** ($29.95/mo, only budget-priced real-time whale feed) and **DefiLlama stablecoin flows** (free) — ADD_LATER: genuinely available, but whale/stablecoin flow is **not a course concept**; adding non-course factors is how the BNB disaster happened. Revisit only if the committee's flow specialist proves value on course-native data first.
- **On-chain generally** — as expected, ruthlessly skipped: the course trades liquidations, OI, funding, dominance and session timing; none are on-chain metrics.

## Priority and sequencing (honest framing)

More feeds do not create edge — the nominator does (LAUNCH_HANDOFF §7). The
recommended order keeps feed work subordinate to it:

1. **Now, alongside nominator work (all $0):** Binance ratio endpoints +
   Fear & Greed + CoinGecko dominance → these upgrade the *committee's*
   context immediately (its flow/sentiment fields currently show
   `data_quality: degraded/missing`).
2. **Next sprint:** Binance websocket layer (markPrice@1s SL checks first —
   it attacks a measured ~$3k/quarter leak — then aggTrade/CVD, then the
   Bybit liquidation stream).
3. **After the heatmap has a consumer** (nominator or committee flow
   specialist asking "is there a magnet above/below?"): CoinAnk 7-day trial →
   $30/mo if Plan1 confirms.
4. RSS news boolean whenever convenient (S effort, low urgency).

Registry integration pattern is uniform: each item is a provider class in
`mm_data_feeds.py` + fields surfaced through `build_flow_snapshot` — and,
per the audit finding, the flow snapshot must ALSO start feeding the
deterministic scorer, not just the LLM context.

Implementation notes from verification: send a real User-Agent when polling
the RSS feeds (Cointelegraph); CoinDesk's feed 308-redirects on the trailing
slash — follow redirects. Binance docs carry a legacy-websocket-URL migration
notice — use the currently documented base URL (`wss://fstream.binance.com`).

---
*Method note: five domain researchers + one adversarial pricing verifier, all
working against live vendor pages and real requests on 2026-07-31 (curl tests,
a live websocket connection, browser rendering for JS-heavy pricing pages).
Verifier outcome: 9 of 10 headline sources fully CONFIRMED; CoinGecko
confirmed with two minor corrections (paid Basic tier is 300 req/min, and the
keyless rate limit is documented only as "IP-based, shared"). Alternative.me
returned identical data on both independent checks. Prices change — recheck
before subscribing.*
