"""Historical replay scanner — run the MM engine's deterministic entry
pipeline against past Binance candles.

WHAT THIS DOES
  For a given symbol and window, step through each 1H bar in the window
  and run `MMEngine._analyze_pair()` as if the bot were scanning live at
  that moment. Record which gate each potential setup passed or died at,
  plus the confluence grade and rejection reason. Print a summary.

WHAT IT DOES NOT DO (yet)
  - Does NOT call the sanity agent (would cost real money). Agent is
    disabled via config flag. Deterministic rules only.
  - Does NOT simulate P&L. Exit behaviour / partial closes / SL progression
    aren't replayed — that's Tier 4. This is "what signals would fire?"
  - External data feeds (liquidation, correlation) gracefully score zero
    since we can't fetch their live values retroactively.

USAGE
  Single symbol:
    python3 scripts/replay_scan.py --symbol BNB --days 30
    python3 scripts/replay_scan.py --symbol NEAR --days 14 --hours-step 4

  Multi-symbol batch (scans each in turn, aggregates at the end):
    python3 scripts/replay_scan.py --symbols BTC,ETH,BNB,SOL --days 7

  Config overrides (A/B test rule changes before deploying):
    python3 scripts/replay_scan.py --symbol BNB --days 30 --min-confluence 40
    python3 scripts/replay_scan.py --symbol BTC --days 14 --max-sl-pct 3.0

  Diagnose low grades (show per-factor hit rates across the window):
    python3 scripts/replay_scan.py --symbol BNB --days 30 --factor-rates

OUTPUT
  Per-scan signal/rejection, plus aggregate funnel + (optional) factor
  hit-rates. Lets you answer "would min_confluence=40 have killed any
  of last month's signals?" and "which factors never fire on BTC?"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

# Make src importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from types import SimpleNamespace

from src.strategy.mm_engine import MMEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Historical candle fetch (Binance public REST)
# ---------------------------------------------------------------------------

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/klines"

TF_TO_BINANCE = {
    "15m": "15m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}


async def _fetch_klines(
    client: httpx.AsyncClient,
    symbol: str,
    tf: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """Fetch klines across a range, paginating through 1500-bar limit."""
    all_rows = []
    cur = start_ms
    while cur < end_ms:
        r = await client.get(
            BINANCE_FAPI,
            params={
                "symbol": symbol,
                "interval": TF_TO_BINANCE[tf],
                "startTime": cur,
                "endTime": end_ms,
                "limit": 1500,
            },
            timeout=30.0,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_rows.extend(batch)
        last_open = batch[-1][0]
        step_ms = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}[tf] * 1000
        cur = last_open + step_ms
        if len(batch) < 1500:
            break
    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        all_rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "_qv", "_tn", "_tbv", "_tqv", "_ign",
        ],
    )
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


async def fetch_history(symbol_spot: str, start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    """Fetch 1H / 4H / 1D / 15m candles covering [start, end] with warmup."""
    # Warmup buffers so EMA-200 etc. have data
    warmup_by_tf = {
        "1h":  timedelta(days=22),   # 500 bars warmup
        "4h":  timedelta(days=50),   # 300 bars warmup
        "1d":  timedelta(days=260),  # 260 bars warmup (EMA-200 on 1D)
        "15m": timedelta(days=3),    # 300 bars warmup
    }
    binance_symbol = symbol_spot.replace("/", "").replace(":USDT", "")
    out: dict[str, pd.DataFrame] = {}
    async with httpx.AsyncClient() as client:
        for tf, warmup in warmup_by_tf.items():
            s = int((start - warmup).timestamp() * 1000)
            e = int(end.timestamp() * 1000)
            df = await _fetch_klines(client, binance_symbol, tf, s, e)
            out[tf] = df
            print(f"  fetched {len(df):>4} {tf:>3} bars for {binance_symbol} "
                  f"({df.index[0] if not df.empty else '—'} → "
                  f"{df.index[-1] if not df.empty else '—'})")
    return out


# ---------------------------------------------------------------------------
# Stub managers so the engine runs without real IO
# ---------------------------------------------------------------------------

class ReplayCandleManager:
    """Candle manager that serves pre-fetched DFs sliced to `as_of`."""
    def __init__(self, history: dict[str, pd.DataFrame]) -> None:
        self.history = history
        self.as_of: datetime | None = None

    async def get_candles(self, symbol: str, tf: str, limit: int = 500) -> pd.DataFrame:
        df = self.history.get(tf)
        if df is None or df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        if self.as_of is not None:
            # Include bars whose open_time is strictly before `as_of`.
            # This mirrors live behaviour — you can't see a bar that hasn't
            # opened yet.
            df = df[df.index < self.as_of]
        return df.tail(limit).copy()


class ReplayExchange:
    """Minimal async exchange stub — only fetch_ticker + get_balance used
    on the scan path. The engine's scan flow doesn't actually place any
    orders in replay because we never get to _enter_trade (signals are
    collected from _analyze_pair return values)."""
    def __init__(self, mgr: ReplayCandleManager, balance_usd: float = 100_000.0) -> None:
        self.mgr = mgr
        self.balance_usd = balance_usd

    async def fetch_ticker(self, symbol: str) -> dict:
        df = self.mgr.history.get("1h")
        if df is None or df.empty:
            return {"last": 0.0}
        if self.mgr.as_of is not None:
            df = df[df.index < self.mgr.as_of]
        if df.empty:
            return {"last": 0.0}
        return {"last": float(df.iloc[-1]["close"])}

    async def get_balance(self) -> dict:
        return {"USDT": self.balance_usd}

    async def place_market_order(self, **kwargs):
        # Never called on scan path in replay
        raise NotImplementedError("replay mode")


class ReplayRepo:
    """No-op repository — all writes silently dropped, all reads return
    empty. The engine's scan path reads:
      - get_cached_candles (via candle_manager, which we replace)
      - get_open_trades (empty — we're simulating from nothing)
      - get_recent_trades_for_symbol (empty — no trade history)
    """
    instance_id = "replay"

    async def get_cached_candles(self, *a, **kw): return []
    async def upsert_candles(self, *a, **kw): pass
    async def get_open_trades(self, *a, **kw): return []
    async def get_recent_trades_for_symbol(self, *a, **kw): return []
    async def get_recent_trades(self, *a, **kw): return []
    async def insert_trade(self, *a, **kw): return {}
    async def update_trade(self, *a, **kw): return {}
    async def insert_signal(self, *a, **kw): return {}
    async def insert_snapshot(self, *a, **kw): return {}
    async def log_api_usage(self, *a, **kw): pass
    async def insert_mm_agent_decision(self, *a, **kw): return {}
    async def get_mm_agent_month_cost(self): return 0.0
    async def get_engine_state(self): return None
    async def upsert_engine_state(self, *a, **kw): return {}
    async def link_signal_to_trade(self, *a, **kw): pass
    async def get_signal_by_symbol_recent(self, *a, **kw): return None
    async def update_signal_components(self, *a, **kw): pass


def _replay_config(**overrides) -> SimpleNamespace:
    """A config stub with the same defaults as production but with the
    sanity agent disabled (we do not call the live API during replay).

    ``overrides`` lets the caller supply any config override that the
    engine honours (mm_min_confluence, mm_max_sl_pct, etc.) for A/B
    testing. Unknown keys are silently ignored — config is permissive.
    """
    defaults = dict(
        instance_id="replay",
        trading_mode="paper",
        initial_balance=100_000.0,
        mm_method_enabled=True,
        mm_scan_interval_minutes=5.0,
        mm_max_positions=20,
        mm_risk_per_trade_pct=1.0,
        mm_max_aggregate_risk_pct=5.0,
        mm_initial_balance=100_000.0,
        mm_min_volume_usd=50_000_000,
        mm_majors_only=True,
        # Agent OFF — no API calls during replay
        anthropic_api_key="",
        mm_sanity_agent_enabled=False,
        mm_sanity_agent_model="claude-opus-4-7",
        mm_sanity_agent_fallback_model="claude-sonnet-4-6",
        mm_sanity_agent_timeout_s=20.0,
        mm_sanity_agent_effort="high",
        mm_sanity_agent_min_confidence=0.0,
        mm_sanity_agent_monthly_budget_usd=600.0,
        markets={},
        leverage=10,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

@dataclass
class BarResult:
    ts: datetime
    stage: str                 # deepest stage reached
    signal: dict | None = None # non-None if signal produced
    rejects: dict[str, int] = field(default_factory=dict)


def _snapshot_stage_counts(engine: MMEngine) -> dict[str, int]:
    return dict(engine._scan_stage_counts)


def _snapshot_rejects(engine: MMEngine) -> dict[str, int]:
    return dict(engine._scan_reject_counts)


def _deepest_stage(after: dict[str, int], before: dict[str, int]) -> str:
    """Find the latest stage whose counter incremented between before/after."""
    stage_order = [
        "candles_ok", "formation_found", "htf_aligned", "level_ok",
        "phase_valid", "direction_ok", "target_acquired", "rr_passed",
        "scored", "confluence_passed", "retest_passed",
        "sanity_agent_passed", "signal_built",
    ]
    deepest = "none"
    for s in stage_order:
        if after.get(s, 0) > before.get(s, 0):
            deepest = s
    return deepest


def _which_reject(after: dict[str, int], before: dict[str, int]) -> str | None:
    for k, v in after.items():
        if v > before.get(k, 0):
            return k
    return None


@dataclass
class SymbolResult:
    """Aggregated result for one symbol over the replay window."""
    symbol: str
    bars: list[BarResult] = field(default_factory=list)
    factor_hits: Counter = field(default_factory=Counter)  # factor_name → count
    score_samples: list[float] = field(default_factory=list)
    grade_counts: Counter = field(default_factory=Counter)


async def replay_single_symbol(
    symbol: str,
    days: int,
    hours_step: int,
    engine_overrides: dict,
) -> SymbolResult:
    """Replay one symbol over the window. Returns aggregated results."""
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    print(f"\n[{symbol}] fetching candles…")
    history = await fetch_history(symbol, start, end)
    if any(v.empty for v in history.values()):
        print(f"[{symbol}] ERROR: missing candles for at least one timeframe.")
        return SymbolResult(symbol=symbol)

    mgr = ReplayCandleManager(history)
    exch = ReplayExchange(mgr)
    repo = ReplayRepo()
    config = _replay_config()

    engine = MMEngine(
        exchange=exch, repo=repo, candle_manager=mgr,
        config=config, scan_interval_minutes=5.0,
    )

    # Apply engine-level overrides (min_confluence, max_sl_pct) AFTER
    # construction — these are instance attributes on the engine, set
    # from constants in __init__ and normally only updated via the
    # engine_state.config_overrides DB path.
    for k, v in engine_overrides.items():
        if hasattr(engine, k):
            setattr(engine, k, v)
            print(f"[{symbol}] override: engine.{k} = {v}")

    bars_1h = history["1h"]
    bars_1h = bars_1h[(bars_1h.index >= start) & (bars_1h.index < end)]

    step = max(1, hours_step)
    result = SymbolResult(symbol=symbol)

    print(f"[{symbol}] stepping through {len(bars_1h)} 1H bars, every {step}h…")
    for ts in bars_1h.index[::step]:
        as_of = ts + timedelta(hours=1)
        mgr.as_of = as_of

        engine._scan_reject_counts = {}
        engine._scan_stage_counts = {}
        engine._scan_factor_hits = {}
        engine._scan_score_samples = []
        engine._scan_grade_counts = {}

        session = engine.session_analyzer.get_current_session(as_of)

        try:
            signal = await engine._analyze_pair(symbol, session, as_of)
        except Exception as e:
            result.bars.append(BarResult(ts=as_of,
                                         stage=f"EXCEPTION:{type(e).__name__}"))
            continue

        # Aggregate factor hits + score samples for the full window.
        for k, v in engine._scan_factor_hits.items():
            result.factor_hits[k] += v
        result.score_samples.extend(engine._scan_score_samples)
        for g, n in engine._scan_grade_counts.items():
            result.grade_counts[g] += n

        stages = dict(engine._scan_stage_counts)
        rejects = dict(engine._scan_reject_counts)
        if signal:
            result.bars.append(BarResult(
                ts=as_of, stage="signal_built",
                signal={
                    "direction": signal.direction,
                    "entry": round(signal.entry_price, 6),
                    "sl": round(signal.stop_loss, 6),
                    "tp1": round(signal.target_l1, 6),
                    "tp2": round(signal.target_l2, 6),
                    "tp3": round(signal.target_l3, 6),
                    "rr": round(signal.risk_reward, 2),
                    "grade": signal.confluence_grade,
                    "score_pct": round(signal.confluence_score, 1),
                    "variant": signal.formation_variant,
                    "entry_type": signal.entry_type,
                    "reason": signal.reason,
                },
            ))
        else:
            stage = _deepest_stage(stages, {})
            reject = _which_reject(rejects, {})
            result.bars.append(BarResult(
                ts=as_of, stage=stage,
                rejects={reject: 1} if reject else {},
            ))
    return result


async def replay(
    symbols: list[str],
    days: int,
    hours_step: int,
    show_rejects: bool,
    factor_rates: bool,
    engine_overrides: dict,
) -> int:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    print(f"Replay window: {start.isoformat()} → {end.isoformat()} ({days} days)")
    print(f"Symbols: {', '.join(symbols)}   step: {hours_step}h")
    if engine_overrides:
        print(f"Overrides: {engine_overrides}")

    symbol_results: list[SymbolResult] = []
    for sym in symbols:
        r = await replay_single_symbol(sym, days, hours_step, engine_overrides)
        symbol_results.append(r)

    # Use the first symbol's results to drive the per-symbol summary,
    # then render a cross-symbol summary if multiple symbols.
    for sr in symbol_results:
        _render_symbol_summary(sr, days, show_rejects, factor_rates)

    if len(symbol_results) > 1:
        _render_cross_symbol_summary(symbol_results, days)

    return 0


def _render_symbol_summary(
    sr: SymbolResult,
    days: int,
    show_rejects: bool,
    factor_rates: bool,
) -> None:
    """Per-symbol summary block. Called once per symbol in the batch."""
    results = sr.bars
    if not results:
        print(f"\n[{sr.symbol}] no results (fetch failure).")
        return

    # Local assignments so the rest of this function reads like the
    # single-symbol flow we had before.
    reject_counts: Counter = Counter()
    for r in results:
        for k in r.rejects:
            reject_counts[k] += 1
    stage_counts = Counter(r.stage for r in results)
    signals = [r for r in results if r.signal]

    # --- Summary block ---
    print("\n" + "=" * 72)
    print(f"REPLAY SUMMARY — {sr.symbol} × {days}d")
    print("=" * 72)

    print(f"\nTotal bars scanned: {len(results)}")
    print(f"Would-have-been signals: {len(signals)}")
    print(f"Exceptions: {sum(1 for r in results if r.stage.startswith('EXCEPTION'))}")

    print("\nDeepest stage distribution:")
    stage_order = [
        "signal_built", "sanity_agent_passed", "retest_passed",
        "confluence_passed", "scored", "rr_passed", "target_acquired",
        "direction_ok", "phase_valid", "level_ok", "htf_aligned",
        "formation_found", "candles_ok", "none",
    ]
    for s in stage_order:
        n = stage_counts.get(s, 0)
        if n > 0:
            print(f"  {s:<24} {n}")
    for s, n in stage_counts.items():
        if s not in stage_order and not s.startswith("EXCEPTION"):
            print(f"  {s:<24} {n}")

    if reject_counts:
        print("\nReject reasons (top 10):")
        for reason, n in reject_counts.most_common(10):
            print(f"  {reason:<30} {n}")

    # Score distribution (when the formation scored)
    if sr.score_samples:
        scores = sorted(sr.score_samples)
        n = len(scores)
        print(f"\nConfluence score distribution (over {n} scored bars):")
        print(f"  min:    {scores[0]:.1f}%")
        print(f"  p25:    {scores[n // 4]:.1f}%")
        print(f"  median: {scores[n // 2]:.1f}%")
        print(f"  p75:    {scores[3 * n // 4]:.1f}%")
        print(f"  max:    {scores[-1]:.1f}%")
        if sr.grade_counts:
            gd = ", ".join(f"{g}={n}" for g, n in sorted(sr.grade_counts.items()))
            print(f"  grades: {gd}")

    # Factor hit rates — the diagnostic for "why are grades low?"
    if factor_rates and sr.factor_hits:
        total_scored = sum(sr.grade_counts.values()) or 1
        print(f"\nFactor hit rate (% of scored bars — {total_scored} total):")
        print(f"  {'factor':<30} {'hits':>5} {'rate':>7}")
        for factor, n in sorted(sr.factor_hits.items(), key=lambda kv: -kv[1]):
            rate = n / total_scored * 100
            print(f"  {factor:<30} {n:>5} {rate:>6.1f}%")

    if signals:
        print(f"\nSignals ({len(signals)}):")
        print(f"  {'timestamp':<20} {'dir':<6} {'grade':<6} "
              f"{'score':<7} {'rr':<5} {'variant':<18} {'entry_type'}")
        for r in signals:
            sig = r.signal
            print(
                f"  {r.ts.strftime('%Y-%m-%d %H:%M'):<20} "
                f"{sig['direction']:<6} {sig['grade']:<6} "
                f"{sig['score_pct']:<7} {sig['rr']:<5} "
                f"{sig['variant']:<18} {sig['entry_type']}"
            )
            print(f"    entry={sig['entry']}  sl={sig['sl']}  tp1={sig['tp1']}")

    if show_rejects:
        print("\nBar-by-bar detail (non-trivial rejections only):")
        for r in results:
            if r.stage in ("none", "candles_ok", "formation_found"):
                continue
            if r.signal:
                continue
            rej = ",".join(r.rejects) if r.rejects else "—"
            print(f"  {r.ts.strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{r.stage:<24} {rej}")


def _render_cross_symbol_summary(
    symbol_results: list[SymbolResult],
    days: int,
) -> None:
    """Cross-symbol aggregate — signals-per-symbol and common-factor hit rates."""
    print("\n" + "=" * 72)
    print(f"CROSS-SYMBOL SUMMARY — {len(symbol_results)} symbols × {days}d")
    print("=" * 72)

    print(f"\n  {'symbol':<18} {'scans':>6} {'signals':>8} {'median_score':>13}")
    for sr in symbol_results:
        scans = len(sr.bars)
        signals = sum(1 for b in sr.bars if b.signal)
        med = (
            sorted(sr.score_samples)[len(sr.score_samples) // 2]
            if sr.score_samples else 0.0
        )
        print(f"  {sr.symbol:<18} {scans:>6} {signals:>8} {med:>12.1f}%")

    # Aggregate factor hits across all symbols
    combined = Counter()
    total_scored = 0
    for sr in symbol_results:
        for k, v in sr.factor_hits.items():
            combined[k] += v
        total_scored += sum(sr.grade_counts.values())
    if combined and total_scored:
        print(f"\nCross-symbol factor hit rate ({total_scored} scored bars total):")
        print(f"  {'factor':<30} {'hits':>5} {'rate':>7}")
        for factor, n in sorted(combined.items(), key=lambda kv: -kv[1]):
            rate = n / total_scored * 100
            print(f"  {factor:<30} {n:>5} {rate:>6.1f}%")


def _normalise_symbol(s: str) -> str:
    s = s.strip().upper()
    if not s.endswith(":USDT"):
        if "/" not in s:
            s = f"{s}/USDT:USDT"
        elif not s.endswith(":USDT"):
            s = f"{s}:USDT"
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sym_group = ap.add_mutually_exclusive_group(required=True)
    sym_group.add_argument("--symbol",
                           help="Single symbol (BNB / BNB/USDT / BNB/USDT:USDT)")
    sym_group.add_argument("--symbols",
                           help="Comma-separated list, e.g. BTC,ETH,BNB,SOL")
    ap.add_argument("--days", type=int, default=30,
                    help="Days of history to replay (default: 30)")
    ap.add_argument("--hours-step", type=int, default=1,
                    help="Scan every N hours (default: 1 — every 1H bar)")
    ap.add_argument("--show-rejects", action="store_true",
                    help="Print bar-by-bar rejection detail")
    ap.add_argument("--factor-rates", action="store_true",
                    help="Print per-factor hit rates (diagnostic)")
    # Engine-level config overrides for A/B testing rule changes
    ap.add_argument("--min-confluence", type=float, default=None,
                    help="Override engine.min_confluence threshold (default 35.0)")
    ap.add_argument("--max-sl-pct", type=float, default=None,
                    help="Override engine.max_sl_pct warning threshold (default 5.0)")
    ap.add_argument("--min-rr", type=float, default=None,
                    help="Override engine.min_rr threshold")
    args = ap.parse_args()

    if args.symbol:
        symbols = [_normalise_symbol(args.symbol)]
    else:
        symbols = [_normalise_symbol(s) for s in args.symbols.split(",") if s.strip()]

    engine_overrides: dict = {}
    if args.min_confluence is not None:
        engine_overrides["min_confluence"] = args.min_confluence
    if args.max_sl_pct is not None:
        engine_overrides["max_sl_pct"] = args.max_sl_pct
    if args.min_rr is not None:
        engine_overrides["min_rr"] = args.min_rr

    return asyncio.run(replay(
        symbols=symbols,
        days=args.days,
        hours_step=args.hours_step,
        show_rejects=args.show_rejects,
        factor_rates=args.factor_rates,
        engine_overrides=engine_overrides,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
