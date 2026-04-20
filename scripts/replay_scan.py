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
  python3 scripts/replay_scan.py --symbol BNB --days 30
  python3 scripts/replay_scan.py --symbol NEAR --days 14 --hours-step 4
  python3 scripts/replay_scan.py --symbol BTC --days 7 --show-rejects

OUTPUT
  Table per signal / rejection, plus aggregate funnel stats at the end.
  Shows the breakdown of where setups died — lets you see at a glance
  "would min_confluence=40 have killed 5 of these 12 signals?"
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


def _replay_config() -> SimpleNamespace:
    """A config stub with the same defaults as production but with the
    sanity agent disabled (we do not call the live API during replay)."""
    return SimpleNamespace(
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


async def replay(symbol: str, days: int, hours_step: int,
                 show_rejects: bool) -> int:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    print(f"Replay window: {start.isoformat()} → {end.isoformat()} ({days} days)")
    print(f"Symbol: {symbol}   step: {hours_step}h")
    print("\nFetching historical candles from Binance…")
    history = await fetch_history(symbol, start, end)
    if any(v.empty for v in history.values()):
        print("ERROR: missing candles for at least one timeframe.")
        return 1

    mgr = ReplayCandleManager(history)
    exch = ReplayExchange(mgr)
    repo = ReplayRepo()
    config = _replay_config()

    engine = MMEngine(
        exchange=exch, repo=repo, candle_manager=mgr,
        config=config, scan_interval_minutes=5.0,
    )

    # Walk 1H bars in the window
    bars_1h = history["1h"]
    bars_1h = bars_1h[(bars_1h.index >= start) & (bars_1h.index < end)]

    results: list[BarResult] = []
    step = max(1, hours_step)

    print(f"\nStepping through {len(bars_1h)} 1H bars, every {step}h…")
    for i, ts in enumerate(bars_1h.index[::step]):
        # "as of" is the END of this bar — the bot would have scanned at close
        as_of = ts + timedelta(hours=1)
        mgr.as_of = as_of

        # Reset per-scan counters and step the engine
        engine._scan_reject_counts = {}
        engine._scan_stage_counts = {}

        session = engine.session_analyzer.get_current_session(as_of)

        try:
            signal = await engine._analyze_pair(symbol, session, as_of)
        except Exception as e:
            results.append(BarResult(ts=as_of, stage=f"EXCEPTION:{type(e).__name__}"))
            continue

        stages = _snapshot_stage_counts(engine)
        rejects = _snapshot_rejects(engine)
        if signal:
            results.append(BarResult(
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
            results.append(BarResult(ts=as_of, stage=stage,
                                     rejects={reject: 1} if reject else {}))

    # --- Summary ---
    print("\n" + "=" * 72)
    print(f"REPLAY SUMMARY — {symbol} × {days}d")
    print("=" * 72)

    stage_counts = Counter(r.stage for r in results)
    reject_counts = Counter()
    for r in results:
        for k in r.rejects:
            reject_counts[k] += 1

    signals = [r for r in results if r.signal]
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
                continue  # noise
            if r.signal:
                continue
            rej = ",".join(r.rejects) if r.rejects else "—"
            print(f"  {r.ts.strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{r.stage:<24} {rej}")

    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", required=True,
                    help="Symbol (accepts BNB, BNB/USDT, BNB/USDT:USDT)")
    ap.add_argument("--days", type=int, default=30,
                    help="Days of history to replay (default: 30)")
    ap.add_argument("--hours-step", type=int, default=1,
                    help="Scan every N hours (default: 1 — every 1H bar)")
    ap.add_argument("--show-rejects", action="store_true",
                    help="Print bar-by-bar rejection detail")
    args = ap.parse_args()

    # Normalise symbol to the form the engine expects
    sym = args.symbol.upper()
    if not sym.endswith(":USDT"):
        if "/" not in sym:
            sym = f"{sym}/USDT:USDT"
        elif not sym.endswith(":USDT"):
            sym = f"{sym}:USDT"

    return asyncio.run(replay(
        symbol=sym,
        days=args.days,
        hours_step=args.hours_step,
        show_rejects=args.show_rejects,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
