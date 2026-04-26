"""GATED predictive backtest — v1 quality filters APPLIED before placing limits.

The earlier predictive_backtest.py was catastrophic (-99.96%) because it
placed limits on every swing-low + swing-high pair without any quality
filtering. 25,429 pre-formations is too permissive.

This version replicates v1's gating logic at the pre-formation stage:
  - HTF trend alignment (4H trend must not oppose trade direction)
  - P1 at LOD/LOW (W) or HOD/HOW (M) — within 0.5% of recent extreme
  - Course variant: P1 and ridge in different sessions (multi-session)
  - Direction validity

Then for setups that survive ALL gates, place a single LIMIT at P1 + 0.5%
(W) or P1 - 0.5% (M). Wait for fill or invalidation.

Hypothesis: with proper filtering, predictive should approach v1's
expectancy with tighter SL geometry on fills.

Usage:
    python3 scripts/predictive_backtest_gated.py --symbols BTC,ETH,... --days 90
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.strategy.mm_formations import (  # noqa: E402
    _find_swing_highs, _find_swing_lows,
    SWING_WINDOW, MIN_PEAK_SEPARATION, MAX_PEAK_SEPARATION,
)
from src.strategy.mm_ema_framework import EMAFramework  # noqa: E402

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/klines"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def fetch_tf(client: httpx.AsyncClient, symbol: str, tf: str, days: int) -> pd.DataFrame:
    """Fetch candles with enough warmup for EMA-800 on 4H + general slack."""
    # Need enough warmup: EMA-800 on 4H requires ~134 days of 4H data.
    if tf == "1h":
        warmup_days = days + 7
    elif tf == "4h":
        warmup_days = days + 200
    else:
        warmup_days = days + 5
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=warmup_days)
    rows = []
    cur = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    interval_ms = {"1h": 3600 * 1000, "4h": 4 * 3600 * 1000}[tf]
    while cur < end_ms:
        r = await client.get(
            BINANCE_FAPI,
            params={"symbol": symbol, "interval": tf, "startTime": cur, "endTime": end_ms, "limit": 1500},
            timeout=30.0,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + interval_ms
        if len(batch) < 1500:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume",
                                      "close_time", "_qv", "_tn", "_tbv", "_tqv", "_ign"])
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def session_for_bar(ts: pd.Timestamp) -> str:
    """Map a UTC timestamp to MMM session label (asia / uk / us)."""
    ny_hour = (ts.hour - 5) % 24  # rough EST; fine for session bucketing
    if 17 <= ny_hour or ny_hour < 2:
        return "asia"
    elif 2 <= ny_hour < 9:
        return "uk"
    else:
        return "us"


# ---------------------------------------------------------------------------
# Quality gates (replicating v1's filtering at pre-formation stage)
# ---------------------------------------------------------------------------


def htf_trend_at(df_4h: pd.DataFrame, ts: pd.Timestamp, ema: EMAFramework) -> str:
    """Compute 4H trend at given timestamp. Returns 'bullish' / 'bearish' / 'sideways'."""
    view = df_4h[df_4h.index <= ts]
    if len(view) < max(ema.periods):
        return "sideways"
    state = ema.get_trend_state(view)
    return state.direction


def is_at_key_level(df_1h: pd.DataFrame, p1_idx: int, p1_price: float, formation_type: str,
                    tol_pct: float = 0.5, lookback_bars: int = 24 * 7) -> bool:
    """True if P1 is at LOD/LOW (W) or HOD/HOW (M).

    Uses last 24 1H bars for LOD/HOD and 168 (7 days) for LOW/HOW.
    """
    # LOD/HOD: last 24 bars before P1
    start_lod = max(0, p1_idx - 24)
    window_lod = df_1h.iloc[start_lod:p1_idx + 1]
    # LOW/HOW: last 168 bars before P1
    start_low = max(0, p1_idx - lookback_bars)
    window_low = df_1h.iloc[start_low:p1_idx + 1]
    if window_lod.empty or window_low.empty:
        return False

    if formation_type == "W":
        lod = float(window_lod["low"].min())
        low = float(window_low["low"].min())
        return (
            abs(p1_price - lod) / lod * 100 < tol_pct
            or abs(p1_price - low) / low * 100 < tol_pct
        )
    else:  # M
        hod = float(window_lod["high"].max())
        how = float(window_low["high"].max())
        return (
            abs(p1_price - hod) / hod * 100 < tol_pct
            or abs(p1_price - how) / how * 100 < tol_pct
        )


def is_multi_session(df_1h: pd.DataFrame, p1_idx: int, ridge_idx: int) -> bool:
    """True if P1 and ridge are in different MMM sessions."""
    if p1_idx >= len(df_1h) or ridge_idx >= len(df_1h):
        return False
    s1 = session_for_bar(df_1h.index[p1_idx])
    s2 = session_for_bar(df_1h.index[ridge_idx])
    return s1 != s2


# ---------------------------------------------------------------------------
# Pre-formation detection with gates
# ---------------------------------------------------------------------------


@dataclass
class GatedPreFormation:
    symbol: str
    type: str
    p1_idx: int
    p1_price: float
    ridge_idx: int
    ridge_price: float
    detected_at_idx: int
    detected_at_ts: datetime
    limit_price: float
    sl_price: float
    timeout_at_idx: int
    invalidation_threshold: float
    htf_trend: str
    at_key_level: bool
    multi_session: bool


def detect_gated(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    symbol: str,
    detect_at_idx: int,
    ema: EMAFramework,
    timeout_bars: int = 48,
) -> list[GatedPreFormation]:
    """Detect pre-formations that pass v1 gates."""
    if detect_at_idx < SWING_WINDOW * 4:
        return []
    view = df_1h.iloc[: detect_at_idx + 1]
    if len(view) < SWING_WINDOW * 4:
        return []

    highs = view["high"].values
    lows = view["low"].values
    swing_lows = _find_swing_lows(lows, window=SWING_WINDOW)
    swing_highs = _find_swing_highs(highs, window=SWING_WINDOW)

    cutoff = detect_at_idx - MAX_PEAK_SEPARATION
    swing_lows = [s for s in swing_lows if s >= cutoff]
    swing_highs = [s for s in swing_highs if s >= cutoff]

    candidates: list[GatedPreFormation] = []
    current_close = float(view["close"].iloc[-1])
    current_ts = view.index[-1]
    htf_trend = htf_trend_at(df_4h, current_ts, ema)

    # ---- W candidates ----
    for sl_idx in swing_lows:
        ridges = [sh for sh in swing_highs if sh > sl_idx]
        if not ridges:
            continue
        ridge_idx = ridges[0]
        if (ridge_idx - sl_idx) < MIN_PEAK_SEPARATION:
            continue
        if (detect_at_idx - sl_idx) > MAX_PEAK_SEPARATION:
            continue
        if (detect_at_idx - ridge_idx) > MAX_PEAK_SEPARATION:
            continue
        # No confirmed peak2 yet
        if any(sl > ridge_idx for sl in swing_lows):
            continue
        # Price must be pulling back (below ridge)
        if current_close >= float(highs[ridge_idx]):
            continue

        p1_price = float(lows[sl_idx])
        ridge_price = float(highs[ridge_idx])

        # GATE 1: HTF — 4H trend must NOT be bearish for a long
        if htf_trend == "bearish":
            continue
        # GATE 2: at_key_level — P1 must be at LOD or LOW
        atkl = is_at_key_level(view, sl_idx, p1_price, "W")
        if not atkl:
            continue
        # GATE 3: multi-session — P1 and ridge in different sessions
        multi = is_multi_session(view, sl_idx, ridge_idx)
        if not multi:
            continue

        candidates.append(GatedPreFormation(
            symbol=symbol, type="W",
            p1_idx=sl_idx, p1_price=p1_price,
            ridge_idx=ridge_idx, ridge_price=ridge_price,
            detected_at_idx=detect_at_idx,
            detected_at_ts=current_ts,
            limit_price=p1_price * 1.005,
            sl_price=p1_price * 0.995,
            timeout_at_idx=detect_at_idx + timeout_bars,
            invalidation_threshold=p1_price * 0.99,
            htf_trend=htf_trend,
            at_key_level=atkl,
            multi_session=multi,
        ))

    # ---- M candidates ----
    for sh_idx in swing_highs:
        troughs = [sl for sl in swing_lows if sl > sh_idx]
        if not troughs:
            continue
        trough_idx = troughs[0]
        if (trough_idx - sh_idx) < MIN_PEAK_SEPARATION:
            continue
        if (detect_at_idx - sh_idx) > MAX_PEAK_SEPARATION:
            continue
        if (detect_at_idx - trough_idx) > MAX_PEAK_SEPARATION:
            continue
        if any(sh > trough_idx for sh in swing_highs):
            continue
        if current_close <= float(lows[trough_idx]):
            continue

        p1_price = float(highs[sh_idx])
        trough_price = float(lows[trough_idx])

        if htf_trend == "bullish":
            continue
        atkl = is_at_key_level(view, sh_idx, p1_price, "M")
        if not atkl:
            continue
        multi = is_multi_session(view, sh_idx, trough_idx)
        if not multi:
            continue

        candidates.append(GatedPreFormation(
            symbol=symbol, type="M",
            p1_idx=sh_idx, p1_price=p1_price,
            ridge_idx=trough_idx, ridge_price=trough_price,
            detected_at_idx=detect_at_idx,
            detected_at_ts=current_ts,
            limit_price=p1_price * 0.995,
            sl_price=p1_price * 1.005,
            timeout_at_idx=detect_at_idx + timeout_bars,
            invalidation_threshold=p1_price * 1.01,
            htf_trend=htf_trend,
            at_key_level=atkl,
            multi_session=multi,
        ))

    return candidates


# ---------------------------------------------------------------------------
# Trade simulation (TPs at 1R/2R/3R, 30/40/30 partial split, BE after TP1)
# ---------------------------------------------------------------------------


@dataclass
class GatedTrade:
    symbol: str
    direction: str
    p1_price: float
    detected_at_ts: datetime
    fill_ts: datetime
    fill_price: float
    sl: float
    htf_trend: str
    exit_ts: datetime | None = None
    exit_price: float = 0.0
    exit_reason: str = ""
    r_multiple: float = 0.0
    realized_pnl_usd: float = 0.0
    risk_usd_at_entry: float = 0.0
    balance_at_entry: float = 0.0


def simulate_trade(
    pf: GatedPreFormation, fill_ts: datetime, forward: pd.DataFrame,
    max_hold_hours: int = 24 * 7, fee_per_side: float = 0.0004,
) -> GatedTrade:
    is_long = pf.type == "W"
    entry = pf.limit_price
    sl_initial = pf.sl_price
    sl_distance = abs(entry - sl_initial)
    if sl_distance <= 0:
        return GatedTrade(
            symbol=pf.symbol, direction="long" if is_long else "short",
            p1_price=pf.p1_price, detected_at_ts=pf.detected_at_ts,
            fill_ts=fill_ts, fill_price=entry, sl=sl_initial,
            htf_trend=pf.htf_trend, exit_reason="invalid_risk",
        )
    # TP at R-multiples (1, 2, 3)
    if is_long:
        tp1 = entry + sl_distance
        tp2 = entry + 2 * sl_distance
        tp3 = entry + 3 * sl_distance
    else:
        tp1 = entry - sl_distance
        tp2 = entry - 2 * sl_distance
        tp3 = entry - 3 * sl_distance
    sl_current = sl_initial
    realized = 0.0
    fees = 0.0
    quantity = 1.0
    p1_qty, p2_qty, p3_qty = 0.30, 0.40, 0.30
    remaining = 1.0
    fees += entry * quantity * fee_per_side
    tp1_hit = tp2_hit = tp3_hit = False
    exit_ts = None
    exit_reason = ""
    exit_price = 0.0
    for ts, row in forward.iterrows():
        bar_high = float(row["high"])
        bar_low = float(row["low"])
        bar_close = float(row["close"])
        elapsed_h = (ts - fill_ts).total_seconds() / 3600
        if elapsed_h > max_hold_hours:
            pnl_per_unit = (bar_close - entry) if is_long else (entry - bar_close)
            realized += pnl_per_unit * (quantity * remaining)
            fees += bar_close * quantity * remaining * fee_per_side
            exit_reason, exit_ts, exit_price = "timeout", ts, bar_close
            break
        sl_touched = (bar_low <= sl_current) if is_long else (bar_high >= sl_current)
        if sl_touched:
            fill = sl_current
            pnl_per_unit = (fill - entry) if is_long else (entry - fill)
            realized += pnl_per_unit * (quantity * remaining)
            fees += fill * quantity * remaining * fee_per_side
            exit_reason, exit_ts, exit_price = "sl", ts, fill
            break
        if not tp1_hit:
            hit = (bar_high >= tp1) if is_long else (bar_low <= tp1)
            if hit:
                pnl_per_unit = (tp1 - entry) if is_long else (entry - tp1)
                realized += pnl_per_unit * p1_qty
                fees += tp1 * p1_qty * fee_per_side
                remaining -= p1_qty
                tp1_hit = True
                sl_current = entry * (1 + 2 * fee_per_side) if is_long else entry * (1 - 2 * fee_per_side)
        if tp1_hit and not tp2_hit:
            hit = (bar_high >= tp2) if is_long else (bar_low <= tp2)
            if hit:
                pnl_per_unit = (tp2 - entry) if is_long else (entry - tp2)
                realized += pnl_per_unit * p2_qty
                fees += tp2 * p2_qty * fee_per_side
                remaining -= p2_qty
                tp2_hit = True
        if tp2_hit and not tp3_hit:
            hit = (bar_high >= tp3) if is_long else (bar_low <= tp3)
            if hit:
                pnl_per_unit = (tp3 - entry) if is_long else (entry - tp3)
                realized += pnl_per_unit * p3_qty
                fees += tp3 * p3_qty * fee_per_side
                remaining -= p3_qty
                tp3_hit = True
                exit_reason, exit_ts, exit_price = "tp3", ts, tp3
                break
    if not exit_reason:
        last = forward.iloc[-1] if len(forward) else None
        if last is not None:
            bar_close = float(last["close"])
            pnl_per_unit = (bar_close - entry) if is_long else (entry - bar_close)
            realized += pnl_per_unit * (quantity * remaining)
            fees += bar_close * quantity * remaining * fee_per_side
            exit_reason = "window_end"
            exit_ts = forward.index[-1]
            exit_price = bar_close
    net_per_unit = (realized - fees) / quantity
    r = net_per_unit / sl_distance
    return GatedTrade(
        symbol=pf.symbol, direction="long" if is_long else "short",
        p1_price=pf.p1_price, detected_at_ts=pf.detected_at_ts,
        fill_ts=fill_ts, fill_price=entry, sl=sl_initial,
        htf_trend=pf.htf_trend, exit_ts=exit_ts, exit_price=exit_price,
        exit_reason=exit_reason, r_multiple=r,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run(symbols: list[str], days: int, starting: float, risk_pct: float,
              max_concurrent: int, timeout_bars: int, json_out: str | None) -> dict:
    print(f"GATED predictive backtest: {days}d × {len(symbols)} symbols")
    print(f"Gates: HTF + at_key_level + multi_session, single limit at P1 ± 0.5%\n")

    ema = EMAFramework()
    print("Fetching candles…")
    data: dict[str, dict[str, pd.DataFrame]] = {}
    async with httpx.AsyncClient() as client:
        for sym_human in symbols:
            sym_b = sym_human.replace("/", "").replace(":USDT", "")
            try:
                df_1h = await fetch_tf(client, sym_b, "1h", days)
                df_4h = await fetch_tf(client, sym_b, "4h", days)
                if df_1h.empty or df_4h.empty:
                    continue
                data[sym_human] = {"1h": df_1h, "4h": df_4h}
                print(f"  {sym_human}: 1h={len(df_1h)}, 4h={len(df_4h)}")
            except Exception as e:
                print(f"  {sym_human}: ERROR {e}")

    print("\nRunning gated detection + simulation…")
    pending: dict[str, dict] = {}
    filled: list[GatedTrade] = []
    skipped: list[dict] = []
    balance = starting
    peak = starting
    max_dd = 0.0
    open_trades: list[GatedTrade] = []

    timeline = []
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    bt_start = end - timedelta(days=days)
    for sym, dfs in data.items():
        df_1h = dfs["1h"]
        for i, ts in enumerate(df_1h.index):
            if ts < bt_start:
                continue
            timeline.append((ts, sym, i, df_1h.iloc[i]))
    timeline.sort(key=lambda x: x[0])
    print(f"Timeline: {len(timeline)} bars across {len(data)} symbols\n")

    def close_due(now: datetime) -> None:
        nonlocal balance, peak, max_dd, open_trades
        still_open = []
        for t in open_trades:
            if t.exit_ts is not None and t.exit_ts <= now:
                t.realized_pnl_usd = t.r_multiple * t.risk_usd_at_entry
                balance += t.realized_pnl_usd
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                filled.append(t)
            else:
                still_open.append(t)
        open_trades = still_open

    for bar_ts, sym, i, bar in timeline:
        df_1h = data[sym]["1h"]
        df_4h = data[sym]["4h"]
        close_due(bar_ts)

        # Check pending limit for this symbol
        if sym in pending:
            pf_state = pending[sym]
            pf: GatedPreFormation = pf_state["pf"]
            invalidated = False
            if pf.type == "W" and float(bar["close"]) <= pf.invalidation_threshold:
                invalidated = True
            elif pf.type == "M" and float(bar["close"]) >= pf.invalidation_threshold:
                invalidated = True
            if i > pf.timeout_at_idx:
                invalidated = True
            if invalidated:
                skipped.append({"symbol": sym, "type": pf.type, "reason": "invalid_or_timeout"})
                del pending[sym]
            else:
                bar_high = float(bar["high"])
                bar_low = float(bar["low"])
                touched = (bar_low <= pf.limit_price) if pf.type == "W" else (bar_high >= pf.limit_price)
                if touched:
                    if len(open_trades) >= max_concurrent:
                        skipped.append({"symbol": sym, "type": pf.type, "reason": "capacity"})
                        del pending[sym]
                    else:
                        forward = df_1h.iloc[i + 1:]
                        if len(forward) > 0:
                            t = simulate_trade(pf, bar_ts, forward, max_hold_hours=24 * 7)
                            t.risk_usd_at_entry = balance * risk_pct
                            t.balance_at_entry = balance
                            open_trades.append(t)
                        del pending[sym]

        # Detect new pre-formations (only if no pending for this symbol)
        if sym not in pending:
            new_pfs = detect_gated(df_1h, df_4h, sym, i, ema, timeout_bars)
            if new_pfs:
                pending[sym] = {"pf": new_pfs[-1]}

    final_ts = max(d["1h"].index[-1] for d in data.values())
    close_due(final_ts + timedelta(days=365))

    # Report
    wins = [t for t in filled if t.realized_pnl_usd > 0]
    losses = [t for t in filled if t.realized_pnl_usd < 0]
    total_pnl = sum(t.realized_pnl_usd for t in filled)
    skip_reasons = Counter(s["reason"] for s in skipped)
    return_pct = (balance / starting - 1) * 100
    annualized = ((balance / starting) ** (365 / days) - 1) * 100
    win_rate = (len(wins) / len(filled) * 100) if filled else 0

    print()
    print("=" * 78)
    print("GATED PREDICTIVE BACKTEST RESULT")
    print("=" * 78)
    print(f"Window:               {days} days × {len(symbols)} symbols")
    print(f"Starting balance:     ${starting:>12,.2f}")
    print(f"Final balance:        ${balance:>12,.2f}")
    print(f"Total P&L:            ${total_pnl:>+12,.2f}")
    print(f"Return %:             {return_pct:>+12.2f}%")
    print(f"Annualized:           {annualized:>+12.2f}%")
    print(f"Peak balance:         ${peak:>12,.2f}")
    print(f"Max DD (peak→trough): {max_dd:>12.2f}%")
    print()
    print(f"Pre-formations passing all gates: {len(filled) + len(skipped)}")
    print(f"Trades filled:                    {len(filled)}")
    print(f"Skipped:                          {len(skipped)}  {dict(skip_reasons)}")
    if filled:
        print(f"Win rate:                         {win_rate:>5.1f}%  ({len(wins)}W/{len(losses)}L)")
        avg_win = sum(t.r_multiple for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.r_multiple for t in losses) / len(losses) if losses else 0
        print(f"Avg win R:                        {avg_win:+.2f}R")
        print(f"Avg loss R:                       {avg_loss:+.2f}R")
        # Exit reason distribution
        exit_counter = Counter(t.exit_reason for t in filled)
        print(f"Exit reasons: {dict(exit_counter)}")

    summary = {
        "starting": starting, "final": balance, "pnl": total_pnl,
        "return_pct": return_pct, "annualized": annualized,
        "peak": peak, "max_dd_pct": max_dd,
        "filled": len(filled), "skipped": len(skipped),
        "wins": len(wins), "losses": len(losses), "win_rate": win_rate,
    }
    if json_out:
        out = Path(json_out)
        with out.open("w") as fh:
            json.dump({
                "summary": summary,
                "trades": [{**asdict(t),
                            "exit_ts": t.exit_ts.isoformat() if t.exit_ts else None,
                            "fill_ts": t.fill_ts.isoformat(),
                            "detected_at_ts": t.detected_at_ts.isoformat()}
                           for t in filled],
            }, fh, indent=2, default=str)
        print(f"\nLedger: {out}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--starting-balance", type=float, default=100_000.0)
    ap.add_argument("--risk-pct", type=float, default=0.01)
    ap.add_argument("--max-concurrent", type=int, default=10)
    ap.add_argument("--timeout-bars", type=int, default=48)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()
    syms = [f"{s.strip().upper()}/USDT:USDT" for s in args.symbols.split(",") if s.strip()]
    asyncio.run(run(
        symbols=syms, days=args.days,
        starting=args.starting_balance, risk_pct=args.risk_pct,
        max_concurrent=args.max_concurrent, timeout_bars=args.timeout_bars,
        json_out=args.json_out,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
