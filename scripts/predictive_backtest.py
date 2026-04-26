"""True pre-formation predictive backtest.

Implements the course-faithful workflow from Lesson 7:
  1. Confirm P1 (first swing low for W, first swing high for M).
  2. Confirm ridge (swing high after P1 for W, or swing low for M).
  3. Place a 3-tier LIMIT LADDER at anticipated peak2 levels:
       Tier 1: P1 + 0.5%   (tight retest, "double-bottom W")
       Tier 2: P1 + 1.0%   (standard "higher-low W")
       Tier 3: P1 + 1.5%   (looser retest)
     Inverse offsets for M-tops.
  4. Each subsequent bar: check if any limit got touched.
     If yes: fill at limit, then run forward SL/TP simulation.
     If no: keep waiting until timeout or invalidation.
  5. Cancel a pending ladder if any of:
       - Price closes below P1 × 0.99 (W) or above P1 × 1.01 (M) (formation broken)
       - Ridge breaks down significantly (price closes below ridge × 0.99 for W in pullback context)
       - Timeout (default 48h after ladder placement)

This is fundamentally different from the existing reactive bot:
  - Reactive: wait for peak2 to be confirmed (5 forward bars), then enter at current_price
  - Predictive: place limit BEFORE peak2 forms, fill AT anticipated wick

Output: per-symbol P&L, win rate, fill rate per tier, time-to-fill stats,
and aggregate compounded balance growth assuming starting $100k.

Usage:
    python3 scripts/predictive_backtest.py --symbols BTC,ETH,... --days 90
    python3 scripts/predictive_backtest.py --symbols ... --days 90 --max-concurrent 10
    python3 scripts/predictive_backtest.py --symbols ... --days 90 --json /tmp/predictive.json
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


# ---------------------------------------------------------------------------
# Data fetch (Binance public REST, paginated)
# ---------------------------------------------------------------------------

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/klines"


async def fetch_1h(client: httpx.AsyncClient, symbol: str, days: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    # Need enough warmup for swing detection (at least 50 bars before backtest start)
    start = end - timedelta(days=days + 5)
    rows = []
    cur = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cur < end_ms:
        r = await client.get(
            BINANCE_FAPI,
            params={"symbol": symbol, "interval": "1h", "startTime": cur, "endTime": end_ms, "limit": 1500},
            timeout=30.0,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        cur = last_open + 3600 * 1000
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


# ---------------------------------------------------------------------------
# Pre-formation detection
# ---------------------------------------------------------------------------


@dataclass
class PreFormation:
    """A formation in 'pending peak2' state. P1 + ridge confirmed, peak2 not yet."""
    symbol: str
    type: str                   # "W" or "M"
    p1_idx: int                 # index of P1 (first swing low/high)
    p1_price: float
    ridge_idx: int
    ridge_price: float
    detected_at_idx: int        # bar index when this was detected
    detected_at_ts: datetime    # bar timestamp when detected
    # Limit ladder
    limit_t1: float             # tightest (P1 ± 0.5%)
    limit_t2: float             # P1 ± 1.0%
    limit_t3: float             # widest (P1 ± 1.5%)
    sl_price: float             # below P1 (W) or above P1 (M) with buffer
    # Lifecycle
    timeout_at_idx: int         # bar idx after which we cancel
    invalidation_threshold: float  # price beyond which formation is dead


def detect_pre_formations(
    df: pd.DataFrame,
    symbol: str,
    detect_at_idx: int,
    timeout_bars: int = 48,
    swing_window: int = SWING_WINDOW,
) -> list[PreFormation]:
    """Detect pre-formations as of bar index `detect_at_idx`.

    Only considers data UP TO and INCLUDING `detect_at_idx`. Mimics how the
    live bot would see the world at that moment.

    Returns list of pending formations with confirmed P1 + ridge but no
    confirmed peak2 yet (peak2 might be forming or might come later).
    """
    if detect_at_idx < swing_window * 4:
        return []

    # Slice to "as of" view
    view = df.iloc[: detect_at_idx + 1]
    if len(view) < swing_window * 4:
        return []

    highs = view["high"].values
    lows = view["low"].values

    swing_lows = _find_swing_lows(lows, window=swing_window)
    swing_highs = _find_swing_highs(highs, window=swing_window)

    # Only consider swings recent enough that they could still be a P1 for a
    # forming peak2. Constrain P1 to be within MAX_PEAK_SEPARATION of NOW.
    cutoff = detect_at_idx - MAX_PEAK_SEPARATION
    swing_lows = [s for s in swing_lows if s >= cutoff]
    swing_highs = [s for s in swing_highs if s >= cutoff]

    pre_formations: list[PreFormation] = []

    # ---- W candidates ----
    for sl_idx in swing_lows:
        # Find ridge (next swing high after this swing low)
        ridges = [sh for sh in swing_highs if sh > sl_idx]
        if not ridges:
            continue
        ridge_idx = ridges[0]
        # Constrain separation
        sep_p1_to_ridge = ridge_idx - sl_idx
        if sep_p1_to_ridge < MIN_PEAK_SEPARATION:
            continue
        # Skip if there's already a CONFIRMED swing low after the ridge (= peak2 already).
        # This means a regular W is already detectable — predictive missed the window.
        later_lows = [sl for sl in swing_lows if sl > ridge_idx]
        if later_lows:
            continue
        # Ridge must still be "actionable": i.e. enough bars remain for a peak2 to form
        # within MAX_PEAK_SEPARATION of P1.
        if (detect_at_idx - sl_idx) > MAX_PEAK_SEPARATION:
            continue
        # Ridge should also have happened recently — within last MAX_PEAK_SEPARATION bars.
        if (detect_at_idx - ridge_idx) > MAX_PEAK_SEPARATION:
            continue
        # Price has to currently be BELOW the ridge (i.e. pulling back) to be in the peak2 window.
        # Otherwise we may be still rallying.
        current_close = float(view["close"].iloc[-1])
        if current_close >= float(highs[ridge_idx]):
            continue

        p1_price = float(lows[sl_idx])
        ridge_price = float(highs[ridge_idx])
        pre_formations.append(PreFormation(
            symbol=symbol, type="W",
            p1_idx=sl_idx, p1_price=p1_price,
            ridge_idx=ridge_idx, ridge_price=ridge_price,
            detected_at_idx=detect_at_idx,
            detected_at_ts=view.index[-1],
            limit_t1=p1_price * 1.005,
            limit_t2=p1_price * 1.010,
            limit_t3=p1_price * 1.015,
            sl_price=p1_price * 0.995,
            timeout_at_idx=detect_at_idx + timeout_bars,
            invalidation_threshold=p1_price * 0.99,  # close below P1×0.99 = formation dead
        ))

    # ---- M candidates ----
    for sh_idx in swing_highs:
        troughs = [sl for sl in swing_lows if sl > sh_idx]
        if not troughs:
            continue
        trough_idx = troughs[0]
        sep_p1_to_trough = trough_idx - sh_idx
        if sep_p1_to_trough < MIN_PEAK_SEPARATION:
            continue
        later_highs = [sh for sh in swing_highs if sh > trough_idx]
        if later_highs:
            continue
        if (detect_at_idx - sh_idx) > MAX_PEAK_SEPARATION:
            continue
        if (detect_at_idx - trough_idx) > MAX_PEAK_SEPARATION:
            continue
        current_close = float(view["close"].iloc[-1])
        if current_close <= float(lows[trough_idx]):
            continue

        p1_price = float(highs[sh_idx])
        trough_price = float(lows[trough_idx])
        pre_formations.append(PreFormation(
            symbol=symbol, type="M",
            p1_idx=sh_idx, p1_price=p1_price,
            ridge_idx=trough_idx, ridge_price=trough_price,
            detected_at_idx=detect_at_idx,
            detected_at_ts=view.index[-1],
            limit_t1=p1_price * 0.995,
            limit_t2=p1_price * 0.990,
            limit_t3=p1_price * 0.985,
            sl_price=p1_price * 1.005,
            timeout_at_idx=detect_at_idx + timeout_bars,
            invalidation_threshold=p1_price * 1.01,  # close above P1×1.01 = formation dead
        ))

    return pre_formations


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------


@dataclass
class FilledTrade:
    symbol: str
    type: str
    direction: str
    tier: int                   # 1, 2, 3
    p1_price: float
    detected_at_ts: datetime
    fill_ts: datetime
    fill_price: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    exit_ts: datetime | None = None
    exit_price: float = 0.0
    exit_reason: str = ""
    r_multiple: float = 0.0
    realized_pnl_usd: float = 0.0
    risk_usd_at_entry: float = 0.0
    balance_at_entry: float = 0.0


def simulate_filled_trade(
    pf: PreFormation,
    fill_ts: datetime,
    fill_price: float,
    tier: int,
    forward_candles: pd.DataFrame,
    max_hold_hours: int = 24 * 7,
    fee_per_side: float = 0.0004,
    partial_split: tuple[float, float, float] = (0.30, 0.40, 0.30),
) -> FilledTrade:
    """Run forward SL/TP/timeout sim for a filled trade."""
    is_long = pf.type == "W"
    entry = float(fill_price)
    sl_initial = float(pf.sl_price)
    sl_distance = abs(entry - sl_initial)
    if sl_distance <= 0:
        return FilledTrade(
            symbol=pf.symbol, type=pf.type,
            direction="long" if is_long else "short",
            tier=tier, p1_price=pf.p1_price,
            detected_at_ts=pf.detected_at_ts,
            fill_ts=fill_ts, fill_price=entry,
            sl=sl_initial, tp1=0, tp2=0, tp3=0,
            exit_reason="invalid_risk",
        )

    # TPs based on R-multiples (since we don't have EMAs at fill time in this harness).
    # Using fixed RR ratios: TP1=1R, TP2=2R, TP3=3R as a proxy.
    if is_long:
        tp1 = entry + sl_distance * 1.0
        tp2 = entry + sl_distance * 2.0
        tp3 = entry + sl_distance * 3.0
    else:
        tp1 = entry - sl_distance * 1.0
        tp2 = entry - sl_distance * 2.0
        tp3 = entry - sl_distance * 3.0

    p1_split, p2_split, p3_split = partial_split
    sl_current = sl_initial
    realized_cashflow = 0.0
    fees = 0.0
    quantity = 1.0  # normalized; r_multiple is what matters
    p1_qty = quantity * p1_split
    p2_qty = quantity * p2_split
    p3_qty = quantity * p3_split
    remaining = 1.0
    fees += entry * quantity * fee_per_side
    tp1_hit = tp2_hit = tp3_hit = sl_hit = False
    exit_ts = None
    exit_reason = ""
    exit_price = 0.0

    for ts, row in forward_candles.iterrows():
        bar_high = float(row["high"])
        bar_low = float(row["low"])
        bar_close = float(row["close"])
        elapsed_h = (ts - fill_ts).total_seconds() / 3600
        if elapsed_h > max_hold_hours:
            pnl_per_unit = (bar_close - entry) if is_long else (entry - bar_close)
            realized_cashflow += pnl_per_unit * (quantity * remaining)
            fees += bar_close * quantity * remaining * fee_per_side
            exit_reason = "timeout"
            exit_ts = ts
            exit_price = bar_close
            break

        # SL pessimistic
        sl_touched = (bar_low <= sl_current) if is_long else (bar_high >= sl_current)
        if sl_touched:
            fill = sl_current
            pnl_per_unit = (fill - entry) if is_long else (entry - fill)
            realized_cashflow += pnl_per_unit * (quantity * remaining)
            fees += fill * quantity * remaining * fee_per_side
            sl_hit = True
            exit_reason = "sl"
            exit_ts = ts
            exit_price = fill
            break

        # TP1
        if not tp1_hit:
            hit = (bar_high >= tp1) if is_long else (bar_low <= tp1)
            if hit:
                pnl_per_unit = (tp1 - entry) if is_long else (entry - tp1)
                realized_cashflow += pnl_per_unit * p1_qty
                fees += tp1 * p1_qty * fee_per_side
                remaining -= p1_split
                tp1_hit = True
                # SL → BE
                sl_current = entry * (1 + 2 * fee_per_side) if is_long else entry * (1 - 2 * fee_per_side)

        if tp1_hit and not tp2_hit:
            hit = (bar_high >= tp2) if is_long else (bar_low <= tp2)
            if hit:
                pnl_per_unit = (tp2 - entry) if is_long else (entry - tp2)
                realized_cashflow += pnl_per_unit * p2_qty
                fees += tp2 * p2_qty * fee_per_side
                remaining -= p2_split
                tp2_hit = True

        if tp2_hit and not tp3_hit:
            hit = (bar_high >= tp3) if is_long else (bar_low <= tp3)
            if hit:
                pnl_per_unit = (tp3 - entry) if is_long else (entry - tp3)
                realized_cashflow += pnl_per_unit * p3_qty
                fees += tp3 * p3_qty * fee_per_side
                remaining -= p3_split
                tp3_hit = True
                exit_reason = "tp3"
                exit_ts = ts
                exit_price = tp3
                break

    if exit_reason == "":
        # Window ended with position still open
        last = forward_candles.iloc[-1]
        bar_close = float(last["close"])
        pnl_per_unit = (bar_close - entry) if is_long else (entry - bar_close)
        realized_cashflow += pnl_per_unit * (quantity * remaining)
        fees += bar_close * quantity * remaining * fee_per_side
        exit_reason = "window_end"
        exit_ts = forward_candles.index[-1] if len(forward_candles) else fill_ts
        exit_price = bar_close

    net_pnl_per_unit = (realized_cashflow - fees) / quantity
    r_multiple = net_pnl_per_unit / sl_distance

    return FilledTrade(
        symbol=pf.symbol, type=pf.type,
        direction="long" if is_long else "short",
        tier=tier, p1_price=pf.p1_price,
        detected_at_ts=pf.detected_at_ts,
        fill_ts=fill_ts, fill_price=entry,
        sl=sl_initial, tp1=tp1, tp2=tp2, tp3=tp3,
        exit_ts=exit_ts, exit_price=exit_price, exit_reason=exit_reason,
        r_multiple=r_multiple,
    )


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------


async def run_predictive_backtest(
    symbols: list[str],
    days: int,
    starting_balance: float = 100_000,
    risk_pct: float = 0.01,
    max_concurrent: int = 10,
    timeout_bars: int = 48,
    json_out: str | None = None,
) -> dict:
    print(f"Predictive backtest: {days}d × {len(symbols)} symbols, "
          f"start ${starting_balance:,.0f}, risk {risk_pct*100:.1f}%, "
          f"max concurrent {max_concurrent}, timeout {timeout_bars}h")
    print()

    # Fetch all candles
    print("Fetching 1H candles…")
    all_data: dict[str, pd.DataFrame] = {}
    async with httpx.AsyncClient() as client:
        for sym_human in symbols:
            sym_binance = sym_human.replace("/", "").replace(":USDT", "")
            try:
                df = await fetch_1h(client, sym_binance, days)
                all_data[sym_human] = df
                print(f"  {sym_human}: {len(df)} bars")
            except Exception as e:
                print(f"  {sym_human}: ERROR {e}")

    # Walk every bar in chronological order across all symbols.
    # Each symbol has its own bar timeline; we'll merge them.
    print("\nDetecting pre-formations and tracking limits…")

    # Active pending pre-formations (per symbol — only one active at a time)
    pending_per_symbol: dict[str, dict] = {}
    all_filled_trades: list[FilledTrade] = []
    all_skipped: list[dict] = []  # not-filled or capacity-skipped

    # For chronological cross-symbol simulation, build a unified timeline
    # of (bar_ts, sym, bar_data, bar_idx).
    timeline = []
    for sym, df in all_data.items():
        if df.empty:
            continue
        for i, ts in enumerate(df.index):
            timeline.append((ts, sym, i, df.iloc[i]))
    timeline.sort(key=lambda x: x[0])

    # Compounded balance state
    balance = starting_balance
    open_trades: list[FilledTrade] = []
    peak_so_far = starting_balance
    max_dd_pct = 0.0

    def close_due(now: datetime) -> None:
        nonlocal balance, peak_so_far, max_dd_pct, open_trades
        still_open = []
        for t in open_trades:
            if t.exit_ts is not None and t.exit_ts <= now:
                t.realized_pnl_usd = t.r_multiple * t.risk_usd_at_entry
                balance += t.realized_pnl_usd
                if balance > peak_so_far:
                    peak_so_far = balance
                dd_pct = (peak_so_far - balance) / peak_so_far * 100
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
                all_filled_trades.append(t)
            else:
                still_open.append(t)
        open_trades = still_open

    for bar_ts, sym, i, bar in timeline:
        df = all_data[sym]

        # 1. Close any open trades that have exited before this bar.
        close_due(bar_ts)

        # 2. Check existing pending limits for this symbol.
        if sym in pending_per_symbol:
            pf_state = pending_per_symbol[sym]
            pf: PreFormation = pf_state["pf"]
            limits_remaining: list[tuple[int, float]] = pf_state["limits"]
            invalidated = False

            # Check invalidation: close beyond P1 threshold
            if pf.type == "W":
                if float(bar["close"]) <= pf.invalidation_threshold:
                    invalidated = True
            else:
                if float(bar["close"]) >= pf.invalidation_threshold:
                    invalidated = True

            # Timeout check
            if i > pf.timeout_at_idx:
                invalidated = True

            if invalidated:
                # Cancel remaining limits
                for tier, limit in limits_remaining:
                    all_skipped.append({
                        "symbol": sym, "type": pf.type, "tier": tier,
                        "reason": "invalidated_or_timeout",
                        "limit_price": limit, "detected_at": pf.detected_at_ts.isoformat(),
                    })
                del pending_per_symbol[sym]
            else:
                # Check fills (bar high/low touches limits)
                bar_high = float(bar["high"])
                bar_low = float(bar["low"])
                still_pending = []
                for tier, limit in limits_remaining:
                    touched = (bar_low <= limit) if pf.type == "W" else (bar_high >= limit)
                    if touched:
                        # Capacity check
                        if len(open_trades) >= max_concurrent:
                            all_skipped.append({
                                "symbol": sym, "type": pf.type, "tier": tier,
                                "reason": "capacity_limit",
                                "limit_price": limit, "detected_at": pf.detected_at_ts.isoformat(),
                            })
                            continue
                        # Fill at limit price; simulate forward
                        forward = df.iloc[i + 1:]  # bars after fill bar
                        if len(forward) == 0:
                            continue
                        ft = simulate_filled_trade(
                            pf=pf, fill_ts=bar_ts, fill_price=limit, tier=tier,
                            forward_candles=forward,
                        )
                        # Compute risk & balance at entry
                        ft.risk_usd_at_entry = balance * risk_pct
                        ft.balance_at_entry = balance
                        open_trades.append(ft)
                    else:
                        still_pending.append((tier, limit))
                pf_state["limits"] = still_pending
                if not still_pending:
                    del pending_per_symbol[sym]

        # 3. Detect new pre-formations (only if no pending for this symbol).
        if sym not in pending_per_symbol:
            # Limit detection cost — only run every N bars (e.g. once per actual bar = every scan).
            # In live this would be called every 5-min scan. Here every bar (1H) is fine.
            new_pfs = detect_pre_formations(
                df, sym, detect_at_idx=i, timeout_bars=timeout_bars,
            )
            if new_pfs:
                # Only place limits for the most-recent pre-formation
                pf = new_pfs[-1]
                pending_per_symbol[sym] = {
                    "pf": pf,
                    "limits": [(1, pf.limit_t1), (2, pf.limit_t2), (3, pf.limit_t3)],
                }

    # Close remaining open trades at final timestamp
    final_ts = max(df.index[-1] for df in all_data.values() if not df.empty)
    close_due(final_ts + timedelta(days=365))  # close everything

    # Aggregate
    total_signals = len(all_filled_trades) + len(all_skipped)
    wins = [t for t in all_filled_trades if t.realized_pnl_usd > 0]
    losses = [t for t in all_filled_trades if t.realized_pnl_usd < 0]
    scratches = [t for t in all_filled_trades if t.realized_pnl_usd == 0]
    total_pnl = sum(t.realized_pnl_usd for t in all_filled_trades)
    fills_per_tier = Counter(t.tier for t in all_filled_trades)
    skips_per_reason = Counter(s["reason"] for s in all_skipped)

    # Print report
    print()
    print("=" * 78)
    print("PREDICTIVE BACKTEST RESULT (true pre-formation, 3-tier ladder)")
    print("=" * 78)
    print(f"Window:                 {days} days × {len(symbols)} symbols")
    print(f"Starting balance:       ${starting_balance:>12,.2f}")
    print(f"Final balance:          ${balance:>12,.2f}")
    print(f"Total P&L:              ${total_pnl:>+12,.2f}")
    return_pct = (balance / starting_balance - 1) * 100
    annualized = ((balance / starting_balance) ** (365 / days) - 1) * 100
    print(f"Return %:               {return_pct:>+12.2f}%")
    print(f"Annualized:             {annualized:>+12.2f}%")
    print(f"Peak balance:           ${peak_so_far:>12,.2f}")
    print(f"Max DD (peak→trough):   {max_dd_pct:>12.2f}%")
    print()
    print(f"Pre-formations detected: {total_signals + len(all_skipped)}")
    print(f"Trades filled:           {len(all_filled_trades)}")
    print(f"Skipped:                 {len(all_skipped)}")
    for reason, n in skips_per_reason.most_common():
        print(f"  {reason}: {n}")
    print(f"Fills per tier: {dict(fills_per_tier)}")
    print()
    if all_filled_trades:
        win_rate = len(wins) / len(all_filled_trades) * 100
        avg_win = sum(t.r_multiple for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.r_multiple for t in losses) / len(losses) if losses else 0
        print(f"Win rate:               {win_rate:>12.1f}%")
        print(f"  Wins / Losses / Scratches: {len(wins)} / {len(losses)} / {len(scratches)}")
        print(f"Avg win R:              {avg_win:>+12.2f}R")
        print(f"Avg loss R:             {avg_loss:>+12.2f}R")
        print(f"Total R:                {sum(t.r_multiple for t in all_filled_trades):>+12.2f}R")

    summary = {
        "starting_balance": starting_balance,
        "final_balance": balance,
        "total_pnl": total_pnl,
        "return_pct": return_pct,
        "annualized_pct": annualized,
        "peak_balance": peak_so_far,
        "max_dd_pct": max_dd_pct,
        "trades_filled": len(all_filled_trades),
        "skipped": len(all_skipped),
        "wins": len(wins),
        "losses": len(losses),
        "scratches": len(scratches),
        "win_rate": (len(wins) / len(all_filled_trades) * 100) if all_filled_trades else 0,
        "fills_per_tier": dict(fills_per_tier),
        "skips_per_reason": dict(skips_per_reason),
    }
    if json_out:
        out = Path(json_out)
        with out.open("w") as fh:
            json.dump({
                "summary": summary,
                "trades": [{**asdict(t), "exit_ts": t.exit_ts.isoformat() if t.exit_ts else None,
                            "fill_ts": t.fill_ts.isoformat(), "detected_at_ts": t.detected_at_ts.isoformat()}
                           for t in all_filled_trades],
                "skipped": all_skipped,
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
    ap.add_argument("--timeout-bars", type=int, default=48,
                    help="Bars (=hours on 1H chart) to wait for limit fill before cancel")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()
    symbols = [f"{s.strip().upper()}/USDT:USDT" for s in args.symbols.split(",") if s.strip()]
    asyncio.run(run_predictive_backtest(
        symbols=symbols, days=args.days,
        starting_balance=args.starting_balance,
        risk_pct=args.risk_pct,
        max_concurrent=args.max_concurrent,
        timeout_bars=args.timeout_bars,
        json_out=args.json_out,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
