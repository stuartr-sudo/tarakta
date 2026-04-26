"""Real compounded backtest with single starting balance and concurrent-
position cap. Tests strategy realism beyond the standalone-trade simulator.

How it differs from the default replay --pnl mode:
  - Default: each trade is simulated standalone against a fixed $100k
    balance with 1% risk = $1,000 per trade. Total P&L = sum of trades.
    No compounding, no capacity limits.
  - This script: starting balance $100k, every trade risks 1% of the
    CURRENT balance (compounding). Maintains an open-positions timeline.
    When a new signal arrives and >= max_concurrent positions are open,
    the signal is SKIPPED (capacity limit).

Reads signal/exit data by re-running the engine against historical
candles, just like replay_scan.py. Produces a balance curve, per-trade
detail, and aggregate metrics.

Usage:
    python3 scripts/compounded_backtest.py --symbols BTC,ETH,... --days 90
    python3 scripts/compounded_backtest.py --symbols ... --days 90 --gate-threshold 3
    python3 scripts/compounded_backtest.py --symbols ... --days 90 --max-concurrent 5

Outputs:
  - Console summary: starting balance, ending balance, per-trade P&L,
    drawdown, max concurrent, signals skipped due to capacity
  - Optional --json writes detailed trade ledger to a file
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

import scripts.replay_scan as rs  # noqa: E402


# ---------------------------------------------------------------------------
# Compounded simulation
# ---------------------------------------------------------------------------


@dataclass
class CompoundedTrade:
    symbol: str
    direction: str
    entry_ts: datetime
    exit_ts: datetime | None
    entry_price: float
    exit_price: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    r_multiple: float                  # outcome from base simulator
    realized_pnl_usd: float = 0.0      # compounded $ amount
    risk_usd_at_entry: float = 0.0     # 1% of balance when opened
    balance_at_entry: float = 0.0
    balance_at_exit: float = 0.0
    skipped_capacity: bool = False
    exit_reason: str = ""


@dataclass
class CompoundedRun:
    starting_balance: float
    final_balance: float
    max_concurrent: int
    peak_balance: float
    trough_balance: float
    max_drawdown_pct: float
    total_signals_seen: int
    trades_taken: int
    skipped_capacity: int
    wins: int
    losses: int
    scratches: int
    total_pnl_usd: float
    total_r: float
    win_rate: float
    avg_win_r: float
    avg_loss_r: float
    trades: list[CompoundedTrade] = field(default_factory=list)


def run_compounded_simulation(
    bars: list,                        # list[BarResult] across all symbols
    starting_balance: float = 100_000,
    risk_pct: float = 0.01,
    max_concurrent: int = 5,
) -> CompoundedRun:
    """Walk all signals chronologically, applying compounding + concurrency cap.

    Algorithm:
      1. Filter bars to those that produced a signal AND had P&L simulated.
      2. Sort by entry timestamp.
      3. Iterate:
         - Close any open trades whose exit_ts <= current signal's entry_ts.
           Add their compounded P&L (r_multiple × risk_at_their_entry) to
           balance.
         - If open_trades count >= max_concurrent: mark this signal skipped
           and continue.
         - Else open new trade: risk_usd = balance × risk_pct, record
           balance_at_entry, expected close at exit_ts.
      4. After all signals processed, close remaining open trades in
         exit-time order.
    """
    # Extract signal-bearing bars with PnL info.
    signaled = [
        b for b in bars
        if b.signal is not None and b.pnl is not None
    ]
    signaled.sort(key=lambda b: b.ts)

    open_trades: list[CompoundedTrade] = []
    completed: list[CompoundedTrade] = []
    skipped: list[CompoundedTrade] = []

    balance = starting_balance
    peak_balance = starting_balance
    trough_balance = starting_balance
    max_concurrent_seen = 0

    # True peak-to-trough drawdown tracking. peak_so_far walks forward
    # only (a new peak resets the reference). max_drawdown is the
    # largest single peak-to-trough drop the balance experienced.
    peak_so_far = starting_balance
    max_drawdown_dollar = 0.0
    max_drawdown_pct = 0.0
    max_dd_peak_at_time = 0.0
    max_dd_trough_at_time = 0.0

    def update_drawdown(current: float) -> None:
        nonlocal peak_so_far, max_drawdown_dollar, max_drawdown_pct
        nonlocal max_dd_peak_at_time, max_dd_trough_at_time
        if current > peak_so_far:
            peak_so_far = current
        dd = peak_so_far - current
        dd_pct = (dd / peak_so_far) * 100 if peak_so_far > 0 else 0.0
        if dd_pct > max_drawdown_pct:
            max_drawdown_pct = dd_pct
            max_drawdown_dollar = dd
            max_dd_peak_at_time = peak_so_far
            max_dd_trough_at_time = current

    def close_due(now: datetime) -> None:
        """Close any open trades whose exit_ts has passed `now`."""
        nonlocal balance, peak_balance, trough_balance, open_trades
        still_open = []
        for t in open_trades:
            if t.exit_ts is not None and t.exit_ts <= now:
                # Compounded $ outcome = r_multiple × risk_usd_at_entry
                t.realized_pnl_usd = t.r_multiple * t.risk_usd_at_entry
                balance += t.realized_pnl_usd
                t.balance_at_exit = balance
                peak_balance = max(peak_balance, balance)
                trough_balance = min(trough_balance, balance)
                update_drawdown(balance)
                completed.append(t)
            else:
                still_open.append(t)
        open_trades = still_open

    for bar in signaled:
        signal_ts = bar.ts
        # Step 1: close any trades that exited before this signal.
        close_due(signal_ts)

        # Step 2: capacity check.
        if len(open_trades) >= max_concurrent:
            # Mark skipped (don't open). Track for telemetry.
            sk = CompoundedTrade(
                symbol=bar.signal["symbol"],
                direction=bar.signal["direction"],
                entry_ts=signal_ts,
                exit_ts=None,
                entry_price=bar.signal["entry"],
                exit_price=0.0,
                sl=bar.signal["sl"],
                tp1=bar.signal.get("tp1", 0),
                tp2=bar.signal.get("tp2", 0),
                tp3=bar.signal.get("tp3", 0),
                r_multiple=0.0,
                skipped_capacity=True,
                exit_reason="capacity_limit",
            )
            skipped.append(sk)
            continue

        # Step 3: open new trade.
        risk_usd = balance * risk_pct
        if risk_usd <= 0:
            # Account blown up; stop.
            continue

        pnl_res = bar.pnl
        t = CompoundedTrade(
            symbol=bar.signal["symbol"],
            direction=bar.signal["direction"],
            entry_ts=signal_ts,
            exit_ts=pnl_res.exit_ts,
            entry_price=pnl_res.entry_price,
            exit_price=pnl_res.exit_price,
            sl=pnl_res.sl_initial,
            tp1=pnl_res.tp1,
            tp2=pnl_res.tp2,
            tp3=pnl_res.tp3,
            r_multiple=pnl_res.r_multiple,
            risk_usd_at_entry=risk_usd,
            balance_at_entry=balance,
            exit_reason=pnl_res.exit_reason,
        )
        open_trades.append(t)
        max_concurrent_seen = max(max_concurrent_seen, len(open_trades))

    # Step 4: close any remaining open trades by their exit_ts.
    open_trades.sort(key=lambda t: t.exit_ts or datetime.max.replace(tzinfo=timezone.utc))
    for t in open_trades:
        if t.exit_ts is None:
            continue
        t.realized_pnl_usd = t.r_multiple * t.risk_usd_at_entry
        balance += t.realized_pnl_usd
        t.balance_at_exit = balance
        peak_balance = max(peak_balance, balance)
        trough_balance = min(trough_balance, balance)
        update_drawdown(balance)
        completed.append(t)
    open_trades = []

    # Compute aggregate metrics.
    wins = [t for t in completed if t.realized_pnl_usd > 0]
    losses = [t for t in completed if t.realized_pnl_usd < 0]
    scratches = [t for t in completed if t.realized_pnl_usd == 0]
    total_pnl = sum(t.realized_pnl_usd for t in completed)
    total_r = sum(t.r_multiple for t in completed)
    avg_win_r = (sum(t.r_multiple for t in wins) / len(wins)) if wins else 0.0
    avg_loss_r = (sum(t.r_multiple for t in losses) / len(losses)) if losses else 0.0
    win_rate = (len(wins) / len(completed) * 100) if completed else 0.0

    all_trades = completed + skipped
    return CompoundedRun(
        starting_balance=starting_balance,
        final_balance=balance,
        max_concurrent=max_concurrent_seen,
        peak_balance=peak_balance,
        trough_balance=trough_balance,
        max_drawdown_pct=max_drawdown_pct,
        total_signals_seen=len(signaled),
        trades_taken=len(completed),
        skipped_capacity=len(skipped),
        wins=len(wins),
        losses=len(losses),
        scratches=len(scratches),
        total_pnl_usd=total_pnl,
        total_r=total_r,
        win_rate=win_rate,
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        trades=all_trades,
    )


# ---------------------------------------------------------------------------
# Driver — collects bars from all symbols, runs compounded sim, prints report
# ---------------------------------------------------------------------------


async def run(
    symbols: list[str],
    days: int,
    starting_balance: float,
    risk_pct: float,
    max_concurrent: int,
    gate_threshold: int,
    json_out: str | None,
) -> int:
    # Patch gate threshold if requested (mirrors the wrapper-script pattern).
    if gate_threshold > 0:
        import src.strategy.mm_engine as eng
        _orig_init = eng.MMEngine.__init__

        def _init_with_gates(self, *a, **kw):
            _orig_init(self, *a, **kw)
            self._gate_threshold = gate_threshold

        eng.MMEngine.__init__ = _init_with_gates  # type: ignore[method-assign]
        print(f"# Gate threshold set: {gate_threshold} of 5\n", flush=True)

    # Run the per-symbol replay (with --pnl forced) to collect signals.
    print(f"Compounded backtest: {days}d × {len(symbols)} symbols, "
          f"start ${starting_balance:,.0f}, risk {risk_pct*100:.1f}%, "
          f"max concurrent {max_concurrent}")
    print("Collecting signals…\n")

    all_bars = []
    for sym in symbols:
        sr = await rs.replay_single_symbol(
            sym, days, hours_step=1, engine_overrides={}, pnl_enabled=True,
        )
        # Tag each bar with the symbol for the compounded sim.
        for b in sr.bars:
            if b.signal is not None and b.pnl is not None:
                b.signal["symbol"] = sym
                all_bars.append(b)
        print(f"[{sym}] {len([b for b in sr.bars if b.signal is not None])} signals")

    print(f"\nTotal signals across universe: {len(all_bars)}")
    print("Running compounded simulation…\n")

    result = run_compounded_simulation(
        all_bars,
        starting_balance=starting_balance,
        risk_pct=risk_pct,
        max_concurrent=max_concurrent,
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    print("=" * 78)
    print("COMPOUNDED BACKTEST RESULT")
    print("=" * 78)
    print(f"Window:                 {days} days × {len(symbols)} symbols")
    print(f"Starting balance:       ${result.starting_balance:>12,.2f}")
    print(f"Final balance:          ${result.final_balance:>12,.2f}")
    print(f"Total P&L:              ${result.total_pnl_usd:>+12,.2f}")
    return_pct = (result.final_balance / result.starting_balance - 1) * 100
    annualized = ((result.final_balance / result.starting_balance) ** (365 / days) - 1) * 100
    print(f"Return %:               {return_pct:>+12.2f}%")
    print(f"Annualized:             {annualized:>+12.2f}%")
    print(f"Peak balance:           ${result.peak_balance:>12,.2f}")
    print(f"Trough balance:         ${result.trough_balance:>12,.2f}")
    print(f"Max drawdown (peak→trough): {result.max_drawdown_pct:>9.2f}%  (true experienced DD)")
    print()
    print(f"Signals seen:           {result.total_signals_seen}")
    print(f"Trades taken:           {result.trades_taken}")
    print(f"Skipped (capacity):     {result.skipped_capacity}")
    print(f"Max concurrent open:    {result.max_concurrent}")
    print()
    if result.trades_taken:
        print(f"Win rate:               {result.win_rate:>12.1f}%")
        print(f"  Wins / Losses / Scratches: {result.wins} / {result.losses} / {result.scratches}")
        print(f"Total R-multiple:       {result.total_r:>+12.2f}R")
        print(f"Avg win R:              {result.avg_win_r:>+12.2f}R")
        print(f"Avg loss R:             {result.avg_loss_r:>+12.2f}R")

    if json_out:
        out_path = Path(json_out)
        with out_path.open("w") as fh:
            json.dump({
                "config": {
                    "symbols": symbols, "days": days,
                    "starting_balance": starting_balance,
                    "risk_pct": risk_pct,
                    "max_concurrent": max_concurrent,
                    "gate_threshold": gate_threshold,
                },
                "summary": {
                    k: v for k, v in asdict(result).items() if k != "trades"
                },
                "trades": [
                    {
                        **{k: (v.isoformat() if isinstance(v, datetime) else v)
                           for k, v in asdict(t).items()},
                    }
                    for t in result.trades
                ],
            }, fh, indent=2, default=str)
        print(f"\nDetailed trade ledger written to {out_path}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True,
                    help="Comma-separated, e.g. BTC,ETH,BNB")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--starting-balance", type=float, default=100_000.0)
    ap.add_argument("--risk-pct", type=float, default=0.01)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--gate-threshold", type=int, default=0,
                    help="0=disabled (default), 1-5 = N of 5 course gates required")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Write detailed trade ledger to JSON file")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    symbols = [rs._normalise_symbol(s) for s in symbols]
    return asyncio.run(run(
        symbols=symbols,
        days=args.days,
        starting_balance=args.starting_balance,
        risk_pct=args.risk_pct,
        max_concurrent=args.max_concurrent,
        gate_threshold=args.gate_threshold,
        json_out=args.json_out,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
