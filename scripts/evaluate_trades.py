#!/usr/bin/env python3
"""Paper-forward evaluation script — compare pre/post metrics.

Compares trade performance before and after the Agent 1/2/3 rollout.
Reads from the Supabase trades table and produces a summary report.

Usage:
    python scripts/evaluate_trades.py [--cutoff YYYY-MM-DD] [--mode paper|live]

The cutoff date separates "before" (pre-AI) from "after" (post-AI) trades.
If not specified, defaults to 7 days ago.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pre/post AI trade metrics")
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="Cutoff date (YYYY-MM-DD) separating pre/post periods. Default: 7 days ago.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="paper",
        choices=["paper", "live"],
        help="Trading mode to filter (default: paper)",
    )
    parser.add_argument(
        "--days-before",
        type=int,
        default=30,
        help="How many days before cutoff to include (default: 30)",
    )
    parser.add_argument(
        "--days-after",
        type=int,
        default=7,
        help="How many days after cutoff to include (default: 7)",
    )
    return parser.parse_args()


def compute_metrics(trades: list[dict]) -> dict[str, Any]:
    """Compute performance metrics from a list of closed trades."""
    if not trades:
        return {
            "count": 0,
            "win_rate": 0.0,
            "avg_pnl_usd": 0.0,
            "avg_pnl_pct": 0.0,
            "total_pnl_usd": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "avg_hold_minutes": 0.0,
            "avg_confluence": 0.0,
            "exit_reasons": {},
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
        }

    wins = [t for t in trades if (t.get("pnl_usd") or 0) > 0]
    losses = [t for t in trades if (t.get("pnl_usd") or 0) <= 0]
    pnls = [t.get("pnl_usd") or 0 for t in trades]
    pnl_pcts = [t.get("pnl_percent") or 0 for t in trades]

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))

    # Hold time in minutes
    hold_times = []
    for t in trades:
        entry = t.get("entry_time")
        exit_ = t.get("exit_time")
        if entry and exit_:
            try:
                dt_entry = datetime.fromisoformat(entry)
                dt_exit = datetime.fromisoformat(exit_)
                hold_times.append((dt_exit - dt_entry).total_seconds() / 60)
            except (ValueError, TypeError):
                pass

    # Exit reason distribution
    exit_reasons: dict[str, int] = {}
    for t in trades:
        reason = t.get("exit_reason") or "unknown"
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Confluence scores
    confluences = [t.get("confluence_score") or 0 for t in trades if t.get("confluence_score")]

    return {
        "count": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "avg_pnl_usd": sum(pnls) / len(pnls) if pnls else 0,
        "avg_pnl_pct": sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0,
        "total_pnl_usd": sum(pnls),
        "max_win": max(pnls) if pnls else 0,
        "max_loss": min(pnls) if pnls else 0,
        "avg_hold_minutes": sum(hold_times) / len(hold_times) if hold_times else 0,
        "avg_confluence": sum(confluences) / len(confluences) if confluences else 0,
        "exit_reasons": exit_reasons,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "expectancy_r": sum(pnls) / len(trades) if trades else 0,
    }


def print_comparison(before: dict, after: dict, cutoff_str: str) -> None:
    """Print side-by-side comparison of pre/post metrics."""
    print("\n" + "=" * 72)
    print(f"  TRADE PERFORMANCE EVALUATION — Cutoff: {cutoff_str}")
    print("=" * 72)

    def fmt(val, suffix="", fmt_str=".2f"):
        if isinstance(val, float):
            return f"{val:{fmt_str}}{suffix}"
        return str(val) + suffix

    def delta(b_val, a_val, higher_is_better=True):
        if b_val == 0:
            return ""
        diff = a_val - b_val
        pct = diff / abs(b_val) * 100 if b_val != 0 else 0
        arrow = "+" if diff > 0 else ""
        color = "" if diff == 0 else ""
        return f"  ({arrow}{pct:.1f}%)"

    rows = [
        ("Trades", before["count"], after["count"], False),
        ("Win Rate", f"{before['win_rate']:.1f}%", f"{after['win_rate']:.1f}%", True),
        ("Wins / Losses", f"{before.get('wins', 0)} / {before.get('losses', 0)}",
         f"{after.get('wins', 0)} / {after.get('losses', 0)}", False),
        ("Avg PnL (USD)", f"${before['avg_pnl_usd']:.2f}", f"${after['avg_pnl_usd']:.2f}", True),
        ("Total PnL (USD)", f"${before['total_pnl_usd']:.2f}", f"${after['total_pnl_usd']:.2f}", True),
        ("Max Win", f"${before['max_win']:.2f}", f"${after['max_win']:.2f}", True),
        ("Max Loss", f"${before['max_loss']:.2f}", f"${after['max_loss']:.2f}", False),
        ("Profit Factor", f"{before['profit_factor']:.2f}", f"{after['profit_factor']:.2f}", True),
        ("Avg Hold (min)", f"{before['avg_hold_minutes']:.0f}", f"{after['avg_hold_minutes']:.0f}", False),
        ("Avg Confluence", f"{before['avg_confluence']:.0f}", f"{after['avg_confluence']:.0f}", True),
    ]

    print(f"\n  {'Metric':<20} {'BEFORE':<18} {'AFTER':<18}")
    print(f"  {'-'*20} {'-'*18} {'-'*18}")
    for label, b, a, _ in rows:
        print(f"  {label:<20} {str(b):<18} {str(a):<18}")

    # Exit reason breakdown
    print(f"\n  Exit Reasons (Before / After):")
    all_reasons = set(list(before.get("exit_reasons", {}).keys()) + list(after.get("exit_reasons", {}).keys()))
    for reason in sorted(all_reasons):
        b_count = before.get("exit_reasons", {}).get(reason, 0)
        a_count = after.get("exit_reasons", {}).get(reason, 0)
        print(f"    {reason:<25} {b_count:>5}  /  {a_count:>5}")

    print("\n" + "=" * 72)


def main() -> None:
    args = parse_args()

    # Determine cutoff
    if args.cutoff:
        cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.days_after)

    before_start = cutoff - timedelta(days=args.days_before)
    after_end = cutoff + timedelta(days=args.days_after)

    print(f"Evaluating trades from {before_start.date()} to {after_end.date()}")
    print(f"Cutoff (pre/post split): {cutoff.date()}")
    print(f"Mode: {args.mode}")

    # Try to connect to Supabase
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")

    if not supabase_url or not supabase_key:
        print("\nWARNING: SUPABASE_URL and SUPABASE_KEY not set.")
        print("Running with sample data for demonstration.\n")

        # Demo with sample data
        sample_before = compute_metrics([])
        sample_after = compute_metrics([])
        print_comparison(sample_before, sample_after, str(cutoff.date()))
        print("\nTo run with real data, set SUPABASE_URL and SUPABASE_KEY environment variables.")
        return

    try:
        from supabase import create_client
        db = create_client(supabase_url, supabase_key)

        # Fetch before trades
        before_result = (
            db.table("trades")
            .select("*")
            .eq("status", "closed")
            .eq("mode", args.mode)
            .gte("exit_time", before_start.isoformat())
            .lt("exit_time", cutoff.isoformat())
            .order("exit_time", desc=True)
            .execute()
        )
        before_trades = before_result.data or []

        # Fetch after trades
        after_result = (
            db.table("trades")
            .select("*")
            .eq("status", "closed")
            .eq("mode", args.mode)
            .gte("exit_time", cutoff.isoformat())
            .lte("exit_time", after_end.isoformat())
            .order("exit_time", desc=True)
            .execute()
        )
        after_trades = after_result.data or []

        before_metrics = compute_metrics(before_trades)
        after_metrics = compute_metrics(after_trades)

        print_comparison(before_metrics, after_metrics, str(cutoff.date()))

        # Per-symbol breakdown if enough trades
        if len(after_trades) >= 5:
            symbols = set(t.get("symbol", "") for t in after_trades)
            print(f"\n  Per-Symbol Breakdown (after cutoff, {len(symbols)} symbols):")
            print(f"  {'Symbol':<15} {'Trades':>7} {'Win%':>7} {'PnL':>10}")
            print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*10}")
            for sym in sorted(symbols):
                sym_trades = [t for t in after_trades if t.get("symbol") == sym]
                m = compute_metrics(sym_trades)
                print(f"  {sym:<15} {m['count']:>7} {m['win_rate']:>6.1f}% ${m['total_pnl_usd']:>8.2f}")

    except ImportError:
        print("ERROR: supabase-py not installed. Run: pip install supabase")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
