"""Trade audit CLI — print every gate, factor, and reasoning step for a
given MM trade in one view.

Usage:
    python3 scripts/trade_audit.py <trade_id>
    python3 scripts/trade_audit.py --symbol BNB   # audit latest BNB MM trade
    python3 scripts/trade_audit.py --last 5       # audit the last 5 MM trades

Reads existing data from Supabase (trades + mm_agent_decisions +
partial_exits tables). No engine execution — this is a forensic tool,
not a replay. For "what would the engine do if I changed rule X" use
scripts/replay_scan.py (coming in Tier 2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make `src` importable when run from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import Settings  # noqa: E402
from src.data.db import Database  # noqa: E402


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def _fmt_price(v) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"${f:,.4f}" if abs(f) >= 0.01 else f"${f:,.8f}"


def _fmt_pct(v, digits: int = 2) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}%"
    except (TypeError, ValueError):
        return str(v)


def _fmt_dt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v.replace("T", " ").split(".")[0] + "Z"
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M:%S UTC")
    return str(v)


def _duration_min(entry: str | None, exit_: str | None) -> str:
    if not entry:
        return "—"
    try:
        e = datetime.fromisoformat(str(entry).replace("Z", "+00:00"))
        x = datetime.fromisoformat(str(exit_).replace("Z", "+00:00")) if exit_ else datetime.now(timezone.utc)
        mins = int((x - e).total_seconds() / 60)
        if mins < 60:
            return f"{mins} min"
        h = mins // 60
        m = mins % 60
        return f"{h}h {m}m"
    except Exception:
        return "—"


def _fetch_trade(db: Database, trade_id: str) -> dict | None:
    result = db.table("trades").select("*").eq("id", trade_id).execute()
    return (result.data or [None])[0]


def _fetch_latest_for_symbol(db: Database, symbol_substring: str) -> dict | None:
    # Match BNB → BNB/USDT:USDT etc.
    result = (
        db.table("trades")
        .select("*")
        .ilike("symbol", f"%{symbol_substring}%")
        .eq("strategy", "mm_method")
        .order("entry_time", desc=True)
        .limit(1)
        .execute()
    )
    return (result.data or [None])[0]


def _fetch_last_n(db: Database, n: int) -> list[dict]:
    result = (
        db.table("trades")
        .select("*")
        .eq("strategy", "mm_method")
        .order("entry_time", desc=True)
        .limit(n)
        .execute()
    )
    return result.data or []


def _fetch_agent_decision(db: Database, trade_id: str, symbol: str,
                          entry_time: str | None) -> dict | None:
    """The agent decision isn't foreign-keyed to trade_id yet (that's a
    v2 followup). Match on (symbol, timestamp ~ entry_time ± 30s)."""
    result = (
        db.table("mm_agent_decisions")
        .select("*")
        .eq("symbol", symbol)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
    rows = result.data or []
    if not entry_time:
        return rows[0] if rows else None
    try:
        et = datetime.fromisoformat(str(entry_time).replace("Z", "+00:00"))
    except Exception:
        return rows[0] if rows else None
    for r in rows:
        try:
            ct = datetime.fromisoformat(str(r.get("created_at", "")).replace("Z", "+00:00"))
            if abs((ct - et).total_seconds()) <= 60:
                return r
        except Exception:
            continue
    # None within 60s → pick the closest
    return min(
        rows,
        key=lambda r: abs(
            (datetime.fromisoformat(str(r.get("created_at", "")).replace("Z", "+00:00")) - et).total_seconds()
        ),
        default=None,
    )


def _fetch_partial_exits(db: Database, trade_id: str) -> list[dict]:
    try:
        result = (
            db.table("partial_exits")
            .select("*")
            .eq("trade_id", trade_id)
            .execute()
        )
        return result.data or []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(trade: dict, agent: dict | None, partials: list[dict]) -> str:
    lines = []
    push = lines.append

    # Header
    push(_hr("═"))
    push(f"  TRADE AUDIT — {trade.get('symbol', '?')} {trade.get('direction', '?').upper()}")
    push(f"  id: {trade.get('id')}")
    push(_hr("═"))

    # --- Setup ---
    push("\n  SETUP")
    push(f"    strategy:          {trade.get('strategy')}")
    push(f"    formation:         {trade.get('mm_formation')} / {trade.get('mm_entry_type')}")
    push(f"    cycle phase:       {trade.get('mm_cycle_phase')}")
    push(f"    confluence grade:  {trade.get('mm_confluence_grade')} ({trade.get('confluence_score')} pts)")

    # --- HTF alignment (migration 018) ---
    push("\n  HTF ALIGNMENT")
    ct = trade.get("counter_trend")
    ct_marker = "⚠️ COUNTER-TREND" if ct else "✓ aligned / neutral"
    push(f"    4H trend:          {trade.get('htf_trend_4h', '—')}")
    push(f"    1D trend:          {trade.get('htf_trend_1d', '—')}")
    push(f"    counter_trend:     {ct_marker}")

    # --- Prices ---
    push("\n  PRICES")
    push(f"    entry:             {_fmt_price(trade.get('entry_price'))}")
    push(f"    SL:                {_fmt_price(trade.get('stop_loss'))}")
    push(f"    original SL:       {_fmt_price(trade.get('original_stop_loss'))}")
    push(f"    TP1:               {_fmt_price(trade.get('take_profit'))}")
    tp_tiers = trade.get("tp_tiers")
    if isinstance(tp_tiers, str):
        try:
            tp_tiers = json.loads(tp_tiers)
        except Exception:
            tp_tiers = {}
    if isinstance(tp_tiers, dict):
        push(f"    TP2:               {_fmt_price(tp_tiers.get('l2'))}")
        push(f"    TP3:               {_fmt_price(tp_tiers.get('l3'))}")

    entry_p = trade.get("entry_price")
    sl_p = trade.get("stop_loss")
    if entry_p and sl_p:
        try:
            e = float(entry_p); s = float(sl_p)
            sl_dist_pct = abs(e - s) / e * 100
            push(f"    SL distance:       {_fmt_pct(sl_dist_pct)}")
            if sl_dist_pct > 5.0:
                push("                       ⚠️  > 5% — wider than course 'cheap stop'")
        except Exception:
            pass

    # --- Size + Risk ---
    push("\n  SIZE & RISK")
    push(f"    entry qty:         {trade.get('original_quantity')}")
    push(f"    remaining qty:     {trade.get('remaining_quantity')}")
    push(f"    notional (entry):  {_fmt_price(trade.get('entry_cost_usd'))}")
    push(f"    margin used:       {_fmt_price(trade.get('margin_used'))}")
    push(f"    leverage (exch):   {trade.get('leverage')}x")
    push(f"    risk (USD):        {_fmt_price(trade.get('risk_usd'))}")
    push(f"    initial R:R:       {trade.get('risk_reward')}")

    # --- SL progression ---
    push("\n  SL LIFECYCLE (course Lessons 47/48)")
    be = trade.get("mm_sl_moved_to_breakeven")
    u50 = trade.get("mm_sl_moved_under_50ema")
    p200 = trade.get("mm_took_200ema_partial")
    push(f"    moved to breakeven:  {'✓' if be else '—'}")
    push(f"    moved under 50 EMA:  {'✓' if u50 else '—'}")
    push(f"    took 200 EMA partial:{'✓' if p200 else '—'}")
    push(f"    current tier:        {trade.get('current_tier') or 0}")

    # --- Agent ---
    push("\n  MM SANITY AGENT (Agent 4)")
    if agent is None:
        push("    (no matching agent decision found — trade may pre-date the agent)")
    else:
        push(f"    decision:          {agent.get('decision')}")
        push(f"    confidence:        {agent.get('confidence')}")
        push(f"    model:             {agent.get('model')}")
        push(f"    latency:           {agent.get('latency_ms')} ms")
        push(f"    cost:              {_fmt_price(agent.get('cost_usd'))}")
        push(f"    concerns:          {agent.get('concerns') or []}")
        push(f"    reason:            {agent.get('reason')}")
        push(f"    prompt version:    {agent.get('prompt_version')}")

    # --- Partial exits ---
    push("\n  PARTIAL EXITS")
    if not partials:
        push("    (none yet)")
    else:
        for p in partials:
            push(
                f"    tier={p.get('tier')} qty={p.get('quantity')} "
                f"price={_fmt_price(p.get('price'))} "
                f"pnl_usd={_fmt_price(p.get('pnl_usd'))} "
                f"at={_fmt_dt(p.get('created_at') or p.get('executed_at'))}"
            )

    # --- Outcome ---
    push("\n  OUTCOME")
    push(f"    status:            {trade.get('status')}")
    push(f"    entry_time:        {_fmt_dt(trade.get('entry_time'))}")
    if trade.get("status") == "closed":
        push(f"    exit_time:         {_fmt_dt(trade.get('exit_time'))}")
        push(f"    exit_price:        {_fmt_price(trade.get('exit_price'))}")
        push(f"    exit_reason:       {trade.get('exit_reason')}")
        push(f"    duration:          {_duration_min(trade.get('entry_time'), trade.get('exit_time'))}")
        push(f"    realized pnl:      {_fmt_price(trade.get('pnl_usd'))}")
    else:
        push(f"    open for:          {_duration_min(trade.get('entry_time'), None)}")

    # --- Entry reason string (human summary from engine) ---
    push("\n  ENGINE REASON STRING")
    push(f"    {trade.get('entry_reason', '—')}")

    push(_hr("═"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trade_id", nargs="?", help="Trade UUID")
    ap.add_argument("--symbol", help="Audit latest MM trade on symbol (e.g. BNB)")
    ap.add_argument("--last", type=int, help="Audit the last N MM trades")
    args = ap.parse_args()

    if not (args.trade_id or args.symbol or args.last):
        ap.print_help()
        return 2

    # Prefer TARAKTA_SUPABASE_URL/KEY if set for ad-hoc auditing; fallback to
    # the main SUPABASE_URL/KEY used by the bot.
    settings = Settings()
    if not settings.supabase_url or not settings.supabase_key:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in env.", file=sys.stderr)
        return 1

    db = Database(settings.supabase_url, settings.supabase_key)

    trades_to_audit: list[dict] = []
    if args.trade_id:
        t = _fetch_trade(db, args.trade_id)
        if t:
            trades_to_audit.append(t)
    elif args.symbol:
        t = _fetch_latest_for_symbol(db, args.symbol.upper())
        if t:
            trades_to_audit.append(t)
    elif args.last:
        trades_to_audit = _fetch_last_n(db, args.last)

    if not trades_to_audit:
        print("No matching trade(s) found.", file=sys.stderr)
        return 1

    for t in trades_to_audit:
        agent = _fetch_agent_decision(
            db,
            trade_id=t["id"],
            symbol=t.get("symbol", ""),
            entry_time=t.get("entry_time"),
        )
        partials = _fetch_partial_exits(db, t["id"])
        print(render(t, agent, partials))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
