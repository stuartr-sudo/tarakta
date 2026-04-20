"""Agent review CLI — is the MM Sanity Agent doing its job?

Reads mm_agent_decisions and trades from Supabase. Joins by timestamp
proximity (agent calls within 120s of the entry_time of the trade that
eventually entered). Produces a forensic report covering:

  - Overall counts (APPROVE / VETO / ERROR), avg confidence, total cost
  - Breakdown by grade, HTF direction, formation variant, cycle phase
  - Outcome distribution on APPROVED trades (wins / scratches / losses)
  - Common concerns across VETOs
  - Per-profile P&L (grade × HTF) — the actionable pattern signal

Usage:
    python3 scripts/agent_review.py --days 7
    python3 scripts/agent_review.py --days 14 --json > /tmp/review.json

No engine execution, pure DB reads. Safe to run anytime.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import Settings  # noqa: E402
from src.data.db import Database  # noqa: E402


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def _fetch_agent_decisions(db: Database, days: int) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    result = (
        db.table("mm_agent_decisions")
        .select("*")
        .gte("created_at", since)
        .order("created_at", desc=False)
        .execute()
    )
    return result.data or []


def _fetch_mm_trades(db: Database, days: int) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    result = (
        db.table("trades")
        .select("*")
        .eq("strategy", "mm_method")
        .gte("entry_time", since)
        .order("entry_time", desc=False)
        .execute()
    )
    return result.data or []


def _parse_ts(s) -> datetime | None:
    if s is None:
        return None
    if isinstance(s, datetime):
        return s
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _match_trade_to_decision(decisions: list[dict], trades: list[dict]) -> dict[str, str | None]:
    """For each trade, find the agent decision that led to it (by symbol
    + entry_time within 120s of decision's created_at).
    Returns {trade_id: decision_id or None}.
    """
    link: dict[str, str | None] = {}
    # Pre-index decisions by symbol for speed
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for d in decisions:
        sym = d.get("symbol") or ""
        by_symbol[sym].append(d)

    for t in trades:
        tid = t.get("id")
        et = _parse_ts(t.get("entry_time"))
        if not tid or et is None:
            continue
        sym = t.get("symbol") or ""
        closest_id = None
        closest_delta = None
        for d in by_symbol.get(sym, []):
            ct = _parse_ts(d.get("created_at"))
            if ct is None:
                continue
            delta = abs((ct - et).total_seconds())
            # Decision must be slightly before the trade entry
            # (agent runs before execute). Allow up to 120s window.
            if delta <= 120:
                if closest_delta is None or delta < closest_delta:
                    closest_delta = delta
                    closest_id = d.get("id")
        link[tid] = closest_id
    return link


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def _overall_counts(decisions: list[dict]) -> dict:
    by_dec: dict[str, dict] = defaultdict(lambda: {"n": 0, "conf_sum": 0.0, "conf_n": 0,
                                                    "latency_sum": 0, "cost_sum": 0.0})
    for d in decisions:
        dec = d.get("decision") or "?"
        b = by_dec[dec]
        b["n"] += 1
        c = d.get("confidence")
        if c is not None:
            try:
                b["conf_sum"] += float(c)
                b["conf_n"] += 1
            except (TypeError, ValueError):
                pass
        try:
            b["latency_sum"] += int(d.get("latency_ms") or 0)
        except (TypeError, ValueError):
            pass
        try:
            b["cost_sum"] += float(d.get("cost_usd") or 0)
        except (TypeError, ValueError):
            pass
    out = {}
    for dec, b in by_dec.items():
        avg_conf = (b["conf_sum"] / b["conf_n"]) if b["conf_n"] else None
        avg_latency = (b["latency_sum"] / b["n"]) if b["n"] else 0
        out[dec] = {
            "n": b["n"],
            "avg_confidence": round(avg_conf, 3) if avg_conf else None,
            "avg_latency_ms": round(avg_latency, 0),
            "total_cost_usd": round(b["cost_sum"], 4),
        }
    return out


def _extract_profile(d: dict) -> tuple[str, str, str]:
    """Return (grade, htf_4h, formation_variant) — the canonical decision profile."""
    ctx = d.get("input_context") or {}
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except Exception:
            ctx = {}
    grade = (ctx.get("grade") or d.get("confluence_grade") or "?").upper()
    htf = (d.get("htf_trend_4h") or ctx.get("htf_trend_4h") or "?").lower()
    variant = (ctx.get("formation_variant") or d.get("formation_variant") or "?").lower()
    return grade, htf, variant


def _breakdown_by_profile(decisions: list[dict]) -> dict:
    """Counts by (decision, grade) and (decision, htf_4h)."""
    by_grade: dict[str, Counter] = defaultdict(Counter)
    by_htf: dict[str, Counter] = defaultdict(Counter)
    by_variant: dict[str, Counter] = defaultdict(Counter)
    for d in decisions:
        dec = d.get("decision") or "?"
        grade, htf, variant = _extract_profile(d)
        by_grade[grade][dec] += 1
        by_htf[htf][dec] += 1
        by_variant[variant][dec] += 1
    return {
        "by_grade": {k: dict(v) for k, v in by_grade.items()},
        "by_htf_4h": {k: dict(v) for k, v in by_htf.items()},
        "by_formation_variant": {k: dict(v) for k, v in by_variant.items()},
    }


def _concern_counts(decisions: list[dict]) -> Counter:
    c = Counter()
    for d in decisions:
        if d.get("decision") != "VETO":
            continue
        concerns = d.get("concerns") or []
        if isinstance(concerns, str):
            try:
                concerns = json.loads(concerns)
            except Exception:
                concerns = []
        for tag in concerns:
            c[tag] += 1
    return c


def _approved_outcomes(
    decisions: list[dict],
    trades: list[dict],
    trade_to_dec: dict[str, str | None],
) -> dict:
    """For APPROVE decisions that led to trades, classify outcomes."""
    # Build decision id → decision map
    dec_by_id = {d.get("id"): d for d in decisions}

    profile_pnl: dict[tuple[str, str], list[float]] = defaultdict(list)
    outcomes = Counter()
    per_trade: list[dict] = []

    for t in trades:
        tid = t.get("id")
        did = trade_to_dec.get(tid)
        if not did:
            continue
        d = dec_by_id.get(did)
        if not d or d.get("decision") != "APPROVE":
            continue
        pnl = t.get("pnl_usd")
        try:
            pnl = float(pnl) if pnl is not None else None
        except (TypeError, ValueError):
            pnl = None
        if pnl is None and t.get("status") == "open":
            outcome = "open"
        elif pnl is None:
            outcome = "unknown"
        elif pnl > 5:
            outcome = "win"
        elif pnl < -5:
            outcome = "loss"
        else:
            outcome = "scratch"
        outcomes[outcome] += 1

        grade, htf, _variant = _extract_profile(d)
        if pnl is not None:
            profile_pnl[(grade, htf)].append(pnl)

        per_trade.append({
            "symbol": t.get("symbol"),
            "direction": t.get("direction"),
            "grade": grade,
            "htf_4h": htf,
            "status": t.get("status"),
            "pnl_usd": pnl,
            "exit_reason": t.get("exit_reason"),
            "outcome": outcome,
            "confidence": d.get("confidence"),
        })
    return {
        "counts": dict(outcomes),
        "per_trade": per_trade,
        "profile_pnl": profile_pnl,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def render_report(
    days: int,
    decisions: list[dict],
    overall: dict,
    breakdown: dict,
    concerns: Counter,
    approved_outcomes: dict,
) -> str:
    lines = []
    p = lines.append

    p(_hr("═"))
    p(f"  MM SANITY AGENT REVIEW — last {days} day(s)")
    p(_hr("═"))

    if not decisions:
        p("\n  No agent decisions in this window. Agent may be disabled, or no")
        p("  setups have reached the sanity-agent gate yet.")
        return "\n".join(lines)

    # --- Overall ---
    p("\n  OVERALL")
    total = sum(v["n"] for v in overall.values())
    p(f"    total calls:       {total}")
    for dec in ("APPROVE", "VETO", "ERROR"):
        if dec not in overall:
            continue
        v = overall[dec]
        pct = (v["n"] / total * 100) if total else 0.0
        conf_str = f"conf={v['avg_confidence']:.2f}" if v["avg_confidence"] is not None else "conf=—"
        p(f"    {dec:<8}          {v['n']:>3}  ({pct:>5.1f}%)  "
          f"{conf_str}  latency={v['avg_latency_ms']:.0f}ms  "
          f"cost=${v['total_cost_usd']:.4f}")
    total_cost = sum(v["total_cost_usd"] for v in overall.values())
    p(f"    TOTAL cost:        ${total_cost:.4f}  "
      f"(${total_cost * 30 / max(days, 1):.2f}/month projected)")

    # --- Breakdown by grade ---
    p("\n  BY CONFLUENCE GRADE")
    grades = breakdown.get("by_grade") or {}
    for g in sorted(grades.keys(), reverse=True):
        row = grades[g]
        approve = row.get("APPROVE", 0)
        veto = row.get("VETO", 0)
        err = row.get("ERROR", 0)
        tot = approve + veto + err
        approve_rate = (approve / tot * 100) if tot else 0.0
        p(f"    Grade {g:<2}  total={tot:>3}  "
          f"APPROVE={approve:>3}  VETO={veto:>3}  ERROR={err:>3}  "
          f"approve-rate={approve_rate:>5.1f}%")

    # --- Breakdown by HTF direction ---
    p("\n  BY HTF (4H) TREND")
    htfs = breakdown.get("by_htf_4h") or {}
    for h in sorted(htfs.keys()):
        row = htfs[h]
        approve = row.get("APPROVE", 0)
        veto = row.get("VETO", 0)
        tot = approve + veto + row.get("ERROR", 0)
        p(f"    {h:<10}  total={tot:>3}  APPROVE={approve:>3}  VETO={veto:>3}")

    # --- Breakdown by formation variant ---
    p("\n  BY FORMATION VARIANT")
    variants = breakdown.get("by_formation_variant") or {}
    for v_name in sorted(variants.keys()):
        row = variants[v_name]
        approve = row.get("APPROVE", 0)
        veto = row.get("VETO", 0)
        tot = approve + veto + row.get("ERROR", 0)
        p(f"    {v_name:<22}  total={tot:>3}  APPROVE={approve:>3}  VETO={veto:>3}")

    # --- Concerns (VETO reasons) ---
    if concerns:
        p("\n  COMMON CONCERNS (VETO reasons, most frequent first)")
        for tag, n in concerns.most_common():
            p(f"    {tag:<28}  {n:>3}")

    # --- Approved outcomes ---
    out = approved_outcomes
    counts = out.get("counts") or {}
    per_trade = out.get("per_trade") or []
    profile_pnl = out.get("profile_pnl") or {}

    p("\n  APPROVED TRADES — actual outcomes")
    if not per_trade:
        p("    (no matched approvals that resulted in trades)")
    else:
        n_total = len(per_trade)
        wins = counts.get("win", 0)
        losses = counts.get("loss", 0)
        scratches = counts.get("scratch", 0)
        opens = counts.get("open", 0)
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) else 0.0
        total_pnl = sum(t.get("pnl_usd") or 0 for t in per_trade)
        p(f"    total approved:    {n_total}  (wins={wins} losses={losses} "
          f"scratches={scratches} open={opens})")
        if wins + losses > 0:
            p(f"    win rate:          {win_rate:.1f}%  (excl. scratches, opens)")
        p(f"    realized pnl:      ${total_pnl:+,.2f}")

    # Profile-level P&L — the actionable signal
    if profile_pnl:
        p("\n  PROFILE P&L (grade × HTF) — where's the edge / anti-edge?")
        p(f"    {'profile':<28}  {'n':>3}  {'sum':>10}  {'avg':>9}  {'worst':>9}  {'best':>9}")
        for (grade, htf), pnls in sorted(profile_pnl.items()):
            n = len(pnls)
            s = sum(pnls)
            avg = s / n if n else 0
            worst = min(pnls)
            best = max(pnls)
            marker = "⚠️ " if s < -20 else ("✓ " if s > 20 else "  ")
            p(f"    {marker}{grade+' + HTF '+htf:<24}  {n:>3}  "
              f"${s:>+9.2f}  ${avg:>+8.2f}  ${worst:>+8.2f}  ${best:>+8.2f}")

    # Per-trade detail (limit to 20 most recent)
    if per_trade:
        p("\n  TRADE-BY-TRADE (most recent 20)")
        p(f"    {'symbol':<16} {'dir':<6} {'gr':<3} {'htf':<9} "
          f"{'outcome':<8} {'pnl':>10} {'exit':<18}")
        for t in per_trade[-20:]:
            pnl_str = f"${t['pnl_usd']:+,.2f}" if t.get("pnl_usd") is not None else "—"
            p(f"    {t['symbol']:<16} {t['direction']:<6} "
              f"{t['grade']:<3} {t['htf_4h']:<9} "
              f"{t['outcome']:<8} {pnl_str:>10} "
              f"{(t.get('exit_reason') or '—'):<18}")

    p(_hr("═"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=7,
                    help="How many days of history to review (default: 7)")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of the human report")
    args = ap.parse_args()

    settings = Settings()
    if not settings.supabase_url or not settings.supabase_key:
        print("ERROR: SUPABASE_URL / SUPABASE_KEY not set.", file=sys.stderr)
        return 1

    db = Database(settings.supabase_url, settings.supabase_key)

    decisions = _fetch_agent_decisions(db, args.days)
    trades = _fetch_mm_trades(db, args.days)
    trade_to_dec = _match_trade_to_decision(decisions, trades)

    overall = _overall_counts(decisions)
    breakdown = _breakdown_by_profile(decisions)
    concerns = _concern_counts(decisions)
    outcomes = _approved_outcomes(decisions, trades, trade_to_dec)

    if args.json:
        # Render profile_pnl as plain dict for JSON
        profile_pnl_json = {}
        for (g, h), pnls in outcomes.get("profile_pnl", {}).items():
            profile_pnl_json[f"{g}|{h}"] = {
                "n": len(pnls),
                "sum_usd": round(sum(pnls), 2),
                "avg_usd": round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
            }
        out = {
            "days": args.days,
            "total_calls": sum(v["n"] for v in overall.values()),
            "overall": overall,
            "breakdown": breakdown,
            "concerns": dict(concerns),
            "approved_outcomes": {
                "counts": outcomes.get("counts") or {},
                "per_trade": outcomes.get("per_trade") or [],
                "profile_pnl": profile_pnl_json,
            },
        }
        print(json.dumps(out, indent=2, default=str))
        return 0

    print(render_report(args.days, decisions, overall, breakdown, concerns, outcomes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
