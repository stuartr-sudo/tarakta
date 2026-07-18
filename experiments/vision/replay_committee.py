"""Committee replay over historical closed trades.

Rebuilds each trade's context from its DB row and asks the NEW committee to
judge it. Nothing is written to the DB and no orders are placed — pure
retrospective judgment scored against known outcomes.
"""
import asyncio
import json
import sys
from datetime import datetime

sys.path.insert(0, "/Users/stuarta/tarakta")

from src.config import Settings
from src.data.db import Database
from src.strategy.mm_committee import MMCommittee

OUT = "/Users/stuarta/tarakta/experiments/vision/results/replay_results.jsonl"

COLS = (
    "id,symbol,direction,entry_price,original_stop_loss,mm_formation,"
    "mm_entry_type,mm_confluence_grade,confluence_score,mm_cycle_phase,"
    "htf_trend_4h,htf_trend_1d,counter_trend,entry_time,exit_reason,"
    "pnl_usd,formation_timeframe"
)


def build_ctx(row: dict) -> dict:
    ts = str(row["entry_time"])
    try:
        dt = datetime.fromisoformat(ts)
        hour, dow = dt.hour, dt.strftime("%A")
    except ValueError:
        hour, dow = 12, "unknown"
    session = "asia" if hour < 7 else ("uk" if hour < 13 else "us")
    return {
        "symbol": row["symbol"],
        "direction": row["direction"],
        "formation_type": row.get("mm_formation") or "?",
        "formation_variant": row.get("mm_entry_type") or "standard",
        "formation_timeframe": row.get("formation_timeframe") or "1h",
        "entry_price": float(row["entry_price"]),
        "sl_ref": float(row["original_stop_loss"]),
        "grade": row.get("mm_confluence_grade") or "?",
        "score_pct": float(row.get("confluence_score") or 0),
        "htf_trend_4h": row.get("htf_trend_4h") or "unknown",
        "htf_trend_1d": row.get("htf_trend_1d") or "unknown",
        "counter_trend": bool(row.get("counter_trend")),
        "weekly_phase": (row.get("mm_cycle_phase") or "").lower(),
        "session_name": session,
        "minutes_in": None,
        "dow": dow,
        "recent_trades": [],
        "data_quality_note": (
            "replay from persisted trade row: flow data unavailable, session "
            "approximated from entry hour, retest/accel fields unavailable"
        ),
    }


async def judge(committee, client, row, sem, done_ids):
    if row["id"] in done_ids:
        return None
    async with sem:
        ctx = build_ctx(row)
        rec = {
            "id": row["id"],
            "symbol": row["symbol"],
            "exit_reason": row["exit_reason"],
            "pnl": round(float(row["pnl_usd"] or 0), 2),
            "grade": ctx["grade"],
        }
        try:
            verdict, blob = await asyncio.wait_for(
                committee._run_committee(client, ctx), timeout=300
            )
            rec["decision"] = verdict.decision
            rec["confidence"] = verdict.confidence
            rec["reason"] = verdict.reason[:220]
            rec["specialists"] = {
                s.get("name"): s.get("alignment")
                for s in (blob.get("specialists") or [])
                if isinstance(s, dict)
            }
        except Exception as e:
            rec["decision"] = "ERROR"
            rec["reason"] = str(e)[:220]
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{rec['decision']:8s} {row['exit_reason']:18s} "
              f"pnl={rec['pnl']:>9.2f} {row['symbol']}", flush=True)
        return rec


async def main() -> None:
    cfg = Settings()
    db = Database(cfg.supabase_url, cfg.supabase_key)

    base = (
        db.table("trades").select(COLS)
        .eq("instance_id", "tarakta-mm")
        .gte("exit_time", "2026-05-01")
    )
    winners = (
        base.in_("exit_reason", ["tp_l3", "friday_uk_exit", "volume_degradation"])
        .execute().data
    )
    stops = (
        db.table("trades").select(COLS)
        .eq("instance_id", "tarakta-mm").gte("exit_time", "2026-05-01")
        .eq("exit_reason", "stop_loss").order("pnl_usd").limit(20)
        .execute().data
    )
    scratches = (
        db.table("trades").select(COLS)
        .eq("instance_id", "tarakta-mm").gte("exit_time", "2026-05-01")
        .eq("exit_reason", "scratch_2h").order("pnl_usd").limit(5)
        .execute().data
    )
    rows = winners + stops + scratches
    print(f"replaying {len(rows)} trades "
          f"({len(winners)} winners, {len(stops)} stops, {len(scratches)} scratches)",
          flush=True)

    done_ids = set()
    try:
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
    except FileNotFoundError:
        pass
    if done_ids:
        print(f"resuming: {len(done_ids)} already judged", flush=True)

    committee = MMCommittee(config=cfg, repo=None)
    client = committee._get_client()
    if client is None:
        print("FATAL: no committee client available", flush=True)
        return
    sem = asyncio.Semaphore(3)
    results = [r for r in await asyncio.gather(
        *[judge(committee, client, row, sem, done_ids) for row in rows]
    ) if r]

    # Merge previously-done records for the scoreboard
    all_recs = {}
    try:
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                all_recs[rec["id"]] = rec
    except FileNotFoundError:
        pass

    def bucket(name, recs):
        n = len(recs)
        veto = [r for r in recs if r["decision"] == "VETO"]
        appr = [r for r in recs if r["decision"] == "APPROVE"]
        err = [r for r in recs if r["decision"] == "ERROR"]
        print(f"\n== {name}: n={n} veto={len(veto)} approve={len(appr)} "
              f"error={len(err)}")
        print(f"   pnl of VETOed:   {sum(r['pnl'] for r in veto):>10.2f}")
        print(f"   pnl of APPROVEd: {sum(r['pnl'] for r in appr):>10.2f}")

    recs = list(all_recs.values())
    print("\n========== SCOREBOARD ==========")
    bucket("winners", [r for r in recs if r["exit_reason"] in
                       ("tp_l3", "friday_uk_exit", "volume_degradation")])
    bucket("stop_losses", [r for r in recs if r["exit_reason"] == "stop_loss"])
    bucket("scratches", [r for r in recs if r["exit_reason"] == "scratch_2h"])
    ok = [r for r in recs if r["decision"] in ("APPROVE", "VETO")]
    actual = sum(r["pnl"] for r in ok)
    filtered = sum(r["pnl"] for r in ok if r["decision"] == "APPROVE")
    print(f"\nSample P&L actual (judged trades):        {actual:>10.2f}")
    print(f"Sample P&L if committee had filtered:     {filtered:>10.2f}")
    print(f"Committee improvement on this sample:     {filtered - actual:>10.2f}")


asyncio.run(main())
