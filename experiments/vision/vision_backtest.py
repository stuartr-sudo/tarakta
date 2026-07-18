"""Vision structure-agent backtest over historical trades.

Phase 1: fetch Binance candles as-of each trade's entry (public API), render
4H + 1H charts. Phase 2: vision agent judges each chart pair using the
course-derived skill. Scored against known outcomes. No DB writes.
"""
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

SCRATCH = "/Users/stuarta/tarakta/experiments/vision"
sys.path.insert(0, SCRATCH)
sys.path.insert(0, "/Users/stuarta/tarakta")

from chart_render import render_chart  # noqa: E402
from vision_call import vision_call, extract_json  # noqa: E402
from src.config import Settings  # noqa: E402
from src.data.db import Database  # noqa: E402

OUT = f"{SCRATCH}/results/vision_backtest_results.jsonl"
CHART_DIR = f"{SCRATCH}/charts"
SKILL_PATH = '/Users/stuarta/tarakta/docs/agent-committee/skills/structure_vision_specialist_DRAFT.md'

COLS = ("id,symbol,direction,entry_price,mm_confluence_grade,entry_time,"
        "exit_reason,pnl_usd")


def fetch_frame(ex, symbol, timeframe, entry_dt, bars=320):
    tf_ms = {"1h": 3600_000, "4h": 14400_000}[timeframe]
    since = int(entry_dt.timestamp() * 1000) - bars * tf_ms
    ohlcv = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=bars + 5)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    # only bars fully CLOSED before entry (replay-faithful)
    cutoff = entry_dt - timedelta(milliseconds=tf_ms)
    return df[df.index <= cutoff].drop(columns=["ts"])


def week_levels(df_1h, entry_dt):
    # approximate week start: most recent Sunday 21:00 UTC before entry
    d = entry_dt
    days_back = (d.weekday() - 6) % 7  # Sunday=6
    start = (d - timedelta(days=days_back)).replace(hour=21, minute=0, second=0, microsecond=0)
    if start > d:
        start -= timedelta(days=7)
    wk = df_1h[df_1h.index >= start]
    if len(wk) < 3:
        return {}
    return {"HOW": float(wk["high"].max()), "LOW": float(wk["low"].min())}


def prepare_charts(rows):
    Path(CHART_DIR).mkdir(exist_ok=True)
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    prepared = []
    for row in rows:
        rid = row["id"][:8]
        entry_dt = datetime.fromisoformat(str(row["entry_time"])).astimezone(timezone.utc)
        try:
            df4 = fetch_frame(ex, row["symbol"], "4h", entry_dt)
            df1 = fetch_frame(ex, row["symbol"], "1h", entry_dt)
            if len(df4) < 60 or len(df1) < 60:
                raise RuntimeError(f"insufficient candles 4h={len(df4)} 1h={len(df1)}")
            lv = week_levels(df1, entry_dt)
            p4 = render_chart(df4, f"{CHART_DIR}/{rid}_4h.png",
                              f"{row['symbol']} 4H — as of {entry_dt:%Y-%m-%d %H:%M} UTC",
                              levels=lv)
            p1 = render_chart(df1, f"{CHART_DIR}/{rid}_1h.png",
                              f"{row['symbol']} 1H — as of {entry_dt:%Y-%m-%d %H:%M} UTC",
                              levels=lv)
            prepared.append((row, p4, p1))
            print(f"charts ok  {row['symbol']:16s} {rid}", flush=True)
        except Exception as e:
            print(f"charts FAIL {row['symbol']:16s} {rid}: {str(e)[:90]}", flush=True)
    return prepared


async def judge(skill, row, p4, p1, sem, done):
    if row["id"] in done:
        return
    async with sem:
        prompt = (
            f"Trade candidate: {row['symbol']} {row['direction'].upper()} at "
            f"{row['entry_price']}. Read the 4H chart file {p4} and then the "
            f"1H chart file {p1} . Apply your checklist as of the final bar of "
            "each chart. Is there a valid M or W setup per the course rules "
            "supporting this trade? Output your JSON verdict only."
        )
        rec = {"id": row["id"], "symbol": row["symbol"],
               "exit_reason": row["exit_reason"],
               "pnl": round(float(row["pnl_usd"] or 0), 2),
               "grade": row.get("mm_confluence_grade")}
        try:
            raw, _ = await vision_call(skill, prompt, timeout_s=240)
            data = extract_json(raw) or {}
            rec.update({
                "formation": data.get("formation"),
                "alignment": data.get("alignment"),
                "location_valid": data.get("location_valid"),
                "svc_present": data.get("svc_present"),
                "confidence": data.get("confidence"),
                "concerns": (data.get("concerns") or [])[:3],
            })
        except Exception as e:
            rec["error"] = str(e)[:180]
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        a = rec.get("alignment")
        print(f"judged {row['symbol']:16s} {row['exit_reason']:18s} "
              f"pnl={rec['pnl']:>9.2f} align={a} loc={rec.get('location_valid')}",
              flush=True)


async def main():
    skill = Path(SKILL_PATH).read_text()
    cfg = Settings()
    db = Database(cfg.supabase_url, cfg.supabase_key)
    winners = (db.table("trades").select(COLS).eq("instance_id", "tarakta-mm")
               .gte("exit_time", "2026-05-01")
               .in_("exit_reason", ["tp_l3", "friday_uk_exit", "volume_degradation"])
               .execute().data)
    stops = (db.table("trades").select(COLS).eq("instance_id", "tarakta-mm")
             .gte("exit_time", "2026-05-01").eq("exit_reason", "stop_loss")
             .order("pnl_usd").limit(20).execute().data)
    scratches = (db.table("trades").select(COLS).eq("instance_id", "tarakta-mm")
                 .gte("exit_time", "2026-05-01").eq("exit_reason", "scratch_2h")
                 .order("pnl_usd").limit(5).execute().data)
    # April agent-era APPROVEs — the profitable configuration's picks
    april = (db.table("mm_agent_decisions")
             .select("id,symbol,direction,created_at,input_context")
             .eq("decision", "APPROVE").lte("created_at", "2026-05-01")
             .execute().data)
    april_rows = []
    for a in april:
        ictx = a.get("input_context") or {}
        entry = ictx.get("entry_price")
        if not entry:
            continue
        april_rows.append({
            "id": f"apr-{a['id']}"[:36], "symbol": a["symbol"],
            "direction": a.get("direction") or ictx.get("direction") or "long",
            "entry_price": entry, "entry_time": a["created_at"],
            "exit_reason": "april_approved", "pnl_usd": 0,
            "mm_confluence_grade": ictx.get("grade"),
        })
    rows = winners + stops + scratches + april_rows
    print(f"sample: {len(rows)} trades ({len(april_rows)} april-approved)", flush=True)

    done = set()
    try:
        with open(OUT) as f:
            done = {json.loads(l)["id"] for l in f}
    except FileNotFoundError:
        pass

    prepared = prepare_charts([r for r in rows if r["id"] not in done])
    sem = asyncio.Semaphore(3)
    await asyncio.gather(*[judge(skill, r, p4, p1, sem, done) for r, p4, p1 in prepared])

    # scoreboard
    recs = [json.loads(l) for l in open(OUT)]
    def bucket(name, sel):
        b = [r for r in recs if sel(r) and "error" not in r]
        if not b:
            print(f"{name}: none judged")
            return
        pos = [r for r in b if (r.get("alignment") or -9) >= 1]
        neg = [r for r in b if (r.get("alignment") or -9) <= -1]
        print(f"{name}: n={len(b)} align>=+1: {len(pos)}  align<=-1: {len(neg)}  "
              f"mean_align={sum((r.get('alignment') or 0) for r in b)/len(b):.2f}  "
              f"loc_valid={sum(1 for r in b if r.get('location_valid'))}/{len(b)}")
    W = ("tp_l3", "friday_uk_exit", "volume_degradation")
    print("\n===== VISION BACKTEST SCOREBOARD =====")
    bucket("winners      ", lambda r: r["exit_reason"] in W)
    bucket("stop_losses  ", lambda r: r["exit_reason"] == "stop_loss")
    bucket("scratches    ", lambda r: r["exit_reason"] == "scratch_2h")
    bucket("april_approved", lambda r: r["exit_reason"] == "april_approved")
    errs = [r for r in recs if "error" in r]
    print(f"errors: {len(errs)}")


asyncio.run(main())
