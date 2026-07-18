"""Positive-control hunt: find real textbook-ish W/M moments in major-coin
history and check the vision judge can approve them."""
import asyncio
import json
import sys

import ccxt
import pandas as pd

SCRATCH = "/Users/stuarta/tarakta/experiments/vision"
sys.path.insert(0, SCRATCH)
sys.path.insert(0, "/Users/stuarta/tarakta")
from chart_render import render_chart  # noqa: E402
from vision_call import vision_call, extract_json  # noqa: E402
from pathlib import Path  # noqa: E402

SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]


def fetch_year_4h(ex, symbol):
    frames, since = [], ex.milliseconds() - 370 * 24 * 3600 * 1000
    for _ in range(4):
        chunk = ex.fetch_ohlcv(symbol, "4h", since=since, limit=1000)
        if not chunk:
            break
        frames.extend(chunk)
        since = chunk[-1][0] + 1
        if len(chunk) < 1000:
            break
    df = pd.DataFrame(frames, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop(columns=["ts"]).drop_duplicates()


def find_candidates(df, kind="W", max_out=3):
    """Heuristic textbook moments: touch of a 30d extreme, sharp leave,
    second touch 6-60 bars later with a volume spike, judged 3 bars after."""
    out = []
    vol20 = df["volume"].rolling(20).mean()
    ext = (df["low"].rolling(180).min() if kind == "W"
           else df["high"].rolling(180).max())
    i = 200
    while i < len(df) - 70 and len(out) < max_out:
        px = df["low"].iloc[i] if kind == "W" else df["high"].iloc[i]
        level = ext.iloc[i]
        near = abs(px - level) / level < 0.004
        spike = df["volume"].iloc[i] > 2.0 * (vol20.iloc[i] or 1)
        if near and spike:
            # sharp leave: >=2.5% away within 5 bars
            fwd = df["close"].iloc[i + 1:i + 6]
            moved = ((fwd.max() - px) / px > 0.025) if kind == "W" else ((px - fwd.min()) / px > 0.025)
            if moved:
                # second touch within 6-60 bars: back within 1% of level
                for j in range(i + 6, min(i + 60, len(df) - 8)):
                    px2 = df["low"].iloc[j] if kind == "W" else df["high"].iloc[j]
                    if abs(px2 - level) / level < 0.01:
                        out.append((i, j, float(level)))
                        i = j + 60
                        break
                else:
                    i += 5
            else:
                i += 1
        else:
            i += 1
    return out


async def main():
    skill = Path('/Users/stuarta/tarakta/docs/agent-committee/skills/structure_vision_specialist_DRAFT.md').read_text()
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    cases = []
    for sym in SYMBOLS:
        df = fetch_year_4h(ex, sym)
        for kind in ("W", "M"):
            for (i, j, level) in find_candidates(df, kind):
                cut = j + 3  # decision moment: 3 bars after second touch
                window = df.iloc[:cut + 1]
                tag = f"{sym.split('/')[0]}_{kind}_{str(df.index[j])[:10]}"
                p = f"{SCRATCH}/pos_{tag}.png"
                lv = {"LEVEL": level,
                      "HOW": float(window['high'].tail(120).max()),
                      "LOW": float(window['low'].tail(120).min())}
                render_chart(window, p, f"{sym} 4H — as of {window.index[-1]:%Y-%m-%d %H:%M}",
                             levels=lv)
                cases.append((tag, kind, p))
        print(f"{sym}: scanned {len(df)} bars", flush=True)
    print(f"candidates found: {len(cases)}", flush=True)

    hits = 0
    for tag, kind, p in cases[:8]:
        prompt = (f"Trade candidate under review. Read the 4H chart file {p} . "
                  "Apply your checklist as of the final bar. Output your JSON verdict only.")
        try:
            raw, _ = await vision_call(skill, prompt)
            d = extract_json(raw) or {}
            ok = d.get("formation") == kind and (d.get("alignment") or -9) >= 1
            hits += ok
            print(f"{'YES' if ok else 'no ':3s} {tag:24s} formation={d.get('formation')} "
                  f"align={d.get('alignment')} loc={d.get('location_valid')} "
                  f"svc={d.get('svc_present')} concerns={[str(c)[:45] for c in (d.get('concerns') or [])[:2]]}",
                  flush=True)
        except Exception as e:
            print(f"ERR {tag:24s} {str(e)[:90]}", flush=True)
    print(f"\npositive gate: {hits}/{min(len(cases),8)} approved "
          f"({'PASSED' if hits >= 2 else 'NOT PASSED'} — need >=2)", flush=True)


asyncio.run(main())
