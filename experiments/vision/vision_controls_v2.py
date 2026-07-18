"""Negative-control gate v2: same 4 synthetic charts, judged under the
course-derived vision skill (not naive shape-spotting)."""
import asyncio
import math
import random
import sys
from pathlib import Path

import pandas as pd

SCRATCH = "/Users/stuarta/tarakta/experiments/vision"
sys.path.insert(0, SCRATCH)
from chart_render import render_chart  # noqa: E402
from vision_call import vision_call, extract_json  # noqa: E402

rng = random.Random(42)


def bars(path_fn, n=140, vol_base=1000.0, spike_at=()):
    rows, price = [], path_fn(0.0)
    for i in range(n):
        target = path_fn(i / (n - 1))
        o = price
        c = price + (target - price) * 0.35 + rng.gauss(0, 0.25)
        hi = max(o, c) + abs(rng.gauss(0, 0.3))
        lo = min(o, c) - abs(rng.gauss(0, 0.3))
        v = vol_base * (1 + abs(rng.gauss(0, 0.3)))
        for s, mult in spike_at:
            if abs(i - s) <= 1:
                v *= mult
                if c > o:
                    lo -= 1.2
                else:
                    hi += 1.2
        rows.append((o, hi, lo, c, v))
        price = c
    idx = pd.date_range("2026-06-01", periods=n, freq="4h")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def w_path(t):
    # decision-moment framing: second low forms near the RIGHT EDGE
    if t < 0.45: return 112 - 42 * (t / 0.45)
    if t < 0.62: return 70 + 15 * ((t - 0.45) / 0.17)
    if t < 0.88: return 85 - 14 * ((t - 0.62) / 0.26)
    return 71 + 5 * ((t - 0.88) / 0.12)


def m_path(t): return 220 - (w_path(t) - 70)
def side_path(t): return 100 + 4 * math.sin(t * 22) + 2 * math.sin(t * 7)
def down_path(t): return 130 - 55 * t + 3 * math.sin(t * 9)


CASES = {
    "ctl_a.png": ("downtrend", down_path, (), lambda d: d.get("alignment", 9) <= -1 or d.get("formation") == "none"),
    "ctl_b.png": ("W bottom", w_path, ((63, 4.0), (123, 3.5)), lambda d: d.get("formation") == "W" and d.get("alignment", -9) >= 0),
    "ctl_c.png": ("chop", side_path, (), lambda d: d.get("alignment", 9) <= -1 or d.get("formation") == "none"),
    "ctl_d.png": ("M top", m_path, ((63, 4.0), (123, 3.5)), lambda d: d.get("formation") == "M" and d.get("alignment", -9) >= 0),
}


async def main():
    skill = Path('/Users/stuarta/tarakta/docs/agent-committee/skills/structure_vision_specialist_DRAFT.md').read_text()
    passed = 0
    for fname, (label, fn, spikes, ok_fn) in CASES.items():
        df = bars(fn, spike_at=spikes)
        levels = {"HOW": float(df["high"].tail(120).max()),
                  "LOW": float(df["low"].tail(120).min())}
        p = f"{SCRATCH}/{fname}"
        render_chart(df, p, "BTC/USDT 4H", levels=levels)
        prompt = (f"Trade candidate under review. Read the 4H chart file {p} . "
                  "Apply your checklist as of the final bar. Output your JSON verdict only.")
        try:
            raw, _ = await vision_call(skill, prompt)
            d = extract_json(raw) or {}
            ok = bool(ok_fn(d))
            passed += ok
            print(f"{'PASS' if ok else 'FAIL':4s} {label:10s} formation={d.get('formation')} "
                  f"align={d.get('alignment')} loc={d.get('location_valid')} "
                  f"svc={d.get('svc_present')} concerns={[str(c)[:40] for c in (d.get('concerns') or [])[:2]]}")
        except Exception as e:
            print(f"FAIL {label:10s} ERROR {str(e)[:100]}")
    print(f"\ngate v2: {passed}/4 {'PASSED' if passed == 4 else 'FAILED'}")


asyncio.run(main())
