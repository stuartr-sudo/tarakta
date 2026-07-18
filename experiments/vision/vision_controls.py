"""Negative-control gate for chart vision: 4 synthetic charts, neutral names."""
import asyncio
import math
import random
import sys

import pandas as pd

sys.path.insert(0, "/Users/stuarta/tarakta/experiments/vision")
from chart_render import render_chart  # noqa: E402
from vision_call import vision_call, extract_json  # noqa: E402

DIR = "/Users/stuarta/tarakta/experiments/vision"
rng = random.Random(42)


def bars(path_fn, n=140, base=100.0, vol_base=1000.0, spike_at=()):
    rows, price = [], base
    for i in range(n):
        target = path_fn(i / (n - 1))
        drift = (target - price) * 0.35
        o = price
        c = price + drift + rng.gauss(0, 0.25)
        hi = max(o, c) + abs(rng.gauss(0, 0.3))
        lo = min(o, c) - abs(rng.gauss(0, 0.3))
        v = vol_base * (1 + abs(rng.gauss(0, 0.3)))
        for s, mult in spike_at:
            if abs(i - s) <= 1:
                v *= mult
                if c > o:
                    lo -= 1.2  # long lower wick at spike
                else:
                    hi += 1.2
        rows.append((o, hi, lo, c, v))
        price = c
    idx = pd.date_range("2026-06-01", periods=n, freq="4h")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def w_path(t):  # decline, low, bounce, retest, rally
    if t < 0.35: return 110 - 40 * (t / 0.35)
    if t < 0.5: return 70 + 18 * ((t - 0.35) / 0.15)
    if t < 0.65: return 88 - 16 * ((t - 0.5) / 0.15)
    return 72 + 45 * ((t - 0.65) / 0.35)


def m_path(t):
    return 220 - (w_path(t) - 70)


def side_path(t):
    return 100 + 4 * math.sin(t * 22) + 2 * math.sin(t * 7)


def down_path(t):
    return 130 - 55 * t + 3 * math.sin(t * 9)


CASES = {  # neutral filenames, no hints
    "chart_a.png": ("none", down_path, ()),
    "chart_b.png": ("W", w_path, ((49, 4.0), (88, 3.5))),
    "chart_c.png": ("none", side_path, ()),
    "chart_d.png": ("M", m_path, ((49, 4.0), (88, 3.5))),
}

SYSTEM = ("You are a precise chart pattern analyst. When asked about a chart "
          "image, read the file, look carefully, and answer with strict JSON only.")


async def main():
    results, passed = [], 0
    for fname, (truth, path_fn, spikes) in CASES.items():
        df = bars(path_fn, spike_at=spikes)
        p = f"{DIR}/{fname}"
        render_chart(df, p, "4H chart")
        prompt = (
            f"Read the file {p} . Then classify the dominant completed or "
            "forming pattern in the chart: 'M' (double top / M shape), 'W' "
            "(double bottom / W shape), or 'none' (no M or W present). "
            'Reply with JSON only: {"pattern": "M"|"W"|"none", "why": "<20 words"}'
        )
        try:
            raw, _ = await vision_call(SYSTEM, prompt)
            data = extract_json(raw) or {}
            got = str(data.get("pattern", "?"))
            ok = got == truth
            passed += ok
            results.append((fname, truth, got, ok, str(data.get("why", ""))[:80]))
        except Exception as e:
            results.append((fname, truth, "ERROR", False, str(e)[:80]))
    for r in results:
        print(f"{'PASS' if r[3] else 'FAIL':4s} {r[0]}  truth={r[1]:4s} got={r[2]:5s}  {r[4]}")
    print(f"\ngate: {passed}/4 {'PASSED' if passed == 4 else 'FAILED'}")


asyncio.run(main())
