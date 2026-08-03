"""Empirical test of the 'reverse all trades' hypothesis (diagnostic only).

Parses the per-symbol "Signals (N):" blocks from a replay_scan.py --pnl log,
then re-simulates each signal TWICE under identical simplified management
(single 2R target, -1R stop, 168h timeout, conservative both-touched=loss):
  1. original direction, original SL distance
  2. reversed direction, mirrored SL distance
Same ruler both sides — isolates directional edge from exit management.
Also reports direction split and forward market drift after each entry.

Usage:
  python3 scripts/replay_reversal_test.py /path/to/replay_pnl.log

First run + findings 2026-08-03 (180d x ETH/BTC/SOL/DOGE/ADA):
see docs/BACKTEST_180D_2026-08-03.md. Headline: raw reversal looks great
(24% -> 50% WR) but the entire effect is serial re-entry clusters; after
dedup to independent ideas BOTH directions are ~0R. Do NOT build an
inverse bot off this (tarakta-mm-inverse already lost live).
"""
import re
import sys
from datetime import datetime, timezone

import ccxt

LOG = sys.argv[1] if len(sys.argv) > 1 else None
if not LOG:
    raise SystemExit("usage: replay_reversal_test.py <replay_pnl_log>")

# ---- parse signals ---------------------------------------------------------
signals = []
symbol = None
sig_re = re.compile(r"^\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s+(long|short)\s+([A-F])\s")
entry_re = re.compile(r"entry=([\d.]+)\s+sl=([\d.]+)\s+tp1=([\d.]+)")
exit_re = re.compile(r"→ (\S+)\s+tiers=(\S+)\s+r=([+-][\d.]+)\s+pnl=\$([+-][\d,.]+)")

pending = None
for line in open(LOG):
    m = re.match(r"^REPLAY SUMMARY — (\S+)", line)
    if m:
        symbol = m.group(1)
        continue
    m = sig_re.match(line)
    if m and symbol:
        pending = {"symbol": symbol, "ts": m.group(1), "dir": m.group(2),
                   "grade": m.group(3)}
        continue
    if pending and (m := entry_re.search(line)):
        pending.update(entry=float(m.group(1)), sl=float(m.group(2)),
                       tp1=float(m.group(3)))
        continue
    if pending and (m := exit_re.search(line)):
        pending.update(orig_exit=m.group(1), orig_r=float(m.group(3)),
                       orig_pnl=float(m.group(4).replace(",", "")))
        signals.append(pending)
        pending = None

print(f"parsed {len(signals)} signals")
longs = [s for s in signals if s["dir"] == "long"]
print(f"direction split: {len(longs)} long / {len(signals) - len(longs)} short\n")
if not signals:
    raise SystemExit("no signals parsed — was the log produced with --pnl?")

# ---- fetch candles ---------------------------------------------------------
first_ts = min(datetime.strptime(s["ts"], "%Y-%m-%d %H:%M") for s in signals)
last_ts = max(datetime.strptime(s["ts"], "%Y-%m-%d %H:%M") for s in signals)
start_ms = int(first_ts.replace(tzinfo=timezone.utc).timestamp() * 1000)
# 168h sim window + slack after the last signal
end_ms = int(last_ts.replace(tzinfo=timezone.utc).timestamp() * 1000) + 200 * 3600_000

ex = ccxt.binance({"options": {"defaultType": "future"}})
candles = {}  # symbol -> list of [ts, o, h, l, c, v]
for sym in sorted({s["symbol"] for s in signals}):
    since = start_ms
    rows = []
    while since < end_ms:
        batch = ex.fetch_ohlcv(sym, "1h", since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 3600_000
    candles[sym] = rows
    print(f"fetched {len(rows)} 1h bars for {sym}", file=sys.stderr)


def bars_after(sym, ts_str):
    ts = int(datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
             .replace(tzinfo=timezone.utc).timestamp() * 1000)
    return [b for b in candles[sym] if b[0] > ts][:168]


def sim(entry, sl, tp, direction, bars):
    """-1R stop / +2R target / 168h timeout; both-touched-in-bar = loss."""
    for b in bars:
        _, o, h, l, c, _ = b[:6]
        if direction == "long":
            if l <= sl:
                return -1.0, "sl"
            if h >= tp:
                return 2.0, "tp"
        else:
            if h >= sl:
                return -1.0, "sl"
            if l <= tp:
                return 2.0, "tp"
    if not bars:
        return 0.0, "nodata"
    c = bars[-1][4]
    d = abs(entry - sl)
    r = (c - entry) / d if direction == "long" else (entry - c) / d
    return max(min(r, 2.0), -1.0), "timeout"


rows = []
for s in signals:
    bars = bars_after(s["symbol"], s["ts"])
    d = abs(s["entry"] - s["sl"])
    if s["dir"] == "long":
        o_r, o_x = sim(s["entry"], s["sl"], s["entry"] + 2 * d, "long", bars)
        r_r, r_x = sim(s["entry"], s["entry"] + d, s["entry"] - 2 * d, "short", bars)
    else:
        o_r, o_x = sim(s["entry"], s["sl"], s["entry"] - 2 * d, "short", bars)
        r_r, r_x = sim(s["entry"], s["entry"] - d, s["entry"] + 2 * d, "long", bars)
    # forward drift 24h/72h, signed toward the ORIGINAL trade direction
    drift = {}
    for h_n in (24, 72):
        if len(bars) >= h_n:
            chg = (bars[h_n - 1][4] - s["entry"]) / s["entry"] * 100
            drift[h_n] = chg if s["dir"] == "long" else -chg
    rows.append({**s, "o_r": o_r, "o_x": o_x, "r_r": r_r, "r_x": r_x,
                 "d24": drift.get(24), "d72": drift.get(72)})

print(f"{'symbol':<16}{'ts':<18}{'dir':<7}{'orig(full)':<12}"
      f"{'orig(2R sim)':<14}{'REV(2R sim)':<13}{'fwd24h%':<9}{'fwd72h%'}")
for r in rows:
    d24 = f"{r['d24']:+.2f}" if r["d24"] is not None else "—"
    d72 = f"{r['d72']:+.2f}" if r["d72"] is not None else "—"
    print(f"{r['symbol']:<16}{r['ts']:<18}{r['dir']:<7}"
          f"{r['orig_r']:+.2f}R {r['orig_exit'][:6]:<5}"
          f"{r['o_r']:+.2f}R {r['o_x'][:4]:<7}"
          f"{r['r_r']:+.2f}R {r['r_x'][:4]:<7}"
          f"{d24:<9}{d72}")

# ---- aggregate: raw and deduped (collapse same-symbol chains < 24h apart) --
def totals(rs):
    o = sum(r["o_r"] for r in rs)
    v = sum(r["r_r"] for r in rs)
    ow = sum(1 for r in rs if r["o_r"] > 0)
    rw = sum(1 for r in rs if r["r_r"] > 0)
    return o, v, ow, rw

deduped = []
last_by_sym = {}
for r in rows:
    ts = datetime.strptime(r["ts"], "%Y-%m-%d %H:%M")
    prev = last_by_sym.get(r["symbol"])
    if prev is None or (ts - prev).total_seconds() > 24 * 3600:
        deduped.append(r)
    last_by_sym[r["symbol"]] = ts

n = len(rows)
o_tot, r_tot, o_w, r_w = totals(rows)
full = sum(r["orig_r"] for r in rows)
print("\n================ VERDICT (same simplified 2R management both sides) ================")
print(f"original engine management : {full:+.2f}R total (the real backtest)")
print(f"original dir, 2R sim       : {o_tot:+.2f}R total, win rate {o_w}/{n} = {o_w/n*100:.0f}%")
print(f"REVERSED dir, 2R sim       : {r_tot:+.2f}R total, win rate {r_w}/{n} = {r_w/n*100:.0f}%")
do, dv, dow, drw = totals(deduped)
dn = len(deduped)
print(f"\nDEDUPED to independent ideas (same-symbol re-entries <24h apart collapsed): {dn} ideas")
print(f"original dir, 2R sim       : {do:+.2f}R, win rate {dow}/{dn}")
print(f"REVERSED dir, 2R sim       : {dv:+.2f}R, win rate {drw}/{dn}")
d24 = [r["d24"] for r in rows if r["d24"] is not None]
if d24:
    print(f"\navg market drift toward original trade: 24h {sum(d24)/len(d24):+.2f}%")
    adverse24 = sum(1 for x in d24 if x < 0)
    print(f"signals where market moved AGAINST the original direction within 24h: "
          f"{adverse24}/{len(d24)}")
