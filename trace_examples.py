"""Trace 2 more trade examples — no outcome, just the setup and entry."""
import asyncio
import sys
from datetime import datetime, timedelta, timezone

import ccxt.async_support as ccxt
import pandas as pd

sys.path.insert(0, ".")

from src.strategy.sweep_detector import SweepDetector
from src.strategy.volume import VolumeAnalyzer
from src.strategy.market_structure import MarketStructureAnalyzer
from src.strategy.sessions import SessionAnalyzer


async def fetch(exchange, symbol, tf, since, limit=1000):
    raw = await exchange.fetch_ohlcv(symbol, tf, since=since, limit=limit)
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def trace_signal(symbol, candles_1h, candles_4h, candles_1d, target):
    window_1h = candles_1h[candles_1h.index <= target].tail(200)
    now = target.to_pydatetime()

    session = SessionAnalyzer().analyze(window_1h, now=now)
    ms = MarketStructureAnalyzer()
    ms_1h = ms.analyze(window_1h, timeframe="1h")
    ms_4h = ms.analyze(candles_4h[candles_4h.index <= target].tail(200), timeframe="4h")
    ms_1d = ms.analyze(candles_1d[candles_1d.index <= target].tail(200), timeframe="1d")

    sweep = SweepDetector().detect(
        candles_1h=window_1h,
        asian_high=session.asian_high,
        asian_low=session.asian_low,
        swing_high=ms_1h.key_levels.get("swing_high"),
        swing_low=ms_1h.key_levels.get("swing_low"),
    )

    vol = VolumeAnalyzer()
    vol_result = vol.analyze(window_1h)

    atr_series = vol._atr(
        window_1h["high"].astype(float),
        window_1h["low"].astype(float),
        window_1h["close"].astype(float),
        period=14,
    )
    avg_vol = window_1h["volume"].astype(float).rolling(20).mean()

    # Print
    print(f"\n{'='*70}")
    print(f"TRADE: {symbol} @ {target.strftime('%b %d, %H:%M UTC')}")
    print(f"{'='*70}")

    print(f"\n--- OVERNIGHT RANGE (Asian 00:00-08:00 UTC) ---")
    print(f"  High: ${session.asian_high:,.2f}  |  Low: ${session.asian_low:,.2f}  |  Range: ${session.asian_high - session.asian_low:,.2f}")

    print(f"\n--- KEY LEVELS ---")
    print(f"  1H swing high: ${ms_1h.key_levels.get('swing_high', 0):,.2f}")
    print(f"  1H swing low:  ${ms_1h.key_levels.get('swing_low', 0):,.2f}")
    print(f"  HTF trend: 4H={ms_4h.trend}, Daily={ms_1d.trend}")

    print(f"\n--- THE SWEEP ---")
    print(f"  Last 8 candles:")
    print(f"  {'Time':>16s}  {'Open':>10s}  {'High':>10s}  {'Low':>10s}  {'Close':>10s}  {'Vol':>8s}  Note")
    for i in range(-8, 0):
        if abs(i) >= len(window_1h):
            continue
        row = window_1h.iloc[i]
        ts = window_1h.index[len(window_1h) + i]
        note = ""

        # Check for sweep
        if session.asian_low > 0 and row["low"] < session.asian_low and row["close"] > session.asian_low:
            note = "SWEEP below Asian low, closed above"
        elif session.asian_high > 0 and row["high"] > session.asian_high and row["close"] < session.asian_high:
            note = "SWEEP above Asian high, closed below"
        sw_low = ms_1h.key_levels.get("swing_low", 0)
        sw_high = ms_1h.key_levels.get("swing_high", 0)
        if not note and sw_low > 0 and row["low"] < sw_low and row["close"] > sw_low:
            note = f"SWEEP below swing low ${sw_low:,.0f}"
        if not note and sw_high > 0 and row["high"] > sw_high and row["close"] < sw_high:
            note = f"SWEEP above swing high ${sw_high:,.0f}"

        # Check for displacement
        idx = len(window_1h) + i
        body = abs(float(row["close"]) - float(row["open"]))
        atr_val = float(atr_series.iloc[idx]) if pd.notna(atr_series.iloc[idx]) else 0
        avg_v = float(avg_vol.iloc[idx]) if pd.notna(avg_vol.iloc[idx]) else 1
        body_r = body / atr_val if atr_val > 0 else 0
        vol_r = float(row["volume"]) / avg_v if avg_v > 0 else 0
        if body_r > 1.5 and vol_r > 1.5:
            d = "BULL" if row["close"] > row["open"] else "BEAR"
            if not note:
                note = f"DISPLACEMENT {d} ({body_r:.1f}x ATR, {vol_r:.1f}x vol)"
            else:
                note += f" + DISP {d}"

        print(f"  {ts.strftime('%m/%d %H:%M'):>16s}  {row['open']:>10,.1f}  {row['high']:>10,.1f}  {row['low']:>10,.1f}  {row['close']:>10,.1f}  {row['volume']:>8,.0f}  {note}")

    print(f"\n--- SIGNAL ---")
    print(f"  Sweep: {sweep.sweep_type} ({sweep.sweep_direction}) — wick to ${sweep.sweep_level:,.2f}, depth ${sweep.sweep_depth:,.2f}")
    print(f"  Displacement: {'YES' if vol_result.displacement_detected else 'NO'} — {vol_result.displacement_direction or 'n/a'} ({vol_result.displacement_strength:.0%} strength, {vol_result.relative_volume:.1f}x RVOL)")

    # Score
    score = 0
    components = []
    if sweep.sweep_detected:
        score += 40
        components.append("Sweep +40")
    if vol_result.displacement_detected and vol_result.displacement_direction == sweep.sweep_direction:
        score += 30
        components.append("Displacement +30")
    if ms_4h.trend == sweep.sweep_direction:
        score += 10
        components.append(f"4H trend +10")
    if ms_1d.trend == sweep.sweep_direction:
        score += 5
        components.append(f"1D trend +5")
    if session.in_post_kill_zone:
        score += 15
        components.append(f"Post-KZ +15")

    print(f"  Score: {' | '.join(components)} = {score}")
    print(f"  Session: {session.current_session} | Post-KZ: {session.in_post_kill_zone} ({session.post_kill_zone_name or 'n/a'})")

    # Entry
    entry = float(window_1h["close"].iloc[-1])
    if sweep.sweep_direction == "swing_low":
        sl = sweep.sweep_level * 0.995
        tp = sweep.target_level if sweep.target_level > entry else entry + abs(entry - sl) * 3
    else:
        sl = sweep.sweep_level * 1.005
        tp = sweep.target_level if sweep.target_level < entry else entry - abs(entry - sl) * 3

    sl_dist = abs(entry - sl)
    tp_dist = abs(tp - entry)
    rr = tp_dist / sl_dist if sl_dist > 0 else 0

    print(f"\n--- ENTRY ---")
    print(f"  Direction: {sweep.sweep_direction.upper()}")
    print(f"  Entry:      ${entry:,.2f}")
    print(f"  Stop Loss:  ${sl:,.2f}  (sweep wick ${sweep.sweep_level:,.2f} minus 0.5%)")
    print(f"  Take Profit: ${tp:,.2f}  (opposite liquidity)")
    print(f"  Risk:       ${sl_dist:,.2f}  |  Reward: ${tp_dist:,.2f}  |  R:R: {rr:.1f}")
    print(f"  Thesis: MMs swept {sweep.sweep_type.replace('_', ' ')} at ${sweep.sweep_level:,.2f}, ")
    if sweep.sweep_direction == "swing_low":
        print(f"           grabbed sell-side stops, now expect move UP toward ${tp:,.2f}")
    else:
        print(f"           grabbed buy-side stops, now expect move DOWN toward ${tp:,.2f}")


async def main():
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    since = int(datetime(2026, 2, 15, tzinfo=timezone.utc).timestamp() * 1000)

    # Example 1: LINK bearish signal (from backtest: swing_high sweep)
    # Example 2: SOL bullish signal
    examples = [
        ("LINK/USDT:USDT", pd.Timestamp("2026-03-04 16:00", tz="UTC")),
        ("SOL/USDT:USDT", pd.Timestamp("2026-03-02 16:00", tz="UTC")),
    ]

    for symbol, target in examples:
        c1h = await fetch(exchange, symbol, "1h", since)
        c4h = await fetch(exchange, symbol, "4h", since)
        c1d = await fetch(exchange, symbol, "1d", since)
        await asyncio.sleep(0.3)

        if c1h.empty:
            print(f"No data for {symbol}")
            continue

        trace_signal(symbol, c1h, c4h, c1d, target)

    await exchange.close()


asyncio.run(main())
