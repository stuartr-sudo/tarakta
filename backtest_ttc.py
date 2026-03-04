"""Backtest: Trade Travel Chill strategy on historical data.

Fetches 30 days of 1H candles from Binance for top pairs,
runs the sweep + displacement pipeline, and reports how many
signals would have fired.
"""
import asyncio
import sys
from datetime import datetime, timedelta, timezone

import ccxt.async_support as ccxt
import pandas as pd

# Add project root
sys.path.insert(0, ".")

from src.strategy.sweep_detector import SweepDetector
from src.strategy.volume import VolumeAnalyzer
from src.strategy.market_structure import MarketStructureAnalyzer
from src.strategy.sessions import SessionAnalyzer


# Top traded futures pairs
PAIRS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
    "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
    "DOT/USDT:USDT", "MATIC/USDT:USDT", "SUI/USDT:USDT", "ARB/USDT:USDT",
    "OP/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT",
]

LOOKBACK_DAYS = 30
ENTRY_THRESHOLD = 70
MIN_RR = 3.0


async def fetch_candles(exchange, symbol, timeframe, days):
    """Fetch historical candles."""
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_candles = []
    limit = 1000

    while True:
        try:
            raw = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except Exception as e:
            print(f"  Error fetching {symbol} {timeframe}: {e}")
            break

        if not raw:
            break

        all_candles.extend(raw)
        since = raw[-1][0] + 1

        if len(raw) < limit:
            break

        await asyncio.sleep(0.1)

    if not all_candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    return df


def simulate_pipeline(candles_1h, candles_4h, candles_1d):
    """Run the strategy pipeline on a sliding window and count signals."""
    sweep_det = SweepDetector()
    vol_analyzer = VolumeAnalyzer()
    ms_analyzer = MarketStructureAnalyzer()
    session_analyzer = SessionAnalyzer()

    signals = []
    sweep_only = 0
    displacement_only = 0
    both_count = 0
    neither = 0

    # Slide through 1H candles, simulating every 2-hour scan
    window_size = 200
    step = 2  # every 2 hours

    for i in range(window_size, len(candles_1h), step):
        window_1h = candles_1h.iloc[max(0, i - window_size):i]
        now = window_1h.index[-1].to_pydatetime()

        # Get 4H/1D windows up to this point
        w4h = candles_4h[candles_4h.index <= now].tail(200)
        w1d = candles_1d[candles_1d.index <= now].tail(200)

        # Market structure
        ms_1h = ms_analyzer.analyze(window_1h, timeframe="1h")
        ms_4h = ms_analyzer.analyze(w4h, timeframe="4h")
        ms_1d = ms_analyzer.analyze(w1d, timeframe="1d")

        # Session
        session = session_analyzer.analyze(window_1h, now=now)

        # Sweep detection
        sweep = sweep_det.detect(
            candles_1h=window_1h,
            asian_high=session.asian_high,
            asian_low=session.asian_low,
            swing_high=ms_1h.key_levels.get("swing_high"),
            swing_low=ms_1h.key_levels.get("swing_low"),
        )

        # Displacement check
        vol = vol_analyzer.analyze(window_1h)
        has_displacement = vol.displacement_detected
        disp_dir = vol.displacement_direction

        # Track what we see
        has_sweep = sweep.sweep_detected
        direction_match = has_displacement and disp_dir == sweep.sweep_direction

        if has_sweep and direction_match:
            both_count += 1

            # Score
            score = 40 + 30  # sweep + displacement

            # HTF alignment
            trend_4h = ms_4h.trend if ms_4h else "ranging"
            trend_1d = ms_1d.trend if ms_1d else "ranging"
            if trend_4h == sweep.sweep_direction:
                score += 10
            if trend_1d == sweep.sweep_direction:
                score += 5

            # Timing
            if session.in_post_kill_zone:
                score += 15

            # R:R check
            entry = float(window_1h["close"].iloc[-1])
            if sweep.sweep_direction == "bullish":
                sl = sweep.sweep_level * 0.995
                tp = sweep.target_level if sweep.target_level > entry else entry + abs(entry - sl) * MIN_RR
                sl_dist = abs(entry - sl)
                tp_dist = abs(tp - entry)
            else:
                sl = sweep.sweep_level * 1.005
                tp = sweep.target_level if sweep.target_level < entry else entry - abs(entry - sl) * MIN_RR
                sl_dist = abs(sl - entry)
                tp_dist = abs(entry - tp)

            rr = tp_dist / sl_dist if sl_dist > 0 else 0

            if score >= ENTRY_THRESHOLD and rr >= MIN_RR:
                signals.append({
                    "time": now,
                    "direction": sweep.sweep_direction,
                    "sweep_type": sweep.sweep_type,
                    "score": score,
                    "rr": round(rr, 1),
                    "entry": round(entry, 4),
                    "sl": round(sl, 4),
                    "tp": round(tp, 4),
                    "htf_4h": trend_4h,
                    "htf_1d": trend_1d,
                    "post_kz": session.in_post_kill_zone,
                })

        elif has_sweep and not direction_match:
            sweep_only += 1
        elif not has_sweep and has_displacement:
            displacement_only += 1
        else:
            neither += 1

    return signals, sweep_only, displacement_only, both_count, neither


async def main():
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    print(f"=== Trade Travel Chill Backtest ({LOOKBACK_DAYS} days) ===\n")
    print(f"Threshold: {ENTRY_THRESHOLD} | Min R:R: {MIN_RR} | Pairs: {len(PAIRS)}")
    print(f"Scanning every 2h | Lookback: 3 candles\n")

    total_signals = []
    total_sweeps_only = 0
    total_displacements_only = 0
    total_both = 0
    total_scans = 0

    for symbol in PAIRS:
        print(f"Fetching {symbol}...", end=" ", flush=True)

        candles_1h = await fetch_candles(exchange, symbol, "1h", LOOKBACK_DAYS)
        candles_4h = await fetch_candles(exchange, symbol, "4h", LOOKBACK_DAYS)
        candles_1d = await fetch_candles(exchange, symbol, "1d", LOOKBACK_DAYS + 5)

        if candles_1h.empty or len(candles_1h) < 200:
            print(f"insufficient data ({len(candles_1h)} candles)")
            continue

        signals, s_only, d_only, both, neither = simulate_pipeline(candles_1h, candles_4h, candles_1d)
        scans = (len(candles_1h) - 200) // 2

        total_signals.extend([(symbol, s) for s in signals])
        total_sweeps_only += s_only
        total_displacements_only += d_only
        total_both += both
        total_scans += scans

        print(f"{len(candles_1h)} candles | {scans} scans | {both} sweep+disp | {len(signals)} tradeable")

        await asyncio.sleep(0.5)

    await exchange.close()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY ({LOOKBACK_DAYS} days, {len(PAIRS)} pairs)")
    print(f"{'='*70}")
    print(f"Total scans:                 {total_scans}")
    print(f"Sweep detected (no disp):    {total_sweeps_only}")
    print(f"Displacement (no sweep):     {total_displacements_only}")
    print(f"Sweep + Displacement:        {total_both}")
    print(f"Tradeable signals (>=70, 3R): {len(total_signals)}")
    print(f"Signals per day:             {len(total_signals) / LOOKBACK_DAYS:.1f}")
    print(f"Signals per pair per day:    {len(total_signals) / LOOKBACK_DAYS / len(PAIRS):.2f}")

    if total_signals:
        print(f"\n{'='*70}")
        print(f"SIGNAL DETAILS (showing all {len(total_signals)})")
        print(f"{'='*70}")

        # Direction breakdown
        dirs = {}
        types = {}
        for sym, s in total_signals:
            d = s["direction"]
            t = s["sweep_type"]
            dirs[d] = dirs.get(d, 0) + 1
            types[t] = types.get(t, 0) + 1

        print(f"\nBy direction: {dirs}")
        print(f"By sweep type: {types}")

        # Score distribution
        scores = [s["score"] for _, s in total_signals]
        print(f"Score range: {min(scores)} - {max(scores)}")
        print(f"R:R range: {min(s['rr'] for _, s in total_signals)} - {max(s['rr'] for _, s in total_signals)}")

        # Show recent signals
        print(f"\nMost recent signals:")
        for sym, s in sorted(total_signals, key=lambda x: x[1]["time"], reverse=True)[:20]:
            print(f"  {s['time'].strftime('%m/%d %H:%M')} {sym:20s} {s['direction']:7s} "
                  f"{s['sweep_type']:12s} score={s['score']} R:R={s['rr']} "
                  f"{'POST-KZ' if s['post_kz'] else ''}")

    # Analysis of the displacement gate
    print(f"\n{'='*70}")
    print(f"GATE ANALYSIS")
    print(f"{'='*70}")
    total_with_sweep = total_sweeps_only + total_both
    if total_with_sweep > 0:
        pass_rate = total_both / total_with_sweep * 100
        print(f"Sweeps detected total:       {total_with_sweep}")
        print(f"Of those, had displacement:  {total_both} ({pass_rate:.0f}%)")
        print(f"Displacement gate blocks:    {total_sweeps_only} ({100-pass_rate:.0f}%)")
    else:
        print("No sweeps detected at all!")


if __name__ == "__main__":
    asyncio.run(main())
