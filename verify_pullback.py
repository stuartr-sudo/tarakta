"""Verify pullback refinement against the 3 trade examples."""
import asyncio
import sys
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import pandas as pd

sys.path.insert(0, ".")

from src.strategy.sweep_detector import SweepDetector
from src.strategy.volume import VolumeAnalyzer
from src.strategy.pullback import PullbackAnalyzer
from src.strategy.market_structure import MarketStructureAnalyzer
from src.strategy.sessions import SessionAnalyzer
from src.strategy.confluence import PostSweepEngine


async def fetch(exchange, symbol, tf, since, limit=1000):
    raw = await exchange.fetch_ohlcv(symbol, tf, since=since, limit=limit)
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def test_signal(symbol, candles_1h, candles_4h, candles_1d, target):
    window_1h = candles_1h[candles_1h.index <= target].tail(200)
    now = target.to_pydatetime()

    session = SessionAnalyzer().analyze(window_1h, now=now)
    ms = MarketStructureAnalyzer()
    ms_1h = ms.analyze(window_1h, timeframe="1h")
    ms_4h = ms.analyze(candles_4h[candles_4h.index <= target].tail(200), timeframe="4h")
    ms_1d = ms.analyze(candles_1d[candles_1d.index <= target].tail(200), timeframe="1d")

    vol = VolumeAnalyzer()
    vol_result = vol.analyze(window_1h)

    # Detect sweep with displacement direction preference (avoids pullback
    # candles being misread as conflicting sweeps)
    sweep = SweepDetector().detect(
        candles_1h=window_1h,
        asian_high=session.asian_high,
        asian_low=session.asian_low,
        swing_high=ms_1h.key_levels.get("swing_high"),
        swing_low=ms_1h.key_levels.get("swing_low"),
        lookback=8,
        prefer_direction=vol_result.displacement_direction,
    )

    # NEW: Pullback detection
    pullback_result = None
    if vol_result.displacement_detected and vol_result.displacement_candle_idx is not None:
        pb = PullbackAnalyzer()
        pullback_result = pb.analyze(
            candles_1h=window_1h,
            displacement_candle_idx=vol_result.displacement_candle_idx,
            direction=vol_result.displacement_direction,
        )

    # Resolve HTF direction
    trend_4h = ms_4h.trend
    trend_1d = ms_1d.trend
    if trend_4h == trend_1d and trend_4h != "ranging":
        htf_direction = trend_4h
    elif trend_4h != "ranging":
        htf_direction = trend_4h
    elif trend_1d != "ranging":
        htf_direction = trend_1d
    else:
        htf_direction = None

    # Score with new engine
    engine = PostSweepEngine(entry_threshold=70.0)
    ms_results = {"1h": ms_1h, "4h": ms_4h, "1d": ms_1d}
    signal = engine.score_signal(
        symbol=symbol,
        current_price=float(window_1h["close"].iloc[-1]),
        sweep_result=sweep,
        displacement_confirmed=vol_result.displacement_detected,
        displacement_direction=vol_result.displacement_direction,
        htf_direction=htf_direction,
        in_post_kill_zone=session.in_post_kill_zone,
        ms_results=ms_results,
        pullback_result=pullback_result,
    )

    print(f"\n{'='*60}")
    print(f"{symbol} @ {target.strftime('%b %d, %H:%M UTC')}")
    print(f"{'='*60}")
    print(f"  Sweep: {sweep.sweep_type} ({sweep.sweep_direction})")
    print(f"  Displacement: {'YES' if vol_result.displacement_detected else 'NO'} ({vol_result.displacement_direction})")

    if pullback_result:
        print(f"  Pullback: {pullback_result.pullback_status} ({pullback_result.retracement_pct:.0%} retracement)")
        print(f"    disp_open=${pullback_result.displacement_open:,.2f}  thrust_peak=${pullback_result.thrust_extreme:,.2f}  current=${pullback_result.current_price:,.2f}")
    else:
        print(f"  Pullback: N/A (no displacement)")

    print(f"  HTF: 4H={trend_4h}, 1D={trend_1d}")
    print(f"  Post-KZ: {session.in_post_kill_zone}")
    print(f"  Score: {signal.score}/100 (threshold: 70)")
    print(f"  Reasons: {signal.reasons}")

    would_trade = signal.score >= 70
    print(f"\n  >>> WOULD TRADE: {'YES' if would_trade else 'NO'}")

    if would_trade:
        entry = signal.entry_price
        old_entry = float(window_1h["close"].iloc[-1])
        print(f"  Entry: ${entry:,.2f} (was ${old_entry:,.2f}, improvement: ${abs(old_entry - entry):,.2f})")

        if sweep.sweep_direction == "bullish":
            sl = sweep.sweep_level * 0.995
            tp = sweep.target_level if sweep.target_level > entry else entry + abs(entry - sl) * 3
        else:
            sl = sweep.sweep_level * 1.005
            tp = sweep.target_level if sweep.target_level < entry else entry - abs(entry - sl) * 3

        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        print(f"  SL: ${sl:,.2f}  TP: ${tp:,.2f}  R:R: {rr:.1f}")

    return would_trade


async def main():
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    since = int(datetime(2026, 2, 15, tzinfo=timezone.utc).timestamp() * 1000)

    examples = [
        ("BTC/USDT:USDT", pd.Timestamp("2026-03-02 16:00", tz="UTC"), "Was profitable (+$3.8K)"),
        ("BTC/USDT:USDT", pd.Timestamp("2026-03-02 20:00", tz="UTC"), "BTC 4h later (pullback scan)"),
        ("SOL/USDT:USDT", pd.Timestamp("2026-03-02 16:00", tz="UTC"), "Was a LOSS (SL hit)"),
        ("SOL/USDT:USDT", pd.Timestamp("2026-03-02 20:00", tz="UTC"), "SOL 4h later (pullback scan)"),
        ("LINK/USDT:USDT", pd.Timestamp("2026-03-04 16:00", tz="UTC"), "Was flat"),
        ("LINK/USDT:USDT", pd.Timestamp("2026-03-04 20:00", tz="UTC"), "LINK 4h later (pullback scan)"),
    ]

    print("PULLBACK REFINEMENT VERIFICATION")
    print("=" * 60)
    print("Testing: would the pullback gate have improved outcomes?")
    print("  - BTC: should still trade (on pullback scan) with better entry")
    print("  - SOL: should be BLOCKED (V-shape, no pullback)")
    print("  - LINK: depends on pullback depth")

    for symbol, target, note in examples:
        c1h = await fetch(exchange, symbol, "1h", since)
        c4h = await fetch(exchange, symbol, "4h", since)
        c1d = await fetch(exchange, symbol, "1d", since)
        await asyncio.sleep(0.3)

        if c1h.empty:
            print(f"\nNo data for {symbol}")
            continue

        print(f"\n  [{note}]")
        test_signal(symbol, c1h, c4h, c1d, target)

    await exchange.close()


asyncio.run(main())
