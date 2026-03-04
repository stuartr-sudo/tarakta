"""Trace a single trade example from the backtest."""
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


async def main():
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    symbol = "BTC/USDT:USDT"
    # Fetch enough data around March 2
    since = int((datetime(2026, 2, 20, tzinfo=timezone.utc)).timestamp() * 1000)

    raw_1h = await exchange.fetch_ohlcv(symbol, "1h", since=since, limit=1000)
    raw_4h = await exchange.fetch_ohlcv(symbol, "4h", since=since, limit=500)
    raw_1d = await exchange.fetch_ohlcv(symbol, "1d", since=since, limit=100)
    await exchange.close()

    def to_df(raw):
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    candles_1h = to_df(raw_1h)
    candles_4h = to_df(raw_4h)
    candles_1d = to_df(raw_1d)

    # Find the window ending at March 2, 16:00 UTC
    target = pd.Timestamp("2026-03-02 16:00", tz="UTC")
    window_1h = candles_1h[candles_1h.index <= target].tail(200)
    now = target.to_pydatetime()

    print(f"=== TRADE TRACE: {symbol} @ {target} ===\n")

    # --- Step 1: What did the Asian session look like? ---
    session_analyzer = SessionAnalyzer()
    session = session_analyzer.analyze(window_1h, now=now)
    print(f"1. SESSION ANALYSIS")
    print(f"   Current session: {session.current_session}")
    print(f"   In post-kill-zone: {session.in_post_kill_zone} ({session.post_kill_zone_name})")
    print(f"   Asian High: ${session.asian_high:,.2f}")
    print(f"   Asian Low:  ${session.asian_low:,.2f}")
    print(f"   Asian Range: ${session.asian_high - session.asian_low:,.2f}")
    print()

    # --- Step 2: Market structure ---
    ms = MarketStructureAnalyzer()
    ms_1h = ms.analyze(window_1h, timeframe="1h")
    ms_4h = ms.analyze(candles_4h[candles_4h.index <= target].tail(200), timeframe="4h")
    ms_1d = ms.analyze(candles_1d[candles_1d.index <= target].tail(200), timeframe="1d")
    print(f"2. MARKET STRUCTURE")
    print(f"   1H trend: {ms_1h.trend}")
    print(f"   4H trend: {ms_4h.trend}")
    print(f"   1D trend: {ms_1d.trend}")
    print(f"   1H swing high: ${ms_1h.key_levels.get('swing_high', 0):,.2f}")
    print(f"   1H swing low:  ${ms_1h.key_levels.get('swing_low', 0):,.2f}")
    print()

    # --- Step 3: Sweep detection ---
    sweep_det = SweepDetector()
    sweep = sweep_det.detect(
        candles_1h=window_1h,
        asian_high=session.asian_high,
        asian_low=session.asian_low,
        swing_high=ms_1h.key_levels.get("swing_high"),
        swing_low=ms_1h.key_levels.get("swing_low"),
    )
    print(f"3. SWEEP DETECTION")
    print(f"   Sweep detected: {sweep.sweep_detected}")
    print(f"   Direction: {sweep.sweep_direction}")
    print(f"   Type: {sweep.sweep_type}")
    print(f"   Sweep level (wick extreme): ${sweep.sweep_level:,.2f}")
    print(f"   Target level (opposite side): ${sweep.target_level:,.2f}")
    print(f"   Sweep depth: ${sweep.sweep_depth:,.2f}")
    print()

    # Show the actual candles around the sweep
    print(f"   Last 6 candles (1H):")
    print(f"   {'Time':>20s}  {'Open':>10s}  {'High':>10s}  {'Low':>10s}  {'Close':>10s}  {'Volume':>12s}")
    for i in range(-6, 0):
        row = window_1h.iloc[i]
        ts = window_1h.index[len(window_1h) + i]
        marker = ""
        if row["low"] < session.asian_low and row["close"] > session.asian_low:
            marker = " <-- BULLISH SWEEP (wicked below Asian low, closed above)"
        elif row["high"] > session.asian_high and row["close"] < session.asian_high:
            marker = " <-- BEARISH SWEEP (wicked above Asian high, closed below)"
        # Check swing levels too
        sw_low = ms_1h.key_levels.get("swing_low", 0)
        sw_high = ms_1h.key_levels.get("swing_high", 0)
        if sw_low > 0 and row["low"] < sw_low and row["close"] > sw_low and not marker:
            marker = f" <-- BULLISH SWEEP (wicked below swing low ${sw_low:,.0f}, closed above)"
        if sw_high > 0 and row["high"] > sw_high and row["close"] < sw_high and not marker:
            marker = f" <-- BEARISH SWEEP (wicked above swing high ${sw_high:,.0f}, closed below)"
        print(f"   {str(ts):>20s}  {row['open']:>10,.1f}  {row['high']:>10,.1f}  {row['low']:>10,.1f}  {row['close']:>10,.1f}  {row['volume']:>12,.0f}{marker}")
    print()

    # --- Step 4: Displacement check ---
    vol = VolumeAnalyzer()
    vol_result = vol.analyze(window_1h)
    print(f"4. DISPLACEMENT CHECK")
    print(f"   Displacement detected: {vol_result.displacement_detected}")
    print(f"   Direction: {vol_result.displacement_direction}")
    print(f"   Strength: {vol_result.displacement_strength:.0%}")
    print(f"   Relative volume: {vol_result.relative_volume:.1f}x")

    # Show what the displacement candle looked like
    atr_series = vol._atr(
        window_1h["high"].astype(float),
        window_1h["low"].astype(float),
        window_1h["close"].astype(float),
        period=14,
    )
    avg_vol = window_1h["volume"].astype(float).rolling(20).mean()
    for i in range(-3, 0):
        idx = len(window_1h) + i
        body = abs(float(window_1h["close"].iloc[idx]) - float(window_1h["open"].iloc[idx]))
        candle_vol = float(window_1h["volume"].iloc[idx])
        atr_val = float(atr_series.iloc[idx]) if pd.notna(atr_series.iloc[idx]) else 0
        avg_v = float(avg_vol.iloc[idx]) if pd.notna(avg_vol.iloc[idx]) else 1
        body_ratio = body / atr_val if atr_val > 0 else 0
        vol_ratio = candle_vol / avg_v if avg_v > 0 else 0
        is_disp = body_ratio > 1.5 and vol_ratio > 1.5
        ts = window_1h.index[idx]
        direction = "bullish" if float(window_1h["close"].iloc[idx]) > float(window_1h["open"].iloc[idx]) else "bearish"
        print(f"   Candle {ts}: body={body:.1f} ({body_ratio:.1f}x ATR), vol={vol_ratio:.1f}x avg "
              f"{'-> DISPLACEMENT ' + direction.upper() if is_disp else ''}")
    print()

    # --- Step 5: Score ---
    score = 0
    reasons = []
    if sweep.sweep_detected:
        score += 40
        reasons.append(f"Sweep: {sweep.sweep_type} ({sweep.sweep_direction})")
    if vol_result.displacement_detected and vol_result.displacement_direction == sweep.sweep_direction:
        score += 30
        reasons.append(f"Displacement: {vol_result.displacement_direction}")
    if ms_4h.trend == sweep.sweep_direction:
        score += 10
        reasons.append(f"4H trend aligned: {ms_4h.trend}")
    if ms_1d.trend == sweep.sweep_direction:
        score += 5
        reasons.append(f"1D trend aligned: {ms_1d.trend}")
    if session.in_post_kill_zone:
        score += 15
        reasons.append(f"Post-kill-zone: {session.post_kill_zone_name}")

    print(f"5. SCORING")
    for r in reasons:
        print(f"   + {r}")
    print(f"   = {score} / 100 (threshold: 70)")
    print()

    # --- Step 6: Entry execution ---
    entry = float(window_1h["close"].iloc[-1])
    if sweep.sweep_direction == "bullish":
        sl = sweep.sweep_level * 0.995
        tp = sweep.target_level if sweep.target_level > entry else entry + abs(entry - sl) * 3
    else:
        sl = sweep.sweep_level * 1.005
        tp = sweep.target_level if sweep.target_level < entry else entry - abs(entry - sl) * 3

    sl_dist = abs(entry - sl)
    tp_dist = abs(tp - entry)
    rr = tp_dist / sl_dist if sl_dist > 0 else 0

    print(f"6. TRADE EXECUTION")
    print(f"   Direction: {sweep.sweep_direction.upper()}")
    print(f"   Entry:     ${entry:,.2f}")
    print(f"   Stop Loss: ${sl:,.2f} (behind sweep wick ${sweep.sweep_level:,.2f} + 0.5% buffer)")
    print(f"   Take Profit: ${tp:,.2f} (opposite liquidity)")
    print(f"   SL distance: ${sl_dist:,.2f}")
    print(f"   TP distance: ${tp_dist:,.2f}")
    print(f"   R:R ratio:   {rr:.1f}")
    print()

    # --- Step 7: What happened next? ---
    print(f"7. WHAT HAPPENED NEXT?")
    future = candles_1h[candles_1h.index > target].head(24)
    if not future.empty:
        hit_tp = False
        hit_sl = False
        for _, row in future.iterrows():
            ts = _
            if sweep.sweep_direction == "bullish":
                if row["low"] <= sl:
                    hit_sl = True
                    print(f"   {ts}: SL HIT at ${sl:,.2f} (low was ${row['low']:,.2f}) -> LOSS")
                    break
                if row["high"] >= tp:
                    hit_tp = True
                    print(f"   {ts}: TP HIT at ${tp:,.2f} (high was ${row['high']:,.2f}) -> WIN")
                    break
            else:
                if row["high"] >= sl:
                    hit_sl = True
                    print(f"   {ts}: SL HIT at ${sl:,.2f} (high was ${row['high']:,.2f}) -> LOSS")
                    break
                if row["low"] <= tp:
                    hit_tp = True
                    print(f"   {ts}: TP HIT at ${tp:,.2f} (low was ${row['low']:,.2f}) -> WIN")
                    break

        if not hit_tp and not hit_sl:
            last_price = float(future["close"].iloc[-1])
            unrealized = (last_price - entry) if sweep.sweep_direction == "bullish" else (entry - last_price)
            print(f"   Neither TP nor SL hit in next 24 hours")
            print(f"   Last price: ${last_price:,.2f} (unrealized: ${unrealized:,.2f})")

        print(f"\n   Price action after entry:")
        print(f"   {'Time':>20s}  {'High':>10s}  {'Low':>10s}  {'Close':>10s}")
        for ts, row in future.iterrows():
            marker = ""
            if sweep.sweep_direction == "bullish":
                if row["high"] >= tp:
                    marker = " <-- TP"
                if row["low"] <= sl:
                    marker = " <-- SL"
            else:
                if row["low"] <= tp:
                    marker = " <-- TP"
                if row["high"] >= sl:
                    marker = " <-- SL"
            print(f"   {str(ts):>20s}  {row['high']:>10,.1f}  {row['low']:>10,.1f}  {row['close']:>10,.1f}{marker}")


asyncio.run(main())
