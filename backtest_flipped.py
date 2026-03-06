"""FLIPPED direction backtest — inverts every trade the original strategy takes.

Hypothesis: if the strategy times entries correctly but picks the wrong direction,
flipping every trade should invert the win rate.

SL/TP for flipped trades uses the sweep's opposite levels:
- Original bullish → Flipped SHORT: SL above target_level, TP below sweep_level
- Original bearish → Flipped LONG: SL below target_level, TP above sweep_level

This is the PRE-leverage-intelligence version of the strategy (no funding/OI/LS data).

Usage:
    python backtest_flipped.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time as time_mod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import ccxt.async_support as ccxt
import pandas as pd

from src.exchange.client import BinanceFuturesClient
from src.strategy.confluence import PostSweepEngine
from src.strategy.market_structure import MarketStructureAnalyzer
from src.strategy.pullback import PullbackAnalyzer
from src.strategy.sessions import SessionAnalyzer
from src.strategy.sweep_detector import SweepDetector
from src.strategy.volume import VolumeAnalyzer

# ── Config ──────────────────────────────────────────────────────────
QUALITY_BASES = BinanceFuturesClient.QUALITY_BASES
FEE_RATE = 0.0004          # 0.04% per side (Binance Futures maker)
INITIAL_BALANCE = 2000.0
LEVERAGE = 3
MAX_RISK_PCT = 0.10
MAX_POSITION_PCT = 0.25
MAX_CONCURRENT = 100
TRAILING_ACTIVATION_RR = 2.0
TRAILING_ATR_MULT = 1.5
MIN_RR = 2.0
COOLDOWN_HOURS = 2.0
ENTRY_THRESHOLD = 70.0
BATCH_SIZE = 8
MIN_VOLUME_USD = 20_000_000
BACKTEST_DAYS = 7
WARMUP_1H = 80


@dataclass
class BacktestPosition:
    symbol: str
    direction: str           # "long" or "short"
    entry_price: float
    quantity: float
    sl: float
    tp: float
    entry_time: datetime
    high_water_mark: float
    original_sl: float
    cost_usd: float
    atr_1h: float
    score: float
    reasons: list[str] = field(default_factory=list)
    exit_price: float = 0.0
    exit_time: datetime | None = None
    exit_reason: str = ""
    pnl: float = 0.0


class FlippedBacktestEngine:
    """Same as BacktestEngine but flips direction on every trade."""

    def __init__(self) -> None:
        self.exchange: ccxt.binance | None = None
        self.ms_analyzer = MarketStructureAnalyzer()
        self.vol_analyzer = VolumeAnalyzer()
        self.session_analyzer = SessionAnalyzer()
        self.sweep_detector = SweepDetector()
        self.pullback_analyzer = PullbackAnalyzer(min_retracement=0.20, max_retracement=0.78)
        self.confluence = PostSweepEngine(entry_threshold=ENTRY_THRESHOLD)

        self.balance = INITIAL_BALANCE
        self.peak_balance = INITIAL_BALANCE
        self.max_drawdown = 0.0
        self.positions: list[BacktestPosition] = []
        self.closed_trades: list[BacktestPosition] = []
        self.cooldowns: dict[str, datetime] = {}
        self._last_request_time: float = 0

    async def run(self) -> None:
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "timeout": 30000,
            "options": {"defaultType": "future", "fetchCurrencies": False},
        })

        try:
            await self.exchange.load_markets()
            pairs = self._get_quality_pairs()
            print(f"[FLIPPED] Found {len(pairs)} quality futures pairs")

            all_candles = await self._fetch_all_candles(pairs)
            valid_pairs = [p for p in pairs if p in all_candles and "1h" in all_candles[p]]
            print(f"[FLIPPED] Fetched candles for {len(valid_pairs)} pairs")

            now = datetime.now(timezone.utc)
            bt_start = now - timedelta(days=BACKTEST_DAYS)

            all_times: set[datetime] = set()
            for sym in valid_pairs:
                df = all_candles[sym]["1h"]
                mask = df.index >= bt_start
                all_times.update(df.index[mask].tolist())
            all_times_sorted = sorted(all_times)
            print(f"[FLIPPED] Stepping through {len(all_times_sorted)} hourly candles ({bt_start.strftime('%b %d')} → {now.strftime('%b %d')})")
            print()

            signals_generated = 0
            for i, current_time in enumerate(all_times_sorted):
                self._monitor_positions(current_time, all_candles)

                for sym in valid_pairs:
                    if sym not in all_candles or "1h" not in all_candles[sym]:
                        continue
                    if any(p.symbol == sym for p in self.positions):
                        continue
                    cd = self.cooldowns.get(sym)
                    if cd and current_time < cd:
                        continue

                    signal = self._analyze_at_time(sym, all_candles[sym], current_time)
                    if signal is None:
                        continue

                    signals_generated += 1
                    self._try_enter_flipped(signal, current_time)

                equity = self._equity(current_time, all_candles)
                if equity > self.peak_balance:
                    self.peak_balance = equity
                dd = (self.peak_balance - equity) / self.peak_balance if self.peak_balance > 0 else 0
                if dd > self.max_drawdown:
                    self.max_drawdown = dd

                if (i + 1) % 24 == 0 or i == len(all_times_sorted) - 1:
                    pct = (i + 1) / len(all_times_sorted) * 100
                    print(f"\r  Progress: {pct:.0f}% ({i+1}/{len(all_times_sorted)}) | "
                          f"Balance: ${self.balance:.2f} | Open: {len(self.positions)} | "
                          f"Closed: {len(self.closed_trades)}", end="", flush=True)

            if all_times_sorted:
                last_time = all_times_sorted[-1]
                for pos in list(self.positions):
                    df = all_candles.get(pos.symbol, {}).get("1h")
                    if df is not None and last_time in df.index:
                        close_price = float(df.loc[last_time, "close"])
                    else:
                        close_price = pos.entry_price
                    self._close_position(pos, close_price, last_time, "backtest_end")

            print("\n")
            self._print_report(signals_generated, len(valid_pairs), bt_start, now)

        finally:
            await self.exchange.close()

    # ── Shared methods (identical to original) ──────────────────────

    def _get_quality_pairs(self) -> list[str]:
        pairs = []
        for symbol, market in self.exchange.markets.items():
            if not market.get("swap") or not market.get("linear"):
                continue
            if market.get("quote") not in ("USDT",):
                continue
            if not market.get("active", True):
                continue
            base = market.get("base", "")
            if base not in QUALITY_BASES:
                continue
            pairs.append(symbol)
        return sorted(pairs)

    async def _fetch_all_candles(self, pairs: list[str]) -> dict:
        all_candles: dict = {}
        total = len(pairs)
        fetched = 0

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        since_1h = now_ms - (BACKTEST_DAYS * 24 + WARMUP_1H) * 3600 * 1000
        since_4h = now_ms - (BACKTEST_DAYS * 24 + 200) * 3600 * 1000
        since_1d = now_ms - 60 * 86400 * 1000

        for batch_start in range(0, total, BATCH_SIZE):
            batch = pairs[batch_start:batch_start + BATCH_SIZE]
            tasks = [self._fetch_pair_candles(sym, since_1h, since_4h, since_1d) for sym in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for sym, result in zip(batch, results):
                if isinstance(result, Exception):
                    continue
                if result:
                    all_candles[sym] = result

            fetched += len(batch)
            print(f"\r  Fetching candles: {fetched}/{total} pairs...", end="", flush=True)

            if batch_start + BATCH_SIZE < total:
                await asyncio.sleep(1.5)

        print(f"\r  Fetching candles: {fetched}/{total} pairs... done")
        return all_candles

    async def _fetch_pair_candles(self, symbol: str, since_1h: int, since_4h: int, since_1d: int) -> dict | None:
        try:
            candles = {}
            for tf, since, limit in [("1h", since_1h, 500), ("4h", since_4h, 500), ("1d", since_1d, 200)]:
                await self._rate_wait()
                raw = await self.exchange.fetch_ohlcv(symbol, tf, since=since, limit=limit)
                if not raw:
                    continue
                df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df.set_index("timestamp", inplace=True)
                candles[tf] = df

            if "1h" not in candles:
                return None
            return candles
        except Exception:
            return None

    async def _rate_wait(self) -> None:
        elapsed = time_mod.time() - self._last_request_time
        if elapsed < 0.3:
            await asyncio.sleep(0.3 - elapsed)
        self._last_request_time = time_mod.time()

    def _analyze_at_time(self, symbol: str, candles: dict, current_time: datetime):
        df_1h = candles.get("1h")
        df_4h = candles.get("4h")
        df_1d = candles.get("1d")

        if df_1h is None:
            return None

        df_1h = df_1h[df_1h.index <= current_time]
        if len(df_1h) < 50:
            return None

        if df_4h is not None:
            df_4h = df_4h[df_4h.index <= current_time]
            if len(df_4h) < 20:
                df_4h = None

        if df_1d is not None:
            df_1d = df_1d[df_1d.index <= current_time]
            if len(df_1d) < 10:
                df_1d = None

        try:
            ms_1h = self.ms_analyzer.analyze(df_1h, timeframe="1h")
        except Exception:
            return None

        ms_4h = self.ms_analyzer.analyze(df_4h, timeframe="4h") if df_4h is not None else None
        ms_1d = self.ms_analyzer.analyze(df_1d, timeframe="1d") if df_1d is not None else None

        session = self.session_analyzer.analyze(df_1h, now=current_time)
        vol = self.vol_analyzer.analyze(df_1h)

        sweep = self.sweep_detector.detect(
            candles_1h=df_1h,
            asian_high=session.asian_high,
            asian_low=session.asian_low,
            swing_high=ms_1h.key_levels.get("swing_high"),
            swing_low=ms_1h.key_levels.get("swing_low"),
            lookback=8,
            prefer_direction=vol.displacement_direction,
        )

        pullback = None
        if vol.displacement_detected and vol.displacement_candle_idx is not None:
            pullback = self.pullback_analyzer.analyze(
                candles_1h=df_1h,
                displacement_candle_idx=vol.displacement_candle_idx,
                direction=vol.displacement_direction,
            )

        ms_results = {"1h": ms_1h}
        if ms_4h:
            ms_results["4h"] = ms_4h
        if ms_1d:
            ms_results["1d"] = ms_1d

        htf_direction = self._resolve_htf(ms_results)
        current_price = float(df_1h["close"].iloc[-1])

        signal = self.confluence.score_signal(
            symbol=symbol,
            current_price=current_price,
            sweep_result=sweep,
            displacement_confirmed=vol.displacement_detected,
            displacement_direction=vol.displacement_direction,
            htf_direction=htf_direction,
            in_post_kill_zone=session.in_post_kill_zone,
            ms_results=ms_results,
            pullback_result=pullback,
        )

        if signal.score < ENTRY_THRESHOLD:
            return None

        signal.sweep_result = sweep
        signal.atr_1h = self._compute_atr(df_1h)
        signal.session_result = session

        return signal

    def _resolve_htf(self, ms_results: dict) -> str | None:
        htf_4h = ms_results.get("4h")
        htf_1d = ms_results.get("1d")
        trend_4h = htf_4h.trend if htf_4h else "ranging"
        trend_1d = htf_1d.trend if htf_1d else "ranging"
        if trend_4h == trend_1d and trend_4h != "ranging":
            return trend_4h
        if trend_4h != "ranging":
            return trend_4h
        if trend_1d != "ranging":
            return trend_1d
        return None

    def _compute_atr(self, df_1h: pd.DataFrame) -> float:
        if len(df_1h) < 15:
            return 0.0
        high = df_1h["high"].astype(float)
        low = df_1h["low"].astype(float)
        close = df_1h["close"].astype(float)
        prev = close.shift(1)
        tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        atr_series = tr.rolling(14).mean()
        val = atr_series.iloc[-1]
        return float(val) if pd.notna(val) else 0.0

    # ── FLIPPED entry logic ─────────────────────────────────────────

    def _try_enter_flipped(self, signal, current_time: datetime) -> None:
        """Enter a position with FLIPPED direction.

        When strategy says "bullish" → go SHORT.
        When strategy says "bearish" → go LONG.

        SL/TP use the sweep's opposite levels:
        - Flipped SHORT (original bullish): SL above target_level, TP below sweep_level
        - Flipped LONG (original bearish): SL below target_level, TP above sweep_level
        """
        if len(self.positions) >= MAX_CONCURRENT:
            return

        # FLIP: bullish signal → short, bearish signal → long
        is_long = signal.direction == "bearish"  # INVERTED
        entry = signal.entry_price
        sweep = signal.sweep_result
        atr = signal.atr_1h

        # SL/TP calculation with flipped sweep levels
        sl = None
        tp = None

        if sweep and sweep.sweep_detected and sweep.sweep_level > 0:
            if is_long:
                # Original was bearish (sweep high), flipped to LONG
                # SL below the sweep's target_level (the swing low the original aimed for)
                if sweep.target_level and sweep.target_level < entry:
                    sl = sweep.target_level * 0.995  # Buffer below
                # TP above the sweep_level (the high that was swept)
                if sweep.sweep_level > entry:
                    tp = sweep.sweep_level * 1.005  # Buffer above
            else:
                # Original was bullish (sweep low), flipped to SHORT
                # SL above the sweep's target_level (the swing high the original aimed for)
                if sweep.target_level and sweep.target_level > entry:
                    sl = sweep.target_level * 1.005  # Buffer above
                # TP below the sweep_level (the low that was swept)
                if sweep.sweep_level < entry:
                    tp = sweep.sweep_level * 0.995  # Buffer below

        # ATR fallback for SL
        if sl is None:
            if atr > 0:
                dist = atr * 2.0
            else:
                dist = entry * 0.03
            sl = entry - dist if is_long else entry + dist

        # Validate SL direction
        if is_long and sl >= entry:
            return
        if not is_long and sl <= entry:
            return

        sl_distance = abs(entry - sl)
        if sl_distance <= 0:
            return

        # Futures leverage: SL must trigger before liquidation
        liq_distance = entry / LEVERAGE
        if sl_distance > liq_distance * 0.8:
            return

        # TP fallback: ATR-based minimum R:R
        min_tp_dist = sl_distance * MIN_RR
        if tp is None:
            if is_long:
                tp = entry + min_tp_dist
            else:
                tp = entry - min_tp_dist
        else:
            # Ensure minimum R:R even with sweep-based TP
            if is_long:
                tp = max(tp, entry + min_tp_dist)
            else:
                tp = min(tp, entry - min_tp_dist)

        # R:R check
        tp_distance = abs(tp - entry)
        rr = tp_distance / sl_distance
        if rr < MIN_RR:
            return

        # Position sizing (identical to original)
        risk_amount = self.balance * MAX_RISK_PCT
        quantity = risk_amount / sl_distance
        cost = quantity * entry
        margin = cost / LEVERAGE
        max_margin = self.balance * MAX_POSITION_PCT

        if margin > max_margin:
            margin = max_margin
            cost = margin * LEVERAGE
            quantity = cost / entry

        if cost < 5.0:
            return

        total_margin = sum(p.cost_usd / LEVERAGE for p in self.positions) + margin
        if total_margin > self.balance:
            return

        entry_fee = cost * FEE_RATE
        self.balance -= entry_fee

        pos = BacktestPosition(
            symbol=signal.symbol,
            direction="long" if is_long else "short",
            entry_price=entry,
            quantity=quantity,
            sl=sl,
            tp=tp,
            entry_time=current_time,
            high_water_mark=entry,
            original_sl=sl,
            cost_usd=cost,
            atr_1h=atr,
            score=signal.score,
            reasons=[f"FLIPPED from {signal.direction}"] + signal.reasons[:2],
        )
        self.positions.append(pos)

    # ── Position monitoring (identical to original) ─────────────────

    def _monitor_positions(self, current_time: datetime, all_candles: dict) -> None:
        for pos in list(self.positions):
            df = all_candles.get(pos.symbol, {}).get("1h")
            if df is None or current_time not in df.index:
                continue

            candle = df.loc[current_time]
            h = float(candle["high"])
            l = float(candle["low"])

            if pos.direction == "long":
                if h > pos.high_water_mark:
                    pos.high_water_mark = h

                if l <= pos.sl:
                    self._close_position(pos, pos.sl, current_time, "sl_hit")
                    continue

                if h >= pos.tp:
                    self._close_position(pos, pos.tp, current_time, "tp_hit")
                    continue

                original_sl_dist = pos.entry_price - pos.original_sl
                if original_sl_dist > 0:
                    unrealized_rr = (pos.high_water_mark - pos.entry_price) / original_sl_dist
                    if unrealized_rr >= TRAILING_ACTIVATION_RR and pos.atr_1h > 0:
                        trailing_sl = pos.high_water_mark - (pos.atr_1h * TRAILING_ATR_MULT)
                        if trailing_sl > pos.sl:
                            pos.sl = trailing_sl
                        if l <= trailing_sl:
                            self._close_position(pos, trailing_sl, current_time, "trailing_stop")
                            continue

            else:  # short
                if l < pos.high_water_mark:
                    pos.high_water_mark = l

                if h >= pos.sl:
                    self._close_position(pos, pos.sl, current_time, "sl_hit")
                    continue

                if l <= pos.tp:
                    self._close_position(pos, pos.tp, current_time, "tp_hit")
                    continue

                original_sl_dist = pos.original_sl - pos.entry_price
                if original_sl_dist > 0:
                    unrealized_rr = (pos.entry_price - pos.high_water_mark) / original_sl_dist
                    if unrealized_rr >= TRAILING_ACTIVATION_RR and pos.atr_1h > 0:
                        trailing_sl = pos.high_water_mark + (pos.atr_1h * TRAILING_ATR_MULT)
                        if trailing_sl < pos.sl:
                            pos.sl = trailing_sl
                        if h >= trailing_sl:
                            self._close_position(pos, trailing_sl, current_time, "trailing_stop")
                            continue

    def _close_position(self, pos: BacktestPosition, exit_price: float, exit_time: datetime, reason: str) -> None:
        if pos.direction == "long":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        exit_cost = pos.quantity * exit_price
        exit_fee = exit_cost * FEE_RATE
        pnl -= exit_fee

        pos.exit_price = exit_price
        pos.exit_time = exit_time
        pos.exit_reason = reason
        pos.pnl = pnl

        self.balance += pnl
        self.positions.remove(pos)
        self.closed_trades.append(pos)

        if reason == "sl_hit":
            self.cooldowns[pos.symbol] = exit_time + timedelta(hours=COOLDOWN_HOURS)

    def _equity(self, current_time: datetime, all_candles: dict) -> float:
        equity = self.balance
        for pos in self.positions:
            df = all_candles.get(pos.symbol, {}).get("1h")
            if df is None:
                continue
            mask = df.index <= current_time
            if not mask.any():
                continue
            price = float(df[mask]["close"].iloc[-1])
            if pos.direction == "long":
                equity += (price - pos.entry_price) * pos.quantity
            else:
                equity += (pos.entry_price - price) * pos.quantity
        return equity

    # ── Report ──────────────────────────────────────────────────────

    def _print_report(self, signals_generated: int, pairs_scanned: int, bt_start: datetime, bt_end: datetime) -> None:
        print("=" * 90)
        print(f"  FLIPPED BACKTEST RESULTS: {bt_start.strftime('%b %d')} - {bt_end.strftime('%b %d, %Y')} ({BACKTEST_DAYS} days)")
        print(f"  (Every trade direction inverted from original strategy)")
        print("=" * 90)
        print()
        print(f"  Pairs scanned:        {pairs_scanned}")
        print(f"  Signals generated:    {signals_generated}")
        print(f"  Trades entered:       {len(self.closed_trades) + len(self.positions)}")
        print(f"  Trades closed:        {len(self.closed_trades)}")
        print(f"  Still open:           {len(self.positions)}")
        print()

        if not self.closed_trades and not self.positions:
            print("  No trades taken.")
            return

        all_trades = self.closed_trades + self.positions
        symbols = set(t.symbol for t in all_trades)
        print(f"  Unique coins traded:  {len(symbols)}")
        print()

        print(f"  {'Symbol':<22} {'Dir':<6} {'Entry':>10} {'SL':>10} {'TP':>10} {'Exit':>10} {'P&L':>10} {'Score':>6} {'Reason':<15}")
        print(f"  {'-'*20} {'-'*4}  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*15}")

        for t in sorted(all_trades, key=lambda x: x.entry_time):
            sym_short = t.symbol.replace("/USDT:USDT", "").replace("/USDT", "")
            exit_str = f"{t.exit_price:.6g}" if t.exit_price else "OPEN"
            pnl_str = f"${t.pnl:+.2f}" if t.exit_price else "---"
            reason_str = t.exit_reason or "open"
            print(f"  {sym_short:<22} {t.direction:<6} {t.entry_price:>10.6g} {t.sl:>10.6g} "
                  f"{t.tp:>10.6g} {exit_str:>10} {pnl_str:>10} {t.score:>6.0f} {reason_str:<15}")

        print()

        closed = self.closed_trades
        if closed:
            wins = [t for t in closed if t.pnl > 0]
            losses = [t for t in closed if t.pnl <= 0]
            total_pnl = sum(t.pnl for t in closed)

            print(f"  Win Rate:             {len(wins)}/{len(closed)} ({len(wins)/len(closed)*100:.1f}%)")
            print(f"  Total P&L:            ${total_pnl:+.2f}")
            print(f"  Avg Win:              ${sum(t.pnl for t in wins)/len(wins):+.2f}" if wins else "  Avg Win:              ---")
            print(f"  Avg Loss:             ${sum(t.pnl for t in losses)/len(losses):+.2f}" if losses else "  Avg Loss:             ---")
            print(f"  Max Drawdown:         {self.max_drawdown*100:.1f}%")

            if wins:
                best = max(closed, key=lambda t: t.pnl)
                sym_short = best.symbol.replace("/USDT:USDT", "").replace("/USDT", "")
                print(f"  Best Trade:           {sym_short} ${best.pnl:+.2f}")
            if losses:
                worst = min(closed, key=lambda t: t.pnl)
                sym_short = worst.symbol.replace("/USDT:USDT", "").replace("/USDT", "")
                print(f"  Worst Trade:          {sym_short} ${worst.pnl:+.2f}")

            print(f"  Final Balance:        ${self.balance:.2f}")

            print()
            print("  Exit Reasons:")
            reasons_count: dict[str, int] = {}
            for t in closed:
                reasons_count[t.exit_reason] = reasons_count.get(t.exit_reason, 0) + 1
            for reason, count in sorted(reasons_count.items()):
                pnl_for = sum(t.pnl for t in closed if t.exit_reason == reason)
                print(f"    {reason:<20} {count:>3} trades  ${pnl_for:+.2f}")

            print()
            print("  By Symbol:")
            sym_stats: dict[str, list] = {}
            for t in closed:
                sym_short = t.symbol.replace("/USDT:USDT", "").replace("/USDT", "")
                sym_stats.setdefault(sym_short, []).append(t)
            for sym, trades in sorted(sym_stats.items(), key=lambda x: sum(t.pnl for t in x[1]), reverse=True):
                total = sum(t.pnl for t in trades)
                w = len([t for t in trades if t.pnl > 0])
                print(f"    {sym:<18} {len(trades):>2} trades  {w}W/{len(trades)-w}L  ${total:+.2f}")

        print()
        print("=" * 90)

        self._save_results(signals_generated, pairs_scanned, bt_start, bt_end)

    def _save_results(self, signals_generated: int, pairs_scanned: int, bt_start: datetime, bt_end: datetime) -> None:
        os.makedirs("backtest_results", exist_ok=True)

        closed = self.closed_trades
        all_trades = closed + self.positions
        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in closed)

        by_reason: dict[str, dict] = {}
        for t in closed:
            r = t.exit_reason
            if r not in by_reason:
                by_reason[r] = {"count": 0, "pnl": 0.0}
            by_reason[r]["count"] += 1
            by_reason[r]["pnl"] = round(by_reason[r]["pnl"] + t.pnl, 2)

        by_symbol: dict[str, dict] = {}
        for t in closed:
            sym = t.symbol.replace("/USDT:USDT", "").replace("/USDT", "")
            if sym not in by_symbol:
                by_symbol[sym] = {"count": 0, "wins": 0, "pnl": 0.0}
            by_symbol[sym]["count"] += 1
            if t.pnl > 0:
                by_symbol[sym]["wins"] += 1
            by_symbol[sym]["pnl"] = round(by_symbol[sym]["pnl"] + t.pnl, 2)

        best_trade = max(closed, key=lambda t: t.pnl) if closed else None
        worst_trade = min(closed, key=lambda t: t.pnl) if closed else None

        result = {
            "type": "flipped",
            "run_time": datetime.now(timezone.utc).isoformat(),
            "period": {
                "start": bt_start.strftime("%Y-%m-%d"),
                "end": bt_end.strftime("%Y-%m-%d"),
                "days": BACKTEST_DAYS,
            },
            "config": {
                "balance": INITIAL_BALANCE,
                "leverage": LEVERAGE,
                "threshold": ENTRY_THRESHOLD,
                "fee_rate": FEE_RATE,
                "min_rr": MIN_RR,
            },
            "summary": {
                "pairs_scanned": pairs_scanned,
                "signals_generated": signals_generated,
                "trades_entered": len(all_trades),
                "trades_closed": len(closed),
                "still_open": len(self.positions),
                "unique_symbols": len(set(t.symbol for t in all_trades)),
                "win_rate": round(len(wins) / len(closed), 3) if closed else 0,
                "total_pnl": round(total_pnl, 2),
                "avg_win": round(sum(t.pnl for t in wins) / len(wins), 2) if wins else 0,
                "avg_loss": round(sum(t.pnl for t in losses) / len(losses), 2) if losses else 0,
                "max_drawdown": round(self.max_drawdown, 4),
                "final_balance": round(self.balance, 2),
                "best_trade": {
                    "symbol": best_trade.symbol.replace("/USDT:USDT", "").replace("/USDT", ""),
                    "pnl": round(best_trade.pnl, 2),
                } if best_trade else None,
                "worst_trade": {
                    "symbol": worst_trade.symbol.replace("/USDT:USDT", "").replace("/USDT", ""),
                    "pnl": round(worst_trade.pnl, 2),
                } if worst_trade else None,
            },
            "trades": [
                {
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "sl": t.sl,
                    "tp": t.tp,
                    "exit_price": t.exit_price,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    "exit_reason": t.exit_reason,
                    "pnl": round(t.pnl, 2),
                    "score": t.score,
                    "reasons": t.reasons,
                }
                for t in sorted(all_trades, key=lambda x: x.entry_time)
            ],
            "by_exit_reason": by_reason,
            "by_symbol": by_symbol,
        }

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results/flipped_{ts}.json"
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Results saved to {filename}")


async def main():
    engine = FlippedBacktestEngine()
    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
