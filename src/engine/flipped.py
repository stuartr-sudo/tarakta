"""Flipped shadow trader — independent scanner with inverted direction.

Runs alongside the main bot on its own 15-minute scan cycle. Uses a SIMPLER
strategy than the main bot: sweep + displacement only (no pullback gate,
no leverage intelligence, no quality whitelist). Every qualifying signal
is direction-flipped: bullish → short, bearish → long.

Wider SL, higher leverage. No exchange orders — purely simulated using
ticker prices. Trades stored in DB with mode='flipped_paper'.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import traceback
from datetime import datetime, timezone

import pandas as pd

from src.config import Settings
from src.data.candles import CandleManager
from src.data.repository import Repository
from src.exchange.models import Position, SignalCandidate
from src.strategy.market_structure import MarketStructureAnalyzer
from src.strategy.sessions import SessionAnalyzer
from src.strategy.sweep_detector import SweepDetector
from src.strategy.volume import VolumeAnalyzer
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Simulated fee rate (taker fee on Binance futures)
SIM_FEE_RATE = 0.0004

# Simplified scoring (no pullback, no leverage intelligence)
# Sweep (35) + Displacement (25) = 60 minimum
# HTF (15) + Timing (15) = bonus
FLIPPED_THRESHOLD = 60.0
FLIPPED_MAX_CONCURRENT = 0  # 0 = no cap; position sizing handles risk naturally
BATCH_SIZE = 8
BATCH_DELAY = 1.5
SCAN_TIMEFRAMES = ["1h", "4h", "1d"]


class FlippedTrader:
    """Shadow bot with its own scanner. Flips every signal and paper-trades it."""

    def __init__(
        self,
        config: Settings,
        repo: Repository,
        candle_manager: CandleManager | None = None,
        exchange=None,
    ) -> None:
        self.enabled = config.flipped_enabled
        self.leverage = config.flipped_leverage
        self.sl_buffer = config.flipped_sl_buffer
        self.min_sl_pct = config.flipped_min_sl_pct
        self.min_rr = config.min_rr_ratio
        self.max_risk_pct = config.flipped_max_risk_pct
        self.max_position_pct = config.flipped_max_position_pct
        self.trailing_activation_rr = config.trailing_activation_rr
        self.trailing_atr_multiplier = config.trailing_atr_multiplier
        self.scan_interval = config.flipped_scan_interval_minutes
        self.min_volume_usd = config.min_volume_usd
        self.quote_currencies = config.quote_currencies
        self.repo = repo
        self.candle_manager = candle_manager
        self.exchange = exchange

        # Simplified strategy components (no pullback, no leverage analyzer)
        self.ms_analyzer = MarketStructureAnalyzer()
        self.vol_analyzer = VolumeAnalyzer()
        self.session_analyzer = SessionAnalyzer()
        self.sweep_detector = SweepDetector()

        # Separate paper balance — independent of main bot
        self.balance = config.flipped_initial_balance
        self.peak_balance = config.flipped_initial_balance
        self.daily_start_balance = config.flipped_initial_balance
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.positions: dict[str, Position] = {}
        self.daily_trade_count: int = 0
        self._scan_count: int = 0
        self.last_scan_time: str | None = None

    # ------------------------------------------------------------------
    # Independent scan loop
    # ------------------------------------------------------------------

    async def run_loop(self) -> None:
        """Independent scan loop — runs every N minutes, completely separate from main bot."""
        if not self.enabled or not self.candle_manager or not self.exchange:
            logger.warning("flipped_loop_disabled", reason="missing dependencies or disabled")
            return

        logger.info(
            "flipped_loop_started",
            interval_minutes=self.scan_interval,
            leverage=self.leverage,
            sl_buffer=self.sl_buffer,
            threshold=FLIPPED_THRESHOLD,
        )

        # Reconcile in-memory positions with DB (handles restarts)
        await self._reconcile_from_db()

        # Start independent position monitor (every 60s)
        monitor_task = asyncio.create_task(self._monitor_loop())

        # Run an immediate scan on startup
        try:
            await self._run_scan()
        except Exception as e:
            logger.error("flipped_startup_scan_failed", error=str(e))

        while True:
            try:
                await asyncio.sleep(self.scan_interval * 60)
                await self._run_scan()
            except asyncio.CancelledError:
                logger.info("flipped_loop_cancelled")
                monitor_task.cancel()
                break
            except Exception as e:
                logger.error(
                    "flipped_scan_error",
                    error=str(e),
                    stack=traceback.format_exc()[:500],
                )
                await asyncio.sleep(60)  # Back off on error

    async def _monitor_loop(self) -> None:
        """Independent position monitor — checks SL/TP every 60 seconds.

        This runs alongside the main engine's _monitor_tick as a safety net.
        Positions are checked here regardless of the main engine's state.
        """
        logger.info("flipped_monitor_loop_started")
        while True:
            try:
                await asyncio.sleep(60)
                if self.positions:
                    await self.monitor_positions()
                    # Persist state after monitoring (in case positions were closed)
                    await self._save_state()
            except asyncio.CancelledError:
                logger.info("flipped_monitor_loop_cancelled")
                break
            except Exception as e:
                logger.error("flipped_monitor_loop_error", error=str(e))

    async def _reconcile_from_db(self) -> None:
        """Ensure in-memory positions match DB open trades.

        On restart, config_overrides might be stale. This queries the DB for
        open flipped_paper trades and adds any missing positions to memory.
        """
        try:
            open_trades = await self.repo.get_open_trades(mode="flipped_paper")
            if not open_trades:
                logger.info("flipped_reconcile_no_open_trades")
                return

            added = 0
            for trade in open_trades:
                symbol = trade["symbol"]
                if symbol in self.positions:
                    continue  # Already tracked

                # Reconstruct Position from DB record
                try:
                    entry_time = datetime.now(timezone.utc)
                    if trade.get("entry_time"):
                        entry_time = datetime.fromisoformat(str(trade["entry_time"]))

                    pos = Position(
                        trade_id=trade.get("id", ""),
                        symbol=symbol,
                        entry_price=float(trade.get("entry_price", 0)),
                        quantity=float(trade.get("entry_quantity", 0)),
                        stop_loss=float(trade.get("stop_loss", 0)),
                        take_profit=float(trade.get("take_profit")) if trade.get("take_profit") else None,
                        high_water_mark=float(trade.get("entry_price", 0)),
                        entry_time=entry_time,
                        cost_usd=float(trade.get("entry_cost_usd", 0)),
                        direction=trade.get("direction", "long"),
                        leverage=int(trade.get("leverage", 1) or 1),
                        margin_used=float(trade.get("margin_used", 0) or 0),
                        liquidation_price=float(trade.get("liquidation_price", 0) or 0),
                        original_quantity=float(trade.get("original_quantity") or trade.get("entry_quantity", 0)),
                        original_stop_loss=float(trade.get("original_stop_loss") or trade.get("stop_loss", 0)),
                        confluence_score=float(trade.get("confluence_score", 0) or 0),
                    )
                    self.positions[symbol] = pos
                    added += 1
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning("flipped_reconcile_failed", symbol=symbol, error=str(e))

            if added > 0:
                logger.info(
                    "flipped_reconcile_added",
                    added=added,
                    total_open=len(self.positions),
                    db_open=len(open_trades),
                )
        except Exception as e:
            logger.error("flipped_reconcile_error", error=str(e))

    async def _run_scan(self) -> None:
        """Run a full scan with simplified strategy, then enter flipped trades."""
        self._scan_count += 1
        logger.info("flipped_scan_start", scan=self._scan_count)

        # Get ALL tradeable pairs (no quality whitelist filter)
        try:
            pairs = await self.exchange.get_tradeable_pairs(
                min_volume_usd=self.min_volume_usd,
                quote_currencies=self.quote_currencies,
                quality_filter=False,  # No whitelist — scan everything
            )
        except Exception as e:
            logger.error("flipped_pair_scan_failed", error=str(e))
            return

        if not pairs:
            logger.warning("flipped_no_pairs")
            return

        # Scan with simplified pipeline
        signals = await self._scan_pairs(pairs)

        # Enter flipped trades (limited by available balance)
        entered = 0
        for signal in signals:
            if FLIPPED_MAX_CONCURRENT > 0 and len(self.positions) >= FLIPPED_MAX_CONCURRENT:
                break
            if self.balance < 5.0:
                break  # No margin left
            if signal.symbol in self.positions:
                continue
            try:
                if await self._try_enter(signal):
                    entered += 1
            except Exception as e:
                logger.warning("flipped_entry_error", symbol=signal.symbol, error=str(e))

        self.last_scan_time = datetime.now(timezone.utc).isoformat()

        logger.info(
            "flipped_scan_complete",
            scan=self._scan_count,
            pairs_scanned=len(pairs),
            signals_found=len(signals),
            trades_entered=entered,
            open_positions=len(self.positions),
            balance=round(self.balance, 2),
        )

    async def _scan_pairs(self, pairs: list[str]) -> list[SignalCandidate]:
        """Simplified scan pipeline: sweep + displacement only. No pullback, no leverage."""
        all_signals: list[SignalCandidate] = []
        total = len(pairs)

        # Suppress noisy sweep_detected logs during flipped scans (floods log buffer)
        sweep_logger = logging.getLogger("src.strategy.sweep_detector")
        prev_level = sweep_logger.level
        sweep_logger.setLevel(logging.WARNING)

        for batch_idx in range(0, total, BATCH_SIZE):
            batch = pairs[batch_idx : batch_idx + BATCH_SIZE]

            tasks = [self._analyze_pair(pair) for pair in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    continue
                if isinstance(result, SignalCandidate) and result.score >= FLIPPED_THRESHOLD:
                    all_signals.append(result)

            gc.collect()
            if batch_idx + BATCH_SIZE < total:
                await asyncio.sleep(BATCH_DELAY)

        # Restore sweep detector logging
        sweep_logger.setLevel(prev_level)

        # Sort by score descending
        all_signals.sort(key=lambda s: s.score, reverse=True)

        if all_signals:
            logger.info(
                "flipped_signals_found",
                count=len(all_signals),
                top=all_signals[0].symbol,
                top_score=all_signals[0].score,
            )

        return all_signals

    async def _analyze_pair(self, symbol: str) -> SignalCandidate:
        """Simplified analysis: sweep + displacement + HTF + timing. No pullback."""
        # 1. Fetch candles
        candles: dict[str, pd.DataFrame] = {}
        for tf in SCAN_TIMEFRAMES:
            candles[tf] = await self.candle_manager.get_candles(symbol, tf, limit=200)

        # 2. Market structure on all TFs
        ms_results = {}
        for tf, df in candles.items():
            ms_results[tf] = self.ms_analyzer.analyze(df, timeframe=tf)

        # 3. Session analysis
        session_result = self.session_analyzer.analyze(candles["1h"])

        # 4. Swing levels from 1H
        swing_high = ms_results["1h"].key_levels.get("swing_high")
        swing_low = ms_results["1h"].key_levels.get("swing_low")

        # 5. Displacement check on 1H
        vol_profile = self.vol_analyzer.analyze(candles["1h"])
        displacement_confirmed = vol_profile.displacement_detected
        displacement_direction = vol_profile.displacement_direction

        # 6. Sweep detection on 1H
        sweep_result = self.sweep_detector.detect(
            candles_1h=candles["1h"],
            asian_high=session_result.asian_high,
            asian_low=session_result.asian_low,
            swing_high=swing_high,
            swing_low=swing_low,
            lookback=8,
            prefer_direction=displacement_direction,
        )

        # 7. Current price + ATR
        current_price = float(candles["1h"]["close"].iloc[-1]) if not candles["1h"].empty else 0
        atr_1h = self._compute_atr(candles.get("1h"))

        # 8. HTF direction
        htf_direction = self._resolve_htf_direction(ms_results)

        # 9. SIMPLIFIED SCORING (no pullback, no leverage)
        score = 0.0
        direction = None
        reasons: list[str] = []
        components: dict[str, float] = {}

        # Sweep (35 pts) — REQUIRED
        if sweep_result.sweep_detected:
            score += 35
            direction = sweep_result.sweep_direction
            reasons.append(f"Sweep: {sweep_result.sweep_type} (depth={sweep_result.sweep_depth:.4f})")
            components["sweep_detected"] = 35
        else:
            return SignalCandidate(
                score=0, direction=None, reasons=["No sweep"],
                symbol=symbol, entry_price=current_price, components={},
            )

        # Displacement (25 pts) — REQUIRED
        if displacement_confirmed and displacement_direction == direction:
            score += 25
            reasons.append(f"Displacement confirmed: {direction}")
            components["displacement_confirmed"] = 25
        else:
            return SignalCandidate(
                score=score, direction=direction, reasons=reasons,
                symbol=symbol, entry_price=current_price, components=components,
            )

        # HTF alignment (15 pts) — bonus
        if htf_direction == direction:
            score += 15
            reasons.append(f"HTF aligned: {direction}")
            components["htf_aligned"] = 15
        elif htf_direction:
            # Partial credit for 4H only
            htf_4h = ms_results.get("4h")
            if htf_4h and htf_4h.trend == direction:
                score += 10
                reasons.append(f"4H aligned: {direction}")
                components["htf_aligned"] = 10

        # Timing (15 pts) — bonus
        if session_result.in_post_kill_zone:
            score += 15
            reasons.append("Post-kill-zone timing")
            components["timing_optimal"] = 15

        # Collect key levels
        key_levels = {}
        for tf, ms in ms_results.items():
            if ms.key_levels.get("swing_high"):
                key_levels[f"{tf}_swing_high"] = ms.key_levels["swing_high"]
            if ms.key_levels.get("swing_low"):
                key_levels[f"{tf}_swing_low"] = ms.key_levels["swing_low"]

        signal = SignalCandidate(
            score=score,
            direction=direction,
            reasons=reasons,
            symbol=symbol,
            entry_price=current_price,
            key_levels=key_levels,
            components=components,
        )
        signal.sweep_result = sweep_result
        signal.atr_1h = atr_1h
        signal.session_result = session_result
        return signal

    @staticmethod
    def _compute_atr(df: pd.DataFrame | None) -> float:
        if df is None or len(df) < 15:
            return 0.0
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev = close.shift(1)
        tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        return float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0

    @staticmethod
    def _resolve_htf_direction(ms_results: dict) -> str | None:
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

    # ------------------------------------------------------------------
    # Trade entry (flipped direction)
    # ------------------------------------------------------------------

    async def _try_enter(self, signal: SignalCandidate) -> bool:
        """Flip direction and simulate entry."""
        # FLIP: bullish → short, bearish → long
        is_long = signal.direction == "bearish"
        direction = "long" if is_long else "short"

        # Calculate flipped SL
        sl_price = self._calculate_sl(signal, is_long)
        if sl_price is None:
            return False

        # Validate SL direction
        if is_long and sl_price >= signal.entry_price:
            return False
        if not is_long and sl_price <= signal.entry_price:
            return False

        # Calculate flipped TP
        tp_price = self._calculate_tp(signal, sl_price, is_long)

        # Validate R:R
        sl_distance = abs(signal.entry_price - sl_price)
        if sl_distance <= 0:
            return False
        tp_distance = abs(tp_price - signal.entry_price) if tp_price else sl_distance * self.min_rr
        rr_ratio = tp_distance / sl_distance
        if rr_ratio < self.min_rr:
            return False

        # Ensure SL within leverage safety (don't get liquidated before SL)
        liq_distance = signal.entry_price / self.leverage
        if sl_distance > liq_distance * 0.8:
            return False

        # Position sizing
        risk_amount = self.balance * self.max_risk_pct
        quantity = risk_amount / sl_distance
        cost_usd = quantity * signal.entry_price
        margin_used = cost_usd / self.leverage

        # Cap by max_position_pct
        max_margin = self.balance * self.max_position_pct
        if margin_used > max_margin:
            margin_used = max_margin
            cost_usd = margin_used * self.leverage
            quantity = cost_usd / signal.entry_price

        # Can't exceed available balance
        if margin_used > self.balance:
            margin_used = self.balance
            cost_usd = margin_used * self.leverage
            quantity = cost_usd / signal.entry_price

        if cost_usd < 5.0:
            return False  # Below exchange minimum

        # Liquidation price
        if is_long:
            liq_price = signal.entry_price * (1 - 1 / self.leverage * 0.95)
        else:
            liq_price = signal.entry_price * (1 + 1 / self.leverage * 0.95)

        # Build position
        position = Position(
            trade_id="",
            symbol=signal.symbol,
            direction=direction,
            entry_price=signal.entry_price,
            quantity=quantity,
            stop_loss=sl_price,
            take_profit=tp_price,
            high_water_mark=signal.entry_price,
            entry_time=datetime.now(timezone.utc),
            cost_usd=cost_usd,
            leverage=self.leverage,
            margin_used=margin_used,
            liquidation_price=liq_price,
            original_quantity=quantity,
            original_stop_loss=sl_price,
            confluence_score=signal.score,
        )

        # Entry fee
        entry_fee = cost_usd * SIM_FEE_RATE

        # Save to DB with mode="flipped_paper"
        trade_record = {
            "symbol": signal.symbol,
            "direction": direction,
            "status": "open",
            "mode": "flipped_paper",
            "entry_price": signal.entry_price,
            "entry_quantity": quantity,
            "entry_cost_usd": cost_usd,
            "entry_order_id": "flipped_sim",
            "entry_time": position.entry_time.isoformat(),
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "risk_usd": quantity * sl_distance,
            "risk_reward": round(rr_ratio, 2),
            "confluence_score": signal.score,
            "signal_reasons": [f"FLIPPED:{signal.direction}\u2192{direction}"] + signal.reasons,
            "timeframes_used": {"htf": "4h", "entry": "1h"},
            "fees_usd": entry_fee,
            "leverage": self.leverage,
            "margin_used": margin_used,
            "liquidation_price": liq_price,
            "original_quantity": quantity,
            "remaining_quantity": quantity,
        }

        db_trade = await self.repo.insert_trade(trade_record)
        if not db_trade:
            logger.warning("flipped_db_insert_failed", symbol=signal.symbol)
            return False

        position.trade_id = db_trade.get("id", "")
        self.positions[signal.symbol] = position
        self.balance -= margin_used
        self.daily_trade_count += 1

        logger.info(
            "flipped_entry",
            symbol=signal.symbol,
            original_direction=signal.direction,
            flipped_direction=direction,
            entry=signal.entry_price,
            sl=sl_price,
            tp=tp_price,
            rr=f"{rr_ratio:.1f}",
            leverage=self.leverage,
            margin=round(margin_used, 2),
            balance=round(self.balance, 2),
        )
        return True

    # ------------------------------------------------------------------
    # Position monitoring
    # ------------------------------------------------------------------

    async def monitor_positions(self, exchange=None, atr_values: dict[str, float] | None = None) -> None:
        """Check flipped positions for SL/TP/trailing stop using live ticker data."""
        exchange = exchange or self.exchange
        if not self.enabled or not self.positions or not exchange:
            return

        atr_values = atr_values or {}
        checked = 0
        closed = 0
        ticker_errors = 0

        # Compute ATR for open positions if not provided
        if self.candle_manager:
            for symbol in list(self.positions.keys()):
                if symbol not in atr_values:
                    try:
                        candles = await self.candle_manager.get_candles(symbol, "1h", limit=30)
                        atr = self._compute_atr(candles)
                        if atr > 0:
                            atr_values[symbol] = atr
                    except Exception:
                        pass

        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            try:
                ticker = await exchange.fetch_ticker(symbol)
                current_price = float(ticker["last"])
                checked += 1
            except Exception as e:
                ticker_errors += 1
                logger.warning("flipped_ticker_failed", symbol=symbol, error=str(e)[:100])
                continue

            # Update high water mark
            if pos.direction == "long":
                if current_price > pos.high_water_mark:
                    pos.high_water_mark = current_price
            else:
                if current_price < pos.high_water_mark:
                    pos.high_water_mark = current_price

            exit_reason = None

            # SL check
            if pos.direction == "long" and current_price <= pos.stop_loss:
                exit_reason = "sl_hit"
            elif pos.direction == "short" and current_price >= pos.stop_loss:
                exit_reason = "sl_hit"

            # TP check
            if not exit_reason and pos.take_profit:
                if pos.direction == "long" and current_price >= pos.take_profit:
                    exit_reason = "tp_hit"
                elif pos.direction == "short" and current_price <= pos.take_profit:
                    exit_reason = "tp_hit"

            # Trailing stop
            if not exit_reason:
                sl_dist = abs(pos.entry_price - pos.original_stop_loss)
                if sl_dist > 0:
                    if pos.direction == "long":
                        r_multiple = (pos.high_water_mark - pos.entry_price) / sl_dist
                    else:
                        r_multiple = (pos.entry_price - pos.high_water_mark) / sl_dist

                    if r_multiple >= self.trailing_activation_rr:
                        atr = atr_values.get(symbol, 0)
                        trail_dist = atr * self.trailing_atr_multiplier if atr > 0 else sl_dist * 0.75

                        if pos.direction == "long":
                            trail_sl = pos.high_water_mark - trail_dist
                            if trail_sl > pos.stop_loss:
                                pos.stop_loss = trail_sl
                            if current_price <= pos.stop_loss:
                                exit_reason = "trailing_stop"
                        else:
                            trail_sl = pos.high_water_mark + trail_dist
                            if trail_sl < pos.stop_loss:
                                pos.stop_loss = trail_sl
                            if current_price >= pos.stop_loss:
                                exit_reason = "trailing_stop"

            if exit_reason:
                logger.info(
                    "flipped_exit_triggered",
                    symbol=symbol,
                    reason=exit_reason,
                    direction=pos.direction,
                    price=current_price,
                    sl=pos.stop_loss,
                    tp=pos.take_profit,
                )
                await self._close_position(symbol, current_price, exit_reason)
                closed += 1

        if checked > 0 or ticker_errors > 0:
            logger.info(
                "flipped_monitor_tick",
                checked=checked,
                closed=closed,
                errors=ticker_errors,
                open=len(self.positions),
            )

    async def _close_position(self, symbol: str, exit_price: float, reason: str) -> None:
        """Simulate closing a flipped position."""
        pos = self.positions.pop(symbol, None)
        if not pos:
            return

        # Calculate PnL
        if pos.direction == "long":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        # Simulate exit fee
        exit_fee = pos.cost_usd * SIM_FEE_RATE
        pnl -= exit_fee

        # Return margin + pnl to balance
        margin = pos.margin_used
        self.balance += margin + pnl
        self.daily_pnl += pnl
        self.total_pnl += pnl

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        # Close in DB
        pnl_pct = (pnl / pos.cost_usd * 100) if pos.cost_usd > 0 else 0
        total_fees = pos.cost_usd * SIM_FEE_RATE + exit_fee
        await self.repo.close_trade(
            trade_id=pos.trade_id,
            exit_price=exit_price,
            exit_quantity=pos.quantity,
            exit_order_id="flipped_sim",
            exit_reason=reason,
            pnl_usd=round(pnl, 4),
            pnl_percent=round(pnl_pct, 2),
            fees_usd=round(total_fees, 4),
        )

        logger.info(
            "flipped_exit",
            symbol=symbol,
            direction=pos.direction,
            reason=reason,
            entry=pos.entry_price,
            exit=exit_price,
            pnl=round(pnl, 2),
            balance=round(self.balance, 2),
        )

    # ------------------------------------------------------------------
    # State persistence (independent of main engine)
    # ------------------------------------------------------------------

    async def _save_state(self) -> None:
        """Persist flipped state to DB (config_overrides JSONB).

        Called after the independent monitor loop closes positions, so state
        is saved even if the main engine's _persist_state hasn't run yet.
        """
        try:
            state = await self.repo.get_engine_state()
            if not state:
                return
            overrides = state.get("config_overrides") or {}
            if not isinstance(overrides, dict):
                overrides = {}
            overrides["flipped_trader"] = self.to_state_dict()
            state["config_overrides"] = overrides
            await self.repo.upsert_engine_state(state)
        except Exception as e:
            logger.warning("flipped_save_state_failed", error=str(e))

    # ------------------------------------------------------------------
    # SL / TP calculation
    # ------------------------------------------------------------------

    def _calculate_sl(self, signal: SignalCandidate, is_long: bool) -> float | None:
        """Calculate SL for flipped trade with wider buffer + minimum distance floor."""
        entry = signal.entry_price
        min_distance = entry * self.min_sl_pct  # Minimum SL distance (1.5% of entry)
        sweep = getattr(signal, "sweep_result", None)

        sl = None
        if sweep is not None and sweep.sweep_detected and sweep.target_level > 0:
            if is_long:
                sl = sweep.target_level * (1 - self.sl_buffer)
                if sl >= entry:
                    sl = None  # Invalid — SL must be below entry for long
            else:
                sl = sweep.target_level * (1 + self.sl_buffer)
                if sl <= entry:
                    sl = None  # Invalid — SL must be above entry for short

        # Enforce minimum SL distance
        if sl is not None:
            actual_distance = abs(entry - sl)
            if actual_distance < min_distance:
                sl = entry - min_distance if is_long else entry + min_distance

        if sl is not None:
            return sl

        # ATR fallback with wider buffer
        atr = getattr(signal, "atr_1h", 0.0) or 0.0
        if atr > 0:
            distance = max(atr * 2.5, min_distance)
        else:
            distance = max(entry * 0.04, min_distance)

        return entry - distance if is_long else entry + distance

    def _calculate_tp(self, signal: SignalCandidate, sl_price: float, is_long: bool) -> float | None:
        """Calculate TP for flipped trade."""
        entry = signal.entry_price
        sl_distance = abs(entry - sl_price)
        min_tp_distance = sl_distance * self.min_rr
        sweep = getattr(signal, "sweep_result", None)

        if is_long:
            min_tp = entry + min_tp_distance
            if sweep and sweep.sweep_level > entry:
                tp = sweep.sweep_level * 1.005
                if tp >= min_tp:
                    return tp
            for tf in ["1h_swing_high", "4h_swing_high", "1d_swing_high"]:
                level = signal.key_levels.get(tf)
                if level and level >= min_tp:
                    return level
            return min_tp
        else:
            min_tp = entry - min_tp_distance
            if sweep and sweep.sweep_level < entry:
                tp = sweep.sweep_level * 0.995
                if tp <= min_tp:
                    return tp
            for tf in ["1h_swing_low", "4h_swing_low", "1d_swing_low"]:
                level = signal.key_levels.get(tf)
                if level and level <= min_tp:
                    return level
            return min_tp

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        self.daily_start_balance = self.balance
        self.daily_pnl = 0.0
        self.daily_trade_count = 0

    def to_state_dict(self) -> dict:
        positions_data = {}
        for sym, pos in self.positions.items():
            positions_data[sym] = {
                "trade_id": pos.trade_id,
                "symbol": pos.symbol,
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "high_water_mark": pos.high_water_mark,
                "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
                "cost_usd": pos.cost_usd,
                "direction": pos.direction,
                "leverage": pos.leverage,
                "margin_used": pos.margin_used,
                "liquidation_price": pos.liquidation_price,
                "original_quantity": pos.original_quantity,
                "original_stop_loss": pos.original_stop_loss,
                "confluence_score": pos.confluence_score,
            }
        return {
            "balance": self.balance,
            "peak_balance": self.peak_balance,
            "daily_start_balance": self.daily_start_balance,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "daily_trade_count": self.daily_trade_count,
            "last_scan_time": self.last_scan_time,
            "positions": positions_data,
        }

    def restore_state(self, data: dict) -> None:
        if not data:
            return
        self.balance = float(data.get("balance", self.balance))
        self.peak_balance = float(data.get("peak_balance", self.peak_balance))
        self.daily_start_balance = float(data.get("daily_start_balance", self.daily_start_balance))
        self.daily_pnl = float(data.get("daily_pnl", 0))
        self.total_pnl = float(data.get("total_pnl", 0))
        self.daily_trade_count = int(data.get("daily_trade_count", 0))
        self.last_scan_time = data.get("last_scan_time")

        positions_data = data.get("positions", {})
        for sym, pd_ in positions_data.items():
            try:
                entry_time = datetime.now(timezone.utc)
                if pd_.get("entry_time"):
                    entry_time = datetime.fromisoformat(str(pd_["entry_time"]))
                self.positions[sym] = Position(
                    trade_id=pd_.get("trade_id", ""),
                    symbol=sym,
                    entry_price=float(pd_.get("entry_price", 0)),
                    quantity=float(pd_.get("quantity", 0)),
                    stop_loss=float(pd_.get("stop_loss", 0)),
                    take_profit=float(pd_.get("take_profit")) if pd_.get("take_profit") else None,
                    high_water_mark=float(pd_.get("high_water_mark", pd_.get("entry_price", 0))),
                    entry_time=entry_time,
                    cost_usd=float(pd_.get("cost_usd", 0)),
                    direction=pd_.get("direction", "long"),
                    leverage=int(pd_.get("leverage", 1) or 1),
                    margin_used=float(pd_.get("margin_used", 0) or 0),
                    liquidation_price=float(pd_.get("liquidation_price", 0) or 0),
                    original_quantity=float(pd_.get("original_quantity", 0) or 0),
                    original_stop_loss=float(pd_.get("original_stop_loss", 0) or 0),
                    confluence_score=float(pd_.get("confluence_score", 0) or 0),
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.warning("flipped_position_restore_failed", symbol=sym, error=str(e))

        logger.info(
            "flipped_state_restored",
            balance=self.balance,
            open_positions=len(self.positions),
            total_pnl=self.total_pnl,
        )
