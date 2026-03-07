"""Flipped shadow trader — independent scanner with inverted direction.

Runs alongside the main bot on its own scan cycle. Uses the same core
pipeline: sweep + displacement + pullback + HTF + timing, with London/NY
ranges for more sweep opportunities. Direction can be flipped or run
normally depending on mode.

Wider SL, higher leverage. No exchange orders — purely simulated using
ticker prices. Trades stored in DB with configurable mode (default: 'flipped_paper').
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
from src.exchange.models import Position, SignalCandidate, TakeProfitTier
from src.strategy.leverage import LeverageAnalyzer
from src.strategy.market_structure import MarketStructureAnalyzer
from src.strategy.pullback import PullbackAnalyzer
from src.strategy.sessions import SessionAnalyzer
from src.strategy.sweep_detector import SweepDetector
from src.strategy.volume import VolumeAnalyzer
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Simulated fee rate (taker fee on Binance futures)
SIM_FEE_RATE = 0.0004

# Scoring: sweep (35) + displacement (25) + pullback (10) + HTF (15) + timing (15)
# Classic path: sweep + displacement = 60.  Alternative: sweep + HTF + timing = 65.
FLIPPED_THRESHOLD = 60.0
FLIPPED_MAX_CONCURRENT = 8  # Cap concurrent positions to limit total exposure
MAX_EXPOSURE_PCT = 0.60  # Don't deploy more than 60% of initial balance as margin
BATCH_SIZE = 16  # Larger batches for faster scans (475 pairs is a lot)
BATCH_DELAY = 0.5  # Shorter delay between batches
SCAN_TIMEFRAMES = ["1h", "4h"]  # Drop 1d — not worth extra API calls for 5pts HTF bonus


class FlippedTrader:
    """Configurable shadow bot with its own scanner.

    Supports both flipped (inverted direction) and normal trading modes.
    Parameters ``flip_direction``, ``mode_name``, and ``state_key`` allow
    multiple independent instances to run side-by-side (e.g. the default
    flipped bot AND a user-configurable "custom" bot).
    """

    def __init__(
        self,
        config: Settings,
        repo: Repository,
        candle_manager: CandleManager | None = None,
        exchange=None,
        *,
        flip_direction: bool = True,
        mode_name: str = "flipped_paper",
        state_key: str = "flipped_trader",
    ) -> None:
        self.config = config  # Store reference for liquidity checks etc.
        self.enabled = config.flipped_enabled
        self.leverage = config.flipped_leverage
        self.sl_buffer = config.flipped_sl_buffer
        self.min_sl_pct = config.flipped_min_sl_pct
        self.min_rr = config.flipped_min_rr_ratio
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

        # Configurable direction & identity
        self.flip_direction = flip_direction
        self.mode_name = mode_name
        self.state_key = state_key
        self.flip_mode = "always_flip"   # Default for flipped bot; custom overrides to "smart_flip"
        self.flip_threshold = 0.50

        # Strategy components
        self.ms_analyzer = MarketStructureAnalyzer()
        self.vol_analyzer = VolumeAnalyzer()
        self.session_analyzer = SessionAnalyzer()
        self.sweep_detector = SweepDetector()
        self.pullback_analyzer = PullbackAnalyzer(
            min_retracement=config.pullback_min_retracement,
            max_retracement=config.pullback_max_retracement,
        )
        self.leverage_analyzer = LeverageAnalyzer()

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
        # Cooldown: don't re-enter a symbol within 4 hours of closing
        self._closed_symbols: dict[str, datetime] = {}  # symbol → close time
        self.REENTRY_COOLDOWN_HOURS = 4

        # Thread-safe flags — set by dashboard API, checked by monitor/scan loops
        import threading
        self._reset_event = threading.Event()
        self._trigger_scan_event = threading.Event()
        self._scanning_active = True  # Default True; custom bot will override to False
        self._initial_balance = config.flipped_initial_balance

        # Throttled logging: suppress repeated ticker errors per symbol
        self._ticker_error_last_logged: dict[str, datetime] = {}
        self._ticker_error_throttle_seconds = 300  # Log same error once per 5 min
        self._monitor_tick_count = 0  # For periodic heartbeat log

    # ------------------------------------------------------------------
    # Runtime settings update (hot-reload from dashboard)
    # ------------------------------------------------------------------

    def update_settings(
        self,
        flip_direction: bool | None = None,
        margin_pct: float | None = None,
        flip_mode: str | None = None,
        flip_threshold: float | None = None,
        leverage: int | None = None,
    ) -> None:
        """Update configurable parameters at runtime (thread-safe for simple scalars)."""
        if flip_direction is not None:
            self.flip_direction = flip_direction
        if margin_pct is not None:
            self.max_position_pct = margin_pct
            self.max_risk_pct = margin_pct
        if flip_mode is not None and flip_mode in ("always_flip", "smart_flip", "normal"):
            self.flip_mode = flip_mode
        if flip_threshold is not None:
            self.flip_threshold = max(0.0, min(1.0, flip_threshold))
        if leverage is not None:
            self.leverage = leverage
            # Also update PaperExchange leverage if paper trading
            if hasattr(self.exchange, "_leverage"):
                self.exchange._leverage = leverage

    def request_reset(self) -> None:
        """Signal the flipped trader to reset on next tick (thread-safe)."""
        self._reset_event.set()

    def request_scan(self) -> None:
        """Signal the flipped trader to run an immediate scan (thread-safe)."""
        self._trigger_scan_event.set()

    def begin_scanning(self) -> None:
        """Enable the scan loop (called by dashboard Begin button)."""
        self._scanning_active = True
        self._trigger_scan_event.set()  # Also trigger an immediate scan
        logger.info("bot_scanning_started", mode=self.mode_name)

    def stop_scanning(self) -> None:
        """Pause the scan loop (keeps monitoring open positions)."""
        self._scanning_active = False
        logger.info("bot_scanning_stopped", mode=self.mode_name)

    async def _check_and_apply_reset(self) -> bool:
        """Check if a reset was requested and apply it. Returns True if reset happened."""
        if not self._reset_event.is_set():
            return False

        self._reset_event.clear()
        logger.info("flipped_reset_applying", reason="dashboard_reset_button")

        # Clear all in-memory state
        self.positions.clear()
        self.balance = self._initial_balance
        self.peak_balance = self._initial_balance
        self.daily_start_balance = self._initial_balance
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.daily_trade_count = 0
        self._scan_count = 0
        self.last_scan_time = None
        self._closed_symbols.clear()

        # Save the clean state to DB (overwrite any stale data)
        await self._save_state()

        logger.info(
            "flipped_reset_complete",
            balance=self.balance,
            positions=len(self.positions),
        )
        return True

    def _purge_incompatible_positions(self) -> None:
        """Remove in-memory positions for symbols that don't belong to this exchange.

        After multi-market split, state may contain non-crypto symbols (like CL=F)
        that were entered when all markets shared the same flipped/custom state.
        These can't be monitored on the current exchange (e.g. Binance).
        """
        exchange_name = getattr(self.exchange, "exchange_name", "")
        is_crypto_exchange = "binance" in exchange_name.lower() or "ccxt" in str(type(self.exchange)).lower()

        if not is_crypto_exchange:
            return  # Non-crypto exchanges can have any symbol format

        to_remove = []
        for sym in self.positions:
            # Crypto symbols contain "/" (e.g. BTC/USDT:USDT), non-crypto don't
            if "/" not in sym:
                to_remove.append(sym)

        for sym in to_remove:
            pos = self.positions.pop(sym)
            logger.info(
                "purged_incompatible_position",
                mode=self.mode_name,
                symbol=sym,
                direction=pos.direction,
                entry_price=pos.entry_price,
                reason="non_crypto_on_crypto_exchange",
            )

    # ------------------------------------------------------------------
    # Independent scan loop
    # ------------------------------------------------------------------

    async def run_loop(self) -> None:
        """Independent scan loop — runs every N minutes, completely separate from main bot."""
        if not self.enabled or not self.candle_manager or not self.exchange:
            logger.warning("flipped_loop_disabled", mode=self.mode_name, reason="missing dependencies or disabled")
            return

        logger.info(
            "flipped_loop_started",
            mode=self.mode_name,
            interval_minutes=self.scan_interval,
            leverage=self.leverage,
            sl_buffer=self.sl_buffer,
            threshold=FLIPPED_THRESHOLD,
            flip_mode=self.flip_mode,
            flip_direction=self.flip_direction,
            scanning_active=self._scanning_active,
            margin_pct=self.max_position_pct,
        )

        # Reconcile in-memory positions with DB (handles restarts)
        await self._reconcile_from_db()

        # Auto-purge positions for symbols the exchange can't handle
        # (e.g. CL=F stuck in crypto engine after multi-market was split)
        self._purge_incompatible_positions()

        # Start independent position monitor (every 60s)
        monitor_task = asyncio.create_task(self._monitor_loop())

        # Run an immediate scan on startup (only if scanning is active)
        if self._scanning_active:
            try:
                await self._run_scan()
            except Exception as e:
                logger.error("flipped_startup_scan_failed", mode=self.mode_name, error=str(e))
        else:
            logger.info("bot_scanning_paused_at_startup", mode=self.mode_name)

        while True:
            try:
                # Sleep in 5s increments so we can respond quickly to trigger requests
                remaining = self.scan_interval * 60
                triggered = False
                while remaining > 0:
                    await asyncio.sleep(min(5, remaining))
                    remaining -= 5
                    if self._trigger_scan_event.is_set():
                        self._trigger_scan_event.clear()
                        triggered = True
                        logger.info("flipped_scan_triggered", reason="manual_trigger")
                        break
                # Check for reset request from dashboard
                if await self._check_and_apply_reset():
                    continue  # Skip scan this tick, state was just reset
                # Check if scanning is active (custom bot may be paused)
                if not self._scanning_active:
                    continue
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
                # Check for reset request from dashboard
                if await self._check_and_apply_reset():
                    continue  # Skip monitoring this tick, state was just reset
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
        """Ensure in-memory positions match DB open trades (bidirectional).

        On restart, config_overrides might be stale. This queries the DB for
        open trades (matching self.mode_name) and:
        1. Adds DB positions missing from memory
        2. REMOVES memory positions missing from DB (orphans from failed resets)
        """
        try:
            open_trades = await self.repo.get_open_trades(mode=self.mode_name)
            db_symbols = {t["symbol"] for t in open_trades} if open_trades else set()

            # --- Remove orphaned positions (in memory but not in DB) ---
            orphaned = [s for s in self.positions if s not in db_symbols]
            if orphaned:
                for sym in orphaned:
                    pos = self.positions.pop(sym)
                    self.balance += pos.margin_used  # Return margin
                    logger.warning(
                        "flipped_reconcile_removed_orphan",
                        symbol=sym,
                        margin_returned=round(pos.margin_used, 2),
                        balance=round(self.balance, 2),
                    )
                # Save cleaned state immediately
                await self._save_state()

            if not open_trades:
                # If all positions were orphans (DB was wiped by reset), reset balance to initial
                if orphaned and not self.positions:
                    logger.info(
                        "flipped_reconcile_full_reset_detected",
                        old_balance=round(self.balance, 2),
                        new_balance=self._initial_balance,
                        orphans_removed=len(orphaned),
                    )
                    self.balance = self._initial_balance
                    self.peak_balance = self._initial_balance
                    self.total_pnl = 0.0
                    self.daily_pnl = 0.0
                    self.daily_start_balance = self._initial_balance
                    self._closed_symbols.clear()
                    await self._save_state()
                else:
                    logger.info(
                        "flipped_reconcile_no_open_trades",
                        orphans_removed=len(orphaned),
                        balance=round(self.balance, 2),
                    )
                return

            # --- Add DB positions missing from memory ---
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
        logger.info("flipped_scan_start", mode=self.mode_name, scan=self._scan_count)

        # Get tradeable pairs — quality whitelist ensures only liquid, established coins
        try:
            pairs = await self.exchange.get_tradeable_pairs(
                min_volume_usd=self.min_volume_usd,
                quote_currencies=self.quote_currencies,
                quality_filter=True,  # Only established coins with reliable liquidity
            )
        except Exception as e:
            logger.error("flipped_pair_scan_failed", error=str(e))
            return

        if not pairs:
            logger.warning("flipped_no_pairs")
            return

        # Scan with simplified pipeline
        signals = await self._scan_pairs(pairs)

        # Clean up expired cooldowns
        now = datetime.now(timezone.utc)
        expired = [s for s, t in self._closed_symbols.items()
                   if (now - t).total_seconds() > self.REENTRY_COOLDOWN_HOURS * 3600]
        for s in expired:
            del self._closed_symbols[s]

        # Log signal details for debugging
        if signals:
            logger.info(
                "flipped_signals_detail",
                signals=[
                    {"symbol": s.symbol, "score": s.score, "dir": s.direction}
                    for s in signals[:10]  # top 10
                ],
            )

        # Enter flipped trades (limited by available balance)
        entered = 0
        skipped_cooldown = 0
        skipped_already_open = 0
        skipped_entry_failed = 0
        for signal in signals:
            if FLIPPED_MAX_CONCURRENT > 0 and len(self.positions) >= FLIPPED_MAX_CONCURRENT:
                break
            if self.balance < 5.0:
                break  # No margin left
            # Total exposure cap: don't deploy more than MAX_EXPOSURE_PCT of initial balance
            total_margin = sum(p.margin_used for p in self.positions.values())
            if total_margin >= self._initial_balance * MAX_EXPOSURE_PCT:
                logger.info(
                    "flipped_exposure_cap_reached",
                    total_margin=round(total_margin, 2),
                    limit=round(self._initial_balance * MAX_EXPOSURE_PCT, 2),
                )
                break
            if signal.symbol in self.positions:
                skipped_already_open += 1
                continue
            # Cooldown: don't re-enter a symbol too soon after closing
            if signal.symbol in self._closed_symbols:
                skipped_cooldown += 1
                continue
            try:
                if await self._try_enter(signal):
                    entered += 1
                else:
                    skipped_entry_failed += 1
            except Exception as e:
                logger.warning("flipped_entry_error", symbol=signal.symbol, error=str(e))

        self.last_scan_time = datetime.now(timezone.utc).isoformat()

        logger.info(
            "flipped_scan_complete",
            mode=self.mode_name,
            scan=self._scan_count,
            pairs_scanned=len(pairs),
            signals_found=len(signals),
            trades_entered=entered,
            skipped_cooldown=skipped_cooldown,
            skipped_already_open=skipped_already_open,
            skipped_entry_failed=skipped_entry_failed,
            open_positions=len(self.positions),
            balance=round(self.balance, 2),
            flip_mode=self.flip_mode,
            margin_pct=self.max_position_pct,
        )

    async def _scan_pairs(self, pairs: list[str]) -> list[SignalCandidate]:
        """Scan pipeline: sweep + displacement + pullback + HTF + timing."""
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
        """Analysis: sweep + displacement + pullback + HTF + timing.

        Now includes London/NY ranges for more sweep opportunities,
        pullback detection for better entries, and no early return
        on displacement miss (lets HTF + timing compensate).
        """
        # 1. Fetch candles
        candles: dict[str, pd.DataFrame] = {}
        for tf in SCAN_TIMEFRAMES:
            candles[tf] = await self.candle_manager.get_candles(symbol, tf, limit=200)

        # 2. Market structure on all TFs
        ms_results = {}
        for tf, df in candles.items():
            ms_results[tf] = self.ms_analyzer.analyze(df, timeframe=tf)

        # 3. Session analysis (Asian + London + NY ranges)
        session_result = self.session_analyzer.analyze(candles["1h"])

        # 4. Swing levels from 1H
        swing_high = ms_results["1h"].key_levels.get("swing_high")
        swing_low = ms_results["1h"].key_levels.get("swing_low")

        # 5. Displacement check on 1H
        vol_profile = self.vol_analyzer.analyze(candles["1h"])
        displacement_confirmed = vol_profile.displacement_detected
        displacement_direction = vol_profile.displacement_direction

        # 5.5 Volume sustainability check: if volume is declining after the
        # displacement spike, treat it as unreliable (likely a one-off liquidation
        # cascade or single whale, not sustained institutional interest).
        if displacement_confirmed and vol_profile.volume_trend == "decreasing":
            displacement_confirmed = False
            displacement_direction = None

        # 6. Sweep detection on 1H (now with London/NY ranges)
        sweep_result = self.sweep_detector.detect(
            candles_1h=candles["1h"],
            asian_high=session_result.asian_high,
            asian_low=session_result.asian_low,
            swing_high=swing_high,
            swing_low=swing_low,
            lookback=8,
            prefer_direction=displacement_direction,
            london_high=session_result.london_high,
            london_low=session_result.london_low,
            ny_high=session_result.ny_high,
            ny_low=session_result.ny_low,
        )

        # 6.5 Pullback detection (requires displacement)
        pullback_result = None
        if displacement_confirmed and vol_profile.displacement_candle_idx is not None:
            pullback_result = self.pullback_analyzer.analyze(
                candles_1h=candles["1h"],
                displacement_candle_idx=vol_profile.displacement_candle_idx,
                direction=displacement_direction,
            )

        # 7. Current price + ATR
        current_price = float(candles["1h"]["close"].iloc[-1]) if not candles["1h"].empty else 0
        atr_1h = self._compute_atr(candles.get("1h"))

        # 8. HTF direction
        htf_direction = self._resolve_htf_direction(ms_results)

        # 9. SCORING (sweep required, everything else contributes)
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

        # Displacement (25 pts) — strongly preferred, no early return
        if displacement_confirmed and displacement_direction == direction:
            score += 25
            reasons.append(f"Displacement confirmed: {direction}")
            components["displacement_confirmed"] = 25
        else:
            if displacement_confirmed and displacement_direction != direction:
                reasons.append(f"Displacement mismatch: sweep={direction}, disp={displacement_direction}")
            else:
                reasons.append("No displacement after sweep")
            components["displacement_confirmed"] = 0

        # Pullback (10 pts) — bonus, better entry price
        if pullback_result is not None and pullback_result.pullback_detected:
            score += 10
            reasons.append(
                f"Pullback: {pullback_result.retracement_pct:.0%} "
                f"retracement ({pullback_result.pullback_status})"
            )
            components["pullback_confirmed"] = 10
            current_price = pullback_result.current_price
        else:
            components["pullback_confirmed"] = 0

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
    # Leverage enrichment for smart flip
    # ------------------------------------------------------------------

    async def _enrich_signal_with_leverage(self, signal: SignalCandidate) -> float:
        """Fetch leverage data and compute sweep flip probability for a signal.

        Returns a float in [0.0, 1.0]. On any error, returns 0.5 (neutral).
        """
        try:
            funding_data = await self.exchange.fetch_funding_rate(signal.symbol)
            oi_data = await self.exchange.fetch_open_interest(signal.symbol)
            ls_data = await self.exchange.fetch_long_short_ratio(signal.symbol)

            funding_rate = float(funding_data.get("funding_rate", 0))
            oi_usd = float(oi_data.get("open_interest_usd", 0))
            ls_ratio = ls_data.get("long_short_ratio")  # May be None

            session_result = getattr(signal, "session_result", None)
            in_kz = session_result.in_kill_zone if session_result else False
            in_pkz = session_result.in_post_kill_zone if session_result else False

            prob = self.leverage_analyzer.compute_sweep_flip_probability(
                signal_direction=signal.direction,
                current_price=signal.entry_price,
                open_interest_usd=oi_usd,
                funding_rate=funding_rate,
                long_short_ratio=ls_ratio,
                in_kill_zone=in_kz,
                in_post_kill_zone=in_pkz,
            )
            return prob
        except Exception as e:
            logger.warning("leverage_enrichment_failed", symbol=signal.symbol, error=str(e)[:100])
            return 0.5  # Neutral fallback — let threshold decide

    # ------------------------------------------------------------------
    # Trade entry (flipped direction)
    # ------------------------------------------------------------------

    async def _try_enter(self, signal: SignalCandidate) -> bool:
        """Flip direction (or not) and simulate entry."""
        # ── Fetch LIVE ticker price ──────────────────────────────────
        # The signal's entry_price comes from the last 1H candle close,
        # which can be up to 59 minutes stale.  Using it directly causes
        # paper trades to "enter" at a price the market has already moved
        # past, leading to instant SL hits.  Fetch the real-time ticker
        # and replace the signal price so all SL/TP/sizing calcs use the
        # actual current price.
        try:
            ticker = await self.exchange.fetch_ticker(signal.symbol)
            live_price = float(ticker["last"])
        except Exception as e:
            logger.warning("flipped_entry_ticker_failed", symbol=signal.symbol, error=str(e)[:100])
            return False

        # Reject if live price drifted >3% from signal — candle data too stale
        price_drift = abs(live_price - signal.entry_price) / signal.entry_price if signal.entry_price else 1.0
        if price_drift > 0.03:
            logger.info(
                "flipped_entry_rejected",
                symbol=signal.symbol,
                reason="price_drift",
                signal_price=signal.entry_price,
                live_price=live_price,
                drift_pct=round(price_drift * 100, 2),
            )
            return False

        # Update signal to use live price for SL/TP/sizing calculations
        signal.entry_price = live_price

        # Determine trade direction based on flip mode
        if self.flip_mode == "always_flip":
            is_long = signal.direction == "bearish"
        elif self.flip_mode == "normal":
            is_long = signal.direction == "bullish"
        elif self.flip_mode == "smart_flip":
            sweep_prob = await self._enrich_signal_with_leverage(signal)
            should_flip = sweep_prob >= self.flip_threshold
            if should_flip:
                is_long = signal.direction == "bearish"
            else:
                is_long = signal.direction == "bullish"
            logger.info(
                "smart_flip_decision",
                mode=self.mode_name,
                symbol=signal.symbol,
                signal_direction=signal.direction,
                sweep_prob=round(sweep_prob, 3),
                threshold=self.flip_threshold,
                flipped=should_flip,
                final_direction="long" if is_long else "short",
            )
        else:
            # Fallback: use legacy flip_direction bool
            if self.flip_direction:
                is_long = signal.direction == "bearish"
            else:
                is_long = signal.direction == "bullish"
        direction = "long" if is_long else "short"

        # Calculate flipped SL
        sl_price = self._calculate_sl(signal, is_long)
        if sl_price is None:
            logger.debug("flipped_entry_rejected", symbol=signal.symbol, reason="sl_calc_failed")
            return False

        # Validate SL direction
        if is_long and sl_price >= signal.entry_price:
            logger.debug("flipped_entry_rejected", symbol=signal.symbol, reason="sl_above_entry_for_long",
                        sl=sl_price, entry=signal.entry_price)
            return False
        if not is_long and sl_price <= signal.entry_price:
            logger.debug("flipped_entry_rejected", symbol=signal.symbol, reason="sl_below_entry_for_short",
                        sl=sl_price, entry=signal.entry_price)
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
            logger.info("flipped_entry_rejected", symbol=signal.symbol, reason="rr_too_low",
                        rr=round(rr_ratio, 2), min_rr=self.min_rr)
            return False

        # Ensure SL within leverage safety (don't get liquidated before SL)
        liq_distance = signal.entry_price / self.leverage
        if sl_distance > liq_distance * 0.8:
            logger.info("flipped_entry_rejected", symbol=signal.symbol, reason="sl_beyond_liquidation",
                        sl_dist_pct=round(sl_distance/signal.entry_price*100, 2))
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

        # Minimum trade size: $min_trade_usd × leverage
        min_notional = self.config.min_trade_usd * self.leverage
        if cost_usd < min_notional:
            logger.info("flipped_skip_below_min_trade", symbol=signal.symbol,
                        cost=f"{cost_usd:.2f}", min=f"{min_notional:.2f}")
            return False

        # ── LIQUIDITY GATE: spread, depth, volume checks ──────────
        try:
            ob = await self.exchange.fetch_order_book(signal.symbol, limit=10)
            ob_bids = ob.get("bids") or []
            ob_asks = ob.get("asks") or []
            if not ob_bids or not ob_asks:
                logger.info("flipped_skip_no_orderbook", symbol=signal.symbol)
                return False

            best_bid = float(ob_bids[0][0])
            best_ask = float(ob_asks[0][0])
            mid = (best_bid + best_ask) / 2
            spread_pct = (best_ask - best_bid) / mid if mid > 0 else 1.0

            if spread_pct > self.config.max_spread_pct:
                logger.info("flipped_skip_wide_spread", symbol=signal.symbol,
                            spread_pct=f"{spread_pct:.4f}", max=f"{self.config.max_spread_pct:.4f}")
                return False

            # Depth across top 5 levels
            bid_depth = sum(float(l[0]) * float(l[1]) for l in ob_bids[:5])
            ask_depth = sum(float(l[0]) * float(l[1]) for l in ob_asks[:5])
            relevant_depth = bid_depth if is_long else ask_depth

            if relevant_depth < self.config.min_ob_depth_usd:
                logger.info("flipped_skip_thin_ob", symbol=signal.symbol,
                            depth=f"{relevant_depth:.0f}", min=f"{self.config.min_ob_depth_usd:.0f}")
                return False

            # 24h volume re-check (ticker already fetched above)
            quote_vol = float(ticker.get("quoteVolume", 0) or 0)
            if quote_vol < self.config.min_volume_usd:
                logger.info("flipped_skip_low_volume", symbol=signal.symbol,
                            vol=f"{quote_vol:,.0f}", min=f"{self.config.min_volume_usd:,.0f}")
                return False

            # Position size vs daily volume
            vol_pct = cost_usd / quote_vol if quote_vol > 0 else 1.0
            if vol_pct > self.config.max_position_volume_pct:
                logger.info("flipped_skip_pos_too_large", symbol=signal.symbol,
                            cost=f"{cost_usd:.2f}", vol=f"{quote_vol:,.0f}",
                            pct=f"{vol_pct:.6f}")
                return False

            logger.info("flipped_liquidity_passed", symbol=signal.symbol,
                        spread=f"{spread_pct:.4f}", depth=f"{relevant_depth:.0f}",
                        vol_24h=f"{quote_vol:,.0f}")
        except Exception as e:
            logger.warning("flipped_liquidity_check_failed", symbol=signal.symbol, error=str(e)[:100])
            return False  # Conservative: skip if can't verify liquidity

        # Liquidation price
        if is_long:
            liq_price = signal.entry_price * (1 - 1 / self.leverage * 0.95)
        else:
            liq_price = signal.entry_price * (1 + 1 / self.leverage * 0.95)

        # Build progressive TP tiers: 33% at TP1 (1R), 33% at TP2 (2R), 34% trailing
        tp_tiers = self._calculate_tp_tiers(
            entry_price=signal.entry_price,
            sl_price=sl_price,
            quantity=quantity,
            is_long=is_long,
        )

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
            tp_tiers=tp_tiers,
        )

        # Entry fee
        entry_fee = cost_usd * SIM_FEE_RATE

        # Save to DB
        direction_label = "FLIPPED" if self.flip_direction else "NORMAL"
        trade_record = {
            "symbol": signal.symbol,
            "direction": direction,
            "status": "open",
            "mode": self.mode_name,
            "entry_price": signal.entry_price,
            "entry_quantity": quantity,
            "entry_cost_usd": cost_usd,
            "entry_order_id": f"{self.state_key}_sim",
            "entry_time": position.entry_time.isoformat(),
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "risk_usd": quantity * sl_distance,
            "risk_reward": round(rr_ratio, 2),
            "confluence_score": signal.score,
            "signal_reasons": [f"{direction_label}:{signal.direction}\u2192{direction}"] + signal.reasons,
            "timeframes_used": {"htf": "4h", "entry": "1h"},
            "fees_usd": entry_fee,
            "leverage": self.leverage,
            "margin_used": margin_used,
            "liquidation_price": liq_price,
            "original_quantity": quantity,
            "remaining_quantity": quantity,
            "tp_tiers": [
                {"level": t.level, "price": t.price, "pct": t.pct, "quantity": t.quantity, "filled": False}
                for t in tp_tiers
            ],
            "current_tier": 0,
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
            mode=self.mode_name,
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
            flip_mode=self.flip_mode,
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
                # Throttle: only log same symbol's ticker error once per 5 min
                now_err = datetime.now(timezone.utc)
                last_logged = self._ticker_error_last_logged.get(symbol)
                if last_logged is None or (now_err - last_logged).total_seconds() >= self._ticker_error_throttle_seconds:
                    logger.warning("flipped_ticker_failed", symbol=symbol, error=str(e)[:100])
                    self._ticker_error_last_logged[symbol] = now_err
                continue

            # Update high water mark
            if pos.direction == "long":
                if current_price > pos.high_water_mark:
                    pos.high_water_mark = current_price
            else:
                if current_price < pos.high_water_mark:
                    pos.high_water_mark = current_price

            exit_reason = None

            # SL check (full exit of remaining quantity)
            # If SL was moved from original (by trailing/progressive TP), label correctly
            if pos.direction == "long" and current_price <= pos.stop_loss:
                if pos.original_stop_loss and pos.stop_loss > pos.original_stop_loss:
                    exit_reason = "trailing_stop"
                else:
                    exit_reason = "sl_hit"
            elif pos.direction == "short" and current_price >= pos.stop_loss:
                if pos.original_stop_loss and pos.stop_loss < pos.original_stop_loss:
                    exit_reason = "trailing_stop"
                else:
                    exit_reason = "sl_hit"

            # Progressive TP tiers — partial exits at 1R and 2R
            if not exit_reason and pos.tp_tiers:
                for tier in pos.tp_tiers:
                    if tier.filled or tier.price is None:
                        continue  # Skip filled tiers and tier 3 (trailing only)
                    tier_hit = False
                    if pos.direction == "long" and current_price >= tier.price:
                        tier_hit = True
                    elif pos.direction == "short" and current_price <= tier.price:
                        tier_hit = True

                    if tier_hit:
                        await self._partial_exit(pos, symbol, current_price, tier)

            # Trailing stop — activates after TP1 milestone or 2R profit
            if not exit_reason:
                sl_dist = abs(pos.entry_price - pos.original_stop_loss)
                if sl_dist > 0:
                    if pos.direction == "long":
                        r_multiple = (pos.high_water_mark - pos.entry_price) / sl_dist
                    else:
                        r_multiple = (pos.entry_price - pos.high_water_mark) / sl_dist

                    # Trailing activates after first partial TP or 2R profit
                    trailing_active = pos.current_tier >= 1 or r_multiple >= self.trailing_activation_rr

                    if trailing_active:
                        atr = atr_values.get(symbol, 0)
                        atr_trail = atr * self.trailing_atr_multiplier if atr > 0 else sl_dist * 0.75

                        # After TP1 but before TP2: use wider trail (1R = SL distance)
                        # so the position has room to reach TP2.  After TP2 (runner
                        # portion): tighten to ATR-based trail to lock in profit.
                        if pos.current_tier >= 2:
                            trail_dist = atr_trail          # Tight trail for runner
                        else:
                            trail_dist = max(atr_trail, sl_dist)  # Wide trail until TP2

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
                    remaining_qty=pos.quantity,
                )
                await self._close_position(symbol, current_price, exit_reason)
                # Cooldown only on stop-loss exits — TP/trailing exits allow
                # immediate re-entry (enables reverse trades after profit)
                if exit_reason == "sl_hit":
                    self._closed_symbols[symbol] = datetime.now(timezone.utc)
                closed += 1

        if checked > 0 or ticker_errors > 0:
            self._monitor_tick_count += 1
            # Only log monitor tick every 5th cycle (every 5 min) to reduce noise
            if self._monitor_tick_count % 5 == 0 or closed > 0:
                logger.info(
                    "flipped_monitor_tick",
                    mode=self.mode_name,
                    checked=checked,
                    closed=closed,
                    errors=ticker_errors,
                    open=len(self.positions),
                    scan_count=self._scan_count,
                    last_scan=self.last_scan_time,
                    scanning=self._scanning_active,
                )

    async def _partial_exit(self, pos: Position, symbol: str, current_price: float, tier: TakeProfitTier) -> None:
        """Simulate a partial exit at a TP tier — take some profit, reduce position."""
        exit_qty = tier.quantity
        if exit_qty <= 0 or exit_qty > pos.quantity:
            return

        # Calculate partial PnL
        if pos.direction == "long":
            pnl = (current_price - pos.entry_price) * exit_qty
        else:
            pnl = (pos.entry_price - current_price) * exit_qty

        exit_fee = exit_qty * current_price * SIM_FEE_RATE
        pnl -= exit_fee

        # Mark tier as filled
        tier.filled = True
        tier.fill_price = current_price
        tier.fill_time = datetime.now(timezone.utc)
        pos.current_tier = tier.level

        # Reduce position quantity
        pos.quantity -= exit_qty

        # Progressive SL: move SL to previous TP level after each tier
        if tier.level == 1:
            # TP1 hit → move SL to breakeven (entry price + fee buffer)
            fee_buffer = pos.entry_price * 0.001  # 0.1% buffer for fees
            if pos.direction == "long":
                be_sl = pos.entry_price + fee_buffer
                if be_sl > pos.stop_loss:
                    pos.stop_loss = be_sl
            else:
                be_sl = pos.entry_price - fee_buffer
                if be_sl < pos.stop_loss:
                    pos.stop_loss = be_sl
        elif tier.level >= 2 and pos.tp_tiers:
            # TP2+ hit → move SL to previous tier's price (lock in profit)
            prev_tier = next(
                (t for t in pos.tp_tiers if t.level == tier.level - 1),
                None,
            )
            if prev_tier and prev_tier.price:
                if pos.direction == "long" and prev_tier.price > pos.stop_loss:
                    pos.stop_loss = prev_tier.price
                elif pos.direction == "short" and prev_tier.price < pos.stop_loss:
                    pos.stop_loss = prev_tier.price
                logger.info("flipped_sl_moved_to_prev_tp", symbol=symbol,
                            tier=tier.level, new_sl=prev_tier.price)

        # Compute margin freed by this partial exit
        partial_margin = (exit_qty / pos.original_quantity) * pos.margin_used if pos.original_quantity > 0 else 0

        # Return freed margin + profit to balance
        self.balance += partial_margin + pnl
        self.daily_pnl += pnl
        self.total_pnl += pnl

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        # Log partial exit to DB
        partial_cost = exit_qty * pos.entry_price
        pnl_pct = (pnl / partial_cost * 100) if partial_cost > 0 else 0

        try:
            await self.repo.log_partial_exit(
                trade_id=pos.trade_id,
                tier=tier.level,
                exit_price=current_price,
                exit_quantity=exit_qty,
                exit_order_id=f"{self.state_key}_sim",
                exit_reason=f"tp{tier.level}_hit",
                pnl_usd=round(pnl, 4),
                pnl_percent=round(pnl_pct, 2),
                fees_usd=round(exit_fee, 4),
                remaining_quantity=pos.quantity,
                new_stop_loss=pos.stop_loss,
            )
        except Exception as e:
            logger.warning("flipped_partial_exit_db_failed", symbol=symbol, tier=tier.level, error=str(e)[:100])

        # Update the trade record in DB with new remaining qty + SL
        try:
            await self.repo.update_trade(pos.trade_id, {
                "remaining_quantity": pos.quantity,
                "current_tier": pos.current_tier,
                "stop_loss": pos.stop_loss,
                "tp_tiers": [
                    {"level": t.level, "price": t.price, "pct": t.pct, "quantity": t.quantity,
                     "filled": t.filled, "fill_price": t.fill_price}
                    for t in pos.tp_tiers
                ] if pos.tp_tiers else None,
            })
        except Exception as e:
            logger.warning("flipped_partial_exit_update_failed", symbol=symbol, error=str(e)[:100])

        logger.info(
            "flipped_partial_exit",
            symbol=symbol,
            tier=tier.level,
            direction=pos.direction,
            price=current_price,
            exit_qty=round(exit_qty, 6),
            remaining_qty=round(pos.quantity, 6),
            pnl=round(pnl, 2),
            new_sl=pos.stop_loss,
            balance=round(self.balance, 2),
        )

        # Save state after each partial exit
        await self._save_state()

    async def _close_position(self, symbol: str, exit_price: float, reason: str) -> None:
        """Simulate closing a flipped position (remaining quantity after any partial exits)."""
        pos = self.positions.pop(symbol, None)
        if not pos:
            return

        # Calculate PnL on remaining quantity only
        if pos.direction == "long":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        # Exit fee on remaining quantity
        remaining_cost = pos.quantity * pos.entry_price
        exit_fee = remaining_cost * SIM_FEE_RATE
        pnl -= exit_fee

        # Compute remaining margin (proportion of original margin)
        if pos.original_quantity > 0:
            remaining_ratio = pos.quantity / pos.original_quantity
        else:
            remaining_ratio = 1.0
        remaining_margin = pos.margin_used * remaining_ratio

        # Return remaining margin + pnl to balance
        self.balance += remaining_margin + pnl
        self.daily_pnl += pnl
        self.total_pnl += pnl

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        # Accumulate total PnL across all partial exits for the DB record
        total_trade_pnl = pnl
        total_trade_fees = remaining_cost * SIM_FEE_RATE + exit_fee
        try:
            partial_exits = await self.repo.get_partial_exits(pos.trade_id)
            total_trade_pnl += sum(float(pe.get("pnl_usd", 0)) for pe in partial_exits)
            total_trade_fees += sum(float(pe.get("fees_usd", 0)) for pe in partial_exits)
        except Exception:
            pass  # If partial exits query fails, just use the final exit PnL

        pnl_pct = (total_trade_pnl / pos.cost_usd * 100) if pos.cost_usd > 0 else 0

        await self.repo.close_trade(
            trade_id=pos.trade_id,
            exit_price=exit_price,
            exit_quantity=pos.quantity,
            exit_order_id=f"{self.state_key}_sim",
            exit_reason=reason,
            pnl_usd=round(total_trade_pnl, 4),
            pnl_percent=round(pnl_pct, 2),
            fees_usd=round(total_trade_fees, 4),
        )

        logger.info(
            "flipped_exit",
            symbol=symbol,
            direction=pos.direction,
            reason=reason,
            entry=pos.entry_price,
            exit=exit_price,
            final_pnl=round(pnl, 2),
            total_pnl=round(total_trade_pnl, 2),
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
            overrides[self.state_key] = self.to_state_dict()
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

    def _calculate_tp_tiers(
        self,
        entry_price: float,
        sl_price: float,
        quantity: float,
        is_long: bool,
    ) -> list[TakeProfitTier]:
        """Build 3-tier progressive TP plan: 33% at 1R, 33% at 2R, 34% trailing."""
        sl_distance = abs(entry_price - sl_price)

        if is_long:
            tp1_price = entry_price + sl_distance * 1.0   # 1R
            tp2_price = entry_price + sl_distance * 2.0   # 2R
        else:
            tp1_price = entry_price - sl_distance * 1.0
            tp2_price = entry_price - sl_distance * 2.0

        tp1_qty = round(quantity * 0.33, 8)
        tp2_qty = round(quantity * 0.33, 8)
        tp3_qty = round(quantity - tp1_qty - tp2_qty, 8)

        return [
            TakeProfitTier(level=1, price=tp1_price, pct=0.33, quantity=tp1_qty),
            TakeProfitTier(level=2, price=tp2_price, pct=0.33, quantity=tp2_qty),
            TakeProfitTier(level=3, price=None, pct=0.34, quantity=tp3_qty),  # trailing only
        ]

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
            pos_data = {
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
                "current_tier": pos.current_tier,
            }
            if pos.tp_tiers:
                pos_data["tp_tiers"] = [
                    {"level": t.level, "price": t.price, "pct": t.pct,
                     "quantity": t.quantity, "filled": t.filled, "fill_price": t.fill_price}
                    for t in pos.tp_tiers
                ]
            positions_data[sym] = pos_data
        return {
            "balance": self.balance,
            "peak_balance": self.peak_balance,
            "daily_start_balance": self.daily_start_balance,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "daily_trade_count": self.daily_trade_count,
            "scan_count": self._scan_count,
            "last_scan_time": self.last_scan_time,
            "positions": positions_data,
            # Configurable settings (survive restarts)
            "flip_direction": self.flip_direction,
            "margin_pct": self.max_position_pct,
            "flip_mode": self.flip_mode,
            "flip_threshold": self.flip_threshold,
            "scanning_active": self._scanning_active,
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
        self._scan_count = int(data.get("scan_count", 0))
        self.last_scan_time = data.get("last_scan_time")
        # Restore configurable settings (if saved)
        if "flip_direction" in data:
            self.flip_direction = bool(data["flip_direction"])
        if "margin_pct" in data:
            self.max_position_pct = float(data["margin_pct"])
            self.max_risk_pct = float(data["margin_pct"])
        if "flip_mode" in data:
            mode = str(data["flip_mode"])
            if mode in ("always_flip", "smart_flip", "normal"):
                self.flip_mode = mode
        if "flip_threshold" in data:
            self.flip_threshold = max(0.0, min(1.0, float(data["flip_threshold"])))
        if "scanning_active" in data:
            self._scanning_active = bool(data["scanning_active"])

        positions_data = data.get("positions", {})
        for sym, pd_ in positions_data.items():
            try:
                entry_time = datetime.now(timezone.utc)
                if pd_.get("entry_time"):
                    entry_time = datetime.fromisoformat(str(pd_["entry_time"]))
                # Rebuild TP tiers from saved state
                tp_tiers = None
                if pd_.get("tp_tiers"):
                    tp_tiers = [
                        TakeProfitTier(
                            level=int(t["level"]),
                            price=float(t["price"]) if t.get("price") is not None else None,
                            pct=float(t.get("pct", 0.33)),
                            quantity=float(t.get("quantity", 0)),
                            filled=bool(t.get("filled", False)),
                            fill_price=float(t.get("fill_price", 0)),
                        )
                        for t in pd_["tp_tiers"]
                    ]

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
                    current_tier=int(pd_.get("current_tier", 0)),
                    tp_tiers=tp_tiers,
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.warning("flipped_position_restore_failed", symbol=sym, error=str(e))

        logger.info(
            "flipped_state_restored",
            balance=self.balance,
            open_positions=len(self.positions),
            total_pnl=self.total_pnl,
        )
