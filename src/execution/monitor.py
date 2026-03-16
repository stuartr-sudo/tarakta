from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from src.exchange.models import ExitSignal, Position
from src.exchange.protocol import TradingHours
from src.exchange.trading_hours import TradingHoursManager
from src.strategy.market_structure import MarketStructureAnalyzer
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Shared market structure analyzer for Agent 3
_ms_analyzer = MarketStructureAnalyzer()

TRAILING_STOP_PCT_FALLBACK = 0.02  # Fallback if ATR not available
TICKER_FETCH_TIMEOUT = 45  # seconds — safety net above ccxt's 30s timeout


class PositionMonitor:
    """Monitors open positions for SL/TP hits and ATR-based trailing stop updates."""

    def __init__(
        self,
        trailing_activation_rr: float = 2.0,
        trailing_atr_multiplier: float = 1.5,
        breakeven_activation_rr: float = 0.5,
        max_hold_hours: float = 4.0,
        stale_close_below_rr: float = 0.0,
        position_agent=None,
        position_agent_interval_minutes: float = 15.0,
    ) -> None:
        self.trailing_activation_rr = trailing_activation_rr
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.breakeven_activation_rr = breakeven_activation_rr
        self.max_hold_hours = max_hold_hours
        self.stale_close_below_rr = stale_close_below_rr
        self._trading_hours_mgr = TradingHoursManager()
        # Throttle repeated ticker errors: only log per symbol once per 5 min
        self._ticker_error_last_logged: dict[str, datetime] = {}
        self._ticker_error_throttle_seconds = 300

        # Agent 3 (Position Manager) — AI-powered position monitoring
        self._position_agent = position_agent
        self._agent3_interval = position_agent_interval_minutes * 60  # seconds
        self._agent3_last_check: dict[str, float] = {}  # symbol → timestamp
        self._agent3_locks: dict[str, asyncio.Lock] = {}  # Req 7: per-symbol concurrency lock
        self._agent3_shadow_mode: bool = False  # Req 8: set from config at runtime
        # Cache of latest prices fetched during check_positions (for drawdown calc)
        self.last_prices: dict[str, float] = {}

    async def check_positions(
        self, positions: dict[str, Position], exchange, atr_values: dict[str, float] | None = None,
        trading_hours: TradingHours | None = None,
        market_filter=None,
        candle_manager=None,
    ) -> list[ExitSignal]:
        """Check all open positions against current prices (algorithmic SL/TP only).

        IMPORTANT: Agent 3 is NOT called here. Call run_agent3_checks() separately
        AFTER processing algorithmic exits, so that slow LLM calls never block
        time-critical TP/SL execution.

        Args:
            positions: Map of symbol -> Position
            exchange: Exchange client for ticker fetches
            atr_values: Map of symbol -> current ATR value (from 1H candles)
            market_filter: Optional market filter for BTC context (Agent 3)
            candle_manager: Optional CandleManager for Agent 3 market data
        """
        if atr_values is None:
            atr_values = {}
        exits: list[ExitSignal] = []
        current_prices: dict[str, float] = {}

        for symbol, pos in positions.items():
            try:
                ticker = await asyncio.wait_for(
                    exchange.fetch_ticker(symbol), timeout=TICKER_FETCH_TIMEOUT
                )
                current_price = float(ticker["last"])
                current_prices[symbol] = current_price
            except asyncio.TimeoutError:
                logger.warning("ticker_fetch_timeout", symbol=symbol, timeout=TICKER_FETCH_TIMEOUT)
                continue
            except Exception as e:
                # Throttle: only log same symbol's ticker error once per 5 min
                now_err = datetime.now(timezone.utc)
                last_logged = self._ticker_error_last_logged.get(symbol)
                if last_logged is None or (now_err - last_logged).total_seconds() >= self._ticker_error_throttle_seconds:
                    logger.warning("ticker_fetch_failed", symbol=symbol, error=str(e))
                    self._ticker_error_last_logged[symbol] = now_err
                continue

            atr = atr_values.get(symbol, 0.0)
            exit_signal = self._evaluate_position(symbol, pos, current_price, atr, trading_hours)
            if exit_signal:
                exits.append(exit_signal)

        # Cache prices for equity/drawdown calculation
        self.last_prices = current_prices
        return exits

    async def run_agent3_checks(
        self, positions: dict[str, Position], exchange,
        atr_values: dict[str, float] | None = None,
        exited_symbols: set[str] | None = None,
        market_filter=None,
        candle_manager=None,
    ) -> list[ExitSignal]:
        """Run Agent 3 position management checks separately from algorithmic SL/TP.

        Called AFTER algorithmic exits have been processed, so slow LLM calls
        never delay time-critical TP/SL execution.
        """
        if not self._position_agent:
            return []
        if atr_values is None:
            atr_values = {}
        if exited_symbols is None:
            exited_symbols = set()

        import time as _time
        now_ts = _time.time()
        current_prices = self.last_prices  # Reuse prices from check_positions
        exits: list[ExitSignal] = []

        for symbol, pos in positions.items():
            if symbol in exited_symbols:
                continue
            if symbol not in current_prices:
                continue
            last_check = self._agent3_last_check.get(symbol, 0)
            if now_ts - last_check < self._agent3_interval:
                continue

            self._agent3_last_check[symbol] = now_ts

            if symbol not in self._agent3_locks:
                self._agent3_locks[symbol] = asyncio.Lock()
            lock = self._agent3_locks[symbol]
            if lock.locked():
                logger.debug("agent3_skipped_locked", symbol=symbol)
                continue

            try:
                async with lock:
                    agent3_exit = await asyncio.wait_for(
                        self._run_agent3(
                            symbol, pos, current_prices[symbol],
                            atr_values.get(symbol, 0.0), market_filter,
                            exchange=exchange,
                            candle_manager=candle_manager,
                        ),
                        timeout=60,  # 60s max for LLM + data fetching
                    )
                    if agent3_exit:
                        if self._agent3_shadow_mode:
                            logger.info(
                                "agent3_shadow_mode_decision",
                                symbol=symbol,
                                reason=agent3_exit.reason,
                                price=agent3_exit.price,
                            )
                        else:
                            exits.append(agent3_exit)
            except asyncio.TimeoutError:
                logger.warning("agent3_timeout", symbol=symbol, timeout=60)
            except Exception as e:
                logger.warning("agent3_check_failed", symbol=symbol, error=str(e))

        return exits

    async def _run_agent3(
        self, symbol: str, pos: Position, current_price: float, atr: float,
        market_filter=None, exchange=None, candle_manager=None,
    ) -> ExitSignal | None:
        """Run Agent 3 for a single position and return ExitSignal if action needed."""
        # Build context
        sl_distance = abs(pos.entry_price - pos.stop_loss) if pos.stop_loss else 0
        if pos.direction == "long":
            unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
            unrealized_rr = (current_price - pos.entry_price) / sl_distance if sl_distance > 0 else 0
        else:
            unrealized_pnl = (pos.entry_price - current_price) * pos.quantity
            unrealized_rr = (pos.entry_price - current_price) / sl_distance if sl_distance > 0 else 0

        unrealized_pnl_pct = (unrealized_pnl / (pos.entry_price * pos.quantity) * 100) if pos.entry_price * pos.quantity > 0 else 0

        held_minutes = 0
        if pos.entry_time:
            held_minutes = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 60

        # Agent 1's reasoning (from dedicated field)
        agent1_reasoning = getattr(pos, "agent1_reasoning", "") or ""

        # BTC context
        btc_trend = "unknown"
        btc_1h_change = 0.0
        if market_filter:
            btc_trend = getattr(market_filter, "_cached_btc_trend", "unknown") or "unknown"
            btc_1h_change = getattr(market_filter, "_cached_btc_1h_change", 0) or 0

        # Current tier
        current_tier = 0
        if pos.tp_tiers:
            current_tier = sum(1 for t in pos.tp_tiers if t.filled)

        # --- Fetch market data for Agent 3 ---
        candles_5m = []
        candles_1h = []
        market_structure = {}
        funding_rate = None

        if candle_manager:
            try:
                df_5m = await candle_manager.get_candles(symbol, "5m", limit=20)
                if df_5m is not None and not df_5m.empty:
                    candles_5m = self._format_candles(df_5m)
            except Exception as e:
                logger.debug("agent3_candle_5m_fetch_failed", symbol=symbol, error=str(e))

            try:
                df_1h = await candle_manager.get_candles(symbol, "1h", limit=50)
                if df_1h is not None and not df_1h.empty:
                    candles_1h = self._format_candles(df_1h, limit=10)
                    # Run market structure analysis on 1h candles
                    try:
                        ms_result = _ms_analyzer.analyze(df_1h, timeframe="1h")
                        market_structure["1h"] = {
                            "trend": ms_result.trend,
                            "strength": ms_result.structure_strength,
                            "last_bos": ms_result.last_bos_direction,
                            "last_choch": ms_result.last_choch_direction,
                        }
                    except Exception as e:
                        logger.debug("agent3_ms_1h_failed", symbol=symbol, error=str(e))
            except Exception as e:
                logger.debug("agent3_candle_1h_fetch_failed", symbol=symbol, error=str(e))

            try:
                df_4h = await candle_manager.get_candles(symbol, "4h", limit=50)
                if df_4h is not None and not df_4h.empty:
                    try:
                        ms_result_4h = _ms_analyzer.analyze(df_4h, timeframe="4h")
                        market_structure["4h"] = {
                            "trend": ms_result_4h.trend,
                            "strength": ms_result_4h.structure_strength,
                            "last_bos": ms_result_4h.last_bos_direction,
                            "last_choch": ms_result_4h.last_choch_direction,
                        }
                    except Exception as e:
                        logger.debug("agent3_ms_4h_failed", symbol=symbol, error=str(e))
            except Exception as e:
                logger.debug("agent3_candle_4h_fetch_failed", symbol=symbol, error=str(e))

        # Fetch funding rate from exchange
        if exchange and hasattr(exchange, "fetch_funding_rate"):
            try:
                fr_data = await exchange.fetch_funding_rate(symbol)
                funding_rate = fr_data.get("funding_rate")
            except Exception as e:
                logger.debug("agent3_funding_rate_failed", symbol=symbol, error=str(e))

        context = {
            "symbol": symbol,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "current_price": current_price,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "unrealized_pnl_usd": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "unrealized_rr": unrealized_rr,
            "held_minutes": held_minutes,
            "atr": atr,
            "current_tier": current_tier,
            "original_quantity": pos.original_quantity or pos.quantity,
            "remaining_quantity": pos.quantity,
            "confluence_score": getattr(pos, "confluence_score", 0),
            "agent1_reasoning": agent1_reasoning,
            "btc_trend": btc_trend,
            "btc_1h_change": btc_1h_change,
            "funding_rate": funding_rate,
            # New market data for Agent 3
            "candles_5m": candles_5m,
            "candles_1h": candles_1h,
            "market_structure": market_structure,
        }

        decision = await self._position_agent.evaluate_position(context)

        # Store Agent 3's decision on the Position for DB persistence
        pos.last_agent3_action = decision.action
        pos.last_agent3_reasoning = decision.reasoning
        pos.agent3_confidence = decision.confidence

        # ══════════════════════════════════════════════════════════════
        # HARD GUARDRAILS — these CANNOT be bypassed by the LLM
        # ══════════════════════════════════════════════════════════════

        is_close_action = decision.action in ("CLOSE_FULL", "CLOSE_PARTIAL")

        # 1. Minimum confidence: CLOSE needs 70+, CLOSE_FULL needs 80+
        min_conf = 80 if decision.action == "CLOSE_FULL" else 70
        if is_close_action and decision.confidence < min_conf:
            logger.info(
                "agent3_blocked_low_confidence",
                symbol=symbol, action=decision.action,
                confidence=decision.confidence, min_required=min_conf,
            )
            decision.action = "HOLD"

        # 2. Minimum hold time: no closes in the first 30 minutes
        if is_close_action and held_minutes < 30:
            logger.info(
                "agent3_blocked_too_early",
                symbol=symbol, action=decision.action,
                held_minutes=f"{held_minutes:.0f}", min_required=30,
            )
            decision.action = "HOLD"

        # 3. No closing at a loss — that's what the SL is for
        if is_close_action and unrealized_rr < 0:
            logger.info(
                "agent3_blocked_negative_rr",
                symbol=symbol, action=decision.action,
                unrealized_rr=f"{unrealized_rr:.2f}",
            )
            decision.action = "HOLD"

        # 4. Cooldown: no back-to-back closes — must wait 15 min after last close action
        last_close_ts = getattr(pos, "_agent3_last_close_ts", 0)
        import time as _time_guard
        if is_close_action and (_time_guard.time() - last_close_ts) < 900:  # 15 min
            logger.info(
                "agent3_blocked_cooldown",
                symbol=symbol, action=decision.action,
                seconds_since_last=f"{_time_guard.time() - last_close_ts:.0f}",
            )
            decision.action = "HOLD"

        # 5. CLOSE_FULL requires unrealized_rr < 0.3 OR 2+ TP tiers already filled
        #    (i.e. only close-full if trade is barely profitable OR already won)
        if decision.action == "CLOSE_FULL":
            tiers_filled = sum(1 for t in (pos.tp_tiers or []) if t.filled)
            if unrealized_rr >= 0.3 and tiers_filled < 2:
                # Trade is profitable and hasn't hit 2 TPs — let the algo handle it
                logger.info(
                    "agent3_blocked_profitable_trade",
                    symbol=symbol, unrealized_rr=f"{unrealized_rr:.2f}",
                    tiers_filled=tiers_filled,
                    reasoning=decision.reasoning[:100],
                )
                decision.action = "HOLD"

        # ══════════════════════════════════════════════════════════════

        if decision.action == "CLOSE_FULL":
            pos._agent3_last_close_ts = _time_guard.time()
            logger.info(
                "agent3_close_full",
                symbol=symbol,
                direction=pos.direction,
                unrealized_rr=f"{unrealized_rr:.2f}",
                reasoning=decision.reasoning[:200],
                confidence=decision.confidence,
            )
            return ExitSignal(symbol=symbol, reason="agent3_closed", price=current_price)

        elif decision.action == "CLOSE_PARTIAL" and decision.partial_close_pct > 0:
            pos._agent3_last_close_ts = _time_guard.time()
            partial_qty = pos.quantity * min(decision.partial_close_pct, 0.5)  # Cap at 50%
            logger.info(
                "agent3_close_partial",
                symbol=symbol,
                partial_pct=f"{decision.partial_close_pct:.0%}",
                partial_qty=partial_qty,
                reasoning=decision.reasoning[:200],
                confidence=decision.confidence,
            )
            return ExitSignal(
                symbol=symbol,
                reason="agent3_partial",
                price=current_price,
                is_partial=True,
                partial_quantity=partial_qty,
            )

        elif decision.action == "TIGHTEN_SL" and decision.suggested_sl > 0:
            # Validate: can only tighten (move SL in profitable direction)
            if pos.direction == "long" and decision.suggested_sl > pos.stop_loss:
                old_sl = pos.stop_loss
                pos.stop_loss = decision.suggested_sl
                pos.last_agent3_sl = f"{old_sl:.6g} → {decision.suggested_sl:.6g}"
                logger.info(
                    "agent3_tighten_sl",
                    symbol=symbol,
                    old_sl=old_sl,
                    new_sl=decision.suggested_sl,
                    reasoning=decision.reasoning[:200],
                    confidence=decision.confidence,
                )
            elif pos.direction == "short" and decision.suggested_sl < pos.stop_loss:
                old_sl = pos.stop_loss
                pos.stop_loss = decision.suggested_sl
                pos.last_agent3_sl = f"{old_sl:.6g} → {decision.suggested_sl:.6g}"
                logger.info(
                    "agent3_tighten_sl",
                    symbol=symbol,
                    old_sl=old_sl,
                    new_sl=decision.suggested_sl,
                    reasoning=decision.reasoning[:200],
                    confidence=decision.confidence,
                )
            # else: Agent tried to widen SL — ignore silently

        # HOLD or invalid tighten → no exit signal
        return None

    @staticmethod
    def _format_candles(df, limit: int = 20) -> list[dict]:
        """Convert a candle DataFrame to a list of dicts for Agent 3 context."""
        rows = []
        for ts, row in df.tail(limit).iterrows():
            rows.append({
                "time": str(ts),
                "open": round(float(row["open"]), 6),
                "high": round(float(row["high"]), 6),
                "low": round(float(row["low"]), 6),
                "close": round(float(row["close"]), 6),
                "volume": round(float(row.get("volume", 0)), 2),
            })
        return rows

    def _evaluate_position(
        self, symbol: str, pos: Position, current_price: float, atr: float,
        trading_hours: TradingHours | None = None,
    ) -> ExitSignal | None:
        """Evaluate a single position for exit conditions."""
        if pos.direction == "long":
            return self._evaluate_long(symbol, pos, current_price, atr, trading_hours)
        else:
            return self._evaluate_short(symbol, pos, current_price, atr, trading_hours)

    def _evaluate_long(
        self, symbol: str, pos: Position, current_price: float, atr: float,
        trading_hours: TradingHours | None = None,
    ) -> ExitSignal | None:
        # Update high water mark
        if current_price > pos.high_water_mark:
            pos.high_water_mark = current_price

        # 1. Check stop loss (price drops below SL) — full exit of remaining quantity
        if current_price <= pos.stop_loss:
            # If SL was moved up from original (by trailing stop or progressive TP),
            # this is a profitable exit — label it correctly
            if pos.original_stop_loss and pos.stop_loss > pos.original_stop_loss:
                reason = "trailing_stop"
            else:
                reason = "sl_hit"
            logger.info(reason, symbol=symbol, direction="long",
                        price=current_price, sl=pos.stop_loss, original_sl=pos.original_stop_loss)
            return ExitSignal(symbol=symbol, reason=reason, price=current_price)

        # 2. Progressive TP tiers (if enabled)
        if pos.tp_tiers:
            for tier in pos.tp_tiers:
                if tier.filled or tier.price is None:
                    continue
                if current_price >= tier.price:
                    logger.info(
                        f"tp{tier.level}_hit", symbol=symbol, direction="long",
                        price=current_price, tp=tier.price, quantity=tier.quantity,
                    )
                    return ExitSignal(
                        symbol=symbol,
                        reason=f"tp{tier.level}_hit",
                        price=current_price,
                        is_partial=True,
                        partial_quantity=tier.quantity,
                        tier=tier.level,
                    )
            # All priced tiers checked; fall through to trailing stop
        elif pos.take_profit and current_price >= pos.take_profit:
            # Legacy single TP mode
            logger.info("tp_hit", symbol=symbol, direction="long",
                        price=current_price, tp=pos.take_profit)
            return ExitSignal(symbol=symbol, reason="tp_hit", price=current_price)

        # 3. Early breakeven protection — move SL to entry once 0.5R in profit
        #    (only if SL hasn't already been moved above entry by TP1 hit)
        original_sl_dist = (pos.entry_price - pos.original_stop_loss) if pos.original_stop_loss else 0
        sl_distance = original_sl_dist if original_sl_dist > 0 else (pos.entry_price - pos.stop_loss)
        if sl_distance > 0:
            unrealized_rr = (current_price - pos.entry_price) / sl_distance
            if (
                unrealized_rr >= self.breakeven_activation_rr
                and pos.stop_loss < pos.entry_price  # Not already at/above breakeven
            ):
                pos.stop_loss = pos.entry_price
                logger.info(
                    "sl_moved_to_breakeven_early",
                    symbol=symbol,
                    direction="long",
                    unrealized_rr=f"{unrealized_rr:.2f}",
                    new_sl=pos.entry_price,
                )

        # 4. Stale trade auto-close — close losing trades open too long
        #    Uses market-open hours only (overnight/weekend doesn't count for stocks/commodities)
        if self.max_hold_hours > 0 and pos.entry_time:
            now_utc = datetime.now(timezone.utc)
            hours_open = self._trading_hours_mgr.market_open_hours_between(
                pos.entry_time, now_utc, trading_hours,
            )
            if hours_open >= self.max_hold_hours:
                stale_rr = (current_price - pos.entry_price) / sl_distance if sl_distance > 0 else 0
                if stale_rr < self.stale_close_below_rr:
                    logger.info(
                        "stale_trade_closed",
                        symbol=symbol,
                        direction="long",
                        hours_open=f"{hours_open:.1f}",
                        unrealized_rr=f"{stale_rr:.2f}",
                        max_hold_hours=self.max_hold_hours,
                    )
                    return ExitSignal(symbol=symbol, reason="stale_close", price=current_price)

        # 5. ATR-based trailing stop — use original SL distance for R:R calculation
        if sl_distance > 0:
            unrealized_rr = (current_price - pos.entry_price) / sl_distance
            if unrealized_rr >= self.trailing_activation_rr:
                # ATR-dynamic trailing: high_water_mark - ATR * multiplier
                if atr > 0:
                    atr_trail = atr * self.trailing_atr_multiplier
                else:
                    # Fallback to percentage if ATR unavailable
                    atr_trail = pos.high_water_mark * TRAILING_STOP_PCT_FALLBACK

                # After TP1 but before TP2: use wider trail (1R = SL distance)
                # so the position has room to reach TP2.  After TP2: tighten.
                tier = getattr(pos, "current_tier", 0) or 0
                if tier >= 2:
                    trail_dist = atr_trail          # Tight trail for runner
                else:
                    trail_dist = max(atr_trail, sl_distance)  # Wide trail until TP2

                trailing_sl = pos.high_water_mark - trail_dist
                if trailing_sl > pos.stop_loss:
                    pos.stop_loss = trailing_sl
                if current_price <= trailing_sl:
                    return ExitSignal(symbol=symbol, reason="trailing_stop",
                                      price=current_price)

        return None

    def _evaluate_short(
        self, symbol: str, pos: Position, current_price: float, atr: float,
        trading_hours: TradingHours | None = None,
    ) -> ExitSignal | None:
        # Update low water mark (for shorts, lower is better)
        if current_price < pos.high_water_mark:
            pos.high_water_mark = current_price

        # 1. Check stop loss (price rises above SL) — full exit of remaining quantity
        if current_price >= pos.stop_loss:
            # If SL was moved down from original (by trailing stop or progressive TP),
            # this is a profitable exit — label it correctly
            if pos.original_stop_loss and pos.stop_loss < pos.original_stop_loss:
                reason = "trailing_stop"
            else:
                reason = "sl_hit"
            logger.info(reason, symbol=symbol, direction="short",
                        price=current_price, sl=pos.stop_loss, original_sl=pos.original_stop_loss)
            return ExitSignal(symbol=symbol, reason=reason, price=current_price)

        # 2. Progressive TP tiers (if enabled)
        if pos.tp_tiers:
            for tier in pos.tp_tiers:
                if tier.filled or tier.price is None:
                    continue
                if current_price <= tier.price:
                    logger.info(
                        f"tp{tier.level}_hit", symbol=symbol, direction="short",
                        price=current_price, tp=tier.price, quantity=tier.quantity,
                    )
                    return ExitSignal(
                        symbol=symbol,
                        reason=f"tp{tier.level}_hit",
                        price=current_price,
                        is_partial=True,
                        partial_quantity=tier.quantity,
                        tier=tier.level,
                    )
        elif pos.take_profit and current_price <= pos.take_profit:
            # Legacy single TP mode
            logger.info("tp_hit", symbol=symbol, direction="short",
                        price=current_price, tp=pos.take_profit)
            return ExitSignal(symbol=symbol, reason="tp_hit", price=current_price)

        # 3. Early breakeven protection — move SL to entry once 0.5R in profit
        #    (only if SL hasn't already been moved below entry by TP1 hit)
        original_sl_dist = (pos.original_stop_loss - pos.entry_price) if pos.original_stop_loss else 0
        sl_distance = original_sl_dist if original_sl_dist > 0 else (pos.stop_loss - pos.entry_price)
        if sl_distance > 0:
            unrealized_rr = (pos.entry_price - current_price) / sl_distance
            if (
                unrealized_rr >= self.breakeven_activation_rr
                and pos.stop_loss > pos.entry_price  # Not already at/below breakeven
            ):
                pos.stop_loss = pos.entry_price
                logger.info(
                    "sl_moved_to_breakeven_early",
                    symbol=symbol,
                    direction="short",
                    unrealized_rr=f"{unrealized_rr:.2f}",
                    new_sl=pos.entry_price,
                )

        # 4. Stale trade auto-close — close losing trades open too long
        #    Uses market-open hours only (overnight/weekend doesn't count for stocks/commodities)
        if self.max_hold_hours > 0 and pos.entry_time:
            now_utc = datetime.now(timezone.utc)
            hours_open = self._trading_hours_mgr.market_open_hours_between(
                pos.entry_time, now_utc, trading_hours,
            )
            if hours_open >= self.max_hold_hours:
                stale_rr = (pos.entry_price - current_price) / sl_distance if sl_distance > 0 else 0
                if stale_rr < self.stale_close_below_rr:
                    logger.info(
                        "stale_trade_closed",
                        symbol=symbol,
                        direction="short",
                        hours_open=f"{hours_open:.1f}",
                        unrealized_rr=f"{stale_rr:.2f}",
                        max_hold_hours=self.max_hold_hours,
                    )
                    return ExitSignal(symbol=symbol, reason="stale_close", price=current_price)

        # 5. ATR-based trailing stop for shorts — use original SL distance for R:R calculation
        if sl_distance > 0:
            unrealized_rr = (pos.entry_price - current_price) / sl_distance
            if unrealized_rr >= self.trailing_activation_rr:
                # ATR-dynamic trailing: low_water_mark + ATR * multiplier
                if atr > 0:
                    atr_trail = atr * self.trailing_atr_multiplier
                else:
                    atr_trail = pos.high_water_mark * TRAILING_STOP_PCT_FALLBACK

                # After TP1 but before TP2: use wider trail (1R = SL distance)
                # so the position has room to reach TP2.  After TP2: tighten.
                tier = getattr(pos, "current_tier", 0) or 0
                if tier >= 2:
                    trail_dist = atr_trail          # Tight trail for runner
                else:
                    trail_dist = max(atr_trail, sl_distance)  # Wide trail until TP2

                trailing_sl = pos.high_water_mark + trail_dist
                if trailing_sl < pos.stop_loss:
                    pos.stop_loss = trailing_sl
                if current_price >= trailing_sl:
                    return ExitSignal(symbol=symbol, reason="trailing_stop",
                                      price=current_price)

        return None
