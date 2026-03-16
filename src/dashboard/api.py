from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import ccxt.async_support as ccxt
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.dashboard.auth import admin_required, login_required
from src.data.repository import Repository
from src.exchange.models import OrderResult
from src.exchange.protocol import get_symbol_category, CATEGORY_LABELS
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _is_crypto_symbol(symbol: str) -> bool:
    """Determine if a symbol is crypto (has /USDT, /BTC, etc.) vs stock/commodity."""
    return "/" in symbol


def _sync_yf_fetch_price(symbol: str) -> float | None:
    """Synchronous yfinance single-ticker price fetch. Called via run_in_executor."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        price = getattr(ticker.fast_info, "last_price", None)
        return float(price) if price else None
    except Exception:
        return None


async def _fetch_prices_multi_market(
    dash_exchange,
    symbols: list[str],
) -> dict[str, float | None]:
    """Fetch prices for a mixed list of crypto + stock/commodity symbols.

    Routes crypto symbols to Binance (via dash_exchange) and
    non-crypto symbols to yfinance.
    """
    crypto_syms = [s for s in symbols if _is_crypto_symbol(s)]
    other_syms = [s for s in symbols if not _is_crypto_symbol(s)]

    price_map: dict[str, float | None] = {}

    # Fetch crypto prices from Binance
    if crypto_syms and dash_exchange:
        try:
            tickers = await asyncio.wait_for(
                dash_exchange.fetch_tickers(crypto_syms), timeout=20,
            )
            for sym in crypto_syms:
                t = tickers.get(sym)
                price_map[sym] = float(t["last"]) if t and t.get("last") else None
        except Exception as e:
            logger.warning("crypto_batch_tickers_failed", error=str(e), fallback="individual")
            for sym in crypto_syms:
                try:
                    ticker = await asyncio.wait_for(
                        dash_exchange.fetch_ticker(sym), timeout=10,
                    )
                    price_map[sym] = float(ticker["last"]) if ticker and ticker.get("last") else None
                except Exception:
                    price_map[sym] = None

    # Fetch non-crypto prices from yfinance (run sync calls in executor)
    if other_syms:
        loop = asyncio.get_event_loop()
        yf_tasks = [loop.run_in_executor(None, _sync_yf_fetch_price, sym) for sym in other_syms]
        yf_results = await asyncio.gather(*yf_tasks, return_exceptions=True)
        for sym, result in zip(other_syms, yf_results):
            if isinstance(result, Exception):
                logger.warning("yf_price_fetch_failed", symbol=sym, error=str(result))
                price_map[sym] = None
            else:
                price_map[sym] = result

    return price_map


class _DashboardExchange:
    """Lightweight ccxt client that initializes lazily on the dashboard's event loop.

    The main engine's exchange client has an asyncio.Semaphore bound to the engine's
    event loop.  The dashboard runs in a separate thread (its own loop), so calling
    the engine's client causes 'attached to a different loop' errors.

    This wrapper creates its own ccxt instance on first use, ensuring all asyncio
    primitives bind to the dashboard's loop.
    """

    def __init__(self, exchange_name: str, api_key: str, api_secret: str, account_type: str = "spot") -> None:
        self._exchange_name = exchange_name
        self._api_key = api_key
        self._api_secret = api_secret
        self._account_type = account_type
        self._exchange = None

    def _ensure_client(self):
        if self._exchange is None:
            if self._account_type == "futures":
                default_type = "future"
            elif self._account_type == "margin":
                default_type = "margin"
            else:
                default_type = "spot"
            self._exchange = ccxt.binance({
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": default_type, "fetchCurrencies": False},
            })
        return self._exchange

    async def fetch_ticker(self, symbol: str) -> dict:
        client = self._ensure_client()
        return await client.fetch_ticker(symbol)

    async def fetch_tickers(self, symbols: list[str]) -> dict:
        """Batch-fetch tickers in a single API call (much less rate-limit pressure)."""
        client = self._ensure_client()
        return await client.fetch_tickers(symbols)

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> OrderResult:
        client = self._ensure_client()
        result = await client.create_order(
            symbol=symbol, type="market", side=side, amount=quantity,
        )
        fee_cost = 0.0
        if result.get("fee") and result["fee"].get("cost"):
            fee_cost = float(result["fee"]["cost"])
        return OrderResult(
            order_id=result["id"],
            symbol=symbol,
            side=side,
            filled_quantity=float(result.get("filled", 0)),
            avg_price=float(result.get("average", 0)),
            fee=fee_cost,
            status=result.get("status", "unknown"),
        )

    async def close(self) -> None:
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None


def create_router(repo: Repository, exchange=None, exchange_name: str = "binance", api_key: str = "", api_secret: str = "", account_type: str = "spot", engine=None, engines: dict | None = None) -> APIRouter:
    router = APIRouter()
    _dash_exchange: _DashboardExchange | None = None
    if api_key and api_secret:
        _dash_exchange = _DashboardExchange(exchange_name, api_key, api_secret, account_type=account_type)

    _all_engines = engines or {}

    @router.get("/portfolio")
    @login_required
    async def get_portfolio(request: Request):
        snapshot = await repo.get_latest_snapshot()
        history = await repo.get_snapshot_history(hours=168)
        return {
            "current": snapshot,
            "history": history,
        }

    @router.get("/trades/open")
    @login_required
    async def get_open_trades(request: Request):
        from src.config import Settings
        _cfg = Settings()
        trades = await repo.get_open_trades(mode=_cfg.trading_mode)
        return {"trades": trades}

    @router.get("/stats")
    @login_required
    async def get_stats(request: Request):
        mode = request.query_params.get("mode")
        return await repo.get_trade_stats(mode=mode)

    @router.get("/signals/recent")
    @login_required
    async def get_recent_signals(request: Request):
        signals = await repo.get_recent_signals(limit=20)
        return {"signals": signals}

    @router.get("/unrealized-pnl")
    @login_required
    async def get_unrealized_pnl(request: Request):
        """Fetch live prices for open positions and compute unrealized P&L."""
        from src.config import Settings
        _cfg = Settings()
        try:
            trades = await repo.get_open_trades(mode=_cfg.trading_mode)
        except Exception as e:
            logger.error("unrealized_pnl_db_failed", error=str(e))
            return {"positions": [], "total_unrealized": 0, "error": "DB error"}

        if not trades:
            return {"positions": [], "total_unrealized": 0}

        positions = []
        total_unrealized = 0.0

        # Deduplicate ticker fetches — multiple trades may share a symbol
        unique_symbols = list({t["symbol"] for t in trades})

        # Multi-market price fetch: crypto via Binance, stocks/commodities via yfinance
        price_map = await _fetch_prices_multi_market(_dash_exchange, unique_symbols)

        for trade in trades:
            symbol = trade["symbol"]
            current_price = price_map.get(symbol)
            entry_price = float(trade.get("entry_price", 0))
            raw_remaining = trade.get("remaining_quantity")
            raw_entry_qty = trade.get("entry_quantity", 0)
            quantity = float(raw_remaining or raw_entry_qty or 0)
            direction = trade.get("direction", "long")
            cost_usd = float(trade.get("entry_cost_usd", 0))

            if current_price is not None:
                if direction == "short":
                    unrealized = (entry_price - current_price) * quantity
                else:
                    unrealized = (current_price - entry_price) * quantity
            else:
                unrealized = 0

            leverage = int(trade.get("leverage", 1) or 1)
            # Use remaining quantity to compute current margin (handles partial exits)
            current_notional = quantity * entry_price
            if leverage > 1:
                effective_cost = current_notional / leverage
            else:
                effective_cost = current_notional

            positions.append({
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "current_price": current_price,
                "quantity": quantity,
                "cost_usd": cost_usd,
                "margin_used": round(effective_cost, 4),
                "unrealized_pnl": round(unrealized, 4),
                "unrealized_pct": round(unrealized / effective_cost * 100, 2) if effective_cost > 0 else 0,
                "leverage": leverage,
                "trade_id": trade.get("id"),
                "sector": CATEGORY_LABELS.get(get_symbol_category(symbol), "Other"),
                "current_tier": trade.get("current_tier", 0) or 0,
                "tp_tiers": trade.get("tp_tiers"),
            })
            total_unrealized += unrealized

        # Fetch partial exit P&L per tier for all open trades
        trade_ids = [p["trade_id"] for p in positions if p.get("trade_id")]
        tier_pnl_map: dict[str, dict[int, float]] = {}  # trade_id → {tier → pnl}
        if trade_ids:
            try:
                for tid in trade_ids:
                    partial_exits = await repo.get_partial_exits(tid)
                    if partial_exits:
                        tier_pnl_map[tid] = {}
                        for pe in partial_exits:
                            tier = int(pe.get("tier", 0))
                            pnl = float(pe.get("pnl_usd", 0))
                            tier_pnl_map[tid][tier] = tier_pnl_map[tid].get(tier, 0) + pnl
            except Exception as e:
                logger.debug("tier_pnl_fetch_failed", error=str(e))

        for pos in positions:
            pos["tier_pnl"] = tier_pnl_map.get(pos.get("trade_id"), {})

        return {
            "positions": positions,
            "total_unrealized": round(total_unrealized, 4),
        }

    @router.get("/trades/{trade_id}/detail")
    @login_required
    async def get_trade_detail(request: Request, trade_id: str):
        """Full trade record + partial exits + agent analysis for expandable detail row."""
        try:
            trades = await repo.get_trades_by_ids([trade_id])
            if not trades:
                return {"error": "Trade not found"}
            trade = trades[0]
            partial_exits = await repo.get_partial_exits(trade_id)

            # Fetch agent analysis from the signal that produced this trade
            agent_analysis = None
            try:
                signal_row = await repo.get_signal_by_trade_id(trade_id)
                if not signal_row:
                    # Fallback: for WAIT_PULLBACK entries where signal wasn't linked yet
                    symbol = trade.get("symbol", "")
                    if symbol:
                        signal_row = await repo.get_signal_by_symbol_recent(symbol)
                if signal_row:
                    components = signal_row.get("components") or {}
                    agent_analysis = components.get("agent_analysis")
            except Exception:
                pass  # Non-critical — just won't show agent section

            return {
                "trade": trade,
                "partial_exits": partial_exits,
                "account_type": account_type,
                "agent_analysis": agent_analysis,
            }
        except Exception as e:
            logger.error("trade_detail_failed", trade_id=trade_id, error=str(e))
            return {"error": str(e)}

    @router.post("/trades/{trade_id}/close")
    @admin_required
    async def close_single_trade(request: Request, trade_id: str):
        """Close a single open position at market price (manual close)."""
        if not _dash_exchange:
            return {"success": False, "error": "No exchange configured"}

        try:
            trades = await repo.get_trades_by_ids([trade_id])
            if not trades:
                return {"success": False, "error": "Trade not found"}
            trade = trades[0]

            if trade.get("status") != "open":
                return {"success": False, "error": "Trade is not open"}

            symbol = trade["symbol"]
            direction = trade.get("direction", "long")
            quantity = float(trade.get("remaining_quantity") or trade.get("entry_quantity", 0))
            close_side = "sell" if direction == "long" else "buy"

            # Get current price
            price_result = await _fetch_prices_multi_market(_dash_exchange, [symbol])
            current_price = price_result.get(symbol)
            if current_price is None:
                return {"success": False, "error": f"Could not fetch price for {symbol}"}

            # Execute close — only hit exchange in live mode
            from src.config import Settings
            _close_cfg = Settings()
            if _close_cfg.trading_mode == "live" and _is_crypto_symbol(symbol) and _dash_exchange:
                result = await _dash_exchange.place_market_order(symbol, close_side, quantity)
                exit_price = result.avg_price or current_price
                fee = result.fee
                order_id = result.order_id
                filled_qty = result.filled_quantity or quantity
            else:
                exit_price = current_price
                fee = 0.0
                order_id = f"paper_close_{symbol}"
                filled_qty = quantity

            entry_price = float(trade.get("entry_price", 0))
            if direction == "short":
                pnl = (entry_price - exit_price) * quantity - fee
            else:
                pnl = (exit_price - entry_price) * quantity - fee

            # Include partial exit PnLs
            total_trade_pnl = pnl
            total_fees = fee
            try:
                partial_exits = await repo.get_partial_exits(trade_id)
                total_trade_pnl += sum(float(pe.get("pnl_usd", 0)) for pe in partial_exits)
                total_fees += sum(float(pe.get("fees_usd", 0)) for pe in partial_exits)
            except Exception:
                pass

            cost = float(trade.get("entry_cost_usd", 0))
            pnl_pct = (total_trade_pnl / cost * 100) if cost > 0 else 0

            await repo.close_trade(
                trade_id=trade_id,
                exit_price=exit_price,
                exit_quantity=filled_qty,
                exit_order_id=order_id,
                exit_reason="manual_close",
                pnl_usd=round(total_trade_pnl, 4),
                pnl_percent=round(pnl_pct, 2),
                fees_usd=round(total_fees, 4),
            )

            # Remove from engine state
            try:
                engine_state = await repo.get_engine_state()
                if engine_state:
                    open_positions = engine_state.get("open_positions", {})
                    open_positions.pop(symbol, None)
                    engine_state["open_positions"] = open_positions
                    await repo.upsert_engine_state(engine_state)
            except Exception as e:
                logger.error("manual_close_state_update_failed", error=str(e))

            logger.info(
                "manual_close_position",
                symbol=symbol,
                direction=direction,
                exit_price=exit_price,
                pnl=round(total_trade_pnl, 4),
            )

            return {
                "success": True,
                "symbol": symbol,
                "exit_price": exit_price,
                "pnl_usd": round(total_trade_pnl, 4),
                "pnl_pct": round(pnl_pct, 2),
            }

        except Exception as e:
            logger.error("manual_close_failed", trade_id=trade_id, error=str(e))
            return {"success": False, "error": str(e)}

    @router.post("/nuke")
    @admin_required
    async def nuke_all_positions(request: Request):
        """Close ALL open main-bot positions at market price immediately."""
        if not _dash_exchange:
            return {"success": False, "error": "No exchange configured"}

        from src.config import Settings
        _cfg = Settings()
        trades = await repo.get_open_trades(mode=_cfg.trading_mode)
        if not trades:
            return {"success": True, "closed": 0, "message": "No open positions to close"}

        closed = []
        errors = []

        for trade in trades:
            symbol = trade["symbol"]
            try:
                direction = trade.get("direction", "long")
                quantity = float(trade.get("remaining_quantity") or trade.get("entry_quantity", 0))

                # Opposite side to close
                close_side = "sell" if direction == "long" else "buy"

                # Get current price (multi-market aware)
                price_result = await _fetch_prices_multi_market(_dash_exchange, [symbol])
                current_price = price_result.get(symbol)
                if current_price is None:
                    errors.append({"symbol": symbol, "error": "Could not fetch current price"})
                    continue

                # Only hit exchange in live mode
                if _cfg.trading_mode == "live" and _is_crypto_symbol(symbol) and _dash_exchange:
                    result = await _dash_exchange.place_market_order(symbol, close_side, quantity)
                    exit_price = result.avg_price or current_price
                    fee = result.fee
                    order_id = result.order_id
                    filled_qty = result.filled_quantity or quantity
                else:
                    # Paper close — just use current price
                    exit_price = current_price
                    fee = 0.0
                    order_id = f"paper_nuke_{symbol}"
                    filled_qty = quantity

                entry_price = float(trade.get("entry_price", 0))

                if direction == "short":
                    pnl = (entry_price - exit_price) * quantity - fee
                else:
                    pnl = (exit_price - entry_price) * quantity - fee

                # Accumulate partial exit PnLs for total
                total_trade_pnl = pnl
                total_fees = fee
                try:
                    partial_exits = await repo.get_partial_exits(trade["id"])
                    total_trade_pnl += sum(float(pe.get("pnl_usd", 0)) for pe in partial_exits)
                    total_fees += sum(float(pe.get("fees_usd", 0)) for pe in partial_exits)
                except Exception:
                    pass

                cost = float(trade.get("entry_cost_usd", 0))
                pnl_pct = (total_trade_pnl / cost * 100) if cost > 0 else 0

                # Update DB
                await repo.close_trade(
                    trade_id=trade["id"],
                    exit_price=exit_price,
                    exit_quantity=filled_qty,
                    exit_order_id=order_id,
                    exit_reason="nuke",
                    pnl_usd=round(total_trade_pnl, 4),
                    pnl_percent=round(pnl_pct, 2),
                    fees_usd=round(total_fees, 4),
                )

                closed.append({
                    "symbol": symbol,
                    "direction": direction,
                    "exit_price": exit_price,
                    "pnl_usd": round(pnl, 4),
                })

                logger.info(
                    "nuke_closed_position",
                    symbol=symbol,
                    direction=direction,
                    exit_price=exit_price,
                    pnl=round(pnl, 4),
                )

            except Exception as e:
                logger.error("nuke_close_failed", symbol=symbol, error=str(e))
                errors.append({"symbol": symbol, "error": str(e)})

        # Update engine_state to clear open_positions
        try:
            engine_state = await repo.get_engine_state()
            if engine_state:
                # Remove nuked positions from engine state
                open_positions = engine_state.get("open_positions", {})
                for item in closed:
                    open_positions.pop(item["symbol"], None)
                engine_state["open_positions"] = open_positions
                await repo.upsert_engine_state(engine_state)
        except Exception as e:
            logger.error("nuke_state_update_failed", error=str(e))

        total_pnl = sum(c["pnl_usd"] for c in closed)
        return {
            "success": len(errors) == 0,
            "closed": len(closed),
            "errors": errors,
            "total_pnl": round(total_pnl, 4),
            "details": closed,
        }

    @router.post("/reset/main")
    @admin_required
    async def reset_main_data(request: Request):
        """Nuclear reset: wipe ALL tables and start completely fresh."""
        from src.config import Settings
        _cfg = Settings()
        try:
            # Nuclear wipe — all tables
            await repo.wipe_all_data()

            # Create fresh engine state
            state = {
                "id": 1,
                "mode": _cfg.trading_mode,
                "status": "running",
                "current_balance": _cfg.initial_balance,
                "peak_balance": _cfg.initial_balance,
                "daily_start_bal": _cfg.initial_balance,
                "daily_pnl_usd": 0,
                "total_pnl_usd": 0,
                "open_positions": {},
                "cycle_count": 0,
                "last_scan_time": None,
                "config_overrides": {},
            }
            await repo.upsert_engine_state(state)

            # Signal the engine to clear in-memory state
            if engine:
                try:
                    if hasattr(engine, "portfolio"):
                        engine.portfolio.open_positions.clear()
                        engine.portfolio.current_balance = _cfg.initial_balance
                        engine.portfolio.peak_balance = _cfg.initial_balance
                        engine.portfolio.daily_pnl = 0
                        engine.portfolio.total_pnl = 0
                        engine.portfolio.daily_start_balance = _cfg.initial_balance
                    if hasattr(engine, "state"):
                        engine.state.open_positions = {}
                        engine.state.current_balance = _cfg.initial_balance
                        engine.state.peak_balance = _cfg.initial_balance
                        engine.state.daily_pnl = 0
                        engine.state.total_pnl = 0
                    logger.info("reset_main_signaled_engine")
                except Exception as e:
                    logger.warning("reset_main_engine_signal_failed", error=str(e))

            logger.info("nuclear_reset_complete")
            return {"success": True, "deleted": {"trades": "all", "snapshots": "all"}, "balance_reset": _cfg.initial_balance}
        except Exception as e:
            logger.error("reset_main_failed", error=str(e))
            return {"success": False, "error": str(e)}

    # ── Main Bot Settings ────────────────────────────────────────────
    # Validation rules: (type, min, max)
    _SETTINGS_VALIDATION: dict = {
        "margin_pct":                  (float, 0.05, 0.40),
        "leverage":                    (int,   1,    100),
        "max_concurrent":              (int,   0,    50),
        "max_risk_pct":                (float, 0.01, 0.10),
        "max_daily_drawdown":          (float, 0.05, 0.30),
        "entry_threshold":             (float, 30.0, 90.0),
        "min_rr_ratio":                (float, 1.0,  5.0),
        "max_hold_hours":              (float, 1.0,  24.0),
        "circuit_breaker_pct":         (float, 0.05, 0.30),
        "max_sl_pct":                  (float, 0.02, 0.10),
        "cooldown_hours":              (float, 0.0,  24.0),
        "max_exposure_pct":            (float, 0.5,  3.0),
        "monday_manipulation_penalty": (float, 0.0,  30.0),
        "monday_manipulation_hours":   (float, 2.0,  16.0),
        "midweek_reversal_bonus":      (float, 0.0,  25.0),
        "midweek_reversal_delay_hours":(float, 0.0,  12.0),
        "weekly_cycle_enabled":        (bool,  None, None),
    }

    @router.post("/main/settings")
    @admin_required
    async def update_main_settings(request: Request):
        """Update trading settings at runtime (takes effect on next trade/cycle)."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        # Parse and validate all provided settings
        parsed: dict = {}
        for key, (typ, lo, hi) in _SETTINGS_VALIDATION.items():
            if key not in body:
                continue
            raw = body[key]
            if typ is bool:
                parsed[key] = bool(raw)
                continue
            val = typ(raw)
            if lo is not None and val < lo:
                return JSONResponse({"error": f"{key} must be >= {lo}"}, status_code=400)
            if hi is not None and val > hi:
                return JSONResponse({"error": f"{key} must be <= {hi}"}, status_code=400)
            parsed[key] = val

        if not parsed:
            return JSONResponse({"error": "No valid settings provided"}, status_code=400)

        # Update in-memory settings on ALL engines
        for eng in _all_engines.values():
            eng.update_settings(**parsed)
        if engine:
            engine.update_settings(**parsed)

        # Persist to DB
        try:
            state = await repo.get_engine_state()
            if state:
                overrides = state.get("config_overrides", {}) or {}
                if not isinstance(overrides, dict):
                    overrides = {}
                main_settings = overrides.get("main_bot_settings", {}) or {}
                if not isinstance(main_settings, dict):
                    main_settings = {}
                main_settings.update(parsed)
                overrides["main_bot_settings"] = main_settings
                state["config_overrides"] = overrides
                await repo.upsert_engine_state(state)
        except Exception as e:
            logger.warning("main_settings_db_save_failed", error=str(e))

        return {"success": True, "settings": _read_all_settings()}

    def _read_all_settings() -> dict:
        """Read current values of all editable settings from engine config."""
        if not engine:
            return {}
        c = engine.config
        return {
            "margin_pct": c.max_position_pct,
            "leverage": c.leverage,
            "scanning_active": engine._scanning_active,
            "max_concurrent": c.max_concurrent,
            "max_risk_pct": c.max_risk_pct,
            "max_daily_drawdown": c.max_daily_drawdown,
            "entry_threshold": c.entry_threshold,
            "min_rr_ratio": c.min_rr_ratio,
            "max_hold_hours": c.max_hold_hours,
            "circuit_breaker_pct": c.circuit_breaker_pct,
            "max_sl_pct": c.max_sl_pct,
            "cooldown_hours": c.cooldown_hours,
            "max_exposure_pct": c.max_exposure_pct,
            "monday_manipulation_penalty": c.monday_manipulation_penalty,
            "monday_manipulation_hours": c.monday_manipulation_hours,
            "midweek_reversal_bonus": c.midweek_reversal_bonus,
            "midweek_reversal_delay_hours": getattr(c, "midweek_reversal_delay_hours", 4.0),
            "weekly_cycle_enabled": c.weekly_cycle_enabled,
        }

    @router.get("/main/settings")
    @login_required
    async def get_main_settings(request: Request):
        """Get current trading settings."""
        return _read_all_settings()

    # ── Main Bot Start/Stop ───────────────────────────────────────────
    @router.post("/main/begin")
    @admin_required
    async def begin_main_scanning(request: Request):
        """Start the main bot's scan loop."""
        if not engine:
            return JSONResponse({"success": False, "error": "Engine not available"}, status_code=503)
        engine.begin_scanning()
        # Persist to DB
        try:
            state = await repo.get_engine_state()
            if state:
                overrides = state.get("config_overrides", {}) or {}
                if not isinstance(overrides, dict):
                    overrides = {}
                main_settings = overrides.get("main_bot_settings", {})
                if not isinstance(main_settings, dict):
                    main_settings = {}
                main_settings["scanning_active"] = True
                overrides["main_bot_settings"] = main_settings
                state["config_overrides"] = overrides
                await repo.upsert_engine_state(state)
        except Exception as e:
            logger.warning("begin_main_persist_failed", error=str(e))
        return {"success": True, "message": "Main bot scanning started"}

    @router.post("/main/stop")
    @admin_required
    async def stop_main_scanning(request: Request):
        """Pause the main bot's scan loop (keeps monitoring open positions)."""
        if not engine:
            return JSONResponse({"success": False, "error": "Engine not available"}, status_code=503)
        engine.stop_scanning()
        # Persist to DB
        try:
            state = await repo.get_engine_state()
            if state:
                overrides = state.get("config_overrides", {}) or {}
                if not isinstance(overrides, dict):
                    overrides = {}
                main_settings = overrides.get("main_bot_settings", {})
                if not isinstance(main_settings, dict):
                    main_settings = {}
                main_settings["scanning_active"] = False
                overrides["main_bot_settings"] = main_settings
                state["config_overrides"] = overrides
                await repo.upsert_engine_state(state)
        except Exception as e:
            logger.warning("stop_main_persist_failed", error=str(e))
        return {"success": True, "message": "Main bot scanning paused (still monitoring positions)"}

    @router.get("/main/status")
    @login_required
    async def get_main_status(request: Request):
        """Get main bot running status."""
        if not engine:
            return {"scanning_active": False, "available": False}
        return {"scanning_active": engine._scanning_active, "available": True}

    # ── Entry Refiner Queue ───────────────────────────────────────────
    @router.get("/refiner/main")
    @login_required
    async def get_main_refiner_queue(request: Request):
        """Get main bot entry refiner queue."""
        if not engine or not engine.main_entry_refiner:
            return {"entries": [], "total_queued": 0, "enabled": False}
        try:
            state = engine.main_entry_refiner.get_state()
            entries = []
            for sym, data in state.get("entries", {}).items():
                entries.append(data)
            return {
                "entries": entries,
                "total_queued": state.get("total_queued", 0),
                "enabled": True,
                "last_check_at": state.get("last_check_at"),
                "total_checks": state.get("total_checks", 0),
                "total_confirmed": state.get("total_confirmed", 0),
                "total_expired": state.get("total_expired", 0),
                "queue_size": len(engine.main_entry_refiner.queue),
            }
        except Exception as e:
            import traceback
            return {
                "entries": [],
                "total_queued": 0,
                "enabled": True,
                "error": str(e),
                "traceback": traceback.format_exc()[:500],
                "queue_size": len(engine.main_entry_refiner.queue),
            }

    # ── Consensus Monitor Queue ────────────────────────────────────────
    @router.get("/consensus/main")
    @login_required
    async def get_main_consensus_queue(request: Request):
        """Get main bot consensus monitor queue."""
        if not engine or not engine.consensus_monitor:
            return {"entries": [], "total_queued": 0, "enabled": False}
        state = engine.consensus_monitor.get_state()
        entries = []
        for sym, data in state.get("entries", {}).items():
            entries.append(data)
        return {"entries": entries, "total_queued": state.get("total_queued", 0), "enabled": True}

    @router.get("/backtests")
    @login_required
    async def list_backtests(request: Request):
        """List available backtest result files."""
        results_dir = Path("backtest_results")
        if not results_dir.exists():
            return {"backtests": []}

        files = []
        for f in sorted(results_dir.glob("*.json"), reverse=True):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                files.append({
                    "filename": f.name,
                    "type": data.get("type", "unknown"),
                    "run_time": data.get("run_time"),
                    "period": data.get("period"),
                    "summary": data.get("summary"),
                })
            except Exception:
                continue
        return {"backtests": files}

    @router.get("/backtests/{filename}")
    @login_required
    async def get_backtest(request: Request, filename: str):
        """Get a specific backtest result."""
        # Sanitize filename to prevent path traversal
        safe_name = os.path.basename(filename)
        filepath = Path("backtest_results") / safe_name
        if not filepath.exists() or not filepath.suffix == ".json":
            return JSONResponse({"error": "Not found"}, status_code=404)
        try:
            with open(filepath) as f:
                return json.load(f)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.get("/agent-stats")
    @login_required
    async def get_agent_stats(request: Request):
        """Get AI entry agent usage statistics and performance."""
        try:
            result: dict = {"status": "ok"}

            # Agent 1 usage stats (strategic analyst)
            if engine and engine.agent_analyst:
                result["agent"] = engine.agent_analyst.get_usage_stats()
                result["agent"]["enabled"] = True
                result["agent"]["available_models"] = engine.agent_analyst.available_models
            else:
                result["agent"] = {"enabled": False, "total_requests": 0}

            # Agent 2 usage stats (refiner monitor — tactical entry timing)
            if engine and getattr(engine, "refiner_agent", None):
                result["refiner_agent"] = engine.refiner_agent.get_usage_stats()
                result["refiner_agent"]["enabled"] = True
            else:
                result["refiner_agent"] = {"enabled": False, "total_requests": 0}

            # Agent-tagged trade performance
            closed_trades = await repo.get_trades(status="closed", per_page=500)
            agent_trades = [t for t in closed_trades if t.get("test_group") == "agent"]
            if agent_trades:
                wins = sum(1 for t in agent_trades if (t.get("pnl_usd") or 0) > 0)
                pnls = [float(t.get("pnl_usd") or 0) for t in agent_trades]
                result["performance"] = {
                    "trade_count": len(agent_trades),
                    "win_rate": round(wins / len(agent_trades) * 100, 1),
                    "total_pnl_usd": round(sum(pnls), 2),
                    "avg_pnl_usd": round(sum(pnls) / len(agent_trades), 2),
                    "best_trade_usd": round(max(pnls), 2),
                    "worst_trade_usd": round(min(pnls), 2),
                }
            else:
                result["performance"] = {"trade_count": 0, "message": "No agent trades yet"}

            return result
        except Exception as e:
            logger.error("agent_stats_failed", error=str(e))
            return {"error": str(e)}

    @router.post("/agent-model")
    @admin_required
    async def set_agent_model(request: Request):
        """Switch agent model at runtime and persist to DB.

        Body: {"agent": "agent1"|"agent2", "model": "gemini-3-pro-preview"|"gemini-3-flash-preview"}
        Backwards-compatible: {"model": "..."} still switches Agent 1.
        """
        try:
            body = await request.json()
            model = body.get("model", "").strip()
            agent_key = body.get("agent", "agent1").strip()
            if not model:
                return {"error": "model is required"}
            if not engine:
                return {"error": "engine not running"}

            # Switch the correct agent
            if agent_key == "agent3":
                if not engine.position_agent:
                    return {"error": "Agent 3 not available (no API key or disabled)"}
                engine.position_agent._model = model
                active = model
                db_field = "agent3_model"
            elif agent_key == "agent2":
                if not engine.refiner_agent:
                    return {"error": "Agent 2 not available (no API key)"}
                active = engine.refiner_agent.set_model(model)
                db_field = "agent2_model"
            else:
                if not engine.agent_analyst:
                    return {"error": "Agent 1 not available (no API key)"}
                active = engine.agent_analyst.set_model(model)
                db_field = "agent1_model"

            # Persist to DB inside config_overrides JSONB (avoids column issues)
            try:
                state = await repo.get_engine_state()
                if state:
                    overrides = state.get("config_overrides") or {}
                    if not isinstance(overrides, dict):
                        overrides = {}
                    agent_models = overrides.get("agent_models") or {}
                    agent_models[agent_key] = active
                    overrides["agent_models"] = agent_models
                    state["config_overrides"] = overrides
                    await repo.upsert_engine_state(state)
            except Exception as e:
                logger.warning("agent_model_db_save_failed", error=str(e))

            # Return both agents' current models
            result = {
                "status": "ok",
                "agent": agent_key,
                "model": active,
                "agent1_model": engine.agent_analyst._model if engine.agent_analyst else None,
                "agent2_model": engine.refiner_agent._model if engine.refiner_agent else None,
                "available_models": (
                    engine.agent_analyst.available_models
                    if engine.agent_analyst
                    else ["gemini-3-pro-preview", "gemini-3-flash-preview"]
                ),
            }
            return result
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            logger.error("agent_model_switch_failed", error=str(e))
            return {"error": str(e)}

    # ── Analytics ──────────────────────────────────────────────────────
    @router.get("/analytics/trades")
    @login_required
    async def get_analytics_trades(request: Request):
        """Closed trades with full signal journey for the analytics page."""
        from src.config import Settings
        _cfg = Settings()
        page = int(request.query_params.get("page", 1))
        per_page = 25
        offset = (page - 1) * per_page
        rows = await repo.get_closed_trades_with_signals(
            mode=_cfg.trading_mode, limit=per_page, offset=offset,
        )

        results = []
        for row in rows:
            trade = row.get("trade", {})
            sig = row.get("signal", {})
            components = sig.get("components") or {}

            results.append({
                "trade": {
                    "id": trade.get("id"),
                    "symbol": trade.get("symbol"),
                    "direction": trade.get("direction"),
                    "entry_price": trade.get("entry_price"),
                    "exit_price": trade.get("exit_price"),
                    "pnl_usd": trade.get("pnl_usd"),
                    "pnl_percent": trade.get("pnl_percent"),
                    "exit_reason": trade.get("exit_reason"),
                    "confluence_score": trade.get("confluence_score"),
                    "signal_reasons": trade.get("signal_reasons"),
                    "leverage": trade.get("leverage"),
                    "entry_time": trade.get("entry_time"),
                    "exit_time": trade.get("exit_time"),
                    "stop_loss": trade.get("stop_loss"),
                    "take_profit": trade.get("take_profit"),
                    "entry_cost_usd": trade.get("entry_cost_usd"),
                },
                "agent": components.get("agent_analysis"),
                "refiner": components.get("refiner_journey"),
                "context": components.get("agent_context"),
                "signal_type": sig.get("signal_type") or components.get("signal_type"),
            })

        return {"trades": results, "page": page, "per_page": per_page}

    @router.get("/analytics/summary")
    @login_required
    async def get_analytics_summary(request: Request):
        """Aggregated stats for analytics charts — computed in Python from closed trades."""
        from src.config import Settings
        _cfg = Settings()
        rows = await repo.get_closed_trades_with_signals(
            mode=_cfg.trading_mode, limit=500, offset=0,
        )

        if not rows:
            return {"total": 0}

        # ── Win rate by confidence bucket ──
        confidence_buckets = {"60-70": [0, 0], "70-80": [0, 0], "80-90": [0, 0], "90+": [0, 0]}
        # ── Win rate by risk level ──
        risk_buckets: dict[str, list[int]] = {}
        # ── Win rate by signal type ──
        type_buckets: dict[str, list[int]] = {}
        # ── Win rate by direction ──
        dir_buckets: dict[str, list[int]] = {}
        # ── Win rate by exit reason ──
        exit_buckets: dict[str, list[int]] = {}
        # ── Refiner stats ──
        refiner_outcomes: dict[str, int] = {"confirmed": 0, "expired": 0, "invalidated": 0}
        improvement_pcts: list[float] = []
        refiner_durations: list[float] = []
        # ── Agent accuracy ──
        entry_distances: list[float] = []

        for row in rows:
            trade = row.get("trade", {})
            components = (row.get("signal", {}).get("components") or {})
            agent = components.get("agent_analysis") or {}
            refiner = components.get("refiner_journey") or {}
            pnl = float(trade.get("pnl_usd", 0) or 0)
            is_win = 1 if pnl >= 0 else 0

            # Confidence buckets
            conf = agent.get("confidence")
            if conf is not None:
                conf = float(conf)
                if conf >= 90:
                    k = "90+"
                elif conf >= 80:
                    k = "80-90"
                elif conf >= 70:
                    k = "70-80"
                else:
                    k = "60-70"
                confidence_buckets[k][0] += is_win
                confidence_buckets[k][1] += 1

            # Risk level
            risk = agent.get("risk", "unknown")
            if risk:
                risk_buckets.setdefault(risk, [0, 0])
                risk_buckets[risk][0] += is_win
                risk_buckets[risk][1] += 1

            # Signal type
            sig_type = row.get("signal_type") or "unknown"
            type_buckets.setdefault(sig_type, [0, 0])
            type_buckets[sig_type][0] += is_win
            type_buckets[sig_type][1] += 1

            # Direction
            direction = trade.get("direction", "unknown")
            dir_buckets.setdefault(direction, [0, 0])
            dir_buckets[direction][0] += is_win
            dir_buckets[direction][1] += 1

            # Exit reason
            exit_reason = trade.get("exit_reason", "unknown") or "unknown"
            exit_buckets.setdefault(exit_reason, [0, 0])
            exit_buckets[exit_reason][0] += is_win
            exit_buckets[exit_reason][1] += 1

            # Refiner journey
            outcome = refiner.get("outcome")
            if outcome and outcome in refiner_outcomes:
                refiner_outcomes[outcome] += 1
            if refiner.get("improvement_pct") is not None:
                improvement_pcts.append(float(refiner["improvement_pct"]))
            if refiner.get("duration_seconds") is not None:
                refiner_durations.append(float(refiner["duration_seconds"]))

            # Agent accuracy: distance between suggested entry and actual entry
            suggested = agent.get("suggested_entry")
            actual = trade.get("entry_price")
            if suggested and actual:
                try:
                    dist = abs(float(suggested) - float(actual)) / float(actual) * 100
                    entry_distances.append(dist)
                except (ValueError, ZeroDivisionError):
                    pass

        def _wr(bucket: list[int]) -> float:
            return round(bucket[0] / bucket[1] * 100, 1) if bucket[1] > 0 else 0

        return {
            "total": len(rows),
            "confidence": {k: {"wins": v[0], "total": v[1], "win_rate": _wr(v)} for k, v in confidence_buckets.items()},
            "risk": {k: {"wins": v[0], "total": v[1], "win_rate": _wr(v)} for k, v in risk_buckets.items()},
            "signal_type": {k: {"wins": v[0], "total": v[1], "win_rate": _wr(v)} for k, v in type_buckets.items()},
            "direction": {k: {"wins": v[0], "total": v[1], "win_rate": _wr(v)} for k, v in dir_buckets.items()},
            "exit_reason": {k: {"wins": v[0], "total": v[1], "win_rate": _wr(v)} for k, v in exit_buckets.items()},
            "refiner": {
                "outcomes": refiner_outcomes,
                "avg_improvement_pct": round(sum(improvement_pcts) / len(improvement_pcts), 2) if improvement_pcts else 0,
                "avg_duration_minutes": round(sum(refiner_durations) / len(refiner_durations) / 60, 1) if refiner_durations else 0,
            },
            "agent_accuracy": {
                "avg_entry_distance_pct": round(sum(entry_distances) / len(entry_distances), 3) if entry_distances else 0,
                "samples": len(entry_distances),
            },
        }

    return router
