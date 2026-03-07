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
from src.strategy.split_test import SplitTestManager
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

    # Collect all engines' custom traders for multi-market propagation
    _all_engines = engines or {}

    def _get_all_custom_traders():
        """Get custom traders from ALL engines (crypto + stocks + commodities)."""
        traders = []
        for eng in _all_engines.values():
            if hasattr(eng, "custom_trader") and eng.custom_trader and eng.custom_trader.enabled:
                traders.append(eng.custom_trader)
        return traders

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
            })
            total_unrealized += unrealized

        return {
            "positions": positions,
            "total_unrealized": round(total_unrealized, 4),
        }

    @router.get("/trades/{trade_id}/detail")
    @login_required
    async def get_trade_detail(request: Request, trade_id: str):
        """Full trade record + partial exits for expandable detail row."""
        try:
            trades = await repo.get_trades_by_ids([trade_id])
            if not trades:
                return {"error": "Trade not found"}
            trade = trades[0]
            partial_exits = await repo.get_partial_exits(trade_id)
            return {
                "trade": trade,
                "partial_exits": partial_exits,
                "account_type": account_type,
            }
        except Exception as e:
            logger.error("trade_detail_failed", trade_id=trade_id, error=str(e))
            return {"error": str(e)}

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

                # For crypto: execute real market order. For stocks/commodities: paper close only.
                if _is_crypto_symbol(symbol) and _dash_exchange:
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

    # --- Flipped Trader API ---

    @router.get("/flipped/unrealized-pnl")
    @login_required
    async def get_flipped_unrealized_pnl(request: Request):
        """Fetch live prices for open flipped positions and compute unrealized P&L."""
        try:
            trades = await repo.get_open_trades(mode="flipped_paper")
        except Exception as e:
            logger.error("flipped_unrealized_pnl_db_failed", error=str(e))
            return {"positions": [], "total_unrealized": 0, "error": "DB error"}

        if not trades:
            return {"positions": [], "total_unrealized": 0}

        positions = []
        total_unrealized = 0.0
        unique_symbols = list({t["symbol"] for t in trades})

        # Multi-market price fetch: crypto via Binance, stocks/commodities via yfinance
        price_map = await _fetch_prices_multi_market(_dash_exchange, unique_symbols)

        for trade in trades:
            symbol = trade["symbol"]
            current_price = price_map.get(symbol)
            entry_price = float(trade.get("entry_price", 0))
            quantity = float(trade.get("entry_quantity", 0))
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
            current_notional = quantity * entry_price
            effective_cost = current_notional / leverage if leverage > 1 else current_notional

            positions.append({
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "current_price": current_price,
                "quantity": quantity,
                "cost_usd": cost_usd,
                "stop_loss": float(trade.get("stop_loss", 0)),
                "take_profit": float(trade.get("take_profit", 0)) if trade.get("take_profit") else None,
                "margin_used": round(effective_cost, 4),
                "unrealized_pnl": round(unrealized, 4),
                "unrealized_pct": round(unrealized / effective_cost * 100, 2) if effective_cost > 0 else 0,
                "leverage": leverage,
                "trade_id": trade.get("id"),
                "confluence_score": float(trade.get("confluence_score", 0) or 0),
                "entry_time": trade.get("entry_time"),
                "signal_reasons": trade.get("signal_reasons", []),
            })
            total_unrealized += unrealized

        return {
            "positions": positions,
            "total_unrealized": round(total_unrealized, 4),
        }

    @router.get("/flipped/trades/open")
    @login_required
    async def get_flipped_open_trades(request: Request):
        trades = await repo.get_open_trades(mode="flipped_paper")
        return {"trades": trades}

    @router.get("/flipped/stats")
    @login_required
    async def get_flipped_stats(request: Request):
        return await repo.get_trade_stats(mode="flipped_paper")

    @router.post("/reset/main")
    @admin_required
    async def reset_main_data(request: Request):
        """Reset main bot: delete trades, snapshots, reset engine state balance."""
        from src.config import Settings
        _cfg = Settings()
        try:
            deleted = await repo.reset_mode_data(mode=_cfg.trading_mode)

            # Reset engine state balance/pnl
            state = await repo.get_engine_state()
            if state:
                state["current_balance"] = _cfg.initial_balance
                state["peak_balance"] = _cfg.initial_balance
                state["daily_pnl_usd"] = 0
                state["total_pnl_usd"] = 0
                state["daily_start_bal"] = _cfg.initial_balance
                state["open_positions"] = {}
                state["cycle_count"] = 0
                state["last_scan_time"] = None
                await repo.upsert_engine_state(state)

            # Signal the engine to clear in-memory open positions
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
                    logger.info("reset_main_signaled_engine")
                except Exception as e:
                    logger.warning("reset_main_engine_signal_failed", error=str(e))

            logger.info("reset_main_complete", deleted=deleted)
            return {"success": True, "deleted": deleted, "balance_reset": _cfg.initial_balance}
        except Exception as e:
            logger.error("reset_main_failed", error=str(e))
            return {"success": False, "error": str(e)}

    @router.post("/reset/flipped")
    @admin_required
    async def reset_flipped_data(request: Request):
        """Reset flipped bot: delete trades, reset in-memory state, reset DB state."""
        from src.config import Settings
        _cfg = Settings()
        try:
            deleted = await repo.reset_mode_data(mode="flipped_paper")

            # Reset flipped state in engine_state config_overrides
            state = await repo.get_engine_state()
            if state:
                overrides = state.get("config_overrides", {})
                if not isinstance(overrides, dict):
                    overrides = {}
                overrides["flipped_trader"] = {
                    "balance": _cfg.flipped_initial_balance,
                    "peak_balance": _cfg.flipped_initial_balance,
                    "total_pnl": 0,
                    "daily_pnl": 0,
                    "positions": {},
                    "last_scan_time": None,
                    "cooldowns": {},
                }
                state["config_overrides"] = overrides
                await repo.upsert_engine_state(state)

            # Signal the running FlippedTrader to reset its in-memory state
            # This prevents the stale in-memory state from overwriting
            # the clean DB state on the next _save_state() call.
            if engine and hasattr(engine, "flipped_trader") and engine.flipped_trader:
                engine.flipped_trader.request_reset()
                logger.info("reset_flipped_signaled_engine")

            logger.info("reset_flipped_complete", deleted=deleted)
            return {"success": True, "deleted": deleted, "balance_reset": _cfg.flipped_initial_balance}
        except Exception as e:
            logger.error("reset_flipped_failed", error=str(e))
            return {"success": False, "error": str(e)}

    # --- Custom Configurable Bot API ---

    @router.get("/custom/unrealized-pnl")
    @login_required
    async def get_custom_unrealized_pnl(request: Request):
        """Fetch live prices for open custom bot positions and compute unrealized P&L."""
        try:
            trades = await repo.get_open_trades(mode="custom_paper")
        except Exception as e:
            logger.error("custom_unrealized_pnl_db_failed", error=str(e))
            return {"positions": [], "total_unrealized": 0, "error": "DB error"}

        if not trades:
            return {"positions": [], "total_unrealized": 0}

        positions = []
        total_unrealized = 0.0
        unique_symbols = list({t["symbol"] for t in trades})

        # Multi-market price fetch: crypto via Binance, stocks/commodities via yfinance
        price_map = await _fetch_prices_multi_market(_dash_exchange, unique_symbols)

        for trade in trades:
            symbol = trade["symbol"]
            current_price = price_map.get(symbol)
            entry_price = float(trade.get("entry_price", 0))
            quantity = float(trade.get("entry_quantity", 0))
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
            current_notional = quantity * entry_price
            effective_cost = current_notional / leverage if leverage > 1 else current_notional

            positions.append({
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "current_price": current_price,
                "quantity": quantity,
                "cost_usd": cost_usd,
                "stop_loss": float(trade.get("stop_loss", 0)),
                "take_profit": float(trade.get("take_profit", 0)) if trade.get("take_profit") else None,
                "margin_used": round(effective_cost, 4),
                "unrealized_pnl": round(unrealized, 4),
                "unrealized_pct": round(unrealized / effective_cost * 100, 2) if effective_cost > 0 else 0,
                "leverage": leverage,
                "trade_id": trade.get("id"),
                "confluence_score": float(trade.get("confluence_score", 0) or 0),
                "entry_time": trade.get("entry_time"),
                "signal_reasons": trade.get("signal_reasons", []),
            })
            total_unrealized += unrealized

        return {
            "positions": positions,
            "total_unrealized": round(total_unrealized, 4),
        }

    @router.get("/custom/trades/open")
    @login_required
    async def get_custom_open_trades(request: Request):
        trades = await repo.get_open_trades(mode="custom_paper")
        return {"trades": trades}

    @router.get("/custom/stats")
    @login_required
    async def get_custom_stats(request: Request):
        return await repo.get_trade_stats(mode="custom_paper")

    @router.post("/custom/settings")
    @admin_required
    async def update_custom_settings(request: Request):
        """Update custom bot direction toggle and margin slider at runtime."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        flip_direction = body.get("flip_direction")
        margin_pct = body.get("margin_pct")
        flip_mode = body.get("flip_mode")
        flip_threshold = body.get("flip_threshold")
        leverage = body.get("leverage")

        # Validate margin_pct range (5% to 40%)
        if margin_pct is not None:
            margin_pct = float(margin_pct)
            if margin_pct < 0.05 or margin_pct > 0.40:
                return JSONResponse({"error": "margin_pct must be between 0.05 and 0.40"}, status_code=400)

        if flip_direction is not None:
            flip_direction = bool(flip_direction)

        # Validate flip_mode
        if flip_mode is not None and flip_mode not in ("always_flip", "smart_flip", "normal"):
            return JSONResponse({"error": "flip_mode must be always_flip, smart_flip, or normal"}, status_code=400)

        # Validate flip_threshold
        if flip_threshold is not None:
            flip_threshold = float(flip_threshold)
            if flip_threshold < 0.0 or flip_threshold > 1.0:
                return JSONResponse({"error": "flip_threshold must be between 0.0 and 1.0"}, status_code=400)

        # Validate leverage range (1x to 100x)
        if leverage is not None:
            leverage = int(leverage)
            if leverage < 1 or leverage > 100:
                return JSONResponse({"error": "leverage must be between 1 and 100"}, status_code=400)

        # Update in-memory settings on ALL engines' custom traders (crypto + stocks + commodities)
        all_traders = _get_all_custom_traders()
        for ct in all_traders:
            ct.update_settings(
                flip_direction=flip_direction,
                margin_pct=margin_pct,
                flip_mode=flip_mode,
                flip_threshold=flip_threshold,
                leverage=leverage,
            )
        # Also update the primary engine's custom trader (even if not in all_traders yet)
        if engine and hasattr(engine, "custom_trader") and engine.custom_trader:
            engine.custom_trader.update_settings(
                flip_direction=flip_direction,
                margin_pct=margin_pct,
                flip_mode=flip_mode,
                flip_threshold=flip_threshold,
                leverage=leverage,
            )
            logger.info(
                "custom_settings_updated",
                flip_direction=engine.custom_trader.flip_direction,
                margin_pct=engine.custom_trader.max_position_pct,
                flip_mode=engine.custom_trader.flip_mode,
                flip_threshold=engine.custom_trader.flip_threshold,
                leverage=engine.custom_trader.leverage,
                engines_updated=len(all_traders),
            )

        # Also persist to DB (config_overrides) so it survives restarts
        try:
            state = await repo.get_engine_state()
            if state:
                overrides = state.get("config_overrides", {})
                if not isinstance(overrides, dict):
                    overrides = {}
                settings = overrides.get("custom_trader_settings", {})
                if flip_direction is not None:
                    settings["flip_direction"] = flip_direction
                if margin_pct is not None:
                    settings["margin_pct"] = margin_pct
                if flip_mode is not None:
                    settings["flip_mode"] = flip_mode
                if flip_threshold is not None:
                    settings["flip_threshold"] = flip_threshold
                if leverage is not None:
                    settings["leverage"] = leverage
                overrides["custom_trader_settings"] = settings
                state["config_overrides"] = overrides
                await repo.upsert_engine_state(state)
        except Exception as e:
            logger.warning("custom_settings_db_save_failed", error=str(e))

        # Return current settings
        ct = engine.custom_trader if engine and hasattr(engine, "custom_trader") else None
        current = {
            "flip_direction": ct.flip_direction if ct else True,
            "margin_pct": ct.max_position_pct if ct else 0.15,
            "flip_mode": ct.flip_mode if ct else "smart_flip",
            "flip_threshold": ct.flip_threshold if ct else 0.50,
            "leverage": ct.leverage if ct else 10,
        }
        return {"success": True, "settings": current}

    @router.post("/custom/trigger-scan")
    @admin_required
    async def trigger_custom_scan(request: Request):
        """Trigger an immediate scan cycle on the custom bot (all markets)."""
        all_traders = _get_all_custom_traders()
        if not all_traders and (not engine or not hasattr(engine, "custom_trader") or not engine.custom_trader):
            return JSONResponse({"success": False, "error": "Custom trader not available"}, status_code=503)

        # Trigger scan on ALL engines' custom traders
        for ct in all_traders:
            ct.request_scan()
        # Also trigger primary engine's custom trader
        if engine and hasattr(engine, "custom_trader") and engine.custom_trader:
            engine.custom_trader.request_scan()
        logger.info("custom_scan_triggered_via_api", engines_triggered=len(all_traders))
        return {"success": True, "message": f"Scan triggered on {max(len(all_traders), 1)} market(s) — will start within ~5 seconds"}

    @router.post("/flipped/trigger-scan")
    @admin_required
    async def trigger_flipped_scan(request: Request):
        """Trigger an immediate scan cycle on the flipped bot."""
        if not engine or not hasattr(engine, "flipped_trader") or not engine.flipped_trader:
            return JSONResponse({"success": False, "error": "Flipped trader not available"}, status_code=503)

        engine.flipped_trader.request_scan()
        logger.info("flipped_scan_triggered_via_api")
        return {"success": True, "message": "Scan triggered — will start within ~5 seconds"}

    # ── Main Bot Settings ────────────────────────────────────────────
    @router.post("/main/settings")
    @admin_required
    async def update_main_settings(request: Request):
        """Update main bot margin % and leverage at runtime (takes effect on next trade)."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        margin_pct = body.get("margin_pct")
        leverage = body.get("leverage")

        # Validate margin_pct range (5% to 40%)
        if margin_pct is not None:
            margin_pct = float(margin_pct)
            if margin_pct < 0.05 or margin_pct > 0.40:
                return JSONResponse({"error": "margin_pct must be between 0.05 and 0.40"}, status_code=400)

        # Validate leverage range (1x to 100x)
        if leverage is not None:
            leverage = int(leverage)
            if leverage < 1 or leverage > 100:
                return JSONResponse({"error": "leverage must be between 1 and 100"}, status_code=400)

        # Update in-memory settings on ALL engines' main bots
        for eng in _all_engines.values():
            eng.update_settings(margin_pct=margin_pct, leverage=leverage)
        # Also update the primary engine
        if engine:
            engine.update_settings(margin_pct=margin_pct, leverage=leverage)

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
                if margin_pct is not None:
                    main_settings["margin_pct"] = margin_pct
                if leverage is not None:
                    main_settings["leverage"] = leverage
                overrides["main_bot_settings"] = main_settings
                state["config_overrides"] = overrides
                await repo.upsert_engine_state(state)
        except Exception as e:
            logger.warning("main_settings_db_save_failed", error=str(e))

        current = {
            "margin_pct": engine.config.max_position_pct if engine else 0.05,
            "leverage": engine.config.leverage if engine else 10,
        }
        return {"success": True, "settings": current}

    @router.get("/main/settings")
    @login_required
    async def get_main_settings(request: Request):
        """Get current main bot settings."""
        if not engine:
            return {"margin_pct": 0.05, "leverage": 10}
        return {
            "margin_pct": engine.config.max_position_pct,
            "leverage": engine.config.leverage,
            "scanning_active": engine._scanning_active,
        }

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
        state = engine.main_entry_refiner.get_state()
        entries = []
        for sym, data in state.get("entries", {}).items():
            entries.append(data)
        return {"entries": entries, "total_queued": state.get("total_queued", 0), "enabled": True}

    @router.get("/refiner/custom")
    @login_required
    async def get_custom_refiner_queue(request: Request):
        """Get custom bot entry refiner queue (all markets)."""
        all_entries = []
        total = 0
        enabled = False

        # Check primary custom trader
        if engine and hasattr(engine, "custom_trader") and engine.custom_trader:
            refiner = engine.custom_trader.entry_refiner
            if refiner:
                enabled = True
                state = refiner.get_state()
                for sym, data in state.get("entries", {}).items():
                    data["market"] = "crypto"
                    all_entries.append(data)
                total += state.get("total_queued", 0)

        # Check multi-market custom traders (skip primary to avoid double-counting)
        for market_name, eng in _all_engines.items():
            if eng is engine:
                continue  # Already checked above as primary
            if hasattr(eng, "custom_trader") and eng.custom_trader:
                refiner = eng.custom_trader.entry_refiner
                if refiner:
                    enabled = True
                    state = refiner.get_state()
                    for sym, data in state.get("entries", {}).items():
                        data["market"] = market_name
                        all_entries.append(data)
                    total += state.get("total_queued", 0)

        return {"entries": all_entries, "total_queued": total, "enabled": enabled}

    # ── Custom Bot Start/Stop ──────────────────────────────────────────
    @router.post("/custom/begin")
    @admin_required
    async def begin_custom_scanning(request: Request):
        """Start the custom bot's scan loop (all markets)."""
        all_traders = _get_all_custom_traders()
        logger.info(
            "custom_begin_api_called",
            has_engine=bool(engine),
            has_custom_trader=bool(engine and hasattr(engine, "custom_trader") and engine.custom_trader),
            total_custom_traders=len(all_traders),
            role=request.session.get("role"),
        )

        if not all_traders and (not engine or not hasattr(engine, "custom_trader") or not engine.custom_trader):
            logger.warning("custom_begin_no_trader", engine_exists=bool(engine))
            return JSONResponse({"success": False, "error": "Custom trader not available"}, status_code=503)

        # Begin scanning on ALL engines' custom traders
        for ct in all_traders:
            ct.begin_scanning()
        # Also begin on primary engine's custom trader
        if engine and hasattr(engine, "custom_trader") and engine.custom_trader:
            engine.custom_trader.begin_scanning()

        # Persist scanning_active=True to DB (shared settings key)
        try:
            state = await repo.get_engine_state()
            if state:
                overrides = state.get("config_overrides", {})
                if not isinstance(overrides, dict):
                    overrides = {}
                # Persist in custom_trader_settings (shared across all markets)
                settings = overrides.get("custom_trader_settings", {})
                if not isinstance(settings, dict):
                    settings = {}
                settings["scanning_active"] = True
                overrides["custom_trader_settings"] = settings
                # Also persist under legacy custom_trader key for backward compat
                custom_data = overrides.get("custom_trader", {})
                if not isinstance(custom_data, dict):
                    custom_data = {}
                custom_data["scanning_active"] = True
                overrides["custom_trader"] = custom_data
                state["config_overrides"] = overrides
                await repo.upsert_engine_state(state)
        except Exception as e:
            logger.warning("begin_custom_persist_failed", error=str(e))

        logger.info(
            "custom_scanning_begun_via_api",
            engines_started=len(all_traders),
        )
        return {"success": True, "message": f"Custom bot scanning started on {max(len(all_traders), 1)} market(s)"}

    @router.post("/custom/stop")
    @admin_required
    async def stop_custom_scanning(request: Request):
        """Pause the custom bot's scan loop on all markets (keeps monitoring open positions)."""
        all_traders = _get_all_custom_traders()
        if not all_traders and (not engine or not hasattr(engine, "custom_trader") or not engine.custom_trader):
            return JSONResponse({"success": False, "error": "Custom trader not available"}, status_code=503)

        # Stop scanning on ALL engines' custom traders
        for ct in all_traders:
            ct.stop_scanning()
        if engine and hasattr(engine, "custom_trader") and engine.custom_trader:
            engine.custom_trader.stop_scanning()

        # Persist scanning_active=False to DB
        try:
            state = await repo.get_engine_state()
            if state:
                overrides = state.get("config_overrides", {})
                if not isinstance(overrides, dict):
                    overrides = {}
                # Shared settings
                settings = overrides.get("custom_trader_settings", {})
                if not isinstance(settings, dict):
                    settings = {}
                settings["scanning_active"] = False
                overrides["custom_trader_settings"] = settings
                # Legacy key
                custom_data = overrides.get("custom_trader", {})
                if not isinstance(custom_data, dict):
                    custom_data = {}
                custom_data["scanning_active"] = False
                overrides["custom_trader"] = custom_data
                state["config_overrides"] = overrides
                await repo.upsert_engine_state(state)
        except Exception as e:
            logger.warning("stop_custom_persist_failed", error=str(e))

        logger.info("custom_scanning_stopped_via_api", engines_stopped=len(all_traders))
        return {"success": True, "message": "Custom bot scanning paused"}

    @router.get("/custom/status")
    @login_required
    async def get_custom_status(request: Request):
        """Get custom bot running status (aggregated across all markets)."""
        all_traders = _get_all_custom_traders()
        if not all_traders and (not engine or not hasattr(engine, "custom_trader") or not engine.custom_trader):
            return {"scanning_active": False, "available": False}
        # Scanning is active if ANY engine's custom trader is scanning
        any_scanning = any(ct._scanning_active for ct in all_traders)
        if not any_scanning and engine and hasattr(engine, "custom_trader") and engine.custom_trader:
            any_scanning = engine.custom_trader._scanning_active
        return {
            "scanning_active": any_scanning,
            "available": True,
            "markets": len(all_traders),
        }

    @router.post("/reset/custom")
    @admin_required
    async def reset_custom_data(request: Request):
        """Reset custom bot: delete trades, reset in-memory state, reset DB state."""
        from src.config import Settings
        _cfg = Settings()
        try:
            deleted = await repo.reset_mode_data(mode="custom_paper")

            # Reset custom state in engine_state config_overrides
            state = await repo.get_engine_state()
            if state:
                overrides = state.get("config_overrides", {})
                if not isinstance(overrides, dict):
                    overrides = {}
                _reset_state = {
                    "balance": _cfg.custom_initial_balance,
                    "peak_balance": _cfg.custom_initial_balance,
                    "total_pnl": 0,
                    "daily_pnl": 0,
                    "positions": {},
                    "last_scan_time": None,
                    "cooldowns": {},
                }
                overrides["custom_trader"] = _reset_state
                # Also reset non-crypto custom trader state keys
                for key in list(overrides.keys()):
                    if key.startswith("custom_trader_") and key != "custom_trader_settings":
                        overrides[key] = dict(_reset_state)
                state["config_overrides"] = overrides
                await repo.upsert_engine_state(state)

            # Signal ALL engines' custom traders to reset
            all_traders = _get_all_custom_traders()
            for ct in all_traders:
                ct.request_reset()
            if engine and hasattr(engine, "custom_trader") and engine.custom_trader:
                engine.custom_trader.request_reset()
            logger.info("reset_custom_signaled_engine", engines_reset=len(all_traders))

            logger.info("reset_custom_complete", deleted=deleted)
            return {"success": True, "deleted": deleted, "balance_reset": _cfg.custom_initial_balance}
        except Exception as e:
            logger.error("reset_custom_failed", error=str(e))
            return {"success": False, "error": str(e)}

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

    @router.get("/split-test")
    @login_required
    async def get_split_test_results(request: Request):
        """Compare performance of control vs LLM-filtered trades."""
        try:
            closed_trades = await repo.get_trades(status="closed", per_page=500)
            # Filter to only trades that have a test_group field
            tagged_trades = [t for t in closed_trades if t.get("test_group")]
            if not tagged_trades:
                return {
                    "status": "no_data",
                    "message": "No split test trades found yet. Enable LLM_ENABLED=true to start the A/B test.",
                    "control": {"trade_count": 0},
                    "llm": {"trade_count": 0},
                }
            stats = SplitTestManager.compute_stats(tagged_trades)
            stats["total_tagged_trades"] = len(tagged_trades)
            return stats
        except Exception as e:
            logger.error("split_test_stats_failed", error=str(e))
            return {"error": str(e)}

    return router
