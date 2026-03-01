from __future__ import annotations

import asyncio

import ccxt.async_support as ccxt
from fastapi import APIRouter, Request

from src.dashboard.auth import login_required
from src.data.repository import Repository
from src.exchange.models import OrderResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class _DashboardExchange:
    """Lightweight ccxt client that initializes lazily on the dashboard's event loop.

    The main engine's KrakenClient has an asyncio.Semaphore bound to the engine's
    event loop.  The dashboard runs in a separate thread (its own loop), so calling
    the engine's client causes 'attached to a different loop' errors.

    This wrapper creates its own ccxt instance on first use, ensuring all asyncio
    primitives bind to the dashboard's loop.
    """

    def __init__(self, api_key: str, api_secret: str) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._exchange: ccxt.kraken | None = None

    def _ensure_client(self) -> ccxt.kraken:
        if self._exchange is None:
            self._exchange = ccxt.kraken({
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
        return self._exchange

    async def fetch_ticker(self, symbol: str) -> dict:
        client = self._ensure_client()
        return await client.fetch_ticker(symbol)

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


def create_router(repo: Repository, exchange=None, api_key: str = "", api_secret: str = "") -> APIRouter:
    router = APIRouter()
    _dash_exchange: _DashboardExchange | None = None
    if api_key and api_secret:
        _dash_exchange = _DashboardExchange(api_key, api_secret)

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
        trades = await repo.get_open_trades()
        return {"trades": trades}

    @router.get("/stats")
    @login_required
    async def get_stats(request: Request):
        return await repo.get_trade_stats()

    @router.get("/signals/recent")
    @login_required
    async def get_recent_signals(request: Request):
        signals = await repo.get_recent_signals(limit=20)
        return {"signals": signals}

    @router.get("/unrealized-pnl")
    @login_required
    async def get_unrealized_pnl(request: Request):
        """Fetch live prices for open positions and compute unrealized P&L."""
        if not _dash_exchange:
            return {"positions": [], "total_unrealized": 0, "error": "No exchange configured"}

        trades = await repo.get_open_trades()
        if not trades:
            return {"positions": [], "total_unrealized": 0}

        positions = []
        total_unrealized = 0.0

        # Deduplicate ticker fetches — multiple trades may share a symbol
        unique_symbols = list({t["symbol"] for t in trades})

        async def fetch_ticker(symbol: str) -> float | None:
            try:
                ticker = await _dash_exchange.fetch_ticker(symbol)
                return float(ticker["last"])
            except Exception as e:
                logger.error("unrealized_pnl_fetch_failed", symbol=symbol, error=str(e))
                return None

        ticker_results = await asyncio.gather(*[fetch_ticker(s) for s in unique_symbols])
        price_map = dict(zip(unique_symbols, ticker_results))

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

            positions.append({
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "current_price": current_price,
                "quantity": quantity,
                "cost_usd": cost_usd,
                "unrealized_pnl": round(unrealized, 4),
                "unrealized_pct": round(unrealized / cost_usd * 100, 2) if cost_usd > 0 else 0,
                "trade_id": trade.get("id"),
            })
            total_unrealized += unrealized

        return {
            "positions": positions,
            "total_unrealized": round(total_unrealized, 4),
        }

    @router.post("/nuke")
    @login_required
    async def nuke_all_positions(request: Request):
        """Close ALL open positions at market price immediately."""
        if not _dash_exchange:
            return {"success": False, "error": "No exchange configured"}

        trades = await repo.get_open_trades()
        if not trades:
            return {"success": True, "closed": 0, "message": "No open positions to close"}

        closed = []
        errors = []

        for trade in trades:
            symbol = trade["symbol"]
            try:
                direction = trade.get("direction", "long")
                quantity = float(trade.get("entry_quantity", 0))

                # Opposite side to close
                close_side = "sell" if direction == "long" else "buy"

                # Get current price before executing
                ticker = await _dash_exchange.fetch_ticker(symbol)
                current_price = float(ticker["last"])

                # Execute market order
                result = await _dash_exchange.place_market_order(symbol, close_side, quantity)

                exit_price = result.avg_price or current_price
                entry_price = float(trade.get("entry_price", 0))

                if direction == "short":
                    pnl = (entry_price - exit_price) * quantity - result.fee
                else:
                    pnl = (exit_price - entry_price) * quantity - result.fee

                cost = float(trade.get("entry_cost_usd", 0))
                pnl_pct = (pnl / cost * 100) if cost > 0 else 0

                # Update DB
                await repo.close_trade(
                    trade_id=trade["id"],
                    exit_price=exit_price,
                    exit_quantity=result.filled_quantity or quantity,
                    exit_order_id=result.order_id,
                    exit_reason="nuke",
                    pnl_usd=round(pnl, 4),
                    pnl_percent=round(pnl_pct, 2),
                    fees_usd=result.fee,
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

    return router
