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


def create_router(repo: Repository, exchange=None, exchange_name: str = "binance", api_key: str = "", api_secret: str = "", account_type: str = "spot") -> APIRouter:
    router = APIRouter()
    _dash_exchange: _DashboardExchange | None = None
    if api_key and api_secret:
        _dash_exchange = _DashboardExchange(exchange_name, api_key, api_secret, account_type=account_type)

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
        if not _dash_exchange:
            return {"positions": [], "total_unrealized": 0, "error": "No exchange configured"}

        try:
            trades = await repo.get_open_trades()
        except Exception as e:
            logger.error("unrealized_pnl_db_failed", error=str(e))
            return {"positions": [], "total_unrealized": 0, "error": "DB error"}

        if not trades:
            return {"positions": [], "total_unrealized": 0}

        positions = []
        total_unrealized = 0.0

        # Deduplicate ticker fetches — multiple trades may share a symbol
        unique_symbols = list({t["symbol"] for t in trades})

        # Use batch fetchTickers (single API call) to avoid rate-limit pressure
        price_map: dict[str, float | None] = {}
        try:
            tickers = await asyncio.wait_for(
                _dash_exchange.fetch_tickers(unique_symbols), timeout=20,
            )
            for sym in unique_symbols:
                t = tickers.get(sym)
                price_map[sym] = float(t["last"]) if t and t.get("last") else None
        except Exception as e:
            logger.warning("batch_fetch_tickers_failed", error=str(e), fallback="individual")
            # Fallback: fetch individually (slower but more resilient to partial failures)
            async def _fetch_one(symbol: str) -> float | None:
                try:
                    ticker = await _dash_exchange.fetch_ticker(symbol)
                    return float(ticker["last"])
                except Exception:
                    return None
            results = await asyncio.gather(*[_fetch_one(s) for s in unique_symbols])
            price_map = dict(zip(unique_symbols, results))

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
                quantity = float(trade.get("remaining_quantity") or trade.get("entry_quantity", 0))

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

                # Accumulate partial exit PnLs for total
                total_trade_pnl = pnl
                total_fees = result.fee
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
                    exit_quantity=result.filled_quantity or quantity,
                    exit_order_id=result.order_id,
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
        if not _dash_exchange:
            return {"positions": [], "total_unrealized": 0, "error": "No exchange configured"}

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

        price_map: dict[str, float | None] = {}
        try:
            tickers = await asyncio.wait_for(
                _dash_exchange.fetch_tickers(unique_symbols), timeout=20,
            )
            for sym in unique_symbols:
                t = tickers.get(sym)
                price_map[sym] = float(t["last"]) if t and t.get("last") else None
        except Exception as e:
            logger.warning("flipped_batch_tickers_failed", error=str(e))
            async def _fetch_one(symbol: str) -> float | None:
                try:
                    ticker = await _dash_exchange.fetch_ticker(symbol)
                    return float(ticker["last"])
                except Exception:
                    return None
            results = await asyncio.gather(*[_fetch_one(s) for s in unique_symbols])
            price_map = dict(zip(unique_symbols, results))

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
