from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from src.data.db import Database
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _exec(query):
    """Execute a Supabase query synchronously (for use with asyncio.to_thread)."""
    return query.execute()


class Repository:
    """All CRUD operations for trades, signals, snapshots, engine state."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # --- Trades ---

    async def insert_trade(self, trade: dict[str, Any]) -> dict:
        try:
            result = await asyncio.to_thread(_exec, self.db.table("trades").insert(trade))
            return result.data[0] if result.data else {}
        except Exception as e:
            err_str = str(e)
            if "PGRST204" in err_str or "column" in err_str.lower():
                # Column doesn't exist in DB — retry without extra columns
                _extra = {"leverage", "margin_used", "liquidation_price", "confluence_components"}
                stripped = {k: v for k, v in trade.items() if k not in _extra}
                try:
                    result = await asyncio.to_thread(
                        _exec, self.db.table("trades").insert(stripped)
                    )
                    logger.warning("insert_trade_stripped_columns", symbol=trade.get("symbol"))
                    return result.data[0] if result.data else {}
                except Exception as e2:
                    logger.error("insert_trade_failed", error=str(e2), symbol=trade.get("symbol"))
                    return {}
            logger.error("insert_trade_failed", error=err_str, symbol=trade.get("symbol"))
            return {}

    async def update_trade(self, trade_id: str, updates: dict[str, Any]) -> dict:
        try:
            updates["updated_at"] = datetime.now(timezone.utc).isoformat()
            result = await asyncio.to_thread(
                _exec, self.db.table("trades").update(updates).eq("id", trade_id)
            )
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error("update_trade_failed", error=str(e), trade_id=trade_id)
            return {}

    async def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_quantity: float,
        exit_order_id: str,
        exit_reason: str,
        pnl_usd: float,
        pnl_percent: float,
        fees_usd: float,
    ) -> dict:
        return await self.update_trade(
            trade_id,
            {
                "status": "closed",
                "exit_price": exit_price,
                "exit_quantity": exit_quantity,
                "exit_order_id": exit_order_id,
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "exit_reason": exit_reason,
                "pnl_usd": pnl_usd,
                "pnl_percent": pnl_percent,
                "fees_usd": fees_usd,
            },
        )

    # --- Partial Exits (Progressive TP) ---

    async def log_partial_exit(
        self,
        trade_id: str,
        tier: int,
        exit_price: float,
        exit_quantity: float,
        exit_order_id: str,
        exit_reason: str,
        pnl_usd: float,
        pnl_percent: float,
        fees_usd: float,
        remaining_quantity: float,
        new_stop_loss: float | None = None,
    ) -> dict:
        """Log a partial exit (TP tier hit)."""
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("partial_exits").insert({
                    "trade_id": trade_id,
                    "tier": tier,
                    "exit_price": exit_price,
                    "exit_quantity": exit_quantity,
                    "exit_order_id": exit_order_id,
                    "exit_reason": exit_reason,
                    "pnl_usd": pnl_usd,
                    "pnl_percent": pnl_percent,
                    "fees_usd": fees_usd,
                    "remaining_quantity": remaining_quantity,
                    "new_stop_loss": new_stop_loss,
                }),
            )
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error("log_partial_exit_failed", error=str(e), trade_id=trade_id, tier=tier)
            return {}

    async def get_partial_exits(self, trade_id: str) -> list[dict]:
        """Get all partial exits for a trade."""
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("partial_exits")
                .select("*")
                .eq("trade_id", trade_id)
                .order("created_at"),
            )
            return result.data or []
        except Exception as e:
            logger.error("get_partial_exits_failed", error=str(e), trade_id=trade_id)
            return []

    async def log_reversal(
        self,
        old_trade_id: str,
        new_trade_id: str,
        symbol: str,
        old_direction: str,
        new_direction: str,
        close_pnl: float,
        signal_score: float,
    ) -> dict:
        """Record a position reversal event for analytics."""
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("reversals").insert({
                    "old_trade_id": old_trade_id,
                    "new_trade_id": new_trade_id,
                    "symbol": symbol,
                    "old_direction": old_direction,
                    "new_direction": new_direction,
                    "close_pnl": close_pnl,
                    "signal_score": signal_score,
                }),
            )
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error("log_reversal_failed", error=str(e), symbol=symbol)
            return {}

    async def get_trades_by_ids(self, trade_ids: list[str]) -> list[dict]:
        if not trade_ids:
            return []
        result = await asyncio.to_thread(
            _exec, self.db.table("trades").select("*").in_("id", trade_ids)
        )
        return result.data or []

    async def get_open_trades(self, mode: str | None = None) -> list[dict]:
        query = self.db.table("trades").select("*").eq("status", "open")
        if mode:
            query = query.eq("mode", mode)
        result = await asyncio.to_thread(_exec, query.order("entry_time", desc=True))
        return result.data or []

    async def get_trades(
        self, status: str = "all", mode: str | None = None, page: int = 1, per_page: int = 25
    ) -> list[dict]:
        query = self.db.table("trades").select("*")
        if status != "all":
            query = query.eq("status", status)
        if mode:
            query = query.eq("mode", mode)
        offset = (page - 1) * per_page
        result = await asyncio.to_thread(
            _exec, query.order("entry_time", desc=True).range(offset, offset + per_page - 1)
        )
        return result.data or []

    async def get_daily_realized_pnl(self, mode: str | None = None) -> float:
        """Sum pnl_usd of trades closed today + partial exits from today (UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00+00:00")

        # Closed trades today
        query = (
            self.db.table("trades")
            .select("pnl_usd")
            .eq("status", "closed")
            .gte("exit_time", today)
        )
        if mode:
            query = query.eq("mode", mode)
        result = await asyncio.to_thread(_exec, query)
        closed_pnl = sum(float(t.get("pnl_usd", 0)) for t in (result.data or []))

        # Partial exits from today on still-open trades
        partial_pnl = await self._get_todays_partial_exit_pnl(today, mode)

        return closed_pnl + partial_pnl

    async def _get_todays_partial_exit_pnl(self, today: str, mode: str | None = None) -> float:
        """Sum partial_exits.pnl_usd created today for still-open trades."""
        try:
            # Get open trade IDs (optionally filtered by mode)
            open_query = self.db.table("trades").select("id").eq("status", "open")
            if mode:
                open_query = open_query.eq("mode", mode)
            open_result = await asyncio.to_thread(_exec, open_query)
            open_ids = [t["id"] for t in (open_result.data or [])]
            if not open_ids:
                return 0.0

            result = await asyncio.to_thread(
                _exec,
                self.db.table("partial_exits")
                .select("pnl_usd")
                .in_("trade_id", open_ids)
                .gte("created_at", today),
            )
            return sum(float(pe.get("pnl_usd", 0)) for pe in (result.data or []))
        except Exception as e:
            logger.error("todays_partial_exit_pnl_failed", error=str(e))
            return 0.0

    async def get_open_trade_partial_pnl(self, mode: str | None = None) -> float:
        """Sum all partial_exits.pnl_usd for currently open trades."""
        try:
            open_query = self.db.table("trades").select("id").eq("status", "open")
            if mode:
                open_query = open_query.eq("mode", mode)
            open_result = await asyncio.to_thread(_exec, open_query)
            open_ids = [t["id"] for t in (open_result.data or [])]
            if not open_ids:
                return 0.0

            result = await asyncio.to_thread(
                _exec,
                self.db.table("partial_exits")
                .select("pnl_usd")
                .in_("trade_id", open_ids),
            )
            return sum(float(pe.get("pnl_usd", 0)) for pe in (result.data or []))
        except Exception as e:
            logger.error("open_trade_partial_pnl_failed", error=str(e))
            return 0.0

    async def get_trade_stats(self, mode: str | None = None) -> dict:
        query = self.db.table("trades").select("*").eq("status", "closed")
        if mode:
            query = query.eq("mode", mode)
        result = await asyncio.to_thread(_exec, query)
        trades = result.data or []

        # Include partial exit PnL from still-open trades
        partial_pnl = await self.get_open_trade_partial_pnl(mode)
        daily_pnl = await self.get_daily_realized_pnl(mode)

        if not trades:
            return {
                "total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl": partial_pnl, "avg_pnl": 0,
                "daily_pnl": daily_pnl,
            }

        # Break-even (pnl=0) counts as a win, not a loss
        wins = [t for t in trades if float(t.get("pnl_usd", 0)) >= 0]
        closed_pnl = sum(float(t.get("pnl_usd", 0)) for t in trades)
        total_pnl = closed_pnl + partial_pnl

        # Best and worst trades
        best_trade = max(trades, key=lambda t: float(t.get("pnl_usd", 0)))
        worst_trade = min(trades, key=lambda t: float(t.get("pnl_usd", 0)))

        # Exit reason breakdown
        exit_reasons: dict[str, int] = {}
        for t in trades:
            reason = t.get("exit_reason", "unknown") or "unknown"
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        return {
            "total": len(trades),
            "wins": len(wins),
            "losses": len(trades) - len(wins),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "total_pnl": total_pnl,
            "avg_pnl": closed_pnl / len(trades) if trades else 0,
            "daily_pnl": daily_pnl,
            "best_pnl": float(best_trade.get("pnl_usd", 0)),
            "best_symbol": best_trade.get("symbol", ""),
            "worst_pnl": float(worst_trade.get("pnl_usd", 0)),
            "worst_symbol": worst_trade.get("symbol", ""),
            "exit_reasons": exit_reasons,
        }

    # --- Signals ---

    async def insert_signal(self, signal: dict[str, Any]) -> dict:
        try:
            result = await asyncio.to_thread(_exec, self.db.table("signals").insert(signal))
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error("insert_signal_failed", error=str(e), symbol=signal.get("symbol"))
            return {}

    async def get_recent_signals(self, limit: int = 10) -> list[dict]:
        result = await asyncio.to_thread(
            _exec,
            self.db.table("signals").select("*").order("created_at", desc=True).limit(limit),
        )
        return result.data or []

    async def get_signals(self, page: int = 1, per_page: int = 50) -> list[dict]:
        offset = (page - 1) * per_page
        result = await asyncio.to_thread(
            _exec,
            self.db.table("signals")
            .select("*")
            .order("created_at", desc=True)
            .range(offset, offset + per_page - 1),
        )
        return result.data or []

    # --- Portfolio Snapshots ---

    async def insert_snapshot(self, snapshot: dict[str, Any]) -> dict:
        result = await asyncio.to_thread(
            _exec, self.db.table("portfolio_snapshots").insert(snapshot)
        )
        return result.data[0] if result.data else {}

    async def get_latest_snapshot(self) -> dict | None:
        result = await asyncio.to_thread(
            _exec,
            self.db.table("portfolio_snapshots")
            .select("*")
            .order("created_at", desc=True)
            .limit(1),
        )
        return result.data[0] if result.data else None

    async def get_snapshot_history(self, hours: int = 168) -> list[dict]:
        result = await asyncio.to_thread(
            _exec,
            self.db.table("portfolio_snapshots")
            .select("*")
            .order("created_at", desc=True)
            .limit(hours * 4),
        )
        return list(reversed(result.data or []))

    # --- Engine State (singleton) ---

    async def get_engine_state(self) -> dict | None:
        result = await asyncio.to_thread(
            _exec, self.db.table("engine_state").select("*").eq("id", 1)
        )
        return result.data[0] if result.data else None

    async def upsert_engine_state(self, state: dict[str, Any]) -> dict:
        state["id"] = 1
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        result = await asyncio.to_thread(
            _exec, self.db.table("engine_state").upsert(state)
        )
        return result.data[0] if result.data else {}

    # --- Error Log ---

    async def log_error(
        self, component: str, level: str, message: str, details: dict | None = None, stack_trace: str | None = None
    ) -> None:
        try:
            await asyncio.to_thread(
                _exec,
                self.db.table("error_log").insert(
                    {
                        "component": component,
                        "level": level,
                        "message": message,
                        "details": details,
                        "stack_trace": stack_trace,
                    }
                ),
            )
        except Exception as e:
            logger.error("failed_to_log_error", error=str(e))

    # --- Candle Cache ---

    async def get_cached_candles(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> list[dict]:
        result = await asyncio.to_thread(
            _exec,
            self.db.table("candle_cache")
            .select("*")
            .eq("symbol", symbol)
            .eq("timeframe", timeframe)
            .order("timestamp", desc=True)
            .limit(limit),
        )
        return list(reversed(result.data or []))

    async def upsert_candles(self, symbol: str, timeframe: str, candles: list[dict]) -> None:
        if not candles:
            return
        rows = []
        for c in candles:
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": c["timestamp"],
                    "open": c["open"],
                    "high": c["high"],
                    "low": c["low"],
                    "close": c["close"],
                    "volume": c["volume"],
                }
            )
        try:
            await asyncio.to_thread(
                _exec,
                self.db.table("candle_cache").upsert(
                    rows, on_conflict="symbol,timeframe,timestamp"
                ),
            )
        except Exception as e:
            logger.error("candle_cache_upsert_failed", error=str(e), symbol=symbol)
