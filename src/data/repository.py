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

    # Known columns in the Supabase trades table
    _TRADE_COLUMNS = {
        "symbol", "direction", "status", "mode", "market",
        "entry_price", "entry_quantity", "entry_cost_usd", "entry_order_id", "entry_time",
        "exit_price", "exit_quantity", "exit_order_id", "exit_time", "exit_reason",
        "stop_loss", "take_profit", "risk_usd", "risk_reward",
        "pnl_usd", "pnl_percent", "fees_usd",
        "confluence_score", "signal_reasons", "timeframes_used",
        "leverage", "margin_used", "liquidation_price",
        "test_group",
        "tp_tiers", "current_tier", "original_quantity", "remaining_quantity",
        "original_stop_loss",
        "last_agent2_action", "last_agent2_reasoning", "last_agent2_urgency",
        "agent2_confidence", "agent2_check_count",
        "last_agent3_action", "last_agent3_reasoning", "agent3_confidence",
        "last_agent3_sl",
        "created_at", "updated_at",
    }

    async def insert_trade(self, trade: dict[str, Any]) -> dict:
        # Only send columns that exist in the DB schema
        clean = {k: v for k, v in trade.items() if k in self._TRADE_COLUMNS}
        try:
            result = await asyncio.to_thread(_exec, self.db.table("trades").insert(clean))
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error("insert_trade_failed", error=str(e), symbol=trade.get("symbol"))
            return {}

    async def update_trade(self, trade_id: str, updates: dict[str, Any]) -> dict:
        try:
            # Only send columns that exist in the DB schema
            clean = {k: v for k, v in updates.items() if k in self._TRADE_COLUMNS}
            clean["updated_at"] = datetime.now(timezone.utc).isoformat()
            result = await asyncio.to_thread(
                _exec, self.db.table("trades").update(clean).eq("id", trade_id)
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

    async def get_recent_trades_for_symbol(
        self, symbol: str, limit: int = 5, mode: str | None = None
    ) -> list[dict]:
        """Fetch recent closed trades for a specific symbol (for Agent 1 feedback loop).

        Returns trades ordered by exit_time desc with fields useful for pattern recognition:
        direction, entry_price, exit_price, pnl_usd, pnl_percent, exit_reason,
        entry_time, exit_time, confluence_score.
        """
        select_fields = (
            "direction, entry_price, exit_price, pnl_usd, pnl_percent, "
            "exit_reason, entry_time, exit_time, confluence_score, stop_loss, take_profit"
        )
        try:
            query = (
                self.db.table("trades")
                .select(select_fields)
                .eq("symbol", symbol)
                .eq("status", "closed")
                .order("exit_time", desc=True)
                .limit(limit)
            )
            if mode:
                query = query.eq("mode", mode)
            result = await asyncio.to_thread(_exec, query)
            return result.data or []
        except Exception as e:
            logger.warning("get_recent_trades_for_symbol_failed", symbol=symbol, error=str(e))
            return []

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

    async def get_signal_by_trade_id(self, trade_id: str) -> dict | None:
        """Fetch the signal that resulted in a specific trade (for agent analysis display)."""
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("signals")
                .select("components")
                .eq("trade_id", trade_id)
                .limit(1),
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.debug("get_signal_by_trade_failed", error=str(e), trade_id=trade_id)
            return None

    async def get_signal_by_symbol_recent(self, symbol: str) -> dict | None:
        """Fallback: fetch the most recent acted-on signal with agent_analysis for a symbol.

        Only returns signals where acted_on=True to avoid showing SKIP analysis
        from a later scan cycle that didn't produce the trade.
        """
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("signals")
                .select("components")
                .eq("symbol", symbol)
                .eq("acted_on", True)
                .not_.is_("components->agent_analysis", "null")
                .order("created_at", desc=True)
                .limit(1),
            )
            if result.data:
                return result.data[0]
            # If no acted-on signal found, fall back to any signal (backward compat)
            result = await asyncio.to_thread(
                _exec,
                self.db.table("signals")
                .select("components")
                .eq("symbol", symbol)
                .not_.is_("components->agent_analysis", "null")
                .order("created_at", desc=True)
                .limit(1),
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.debug("get_signal_by_symbol_failed", error=str(e), symbol=symbol)
            return None

    async def link_signal_to_trade(self, symbol: str, trade_id: str) -> None:
        """Link the most recent unlinked signal for a symbol to a trade_id.

        Called when the entry refiner opens a trade from a WAIT_PULLBACK signal,
        so the dashboard can later display the agent analysis.
        """
        try:
            await asyncio.to_thread(
                _exec,
                self.db.table("signals")
                .update({"trade_id": trade_id, "acted_on": True})
                .eq("symbol", symbol)
                .is_("trade_id", "null")
                .not_.is_("components->agent_analysis", "null")
                .order("created_at", desc=True)
                .limit(1),
            )
        except Exception as e:
            logger.debug("link_signal_to_trade_failed", error=str(e), symbol=symbol)

    async def update_signal_components(self, symbol: str, trade_id: str, extra: dict) -> None:
        """Merge extra data (e.g. refiner_journey) into an existing signal's components JSON.

        Finds the signal linked to trade_id, reads its current components,
        merges `extra` into it, and writes back.
        """
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("signals")
                .select("id, components")
                .eq("trade_id", trade_id)
                .limit(1),
            )
            if not result.data:
                return
            row = result.data[0]
            components = row.get("components") or {}
            components.update(extra)
            await asyncio.to_thread(
                _exec,
                self.db.table("signals")
                .update({"components": components})
                .eq("id", row["id"]),
            )
        except Exception as e:
            logger.debug("update_signal_components_failed", error=str(e), trade_id=trade_id)

    async def get_closed_trades_with_signals(
        self, mode: str, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        """Fetch closed trades joined with their signal data for the analytics page.

        Returns trades with embedded signal components (agent_analysis, refiner_journey, etc.).
        """
        try:
            # Fetch closed trades
            result = await asyncio.to_thread(
                _exec,
                self.db.table("trades")
                .select("*")
                .eq("status", "closed")
                .eq("mode", mode)
                .order("exit_time", desc=True)
                .range(offset, offset + limit - 1),
            )
            trades = result.data or []
            if not trades:
                return []

            # Fetch signals for these trades in one batch
            trade_ids = [t["id"] for t in trades if t.get("id")]
            signal_map: dict[str, dict] = {}
            if trade_ids:
                sig_result = await asyncio.to_thread(
                    _exec,
                    self.db.table("signals")
                    .select("trade_id, components, signal_type, action, created_at")
                    .in_("trade_id", trade_ids),
                )
                for s in (sig_result.data or []):
                    signal_map[s["trade_id"]] = s

            # Merge
            merged = []
            for t in trades:
                sig = signal_map.get(t.get("id"), {})
                merged.append({
                    "trade": t,
                    "signal": sig,
                })
            return merged
        except Exception as e:
            logger.error("get_closed_trades_with_signals_failed", error=str(e))
            return []

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

    async def reset_mode_data(self, mode: str) -> dict:
        """Reset trades and snapshots for a specific mode (e.g. 'paper' or 'flipped_paper').

        Returns summary of what was deleted.
        """
        deleted = {"trades": 0, "snapshots": 0}

        # Delete trades for this mode
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("trades").delete().eq("mode", mode).gte("created_at", "1970-01-01"),
            )
            deleted["trades"] = len(result.data) if result.data else 0
            logger.info("reset_mode_trades", mode=mode, count=deleted["trades"])
        except Exception as e:
            logger.warning("reset_mode_trades_failed", mode=mode, error=str(e))

        # Delete portfolio snapshots (shared — only clear if resetting main)
        if mode not in ("flipped_paper", "custom_paper"):
            try:
                result = await asyncio.to_thread(
                    _exec,
                    self.db.table("portfolio_snapshots").delete().neq("id", 0),
                )
                deleted["snapshots"] = len(result.data) if result.data else 0
                logger.info("reset_mode_snapshots", mode=mode, count=deleted["snapshots"])
            except Exception as e:
                logger.warning("reset_mode_snapshots_failed", mode=mode, error=str(e))

        return deleted

    async def wipe_all_data(self) -> None:
        """Nuclear reset — delete ALL data from every table. Used by FORCE_RESET."""
        # Tables with integer PK (use neq id 0)
        int_pk_tables = ["engine_state", "portfolio_snapshots", "error_log"]
        # Tables with UUID PK (use gte created_at)
        uuid_pk_tables = ["signals", "partial_exits", "reversals", "trades"]
        # Tables with composite PK
        timestamp_tables = ["candle_cache"]

        for table in int_pk_tables:
            try:
                await asyncio.to_thread(
                    _exec, self.db.table(table).delete().neq("id", 0)
                )
                logger.info("wipe_table_ok", table=table)
            except Exception as e:
                logger.warning("wipe_table_failed", table=table, error=str(e))

        for table in uuid_pk_tables:
            try:
                await asyncio.to_thread(
                    _exec,
                    self.db.table(table).delete().gte("created_at", "1970-01-01"),
                )
                logger.info("wipe_table_ok", table=table)
            except Exception as e:
                logger.warning("wipe_table_failed", table=table, error=str(e))

        for table in timestamp_tables:
            try:
                await asyncio.to_thread(
                    _exec,
                    self.db.table(table).delete().gte("timestamp", "1970-01-01"),
                )
                logger.info("wipe_table_ok", table=table)
            except Exception as e:
                logger.warning("wipe_table_failed", table=table, error=str(e))

        logger.info("wipe_all_data_complete")

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
