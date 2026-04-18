from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from src.data.db import Database
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _exec(query):
    """Execute a Supabase query synchronously (for use with asyncio.to_thread)."""
    return query.execute()


class Repository:
    """All CRUD operations for trades, signals, snapshots, engine state.

    Every query is scoped to ``instance_id`` so multiple bot instances can
    share the same Supabase database without interfering with each other.
    """

    def __init__(self, db: Database, instance_id: str = "main") -> None:
        self.db = db
        self.instance_id = instance_id

    # --- Trades ---

    # Known columns in the Supabase trades table
    _TRADE_COLUMNS = {
        "symbol", "direction", "status", "mode",
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
        "instance_id",
        "strategy", "entry_reason", "mm_formation", "mm_cycle_phase", "mm_confluence_grade",
        # Per-trade MM lifecycle state (migration 017) — must persist across
        # restarts or SL tightening / SVC invalidation / Refund Zone / 200 EMA
        # partial deduplication are all silently disabled.
        "mm_entry_type", "mm_peak2_wick_price",
        "mm_svc_high", "mm_svc_low",
        "mm_sl_moved_to_breakeven", "mm_sl_moved_under_50ema",
        "mm_took_200ema_partial",
        # HTF trend persistence (migration 018) — records the 4H/1D EMA trend
        # direction at entry and whether the trade fought the 4H trend.
        # Without these, counter-trend losses are invisible in post-mortems.
        "htf_trend_4h", "htf_trend_1d", "counter_trend",
        "created_at", "updated_at",
    }

    async def insert_trade(self, trade: dict[str, Any]) -> dict:
        # Tag with instance_id
        trade["instance_id"] = self.instance_id
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
        query = (
            self.db.table("trades").select("*")
            .eq("status", "open")
            .eq("instance_id", self.instance_id)
        )
        if mode:
            query = query.eq("mode", mode)
        result = await asyncio.to_thread(_exec, query.order("entry_time", desc=True))
        return result.data or []

    async def get_recent_trades_for_symbol(
        self, symbol: str, limit: int = 5, mode: str | None = None
    ) -> list[dict]:
        """Fetch recent closed trades for a specific symbol (for Agent 1 feedback loop)."""
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
                .eq("instance_id", self.instance_id)
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
        query = self.db.table("trades").select("*").eq("instance_id", self.instance_id)
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
            .eq("instance_id", self.instance_id)
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
            open_query = (
                self.db.table("trades").select("id")
                .eq("status", "open")
                .eq("instance_id", self.instance_id)
            )
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
            open_query = (
                self.db.table("trades").select("id")
                .eq("status", "open")
                .eq("instance_id", self.instance_id)
            )
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
        query = (
            self.db.table("trades").select("*")
            .eq("status", "closed")
            .eq("instance_id", self.instance_id)
        )
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
        signal["instance_id"] = self.instance_id
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
        """Fallback: fetch the most recent acted-on signal with agent_analysis for a symbol."""
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("signals")
                .select("components")
                .eq("symbol", symbol)
                .eq("instance_id", self.instance_id)
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
                .eq("instance_id", self.instance_id)
                .not_.is_("components->agent_analysis", "null")
                .order("created_at", desc=True)
                .limit(1),
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.debug("get_signal_by_symbol_failed", error=str(e), symbol=symbol)
            return None

    async def link_signal_to_trade(self, symbol: str, trade_id: str) -> None:
        """Link the most recent unlinked signal for a symbol to a trade_id."""
        try:
            await asyncio.to_thread(
                _exec,
                self.db.table("signals")
                .update({"trade_id": trade_id, "acted_on": True})
                .eq("symbol", symbol)
                .eq("instance_id", self.instance_id)
                .is_("trade_id", "null")
                .not_.is_("components->agent_analysis", "null")
                .order("created_at", desc=True)
                .limit(1),
            )
        except Exception as e:
            logger.debug("link_signal_to_trade_failed", error=str(e), symbol=symbol)

    async def update_signal_components(self, symbol: str, trade_id: str, extra: dict) -> None:
        """Merge extra data (e.g. refiner_journey) into an existing signal's components JSON."""
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
        """Fetch closed trades joined with their signal data for the analytics page."""
        try:
            # Fetch closed trades
            result = await asyncio.to_thread(
                _exec,
                self.db.table("trades")
                .select("*")
                .eq("status", "closed")
                .eq("mode", mode)
                .eq("instance_id", self.instance_id)
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
            self.db.table("signals")
            .select("*")
            .eq("instance_id", self.instance_id)
            .order("created_at", desc=True)
            .limit(limit),
        )
        return result.data or []

    async def get_signals(self, page: int = 1, per_page: int = 50) -> list[dict]:
        offset = (page - 1) * per_page
        result = await asyncio.to_thread(
            _exec,
            self.db.table("signals")
            .select("*")
            .eq("instance_id", self.instance_id)
            .order("created_at", desc=True)
            .range(offset, offset + per_page - 1),
        )
        return result.data or []

    # --- Portfolio Snapshots ---

    async def insert_snapshot(self, snapshot: dict[str, Any]) -> dict:
        snapshot["instance_id"] = self.instance_id
        result = await asyncio.to_thread(
            _exec, self.db.table("portfolio_snapshots").insert(snapshot)
        )
        return result.data[0] if result.data else {}

    async def get_latest_snapshot(self) -> dict | None:
        result = await asyncio.to_thread(
            _exec,
            self.db.table("portfolio_snapshots")
            .select("*")
            .eq("instance_id", self.instance_id)
            .order("created_at", desc=True)
            .limit(1),
        )
        return result.data[0] if result.data else None

    async def get_snapshot_history(self, hours: int = 168) -> list[dict]:
        result = await asyncio.to_thread(
            _exec,
            self.db.table("portfolio_snapshots")
            .select("*")
            .eq("instance_id", self.instance_id)
            .order("created_at", desc=True)
            .limit(hours * 4),
        )
        return list(reversed(result.data or []))

    # --- Engine State (per-instance) ---

    async def get_engine_state(self) -> dict | None:
        result = await asyncio.to_thread(
            _exec,
            self.db.table("engine_state")
            .select("*")
            .eq("instance_id", self.instance_id),
        )
        return result.data[0] if result.data else None

    async def upsert_engine_state(self, state: dict[str, Any]) -> dict:
        state["instance_id"] = self.instance_id
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        result = await asyncio.to_thread(
            _exec, self.db.table("engine_state").upsert(state, on_conflict="instance_id")
        )
        return result.data[0] if result.data else {}

    async def reset_mode_data(self, mode: str) -> dict:
        """Reset trades and snapshots for a specific mode (e.g. 'paper' or 'flipped_paper')."""
        deleted = {"trades": 0, "snapshots": 0}

        # Delete trades for this mode and instance
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("trades")
                .delete()
                .eq("mode", mode)
                .eq("instance_id", self.instance_id)
                .gte("created_at", "1970-01-01"),
            )
            deleted["trades"] = len(result.data) if result.data else 0
            logger.info("reset_mode_trades", mode=mode, count=deleted["trades"])
        except Exception as e:
            logger.warning("reset_mode_trades_failed", mode=mode, error=str(e))

        # Delete portfolio snapshots for this instance
        if mode not in ("flipped_paper", "custom_paper"):
            try:
                result = await asyncio.to_thread(
                    _exec,
                    self.db.table("portfolio_snapshots")
                    .delete()
                    .eq("instance_id", self.instance_id)
                    .neq("id", 0),
                )
                deleted["snapshots"] = len(result.data) if result.data else 0
                logger.info("reset_mode_snapshots", mode=mode, count=deleted["snapshots"])
            except Exception as e:
                logger.warning("reset_mode_snapshots_failed", mode=mode, error=str(e))

        return deleted

    async def wipe_all_data(self) -> None:
        """Nuclear reset — delete ALL data for THIS INSTANCE from every table."""
        # First, collect trade IDs so we can delete FK-dependent rows
        trade_ids: list[str] = []
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("trades").select("id").eq("instance_id", self.instance_id),
            )
            trade_ids = [t["id"] for t in (result.data or [])]
        except Exception:
            pass

        # Delete FK-dependent rows BEFORE trades
        if trade_ids:
            for dep_table in ["trade_lessons", "partial_exits"]:
                try:
                    await asyncio.to_thread(
                        _exec,
                        self.db.table(dep_table)
                        .delete()
                        .in_("trade_id", trade_ids)
                        .gte("created_at", "1970-01-01"),
                    )
                    logger.info("wipe_table_ok", table=dep_table, instance_id=self.instance_id)
                except Exception as e:
                    logger.warning("wipe_table_failed", table=dep_table, error=str(e))

        # Now safe to delete trades and signals
        for table in ["trades", "signals"]:
            try:
                await asyncio.to_thread(
                    _exec,
                    self.db.table(table)
                    .delete()
                    .eq("instance_id", self.instance_id)
                    .gte("created_at", "1970-01-01"),
                )
                logger.info("wipe_table_ok", table=table, instance_id=self.instance_id)
            except Exception as e:
                logger.warning("wipe_table_failed", table=table, error=str(e))

        # Engine state for this instance
        try:
            await asyncio.to_thread(
                _exec,
                self.db.table("engine_state")
                .delete()
                .eq("instance_id", self.instance_id),
            )
            logger.info("wipe_table_ok", table="engine_state", instance_id=self.instance_id)
        except Exception as e:
            logger.warning("wipe_table_failed", table="engine_state", error=str(e))

        # Portfolio snapshots for this instance
        try:
            await asyncio.to_thread(
                _exec,
                self.db.table("portfolio_snapshots")
                .delete()
                .eq("instance_id", self.instance_id)
                .neq("id", 0),
            )
            logger.info("wipe_table_ok", table="portfolio_snapshots", instance_id=self.instance_id)
        except Exception as e:
            logger.warning("wipe_table_failed", table="portfolio_snapshots", error=str(e))

        # Candle cache is shared — don't wipe
        logger.info("wipe_all_data_complete", instance_id=self.instance_id)

    # --- Trade Lessons (shared across instances — self-improving feedback loop) ---

    async def insert_lesson(self, lesson: dict[str, Any]) -> dict:
        """Insert a new AI-generated trade lesson."""
        try:
            result = await asyncio.to_thread(
                _exec, self.db.table("trade_lessons").insert(lesson)
            )
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.warning("insert_lesson_failed", error=str(e))
            return {}

    async def get_recent_lessons(
        self,
        applies_to: str | None = None,
        limit: int = 10,
        min_severity: str | None = None,
    ) -> list[dict]:
        """Fetch recent lessons, optionally filtered by agent and severity."""
        try:
            query = (
                self.db.table("trade_lessons")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
            )
            if applies_to:
                query = query.contains("applies_to", [applies_to])
            if min_severity == "high":
                query = query.in_("severity", ["high", "critical"])
            elif min_severity == "critical":
                query = query.eq("severity", "critical")
            result = await asyncio.to_thread(_exec, query)
            return result.data or []
        except Exception as e:
            logger.warning("get_recent_lessons_failed", error=str(e))
            return []

    async def get_lessons_for_symbol(self, symbol: str, limit: int = 5) -> list[dict]:
        """Fetch recent lessons for a specific symbol."""
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("trade_lessons")
                .select("*")
                .eq("symbol", symbol)
                .order("created_at", desc=True)
                .limit(limit),
            )
            return result.data or []
        except Exception as e:
            logger.warning("get_lessons_for_symbol_failed", symbol=symbol, error=str(e))
            return []

    async def increment_lesson_applied(self, lesson_id: str) -> None:
        """Increment times_applied for a lesson (called when lesson is shown to an agent)."""
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("trade_lessons").select("times_applied").eq("id", lesson_id),
            )
            if result.data:
                current = result.data[0].get("times_applied", 0) or 0
                await asyncio.to_thread(
                    _exec,
                    self.db.table("trade_lessons")
                    .update({"times_applied": current + 1})
                    .eq("id", lesson_id),
                )
        except Exception:
            pass  # Non-critical

    async def mark_lesson_helped(self, lesson_id: str) -> None:
        """Mark that a lesson contributed to a winning trade."""
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("trade_lessons")
                .select("times_helped, times_applied")
                .eq("id", lesson_id),
            )
            if result.data:
                helped = (result.data[0].get("times_helped", 0) or 0) + 1
                applied = result.data[0].get("times_applied", 0) or 1
                effectiveness = helped / max(applied, 1)
                await asyncio.to_thread(
                    _exec,
                    self.db.table("trade_lessons")
                    .update({"times_helped": helped, "effectiveness": round(effectiveness, 3)})
                    .eq("id", lesson_id),
                )
        except Exception:
            pass  # Non-critical

    async def get_lesson_stats(self) -> dict:
        """Get summary stats for the lesson system."""
        try:
            result = await asyncio.to_thread(
                _exec, self.db.table("trade_lessons").select("severity, outcome, lesson_type")
            )
            lessons = result.data or []
            if not lessons:
                return {"total": 0}
            return {
                "total": len(lessons),
                "by_type": {t: sum(1 for l in lessons if l.get("lesson_type") == t)
                            for t in set(l.get("lesson_type", "") for l in lessons)},
                "by_outcome": {o: sum(1 for l in lessons if l.get("outcome") == o)
                               for o in ("win", "loss")},
                "by_severity": {s: sum(1 for l in lessons if l.get("severity") == s)
                                for s in ("low", "medium", "high", "critical")},
            }
        except Exception as e:
            logger.warning("get_lesson_stats_failed", error=str(e))
            return {"total": 0}

    # --- All Instances (for dashboard instance switcher) ---

    async def get_all_instances(self) -> list[dict]:
        """Fetch all known instance IDs and their current state."""
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("engine_state")
                .select("instance_id, status, mode, current_balance, daily_pnl_usd, total_pnl_usd, updated_at")
                .order("instance_id"),
            )
            return result.data or []
        except Exception:
            return []

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

    # --- Candle Cache (shared across instances) ---

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

    # --- API Usage Tracking ---

    async def log_api_usage(self, usage: dict[str, Any]) -> None:
        """Insert a row into api_usage. Fire-and-forget, never raises."""
        usage["instance_id"] = self.instance_id
        try:
            await asyncio.to_thread(
                _exec, self.db.table("api_usage").insert(usage)
            )
        except Exception as e:
            logger.debug("log_api_usage_failed", error=str(e))

    async def get_usage_summary(self, days: int = 30) -> list[dict]:
        """Daily aggregates: date, total_cost, request_count, input_tokens, output_tokens."""
        since = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        from datetime import timedelta
        since -= timedelta(days=days)
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("api_usage")
                .select("created_at, cost_usd, input_tokens, output_tokens")
                .eq("instance_id", self.instance_id)
                .gte("created_at", since.isoformat())
                .order("created_at"),
            )
            rows = result.data or []
            # Aggregate by day client-side (Supabase doesn't do GROUP BY)
            daily: dict[str, dict] = {}
            for r in rows:
                day = r["created_at"][:10]  # YYYY-MM-DD
                if day not in daily:
                    daily[day] = {"date": day, "cost_usd": 0, "requests": 0, "input_tokens": 0, "output_tokens": 0}
                daily[day]["cost_usd"] += float(r.get("cost_usd", 0))
                daily[day]["requests"] += 1
                daily[day]["input_tokens"] += int(r.get("input_tokens", 0))
                daily[day]["output_tokens"] += int(r.get("output_tokens", 0))
            return sorted(daily.values(), key=lambda x: x["date"])
        except Exception as e:
            logger.warning("get_usage_summary_failed", error=str(e))
            return []

    async def get_usage_by_model(self, days: int = 30) -> list[dict]:
        """Group by model: model, total_cost, total_requests."""
        from datetime import timedelta
        since = datetime.now(timezone.utc) - timedelta(days=days)
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("api_usage")
                .select("model, cost_usd, input_tokens, output_tokens")
                .eq("instance_id", self.instance_id)
                .gte("created_at", since.isoformat()),
            )
            rows = result.data or []
            models: dict[str, dict] = {}
            for r in rows:
                m = r["model"]
                if m not in models:
                    models[m] = {"model": m, "cost_usd": 0, "requests": 0, "input_tokens": 0, "output_tokens": 0}
                models[m]["cost_usd"] += float(r.get("cost_usd", 0))
                models[m]["requests"] += 1
                models[m]["input_tokens"] += int(r.get("input_tokens", 0))
                models[m]["output_tokens"] += int(r.get("output_tokens", 0))
            return sorted(models.values(), key=lambda x: x["cost_usd"], reverse=True)
        except Exception as e:
            logger.warning("get_usage_by_model_failed", error=str(e))
            return []

    async def get_usage_by_caller(self, days: int = 30) -> list[dict]:
        """Group by caller: caller, total_cost, total_requests, tokens."""
        from datetime import timedelta
        since = datetime.now(timezone.utc) - timedelta(days=days)
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("api_usage")
                .select("caller, cost_usd, input_tokens, output_tokens")
                .eq("instance_id", self.instance_id)
                .gte("created_at", since.isoformat()),
            )
            rows = result.data or []
            callers: dict[str, dict] = {}
            for r in rows:
                c = r["caller"]
                if c not in callers:
                    callers[c] = {"caller": c, "cost_usd": 0, "requests": 0, "input_tokens": 0, "output_tokens": 0}
                callers[c]["cost_usd"] += float(r.get("cost_usd", 0))
                callers[c]["requests"] += 1
                callers[c]["input_tokens"] += int(r.get("input_tokens", 0))
                callers[c]["output_tokens"] += int(r.get("output_tokens", 0))
            return sorted(callers.values(), key=lambda x: x["cost_usd"], reverse=True)
        except Exception as e:
            logger.warning("get_usage_by_caller_failed", error=str(e))
            return []

    async def get_usage_totals(self, days: int = 30) -> dict:
        """Current period totals for header stat cards."""
        now = datetime.now(timezone.utc)
        # Month start
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("api_usage")
                .select("cost_usd, created_at")
                .eq("instance_id", self.instance_id)
                .gte("created_at", month_start.isoformat()),
            )
            rows = result.data or []
            month_cost = sum(float(r.get("cost_usd", 0)) for r in rows)
            month_requests = len(rows)

            # Daily avg and projection
            days_elapsed = max((now - month_start).days, 1)
            import calendar
            days_in_month = calendar.monthrange(now.year, now.month)[1]
            daily_avg = month_cost / days_elapsed
            projected = daily_avg * days_in_month

            return {
                "month_cost": round(month_cost, 4),
                "month_requests": month_requests,
                "daily_avg": round(daily_avg, 4),
                "projected": round(projected, 2),
                "period": now.strftime("%Y-%m"),
            }
        except Exception as e:
            logger.warning("get_usage_totals_failed", error=str(e))
            return {"month_cost": 0, "month_requests": 0, "daily_avg": 0, "projected": 0, "period": now.strftime("%Y-%m")}

    async def get_month_usage_cost(self) -> float:
        """Quick query: total cost this calendar month. Used for alert banner."""
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        try:
            result = await asyncio.to_thread(
                _exec,
                self.db.table("api_usage")
                .select("cost_usd")
                .eq("instance_id", self.instance_id)
                .gte("created_at", month_start.isoformat()),
            )
            return sum(float(r.get("cost_usd", 0)) for r in (result.data or []))
        except Exception:
            return 0.0
