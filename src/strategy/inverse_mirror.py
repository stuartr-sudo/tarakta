"""Inverse mirror bot for Tarakta MM trades.

This runtime does not identify setups itself. It watches a source Tarakta
instance's persisted MM trades and opens the opposite exposure in its own
instance. That makes the inverse bot track what the source bot actually takes,
including sanity-agent vetoes and any other upstream filters.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _opposite_direction(direction: str) -> str:
    return "short" if direction == "long" else "long"


def _entry_side(direction: str) -> str:
    return "buy" if direction == "long" else "sell"


def _close_side(direction: str) -> str:
    return "sell" if direction == "long" else "buy"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _source_id_from_reason(reason: str | None) -> str | None:
    if not reason:
        return None
    marker = "inverse_of:"
    if marker not in reason:
        return None
    return reason.split(marker, 1)[1].split()[0].strip() or None


def _partial_id(row: dict) -> str:
    return str(row.get("id") or f"{row.get('trade_id')}:{row.get('tier')}:{row.get('exit_quantity')}")


def _mirrored_partial_id(reason: str | None) -> str | None:
    if not reason:
        return None
    marker = "inverse_partial_of:"
    if marker not in reason:
        return None
    return reason.split(marker, 1)[1].split()[0].strip() or None


class InverseMirrorEngine:
    """Mirror source MM trades into opposite-side trades for this instance."""

    def __init__(self, exchange, repo, source_repo, config) -> None:
        self.exchange = exchange
        self.repo = repo
        self.source_repo = source_repo
        self.config = config
        self.source_strategy = getattr(config, "inverse_source_strategy", "mm_method")
        self.inverse_strategy = getattr(config, "inverse_strategy_tag", "mm_inverse")
        self.poll_interval = float(getattr(config, "inverse_poll_interval_seconds", 15.0) or 15.0)
        self.quantity_multiplier = float(getattr(config, "inverse_quantity_multiplier", 1.0) or 1.0)
        self.trade_lookback = int(getattr(config, "inverse_trade_lookback", 500) or 500)
        self._running = False
        self._restored = False

    async def run(self) -> None:
        self._running = True
        await self._restore_exchange_positions()
        logger.info(
            "inverse_mirror_started",
            source_instance=getattr(self.source_repo, "instance_id", ""),
            target_instance=getattr(self.repo, "instance_id", ""),
            source_strategy=self.source_strategy,
            inverse_strategy=self.inverse_strategy,
            poll_interval=self.poll_interval,
        )
        while self._running:
            await self.sync_once()
            await asyncio.sleep(self.poll_interval)

    async def shutdown(self) -> None:
        self._running = False

    async def sync_once(self) -> None:
        """Perform one mirror pass."""
        source_trades = [
            t for t in await self.source_repo.get_trades(per_page=self.trade_lookback)
            if t.get("strategy") == self.source_strategy
        ]
        target_trades = [
            t for t in await self.repo.get_trades(per_page=self.trade_lookback)
            if t.get("strategy") == self.inverse_strategy
        ]

        source_by_id = {str(t.get("id")): t for t in source_trades if t.get("id")}
        mirrors = {
            source_id: t
            for t in target_trades
            if (source_id := _source_id_from_reason(t.get("entry_reason")))
        }

        for source in source_trades:
            source_id = str(source.get("id") or "")
            if not source_id or source.get("status") != "open":
                continue
            mirror = mirrors.get(source_id)
            if mirror is None:
                mirror = await self._open_inverse(source)
                if mirror:
                    mirrors[source_id] = mirror
            if mirror and mirror.get("status") == "open":
                await self._mirror_partials(source, mirror)

        for mirror in target_trades:
            if mirror.get("status") != "open":
                continue
            source_id = _source_id_from_reason(mirror.get("entry_reason"))
            source = source_by_id.get(source_id or "")
            if source is None or source.get("status") == "closed":
                await self._close_mirror(mirror, source, reason="source_closed")

    async def _restore_exchange_positions(self) -> None:
        if self._restored or not hasattr(self.exchange, "restore_positions"):
            return
        target_trades = [
            t for t in await self.repo.get_trades(status="open", per_page=self.trade_lookback)
            if t.get("strategy") == self.inverse_strategy
        ]
        positions = {}
        for t in target_trades:
            symbol = t.get("symbol")
            if not symbol:
                continue
            qty = _as_float(t.get("remaining_quantity") or t.get("entry_quantity"))
            if qty <= 0:
                continue
            positions[symbol] = SimpleNamespace(
                direction=t.get("direction", "long"),
                quantity=qty,
                entry_price=_as_float(t.get("entry_price")),
                margin_used=_as_float(t.get("margin_used")),
                cost_usd=_as_float(t.get("entry_cost_usd")),
                leverage=_as_float(t.get("leverage"), 0.0),
            )
        if positions:
            self.exchange.restore_positions(positions)
        self._restored = True

    async def _open_inverse(self, source: dict) -> dict | None:
        source_id = str(source.get("id") or "")
        symbol = source.get("symbol")
        source_direction = source.get("direction")
        if not source_id or not symbol or source_direction not in {"long", "short"}:
            return None

        direction = _opposite_direction(source_direction)
        qty = _as_float(source.get("remaining_quantity") or source.get("entry_quantity"))
        qty *= self.quantity_multiplier
        if qty <= 0:
            return None

        side = _entry_side(direction)
        try:
            result = await self.exchange.place_market_order(symbol=symbol, side=side, quantity=qty)
        except Exception as e:
            logger.warning("inverse_open_failed", source_trade_id=source_id, symbol=symbol, error=str(e))
            return None
        if not result or result.status != "closed":
            logger.info("inverse_open_not_filled", source_trade_id=source_id, symbol=symbol)
            return None

        fill_price = _as_float(result.avg_price, _as_float(source.get("entry_price")))
        filled_qty = _as_float(result.filled_quantity)
        leverage = _as_float(getattr(self.exchange, "leverage", None), _as_float(source.get("leverage"), 1.0))
        if leverage <= 0:
            leverage = 1.0
        stop_loss = self._inverse_stop_loss(source, fill_price, direction)
        take_profit = self._inverse_take_profit(source, fill_price)
        cost_usd = fill_price * filled_qty
        margin = cost_usd / leverage
        risk_usd = abs(fill_price - stop_loss) * filled_qty if stop_loss > 0 else 0.0

        row = await self.repo.insert_trade({
            "symbol": symbol,
            "direction": direction,
            "entry_price": fill_price,
            "entry_quantity": filled_qty,
            "original_quantity": filled_qty,
            "remaining_quantity": filled_qty,
            "stop_loss": stop_loss,
            "original_stop_loss": stop_loss,
            "take_profit": take_profit,
            "tp_tiers": json.dumps(self._inverse_tiers(source, fill_price)),
            "margin_used": round(margin, 2),
            "entry_cost_usd": round(cost_usd, 2),
            "risk_usd": round(risk_usd, 2),
            "leverage": int(round(leverage)),
            "instance_id": getattr(self.config, "instance_id", "inverse"),
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "strategy": self.inverse_strategy,
            "entry_reason": f"inverse_of:{source_id} source_direction={source_direction}",
            "mode": getattr(self.config, "trading_mode", "paper"),
            "status": "open",
        })
        logger.info(
            "inverse_trade_opened",
            source_trade_id=source_id,
            mirror_trade_id=row.get("id"),
            symbol=symbol,
            source_direction=source_direction,
            direction=direction,
            quantity=filled_qty,
        )
        return row or {}

    def _inverse_stop_loss(self, source: dict, entry: float, direction: str) -> float:
        source_tp = _as_float(source.get("take_profit"))
        if source_tp > 0:
            return source_tp
        distance = abs(_as_float(source.get("entry_price"), entry) - _as_float(source.get("stop_loss"), entry))
        return entry + distance if direction == "short" else entry - distance

    def _inverse_take_profit(self, source: dict, entry: float) -> float:
        source_sl = _as_float(source.get("stop_loss"))
        return source_sl if source_sl > 0 else entry

    def _inverse_tiers(self, source: dict, entry: float) -> dict:
        source_sl = _as_float(source.get("stop_loss"))
        tp = source_sl if source_sl > 0 else entry
        return {"l2": tp, "l3": tp}

    async def _mirror_partials(self, source: dict, mirror: dict) -> None:
        source_id = str(source.get("id") or "")
        mirror_id = str(mirror.get("id") or "")
        if not source_id or not mirror_id:
            return

        source_partials = await self.source_repo.get_partial_exits(source_id)
        if not source_partials:
            return
        mirror_partials = await self.repo.get_partial_exits(mirror_id)
        mirrored_ids = {
            pid for p in mirror_partials
            if (pid := _mirrored_partial_id(p.get("exit_reason")))
        }

        mirror_entry_time = _parse_dt(mirror.get("entry_time"))
        remaining = _as_float(mirror.get("remaining_quantity") or mirror.get("entry_quantity"))
        for partial in source_partials:
            partial_id = _partial_id(partial)
            if partial_id in mirrored_ids:
                continue
            partial_time = _parse_dt(partial.get("created_at"))
            if mirror_entry_time and partial_time and partial_time < mirror_entry_time:
                continue
            close_qty = min(
                remaining,
                _as_float(partial.get("exit_quantity")) * self.quantity_multiplier,
            )
            if close_qty <= 0:
                continue
            remaining = await self._close_quantity(
                mirror,
                close_qty,
                reason=f"inverse_partial_of:{partial_id}",
                tier=int(_as_float(partial.get("tier"), 0)),
            )
            mirror["remaining_quantity"] = remaining
            if remaining <= 0:
                break

    async def _close_mirror(self, mirror: dict, source: dict | None, reason: str) -> None:
        remaining = _as_float(mirror.get("remaining_quantity") or mirror.get("entry_quantity"))
        if remaining <= 0:
            return
        await self._close_quantity(mirror, remaining, reason=reason, tier=0, final=True, source=source)

    async def _close_quantity(
        self,
        mirror: dict,
        quantity: float,
        reason: str,
        tier: int = 0,
        final: bool = False,
        source: dict | None = None,
    ) -> float:
        symbol = mirror.get("symbol")
        direction = mirror.get("direction")
        mirror_id = str(mirror.get("id") or "")
        if not symbol or direction not in {"long", "short"} or not mirror_id:
            return _as_float(mirror.get("remaining_quantity"))

        side = _close_side(direction)
        try:
            result = await self.exchange.place_market_order(symbol=symbol, side=side, quantity=quantity)
        except Exception as e:
            logger.warning("inverse_close_failed", mirror_trade_id=mirror_id, symbol=symbol, error=str(e))
            return _as_float(mirror.get("remaining_quantity"))
        if not result or result.status != "closed":
            return _as_float(mirror.get("remaining_quantity"))

        exit_price = _as_float(result.avg_price, _as_float(source.get("exit_price") if source else None))
        entry_price = _as_float(mirror.get("entry_price"))
        filled_qty = _as_float(result.filled_quantity, quantity)
        if direction == "long":
            pnl_usd = (exit_price - entry_price) * filled_qty
        else:
            pnl_usd = (entry_price - exit_price) * filled_qty
        entry_notional = entry_price * filled_qty
        pnl_pct = (pnl_usd / entry_notional * 100.0) if entry_notional > 0 else 0.0
        remaining = max(0.0, _as_float(mirror.get("remaining_quantity") or mirror.get("entry_quantity")) - filled_qty)

        if final or remaining <= 0:
            await self.repo.update_trade(mirror_id, {
                "status": "closed",
                "exit_price": exit_price,
                "exit_quantity": filled_qty,
                "exit_order_id": getattr(result, "order_id", ""),
                "exit_reason": reason,
                "pnl_usd": round(pnl_usd, 4),
                "pnl_percent": round(pnl_pct, 4),
                "fees_usd": round(_as_float(getattr(result, "fee", 0.0)), 4),
                "remaining_quantity": 0.0,
                "exit_time": datetime.now(timezone.utc).isoformat(),
            })
        else:
            await self.repo.update_trade(mirror_id, {
                "remaining_quantity": remaining,
                "current_tier": tier,
            })
            await self.repo.log_partial_exit(
                trade_id=mirror_id,
                tier=tier,
                exit_price=exit_price,
                exit_quantity=filled_qty,
                exit_order_id=str(getattr(result, "order_id", "") or ""),
                exit_reason=reason,
                pnl_usd=round(pnl_usd, 4),
                pnl_percent=round(pnl_pct, 4),
                fees_usd=round(_as_float(getattr(result, "fee", 0.0)), 4),
                remaining_quantity=remaining,
                new_stop_loss=_as_float(mirror.get("stop_loss")),
            )

        logger.info(
            "inverse_trade_reduced",
            mirror_trade_id=mirror_id,
            symbol=symbol,
            side=side,
            quantity=filled_qty,
            remaining=remaining,
            reason=reason,
            pnl_usd=round(pnl_usd, 4),
        )
        return remaining
