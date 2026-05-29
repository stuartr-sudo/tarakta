from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.exchange.models import OrderResult
from src.strategy.inverse_mirror import InverseMirrorEngine


class FakeExchange:
    def __init__(self, price: float = 100.0) -> None:
        self.price = price
        self.leverage = 10
        self.orders: list[dict] = []
        self.restored = None

    def restore_positions(self, positions: dict) -> None:
        self.restored = positions

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> OrderResult:
        self.orders.append({"symbol": symbol, "side": side, "quantity": quantity})
        return OrderResult(
            order_id=f"order-{len(self.orders)}",
            symbol=symbol,
            side=side,
            filled_quantity=quantity,
            avg_price=self.price,
            fee=0.0,
            status="closed",
        )


class FakeRepo:
    def __init__(self, instance_id: str, trades: list[dict] | None = None) -> None:
        self.instance_id = instance_id
        self.trades = trades or []
        self.partials: dict[str, list[dict]] = {}

    async def get_trades(self, status: str = "all", mode=None, page: int = 1, per_page: int = 25) -> list[dict]:
        rows = [dict(t) for t in self.trades]
        if status != "all":
            rows = [t for t in rows if t.get("status") == status]
        return rows[:per_page]

    async def insert_trade(self, trade: dict) -> dict:
        row = dict(trade)
        row["id"] = row.get("id") or f"{self.instance_id}-trade-{len(self.trades) + 1}"
        self.trades.append(row)
        return dict(row)

    async def update_trade(self, trade_id: str, updates: dict) -> dict:
        for trade in self.trades:
            if str(trade.get("id")) == str(trade_id):
                trade.update(updates)
                return dict(trade)
        return {}

    async def get_partial_exits(self, trade_id: str) -> list[dict]:
        return [dict(p) for p in self.partials.get(str(trade_id), [])]

    async def log_partial_exit(self, **kwargs) -> dict:
        trade_id = str(kwargs["trade_id"])
        row = dict(kwargs)
        row["id"] = row.get("id") or f"partial-{len(self.partials.get(trade_id, [])) + 1}"
        self.partials.setdefault(trade_id, []).append(row)
        return dict(row)


def _source_trade(**overrides) -> dict:
    now = datetime.now(timezone.utc)
    row = {
        "id": "source-1",
        "strategy": "mm_method",
        "status": "open",
        "symbol": "BTC/USDT:USDT",
        "direction": "long",
        "entry_price": 100.0,
        "entry_quantity": 2.0,
        "remaining_quantity": 2.0,
        "stop_loss": 95.0,
        "take_profit": 110.0,
        "leverage": 10,
        "entry_time": now.isoformat(),
        "created_at": now.isoformat(),
    }
    row.update(overrides)
    return row


def _engine(source_repo: FakeRepo, target_repo: FakeRepo, exchange: FakeExchange) -> InverseMirrorEngine:
    config = SimpleNamespace(
        instance_id=target_repo.instance_id,
        trading_mode="paper",
        inverse_source_strategy="mm_method",
        inverse_strategy_tag="mm_inverse",
        inverse_poll_interval_seconds=15,
        inverse_quantity_multiplier=1.0,
        inverse_trade_lookback=100,
    )
    return InverseMirrorEngine(exchange=exchange, repo=target_repo, source_repo=source_repo, config=config)


@pytest.mark.asyncio
async def test_inverse_mirror_opens_opposite_trade_for_source_open():
    source_repo = FakeRepo("tarakta-mm", trades=[_source_trade()])
    target_repo = FakeRepo("tarakta-mm-inverse")
    exchange = FakeExchange(price=100.0)

    await _engine(source_repo, target_repo, exchange).sync_once()

    assert exchange.orders == [{"symbol": "BTC/USDT:USDT", "side": "sell", "quantity": 2.0}]
    assert len(target_repo.trades) == 1
    mirror = target_repo.trades[0]
    assert mirror["strategy"] == "mm_inverse"
    assert mirror["direction"] == "short"
    assert mirror["entry_reason"].startswith("inverse_of:source-1")
    assert mirror["stop_loss"] == 110.0
    assert mirror["take_profit"] == 95.0


@pytest.mark.asyncio
async def test_inverse_mirror_closes_when_source_closes():
    source = _source_trade(status="closed", exit_price=90.0, remaining_quantity=0.0)
    source_repo = FakeRepo("tarakta-mm", trades=[source])
    target_repo = FakeRepo("tarakta-mm-inverse", trades=[
        {
            "id": "mirror-1",
            "strategy": "mm_inverse",
            "status": "open",
            "symbol": "BTC/USDT:USDT",
            "direction": "short",
            "entry_price": 100.0,
            "entry_quantity": 2.0,
            "remaining_quantity": 2.0,
            "entry_reason": "inverse_of:source-1",
        },
    ])
    exchange = FakeExchange(price=90.0)

    await _engine(source_repo, target_repo, exchange).sync_once()

    assert exchange.orders == [{"symbol": "BTC/USDT:USDT", "side": "buy", "quantity": 2.0}]
    mirror = target_repo.trades[0]
    assert mirror["status"] == "closed"
    assert mirror["remaining_quantity"] == 0.0
    assert mirror["pnl_usd"] == pytest.approx(20.0)
    assert mirror["exit_reason"] == "source_closed"


@pytest.mark.asyncio
async def test_inverse_mirror_reduces_position_for_source_partial():
    source = _source_trade(entry_quantity=100.0, remaining_quantity=70.0)
    partial_time = datetime.now(timezone.utc) + timedelta(seconds=1)
    source_repo = FakeRepo("tarakta-mm", trades=[source])
    source_repo.partials["source-1"] = [
        {
            "id": "source-partial-1",
            "trade_id": "source-1",
            "tier": 1,
            "exit_quantity": 30.0,
            "created_at": partial_time.isoformat(),
        },
    ]
    target_repo = FakeRepo("tarakta-mm-inverse", trades=[
        {
            "id": "mirror-1",
            "strategy": "mm_inverse",
            "status": "open",
            "symbol": "BTC/USDT:USDT",
            "direction": "short",
            "entry_price": 100.0,
            "entry_quantity": 100.0,
            "remaining_quantity": 100.0,
            "stop_loss": 110.0,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "entry_reason": "inverse_of:source-1",
        },
    ])
    exchange = FakeExchange(price=110.0)

    await _engine(source_repo, target_repo, exchange).sync_once()

    assert exchange.orders == [{"symbol": "BTC/USDT:USDT", "side": "buy", "quantity": 30.0}]
    mirror = target_repo.trades[0]
    assert mirror["remaining_quantity"] == pytest.approx(70.0)
    assert target_repo.partials["mirror-1"][0]["exit_reason"] == "inverse_partial_of:source-partial-1"
