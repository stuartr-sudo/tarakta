"""Websocket feed + engine fast-stop loop: parsing, staleness, close guard."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from src.strategy.mm_engine import MMEngine
from src.strategy.mm_ws_feed import BinanceWsFeed, normalize_symbol


def _mark(sym: str, price: float) -> dict:
    return {"stream": f"{sym.lower()}@markPrice@1s",
            "data": {"e": "markPriceUpdate", "s": sym, "p": str(price)}}


def _force(sym: str, side: str, qty: float, ap: float) -> dict:
    return {"stream": "!forceOrder@arr",
            "data": {"e": "forceOrder",
                     "o": {"s": sym, "S": side, "q": str(qty), "ap": str(ap)}}}


def test_normalize_symbol():
    assert normalize_symbol("BTC/USDT:USDT") == "btcusdt"
    assert normalize_symbol("BTCUSDT") == "btcusdt"
    assert normalize_symbol("near/USDT:USDT") == "nearusdt"


def test_mark_price_parse_and_age():
    feed = BinanceWsFeed(["BTC/USDT:USDT"])
    feed._handle_message(_mark("BTCUSDT", 60123.5))
    pq = feed.get_price("BTC/USDT:USDT")
    assert pq is not None
    price, age = pq
    assert price == 60123.5
    assert age < 1.0
    assert feed.get_price("ETH/USDT:USDT") is None


def test_bad_payloads_ignored():
    feed = BinanceWsFeed()
    feed._handle_message({})
    feed._handle_message({"data": {"e": "markPriceUpdate", "s": "BTCUSDT", "p": "junk"}})
    feed._handle_message({"data": {"e": "forceOrder", "o": {"s": "", "S": "SELL"}}})
    assert feed._prices == {}
    assert len(feed._liqs) == 0


def test_liquidation_stats_sides_and_window():
    feed = BinanceWsFeed()
    feed._handle_message(_force("BTCUSDT", "SELL", 2.0, 60000))   # longs liquidated
    feed._handle_message(_force("BTCUSDT", "BUY", 1.0, 60000))    # shorts liquidated
    feed._handle_message(_force("ETHUSDT", "SELL", 10.0, 1800))   # other symbol
    stats = feed.liquidation_stats("BTC/USDT:USDT", window_s=300)
    assert stats["events"] == 2
    assert stats["long_liq_usd"] == 120000.0
    assert stats["short_liq_usd"] == 60000.0
    # Age one event out of the window
    sym, side, notional, _ts = feed._liqs[0]
    feed._liqs[0] = (sym, side, notional, time.monotonic() - 1000)
    aged = feed.liquidation_stats("BTC/USDT:USDT", window_s=300)
    assert aged["events"] == 1


def test_stream_url_and_ensure_symbols():
    feed = BinanceWsFeed()
    assert feed._stream_url() is None
    feed.ensure_symbols(["BTC/USDT:USDT", "ETH/USDT:USDT"])
    url = feed._stream_url()
    assert "btcusdt@markPrice@1s" in url
    assert "ethusdt@markPrice@1s" in url
    assert "!forceOrder@arr" in url
    assert feed._resubscribe.is_set()
    feed._resubscribe.clear()
    feed.ensure_symbols(["BTC/USDT:USDT"])  # subset: no change
    assert not feed._resubscribe.is_set()
    feed.ensure_symbols([])  # never drops
    assert len(feed._desired) == 2


def test_backoff_schedule_caps():
    assert BinanceWsFeed._backoff(0) == 1.0
    assert BinanceWsFeed._backoff(2) == 5.0
    assert BinanceWsFeed._backoff(99) == 30.0


def _engine() -> MMEngine:
    return MMEngine(exchange=None, repo=None, candle_manager=None, config=None)


async def test_close_position_reentrancy_guard_simultaneous():
    eng = _engine()
    calls: list[str] = []

    async def slow_inner(pos, price, reason):
        calls.append(reason)
        await asyncio.sleep(0.05)
        eng.positions.pop(pos.symbol, None)

    eng._close_position_inner = slow_inner  # type: ignore[method-assign]
    pos = SimpleNamespace(trade_id="t1", symbol="BTC/USDT:USDT")
    eng.positions = {"BTC/USDT:USDT": pos}
    await asyncio.gather(
        eng._close_position(pos, 100.0, "stop_loss"),
        eng._close_position(pos, 100.0, "stop_loss"),
    )
    assert calls == ["stop_loss"]  # second call bounced off the in-flight guard
    assert "t1" not in eng._closing_trades  # guard released


async def test_close_position_sequential_stale_reference_bounced():
    """Review finding 1: a close arriving AFTER a completed close (stale pos
    held across the manage cycle's awaits) must be a no-op — otherwise it
    fires a duplicate order and the paper exchange opens a phantom reverse
    position."""
    eng = _engine()
    calls: list[str] = []

    async def inner(pos, price, reason):
        calls.append(reason)
        eng.positions.pop(pos.symbol, None)

    eng._close_position_inner = inner  # type: ignore[method-assign]
    pos = SimpleNamespace(trade_id="t1", symbol="BTC/USDT:USDT")
    eng.positions = {"BTC/USDT:USDT": pos}

    await eng._close_position(pos, 100.0, "stop_loss")   # fast-stop close
    assert calls == ["stop_loss"]
    # ...manage cycle resumes later with its stale reference:
    await eng._close_position(pos, 99.0, "stop_loss")
    await eng._close_position(pos, 99.0, "scratch_2h")
    assert calls == ["stop_loss"]  # no second close of any kind


async def test_take_partial_and_mark_exited_bounce_stale_reference():
    """Review finding 2: partial-exit paths share the stale-pos hazard."""
    eng = _engine()
    pos = SimpleNamespace(trade_id="t1", symbol="BTC/USDT:USDT",
                          partial_closed_pct=0.0)
    eng.positions = {}  # already closed by the fast-stop loop
    # Must return before touching the exchange (exchange=None would raise).
    await eng._take_partial(pos, 2, 100.0)
    await eng._mark_fully_exited_after_partial(pos, 100.0, reason="tp_l2",
                                               final_exit_qty=1.0)


async def test_fast_stop_check_triggers_and_respects_staleness():
    eng = _engine()
    closed: list[tuple[str, float, str]] = []

    async def record_close(pos, price, reason):
        closed.append((pos.trade_id, price, reason))
        eng.positions.pop("BTC/USDT:USDT", None)

    eng._close_position = record_close  # type: ignore[method-assign]
    pos = SimpleNamespace(trade_id="t1", direction="long", stop_loss=100.0,
                          symbol="BTC/USDT:USDT")
    eng.positions = {"BTC/USDT:USDT": pos}

    class FakeFeed:
        def __init__(self, price, age):
            self.price, self.age = price, age
        def get_price(self, sym):
            return (self.price, self.age)

    # Stale price → no action even though below SL
    eng._ws_feed = FakeFeed(price=99.0, age=60.0)
    await eng._fast_stop_check_once(max_age=15.0)
    assert closed == []

    # Fresh price above SL → no action
    eng._ws_feed = FakeFeed(price=101.0, age=1.0)
    await eng._fast_stop_check_once(max_age=15.0)
    assert closed == []
    assert eng._last_prices["BTC/USDT:USDT"] == 101.0  # dashboard cache updated

    # Fresh price at/below SL → close fires through the normal path
    eng._ws_feed = FakeFeed(price=99.5, age=1.0)
    await eng._fast_stop_check_once(max_age=15.0)
    assert closed == [("t1", 99.5, "stop_loss")]


async def test_fast_stop_skips_trades_already_closing():
    eng = _engine()
    closed: list[str] = []

    async def record_close(pos, price, reason):
        closed.append(pos.trade_id)

    eng._close_position = record_close  # type: ignore[method-assign]
    pos = SimpleNamespace(trade_id="t1", direction="short", stop_loss=100.0,
                          symbol="ETH/USDT:USDT")
    eng.positions = {"ETH/USDT:USDT": pos}
    eng._closing_trades.add("t1")  # manage cycle mid-close

    class FakeFeed:
        def get_price(self, sym):
            return (101.0, 0.5)  # short stopped out (price >= SL)

    eng._ws_feed = FakeFeed()
    await eng._fast_stop_check_once(max_age=15.0)
    assert closed == []
