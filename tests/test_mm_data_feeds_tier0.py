"""Tier-0 free feed providers: dominance, sentiment, news, extended ratios."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.strategy.mm_data_feeds import (
    CoinGeckoDominanceProvider,
    DataFeedRegistry,
    FearGreedSentimentProvider,
    RssNewsProvider,
)


def _rss(items: list[tuple[str, datetime]]) -> str:
    rows = "".join(
        f"<item><title>{t}</title>"
        f"<pubDate>{p.strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate></item>"
        for t, p in items
    )
    return f"<rss><channel>{rows}</channel></rss>"


def test_rss_parse_and_keyword_hit():
    p = RssNewsProvider()
    now = datetime.now(timezone.utc)
    items = p._parse_feed(_rss([("Exchange hacked for $100M", now)]))
    assert len(items) == 1
    assert p._keyword_hit(items[0][0])
    assert not p._keyword_hit("Bitcoin price rises modestly")


def test_rss_two_feed_coincidence_required():
    p = RssNewsProvider()
    now = datetime.now(timezone.utc)
    hit = [("Major exchange hacked", now - timedelta(minutes=5))]
    calm = [("Market update: sideways day", now - timedelta(minutes=5))]
    one_feed = p.evaluate([p._parse_feed(_rss(hit)), p._parse_feed(_rss(calm))])
    assert one_feed.available
    assert one_feed.next_high_impact is None  # one outlet = noise

    hit2 = [("Exchange halts withdrawals after exploit", now - timedelta(minutes=10))]
    both = p.evaluate([p._parse_feed(_rss(hit)), p._parse_feed(_rss(hit2))])
    assert both.next_high_impact is not None  # two outlets = event
    assert both.minutes_to_next == 0.0


def test_rss_old_headlines_outside_window_ignored():
    p = RssNewsProvider()
    old = datetime.now(timezone.utc) - timedelta(hours=3)
    result = p.evaluate([
        p._parse_feed(_rss([("Massive hack confirmed", old)])),
        p._parse_feed(_rss([("Second outlet: hack confirmed", old)])),
    ])
    assert result.next_high_impact is None


def test_dominance_trend_cold_start_and_direction():
    p = CoinGeckoDominanceProvider()
    # Cold start: no history yet → trend unknown
    assert p._trend(60.0, 1) == ""
    # Seed a sample slightly over an hour old, then compare
    old = datetime.now(timezone.utc) - timedelta(seconds=3700)
    p._history = [(old, 60.0, 10.0, 5.0, 25.0)]
    assert p._trend(60.5, 1) == "rising"
    assert p._trend(59.5, 1) == "falling"
    assert p._trend(60.05, 1) == "flat"


async def test_fear_greed_parses_canned_payload(monkeypatch):
    class _Resp:
        status_code = 200
        def json(self):
            return {"data": [{"value": "25", "value_classification": "Extreme Fear"}]}

    class _Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None): return _Resp()

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    result = await FearGreedSentimentProvider().fetch_sentiment()
    assert result.available
    assert result.fear_greed_index == 25


def test_registry_defaults_are_real_not_stub():
    status = DataFeedRegistry().get_status()
    assert status["dominance"] is True
    assert status["sentiment"] is True
    assert status["news"] is True
    # untouched stubs stay stubs
    assert status["tradinglite"] is False
    assert status["options"] is False


async def test_flow_snapshot_carries_new_fields(monkeypatch):
    registry = DataFeedRegistry()

    class _Dom:
        async def fetch_dominances(self):
            from src.strategy.mm_data_feeds import DominanceData
            return DominanceData(
                available=True, btc_dominance_pct=61.2,
                btc_dominance_trend="rising", usdt_dominance_pct=4.9,
                usdt_dominance_trend="flat", is_alt_season=False,
            )

    class _Senti:
        async def fetch_sentiment(self):
            from src.strategy.mm_data_feeds import SentimentData
            return SentimentData(available=True, fear_greed_index=25)

    class _News:
        async def fetch_upcoming(self, hours_ahead=72):
            from src.strategy.mm_data_feeds import NewsCalendarData
            return NewsCalendarData(available=True)

    class _Unavailable:
        async def fetch_liquidation_data(self, symbol):
            from src.strategy.mm_data_feeds import HyblockData
            return HyblockData(available=False)
        async def fetch_funding(self, symbol):
            from src.strategy.mm_data_feeds import FundingData
            return FundingData(available=False)
        async def fetch_orderbook(self, symbol):
            from src.strategy.mm_data_feeds import OrderBookData
            return OrderBookData(available=False)

    stub = _Unavailable()
    registry.hyblock = stub
    registry.funding = stub
    registry.orderbook = stub
    registry.dominance = _Dom()
    registry.sentiment = _Senti()
    registry.news = _News()

    snap = await registry.build_flow_snapshot("BTC/USDT:USDT")
    assert snap["available"] is True
    assert snap["btc_dominance_pct"] == 61.2
    assert snap["fear_greed_index"] == 25
    assert snap["news_event_now"] is False
    assert "taker_buy_sell_ratio" in snap
