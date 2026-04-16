"""Tests for MM Engine external data feed interfaces (C1).

Covers:
  1. Each stub provider returns available=False
  2. HyblockData dataclass has all required fields
  3. NewsCalendarData and NewsEvent dataclass fields
  4. OptionsExpiryData dataclass fields
  5. DominanceData dataclass fields
  6. CorrelationData and CorrelationSignal dataclass fields
  7. SentimentData dataclass fields
  8. DataFeedRegistry.get_status() returns correct dict
  9. get_status() returns False for all stubs
  10. get_status() returns True for non-stub providers
"""
from __future__ import annotations

import asyncio
from dataclasses import fields
from datetime import datetime, timezone

import pytest

from src.strategy.mm_data_feeds import (
    # Dataclasses
    LiquidationCluster,
    HyblockData,
    LimitOrderCluster,
    HeatMapData,
    NewsEvent,
    NewsCalendarData,
    OptionsExpiryData,
    DominanceData,
    CorrelationData,
    CorrelationSignal,
    SentimentData,
    # Stub providers
    StubHyblockProvider,
    StubTradingLiteProvider,
    StubNewsProvider,
    StubOptionsProvider,
    StubDominanceProvider,
    StubCorrelationProvider,
    StubSentimentProvider,
    # Registry
    DataFeedRegistry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(coro):
    """Run an async coroutine synchronously in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _field_names(cls) -> set[str]:
    """Return set of dataclass field names for a given class."""
    return {f.name for f in fields(cls)}


# ---------------------------------------------------------------------------
# 1. All stubs return available=False
# ---------------------------------------------------------------------------


class TestStubsReturnUnavailable:
    def test_hyblock_stub_unavailable(self):
        stub = StubHyblockProvider()
        data = run(stub.fetch_liquidations("BTCUSDT"))
        assert isinstance(data, HyblockData)
        assert data.available is False

    def test_tradinglite_stub_unavailable(self):
        stub = StubTradingLiteProvider()
        data = run(stub.fetch_heatmap("BTCUSDT"))
        assert isinstance(data, HeatMapData)
        assert data.available is False

    def test_news_stub_unavailable(self):
        stub = StubNewsProvider()
        data = run(stub.fetch_upcoming(hours_ahead=24))
        assert isinstance(data, NewsCalendarData)
        assert data.available is False

    def test_options_stub_unavailable(self):
        stub = StubOptionsProvider()
        data = run(stub.fetch_next_expiry("BTCUSDT"))
        assert isinstance(data, OptionsExpiryData)
        assert data.available is False

    def test_dominance_stub_unavailable(self):
        stub = StubDominanceProvider()
        data = run(stub.fetch_dominances())
        assert isinstance(data, DominanceData)
        assert data.available is False

    def test_correlation_stub_unavailable(self):
        stub = StubCorrelationProvider()
        data = run(stub.fetch_correlations("long"))
        assert isinstance(data, CorrelationData)
        assert data.available is False

    def test_correlation_signal_stub_zero_confidence(self):
        stub = StubCorrelationProvider()
        sig = run(stub.fetch_correlation_signal())
        assert isinstance(sig, CorrelationSignal)
        assert sig.dxy_divergence is False
        assert sig.confidence == 0.0

    def test_sentiment_stub_unavailable(self):
        stub = StubSentimentProvider()
        data = run(stub.fetch_sentiment())
        assert isinstance(data, SentimentData)
        assert data.available is False


# ---------------------------------------------------------------------------
# 2. HyblockData fields
# ---------------------------------------------------------------------------


class TestHyblockDataFields:
    def test_required_fields_present(self):
        required = {"available", "delta", "delta_level", "liquidation_clusters", "timestamp"}
        assert required.issubset(_field_names(HyblockData))

    def test_default_values(self):
        data = HyblockData()
        assert data.available is False
        assert data.delta is None
        assert data.delta_level is None
        assert data.liquidation_clusters is None
        assert data.timestamp is None

    def test_populated_instance(self):
        cluster = LiquidationCluster(
            price=50000.0,
            amount=1_000_000.0,
            leverage="100x",
            direction="long_liq",
        )
        data = HyblockData(
            available=True,
            delta=0.35,
            delta_level="high",
            liquidation_clusters=[cluster],
            timestamp=datetime.now(tz=timezone.utc),
        )
        assert data.available is True
        assert data.delta == 0.35
        assert data.delta_level == "high"
        assert len(data.liquidation_clusters) == 1
        assert data.liquidation_clusters[0].leverage == "100x"
        assert data.timestamp is not None

    def test_liquidation_cluster_direction_values(self):
        long_liq = LiquidationCluster(50000.0, 500_000.0, "50x", "long_liq")
        short_liq = LiquidationCluster(48000.0, 200_000.0, "25x", "short_liq")
        assert long_liq.direction == "long_liq"
        assert short_liq.direction == "short_liq"


# ---------------------------------------------------------------------------
# 3. NewsCalendarData and NewsEvent fields
# ---------------------------------------------------------------------------


class TestNewsCalendarDataFields:
    def test_news_event_required_fields(self):
        required = {"title", "currency", "impact", "forecast", "previous", "time"}
        assert required.issubset(_field_names(NewsEvent))

    def test_news_calendar_data_required_fields(self):
        required = {"available", "upcoming_events", "next_high_impact", "minutes_to_next"}
        assert required.issubset(_field_names(NewsCalendarData))

    def test_news_event_construction(self):
        now = datetime.now(tz=timezone.utc)
        event = NewsEvent(
            title="Non-Farm Payrolls",
            currency="USD",
            impact="red",
            forecast="180K",
            previous="177K",
            time=now,
        )
        assert event.title == "Non-Farm Payrolls"
        assert event.currency == "USD"
        assert event.impact == "red"
        assert event.forecast == "180K"
        assert event.previous == "177K"
        assert event.time == now

    def test_news_event_optional_fields_can_be_none(self):
        event = NewsEvent(
            title="FOMC Meeting Minutes",
            currency="USD",
            impact="red",
            forecast=None,
            previous=None,
            time=datetime.now(tz=timezone.utc),
        )
        assert event.forecast is None
        assert event.previous is None

    def test_news_calendar_data_defaults(self):
        data = NewsCalendarData()
        assert data.available is False
        assert data.upcoming_events == []
        assert data.next_high_impact is None
        assert data.minutes_to_next is None

    def test_news_calendar_data_with_events(self):
        now = datetime.now(tz=timezone.utc)
        event = NewsEvent(
            title="CPI", currency="USD", impact="red",
            forecast="3.1%", previous="3.2%", time=now,
        )
        data = NewsCalendarData(
            available=True,
            upcoming_events=[event],
            next_high_impact=event,
            minutes_to_next=45.0,
        )
        assert data.available is True
        assert len(data.upcoming_events) == 1
        assert data.next_high_impact.title == "CPI"
        assert data.minutes_to_next == 45.0

    def test_impact_values(self):
        """Verify impact field accepts the three course-defined levels."""
        for impact in ("red", "orange", "yellow"):
            event = NewsEvent(
                title="test", currency="USD", impact=impact,
                forecast=None, previous=None,
                time=datetime.now(tz=timezone.utc),
            )
            assert event.impact == impact


# ---------------------------------------------------------------------------
# 4. OptionsExpiryData fields
# ---------------------------------------------------------------------------


class TestOptionsExpiryDataFields:
    def test_required_fields_present(self):
        required = {
            "available", "next_expiry_date", "is_quad_witching",
            "max_pain_price", "total_notional_usd",
            "calls_notional", "puts_notional", "put_call_ratio",
        }
        assert required.issubset(_field_names(OptionsExpiryData))

    def test_default_values(self):
        data = OptionsExpiryData()
        assert data.available is False
        assert data.next_expiry_date is None
        assert data.is_quad_witching is False
        assert data.max_pain_price is None
        assert data.total_notional_usd == 0.0
        assert data.put_call_ratio is None

    def test_put_call_ratio_calculation(self):
        data = OptionsExpiryData(
            available=True,
            calls_notional=1_000_000.0,
            puts_notional=1_500_000.0,
            put_call_ratio=1.5,
        )
        assert data.put_call_ratio == 1.5
        assert data.puts_notional > data.calls_notional


# ---------------------------------------------------------------------------
# 5. DominanceData fields
# ---------------------------------------------------------------------------


class TestDominanceDataFields:
    def test_required_fields_present(self):
        required = {
            "available",
            "btc_dominance_pct", "btc_dominance_trend",
            "eth_dominance_pct", "eth_dominance_trend",
            "usdt_dominance_pct", "usdt_dominance_trend",
            "is_alt_season", "is_degen_season",
        }
        assert required.issubset(_field_names(DominanceData))

    def test_default_values(self):
        data = DominanceData()
        assert data.available is False
        assert data.btc_dominance_pct == 0.0
        assert data.is_alt_season is False
        assert data.is_degen_season is False

    def test_trend_field_values(self):
        data = DominanceData(
            available=True,
            btc_dominance_pct=54.3,
            btc_dominance_trend="falling",
            eth_dominance_pct=17.2,
            eth_dominance_trend="rising",
            usdt_dominance_pct=7.1,
            usdt_dominance_trend="falling",
            is_alt_season=True,
        )
        assert data.is_alt_season is True
        assert data.btc_dominance_trend == "falling"


# ---------------------------------------------------------------------------
# 6. CorrelationData and CorrelationSignal fields
# ---------------------------------------------------------------------------


class TestCorrelationFields:
    def test_correlation_data_fields(self):
        required = {
            "available", "btc_dxy_correlation",
            "btc_nasdaq_correlation", "aligns_with_trade_direction",
        }
        assert required.issubset(_field_names(CorrelationData))

    def test_correlation_signal_fields(self):
        required = {
            "dxy_divergence", "dxy_direction",
            "implied_btc_direction", "sp500_aligned", "confidence",
        }
        assert required.issubset(_field_names(CorrelationSignal))

    def test_correlation_signal_construction(self):
        sig = CorrelationSignal(
            dxy_divergence=True,
            dxy_direction="up",
            implied_btc_direction="down",
            sp500_aligned=True,
            confidence=0.8,
        )
        assert sig.dxy_divergence is True
        assert sig.implied_btc_direction == "down"
        assert sig.confidence == 0.8

    def test_correlation_data_defaults(self):
        data = CorrelationData()
        assert data.available is False
        assert data.btc_dxy_correlation == 0.0
        assert data.aligns_with_trade_direction is False


# ---------------------------------------------------------------------------
# 7. SentimentData fields
# ---------------------------------------------------------------------------


class TestSentimentDataFields:
    def test_required_fields_present(self):
        required = {"available", "fear_greed_index", "augmento_score"}
        assert required.issubset(_field_names(SentimentData))

    def test_defaults(self):
        data = SentimentData()
        assert data.available is False
        assert data.fear_greed_index is None
        assert data.augmento_score is None

    def test_extreme_fear_value(self):
        data = SentimentData(available=True, fear_greed_index=10)
        assert data.fear_greed_index == 10  # Extreme Fear range

    def test_extreme_greed_value(self):
        data = SentimentData(available=True, fear_greed_index=90)
        assert data.fear_greed_index == 90  # Extreme Greed range


# ---------------------------------------------------------------------------
# 8 & 9. DataFeedRegistry get_status()
# ---------------------------------------------------------------------------


class TestDataFeedRegistryStatus:
    def test_all_stubs_return_false_status(self):
        registry = DataFeedRegistry()
        status = registry.get_status()

        assert isinstance(status, dict)
        expected_keys = {
            "hyblock", "tradinglite", "news", "options",
            "dominance", "correlation", "sentiment",
        }
        assert set(status.keys()) == expected_keys

        for provider_name, is_available in status.items():
            assert is_available is False, (
                f"Provider '{provider_name}' should be False (stub) but got True"
            )

    def test_status_has_all_seven_providers(self):
        registry = DataFeedRegistry()
        status = registry.get_status()
        assert len(status) == 7

    def test_status_values_are_booleans(self):
        registry = DataFeedRegistry()
        status = registry.get_status()
        for key, val in status.items():
            assert isinstance(val, bool), f"Status for '{key}' should be bool, got {type(val)}"

    def test_custom_provider_shows_as_available(self):
        """A non-stub provider class should return True in get_status()."""

        class RealHyblockProvider:
            """Minimal fake real provider (not named Stub*)."""
            async def fetch_liquidations(self, symbol: str) -> HyblockData:
                return HyblockData(available=True)

        registry = DataFeedRegistry(hyblock=RealHyblockProvider())
        status = registry.get_status()

        assert status["hyblock"] is True
        # All others still False (stubbed)
        for key in ("tradinglite", "news", "options", "dominance", "correlation", "sentiment"):
            assert status[key] is False

    def test_stub_default_registry_instantiation(self):
        """Registry can be constructed with no arguments."""
        registry = DataFeedRegistry()
        assert registry.hyblock is not None
        assert registry.news is not None
        assert registry.sentiment is not None
