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
from unittest.mock import patch, MagicMock

import pandas as pd
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
    # Real providers
    BinanceLiquidationProvider,
    YFinanceCorrelationProvider,
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
        """Default registry: hyblock=True (Binance), correlation=True/False (yfinance),
        rest are False (still stubbed)."""
        registry = DataFeedRegistry()
        status = registry.get_status()

        assert isinstance(status, dict)
        expected_keys = {
            "hyblock", "tradinglite", "news", "options",
            "dominance", "correlation", "sentiment",
        }
        assert set(status.keys()) == expected_keys

        # hyblock is now live via BinanceLiquidationProvider — must be True
        assert status["hyblock"] is True, (
            "hyblock should be True (BinanceLiquidationProvider is the default)"
        )

        # These remain stubs — must be False
        always_stub_keys = ("tradinglite", "news", "options", "dominance", "sentiment")
        for provider_name in always_stub_keys:
            assert status[provider_name] is False, (
                f"Provider '{provider_name}' should be False (stub) but got True"
            )
        # correlation is either YFinance (True) or Stub (False) depending on yfinance install
        assert isinstance(status["correlation"], bool)

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
        # These are always stubs regardless of yfinance availability
        for key in ("tradinglite", "news", "options", "dominance", "sentiment"):
            assert status[key] is False

    def test_stub_default_registry_instantiation(self):
        """Registry can be constructed with no arguments."""
        registry = DataFeedRegistry()
        assert registry.hyblock is not None
        assert registry.news is not None
        assert registry.sentiment is not None

    def test_hyblock_default_is_binance_provider(self):
        """Default hyblock provider is BinanceLiquidationProvider (free, no key)."""
        registry = DataFeedRegistry()
        assert isinstance(registry.hyblock, BinanceLiquidationProvider)

    def test_yfinance_provider_is_default_when_installed(self):
        """Default registry uses YFinanceCorrelationProvider when yfinance is importable."""
        try:
            import yfinance  # noqa: F401
            yfinance_available = True
        except ImportError:
            yfinance_available = False

        registry = DataFeedRegistry()
        if yfinance_available:
            assert isinstance(registry.correlation, YFinanceCorrelationProvider)
        else:
            assert isinstance(registry.correlation, StubCorrelationProvider)


# ---------------------------------------------------------------------------
# YFinanceCorrelationProvider tests (mocked — no real network calls)
# ---------------------------------------------------------------------------


def _make_yf_data(dxy_start: float, dxy_end: float, sp500_start: float, sp500_end: float) -> MagicMock:
    """Build a minimal fake yfinance download result with 14 rows.

    Returns a MagicMock that behaves like yfinance's multi-ticker DataFrame:
    ``data["Close"]`` returns a DataFrame keyed by ticker symbol.
    """
    import numpy as np

    n = 14
    dxy_prices = np.linspace(dxy_start, dxy_end, n)
    sp500_prices = np.linspace(sp500_start, sp500_end, n)
    idx = pd.date_range("2026-04-17 08:00", periods=n, freq="5min", tz="UTC")

    close_df = pd.DataFrame(
        {
            "DX-Y.NYB": dxy_prices,
            "^GSPC": sp500_prices,
            "^IXIC": sp500_prices,  # NASDAQ mirrors S&P in test data
        },
        index=idx,
    )

    # Mock the outer DataFrame: data["Close"] → close_df; data.empty → False; len(data) → n
    mock_data = MagicMock()
    mock_data.empty = False
    mock_data.__len__ = lambda self: n
    mock_data.__getitem__ = lambda self, key: close_df if key == "Close" else pd.DataFrame()
    return mock_data


class TestYFinanceCorrelationProvider:
    """Tests for YFinanceCorrelationProvider using mocked yfinance.download."""

    def _make_provider(self):
        return YFinanceCorrelationProvider()

    def test_dxy_up_implies_btc_down(self):
        """When DXY rises, implied BTC direction should be 'down'."""
        provider = self._make_provider()
        # DXY rises 0.5% (> threshold 0.3%), S&P also falls (aligned with BTC down)
        fake_data = _make_yf_data(
            dxy_start=104.0, dxy_end=104.52,       # +0.5%
            sp500_start=5200.0, sp500_end=5174.0,  # falling (aligned: BTC down)
        )
        with patch("yfinance.download", return_value=fake_data):
            sig = run(provider.fetch_correlation_signal(btc_price_change_pct=0.0))

        assert sig.available is True
        assert sig.dxy_direction == "up"
        assert sig.implied_btc_direction == "down"
        assert sig.sp500_aligned is True

    def test_dxy_down_implies_btc_up(self):
        """When DXY falls, implied BTC direction should be 'up'."""
        provider = self._make_provider()
        fake_data = _make_yf_data(
            dxy_start=104.0, dxy_end=103.48,       # -0.5%
            sp500_start=5200.0, sp500_end=5226.0,  # rising (aligned: BTC up)
        )
        with patch("yfinance.download", return_value=fake_data):
            sig = run(provider.fetch_correlation_signal(btc_price_change_pct=0.0))

        assert sig.dxy_direction == "down"
        assert sig.implied_btc_direction == "up"
        assert sig.sp500_aligned is True

    def test_divergence_detected_when_dxy_moves_and_btc_flat(self):
        """Divergence flag is True when DXY moved >0.3% but BTC barely moved."""
        provider = self._make_provider()
        fake_data = _make_yf_data(
            dxy_start=104.0, dxy_end=104.52,  # +0.5%
            sp500_start=5200.0, sp500_end=5200.0,
        )
        with patch("yfinance.download", return_value=fake_data):
            sig = run(provider.fetch_correlation_signal(btc_price_change_pct=0.05))

        assert sig.dxy_divergence is True

    def test_no_divergence_when_btc_already_moved(self):
        """No divergence when BTC has already caught up (>= 0.1% move)."""
        provider = self._make_provider()
        fake_data = _make_yf_data(
            dxy_start=104.0, dxy_end=104.52,
            sp500_start=5200.0, sp500_end=5200.0,
        )
        with patch("yfinance.download", return_value=fake_data):
            sig = run(provider.fetch_correlation_signal(btc_price_change_pct=0.5))

        assert sig.dxy_divergence is False

    def test_no_divergence_when_dxy_move_below_threshold(self):
        """No divergence flag when DXY move is below 0.3% threshold."""
        provider = self._make_provider()
        fake_data = _make_yf_data(
            dxy_start=104.0, dxy_end=104.1,  # only +0.096%
            sp500_start=5200.0, sp500_end=5200.0,
        )
        with patch("yfinance.download", return_value=fake_data):
            sig = run(provider.fetch_correlation_signal(btc_price_change_pct=0.0))

        assert not sig.dxy_divergence

    def test_confidence_scales_with_dxy_move(self):
        """Confidence increases with DXY move magnitude."""
        provider = self._make_provider()
        # ~1% DXY move → full confidence
        fake_data = _make_yf_data(
            dxy_start=104.0, dxy_end=105.04,
            sp500_start=5200.0, sp500_end=5200.0,
        )
        with patch("yfinance.download", return_value=fake_data):
            sig = run(provider.fetch_correlation_signal())

        assert sig.confidence > 0.5

    def test_sp500_aligned_bonus_raises_confidence(self):
        """sp500_aligned adds 0.2 to confidence (capped at 1.0)."""
        provider = self._make_provider()
        # Without sp500 alignment (flat S&P, not aligned with DXY up → BTC down)
        fake_data_no_align = _make_yf_data(
            dxy_start=104.0, dxy_end=104.52,
            sp500_start=5200.0, sp500_end=5226.0,  # rising = NOT aligned with BTC down
        )
        fake_data_aligned = _make_yf_data(
            dxy_start=104.0, dxy_end=104.52,
            sp500_start=5200.0, sp500_end=5174.0,  # falling = aligned with BTC down
        )
        with patch("yfinance.download", return_value=fake_data_no_align):
            sig_no = run(provider.fetch_correlation_signal())
        provider._cache_time = None  # bust cache for second call
        with patch("yfinance.download", return_value=fake_data_aligned):
            sig_yes = run(provider.fetch_correlation_signal())

        assert sig_yes.sp500_aligned is True
        assert not sig_no.sp500_aligned
        # Aligned confidence should be higher than non-aligned
        assert sig_yes.confidence > sig_no.confidence

    def test_returns_unavailable_on_empty_data(self):
        """Returns CorrelationSignal(available=False) when yfinance returns empty."""
        provider = self._make_provider()
        empty = MagicMock()
        empty.empty = True
        empty.__len__ = lambda self: 0
        with patch("yfinance.download", return_value=empty):
            sig = run(provider.fetch_correlation_signal())

        assert sig.available is False

    def test_returns_unavailable_on_exception(self):
        """Returns CorrelationSignal(available=False) when yfinance raises."""
        provider = self._make_provider()
        with patch("yfinance.download", side_effect=RuntimeError("network error")):
            sig = run(provider.fetch_correlation_signal())

        assert sig.available is False

    def test_cache_is_used_on_second_call(self):
        """Second call within TTL returns cached result without re-calling yfinance."""
        provider = self._make_provider()
        fake_data = _make_yf_data(104.0, 104.52, 5200.0, 5174.0)
        with patch("yfinance.download", return_value=fake_data) as mock_dl:
            run(provider.fetch_correlation_signal())
            run(provider.fetch_correlation_signal())

        # Should only be called once — second call hits cache
        assert mock_dl.call_count == 1

    def test_fetch_correlations_returns_correlation_data(self):
        """fetch_correlations returns CorrelationData with correct alignment."""
        provider = self._make_provider()
        fake_data = _make_yf_data(104.0, 104.52, 5200.0, 5174.0)  # DXY up → BTC down
        with patch("yfinance.download", return_value=fake_data):
            corr = run(provider.fetch_correlations("short"))  # short = aligned with BTC down

        assert corr.available is True
        assert corr.aligns_with_trade_direction is True

    def test_fetch_correlations_misaligned(self):
        """fetch_correlations returns misaligned when direction contradicts DXY signal."""
        provider = self._make_provider()
        fake_data = _make_yf_data(104.0, 104.52, 5200.0, 5174.0)  # DXY up → BTC down
        with patch("yfinance.download", return_value=fake_data):
            corr = run(provider.fetch_correlations("long"))  # long contradicts BTC down

        assert corr.available is True
        assert corr.aligns_with_trade_direction is False


# ---------------------------------------------------------------------------
# BinanceLiquidationProvider tests (mocked httpx — no real network calls)
# ---------------------------------------------------------------------------


def _binance_top_pos_response(long_pct: float, short_pct: float) -> list[dict]:
    """Build a minimal Binance topLongShortPositionRatio response."""
    return [
        {
            "symbol": "BTCUSDT",
            "longAccount": str(long_pct),
            "shortAccount": str(short_pct),
            "longShortRatio": str(long_pct / short_pct) if short_pct else "1",
            "timestamp": 1713340800000,
        }
    ]


class TestBinanceLiquidationProvider:
    """Tests for BinanceLiquidationProvider using mocked httpx."""

    def _make_provider(self) -> BinanceLiquidationProvider:
        return BinanceLiquidationProvider()

    def _mock_client(self, long_pct: float, short_pct: float):
        """Return a context-manager mock for httpx.AsyncClient that returns fake data."""
        from unittest.mock import AsyncMock, MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _binance_top_pos_response(long_pct, short_pct)

        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_resp)

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        return mock_cm

    def test_delta_calculation(self):
        """Delta = long_pct - short_pct. Use 0.70-0.30=0.40 for an unambiguous value."""
        provider = self._make_provider()
        with patch("httpx.AsyncClient", return_value=self._mock_client(0.70, 0.30)):
            data = run(provider.fetch_liquidation_data("BTCUSDT"))

        assert data.available is True
        assert abs(data.delta - 0.40) < 0.001  # 0.70 - 0.30 = 0.40

    def test_delta_level_low(self):
        """Delta < 5% → level='low'."""
        provider = self._make_provider()
        with patch("httpx.AsyncClient", return_value=self._mock_client(0.52, 0.48)):
            data = run(provider.fetch_liquidation_data("BTCUSDT"))

        assert data.delta_level == "low"

    def test_delta_level_medium(self):
        """Delta 5-10% → level='medium' (e.g. 0.57 - 0.43 = 0.14? No — use 0.07 imbalance)."""
        # 0.535 - 0.465 = 0.07 → strictly inside [0.05, 0.10) → medium
        provider = self._make_provider()
        with patch("httpx.AsyncClient", return_value=self._mock_client(0.535, 0.465)):
            data = run(provider.fetch_liquidation_data("BTCUSDT"))

        assert data.delta_level == "medium"

    def test_delta_level_high(self):
        """Delta 10-20% → level='high'. Use 0.57-0.43=0.14 (clearly inside [0.10, 0.20))."""
        provider = self._make_provider()
        # 0.57 - 0.43 = 0.14, which is in [0.10, 0.20) → high
        with patch("httpx.AsyncClient", return_value=self._mock_client(0.57, 0.43)):
            data = run(provider.fetch_liquidation_data("BTCUSDT"))

        assert data.delta_level == "high"

    def test_delta_level_extreme(self):
        """Delta >= 30% → level='extreme'. Use 0.65-0.35=0.30 but via string to avoid FP."""
        provider = self._make_provider()
        # 0.66 - 0.34 = 0.32 (above 0.30 threshold)
        with patch("httpx.AsyncClient", return_value=self._mock_client(0.66, 0.34)):
            data = run(provider.fetch_liquidation_data("BTCUSDT"))

        assert data.delta_level == "extreme"

    def test_cluster_generated_when_high_delta(self):
        """Cluster entry generated when delta >= HIGH threshold (0.20). Use 0.25 imbalance."""
        provider = self._make_provider()
        # 0.625 - 0.375 = 0.25, clearly above 0.20 threshold
        with patch("httpx.AsyncClient", return_value=self._mock_client(0.625, 0.375)):
            data = run(provider.fetch_liquidation_data("BTCUSDT"))

        assert data.available is True
        assert isinstance(data.liquidation_clusters, list)
        assert len(data.liquidation_clusters) == 1
        cluster = data.liquidation_clusters[0]
        assert cluster["direction"] == "longs_exposed"
        assert cluster["imbalance_pct"] == pytest.approx(25.0, abs=0.2)

    def test_cluster_shorts_exposed_when_negative_delta(self):
        """When more shorts than longs → cluster direction = 'shorts_exposed'."""
        provider = self._make_provider()
        # 0.375 - 0.625 = -0.25 (abs = 0.25 >= 0.20)
        with patch("httpx.AsyncClient", return_value=self._mock_client(0.375, 0.625)):
            data = run(provider.fetch_liquidation_data("BTCUSDT"))

        assert data.available is True
        cluster = data.liquidation_clusters[0]
        assert cluster["direction"] == "shorts_exposed"

    def test_no_cluster_when_low_delta(self):
        """No cluster entry when delta is below HIGH threshold."""
        provider = self._make_provider()
        with patch("httpx.AsyncClient", return_value=self._mock_client(0.52, 0.48)):
            data = run(provider.fetch_liquidation_data("BTCUSDT"))

        assert data.available is True
        assert data.liquidation_clusters == []

    def test_symbol_normalisation(self):
        """CCXT-style 'BTC/USDT:USDT' is normalised to 'BTCUSDT' for the API call."""
        provider = self._make_provider()
        mock_cm = self._mock_client(0.55, 0.45)
        with patch("httpx.AsyncClient", return_value=mock_cm) as mock_cls:
            run(provider.fetch_liquidation_data("BTC/USDT:USDT"))

        # Inspect the params passed to .get()
        client_instance = mock_cm.__aenter__.return_value
        call_kwargs = client_instance.get.call_args
        params = call_kwargs[1].get("params") or call_kwargs[0][1] if len(call_kwargs[0]) > 1 else {}
        if not params and call_kwargs.kwargs:
            params = call_kwargs.kwargs.get("params", {})
        assert params.get("symbol") == "BTCUSDT"

    def test_cache_returns_same_result_within_ttl(self):
        """Second call within TTL returns cached result without calling httpx again."""
        provider = self._make_provider()
        mock_cm = self._mock_client(0.57, 0.43)
        with patch("httpx.AsyncClient", return_value=mock_cm) as mock_cls:
            result1 = run(provider.fetch_liquidation_data("BTCUSDT"))
            result2 = run(provider.fetch_liquidation_data("BTCUSDT"))

        # AsyncClient should only be instantiated once — cache hit on second call
        assert mock_cls.call_count == 1
        assert result1.delta == result2.delta

    def test_returns_unavailable_on_non_200(self):
        """Returns HyblockData(available=False) when Binance returns a non-200 status."""
        from unittest.mock import AsyncMock, MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 429  # rate limit
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_resp)
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        provider = self._make_provider()
        with patch("httpx.AsyncClient", return_value=mock_cm):
            data = run(provider.fetch_liquidation_data("BTCUSDT"))

        assert data.available is False

    def test_returns_unavailable_on_network_error(self):
        """Returns HyblockData(available=False) gracefully when httpx raises."""
        from unittest.mock import AsyncMock, MagicMock

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("network error"))
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        provider = self._make_provider()
        with patch("httpx.AsyncClient", return_value=mock_cm):
            data = run(provider.fetch_liquidation_data("BTCUSDT"))

        assert data.available is False

    def test_fetch_liquidations_alias(self):
        """fetch_liquidations() is an alias for fetch_liquidation_data()."""
        provider = self._make_provider()
        with patch("httpx.AsyncClient", return_value=self._mock_client(0.60, 0.40)):
            data = run(provider.fetch_liquidations("BTCUSDT"))

        assert data.available is True
        assert data.delta is not None

    def test_timestamp_is_set(self):
        """Result includes a UTC timestamp."""
        provider = self._make_provider()
        with patch("httpx.AsyncClient", return_value=self._mock_client(0.55, 0.45)):
            data = run(provider.fetch_liquidation_data("BTCUSDT"))

        assert data.timestamp is not None
        assert data.timestamp.tzinfo is not None
