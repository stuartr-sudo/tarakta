"""MM Method external data feed interfaces (lessons 25, 26, 27, 29-32).

Every factor the course teaches is represented here as an interface. When
credentials / subscriptions exist, drop in a real implementation of the
relevant `Provider`. Until then each provider returns the neutral/empty
value and the confluence scorer naturally scores that factor as 0.

Providers covered (all currently UNWIRED — awaiting API access):
  - `HyblockProvider`       — liquidation-level clusters (lesson 25, 27)
  - `TradingLiteProvider`   — limit-order heat map (lesson 25, 26)
  - `NewsProvider`          — Forex Factory calendar (lesson 32)
  - `OptionsProvider`       — options expiry / Max Pain / P-C ratio (lesson 33)
  - `DominanceProvider`     — BTC.D / ETH.D / USDT.D (lesson 31)
  - `CorrelationProvider`   — BTC vs DXY / NASDAQ (lesson 30)
  - `SentimentProvider`     — Fear & Greed / augmento.ai (lesson 32)

Each returns a dataclass with `.available: bool` so callers can cheaply
detect "no data yet" and leave the corresponding context flag False.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


# ---------------------------------------------------------------------------
# Hyblock (liquidation levels)
# ---------------------------------------------------------------------------


@dataclass
class LiquidationCluster:
    """A single liquidation cluster (price level + estimated leverage)."""
    price: float
    size_usd: float
    leverage_bucket: str  # "25x", "50x", "100x" per course lesson 27
    direction: str        # "long_liq" or "short_liq"


@dataclass
class LiquidationData:
    available: bool = False
    delta_usd: float = 0.0             # Long - Short liquidation totals ($)
    total_positions: float = 0.0       # Raw total (want >1000 per lesson 27)
    nearest_cluster_above: LiquidationCluster | None = None
    nearest_cluster_below: LiquidationCluster | None = None


class HyblockProvider(Protocol):
    async def fetch_liquidations(self, symbol: str) -> LiquidationData: ...


class StubHyblockProvider:
    """Returns `available=False` until real credentials are provided."""
    async def fetch_liquidations(self, symbol: str) -> LiquidationData:
        return LiquidationData(available=False)


# ---------------------------------------------------------------------------
# TradingLite (heat map / limit-order clusters)
# ---------------------------------------------------------------------------


@dataclass
class LimitOrderCluster:
    price: float
    size_usd: float
    side: str  # "bid" or "ask"


@dataclass
class HeatMapData:
    available: bool = False
    largest_bid_cluster: LimitOrderCluster | None = None
    largest_ask_cluster: LimitOrderCluster | None = None


class TradingLiteProvider(Protocol):
    async def fetch_heatmap(self, symbol: str) -> HeatMapData: ...


class StubTradingLiteProvider:
    async def fetch_heatmap(self, symbol: str) -> HeatMapData:
        return HeatMapData(available=False)


# ---------------------------------------------------------------------------
# News (Forex Factory style calendar)
# ---------------------------------------------------------------------------


@dataclass
class NewsEvent:
    timestamp: datetime
    currency: str          # "USD", "EUR", "GBP" etc
    impact: str            # "red", "orange", "yellow"
    title: str


@dataclass
class NewsData:
    available: bool = False
    upcoming: list[NewsEvent] = field(default_factory=list)
    next_red_within_hours: float | None = None  # Hours until next red event


class NewsProvider(Protocol):
    async def fetch_upcoming(self, hours_ahead: float = 72) -> NewsData: ...


class StubNewsProvider:
    async def fetch_upcoming(self, hours_ahead: float = 72) -> NewsData:
        return NewsData(available=False)


# ---------------------------------------------------------------------------
# Options expiry (basedmoney.io style)
# ---------------------------------------------------------------------------


@dataclass
class OptionsExpiryData:
    available: bool = False
    next_expiry_date: datetime | None = None
    is_quad_witching: bool = False  # 3rd Friday of Mar/Jun/Sep/Dec
    max_pain_price: float | None = None
    total_notional_usd: float = 0.0
    calls_notional: float = 0.0
    puts_notional: float = 0.0
    put_call_ratio: float | None = None


class OptionsProvider(Protocol):
    async def fetch_next_expiry(self, symbol: str) -> OptionsExpiryData: ...


class StubOptionsProvider:
    async def fetch_next_expiry(self, symbol: str) -> OptionsExpiryData:
        return OptionsExpiryData(available=False)


# ---------------------------------------------------------------------------
# Dominance (BTC.D / ETH.D / USDT.D)
# ---------------------------------------------------------------------------


@dataclass
class DominanceData:
    available: bool = False
    btc_dominance_pct: float = 0.0
    btc_dominance_trend: str = ""  # "rising" | "falling" | "flat"
    eth_dominance_pct: float = 0.0
    eth_dominance_trend: str = ""
    usdt_dominance_pct: float = 0.0
    usdt_dominance_trend: str = ""
    is_alt_season: bool = False  # BTC.D falling + USDT.D falling + ETH.D rising
    is_degen_season: bool = False  # TOTAL3 rising


class DominanceProvider(Protocol):
    async def fetch_dominances(self) -> DominanceData: ...


class StubDominanceProvider:
    async def fetch_dominances(self) -> DominanceData:
        return DominanceData(available=False)


# ---------------------------------------------------------------------------
# Correlation (BTC vs DXY / NASDAQ)
# ---------------------------------------------------------------------------


@dataclass
class CorrelationData:
    available: bool = False
    btc_dxy_correlation: float = 0.0  # -1..+1 (negative expected)
    btc_nasdaq_correlation: float = 0.0  # -1..+1 (positive expected)
    # For confluence: alignment flag used by the scorer
    aligns_with_trade_direction: bool = False


@dataclass
class CorrelationSignal:
    """Pre-positioning signal from macro correlation divergence (Lesson D9 / 19).

    Course teaches (Lesson 19):
      - DXY up = BTC down (inverse correlation).
      - S&P / NASDAQ correlated with BTC (positive).
      - When DXY moves but BTC hasn't reacted yet → position before BTC catches up.

    Fields:
        dxy_divergence: True when DXY has moved significantly but BTC hasn't
            reacted yet (divergence = pre-positioning opportunity).
        dxy_direction: Direction DXY has moved: "up" or "down".
        implied_btc_direction: Expected BTC direction = opposite of DXY direction.
            ("up" if DXY went down; "down" if DXY went up)
        sp500_aligned: True when S&P 500 / NASDAQ move confirms the implied BTC
            direction (additional confluence).
        confidence: 0.0-1.0 signal confidence. 0.0 = no signal / not available.
    """
    dxy_divergence: bool = False
    dxy_direction: str = ""          # "up" or "down"
    implied_btc_direction: str = ""  # "up" (BTC) or "down" (BTC) — opposite of DXY
    sp500_aligned: bool = False
    confidence: float = 0.0


class CorrelationProvider(Protocol):
    async def fetch_correlations(self, direction: str) -> CorrelationData: ...
    async def fetch_correlation_signal(self) -> "CorrelationSignal": ...


class StubCorrelationProvider:
    async def fetch_correlations(self, direction: str) -> CorrelationData:
        return CorrelationData(available=False)

    async def fetch_correlation_signal(self) -> CorrelationSignal:
        """Returns a zero-confidence signal when no real provider is wired."""
        return CorrelationSignal(dxy_divergence=False, confidence=0.0)


# ---------------------------------------------------------------------------
# Sentiment (Fear & Greed / augmento.ai)
# ---------------------------------------------------------------------------


@dataclass
class SentimentData:
    available: bool = False
    fear_greed_index: int | None = None  # 0..100
    augmento_score: float | None = None  # 0..1 bull/bear


class SentimentProvider(Protocol):
    async def fetch_sentiment(self) -> SentimentData: ...


class StubSentimentProvider:
    async def fetch_sentiment(self) -> SentimentData:
        return SentimentData(available=False)


# ---------------------------------------------------------------------------
# Default registry — swap stubs for real providers when credentials arrive
# ---------------------------------------------------------------------------


@dataclass
class DataFeedRegistry:
    """One-stop registry the MMEngine can call. Defaults to stubs.

    To wire a real provider, instantiate this with your implementation:

        registry = DataFeedRegistry(
            hyblock=MyRealHyblockProvider(api_key=...),
            news=MyForexFactoryScraper(),
            # others still stubbed
        )
    """
    hyblock: HyblockProvider = field(default_factory=StubHyblockProvider)
    tradinglite: TradingLiteProvider = field(default_factory=StubTradingLiteProvider)
    news: NewsProvider = field(default_factory=StubNewsProvider)
    options: OptionsProvider = field(default_factory=StubOptionsProvider)
    dominance: DominanceProvider = field(default_factory=StubDominanceProvider)
    correlation: CorrelationProvider = field(default_factory=StubCorrelationProvider)
    sentiment: SentimentProvider = field(default_factory=StubSentimentProvider)
