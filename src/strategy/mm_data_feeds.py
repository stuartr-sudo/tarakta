"""MM Method external data feed interfaces (lessons 25, 26, 27, 29-32).

Every factor the course teaches is represented here as an interface. When
credentials / subscriptions exist, drop in a real implementation of the
relevant ``Provider``. Until then each provider returns the neutral/empty
value and the confluence scorer naturally scores that factor as 0.

Providers covered (all currently UNWIRED — awaiting API access):
  - ``HyblockProvider``       — liquidation-level clusters + delta (lesson 25, 27)
  - ``TradingLiteProvider``   — limit-order heat map (lesson 25, 26)
  - ``NewsProvider``          — Forex Factory calendar (lesson 32)
  - ``OptionsProvider``       — options expiry / Max Pain / P-C ratio (lesson 33)
  - ``DominanceProvider``     — BTC.D / ETH.D / USDT.D (lesson 31)
  - ``CorrelationProvider``   — BTC vs DXY / NASDAQ (lesson 30)
  - ``SentimentProvider``     — Fear & Greed / augmento.ai (lesson 32)

Each returns a dataclass with ``.available: bool`` so callers can cheaply
detect "no data yet" and leave the corresponding context flag False.

Real implementations should be registered in ``DataFeedRegistry``::

    from my_providers import HyblockClient, ForexFactoryScraper

    registry = DataFeedRegistry(
        hyblock=HyblockClient(api_key="..."),
        news=ForexFactoryScraper(),
    )

Call ``registry.get_status()`` to see which providers are live at runtime.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyblock (liquidation levels + delta)
# ---------------------------------------------------------------------------


@dataclass
class LiquidationCluster:
    """A single liquidation cluster at a specific price level.

    Course lesson 27 teaches that clusters at 25×, 50×, and 100× leverage
    act as magnet levels — price is drawn toward them before reversing.
    """
    price: float
    amount: float           # Estimated USD notional of liquidations
    leverage: str           # "25x" | "50x" | "100x" per course lesson 27
    direction: str          # "long_liq" | "short_liq"


@dataclass
class HyblockData:
    """Hyblock liquidation and delta data.

    API required: Hyblock Capital (hyblock.capital) — paid subscription.
    Endpoint: ``/api/v2/public/liquidation-levels`` (symbol, timeframe).

    Fields:
        available: False until real credentials are wired.
        delta: Long-vs-short exposure (positive = more longs exposed to
            liquidation; negative = more shorts exposed).
        delta_level: Qualitative delta reading: "low" | "medium" | "high"
            | "extreme". "Extreme" suggests a likely squeeze reversal.
        liquidation_clusters: Ordered list of price clusters where mass
            liquidations would occur. Each has a price, USD amount, and
            leverage bucket. None when not available.
        timestamp: When the data was fetched (for staleness checks).
    """
    available: bool = False
    delta: float | None = None          # Positive = more longs exposed
    delta_level: str | None = None      # "low" | "medium" | "high" | "extreme"
    liquidation_clusters: list[LiquidationCluster] | None = None
    timestamp: datetime | None = None


class HyblockProvider(Protocol):
    """Protocol for Hyblock liquidation data providers.

    Real implementations must:
      1. Authenticate via Hyblock Capital API (hyblock.capital).
      2. Fetch ``/api/v2/public/liquidation-levels`` for the given symbol.
      3. Compute delta from long_liq vs short_liq totals.
      4. Map delta magnitude to delta_level thresholds.
    """
    async def fetch_liquidations(self, symbol: str) -> HyblockData: ...


class StubHyblockProvider:
    """Returns ``available=False`` until real Hyblock credentials are provided.

    To use real data: implement ``HyblockProvider`` and register it in
    ``DataFeedRegistry(hyblock=YourRealProvider())``.
    """
    async def fetch_liquidations(self, symbol: str) -> HyblockData:
        return HyblockData(available=False)


# ---------------------------------------------------------------------------
# TradingLite (heat map / limit-order clusters)
# ---------------------------------------------------------------------------


@dataclass
class LimitOrderCluster:
    """A cluster of large limit orders at a specific price level.

    Course lesson 26 teaches that large visible limit orders on the heat map
    act as support/resistance — price tends to react at these levels.
    """
    price: float
    size_usd: float     # Estimated USD notional of the clustered orders
    side: str           # "bid" (buy-side) or "ask" (sell-side)


@dataclass
class HeatMapData:
    """TradingLite order-flow heat-map data.

    API required: TradingLite (tradinglite.com) — paid subscription.
    Provides real-time large limit-order cluster visibility.

    Fields:
        available: False until real credentials are wired.
        largest_bid_cluster: Strongest buy-side limit cluster detected.
        largest_ask_cluster: Strongest sell-side limit cluster detected.
    """
    available: bool = False
    largest_bid_cluster: LimitOrderCluster | None = None
    largest_ask_cluster: LimitOrderCluster | None = None


class TradingLiteProvider(Protocol):
    """Protocol for TradingLite heat-map providers.

    Real implementations must connect to TradingLite's WebSocket stream or
    REST API and parse the limit-order cluster data.
    """
    async def fetch_heatmap(self, symbol: str) -> HeatMapData: ...


class StubTradingLiteProvider:
    """Returns ``available=False`` until real TradingLite credentials are provided."""
    async def fetch_heatmap(self, symbol: str) -> HeatMapData:
        return HeatMapData(available=False)


# ---------------------------------------------------------------------------
# News Calendar (Forex Factory style)
# ---------------------------------------------------------------------------


@dataclass
class NewsEvent:
    """A single economic calendar event.

    Course lesson 32 teaches that red/orange events are "no-trade" windows —
    avoid entering within 15 minutes either side of them.

    Fields:
        title: Event description (e.g. "Non-Farm Payrolls", "CPI", "FOMC").
        currency: Affected currency code (e.g. "USD", "EUR", "GBP").
        impact: Forex Factory colour coding:
            - "red"    → high impact (must avoid)
            - "orange" → medium impact (caution)
            - "yellow" → low impact (informational)
        forecast: Analyst consensus estimate, or None if not published.
        previous: Previous reading, or None if unavailable.
        time: UTC datetime of the event.
    """
    title: str
    currency: str
    impact: str          # "red" | "orange" | "yellow"
    forecast: str | None
    previous: str | None
    time: datetime


@dataclass
class NewsCalendarData:
    """Forex Factory (or equivalent) news calendar data.

    API required: Forex Factory (forexfactory.com) web scrape or a paid
    calendar API (e.g. Tradays, MyfxBook economic calendar API).

    Fields:
        available: False until real data source is wired.
        upcoming_events: All events in the next N hours, sorted ascending
            by time.
        next_high_impact: The soonest red-impact event, or None if none
            scheduled in the look-ahead window.
        minutes_to_next: Minutes until ``next_high_impact``, or None.
    """
    available: bool = False
    upcoming_events: list[NewsEvent] = field(default_factory=list)
    next_high_impact: NewsEvent | None = None
    minutes_to_next: float | None = None


class NewsProvider(Protocol):
    """Protocol for economic calendar providers.

    Real implementations must:
      1. Fetch upcoming events for the next ``hours_ahead`` hours.
      2. Filter and sort by time.
      3. Identify the soonest red/high-impact event.
      4. Compute ``minutes_to_next`` from ``datetime.utcnow()``.

    Recommended sources: forexfactory.com scrape, Tradays API, or MyfxBook.
    """
    async def fetch_upcoming(self, hours_ahead: float = 72) -> NewsCalendarData: ...


class StubNewsProvider:
    """Returns ``available=False`` until a real calendar provider is wired."""
    async def fetch_upcoming(self, hours_ahead: float = 72) -> NewsCalendarData:
        return NewsCalendarData(available=False)


# ---------------------------------------------------------------------------
# Options expiry (basedmoney.io / Deribit style)
# ---------------------------------------------------------------------------


@dataclass
class OptionsExpiryData:
    """Options market data for expiry-driven price targets.

    API required: Deribit public API (deribit.com/api/v2) — free for BTC/ETH.
    For altcoins: basedmoney.io or similar options analytics.

    Course lesson 33 teaches:
      - Max Pain price = where options sellers (MMs) profit most → price magnet.
      - Quad-witching Fridays (3rd Friday of Mar/Jun/Sep/Dec) = high volatility.
      - High put-call ratio → bearish sentiment (many puts purchased).

    Fields:
        available: False until real data source is wired.
        next_expiry_date: Next options expiry datetime (UTC).
        is_quad_witching: True on the 3rd Friday of Mar/Jun/Sep/Dec.
        max_pain_price: Price level where most options expire worthless
            (combined P&L minimised for option holders).
        total_notional_usd: Total open interest in USD.
        calls_notional: USD value of open call options.
        puts_notional: USD value of open put options.
        put_call_ratio: puts / calls ratio. > 1.0 = more puts (bearish).
    """
    available: bool = False
    next_expiry_date: datetime | None = None
    is_quad_witching: bool = False
    max_pain_price: float | None = None
    total_notional_usd: float = 0.0
    calls_notional: float = 0.0
    puts_notional: float = 0.0
    put_call_ratio: float | None = None


class OptionsProvider(Protocol):
    """Protocol for options data providers."""
    async def fetch_next_expiry(self, symbol: str) -> OptionsExpiryData: ...


class StubOptionsProvider:
    """Returns ``available=False`` until a real options provider is wired."""
    async def fetch_next_expiry(self, symbol: str) -> OptionsExpiryData:
        return OptionsExpiryData(available=False)


# ---------------------------------------------------------------------------
# Dominance (BTC.D / ETH.D / USDT.D)
# ---------------------------------------------------------------------------


@dataclass
class DominanceData:
    """Crypto market dominance data.

    API required: CoinMarketCap or CoinGecko global market data endpoint.
    Free tiers available for both.

    Course lesson 31 teaches:
      - BTC.D rising → capital flowing into BTC, away from alts (risk-off alts).
      - BTC.D falling + USDT.D falling + ETH.D rising → alt season.
      - USDT.D rising → market selling into stables (risk-off entire market).
      - TOTAL3 (all ex-BTC+ETH) rising → "degen season" for small caps.

    Fields:
        available: False until real data source is wired.
        btc_dominance_pct: BTC market cap as % of total crypto market cap.
        btc_dominance_trend: "rising" | "falling" | "flat"
        eth_dominance_pct: ETH market cap as % of total.
        eth_dominance_trend: "rising" | "falling" | "flat"
        usdt_dominance_pct: USDT market cap as % of total (stablecoin dominance).
        usdt_dominance_trend: "rising" | "falling" | "flat"
        is_alt_season: True when BTC.D falling + USDT.D falling + ETH.D rising.
        is_degen_season: True when TOTAL3 (small-cap alts) is rising.
    """
    available: bool = False
    btc_dominance_pct: float = 0.0
    btc_dominance_trend: str = ""  # "rising" | "falling" | "flat"
    eth_dominance_pct: float = 0.0
    eth_dominance_trend: str = ""
    usdt_dominance_pct: float = 0.0
    usdt_dominance_trend: str = ""
    is_alt_season: bool = False
    is_degen_season: bool = False


class DominanceProvider(Protocol):
    """Protocol for crypto dominance data providers."""
    async def fetch_dominances(self) -> DominanceData: ...


class StubDominanceProvider:
    """Returns ``available=False`` until a real dominance provider is wired."""
    async def fetch_dominances(self) -> DominanceData:
        return DominanceData(available=False)


# ---------------------------------------------------------------------------
# Correlation (BTC vs DXY / NASDAQ)
# ---------------------------------------------------------------------------


@dataclass
class CorrelationData:
    """BTC macro correlation data (DXY and NASDAQ/S&P 500).

    API required: Any financial data API with DXY and SPX/NASDAQ tickers
    (e.g. Alpha Vantage, Yahoo Finance unofficial API, TradingEconomics).

    Fields:
        available: False until real data source is wired.
        btc_dxy_correlation: Pearson correlation coefficient BTC vs DXY
            over last N days. Expected range: -1 to +1. Typically negative
            (DXY rises → BTC falls).
        btc_nasdaq_correlation: Pearson correlation BTC vs NASDAQ.
            Typically positive (risk-on assets move together).
        aligns_with_trade_direction: Pre-computed alignment flag for the
            confluence scorer. True when macro correlation confirms the
            intended trade direction.
    """
    available: bool = False
    btc_dxy_correlation: float = 0.0      # -1..+1 (negative expected)
    btc_nasdaq_correlation: float = 0.0   # -1..+1 (positive expected)
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
        implied_btc_direction: Expected BTC direction — opposite of DXY direction.
            ("up" if DXY went down; "down" if DXY went up)
        sp500_aligned: True when S&P 500 / NASDAQ move confirms the implied BTC
            direction (additional confluence).
        confidence: 0.0-1.0 signal confidence. 0.0 = no signal / not available.
    """
    available: bool = True             # False when data fetch failed
    dxy_divergence: bool = False
    dxy_direction: str = ""           # "up" or "down"
    implied_btc_direction: str = ""   # "up" (BTC) or "down" (BTC) — opposite of DXY
    sp500_aligned: bool = False
    confidence: float = 0.0


class CorrelationProvider(Protocol):
    """Protocol for macro correlation data providers."""
    async def fetch_correlations(self, direction: str) -> CorrelationData: ...
    async def fetch_correlation_signal(self) -> "CorrelationSignal": ...


class StubCorrelationProvider:
    """Returns ``available=False`` and zero-confidence signals until a real
    correlation provider is wired.

    API required: DXY and SPX/NASDAQ price history (e.g. Alpha Vantage free
    tier, Yahoo Finance, or TradingEconomics paid API).
    """
    async def fetch_correlations(self, direction: str) -> CorrelationData:
        return CorrelationData(available=False)

    async def fetch_correlation_signal(self) -> CorrelationSignal:
        return CorrelationSignal(available=False, dxy_divergence=False, confidence=0.0)


class YFinanceCorrelationProvider:
    """Real correlation provider using free yfinance data.

    Course Lesson 19: DXY (US Dollar Index) inverse-correlates with BTC.
    S&P 500 and NASDAQ generally correlate WITH BTC. When DXY moves but
    BTC hasn't yet → position before BTC catches up.

    Uses yfinance (no API key required — free Yahoo Finance data).
    5-minute bars over the last trading day, cached for 5 minutes to
    avoid hammering the upstream service.
    """

    SYMBOLS = {"dxy": "DX-Y.NYB", "sp500": "^GSPC", "nasdaq": "^IXIC"}
    DIVERGENCE_THRESHOLD_PCT = 0.3  # 0.3% move in DXY without BTC reaction
    CACHE_TTL_SECONDS = 300  # 5-min cache to avoid hammering yfinance

    def __init__(self) -> None:
        self._cache: dict = {}
        self._cache_time: datetime | None = None

    async def fetch_correlations(self, direction: str) -> CorrelationData:
        """Return basic correlation data (direction alignment).

        For full pre-positioning signal use ``fetch_correlation_signal``.
        """
        signal = await self.fetch_correlation_signal()
        if not signal.confidence:
            return CorrelationData(available=False)
        aligned = (
            (direction == "long" and signal.implied_btc_direction == "up")
            or (direction == "short" and signal.implied_btc_direction == "down")
        )
        return CorrelationData(
            available=True,
            aligns_with_trade_direction=aligned,
        )

    async def fetch_correlation_signal(self, btc_price_change_pct: float = 0.0) -> CorrelationSignal:
        """Fetch DXY/SPX/NDX recent moves and detect divergence from BTC.

        Returns a CorrelationSignal with:
        - dxy_divergence: True if DXY moved >0.3% but BTC moved <0.1%
        - dxy_direction: "up" or "down"
        - implied_btc_direction: opposite of dxy
        - sp500_aligned: True if S&P moved same direction as implied BTC
        - confidence: 0-1
        """
        from datetime import timezone

        # Use cache if fresh
        if self._cache_time and (
            datetime.now(timezone.utc) - self._cache_time
        ).total_seconds() < self.CACHE_TTL_SECONDS:
            return self._cache.get("signal", CorrelationSignal(available=False))

        try:
            import yfinance as yf

            # Fetch last trading day of 5-min bars (free, no key needed)
            tickers = list(self.SYMBOLS.values())
            data = yf.download(
                tickers,
                period="1d",
                interval="5m",
                progress=False,
                auto_adjust=True,
            )

            if data.empty or len(data) < 2:
                return CorrelationSignal(available=False)

            # Compute % change over last hour (12 × 5-min bars)
            close = data["Close"]
            recent = close.iloc[-12:]
            if len(recent) < 2:
                return CorrelationSignal(available=False)

            dxy_col = self.SYMBOLS["dxy"]
            sp500_col = self.SYMBOLS["sp500"]

            dxy_start = recent[dxy_col].iloc[0]
            dxy_end = recent[dxy_col].iloc[-1]
            sp500_start = recent[sp500_col].iloc[0]
            sp500_end = recent[sp500_col].iloc[-1]

            if not dxy_start or not sp500_start:
                return CorrelationSignal(available=False)

            dxy_change = (dxy_end - dxy_start) / dxy_start * 100
            sp500_change = (sp500_end - sp500_start) / sp500_start * 100

            dxy_direction = "up" if dxy_change > 0 else "down"
            implied_btc = "down" if dxy_direction == "up" else "up"
            sp500_aligned = (sp500_change > 0 and implied_btc == "up") or (
                sp500_change < 0 and implied_btc == "down"
            )

            # Divergence: DXY moved significantly but BTC barely budged
            divergence = (
                abs(dxy_change) > self.DIVERGENCE_THRESHOLD_PCT
                and abs(btc_price_change_pct) < 0.1
            )

            confidence = min(1.0, abs(dxy_change) / 1.0)  # 1% DXY move = full confidence
            if sp500_aligned:
                confidence = min(1.0, confidence + 0.2)

            signal = CorrelationSignal(
                available=True,
                dxy_divergence=divergence,
                dxy_direction=dxy_direction,
                implied_btc_direction=implied_btc,
                sp500_aligned=sp500_aligned,
                confidence=confidence,
            )
            self._cache["signal"] = signal
            self._cache_time = datetime.now(timezone.utc)
            return signal
        except Exception as e:
            logger.warning("yfinance_correlation_failed: %s", e)
            return CorrelationSignal(available=False)


# ---------------------------------------------------------------------------
# Sentiment (Fear & Greed / augmento.ai)
# ---------------------------------------------------------------------------


@dataclass
class SentimentData:
    """Crypto market sentiment data.

    API required:
      - Fear & Greed Index: alternative.me/crypto/fear-and-greed-index/api/
        (free, no key required).
      - Augmento: augmento.ai API (paid, NLP-based bull/bear scoring from
        social media).

    Course lesson 32 teaches:
      - Extreme Fear (0-25) → contrarian buy signal.
      - Extreme Greed (75-100) → contrarian sell signal.
      - Augmento bull/bear score confirms short-term sentiment bias.

    Fields:
        available: False until at least one data source is wired.
        fear_greed_index: 0..100. 0 = extreme fear, 100 = extreme greed.
        augmento_score: 0.0..1.0. > 0.5 = bullish social sentiment.
    """
    available: bool = False
    fear_greed_index: int | None = None   # 0..100
    augmento_score: float | None = None   # 0..1 (bull/bear from social media)


class SentimentProvider(Protocol):
    """Protocol for sentiment data providers."""
    async def fetch_sentiment(self) -> SentimentData: ...


class StubSentimentProvider:
    """Returns ``available=False`` until a real sentiment provider is wired.

    Fear & Greed is free (alternative.me/crypto/fear-and-greed-index/api/)
    and is the lowest-effort provider to implement first.
    """
    async def fetch_sentiment(self) -> SentimentData:
        return SentimentData(available=False)


# ---------------------------------------------------------------------------
# Default registry — swap stubs for real providers when credentials arrive
# ---------------------------------------------------------------------------


def _default_correlation_provider():
    """Return YFinanceCorrelationProvider if yfinance is installed, else Stub."""
    try:
        import yfinance  # noqa: F401 — availability check only
        return YFinanceCorrelationProvider()
    except ImportError:
        return StubCorrelationProvider()


@dataclass
class DataFeedRegistry:
    """One-stop registry the MMEngine can call. Defaults to stubs.

    To wire a real provider, instantiate with your implementation::

        registry = DataFeedRegistry(
            hyblock=MyRealHyblockProvider(api_key="..."),
            news=MyForexFactoryScraper(),
            # others still stubbed
        )

    Call ``get_status()`` to inspect which providers are currently live.
    """
    hyblock: HyblockProvider = field(default_factory=StubHyblockProvider)
    tradinglite: TradingLiteProvider = field(default_factory=StubTradingLiteProvider)
    news: NewsProvider = field(default_factory=StubNewsProvider)
    options: OptionsProvider = field(default_factory=StubOptionsProvider)
    dominance: DominanceProvider = field(default_factory=StubDominanceProvider)
    correlation: CorrelationProvider = field(default_factory=_default_correlation_provider)
    sentiment: SentimentProvider = field(default_factory=StubSentimentProvider)


    def get_status(self) -> dict[str, bool]:
        """Return availability status of all registered providers.

        A provider is considered "available" if it is NOT a stub (i.e. its
        class name does not start with "Stub"). This gives a quick runtime
        overview of which external feeds are wired.

        Returns:
            Dict mapping provider name to True (real) or False (stubbed).

        Example::

            registry.get_status()
            # {"hyblock": False, "tradinglite": False, "news": False,
            #  "options": False, "dominance": False, "correlation": False,
            #  "sentiment": False}
        """
        providers = {
            "hyblock": self.hyblock,
            "tradinglite": self.tradinglite,
            "news": self.news,
            "options": self.options,
            "dominance": self.dominance,
            "correlation": self.correlation,
            "sentiment": self.sentiment,
        }
        return {
            name: not type(provider).__name__.startswith("Stub")
            for name, provider in providers.items()
        }
