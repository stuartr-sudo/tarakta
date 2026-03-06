"""Exchange abstraction layer — protocols and market metadata.

Defines the common interface that all exchange connectors must satisfy.
Uses typing.Protocol for structural subtyping — existing Binance classes
satisfy this automatically via duck typing without inheritance changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Protocol, runtime_checkable

import pandas as pd

from src.exchange.models import OrderResult


@dataclass
class TradingHours:
    """When a market is open for trading. None means 24/7 (crypto)."""
    timezone: str                       # e.g. "America/New_York", "UTC"
    open_time: time                     # Market open (e.g. 09:30)
    close_time: time                    # Market close (e.g. 16:00)
    trading_days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri


@dataclass
class MarketInfo:
    """Market metadata for a connector."""
    market_type: str                    # "crypto", "stocks", "commodities"
    trading_hours: TradingHours | None  # None = 24/7
    symbol_format: str                  # "BASE/QUOTE" (crypto) or "TICKER" (stocks)
    supports_shorting: bool
    max_leverage: int
    default_fee_rate: float
    min_order_value: float
    currency: str                       # "USD", "USDT"
    tv_exchange_prefix: str             # "BINANCE", "NASDAQ", "COMEX", etc.


# Default market info for crypto (Binance) — used by existing clients
CRYPTO_MARKET_INFO = MarketInfo(
    market_type="crypto",
    trading_hours=None,  # 24/7
    symbol_format="BASE/QUOTE",
    supports_shorting=True,
    max_leverage=10,
    default_fee_rate=0.001,
    min_order_value=10.0,
    currency="USDT",
    tv_exchange_prefix="BINANCE",
)

CRYPTO_FUTURES_MARKET_INFO = MarketInfo(
    market_type="crypto",
    trading_hours=None,
    symbol_format="BASE/QUOTE",
    supports_shorting=True,
    max_leverage=125,
    default_fee_rate=0.0004,
    min_order_value=5.0,
    currency="USDT",
    tv_exchange_prefix="BINANCE",
)

US_STOCKS_MARKET_INFO = MarketInfo(
    market_type="stocks",
    trading_hours=TradingHours(
        timezone="America/New_York",
        open_time=time(9, 30),
        close_time=time(16, 0),
        trading_days=[0, 1, 2, 3, 4],
    ),
    symbol_format="TICKER",
    supports_shorting=False,  # Data-only connectors can't short
    max_leverage=1,
    default_fee_rate=0.0,
    min_order_value=1.0,
    currency="USD",
    tv_exchange_prefix="NASDAQ",
)

COMMODITIES_MARKET_INFO = MarketInfo(
    market_type="commodities",
    trading_hours=TradingHours(
        timezone="America/New_York",
        open_time=time(8, 20),
        close_time=time(13, 30),
        trading_days=[0, 1, 2, 3, 4],
    ),
    symbol_format="TICKER",
    supports_shorting=False,
    max_leverage=1,
    default_fee_rate=0.0,
    min_order_value=1.0,
    currency="USD",
    tv_exchange_prefix="COMEX",
)


@runtime_checkable
class ExchangeProtocol(Protocol):
    """Minimal interface every exchange connector must satisfy.

    Existing Binance clients already implement all these methods.
    New connectors (yfinance, Alpaca, IBKR) must also implement them.
    """

    # --- Market Data ---
    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int = 200, since: int | None = None,
    ) -> pd.DataFrame: ...

    async def fetch_ticker(self, symbol: str) -> dict: ...

    async def fetch_order_book(self, symbol: str, limit: int = 5) -> dict: ...

    # --- Trading ---
    async def place_market_order(
        self, symbol: str, side: str, quantity: float,
    ) -> OrderResult: ...

    async def place_limit_order(
        self, symbol: str, side: str, quantity: float, price: float,
    ) -> OrderResult: ...

    # --- Account ---
    async def get_balance(self) -> dict[str, float]: ...

    # --- Discovery ---
    async def get_tradeable_pairs(
        self, min_volume_usd: float = 50_000, quote_currencies: list[str] | None = None, **kwargs,
    ) -> list[str]: ...

    def get_24h_volume(self, symbol: str) -> float: ...

    # --- Properties ---
    @property
    def exchange_name(self) -> str: ...

    @property
    def taker_fee_rate(self) -> float: ...

    @property
    def min_order_usd(self) -> float: ...

    # --- Lifecycle ---
    async def close(self) -> None: ...


@runtime_checkable
class FuturesCapable(Protocol):
    """Optional: exchange supports futures-specific data (crypto only)."""

    async def fetch_open_interest(self, symbol: str) -> dict: ...
    async def fetch_funding_rate(self, symbol: str) -> dict: ...
    async def fetch_long_short_ratio(self, symbol: str) -> dict: ...
    async def set_leverage(self, symbol: str) -> None: ...
    async def get_position_risk(self, symbol: str) -> dict: ...

    @property
    def account_type(self) -> str: ...

    @property
    def leverage(self) -> int: ...


def parse_symbol_base(symbol: str) -> str:
    """Extract base asset from any symbol format.

    Handles:
      - Crypto: "BTC/USDT" -> "BTC", "BTC/USDT:USDT" -> "BTC"
      - Stocks: "AAPL" -> "AAPL"
      - Commodities: "GC=F" -> "GC=F"
    """
    if "/" in symbol:
        base = symbol.split("/")[0]
        return base.split(":")[0] if ":" in base else base
    return symbol
