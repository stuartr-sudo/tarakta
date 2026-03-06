"""yfinance connector — free data for stocks, commodities, ETFs.

Data-only connector (no live trading). Use PaperExchange wrapper for simulated fills.
yfinance is synchronous, so all calls are wrapped in run_in_executor.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.exchange.factory import register_connector
from src.exchange.models import OrderResult
from src.exchange.protocol import (
    COMMODITIES_MARKET_INFO,
    US_STOCKS_MARKET_INFO,
    MarketInfo,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Map Tarakta timeframes to yfinance intervals
_TF_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "1h",  # yfinance doesn't have 4h — we fetch 1h and resample
    "1d": "1d", "1w": "1wk",
}

# Map timeframe to period (how far back to fetch)
_TF_PERIOD = {
    "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
    "1h": "730d", "4h": "730d", "1d": "5y", "1w": "10y",
}


def _resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h candles to 4h candles."""
    if df.empty:
        return df
    resampled = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])
    return resampled


class _YFinanceBase:
    """Base class for yfinance connectors. Handles data fetching; no live trading."""

    def __init__(self, symbol_universe: list[str] | None = None, **kwargs) -> None:
        self._symbol_universe = symbol_universe or []
        self._volume_map: dict[str, float] = {}
        self._ticker_cache: dict[str, tuple[float, dict]] = {}  # symbol -> (timestamp, data)
        self._cache_ttl = 30  # seconds

    def _sync_fetch_candles(self, symbol: str, yf_interval: str, yf_period: str, limit: int) -> pd.DataFrame:
        """Synchronous yfinance download — called via run_in_executor."""
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=yf_period, interval=yf_interval)

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Keep only OHLCV columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0

        df = df[["open", "high", "low", "close", "volume"]]

        # Ensure UTC timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Limit to requested number of candles
        if len(df) > limit:
            df = df.iloc[-limit:]

        return df

    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int = 200, since: int | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles from Yahoo Finance."""
        yf_interval = _TF_MAP.get(timeframe, timeframe)
        yf_period = _TF_PERIOD.get(timeframe, "730d")
        need_resample = timeframe == "4h"

        if need_resample:
            # Fetch 1h candles then resample to 4h
            fetch_limit = limit * 4 + 10  # Extra to account for resampling
            df = await asyncio.get_event_loop().run_in_executor(
                None, self._sync_fetch_candles, symbol, "1h", yf_period, fetch_limit,
            )
            if not df.empty:
                df = _resample_to_4h(df)
                if len(df) > limit:
                    df = df.iloc[-limit:]
        else:
            df = await asyncio.get_event_loop().run_in_executor(
                None, self._sync_fetch_candles, symbol, yf_interval, yf_period, limit,
            )

        return df

    def _sync_fetch_ticker(self, symbol: str) -> dict:
        """Synchronous ticker fetch."""
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = ticker.fast_info

        last_price = float(getattr(info, "last_price", 0) or 0)
        prev_close = float(getattr(info, "previous_close", last_price) or last_price)
        market_cap = float(getattr(info, "market_cap", 0) or 0)

        # yfinance doesn't provide real-time bid/ask reliably — estimate from last price
        spread = last_price * 0.001  # 0.1% synthetic spread
        bid = last_price - spread / 2
        ask = last_price + spread / 2

        return {
            "symbol": symbol,
            "last": last_price,
            "bid": bid,
            "ask": ask,
            "quoteVolume": float(getattr(info, "last_volume", 0) or 0) * last_price,
            "previousClose": prev_close,
            "marketCap": market_cap,
        }

    async def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current price data. Cached for 30 seconds."""
        now = asyncio.get_event_loop().time()
        cached = self._ticker_cache.get(symbol)
        if cached and now - cached[0] < self._cache_ttl:
            return cached[1]

        ticker = await asyncio.get_event_loop().run_in_executor(
            None, self._sync_fetch_ticker, symbol,
        )
        self._ticker_cache[symbol] = (now, ticker)
        return ticker

    async def fetch_order_book(self, symbol: str, limit: int = 5) -> dict:
        """Synthetic order book from bid/ask spread (yfinance has no real order book)."""
        ticker = await self.fetch_ticker(symbol)
        bid = ticker.get("bid", 0)
        ask = ticker.get("ask", 0)
        last = ticker.get("last", 0)

        if bid <= 0:
            bid = last * 0.9995
        if ask <= 0:
            ask = last * 1.0005

        # Generate synthetic depth levels
        spread = ask - bid
        bids = [[bid - i * spread * 0.1, 100.0] for i in range(limit)]
        asks = [[ask + i * spread * 0.1, 100.0] for i in range(limit)]

        return {"bids": bids, "asks": asks}

    async def get_tradeable_pairs(
        self, min_volume_usd: float = 0, quote_currencies: list[str] | None = None, **kwargs,
    ) -> list[str]:
        """Return the configured symbol universe (no dynamic discovery for stocks)."""
        return list(self._symbol_universe)

    def get_24h_volume(self, symbol: str) -> float:
        return self._volume_map.get(symbol, 0.0)

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> OrderResult:
        raise NotImplementedError("yfinance is data-only. Use PaperExchange wrapper for simulated trading.")

    async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> OrderResult:
        raise NotImplementedError("yfinance is data-only. Use PaperExchange wrapper for simulated trading.")

    async def get_balance(self) -> dict[str, float]:
        raise NotImplementedError("yfinance is data-only. Use PaperExchange wrapper.")

    async def close(self) -> None:
        """No persistent connections to close."""
        pass


class YFinanceStocksConnector(_YFinanceBase):
    """yfinance connector for US stocks (AAPL, MSFT, TSLA, etc.)."""

    @property
    def exchange_name(self) -> str:
        return "yfinance_stocks"

    @property
    def taker_fee_rate(self) -> float:
        return 0.0  # Commission-free for modern brokers

    @property
    def min_order_usd(self) -> float:
        return 1.0  # Fractional shares

    @property
    def market_info(self) -> MarketInfo:
        return US_STOCKS_MARKET_INFO


class YFinanceCommoditiesConnector(_YFinanceBase):
    """yfinance connector for commodities (GC=F, SI=F, CL=F, etc.)."""

    @property
    def exchange_name(self) -> str:
        return "yfinance_commodities"

    @property
    def taker_fee_rate(self) -> float:
        return 0.0

    @property
    def min_order_usd(self) -> float:
        return 1.0

    @property
    def market_info(self) -> MarketInfo:
        return COMMODITIES_MARKET_INFO


# Register both connectors
register_connector("yfinance_stocks", YFinanceStocksConnector)
register_connector("yfinance_commodities", YFinanceCommoditiesConnector)
