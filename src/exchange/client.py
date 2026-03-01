from __future__ import annotations

import asyncio
import time

import ccxt.async_support as ccxt
import pandas as pd

from src.exchange.models import OrderResult
from src.utils.logging import get_logger
from src.utils.retry import async_retry

logger = get_logger(__name__)


class KrakenClient:
    """Async wrapper around ccxt Kraken with rate limiting and retry logic."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        self.exchange = ccxt.kraken(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        self._semaphore = asyncio.Semaphore(3)
        self._last_request_time: float = 0
        self._markets_loaded = False

    async def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            await self.exchange.load_markets()
            self._markets_loaded = True

    async def _rate_limit_wait(self) -> None:
        """Enforce minimum 1-second gap between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.0:
            await asyncio.sleep(1.0 - elapsed)

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int = 200, since: int | None = None
    ) -> pd.DataFrame:
        """Fetch OHLCV candles and return as DataFrame."""
        async with self._semaphore:
            await self._rate_limit_wait()
            kwargs: dict = {"limit": limit}
            if since is not None:
                kwargs["since"] = since
            raw = await self.exchange.fetch_ohlcv(symbol, timeframe, **kwargs)
            self._last_request_time = time.time()

        if not raw:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> OrderResult:
        """Place a spot market order."""
        async with self._semaphore:
            await self._rate_limit_wait()
            result = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=quantity,
            )
            self._last_request_time = time.time()

        fee_cost = 0.0
        if result.get("fee") and result["fee"].get("cost"):
            fee_cost = float(result["fee"]["cost"])

        return OrderResult(
            order_id=result["id"],
            symbol=symbol,
            side=side,
            filled_quantity=float(result.get("filled", 0)),
            avg_price=float(result.get("average", 0)),
            fee=fee_cost,
            status=result.get("status", "unknown"),
        )

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def get_balance(self) -> dict[str, float]:
        """Get account balances (non-zero only)."""
        async with self._semaphore:
            await self._rate_limit_wait()
            balance = await self.exchange.fetch_balance()
            self._last_request_time = time.time()
        return {k: float(v) for k, v in balance.get("free", {}).items() if float(v) > 0}

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current ticker for a symbol."""
        async with self._semaphore:
            await self._rate_limit_wait()
            ticker = await self.exchange.fetch_ticker(symbol)
            self._last_request_time = time.time()
        return ticker

    # Fiat currencies to exclude (we only want crypto base assets)
    FIAT_BASES = frozenset({
        "USD", "EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "NZD",
        "USDT", "USDC", "DAI", "PYUSD", "TUSD", "BUSD", "UST",
    })

    async def get_tradeable_pairs(
        self,
        min_volume_usd: float = 50_000,
        quote_currencies: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[str]:
        """Return all crypto pairs meeting volume and quote currency filters."""
        if quote_currencies is None:
            quote_currencies = ["USD", "USDT"]
        if exclude is None:
            exclude = ["USDT/USD", "USDC/USD", "DAI/USD", "PYUSD/USD"]

        await self._ensure_markets()

        candidates = []
        for symbol, market in self.exchange.markets.items():
            if not market.get("active") or not market.get("spot"):
                continue
            if market.get("quote") not in quote_currencies:
                continue
            if symbol in exclude:
                continue
            # Skip forex / stablecoin-vs-fiat pairs — crypto only
            base = market.get("base", "")
            if base in self.FIAT_BASES:
                continue
            candidates.append(symbol)

        if not candidates:
            return []

        # Fetch tickers to filter by volume
        async with self._semaphore:
            await self._rate_limit_wait()
            tickers = await self.exchange.fetch_tickers(candidates)
            self._last_request_time = time.time()

        filtered = []
        volume_map = {}
        for symbol, ticker in tickers.items():
            quote_vol = float(ticker.get("quoteVolume") or 0)
            if quote_vol >= min_volume_usd:
                filtered.append(symbol)
                volume_map[symbol] = quote_vol

        logger.info("pairs_scanned", total_candidates=len(candidates), filtered=len(filtered), min_volume=min_volume_usd)
        self._last_volume_map = volume_map
        return sorted(filtered)

    def get_24h_volume(self, symbol: str) -> float:
        """Return the last known 24h quote volume for a symbol."""
        return getattr(self, "_last_volume_map", {}).get(symbol, 0.0)

    async def close(self) -> None:
        await self.exchange.close()
