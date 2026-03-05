from __future__ import annotations

import asyncio
import time

import ccxt.async_support as ccxt
import pandas as pd

from src.exchange.models import OrderResult
from src.utils.logging import get_logger
from src.utils.retry import async_retry

logger = get_logger(__name__)


class BinanceClient:
    """Async wrapper around ccxt Binance with rate limiting and retry logic."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "timeout": 30000,
                "options": {
                    "defaultType": "spot",
                    "fetchCurrencies": False,
                },
            }
        )
        self._semaphore = asyncio.Semaphore(10)
        self._last_request_time: float = 0
        self._markets_loaded = False

    async def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            await self.exchange.load_markets()
            self._markets_loaded = True

    async def _rate_limit_wait(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.5:
            await asyncio.sleep(0.5 - elapsed)

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int = 200, since: int | None = None
    ) -> pd.DataFrame:
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
        async with self._semaphore:
            await self._rate_limit_wait()
            result = await self.exchange.create_order(
                symbol=symbol, type="market", side=side, amount=quantity,
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
    async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> OrderResult:
        """Place a spot limit order (maker fee)."""
        async with self._semaphore:
            await self._rate_limit_wait()
            result = await self.exchange.create_order(
                symbol=symbol, type="limit", side=side, amount=quantity, price=price,
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
            avg_price=float(result.get("average", 0) or price),
            fee=fee_cost,
            status=result.get("status", "unknown"),
        )

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def fetch_order_book(self, symbol: str, limit: int = 5) -> dict:
        """Fetch order book. Returns {'bids': [[price, qty], ...], 'asks': [[price, qty], ...]}."""
        async with self._semaphore:
            await self._rate_limit_wait()
            ob = await self.exchange.fetch_order_book(symbol, limit=limit)
            self._last_request_time = time.time()
        return ob

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def get_balance(self) -> dict[str, float]:
        async with self._semaphore:
            await self._rate_limit_wait()
            balance = await self.exchange.fetch_balance()
            self._last_request_time = time.time()
        return {k: float(v) for k, v in balance.get("free", {}).items() if float(v) > 0}

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def fetch_ticker(self, symbol: str) -> dict:
        async with self._semaphore:
            await self._rate_limit_wait()
            ticker = await self.exchange.fetch_ticker(symbol)
            self._last_request_time = time.time()
        return ticker

    FIAT_BASES = frozenset({
        "USD", "EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "NZD",
        "USDT", "USDC", "DAI", "PYUSD", "TUSD", "BUSD", "UST",
        "FDUSD", "USDP",
    })

    async def get_tradeable_pairs(
        self,
        min_volume_usd: float = 50_000,
        quote_currencies: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[str]:
        if quote_currencies is None:
            quote_currencies = ["USDT"]
        if exclude is None:
            exclude = ["USDT/BUSD", "USDC/USDT", "BUSD/USDT", "FDUSD/USDT"]

        await self._ensure_markets()

        candidates = []
        for symbol, market in self.exchange.markets.items():
            if not market.get("active") or not market.get("spot"):
                continue
            if market.get("quote") not in quote_currencies:
                continue
            if symbol in exclude:
                continue
            base = market.get("base", "")
            if base in self.FIAT_BASES:
                continue
            candidates.append(symbol)

        if not candidates:
            return []

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
        return getattr(self, "_last_volume_map", {}).get(symbol, 0.0)

    @property
    def exchange_name(self) -> str:
        return "binance"

    @property
    def taker_fee_rate(self) -> float:
        return 0.001

    @property
    def min_order_usd(self) -> float:
        return 10.0

    async def close(self) -> None:
        await self.exchange.close()


class BinanceFuturesClient:
    """Async wrapper around ccxt Binance USDM Futures (perpetual contracts)."""

    def __init__(self, api_key: str, api_secret: str, leverage: int = 1, margin_mode: str = "isolated") -> None:
        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "timeout": 30000,
                "options": {
                    "defaultType": "future",
                    "defaultMarginMode": margin_mode,
                    "fetchCurrencies": False,
                },
            }
        )
        self._leverage = leverage
        self._margin_mode = margin_mode
        self._semaphore = asyncio.Semaphore(10)
        self._last_request_time: float = 0
        self._markets_loaded = False
        self._leverage_set: set[str] = set()

    @property
    def account_type(self) -> str:
        return "futures"

    @property
    def leverage(self) -> int:
        return self._leverage

    async def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            await self.exchange.load_markets()
            self._markets_loaded = True

    async def _rate_limit_wait(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.5:
            await asyncio.sleep(0.5 - elapsed)

    async def set_leverage(self, symbol: str) -> None:
        """Set leverage and margin mode for a symbol (once per symbol per session)."""
        if symbol in self._leverage_set:
            return
        async with self._semaphore:
            await self._rate_limit_wait()
            try:
                await self.exchange.set_margin_mode(self._margin_mode, symbol)
            except Exception:
                pass  # May already be set
            try:
                await self.exchange.set_leverage(self._leverage, symbol)
            except Exception:
                pass  # May already be set
            self._last_request_time = time.time()
        self._leverage_set.add(symbol)
        logger.info("futures_leverage_set", symbol=symbol, leverage=self._leverage, mode=self._margin_mode)

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int = 200, since: int | None = None
    ) -> pd.DataFrame:
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
        await self.set_leverage(symbol)
        async with self._semaphore:
            await self._rate_limit_wait()
            result = await self.exchange.create_order(
                symbol=symbol, type="market", side=side, amount=quantity,
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
    async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> OrderResult:
        """Place a futures limit order (maker fee)."""
        await self.set_leverage(symbol)
        async with self._semaphore:
            await self._rate_limit_wait()
            result = await self.exchange.create_order(
                symbol=symbol, type="limit", side=side, amount=quantity, price=price,
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
            avg_price=float(result.get("average", 0) or price),
            fee=fee_cost,
            status=result.get("status", "unknown"),
        )

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def fetch_order_book(self, symbol: str, limit: int = 5) -> dict:
        """Fetch order book."""
        async with self._semaphore:
            await self._rate_limit_wait()
            ob = await self.exchange.fetch_order_book(symbol, limit=limit)
            self._last_request_time = time.time()
        return ob

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def get_balance(self) -> dict[str, float]:
        async with self._semaphore:
            await self._rate_limit_wait()
            balance = await self.exchange.fetch_balance()
            self._last_request_time = time.time()
        return {k: float(v) for k, v in balance.get("free", {}).items() if float(v) > 0}

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def fetch_ticker(self, symbol: str) -> dict:
        async with self._semaphore:
            await self._rate_limit_wait()
            ticker = await self.exchange.fetch_ticker(symbol)
            self._last_request_time = time.time()
        return ticker

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def get_position_risk(self, symbol: str) -> dict:
        """Fetch position risk info: margin ratio, liquidation price, unrealized PnL."""
        async with self._semaphore:
            await self._rate_limit_wait()
            positions = await self.exchange.fetch_positions([symbol])
            self._last_request_time = time.time()

        for pos in positions:
            if pos.get("symbol") == symbol and float(pos.get("contracts", 0)) > 0:
                return {
                    "liquidation_price": float(pos.get("liquidationPrice", 0) or 0),
                    "margin_ratio": float(pos.get("marginRatio", 0) or 0),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0) or 0),
                    "margin_used": float(pos.get("initialMargin", 0) or 0),
                    "notional": float(pos.get("notional", 0) or 0),
                }
        return {}

    FIAT_BASES = frozenset({
        "USD", "EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "NZD",
        "USDT", "USDC", "DAI", "PYUSD", "TUSD", "BUSD", "UST",
        "FDUSD", "USDP",
    })

    # Top coins by market cap — established projects only, no meme coins
    QUALITY_BASES = frozenset({
        # Layer 1s
        "BTC", "ETH", "SOL", "ADA", "AVAX", "DOT", "ATOM", "NEAR",
        "SUI", "APT", "SEI", "TIA", "INJ", "FTM", "ALGO", "EGLD",
        "HBAR", "ICP", "FIL", "TON", "TRX", "XLM", "XRP", "EOS",
        "FLOW", "MINA", "CELO", "ONE", "KAVA", "VET", "THETA",
        # Layer 2s / Scaling
        "MATIC", "POL", "ARB", "OP", "IMX", "STRK", "MNT", "METIS",
        "ZK", "MANTA", "BLAST",
        # DeFi
        "UNI", "AAVE", "MKR", "SNX", "COMP", "CRV", "SUSHI", "YFI",
        "DYDX", "GMX", "PENDLE", "JUP", "RAY", "JTO", "PYTH",
        "LDO", "RPL", "FXS", "LQTY", "BAL", "1INCH",
        # Infrastructure / Oracles
        "LINK", "GRT", "API3", "BAND", "REN",
        # Storage / Compute
        "AR", "RENDER", "AKT", "TAO", "FET", "AGIX",
        # Gaming / Metaverse
        "AXS", "SAND", "MANA", "GALA", "ENJ", "RONIN", "PIXEL",
        # Exchange tokens
        "BNB", "OKB", "CRO",
        # Privacy
        "XMR", "ZEC",
        # Interoperability
        "RUNE", "ZRO", "W", "WOO", "STX",
        # Other established
        "LTC", "BCH", "ETC", "DOGE", "SHIB", "PEPE",
        "BONK", "WIF", "FLOKI",  # Top memes only (high liquidity)
        "ONDO", "ENS", "SSV", "EIGEN", "ETHFI",
        "WLD", "JASMY", "CHZ", "MASK", "BLUR",
    })

    async def get_tradeable_pairs(
        self,
        min_volume_usd: float = 50_000,
        quote_currencies: list[str] | None = None,
        exclude: list[str] | None = None,
        quality_filter: bool = True,
    ) -> list[str]:
        if quote_currencies is None:
            quote_currencies = ["USDT"]
        if exclude is None:
            exclude = []

        await self._ensure_markets()

        candidates = []
        for symbol, market in self.exchange.markets.items():
            if not market.get("active"):
                continue
            # Filter for linear (USDT-margined) perpetual contracts
            if not market.get("linear"):
                continue
            if not market.get("swap"):
                continue
            if market.get("quote") not in quote_currencies:
                continue
            if symbol in exclude:
                continue
            base = market.get("base", "")
            if base in self.FIAT_BASES:
                continue
            if quality_filter and base not in self.QUALITY_BASES:
                continue
            candidates.append(symbol)

        if not candidates:
            return []

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

        logger.info("futures_pairs_scanned", total_candidates=len(candidates), filtered=len(filtered), min_volume=min_volume_usd, quality_filter=quality_filter)
        self._last_volume_map = volume_map
        return sorted(filtered)

    def get_24h_volume(self, symbol: str) -> float:
        return getattr(self, "_last_volume_map", {}).get(symbol, 0.0)

    @property
    def exchange_name(self) -> str:
        return "binance_futures"

    @property
    def taker_fee_rate(self) -> float:
        return 0.0004  # 0.04% futures taker fee

    @property
    def min_order_usd(self) -> float:
        return 5.0

    async def close(self) -> None:
        await self.exchange.close()


class BinanceMarginClient:
    """Async wrapper around ccxt Binance Cross/Isolated Margin (spot-based shorting)."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "timeout": 30000,
                "options": {
                    "defaultType": "margin",
                    "fetchCurrencies": False,
                },
            }
        )
        self._semaphore = asyncio.Semaphore(10)
        self._last_request_time: float = 0
        self._markets_loaded = False

    @property
    def account_type(self) -> str:
        return "margin"

    async def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            await self.exchange.load_markets()
            self._markets_loaded = True

    async def _rate_limit_wait(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.5:
            await asyncio.sleep(0.5 - elapsed)

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int = 200, since: int | None = None
    ) -> pd.DataFrame:
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
    async def place_market_order(self, symbol: str, side: str, quantity: float, closing: bool = False) -> OrderResult:
        """Place a margin market order. For shorts: auto-borrow on open, auto-repay on close."""
        params = {}
        if side == "sell" and not closing:
            # Opening a short — auto-borrow
            params = {"type": "margin", "sideEffectType": "MARGIN_BUY"}
        elif side == "buy" and closing:
            # Closing a short — auto-repay
            params = {"type": "margin", "sideEffectType": "AUTO_REPAY"}

        async with self._semaphore:
            await self._rate_limit_wait()
            result = await self.exchange.create_order(
                symbol=symbol, type="market", side=side, amount=quantity,
                params=params,
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
    async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float, closing: bool = False) -> OrderResult:
        """Place a margin limit order (maker fee)."""
        params = {}
        if side == "sell" and not closing:
            params = {"type": "margin", "sideEffectType": "MARGIN_BUY"}
        elif side == "buy" and closing:
            params = {"type": "margin", "sideEffectType": "AUTO_REPAY"}

        async with self._semaphore:
            await self._rate_limit_wait()
            result = await self.exchange.create_order(
                symbol=symbol, type="limit", side=side, amount=quantity, price=price,
                params=params,
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
            avg_price=float(result.get("average", 0) or price),
            fee=fee_cost,
            status=result.get("status", "unknown"),
        )

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def fetch_order_book(self, symbol: str, limit: int = 5) -> dict:
        """Fetch order book."""
        async with self._semaphore:
            await self._rate_limit_wait()
            ob = await self.exchange.fetch_order_book(symbol, limit=limit)
            self._last_request_time = time.time()
        return ob

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def get_balance(self) -> dict[str, float]:
        async with self._semaphore:
            await self._rate_limit_wait()
            balance = await self.exchange.fetch_balance({"type": "margin"})
            self._last_request_time = time.time()
        return {k: float(v) for k, v in balance.get("free", {}).items() if float(v) > 0}

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def get_margin_info(self) -> dict:
        """Return available margin, used margin, and margin level."""
        async with self._semaphore:
            await self._rate_limit_wait()
            balance = await self.exchange.fetch_balance({"type": "margin"})
            self._last_request_time = time.time()
        info = balance.get("info", {})
        return {
            "total_net_asset": float(info.get("totalNetAssetOfBtc", 0)),
            "margin_level": float(info.get("marginLevel", 0)),
            "total_liability": float(info.get("totalLiabilityOfBtc", 0)),
        }

    @async_retry(max_attempts=3, base_delay=5.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    async def fetch_ticker(self, symbol: str) -> dict:
        async with self._semaphore:
            await self._rate_limit_wait()
            ticker = await self.exchange.fetch_ticker(symbol)
            self._last_request_time = time.time()
        return ticker

    FIAT_BASES = frozenset({
        "USD", "EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "NZD",
        "USDT", "USDC", "DAI", "PYUSD", "TUSD", "BUSD", "UST",
        "FDUSD", "USDP",
    })

    async def get_tradeable_pairs(
        self,
        min_volume_usd: float = 50_000,
        quote_currencies: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[str]:
        if quote_currencies is None:
            quote_currencies = ["USDT"]
        if exclude is None:
            exclude = ["USDT/BUSD", "USDC/USDT", "BUSD/USDT", "FDUSD/USDT"]

        await self._ensure_markets()

        candidates = []
        for symbol, market in self.exchange.markets.items():
            if not market.get("active") or not market.get("spot"):
                continue
            if not market.get("margin"):
                continue  # Must be margin-eligible
            if market.get("quote") not in quote_currencies:
                continue
            if symbol in exclude:
                continue
            base = market.get("base", "")
            if base in self.FIAT_BASES:
                continue
            candidates.append(symbol)

        if not candidates:
            return []

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

        logger.info("margin_pairs_scanned", total_candidates=len(candidates), filtered=len(filtered), min_volume=min_volume_usd)
        self._last_volume_map = volume_map
        return sorted(filtered)

    def get_24h_volume(self, symbol: str) -> float:
        return getattr(self, "_last_volume_map", {}).get(symbol, 0.0)

    @property
    def exchange_name(self) -> str:
        return "binance_margin"

    @property
    def taker_fee_rate(self) -> float:
        return 0.001  # same as spot

    @property
    def min_order_usd(self) -> float:
        return 10.0

    async def close(self) -> None:
        await self.exchange.close()


def create_exchange(name: str, api_key: str, api_secret: str, account_type: str = "spot", leverage: int = 1, margin_mode: str = "isolated"):
    """Factory: create Binance exchange client by account type."""
    if account_type == "futures":
        return BinanceFuturesClient(api_key, api_secret, leverage=leverage, margin_mode=margin_mode)
    elif account_type == "margin":
        return BinanceMarginClient(api_key, api_secret)
    return BinanceClient(api_key, api_secret)
