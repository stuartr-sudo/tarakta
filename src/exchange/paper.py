from __future__ import annotations

import asyncio
from uuid import uuid4

import pandas as pd

from src.exchange.client import KrakenClient
from src.exchange.models import OrderResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

SIMULATED_SLIPPAGE = 0.001  # 0.1%
SIMULATED_FEE_RATE = 0.0026  # Kraken taker fee


class PaperExchange:
    """
    Paper trading exchange mock.
    Uses real market data from live exchange, simulates order fills.
    Same interface as KrakenClient for drop-in replacement.
    """

    def __init__(self, initial_balance: float, live_exchange: KrakenClient) -> None:
        self.balance: dict[str, float] = {"USD": initial_balance}
        self.live = live_exchange
        self.order_history: list[dict] = []
        logger.info("paper_exchange_init", balance=initial_balance)

    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int = 200, since: int | None = None
    ) -> pd.DataFrame:
        """Delegate to live exchange for real market data."""
        return await self.live.fetch_candles(symbol, timeframe, limit=limit, since=since)

    async def fetch_ticker(self, symbol: str) -> dict:
        """Delegate to live exchange for real prices."""
        return await self.live.fetch_ticker(symbol)

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> OrderResult:
        """Simulate a market order fill at current price with slippage."""
        ticker = await self.live.fetch_ticker(symbol)
        price = float(ticker["last"])

        if side == "buy":
            fill_price = price * (1 + SIMULATED_SLIPPAGE)
            cost = quantity * fill_price
            fee = cost * SIMULATED_FEE_RATE

            base = symbol.split("/")[0]
            short_key = f"SHORT_{base}"

            if self.balance.get(short_key, 0) >= quantity * 0.99:
                # Closing a short — return collateral, deduct buy-back cost + fee
                self.balance[short_key] = max(0, self.balance[short_key] - quantity)
                # Short open credited revenue; now debit the buy-back cost + fee
                self.balance["USD"] = self.balance.get("USD", 0) - cost - fee
            else:
                # Opening a long — buy with USD
                total_cost = cost + fee
                usd_balance = self.balance.get("USD", 0)
                if total_cost > usd_balance:
                    raise ValueError(
                        f"Insufficient paper balance: need ${total_cost:.2f}, have ${usd_balance:.2f}"
                    )
                self.balance["USD"] = usd_balance - total_cost
                self.balance[base] = self.balance.get(base, 0) + quantity
        else:
            fill_price = price * (1 - SIMULATED_SLIPPAGE)
            revenue = quantity * fill_price
            fee = revenue * SIMULATED_FEE_RATE

            base = symbol.split("/")[0]
            base_balance = self.balance.get(base, 0)

            if base_balance >= quantity * 0.99:
                # Closing a long position — sell existing holdings
                self.balance[base] = max(0, base_balance - quantity)
                self.balance["USD"] = self.balance.get("USD", 0) + revenue - fee
            else:
                # Opening a short position — receive sale proceeds, track obligation
                usd_balance = self.balance.get("USD", 0)
                # Credit the sale revenue minus fee
                self.balance["USD"] = usd_balance + revenue - fee
                short_key = f"SHORT_{base}"
                self.balance[short_key] = self.balance.get(short_key, 0) + quantity

        order_id = f"paper-{uuid4().hex[:8]}"
        result = OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            filled_quantity=quantity,
            avg_price=fill_price,
            fee=fee,
            status="closed",
        )
        self.order_history.append(
            {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": fill_price,
                "fee": fee,
            }
        )

        logger.info(
            "paper_order",
            side=side,
            symbol=symbol,
            quantity=quantity,
            price=fill_price,
            fee=fee,
            usd_balance=self.balance.get("USD", 0),
        )
        return result

    async def get_balance(self) -> dict[str, float]:
        return {k: v for k, v in self.balance.items() if v > 0}

    async def get_tradeable_pairs(
        self,
        min_volume_usd: float = 50_000,
        quote_currencies: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[str]:
        """Delegate to live exchange."""
        return await self.live.get_tradeable_pairs(min_volume_usd, quote_currencies, exclude)

    async def close(self) -> None:
        await self.live.close()
