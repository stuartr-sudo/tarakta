from __future__ import annotations

import asyncio
from uuid import uuid4

import pandas as pd

from src.exchange.models import OrderResult
from src.exchange.protocol import parse_symbol_base
from src.utils.logging import get_logger

logger = get_logger(__name__)

SIMULATED_SLIPPAGE = 0.001  # 0.1%


class PaperExchange:
    """
    Paper trading exchange mock.
    Uses real market data from live exchange, simulates order fills.
    Same interface as exchange clients for drop-in replacement.
    """

    def __init__(self, initial_balance: float, live_exchange, account_type: str = "spot", leverage: int = 1) -> None:
        self.balance: dict[str, float] = {"USD": initial_balance}
        self.live = live_exchange
        self.order_history: list[dict] = []
        self._fee_rate = getattr(live_exchange, "taker_fee_rate", 0.0026)
        self._account_type = account_type
        self._leverage = leverage
        logger.info("paper_exchange_init", balance=initial_balance, account_type=account_type, leverage=leverage)

    def restore_positions(self, open_positions: dict) -> None:
        """Restore internal position tracking from engine state after restart.

        Without this, partial/full exits would fail to close positions correctly
        because the PaperExchange wouldn't know about them (it resets on restart).

        Args:
            open_positions: dict of symbol -> Position from engine state
        """
        if not open_positions:
            return

        total_margin_deployed = 0.0
        restored = 0

        for symbol, pos in open_positions.items():
            base = parse_symbol_base(symbol)
            leverage = pos.leverage or self._leverage

            if self._account_type == "futures":
                if pos.direction == "long":
                    key = f"LONG_{base}"
                    price_key = f"LONG_PRICE_{base}"
                else:
                    key = f"SHORT_{base}"
                    price_key = f"SHORT_PRICE_{base}"

                # Track the position quantity and entry price
                self.balance[key] = self.balance.get(key, 0) + pos.quantity
                self.balance[price_key] = pos.entry_price

                # Deduct margin from USD balance
                margin = pos.margin_used or (pos.cost_usd / leverage)
                total_margin_deployed += margin
            else:
                # Spot: track base tokens (long) or SHORT keys (short)
                if pos.direction == "long":
                    self.balance[base] = self.balance.get(base, 0) + pos.quantity
                    total_margin_deployed += pos.cost_usd
                else:
                    key = f"SHORT_{base}"
                    self.balance[key] = self.balance.get(key, 0) + pos.quantity
                    total_margin_deployed += pos.cost_usd

            restored += 1

        # Adjust USD balance: the initial_balance already accounts for all capital,
        # but we need to deduct what's deployed in positions
        self.balance["USD"] = self.balance.get("USD", 0) - total_margin_deployed

        logger.info(
            "paper_positions_restored",
            count=restored,
            margin_deployed=round(total_margin_deployed, 2),
            usd_balance=round(self.balance.get("USD", 0), 2),
        )

    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int = 200, since: int | None = None
    ) -> pd.DataFrame:
        """Delegate to live exchange for real market data."""
        return await self.live.fetch_candles(symbol, timeframe, limit=limit, since=since)

    async def fetch_ticker(self, symbol: str) -> dict:
        """Delegate to live exchange for real prices."""
        return await self.live.fetch_ticker(symbol)

    async def fetch_order_book(self, symbol: str, limit: int = 5) -> dict:
        """Delegate to live exchange for real order book."""
        return await self.live.fetch_order_book(symbol, limit=limit)

    async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> OrderResult:
        """Simulate a limit order fill at the specified price (instant fill for paper trading)."""
        # For paper trading, treat limit orders as immediate fills at the limit price
        # This is a simplification — real limit orders may not fill immediately
        ticker = await self.live.fetch_ticker(symbol)
        last_price = float(ticker["last"])

        # Use the limit price for fill, but apply minimal slippage
        fill_price = price

        if self._account_type == "futures":
            return await self._futures_limit_order(symbol, side, quantity, fill_price)

        base = parse_symbol_base(symbol)
        if side == "buy":
            cost = quantity * fill_price
            fee = cost * self._fee_rate
            short_key = f"SHORT_{base}"

            if self.balance.get(short_key, 0) >= quantity * 0.99:
                self.balance[short_key] = max(0, self.balance[short_key] - quantity)
                self.balance["USD"] = self.balance.get("USD", 0) - cost - fee
            else:
                total_cost = cost + fee
                usd_balance = self.balance.get("USD", 0)
                if total_cost > usd_balance:
                    raise ValueError(
                        f"Insufficient paper balance: need ${total_cost:.2f}, have ${usd_balance:.2f}"
                    )
                self.balance["USD"] = usd_balance - total_cost
                self.balance[base] = self.balance.get(base, 0) + quantity
        else:
            revenue = quantity * fill_price
            fee = revenue * self._fee_rate
            base_balance = self.balance.get(base, 0)

            if base_balance >= quantity * 0.99:
                self.balance[base] = max(0, base_balance - quantity)
                self.balance["USD"] = self.balance.get("USD", 0) + revenue - fee
            else:
                usd_balance = self.balance.get("USD", 0)
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
        self.order_history.append({
            "order_id": order_id, "symbol": symbol, "side": side,
            "quantity": quantity, "price": fill_price, "fee": fee,
        })
        logger.info(
            "paper_limit_order",
            side=side, symbol=symbol, quantity=quantity,
            price=fill_price, fee=fee,
            usd_balance=self.balance.get("USD", 0),
        )
        return result

    async def _futures_limit_order(self, symbol: str, side: str, quantity: float, fill_price: float) -> OrderResult:
        """Simulate a futures limit order with leverage."""
        leverage = self._leverage
        base = parse_symbol_base(symbol)
        long_key = f"LONG_{base}"
        short_key = f"SHORT_{base}"
        notional = quantity * fill_price
        fee = notional * self._fee_rate
        margin_required = notional / leverage

        if side == "buy":
            if self.balance.get(short_key, 0) >= quantity * 0.99:
                open_price_key = f"SHORT_PRICE_{base}"
                open_price = self.balance.get(open_price_key, fill_price)
                pnl = (open_price - fill_price) * quantity
                open_margin = (open_price * quantity) / leverage
                self.balance["USD"] = self.balance.get("USD", 0) + open_margin + pnl - fee
                self.balance[short_key] = max(0, self.balance[short_key] - quantity)
                if self.balance[short_key] == 0:
                    self.balance.pop(open_price_key, None)
            else:
                usd_balance = self.balance.get("USD", 0)
                if margin_required + fee > usd_balance:
                    raise ValueError(
                        f"Insufficient paper margin: need ${margin_required + fee:.2f}, have ${usd_balance:.2f}"
                    )
                self.balance["USD"] = usd_balance - margin_required - fee
                self.balance[long_key] = self.balance.get(long_key, 0) + quantity
                self.balance[f"LONG_PRICE_{base}"] = fill_price
        else:
            if self.balance.get(long_key, 0) >= quantity * 0.99:
                open_price_key = f"LONG_PRICE_{base}"
                open_price = self.balance.get(open_price_key, fill_price)
                pnl = (fill_price - open_price) * quantity
                open_margin = (open_price * quantity) / leverage
                self.balance["USD"] = self.balance.get("USD", 0) + open_margin + pnl - fee
                self.balance[long_key] = max(0, self.balance[long_key] - quantity)
                if self.balance[long_key] == 0:
                    self.balance.pop(open_price_key, None)
            else:
                usd_balance = self.balance.get("USD", 0)
                if margin_required + fee > usd_balance:
                    raise ValueError(
                        f"Insufficient paper margin: need ${margin_required + fee:.2f}, have ${usd_balance:.2f}"
                    )
                self.balance["USD"] = usd_balance - margin_required - fee
                self.balance[short_key] = self.balance.get(short_key, 0) + quantity
                self.balance[f"SHORT_PRICE_{base}"] = fill_price

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
        self.order_history.append({
            "order_id": order_id, "symbol": symbol, "side": side,
            "quantity": quantity, "price": fill_price, "fee": fee,
        })
        logger.info(
            "paper_futures_limit_order",
            side=side, symbol=symbol, quantity=quantity,
            price=fill_price, fee=fee, leverage=leverage,
            usd_balance=self.balance.get("USD", 0),
        )
        return result

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> OrderResult:
        """Simulate a market order fill at current price with slippage."""
        ticker = await self.live.fetch_ticker(symbol)
        price = float(ticker["last"])

        if self._account_type == "futures":
            return await self._futures_order(symbol, side, quantity, price)

        if side == "buy":
            fill_price = price * (1 + SIMULATED_SLIPPAGE)
            cost = quantity * fill_price
            fee = cost * self._fee_rate

            base = parse_symbol_base(symbol)
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
            fee = revenue * self._fee_rate

            base = parse_symbol_base(symbol)
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

    async def _futures_order(self, symbol: str, side: str, quantity: float, price: float) -> OrderResult:
        """Simulate a futures order with leverage. Margin-based accounting."""
        leverage = self._leverage
        base = parse_symbol_base(symbol)
        long_key = f"LONG_{base}"
        short_key = f"SHORT_{base}"

        if side == "buy":
            fill_price = price * (1 + SIMULATED_SLIPPAGE)
            notional = quantity * fill_price
            fee = notional * self._fee_rate
            margin_required = notional / leverage

            if self.balance.get(short_key, 0) >= quantity * 0.99:
                # Closing a short — return margin, apply PnL
                open_price_key = f"SHORT_PRICE_{base}"
                open_price = self.balance.get(open_price_key, fill_price)
                pnl = (open_price - fill_price) * quantity
                open_margin = (open_price * quantity) / leverage
                self.balance["USD"] = self.balance.get("USD", 0) + open_margin + pnl - fee
                self.balance[short_key] = max(0, self.balance[short_key] - quantity)
                if self.balance[short_key] == 0:
                    self.balance.pop(open_price_key, None)
            else:
                # Opening a long
                usd_balance = self.balance.get("USD", 0)
                if margin_required + fee > usd_balance:
                    raise ValueError(
                        f"Insufficient paper margin: need ${margin_required + fee:.2f}, have ${usd_balance:.2f}"
                    )
                self.balance["USD"] = usd_balance - margin_required - fee
                self.balance[long_key] = self.balance.get(long_key, 0) + quantity
                self.balance[f"LONG_PRICE_{base}"] = fill_price
        else:
            fill_price = price * (1 - SIMULATED_SLIPPAGE)
            notional = quantity * fill_price
            fee = notional * self._fee_rate
            margin_required = notional / leverage

            if self.balance.get(long_key, 0) >= quantity * 0.99:
                # Closing a long — return margin, apply PnL
                open_price_key = f"LONG_PRICE_{base}"
                open_price = self.balance.get(open_price_key, fill_price)
                pnl = (fill_price - open_price) * quantity
                open_margin = (open_price * quantity) / leverage
                self.balance["USD"] = self.balance.get("USD", 0) + open_margin + pnl - fee
                self.balance[long_key] = max(0, self.balance[long_key] - quantity)
                if self.balance[long_key] == 0:
                    self.balance.pop(open_price_key, None)
            else:
                # Opening a short
                usd_balance = self.balance.get("USD", 0)
                if margin_required + fee > usd_balance:
                    raise ValueError(
                        f"Insufficient paper margin: need ${margin_required + fee:.2f}, have ${usd_balance:.2f}"
                    )
                self.balance["USD"] = usd_balance - margin_required - fee
                self.balance[short_key] = self.balance.get(short_key, 0) + quantity
                self.balance[f"SHORT_PRICE_{base}"] = fill_price

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
        self.order_history.append({
            "order_id": order_id, "symbol": symbol, "side": side,
            "quantity": quantity, "price": fill_price, "fee": fee,
        })
        logger.info(
            "paper_futures_order",
            side=side, symbol=symbol, quantity=quantity,
            price=fill_price, fee=fee, leverage=leverage,
            usd_balance=self.balance.get("USD", 0),
        )
        return result

    async def get_balance(self) -> dict[str, float]:
        return {k: v for k, v in self.balance.items() if v > 0}

    async def get_tradeable_pairs(
        self,
        min_volume_usd: float = 50_000,
        quote_currencies: list[str] | None = None,
        **kwargs,
    ) -> list[str]:
        """Delegate to live exchange."""
        return await self.live.get_tradeable_pairs(min_volume_usd, quote_currencies, **kwargs)

    def get_24h_volume(self, symbol: str) -> float:
        """Delegate to live exchange."""
        return self.live.get_24h_volume(symbol)

    async def fetch_trades(self, symbol: str, limit: int = 1000) -> list[dict]:
        """Delegate to live exchange for real trade data (footprint analysis)."""
        return await self.live.fetch_trades(symbol, limit=limit)

    async def fetch_open_interest(self, symbol: str) -> dict:
        """Delegate to live exchange."""
        return await self.live.fetch_open_interest(symbol)

    async def fetch_funding_rate(self, symbol: str) -> dict:
        """Delegate to live exchange."""
        return await self.live.fetch_funding_rate(symbol)

    async def fetch_long_short_ratio(self, symbol: str) -> dict:
        """Delegate to live exchange."""
        return await self.live.fetch_long_short_ratio(symbol)

    @property
    def exchange_name(self) -> str:
        return getattr(self.live, "exchange_name", "unknown")

    @property
    def taker_fee_rate(self) -> float:
        return self._fee_rate

    @property
    def min_order_usd(self) -> float:
        return getattr(self.live, "min_order_usd", 5.0)

    @property
    def account_type(self) -> str:
        return self._account_type

    @property
    def leverage(self) -> int:
        return self._leverage

    @property
    def market_info(self):
        """Delegate to live exchange for market metadata."""
        return getattr(self.live, "market_info", None)

    async def get_position_risk(self, symbol: str) -> dict:
        """Simulated position risk for futures paper trading."""
        if self._account_type != "futures":
            return {}
        base = parse_symbol_base(symbol)
        for prefix, direction in [("LONG_", "long"), ("SHORT_", "short")]:
            key = f"{prefix}{base}"
            qty = self.balance.get(key, 0)
            if qty > 0:
                open_price = self.balance.get(f"{prefix}PRICE_{base}", 0)
                if open_price <= 0:
                    continue
                notional = qty * open_price
                margin_used = notional / self._leverage
                if direction == "long":
                    liq_price = open_price * (1 - 1 / self._leverage * 0.95)
                else:
                    liq_price = open_price * (1 + 1 / self._leverage * 0.95)
                return {
                    "liquidation_price": liq_price,
                    "margin_used": margin_used,
                    "notional": notional,
                }
        return {}

    async def set_leverage(self, symbol: str) -> None:
        """No-op for paper trading."""
        pass

    async def close(self) -> None:
        await self.live.close()
