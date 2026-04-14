from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    filled_quantity: float
    avg_price: float
    fee: float
    status: str
