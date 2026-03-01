from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    filled_quantity: float
    avg_price: float
    fee: float
    status: str


@dataclass
class Position:
    trade_id: str
    symbol: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float | None
    high_water_mark: float
    entry_time: datetime
    cost_usd: float = 0.0
    direction: str = "long"  # "long" or "short"


@dataclass
class PositionSize:
    valid: bool
    quantity: float = 0.0
    cost_usd: float = 0.0
    risk_usd: float = 0.0
    risk_pct: float = 0.0
    reason: str = ""


@dataclass
class TradeValidation:
    allowed: bool
    reason: str = ""


@dataclass
class ExitSignal:
    symbol: str
    reason: str  # "sl_hit", "tp_hit", "trailing_stop", "circuit_breaker"
    price: float


@dataclass
class SweepEvent:
    level: float
    direction: str  # "bullish_sweep" or "bearish_sweep"
    candle_idx: int


@dataclass
class OrderBlock:
    direction: str  # "bullish" or "bearish"
    top: float
    bottom: float
    volume: float
    strength: float
    candle_idx: int


@dataclass
class FairValueGap:
    direction: str  # "bullish" or "bearish"
    top: float
    bottom: float
    candle_idx: int
    midpoint: float


@dataclass
class MarketStructureResult:
    trend: str  # "bullish", "bearish", "ranging"
    key_levels: dict
    last_bos_direction: int | None
    last_choch_direction: int | None
    structure_strength: float
    swing_highs_lows: object = None  # pandas DataFrame
    bos_choch: object = None  # pandas DataFrame


@dataclass
class LiquidityResult:
    active_pools: list
    recent_sweeps: list[SweepEvent]
    nearest_buy_liquidity: float | None
    nearest_sell_liquidity: float | None
    sweep_detected_recently: bool


@dataclass
class OrderBlockResult:
    active_order_blocks: list[OrderBlock]
    price_in_order_block: OrderBlock | None
    nearest_bullish_ob: OrderBlock | None
    nearest_bearish_ob: OrderBlock | None


@dataclass
class FVGResult:
    active_fvgs: list[FairValueGap]
    price_in_fvg: FairValueGap | None
    nearest_bullish_fvg: FairValueGap | None
    nearest_bearish_fvg: FairValueGap | None


@dataclass
class SignalCandidate:
    score: float
    direction: str | None
    reasons: list[str] = field(default_factory=list)
    symbol: str = ""
    entry_price: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ob_context: OrderBlock | None = None
    fvg_context: FairValueGap | None = None
    key_levels: dict = field(default_factory=dict)


@dataclass
class CircuitBreakerStatus:
    triggered: bool
    reason: str = ""
    severity: str = ""  # "warning" or "critical"
    resume_at: datetime | None = None
