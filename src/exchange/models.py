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
class TrancheFill:
    """Result of a single tranche in progressive order execution."""
    tranche_num: int
    order_id: str
    price: float           # fill price for this tranche
    quantity_requested: float
    quantity_filled: float
    fee: float
    status: str            # "filled", "partial", "missed", "aborted"


@dataclass
class ProgressiveFillResult:
    """Aggregated result of a progressive (multi-tranche) order."""
    total_requested: float
    total_filled: float
    total_missed: float
    vwap: float               # volume-weighted average fill price
    tranches: list[TrancheFill] = field(default_factory=list)
    total_fees: float = 0.0
    fully_filled: bool = False
    num_tranches_attempted: int = 0
    num_tranches_filled: int = 0
    abort_reason: str = ""    # "" if completed normally


@dataclass
class TakeProfitTier:
    """One tier of a progressive take-profit plan."""
    level: int              # 1, 2, or 3
    price: float | None     # TP target price (None for tier 3 = trailing-only)
    pct: float              # fraction of original quantity (0.33, 0.33, 0.34)
    quantity: float          # actual quantity for this tier (computed at entry)
    filled: bool = False
    fill_price: float = 0.0
    fill_time: datetime | None = None


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
    leverage: int = 1
    margin_used: float = 0.0  # actual margin locked (cost_usd / leverage)
    liquidation_price: float = 0.0
    # Progressive TP fields
    tp_tiers: list[TakeProfitTier] | None = None  # None = legacy single-TP mode
    original_quantity: float = 0.0   # quantity at entry (before partial closes)
    original_stop_loss: float = 0.0  # original SL before breakeven move
    current_tier: int = 0            # 0=no tiers hit, 1=TP1 hit, 2=TP2 hit
    confluence_score: float = 0.0    # confluence score at entry (for post-trade analysis)
    agent1_reasoning: str = ""       # Agent 1's thesis for Agent 3 context
    # Agent 3 latest decision (written by monitor, saved to DB by core)
    last_agent3_action: str = ""     # HOLD, TIGHTEN_SL, CLOSE_PARTIAL, CLOSE_FULL
    last_agent3_reasoning: str = ""  # Agent 3's analysis text
    agent3_confidence: float = 0.0   # 0-100
    last_agent3_sl: str = ""         # "old → new" for SL changes


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
    reason: str  # "sl_hit", "tp_hit", "tp1_hit", "tp2_hit", "trailing_stop", "circuit_breaker"
    price: float
    is_partial: bool = False       # True for TP1/TP2 partial exits
    partial_quantity: float = 0.0  # exact quantity to close for partial exits
    tier: int = 0                  # which TP tier (1, 2, or 0 for full exit)


@dataclass
class SweepEvent:
    level: float
    direction: str  # "bullish_sweep" or "bearish_sweep"
    candle_idx: int


@dataclass
class SweepResult:
    """Result of completed liquidity sweep detection.

    A completed sweep = price wicked through a level and closed back
    on the other side, meaning MMs grabbed the liquidity.
    """
    sweep_detected: bool
    sweep_direction: str | None    # "swing_low" (sell-side liquidity taken) or "swing_high" (buy-side liquidity taken)
    sweep_level: float             # Wick extreme (for SL placement)
    sweep_type: str | None         # "asian_low", "asian_high", "swing_low", "swing_high"
    target_level: float            # Opposite side liquidity (for TP)
    sweep_depth: float             # How far past the level
    htf_continuation: bool = False # True when trade direction was overridden by HTF context



@dataclass
class PullbackResult:
    """Result of pullback detection after a displacement move."""
    pullback_detected: bool
    retracement_pct: float          # 0.0-1.0: how far price has retraced
    displacement_open: float        # origin of the displacement candle
    thrust_extreme: float           # highest high (bullish) or lowest low (bearish) after displacement
    current_price: float
    optimal_entry: float            # suggested entry price
    pullback_status: str            # "waiting", "optimal", "failed"


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
class LeverageProfile:
    """Leverage intelligence for a futures symbol."""
    open_interest_usd: float          # Total OI in USD
    funding_rate: float               # Current 8h funding rate (e.g. 0.0001 = 0.01%)
    long_short_ratio: float | None    # Top traders L/S ratio (>1 = more longs)
    crowded_side: str | None          # "long" or "short" or None
    crowding_intensity: float         # 0.0-1.0 how extreme the crowding is
    funding_bias: str | None          # "long_pay" or "short_pay" or None
    liquidation_clusters: list[dict] = field(default_factory=list)
    nearest_long_liq: float = 0.0     # Closest long liquidation level below price
    nearest_short_liq: float = 0.0    # Closest short liquidation level above price
    sweep_aligns_with_crowding: bool = False
    judas_swing_probability: float = 0.0


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
    components: dict[str, float] = field(default_factory=dict)
    atr_1h: float = 0.0  # 14-period ATR on 1H candles for SL floor
    crt_result: object | None = None     # CRTResult from strategy.crt
    session_result: object | None = None  # SessionResult from strategy.sessions
    sweep_result: object | None = None   # SweepResult from strategy.sweep_detector
    leverage_profile: object | None = None  # LeverageProfile from strategy.leverage
    # Hyper-Watchlist fields
    watchlist_promoted: bool = False           # True if signal graduated from watchlist
    watchlist_duration_seconds: float = 0.0    # How long it was monitored before graduating
    htf_direction_cache: str | None = None     # Preserved HTF direction for watchlist re-scoring
    # Post-sweep entry refinement fields
    refined_entry: bool = False              # True if entry was refined on 5m
    refinement_duration_seconds: float = 0.0 # How long 5m monitoring took
    original_1h_price: float = 0.0           # The stale 1H close we avoided entering at
    # AI agent fields
    agent_target_entry: float | None = None  # Target entry price from agent WAIT_PULLBACK
    agent_entry_zone_high: float | None = None  # Agent-suggested entry zone upper bound
    agent_entry_zone_low: float | None = None   # Agent-suggested entry zone lower bound
    # 1H thrust data for refiner (persisted from scanner's PullbackResult)
    thrust_extreme_1h: float | None = None      # highest 1H high (bullish) or lowest 1H low (bearish) after displacement
    displacement_open_1h: float | None = None    # open of the 1H displacement candle
    # Fibonacci retracement levels (computed from displacement move for agent context)
    fibonacci_levels: dict = field(default_factory=dict)
    # Extended ICT/SMC context for AI agent (populated post-scoring, not used in formula)
    agent_context: dict = field(default_factory=dict)
    # Pullback plan for WAIT_PULLBACK entries (formal pending order contract)
    pullback_plan: PullbackPlan | None = None


@dataclass
class PullbackPlan:
    """Formal pending order plan for WAIT_PULLBACK entries.

    Created when Agent 1 returns WAIT_PULLBACK. Enforces zone boundaries,
    expiry, invalidation, and slippage limits at all entry paths.
    No fallback to market order — if any check fails, DO_NOT_ENTER.
    """

    zone_low: float
    zone_high: float
    created_at: datetime
    expires_at: datetime
    invalidation_level: float = 0.0   # Price that kills the setup (e.g., sweep low break)
    max_chase_bps: float = 3.0        # Max basis points of slippage beyond zone edge
    zone_tolerance_bps: float = 2.0   # Tiny tolerance for zone boundary check
    valid_for_candles: int = 6         # Number of 5m candles (6 = 30 min)
    direction: str = ""                # "swing_low" (long setup) or "swing_high" (short setup)
    limit_price: float = 0.0          # Computed zone-aware limit price
    original_suggested_entry: float = 0.0  # Agent 1's suggested_entry
    zone_updates: int = 0              # How many times Agent 2 adjusted zone
    # Phase gate: price must reach must_reach_price BEFORE pullback entry is allowed.
    # For longs: price must rise to this level first (then pull back to zone).
    # For shorts: price must drop to this level first (then bounce to zone).
    # 0.0 = no gate (pullback can start from current price area).
    must_reach_price: float = 0.0
    must_reach_price_reached: bool = False  # Flipped to True once price reaches gate

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    def price_in_zone(self, price: float) -> bool:
        """Check if price is inside zone with tolerance."""
        tol = self.zone_high * (self.zone_tolerance_bps / 10000)
        return (self.zone_low - tol) <= price <= (self.zone_high + tol)

    def slippage_ok(self, price: float) -> bool:
        """Check if price slippage from zone is within max_chase_bps."""
        if self.direction in ("bullish", "long", "swing_low"):
            # For longs, slippage = buying above zone_high
            if price <= self.zone_high:
                return True
            slip_bps = (price - self.zone_high) / self.zone_high * 10000
        else:
            # For shorts, slippage = selling below zone_low
            if price >= self.zone_low:
                return True
            slip_bps = (self.zone_low - price) / self.zone_low * 10000
        return slip_bps <= self.max_chase_bps

    def invalidation_hit(self, price: float) -> bool:
        """Check if price has broken the invalidation level."""
        if self.invalidation_level <= 0:
            return False
        if self.direction in ("bullish", "long", "swing_low"):
            return price < self.invalidation_level
        else:
            return price > self.invalidation_level

    def check_must_reach_price(self, price: float, tolerance_pct: float = 0.02) -> bool:
        """Check if price has reached the must_reach_price gate (with tolerance).

        Returns True if the gate has been reached (or no gate set).
        Once reached, must_reach_price_reached is permanently set to True.

        tolerance_pct: How close price needs to get (0.02 = within 2%).
            Example: must_reach=100, tolerance=0.02 → longs pass at 98+.
        """
        if self.must_reach_price <= 0 or self.must_reach_price_reached:
            return True
        if self.direction in ("bullish", "long", "swing_low"):
            # For longs: price must reach within tolerance of must_reach_price
            threshold = self.must_reach_price * (1 - tolerance_pct)
            if price >= threshold:
                self.must_reach_price_reached = True
                return True
        else:
            # For shorts: price must reach within tolerance of must_reach_price
            threshold = self.must_reach_price * (1 + tolerance_pct)
            if price <= threshold:
                self.must_reach_price_reached = True
                return True
        return False

    def pullback_allowed(self, price: float) -> bool:
        """Check if pullback entry is allowed — must_reach_price must have been reached first."""
        if self.must_reach_price <= 0:
            return True  # No gate set
        self.check_must_reach_price(price)
        return self.must_reach_price_reached

    def compute_limit_price(self) -> float:
        """Compute optimal limit price within zone.

        Longs: lower quarter of zone (buy low).
        Shorts: upper quarter of zone (sell high).
        """
        zone_range = self.zone_high - self.zone_low
        if self.direction in ("bullish", "long", "swing_low"):
            return self.zone_low + 0.25 * zone_range
        else:
            return self.zone_high - 0.25 * zone_range

    def zone_str(self) -> str:
        return f"{round(self.zone_low, 6)}-{round(self.zone_high, 6)}"


@dataclass
class CircuitBreakerStatus:
    triggered: bool
    reason: str = ""
    severity: str = ""  # "warning" or "critical"
    resume_at: datetime | None = None
