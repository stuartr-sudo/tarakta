from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class MarketConfig(BaseModel):
    """Configuration for one market connector (crypto, stocks, commodities, etc.)."""
    connector: str = ""                 # "binance_futures", "yfinance_stocks", "yfinance_commodities", "alpaca"
    enabled: bool = True
    api_key: str = ""
    api_secret: str = ""
    account_type: str = "spot"          # connector-specific
    leverage: int = 1
    margin_mode: str = "isolated"
    # Market-specific overrides
    min_volume_usd: float = 20_000_000
    scan_interval_minutes: int = 15
    quality_filter: bool = True
    quote_currencies: list[str] = Field(default_factory=lambda: ["USDT"])
    initial_balance: float = 10000.0
    # Symbol universe (for stocks/commodities — the equivalent of QUALITY_BASES)
    symbol_universe: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    # Exchange (Binance — legacy flat fields, still work for single-market crypto)
    exchange_name: str = "binance"
    binance_api_key: str = ""
    binance_api_secret: str = ""

    # Supabase
    supabase_url: str = ""
    supabase_key: str = ""

    # Dashboard
    dashboard_username: str = "admin"
    dashboard_password_hash: str = ""
    viewer_username: str = "viewer"
    viewer_password_hash: str = ""
    session_secret: str = "change-me-to-a-random-32-char-string"

    # Account type
    account_type: Literal["spot", "margin", "futures"] = "spot"
    leverage: int = 10  # 1x–10x (futures/margin)
    margin_mode: Literal["isolated", "cross"] = "isolated"  # futures only

    # Trading
    trading_mode: Literal["paper", "live"] = "paper"
    initial_balance: float = 10000.0
    entry_threshold: float = 60.0  # Requires sweep + displacement at minimum (pullback is bonus)
    max_risk_pct: float = 0.02  # Max 2% of balance lost per trade (SL distance)
    max_position_pct: float = 0.25  # Max 25% of balance allocated per trade (margin)
    max_exposure_pct: float = 1.0  # Allow full budget deployment across positions
    max_concurrent: int = 10  # Max concurrent open positions (0 = unlimited)
    max_sector_positions: int = 3  # Max positions per sector/category (0 = unlimited)
    max_daily_drawdown: float = 0.10
    circuit_breaker_pct: float = 0.15
    min_rr_ratio: float = 2.0  # Minimum 2:1 reward-to-risk
    sl_buffer: float = 0.03  # 3% SL buffer beyond sweep level (wick protection)
    min_sl_pct: float = 0.02  # Minimum SL distance = 2% of entry price
    max_sl_pct: float = 0.05  # Maximum SL distance = 5% of entry (skip trades needing wider stops)
    cooldown_hours: float = 2.0  # Cooldown after stop-loss before re-entering same symbol
    max_daily_trades: int = 15  # Allow more trades to deploy full balance
    min_trade_usd: float = 150.0  # Minimum margin per trade ($150 × leverage = min notional)

    # Progressive take-profit tiers
    # TP1 hit → close 33%, move SL to breakeven
    # TP2 hit → close 33%, move SL to TP1 price
    # TP3    → remaining 34% via trailing stop
    tp_tiers_enabled: bool = True
    tp1_rr: float = 1.0     # TP1 at 1R
    tp2_rr: float = 2.0     # TP2 at 2R
    tp1_pct: float = 0.33   # close 33% at TP1
    tp2_pct: float = 0.33   # close 33% at TP2
    tp3_pct: float = 0.34   # remaining 34% via trailing stop
    move_sl_to_be_after_tp1: bool = True  # move SL to breakeven after TP1

    # Pullback entry timing
    pullback_min_retracement: float = 0.20  # Min retracement to consider pullback valid
    pullback_max_retracement: float = 0.78  # Max retracement before setup is invalid

    # Trailing stop (Travel phase)
    trailing_activation_rr: float = 0.5  # Activate trailing after 0.5R profit
    trailing_atr_multiplier: float = 1.5  # Trail at 1.5x ATR from high water mark

    # Early breakeven protection — move SL to entry price once trade is 0.5R in profit
    breakeven_activation_rr: float = 0.5  # Move SL to entry at 0.5R profit (before TP1)

    # Stale trade auto-close — close losing trades that haven't worked out
    max_hold_hours: float = 4.0  # Auto-close trades open longer than this IF in negative
    stale_close_below_rr: float = 0.0  # Only auto-close if trade is in the red (negative PnL)

    # Weekly cycle — Fake Move Monday & Mid-Week Reversal (ICT concepts)
    weekly_cycle_enabled: bool = True  # Master toggle for weekly cycle features
    monday_manipulation_penalty: float = 15.0  # Score penalty during Monday manipulation window
    monday_manipulation_hours: float = 8.0  # Hours after weekly open (Mon 00:00 UTC) to apply penalty
    midweek_reversal_bonus: float = 10.0  # Bonus for counter-trend signals on Wed/Thu

    # Market-level cross-reference filters
    btc_macro_gate_enabled: bool = True  # Hard gate: only trade with BTC trend
    market_breadth_enabled: bool = True  # Block minority-direction signals
    market_breadth_threshold: float = 0.70  # 70%+ signals in one direction → filter minority
    funding_gate_enabled: bool = True  # Block signals in crowded funding direction
    funding_gate_threshold: float = 0.0005  # 0.05% per 8h = extreme funding
    signal_persistence_scans: int = 2  # Signal must appear in N consecutive scans
    max_per_correlation_cluster: int = 2  # Max positions per correlated coin group

    # Signal reversal — disabled for Trade Travel Chill (no reversals, accept the loss)
    reversal_enabled: bool = False
    reversal_min_score: float = 70.0       # Legacy, inactive
    reversal_min_hold_minutes: int = 60    # Legacy, inactive
    reversal_cooldown_minutes: int = 120   # Legacy, inactive

    # Scanning
    scan_interval_minutes: int = 15  # Scan every 15 min to catch pullback entries
    min_volume_usd: float = 20_000_000  # $20M min 24h volume — filters to top ~100 coins
    quality_filter: bool = True  # Only scan established coins (QUALITY_BASES whitelist)
    max_position_volume_pct: float = 0.001  # Position size must be < 0.1% of 24h volume
    max_spread_pct: float = 0.002  # Max 0.2% bid-ask spread — skip illiquid pairs
    min_ob_depth_usd: float = 1000.0  # Min $1,000 depth at best bid/ask — ensures exit liquidity
    quote_currencies: list[str] = Field(default_factory=lambda: ["USD", "USDT"])

    # Hugging Face Inference API
    hf_api_token: str = ""  # For FinBERT sentiment + zero-shot classification

    # LLM Split Test (Claude trade analyst)
    llm_enabled: bool = False
    llm_api_key: str = ""  # Anthropic API key
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_timeout_seconds: float = 15.0
    llm_split_ratio: float = 0.5  # 0.5 = 50% of signals go through LLM
    llm_min_confidence: float = 40.0  # Reject approvals below this confidence (0-100)
    llm_fallback_approve: bool = True  # If API fails, approve trade by default

    # Dynamic strategy weights
    dynamic_weights_enabled: bool = False  # Adjust confluence weights based on trade outcomes

    # Flipped shadow bot — DISABLED (not producing results)
    flipped_enabled: bool = False
    flipped_leverage: int = 5  # Higher leverage for flipped trades
    flipped_sl_buffer: float = 0.03  # 3% SL buffer — extra room for wicks
    flipped_min_sl_pct: float = 0.02  # Minimum SL distance = 2% of entry price
    flipped_initial_balance: float = 10000.0  # Separate paper balance (disabled)
    flipped_scan_interval_minutes: int = 15  # Independent scan cycle (not tied to main bot)
    flipped_max_position_pct: float = 0.15  # 15% of balance as margin per trade (bigger than main)
    flipped_max_risk_pct: float = 0.15  # 15% of balance at risk per trade (bigger than main)
    flipped_min_rr_ratio: float = 2.0  # Looser R:R for flipped (main uses 3.0)

    # Custom configurable bot — second shadow bot with adjustable direction & margin
    custom_enabled: bool = True
    custom_initial_balance: float = 10000.0
    custom_leverage: int = 10
    custom_flip_direction: bool = True       # Default starts flipped (user can toggle)
    custom_margin_pct: float = 0.15          # 15% default (user can slide 5%–40%)
    custom_flip_mode: str = "smart_flip"     # "always_flip" | "smart_flip" | "normal"
    custom_flip_threshold: float = 0.50      # 0.0–1.0 — only flip when sweep prob > this
    custom_scan_interval_minutes: int = 20   # Offset from flipped's 15min

    # Hyper-Watchlist Monitor — promotes near-miss signals to a fast 5m monitor
    watchlist_enabled: bool = True
    watchlist_monitor_interval_seconds: int = 150   # Check every 2.5 min
    watchlist_expiry_hours: float = 3.0             # Release after 3 hours
    watchlist_max_size: int = 10                    # Max symbols to hyper-watch
    watchlist_min_score: float = 35.0               # Must have at least a sweep (35 pts)

    # OTE Entry Refinement — wait for optimal pullback on 5m before entering
    # Sweep signals: wait for price to retrace into OTE zone (50-79% Fib)
    # Breakout signals: wait for price to retest the breakout level + bounce
    entry_refiner_enabled: bool = True
    entry_refiner_check_interval_seconds: int = 60   # Check every 60s in monitor loop
    entry_refiner_expiry_minutes: float = 30.0       # Max wait for 5m confirmation
    entry_refiner_max_queue: int = 5                  # Max signals queued for refinement
    ote_min_retracement: float = 0.50   # OTE zone starts at 50% Fibonacci retracement
    ote_max_retracement: float = 0.79   # OTE zone ends at 79% (beyond = setup failed)
    ote_skip_on_expiry: bool = True     # No pullback within window = skip trade (don't chase)

    # Market Consensus Check — portfolio + BTC alignment before entry
    consensus_enabled: bool = True
    consensus_portfolio_penalty: float = 10.0    # Score penalty when portfolio bias alone disagrees
    consensus_btc_penalty: float = 15.0          # Total penalty when BTC also disagrees
    consensus_monitor_expiry_minutes: float = 30.0  # How long to monitor before expiring
    consensus_min_positions: int = 3             # Min open positions before consensus applies
    consensus_profitable_threshold: float = 0.0  # P&L threshold to count as "profitable"
    consensus_max_queue: int = 10                # Max signals in consensus monitor

    # Multi-market configuration
    # Each key is a market name like "crypto", "stocks", "commodities"
    # Populated from MARKET_* env vars or left empty for backward-compat (uses legacy flat fields)
    markets: dict[str, MarketConfig] = Field(default_factory=dict)

    # Extra market env vars (parsed in __init__)
    market_stocks_connector: str = ""
    market_stocks_symbol_universe: str = ""  # Comma-separated: "AAPL,MSFT,GOOGL"
    market_stocks_initial_balance: float = 10000.0
    market_commodities_connector: str = ""
    market_commodities_symbol_universe: str = ""  # Comma-separated: "GC=F,SI=F,CL=F"
    market_commodities_initial_balance: float = 10000.0

    # One-time force reset — set FORCE_RESET=true to wipe all data on next startup
    force_reset: bool = False

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    # Server
    port: int = 8080

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.leverage < 1:
            self.leverage = 1
        if self.leverage > 10:
            self.leverage = 10
        if self.account_type == "spot":
            self.leverage = 1

        # Build markets dict from env vars if not explicitly set
        self._build_markets()

    def _build_markets(self) -> None:
        """Auto-construct market configs from flat env vars."""
        # Always include crypto from legacy flat fields (backward compat)
        if "crypto" not in self.markets and self.binance_api_key:
            account_type = self.account_type
            connector = f"binance_{account_type}" if account_type != "spot" else "binance_spot"
            self.markets["crypto"] = MarketConfig(
                connector=connector,
                api_key=self.binance_api_key,
                api_secret=self.binance_api_secret,
                account_type=account_type,
                leverage=self.leverage,
                margin_mode=self.margin_mode,
                min_volume_usd=self.min_volume_usd,
                scan_interval_minutes=self.scan_interval_minutes,
                quality_filter=self.quality_filter,
                quote_currencies=self.quote_currencies,
                initial_balance=self.initial_balance,
            )

        # Stocks from MARKET_STOCKS_* env vars
        # Inherit global leverage + account_type so paper trading simulates margin
        if "stocks" not in self.markets and self.market_stocks_connector:
            universe = [s.strip() for s in self.market_stocks_symbol_universe.split(",") if s.strip()]
            self.markets["stocks"] = MarketConfig(
                connector=self.market_stocks_connector,
                symbol_universe=universe,
                initial_balance=self.market_stocks_initial_balance,
                leverage=self.leverage,           # Inherit global leverage (e.g. 10x)
                account_type=self.account_type,   # Inherit global account type (e.g. "futures")
                scan_interval_minutes=15,
                quality_filter=False,  # Stocks use symbol_universe instead
                min_volume_usd=0,      # No volume filter for stocks
            )

        # Commodities from MARKET_COMMODITIES_* env vars
        # Inherit global leverage + account_type so paper trading simulates margin
        if "commodities" not in self.markets and self.market_commodities_connector:
            universe = [s.strip() for s in self.market_commodities_symbol_universe.split(",") if s.strip()]
            self.markets["commodities"] = MarketConfig(
                connector=self.market_commodities_connector,
                symbol_universe=universe,
                initial_balance=self.market_commodities_initial_balance,
                leverage=self.leverage,           # Inherit global leverage (e.g. 10x)
                account_type=self.account_type,   # Inherit global account type (e.g. "futures")
                scan_interval_minutes=15,
                quality_filter=False,
                min_volume_usd=0,
            )
