from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Exchange (Binance only)
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
    leverage: int = 1  # 1x–10x (futures/margin)
    margin_mode: Literal["isolated", "cross"] = "isolated"  # futures only

    # Trading
    trading_mode: Literal["paper", "live"] = "paper"
    initial_balance: float = 2000.0
    entry_threshold: float = 70.0  # Requires sweep + displacement at minimum
    max_risk_pct: float = 0.10  # Max 10% of balance lost per trade (SL distance)
    max_position_pct: float = 0.25  # Max 25% of balance allocated per trade (margin)
    max_exposure_pct: float = 1.0  # Allow full budget deployment across positions
    max_concurrent: int = 100
    max_daily_drawdown: float = 0.10
    circuit_breaker_pct: float = 0.15
    min_rr_ratio: float = 2.0  # Minimum 2:1 reward-to-risk
    cooldown_hours: float = 2.0  # Cooldown after stop-loss before re-entering same symbol
    max_daily_trades: int = 15  # Allow more trades to deploy full balance

    # Progressive take-profit tiers (disabled for Trade Travel Chill)
    tp_tiers_enabled: bool = False  # Travel: single exit via trailing stop
    tp1_rr: float = 1.0     # TP1 at 1R (legacy, inactive)
    tp2_rr: float = 2.0     # TP2 at 2R (legacy, inactive)
    tp1_pct: float = 0.33   # close 33% at TP1 (legacy, inactive)
    tp2_pct: float = 0.33   # close 33% at TP2 (legacy, inactive)
    tp3_pct: float = 0.34   # remaining 34% via trailing stop (legacy, inactive)
    move_sl_to_be_after_tp1: bool = True  # move SL to breakeven after TP1 (legacy, inactive)

    # Pullback entry timing
    pullback_min_retracement: float = 0.20  # Min retracement to consider pullback valid
    pullback_max_retracement: float = 0.78  # Max retracement before setup is invalid

    # Trailing stop (Travel phase)
    trailing_activation_rr: float = 2.0  # Activate trailing after 2R profit
    trailing_atr_multiplier: float = 1.5  # Trail at 1.5x ATR from high water mark

    # Signal reversal — disabled for Trade Travel Chill (no reversals, accept the loss)
    reversal_enabled: bool = False
    reversal_min_score: float = 70.0       # Legacy, inactive
    reversal_min_hold_minutes: int = 60    # Legacy, inactive
    reversal_cooldown_minutes: int = 120   # Legacy, inactive

    # Scanning
    scan_interval_minutes: int = 15  # Scan every 15 min to catch pullback entries
    min_volume_usd: float = 1_000_000
    max_position_volume_pct: float = 0.001  # Position size must be < 0.1% of 24h volume
    max_spread_pct: float = 0.003  # Max 0.3% bid-ask spread — skip illiquid pairs
    min_ob_depth_usd: float = 50.0  # Min $50 depth at best bid/ask level
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
