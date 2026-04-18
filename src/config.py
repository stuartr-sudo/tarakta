from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class MarketConfig(BaseModel):
    """Configuration for the exchange connection."""
    connector: str = ""
    enabled: bool = True
    api_key: str = ""
    api_secret: str = ""
    account_type: str = "futures"
    leverage: int = 10
    margin_mode: str = "isolated"
    min_volume_usd: float = 5_000_000
    scan_interval_minutes: int = 5
    quality_filter: bool = True
    quote_currencies: list[str] = Field(default_factory=lambda: ["USDT"])
    initial_balance: float = 10000.0
    symbol_universe: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    # Instance isolation
    instance_id: str = "main"

    # Exchange (Binance)
    exchange_name: str = "binance"
    binance_api_key: str = ""
    binance_api_secret: str = ""

    # Supabase
    supabase_url: str = ""
    supabase_key: str = ""

    # Dashboard auth
    dashboard_username: str = "admin"
    dashboard_password_hash: str = ""
    viewer_username: str = "viewer"
    viewer_password_hash: str = ""
    session_secret: str = "change-me-to-a-random-32-char-string"

    # Account
    account_type: Literal["spot", "margin", "futures"] = "futures"
    leverage: int = 10
    margin_mode: Literal["isolated", "cross"] = "isolated"

    # Trading
    trading_mode: Literal["paper", "live"] = "paper"
    initial_balance: float = 10000.0

    # MM Method Engine
    mm_method_enabled: bool = True
    mm_scan_interval_minutes: float = 5.0
    mm_max_positions: int = 3
    mm_risk_per_trade_pct: float = 1.0
    mm_initial_balance: float = 10000.0
    # Pair selection — course says MM Method is a majors strategy. Separate
    # from the SMC engine's `min_volume_usd` so we don't disturb that.
    mm_min_volume_usd: float = 50_000_000  # 50M USD/24h — filters shitcoins
    # Majors-only is ON by default: course says MM Method is a majors
    # strategy (lessons 1-3, 53). Restricts to BTC/ETH/SOL/BNB/top-20.
    # Flip to False to loosen to the 50M-volume universe.
    mm_majors_only: bool = True

    # MM Sanity Agent (Agent 4) — LLM guardrail that reviews every MM setup
    # that survives the deterministic rules and vetoes ones that don't pass
    # a course-fluent sanity check. See docs/MM_SANITY_AGENT_DESIGN.md.
    anthropic_api_key: str = ""
    mm_sanity_agent_enabled: bool = True
    mm_sanity_agent_model: str = "claude-opus-4-7"
    mm_sanity_agent_fallback_model: str = "claude-sonnet-4-6"
    mm_sanity_agent_thinking_budget: int = 4000  # tokens reserved for extended thinking
    mm_sanity_agent_timeout_s: float = 20.0
    mm_sanity_agent_min_confidence: float = 0.0  # 0 = honour every VETO (no shadow)
    mm_sanity_agent_monthly_budget_usd: float = 600.0

    # Scanning defaults
    min_volume_usd: float = 5_000_000
    scan_interval_minutes: int = 5
    quality_filter: bool = True
    quote_currencies: list[str] = Field(default_factory=lambda: ["USDT"])

    # Multi-market configuration
    markets: dict[str, MarketConfig] = Field(default_factory=dict)

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

        # Build markets dict from env vars if not explicitly set
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
