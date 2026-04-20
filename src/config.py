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
    # Hard-ceiling sanity backstop, NOT the real limit — the actual
    # constraint is mm_max_aggregate_risk_pct (aggregate-risk budget).
    # Raised 3 → 20 on 2026-04-20 (~size of majors universe) because the
    # old 3-cap was a human-attention limit from the course, not a
    # bot-appropriate one.
    mm_max_positions: int = 20
    # Aggregate open risk cap across ALL concurrent positions, as % of
    # account balance. Course rule is "1% per trade"; this expresses the
    # same principle at portfolio level. Default 5.0 allows ~5 open
    # trades at 1% risk each (or more if SLs are tight and notional-cap
    # shrinks per-trade risk below 1%). The engine refuses to open a
    # new trade when aggregate_open_risk + proposed_trade_risk > cap.
    mm_max_aggregate_risk_pct: float = 5.0
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
    # Adaptive thinking effort on Opus 4.7 / Sonnet 4.6. Accepted values:
    # "low" | "medium" | "high" | "max". Default "high" — this is a
    # money-critical judgement task, not a classification. Opus 4.7
    # rejects the legacy thinking={"type":"enabled","budget_tokens":N}
    # shape with a 400 (invalid_request_error); adaptive+effort is the
    # only supported mode. Hit live on 2026-04-20 00:12 UTC — the first
    # real setup of the week (NEAR long) approved via fail-open because
    # the agent errored. This parameter replaces mm_sanity_agent_thinking_budget.
    mm_sanity_agent_effort: str = "high"
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
