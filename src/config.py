from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Exchange
    kraken_api_key: str = ""
    kraken_api_secret: str = ""

    # Supabase
    supabase_url: str = ""
    supabase_key: str = ""

    # Dashboard
    dashboard_username: str = "admin"
    dashboard_password_hash: str = ""
    session_secret: str = "change-me-to-a-random-32-char-string"

    # Trading
    trading_mode: Literal["paper", "live"] = "paper"
    initial_balance: float = 100.0
    entry_threshold: float = 65.0
    max_risk_pct: float = 0.10  # Max 10% of balance lost per trade (SL distance)
    max_position_pct: float = 0.10  # Max 10% of balance allocated per trade
    max_exposure_pct: float = 1.0  # Allow full budget deployment across positions
    max_concurrent: int = 20
    max_daily_drawdown: float = 0.10
    circuit_breaker_pct: float = 0.15
    min_rr_ratio: float = 2.0
    cooldown_hours: float = 4.0  # Hours to wait before re-entering a symbol after SL hit

    # Scanning
    scan_interval_minutes: int = 15
    min_volume_usd: float = 1_000_000
    max_position_volume_pct: float = 0.001  # Position size must be < 0.1% of 24h volume
    quote_currencies: list[str] = Field(default_factory=lambda: ["USD", "USDT"])

    # Hugging Face Inference API
    hf_api_token: str = ""  # For FinBERT sentiment + zero-shot classification

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    # Server
    port: int = 8080

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
