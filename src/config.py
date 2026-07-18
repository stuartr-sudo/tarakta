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
    # Runtime role. "mm" runs the normal scanner. "inverse_mirror" runs a
    # separate bot that mirrors another instance's MM trades in the opposite
    # direction.
    bot_role: Literal["mm", "inverse_mirror"] = "mm"

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
    mm_dashboard_strategy: str = "mm_method"
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
    # Course Lesson 53 distinguishes the absolute 1.4R "don't get out of
    # bed" floor from the standard 3R minimum. Linda cascade logic may
    # intentionally lower this per-signal; the normal runtime default is 3R.
    mm_min_rr: float = 3.0
    # Grade C threshold. Historical rows may have lower thresholds, but new
    # runtime decisions should use percent-based Grade C by default.
    mm_min_confluence: float = 40.0
    mm_min_formation_quality: float = 0.4
    mm_max_sl_pct: float = 5.0
    # Empirically validated deterministic gate from docs/STATUS_2026-04-28.
    # 0 still disables it when deliberately overridden.
    mm_gate_threshold: int = 3
    mm_cooldown_hours: float = 4.0
    mm_leverage: int = 10
    # Max distance from entry to TP1, as % of entry. Engineering cap, NOT
    # an explicit course rule — the course doesn't give a numeric bound
    # on target distance. The intent is to reject 1H/intraday formation
    # entries whose nearest natural target is a multi-week structural
    # level (e.g. BTC 2026-04-20 where every EMA was below entry and the
    # cascade landed on a vector 22% away). Disable by setting to 0.
    # Tune up if you see rejected setups that would have worked; tune
    # down if wide-target trades keep slipping through.
    mm_max_tp1_distance_pct: float = 10.0
    # Max slippage (%) from the 2nd-peak wick ("retest level" per course
    # Lesson 20 / 47) before we skip a setup. 0 disables. See engine
    # comment at the entry-price block for the trade-data rationale.
    mm_max_entry_slippage_pct: float = 1.0
    # Scratch rule breakeven-distance threshold in R-multiples.
    # Course Lesson 13 [44:00] says a correct trade should move into
    # enough profit within two hours to move stop to breakeven. The
    # engine therefore checks whether peak MFE reached the BE ladder's
    # minimum distance, not an arbitrary "substantial profit" proxy.
    # Set to 0 to effectively disable the scratch rule.
    mm_scratch_be_distance_r: float = 0.2
    # Compatibility alias for older env/settings rows. Prefer
    # mm_scratch_be_distance_r for new code.
    mm_scratch_mfe_threshold_r: float = 0.2
    # 4H/daily formation scratch window. Lesson 13 [102:30] only gives
    # directional permission to hold longer for 4H/daily structures; 2
    # closed 4H bars is the conservative inferred default.
    mm_scratch_window_4h_bars: int = 2
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
    # only supported mode. This parameter replaces
    # mm_sanity_agent_thinking_budget.
    mm_sanity_agent_effort: str = "high"
    mm_sanity_agent_timeout_s: float = 20.0
    mm_sanity_agent_min_confidence: float = 0.0  # 0 = honour every VETO (no shadow)
    mm_sanity_agent_monthly_budget_usd: float = 600.0
    # Tier 2 learning-loop lookback: how many days of past APPROVE
    # outcomes are aggregated and shown to the agent as "your own
    # track record" per Rubric 8. 14 balances signal (enough samples)
    # and recency (regime changes). 0 disables the learning loop
    # (reverts to pre-Tier-2 behaviour).
    mm_sanity_agent_outcome_lookback_days: int = 14
    # Minimum number of CLOSED samples (wins+losses+scratches) a
    # GRADE|HTF_4H profile bucket must have before it's surfaced to the
    # agent in Rubric 8. Smaller buckets get filtered to an
    # "(insufficient data — skip Rubric 8)" line. Added 2026-04-22 after
    # v44 shipped rubric_v=2 and the agent vetoed ~100% of setups on
    # 1W/4L (n=5) profiles — classic small-sample overfitting.
    # 20 = roughly the sample size at which a 75% vs 25% win rate is
    # distinguishable from variance at 95% CI. Lower values trade
    # statistical rigour for faster feedback-loop activation.
    mm_sanity_agent_outcome_min_n: int = 20
    # Per-setup decision cache (P2 fix 2026-04-22). The 1H formation
    # detector re-generates the same setup every 5-min scan, so without
    # a cache we pay for the same Opus 4.7 call repeatedly. On
    # 2026-04-21 we logged 82 identical VETOs on one DOGE long setup in
    # 6h. This cache keys by
    # (symbol, direction, formation_variant, round(entry_price, 4))
    # and returns the prior verdict on subsequent calls within the TTL.
    # cache_ttl_seconds: 1800s (30 min) — long enough to collapse the
    # 5-min rescan storm, short enough that a stale veto can't outlive
    # a regime change. Set to 0 to disable the cache entirely.
    mm_sanity_agent_cache_ttl_seconds: float = 1800.0
    # price_drift_pct: any current vs cached entry-price drift above
    # this threshold invalidates the cache even within TTL. 0.5%
    # matches the course's "1% retest tolerance" halved — well inside
    # noise for a real retest, but catches a genuinely new formation
    # price. Set high (e.g. 100) to effectively disable drift check.
    mm_sanity_agent_cache_price_drift_pct: float = 0.5

    # MM Agent Committee — disabled by default. Shadow mode returns APPROVE
    # to the engine while logging what the committee would have done.
    mm_committee_enabled: bool = False
    mm_committee_mode: Literal["shadow", "veto"] = "shadow"
    mm_committee_specialist_model: str = "claude-haiku-4-5-20251001"
    mm_committee_head_trader_model: str = "claude-sonnet-4-6"
    mm_committee_escalation_model: str = "claude-opus-4-8"
    mm_committee_timeout_s: float = 30.0
    mm_committee_monthly_budget_usd: float = 600.0
    # Claude Code CLI fallback — used when anthropic_api_key is empty. Runs
    # committee calls through the locally-authenticated `claude` binary
    # (subscription login), so no API key is required. cli_path empty =
    # autodetect via PATH + Homebrew locations. cli_timeout_s is per model
    # call; the committee-wide deadline scales off it (2 stages + slack).
    mm_committee_cli_enabled: bool = True
    mm_committee_cli_path: str = ""
    mm_committee_cli_timeout_s: float = 120.0
    # Long-lived headless auth for the CLI backend, minted by
    # `claude setup-token` (browser flow, ~30s, user-run). Needed because a
    # daemonised bot cannot rely on the interactive Keychain login staying
    # fresh — observed 2026-07-18: Keychain CLI credentials 401'd while the
    # desktop app's own session worked. Also the auth story for Fly later
    # (set as a secret). Empty = rely on ambient CLI login.
    claude_code_oauth_token: str = ""

    # Inverse mirror bot. This mode does not scan for setups. It follows the
    # source instance's persisted trades and opens/closes the opposite side in
    # this instance.
    inverse_source_instance_id: str = "tarakta-mm"
    inverse_instance_id: str = "tarakta-mm-inverse"
    inverse_source_strategy: str = "mm_method"
    inverse_strategy_tag: str = "mm_inverse"
    inverse_poll_interval_seconds: float = 15.0
    inverse_quantity_multiplier: float = 1.0
    inverse_trade_lookback: int = 500

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
