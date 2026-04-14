"""Tool -> Endpoint mapping for the review widget."""

TOOL_ENDPOINT_MAP: dict[str, list[str]] = {
    # Trading Engine
    "Scanner": [
        "Signal detection",
        "SMC confluence scoring",
        "Volume analysis",
        "Candle cache",
    ],
    "Agent 1 (Entry Analyst)": [
        "Entry analysis",
        "Tool-use chain",
        "SL/TP suggestions",
        "Context building",
    ],
    "Agent 2 (Refiner)": [
        "ENTER/WAIT decisions",
        "Order book analysis",
        "Footprint data",
        "Advisor context injection",
    ],
    "Agent 3 (Position Manager)": [
        "SL tightening",
        "TP extension",
        "Position monitoring",
    ],
    "Risk Manager": [
        "Drawdown limits",
        "Circuit breaker",
        "Max concurrent trades",
        "Position sizing",
    ],
    "Trade Executor": [
        "Order placement",
        "Binance API",
        "Leverage management",
        "Error handling",
    ],
    "Lesson Generator": [
        "Post-trade analysis",
        "trade_lessons table",
        "Feedback loop injection",
    ],
    # Dashboard
    "Overview Page": [
        "Portfolio chart",
        "PnL display",
        "Balance tracking",
    ],
    "Trades Page": [
        "Trade history",
        "Filtering",
        "Export",
    ],
    "Signals Page": [
        "Signal list",
        "Scoring display",
    ],
    "Settings Page": [
        "Engine config",
        "Risk parameters",
        "Scanning parameters",
    ],
    "API Routes": [
        "REST endpoints",
        "Authentication",
        "Session management",
    ],
    "Charts/Visualization": [
        "TradingView widgets",
        "Performance charts",
    ],
    # Data & Infrastructure
    "Repository Layer": [
        "Supabase queries",
        "Data access patterns",
    ],
    "Candle Cache": [
        "OHLCV data fetching",
        "Caching",
        "Staleness",
    ],
    "Exchange Connection": [
        "CCXT async",
        "Binance API",
        "Rate limiting",
        "Reconnection",
    ],
    "RAG System": [
        "Knowledge chunks",
        "Hybrid search",
        "Trade history retrieval",
    ],
    # Advisory & Intelligence
    "Daily Advisor": [
        "Claude Agent SDK",
        "Insight generation",
        "Context injection",
    ],
    "MM Method Engine": [
        "Parallel trading engine",
        "State management",
    ],
    "Portfolio Snapshots": [
        "Performance tracking",
        "Equity curve",
    ],
    # Other
    "Auth": [
        "Login",
        "Session management",
        "Password hashing",
    ],
    "Deployment": [
        "Fly.io",
        "Docker",
        "Health checks",
        "OOM management",
    ],
    "Logging": [
        "Structured logging",
        "Log levels",
        "Error tracking",
    ],
    "General": [
        "UI/CSS",
        "Navigation",
        "Configuration",
        "Environment variables",
    ],
}

REQUEST_TYPES = [
    ("bug", "Bug"),
    ("question", "Question"),
    ("improvement", "Improvement"),
    ("console_error", "Console/Log Error"),
    ("change_request", "Change Request"),
    ("strategy_review", "Strategy Review"),
    ("prompt_review", "Prompt Review"),
    ("claude_md_update", "CLAUDE.md Update"),
]
