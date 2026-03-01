-- Tarakta Trading Bot - Initial Schema
-- Run this in your Supabase SQL editor

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    status          TEXT NOT NULL DEFAULT 'open'
                    CHECK (status IN ('open', 'closed', 'cancelled')),
    mode            TEXT NOT NULL CHECK (mode IN ('live', 'paper')),

    -- Entry
    entry_price     DECIMAL(20, 8) NOT NULL,
    entry_quantity  DECIMAL(20, 8) NOT NULL,
    entry_cost_usd  DECIMAL(12, 4) NOT NULL,
    entry_order_id  TEXT,
    entry_time      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Exit
    exit_price      DECIMAL(20, 8),
    exit_quantity   DECIMAL(20, 8),
    exit_order_id   TEXT,
    exit_time       TIMESTAMPTZ,
    exit_reason     TEXT,

    -- Risk params
    stop_loss       DECIMAL(20, 8) NOT NULL,
    take_profit     DECIMAL(20, 8),
    risk_usd        DECIMAL(12, 4) NOT NULL,
    risk_reward     DECIMAL(6, 2),

    -- Result
    pnl_usd         DECIMAL(12, 4),
    pnl_percent     DECIMAL(8, 4),
    fees_usd        DECIMAL(12, 4),

    -- Signal metadata
    confluence_score DECIMAL(5, 2) NOT NULL,
    signal_reasons  JSONB,
    timeframes_used JSONB,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_mode ON trades(mode);

-- Signals table
CREATE TABLE IF NOT EXISTS signals (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    score           DECIMAL(5, 2) NOT NULL,
    reasons         JSONB NOT NULL,
    components      JSONB NOT NULL,
    current_price   DECIMAL(20, 8) NOT NULL,
    acted_on        BOOLEAN NOT NULL DEFAULT FALSE,
    trade_id        UUID REFERENCES trades(id),
    scan_cycle      INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_score ON signals(score DESC);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);

-- Candle cache
CREATE TABLE IF NOT EXISTS candle_cache (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    open            DECIMAL(20, 8) NOT NULL,
    high            DECIMAL(20, 8) NOT NULL,
    low             DECIMAL(20, 8) NOT NULL,
    close           DECIMAL(20, 8) NOT NULL,
    volume          DECIMAL(20, 8) NOT NULL,
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_candles_lookup ON candle_cache(symbol, timeframe, timestamp DESC);

-- Portfolio snapshots
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    balance_usd     DECIMAL(12, 4) NOT NULL,
    equity_usd      DECIMAL(12, 4) NOT NULL,
    open_positions  INTEGER NOT NULL DEFAULT 0,
    daily_pnl_usd   DECIMAL(12, 4) NOT NULL DEFAULT 0,
    total_pnl_usd   DECIMAL(12, 4) NOT NULL DEFAULT 0,
    drawdown_pct    DECIMAL(8, 4) NOT NULL DEFAULT 0,
    peak_balance    DECIMAL(12, 4) NOT NULL,
    mode            TEXT NOT NULL,
    cycle_number    INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snapshots_time ON portfolio_snapshots(created_at DESC);

-- Engine state (singleton)
CREATE TABLE IF NOT EXISTS engine_state (
    id              INTEGER PRIMARY KEY DEFAULT 1,
    status          TEXT NOT NULL,
    mode            TEXT NOT NULL,
    open_positions  JSONB NOT NULL DEFAULT '{}',
    daily_pnl_usd   DECIMAL(12, 4) NOT NULL DEFAULT 0,
    daily_start_bal DECIMAL(12, 4) NOT NULL,
    peak_balance    DECIMAL(12, 4) NOT NULL,
    current_balance DECIMAL(12, 4) NOT NULL,
    last_scan_time  TIMESTAMPTZ,
    cycle_count     INTEGER NOT NULL DEFAULT 0,
    config_overrides JSONB DEFAULT '{}',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (id = 1)
);

-- Error log
CREATE TABLE IF NOT EXISTS error_log (
    id              BIGSERIAL PRIMARY KEY,
    level           TEXT NOT NULL,
    component       TEXT NOT NULL,
    message         TEXT NOT NULL,
    details         JSONB,
    stack_trace     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_errors_time ON error_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_errors_level ON error_log(level);

-- Enable Row Level Security (disable for service role key usage)
-- If you want RLS, configure policies for your use case
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE candle_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE engine_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE error_log ENABLE ROW LEVEL SECURITY;

-- Allow service role full access
CREATE POLICY "service_role_all" ON trades FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all" ON signals FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all" ON candle_cache FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all" ON portfolio_snapshots FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all" ON engine_state FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all" ON error_log FOR ALL USING (true) WITH CHECK (true);
