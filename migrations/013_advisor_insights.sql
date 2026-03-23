-- Stores structured findings from the trade advisor agent.
-- One row per instance per day; upserted on each daily run.

CREATE TABLE IF NOT EXISTS advisor_insights (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    instance_id TEXT NOT NULL DEFAULT 'main',
    run_date DATE NOT NULL,
    signals_analyzed INT NOT NULL DEFAULT 0,
    simulated_winners INT NOT NULL DEFAULT 0,
    simulated_losers INT NOT NULL DEFAULT 0,
    win_rate_pct DECIMAL(5,2),
    top_missed TEXT,          -- JSON array of best missed trades
    patterns TEXT,            -- JSON: common traits of winners
    recommendations TEXT,     -- JSON array of specific recommendations
    full_analysis TEXT,       -- The raw text analysis from the advisor
    cost_usd DECIMAL(8,4) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(instance_id, run_date)
);

CREATE INDEX IF NOT EXISTS idx_advisor_insights_instance_date
    ON advisor_insights(instance_id, run_date DESC);
