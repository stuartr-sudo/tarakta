-- Add MM Method columns to trades table
-- These columns support the standalone Market Makers Method engine

ALTER TABLE trades ADD COLUMN IF NOT EXISTS strategy text;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS entry_reason text;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_formation text;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_cycle_phase text;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_confluence_grade text;

-- Index for filtering MM trades in dashboard
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy);
