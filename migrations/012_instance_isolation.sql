-- Instance Isolation: allow multiple bot instances to share one Supabase DB.
-- Each instance is identified by instance_id (e.g. 'main', 'simple').
-- Trades, signals, snapshots, and engine_state are partitioned by instance_id.
-- Shared tables (candle_cache, trade_lessons, knowledge_*) remain global.

-- 1. engine_state: remove singleton CHECK, add instance_id as new PK
ALTER TABLE engine_state DROP CONSTRAINT IF EXISTS engine_state_pkey CASCADE;
ALTER TABLE engine_state DROP CONSTRAINT IF EXISTS engine_state_id_check;
-- Drop any CHECK constraint named with the pattern (Supabase may name it differently)
DO $$
BEGIN
    EXECUTE (
        SELECT string_agg('ALTER TABLE engine_state DROP CONSTRAINT ' || conname || ';', ' ')
        FROM pg_constraint
        WHERE conrelid = 'engine_state'::regclass
          AND contype = 'c'
    );
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

ALTER TABLE engine_state ADD COLUMN IF NOT EXISTS instance_id TEXT NOT NULL DEFAULT 'main';
-- Remove the old id=1 default so new rows don't collide
ALTER TABLE engine_state ALTER COLUMN id DROP DEFAULT;
-- New PK: one row per instance
ALTER TABLE engine_state ADD PRIMARY KEY (instance_id);

-- 2. trades: add instance_id column
ALTER TABLE trades ADD COLUMN IF NOT EXISTS instance_id TEXT NOT NULL DEFAULT 'main';
CREATE INDEX IF NOT EXISTS idx_trades_instance ON trades (instance_id, status, entry_time DESC);

-- 3. signals: add instance_id column
ALTER TABLE signals ADD COLUMN IF NOT EXISTS instance_id TEXT NOT NULL DEFAULT 'main';
CREATE INDEX IF NOT EXISTS idx_signals_instance ON signals (instance_id, created_at DESC);

-- 4. portfolio_snapshots: add instance_id column
ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS instance_id TEXT NOT NULL DEFAULT 'main';
CREATE INDEX IF NOT EXISTS idx_snapshots_instance ON portfolio_snapshots (instance_id, created_at DESC);
