-- Progressive take-profit: partial exits tracking table
CREATE TABLE IF NOT EXISTS partial_exits (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    trade_id uuid REFERENCES trades(id) NOT NULL,
    tier integer NOT NULL,
    exit_price float NOT NULL,
    exit_quantity float NOT NULL,
    exit_order_id text,
    exit_reason text NOT NULL,
    pnl_usd float NOT NULL,
    pnl_percent float,
    fees_usd float DEFAULT 0,
    remaining_quantity float NOT NULL,
    new_stop_loss float,
    created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_partial_exits_trade_id ON partial_exits(trade_id);

-- Extend trades table for progressive TP tracking
ALTER TABLE trades ADD COLUMN IF NOT EXISTS tp_tiers jsonb;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS current_tier integer DEFAULT 0;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS remaining_quantity float;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS original_quantity float;
