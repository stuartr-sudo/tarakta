-- Signal reversal tracking table
CREATE TABLE IF NOT EXISTS reversals (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    old_trade_id uuid REFERENCES trades(id),
    new_trade_id uuid REFERENCES trades(id),
    symbol text NOT NULL,
    old_direction text NOT NULL,
    new_direction text NOT NULL,
    close_pnl float NOT NULL,
    signal_score float NOT NULL,
    created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_reversals_symbol ON reversals(symbol);
