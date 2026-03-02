-- Add leverage, margin, and liquidation tracking to trades table
-- Required for margin & futures trading support

ALTER TABLE trades ADD COLUMN IF NOT EXISTS leverage       INTEGER NOT NULL DEFAULT 1;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS margin_used    DECIMAL(12, 4);
ALTER TABLE trades ADD COLUMN IF NOT EXISTS liquidation_price DECIMAL(30, 12);

-- Add total_pnl tracking to engine_state
ALTER TABLE engine_state ADD COLUMN IF NOT EXISTS total_pnl_usd DECIMAL(12, 4) NOT NULL DEFAULT 0;

-- Update existing open trades with the configured leverage (3x futures)
-- so that reconciliation uses correct margin calculations
-- Run this ONLY if you know all existing trades were opened at 3x leverage:
-- UPDATE trades SET leverage = 3 WHERE status = 'open' AND leverage = 1;
