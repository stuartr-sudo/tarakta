-- Add Agent 3 (Position Manager) reasoning columns to trades table
-- Stores the last Agent 3 decision and reasoning for dashboard display

ALTER TABLE trades
    ADD COLUMN IF NOT EXISTS last_agent3_action TEXT,
    ADD COLUMN IF NOT EXISTS last_agent3_reasoning TEXT,
    ADD COLUMN IF NOT EXISTS agent3_confidence REAL,
    ADD COLUMN IF NOT EXISTS last_agent3_sl TEXT;
