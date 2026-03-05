-- Add flipped_trader JSONB column to engine_state table
-- Stores the flipped shadow trader's balance, positions, and P&L state

ALTER TABLE engine_state
    ADD COLUMN IF NOT EXISTS flipped_trader JSONB DEFAULT '{}';
