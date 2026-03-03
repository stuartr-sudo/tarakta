-- Add test_group column to trades table for LLM split test tracking
-- Values: "control" (default system) or "llm" (LLM-evaluated)

ALTER TABLE trades
    ADD COLUMN IF NOT EXISTS test_group TEXT DEFAULT 'control';
