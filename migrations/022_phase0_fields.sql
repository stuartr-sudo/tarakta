-- Phase 0 engine persistence fields for MM Agent Committee rollout.
-- formation_timeframe must survive restarts so the scratch-window rule
-- continues to use the same 1H/15m wall-clock or 4H-bar logic after restore.

ALTER TABLE trades
ADD COLUMN IF NOT EXISTS formation_timeframe text;

ALTER TABLE signals
ADD COLUMN IF NOT EXISTS formation_timeframe text;
