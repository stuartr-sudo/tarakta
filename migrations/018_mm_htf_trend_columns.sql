-- Persist Higher-Timeframe (HTF) trend state on each MM trade at entry.
-- Without these, post-mortems cannot distinguish trend-aligned losses from
-- counter-trend losses (which is how the BNB 2026-04-17 short slipped through:
-- the 4H stack was bullish, the bot shorted, but nothing on the trade row
-- recorded that fact — see src/strategy/mm_engine.py:1546 "kept for future use"
-- and the accompanying audit).
--
-- These columns are written at insert-time and never updated afterward.
--
-- htf_trend_4h / htf_trend_1d: "bullish" | "bearish" | "sideways"
-- counter_trend: true when trade_direction opposes a non-sideways 4H trend.

ALTER TABLE trades ADD COLUMN IF NOT EXISTS htf_trend_4h text;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS htf_trend_1d text;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS counter_trend boolean DEFAULT false;
