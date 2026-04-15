-- MM Engine lifecycle fields that must survive engine restarts.
-- Without these persisted, a restart loses SL-tightening progress (breakeven /
-- under-50EMA), SVC invalidation zone, Refund Zone trigger, 200-EMA partial
-- deduplication, and the aggressive/conservative entry type. Silent data loss.
--
-- Course references: lessons 20, 23 (SVC), 47/48 (SL progression),
-- 49 (Refund Zone), C6 (200 EMA hammer partial).

ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_entry_type text;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_peak2_wick_price numeric;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_svc_high numeric;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_svc_low numeric;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_sl_moved_to_breakeven boolean DEFAULT false;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_sl_moved_under_50ema boolean DEFAULT false;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_took_200ema_partial boolean DEFAULT false;
