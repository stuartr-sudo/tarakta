-- Prevent duplicate TP-tier audit rows for the same trade.
--
-- The engine treats TP tiers as cumulative one-shot events. If two workers or
-- a restart race logs the same tier twice, dashboards and inverse mirroring can
-- double-count the exit. Keep the earliest row and enforce one row per tier.

WITH ranked AS (
    SELECT
        id,
        row_number() OVER (
            PARTITION BY trade_id, tier
            ORDER BY created_at ASC, id ASC
        ) AS rn
    FROM partial_exits
)
DELETE FROM partial_exits p
USING ranked r
WHERE p.id = r.id
  AND r.rn > 1;

CREATE UNIQUE INDEX IF NOT EXISTS idx_partial_exits_trade_tier_unique
ON partial_exits(trade_id, tier);
