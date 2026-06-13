-- MM Agent Committee decision details.
-- The existing mm_agent_decisions row remains the decision ledger; this JSONB
-- column carries specialist/head-trader evidence without creating a second
-- source of truth.

ALTER TABLE mm_agent_decisions
ADD COLUMN IF NOT EXISTS committee jsonb;
