-- MM Sanity Agent (Agent 4) DB contract.
--
-- Two changes:
--   1. Columns on `trades` for approved setups that entered (so every
--      live trade carries the agent's verdict + reasoning).
--   2. Standalone `mm_agent_decisions` table that logs EVERY call the
--      agent makes, including VETOs that never became trades and ERRORs
--      where the API failed. This is the primary observability surface.
--
-- See docs/MM_SANITY_AGENT_DESIGN.md §5.

-- ---------------------------------------------------------------------
-- (1) Columns on `trades`
-- ---------------------------------------------------------------------
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_agent_decision text;      -- 'APPROVE' | NULL (VETOs never reach trades)
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_agent_reason text;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_agent_confidence numeric; -- 0.0-1.0
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_agent_model text;         -- e.g. 'claude-opus-4-7'
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_agent_concerns jsonb;     -- list of concern tags

-- ---------------------------------------------------------------------
-- (2) mm_agent_decisions — log every call
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mm_agent_decisions (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol             text NOT NULL,
    created_at         timestamptz NOT NULL DEFAULT now(),
    instance_id        text,                -- matches trades.instance_id
    cycle_count        integer,
    formation_type     text,                -- 'M' | 'W'
    formation_variant  text,
    confluence_grade   text,                -- 'A+'|'A'|'B+'|'B'|'C'|'F'
    confluence_pct     numeric,
    direction          text,                -- 'long' | 'short'
    decision           text NOT NULL,       -- 'APPROVE' | 'VETO' | 'ERROR'
    reason             text,
    confidence         numeric,
    htf_trend_4h       text,                -- 'bullish' | 'bearish' | 'sideways'
    htf_trend_1d       text,
    counter_trend      boolean,
    concerns           jsonb,               -- list of concern tags
    input_context      jsonb,               -- the full payload sent to the model
    raw_response       text,                -- verbatim model output, for debugging
    model              text,                -- 'claude-opus-4-7' etc
    prompt_version     text,                -- 'prompt_v=1 rubric_v=1' etc
    latency_ms         integer,
    cost_usd           numeric,
    trade_id           uuid REFERENCES trades(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_mm_agent_decisions_symbol_time
    ON mm_agent_decisions (symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mm_agent_decisions_decision
    ON mm_agent_decisions (decision);
CREATE INDEX IF NOT EXISTS idx_mm_agent_decisions_instance
    ON mm_agent_decisions (instance_id, created_at DESC);
