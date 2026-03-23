-- Track all OpenAI API calls made by Tarakta agents
CREATE TABLE IF NOT EXISTS api_usage (
    id BIGSERIAL PRIMARY KEY,
    instance_id TEXT NOT NULL DEFAULT 'main',
    caller TEXT NOT NULL,          -- 'agent1', 'agent2', 'agent3', 'lessons', 'rag'
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd NUMERIC(10, 6) NOT NULL DEFAULT 0,
    trade_id UUID,                 -- nullable, links to trades table
    signal_id UUID,                -- nullable, links to signals table
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_api_usage_instance_created ON api_usage (instance_id, created_at DESC);
CREATE INDEX idx_api_usage_caller ON api_usage (caller);
CREATE INDEX idx_api_usage_model ON api_usage (model);
