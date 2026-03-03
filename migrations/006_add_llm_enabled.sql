-- Add llm_enabled column to engine_state table
-- Required for the LLM toggle feature on the settings page

ALTER TABLE engine_state
    ADD COLUMN IF NOT EXISTS llm_enabled BOOLEAN NOT NULL DEFAULT FALSE;
