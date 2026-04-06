-- Review Requests
CREATE TABLE review_requests (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  tool text NOT NULL,
  endpoint text,
  type text NOT NULL CHECK (type IN ('bug','question','improvement','console_error','change_request','strategy_review','prompt_review','claude_md_update')),
  title text NOT NULL,
  description text,
  screenshot_url text,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','in_progress','resolved','needs_info','closed')),
  priority text NOT NULL DEFAULT 'medium' CHECK (priority IN ('low','medium','high')),
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  resolved_at timestamptz
);

CREATE TABLE review_comments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  request_id uuid REFERENCES review_requests(id) ON DELETE CASCADE NOT NULL,
  author text NOT NULL CHECK (author IN ('user','claude')),
  content text NOT NULL,
  commit_hash text,
  created_at timestamptz DEFAULT now()
);

-- No RLS needed — Tarakta uses server-side auth, not Supabase auth
-- Service role key is used for all DB access

-- Auto-update trigger (create function first if it doesn't exist)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_review_requests_updated_at
  BEFORE UPDATE ON review_requests
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Indexes
CREATE INDEX idx_review_requests_status ON review_requests(status);
CREATE INDEX idx_review_requests_user ON review_requests(user_id);
CREATE INDEX idx_review_comments_request ON review_comments(request_id);
