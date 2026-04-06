# Review & Improvement Widget — Implementation Guide for Tarakta

**Date:** 2026-04-07
**Target App:** Tarakta (FastAPI + Jinja2 + Supabase)
**Supabase Project:** `uounrdaescblpgwkgbdq` (ap-southeast-1)
**Deploy:** `fly deploy --depot=false --remote-only --app tarakta-expanded`

---

## Overview

A floating review/improvement widget that appears on every page of the Tarakta dashboard. It lets you quickly log bugs, questions, improvement ideas, and other requests — all targeted at specific subsystems and their components. A Claude Code scheduled task picks up requests hourly, investigates the codebase, fixes bugs or writes answers, deploys fixes, and posts findings back.

### Three Components

1. **Floating Widget** (bottom-left corner) — quick submit form
2. **Review Dashboard** (slide-out panel) — manage, filter, comment on requests
3. **CRON Processor** — Claude Code scheduled task that autonomously processes requests

---

## 1. Database Schema

### Table: `review_requests`

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | uuid | PK, DEFAULT gen_random_uuid() | |
| `user_id` | text | NOT NULL | Dashboard username (e.g. 'admin') — Tarakta uses basic auth, not Supabase auth |
| `tool` | text | NOT NULL | Subsystem name from hardcoded list |
| `endpoint` | text | | Specific component/module (optional) |
| `type` | text | NOT NULL, CHECK | One of 8 types (see below) |
| `title` | text | NOT NULL | Short summary |
| `description` | text | | Full details, logs, error messages |
| `screenshot_url` | text | | Supabase storage path |
| `status` | text | NOT NULL, DEFAULT 'pending', CHECK | pending, in_progress, resolved, needs_info, closed |
| `priority` | text | NOT NULL, DEFAULT 'medium', CHECK | low, medium, high |
| `created_at` | timestamptz | DEFAULT now() | |
| `updated_at` | timestamptz | DEFAULT now() | |
| `resolved_at` | timestamptz | | Set when status -> resolved/closed, cleared on re-open |

### Table: `review_comments`

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | uuid | PK, DEFAULT gen_random_uuid() | |
| `request_id` | uuid | FK -> review_requests(id) ON DELETE CASCADE | |
| `author` | text | NOT NULL, CHECK ('user','claude') | |
| `content` | text | NOT NULL | Comment body — findings, answers, or clarifications |
| `commit_hash` | text | | If Claude made a code fix |
| `created_at` | timestamptz | DEFAULT now() | |

### Migration SQL

```sql
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
```

Save as: `migrations/016_review_requests.sql`

---

## 2. Request Types (8)

| Type | Label | When to Use |
|------|-------|-------------|
| `bug` | Bug | Something is broken or producing wrong results |
| `question` | Question | How/why does X work? |
| `improvement` | Improvement | New feature or enhancement (replaces "feature" — more relevant for a trading bot) |
| `console_error` | Console/Log Error | Paste a log error or stack trace |
| `change_request` | Change Request | Modify existing behavior |
| `strategy_review` | Strategy Review | Audit a trading strategy, signal logic, or agent prompt for correctness (replaces "learn_screenshot") |
| `prompt_review` | Prompt Review | Audit agent prompt construction for coherence |
| `claude_md_update` | CLAUDE.md Update | Update documentation in CLAUDE.md |

---

## 3. Tool -> Endpoint Map

Every subsystem in Tarakta with its specific components. The endpoint dropdown filters based on selected tool.

### Trading Engine
- **Scanner**: Signal detection, SMC confluence scoring, volume analysis, candle cache
- **Agent 1 (Entry Analyst)**: Entry analysis, tool-use chain, SL/TP suggestions, context building
- **Agent 2 (Refiner)**: ENTER/WAIT decisions, order book analysis, footprint data, advisor context injection
- **Agent 3 (Position Manager)**: SL tightening, TP extension, position monitoring
- **Risk Manager**: Drawdown limits, circuit breaker, max concurrent trades, position sizing
- **Trade Executor**: Order placement, Binance API, leverage management, error handling
- **Lesson Generator**: Post-trade analysis, trade_lessons table, feedback loop injection

### Dashboard
- **Overview Page**: Portfolio chart, PnL display, balance tracking
- **Trades Page**: Trade history, filtering, export
- **Signals Page**: Signal list, scoring display
- **Settings Page**: Engine config, risk parameters, scanning parameters
- **API Routes**: REST endpoints, authentication, session management
- **Charts/Visualization**: TradingView widgets, performance charts

### Data & Infrastructure
- **Repository Layer**: Supabase queries, data access patterns
- **Candle Cache**: OHLCV data fetching, caching, staleness
- **Exchange Connection**: CCXT async, Binance API, rate limiting, reconnection
- **RAG System**: Knowledge chunks, hybrid search, trade history retrieval

### Advisory & Intelligence
- **Daily Advisor**: Claude Agent SDK, insight generation, context injection
- **MM Method Engine**: Parallel trading engine, state management
- **Portfolio Snapshots**: Performance tracking, equity curve

### Other
- **Auth**: Login, session management, password hashing
- **Deployment**: Fly.io, Docker, health checks, OOM management
- **Logging**: Structured logging, log levels, error tracking
- **General**: UI/CSS, navigation, configuration, environment variables

---

## 4. API Endpoints

All endpoints go in `src/dashboard/api.py` (or a new `src/dashboard/reviews_api.py`).

### Routes

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | `/api/reviews` | Required | Create a new request |
| GET | `/api/reviews` | Required | List all requests (query: `?status=pending`) |
| GET | `/api/reviews/{id}` | Required | Get single request + comments |
| PATCH | `/api/reviews/{id}` | Required | Update status, priority |
| POST | `/api/reviews/{id}/comments` | Required | Add a comment |
| POST | `/api/reviews/upload` | Required | Upload screenshot to Supabase storage |

### Endpoint Details

**POST /api/reviews** — Create request
```python
@router.post("/api/reviews")
async def create_review(request: Request):
    body = await request.json()
    # Validate: tool, type, title required
    # Insert into review_requests with user_id from session
    # Return: { "request": { ...row } }
```

**GET /api/reviews** — List requests
```python
@router.get("/api/reviews")
async def list_reviews(request: Request, status: str = None):
    # Query review_requests filtered by user_id
    # Optional status filter
    # Order by created_at ASC
    # Return: { "requests": [...] }
```

**GET /api/reviews/{id}** — Get request + comments
```python
@router.get("/api/reviews/{review_id}")
async def get_review(request: Request, review_id: str):
    # Fetch request + all comments ordered by created_at
    # Return: { "request": { ...row, "comments": [...] } }
```

**PATCH /api/reviews/{id}** — Update status/priority
```python
@router.patch("/api/reviews/{review_id}")
async def update_review(request: Request, review_id: str):
    body = await request.json()
    # Update status and/or priority
    # resolved_at rule: set on resolved/closed, clear on pending
    # Return: { "request": { ...row } }
```

**POST /api/reviews/{id}/comments** — Add comment
```python
@router.post("/api/reviews/{review_id}/comments")
async def add_comment(request: Request, review_id: str):
    body = await request.json()
    # Insert into review_comments
    # Return: { "comment": { ...row } }
```

**POST /api/reviews/upload** — Upload screenshot
```python
@router.post("/api/reviews/upload")
async def upload_screenshot(request: Request):
    form = await request.form()
    file = form.get("image")
    # Validate image/*, max 5MB
    # Upload to Supabase storage: media/reviews/{user_id}/{uuid}.{ext}
    # Return: { "url": public_url }
```

---

## 5. Floating Widget UI

### Position & Styling

The widget is a **fixed-position circular button** in the **bottom-left corner** of every dashboard page. It mirrors the teal/magenta design system already in `style.css`.

**Button specs:**
- Position: `fixed`, `bottom: 24px`, `left: 24px`, `z-index: 9999`
- Size: 56px circle
- Background: `var(--bg-deep)` (#0a0f1e)
- Border: 2px solid with teal glow — `border: 2px solid rgba(0, 229, 212, 0.6)`
- Shadow: `box-shadow: 0 0 15px rgba(0, 229, 212, 0.4)`
- Icon: Pencil/edit SVG, white, 24px
- Badge: Red circle (absolute top-right) showing pending + needs_info count, hidden if 0
- Hover: Scale 1.05 + brighter glow

### Submit Form (popup)

Opens above the button when clicked. ~350px wide card.

**Styling:**
- Background: `var(--bg-card)` (#141b2d)
- Border: `1px solid var(--border)` (rgba(255,255,255,0.06))
- Border-radius: 12px
- Box-shadow: large dark shadow

**Fields:**
1. **Tool** — `<select>` required, flat alphabetical list of all tools
2. **Endpoint** — `<select>` optional, filters by selected tool, hidden if no tool selected
3. **Type** — `<select>` required, 8 types
4. **Priority** — 3 pill buttons (Low / Medium / High), Medium default
   - Active: teal background (`var(--teal)`)
   - Inactive: dark surface (`var(--bg-surface)`)
5. **Title** — `<input>` required, placeholder "Brief summary..."
6. **Description** — `<textarea>` rows=3, placeholder "Details, logs, steps to reproduce..."
7. **Screenshot** — File input button + paste handler (clipboard API). Immediate upload on select/paste. Thumbnail preview.
8. **Submit** button — teal background, submits to `POST /api/reviews`

**"View All"** link in the header opens the dashboard panel.

### Implementation

Since Tarakta uses Jinja2 templates, the widget is:
1. A Jinja2 **partial template** (`templates/partials/review_widget.html`) included in `base.html`
2. **Vanilla JS** for open/close, form submission, paste handling
3. **CSS** added to `style.css` using existing design tokens

Include in `base.html`:
```html
{% include "partials/review_widget.html" %}
```

---

## 6. Dashboard Panel (Slide-Over)

### Layout

A **left-side slide-over panel** (~600px wide) that opens when "View All" is clicked.

**Styling:**
- Position: fixed, inset-y 0, left 0
- Background: `var(--bg-deep)`
- Border-right: `1px solid var(--border)`
- Transition: `transform 0.3s ease`
- Closed: `transform: translateX(-100%)`
- Open: `transform: translateX(0)`
- Overlay: semi-transparent black behind panel

### Header
- Title: "Review Requests"
- Close button (X)
- Filter chips: All / Pending / In Progress / Needs Info / Resolved / Closed
- Each chip shows count

### Request List

Cards sorted by priority (high first), then oldest first. Each card shows:
- Status badge (color-coded):
  - Pending: amber/yellow
  - In Progress: teal
  - Needs Info: orange
  - Resolved: green
  - Closed: gray
- Type badge
- Title
- Tool + Endpoint tags
- Relative time ("2h ago")
- Click to expand

### Expanded Request View
- Full description (monospace, preserves formatting for logs)
- Screenshot thumbnail (if any)
- **Comment thread**: User comments left-aligned, Claude comments right-aligned with teal accent
- **Add comment**: textarea + submit button
- **Status buttons** (conditional):
  - Pending → Close
  - In Progress → Close
  - Needs Info → Re-open, Close
  - Resolved → Re-open, Close
  - Closed → Re-open

### Implementation

The dashboard is a second partial template + JS:
1. `templates/partials/review_panel.html` — the slide-over HTML
2. JS in the same file or a `review-widget.js` script
3. All data fetched via `fetch('/api/reviews')` on open

---

## 7. CSS Additions

Add to `src/dashboard/static/style.css`:

```css
/* ── Review Widget ── */

.review-widget-btn {
  position: fixed;
  bottom: 24px;
  left: 24px;
  z-index: 9999;
  width: 56px;
  height: 56px;
  border-radius: 50%;
  background: var(--bg-deep);
  border: 2px solid rgba(0, 229, 212, 0.6);
  box-shadow: 0 0 15px rgba(0, 229, 212, 0.4);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform 0.2s, box-shadow 0.2s;
}
.review-widget-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 0 25px rgba(0, 229, 212, 0.6);
}
.review-widget-btn svg {
  width: 24px;
  height: 24px;
  color: white;
}
.review-widget-badge {
  position: absolute;
  top: -4px;
  right: -4px;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #ef4444;
  color: white;
  font-size: 11px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
}

.review-form-card {
  position: absolute;
  bottom: 64px;
  left: 0;
  width: 350px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: 0 20px 40px rgba(0,0,0,0.4);
  padding: 16px;
  z-index: 9999;
}

.review-panel-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.5);
  z-index: 9998;
  opacity: 0;
  transition: opacity 0.3s;
  pointer-events: none;
}
.review-panel-overlay.open {
  opacity: 1;
  pointer-events: auto;
}

.review-panel {
  position: fixed;
  top: 0;
  left: 0;
  bottom: 0;
  width: 600px;
  max-width: 95vw;
  background: var(--bg-deep);
  border-right: 1px solid var(--border);
  box-shadow: 10px 0 30px rgba(0,0,0,0.3);
  z-index: 9999;
  transform: translateX(-100%);
  transition: transform 0.3s ease;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}
.review-panel.open {
  transform: translateX(0);
}

.review-status-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 9999px;
  font-size: 11px;
  font-weight: 600;
  border: 1px solid;
}
.review-status-pending { background: rgba(245,158,11,0.15); color: #fbbf24; border-color: rgba(245,158,11,0.3); }
.review-status-in_progress { background: rgba(0,229,212,0.15); color: var(--teal); border-color: rgba(0,229,212,0.3); }
.review-status-needs_info { background: rgba(249,115,22,0.15); color: #fb923c; border-color: rgba(249,115,22,0.3); }
.review-status-resolved { background: rgba(34,197,94,0.15); color: #4ade80; border-color: rgba(34,197,94,0.3); }
.review-status-closed { background: rgba(148,163,184,0.15); color: #94a3b8; border-color: rgba(148,163,184,0.3); }

.review-comment-user {
  background: rgba(255,255,255,0.05);
  border-radius: 8px;
  padding: 8px 12px;
  margin-right: 40px;
}
.review-comment-claude {
  background: rgba(0,229,212,0.08);
  border-radius: 8px;
  padding: 8px 12px;
  margin-left: 40px;
  border-left: 2px solid var(--teal);
}
```

---

## 8. File Structure

```
NEW FILES:
  migrations/016_review_requests.sql                    — DB migration
  src/dashboard/reviews_api.py                          — FastAPI routes for review CRUD
  src/dashboard/templates/partials/review_widget.html   — floating button + submit form
  src/dashboard/templates/partials/review_panel.html    — slide-over dashboard panel
  src/dashboard/static/review-widget.js                 — widget + panel JS logic
  src/dashboard/review_tool_map.py                      — tool -> endpoint mapping data

MODIFIED FILES:
  src/dashboard/app.py                                  — register reviews_api router
  src/dashboard/templates/base.html                     — include widget partial
  src/dashboard/static/style.css                        — add widget/panel CSS
```

---

## 9. CRON Scheduled Task

### Configuration

- **Task ID**: `tarakta-review-processor`
- **Model**: `claude-opus-4-6`
- **Schedule**: `0 7-23 * * *` (hourly, 7am-11pm NZT local time)
- **Working directory**: `/Users/stuarta/tarakta`

### Task Prompt

```
Process the oldest pending review request from the Tarakta app.

Use the Supabase MCP execute_sql tool with project_id 'uounrdaescblpgwkgbdq' for all database operations.

App URL: https://tarakta-expanded.fly.dev
Login: [DASHBOARD_USERNAME] / [DASHBOARD_PASSWORD]

For bug fixes, after deploying you can verify via Firecrawl browser session:
- Create session with profile { name: "tarakta-admin", saveChanges: true }
- Navigate to the relevant dashboard page, log in if needed
- Test that the fix works
- Delete the browser session when done

1. Atomically claim the oldest pending request (prevents double-processing):
   execute_sql: UPDATE review_requests SET status = 'in_progress' WHERE id = (SELECT id FROM review_requests WHERE status = 'pending' ORDER BY CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 WHEN 'low' THEN 3 END, created_at ASC LIMIT 1) RETURNING *
   - If no rows returned, respond "No pending requests." and stop.

2. Fetch existing comments for context:
   execute_sql: SELECT * FROM review_comments WHERE request_id = '{id}' ORDER BY created_at ASC

3. Read the request fields: tool, endpoint, type, title, description. Process based on type using the Tarakta codebase at /Users/stuarta/tarakta/:
   - bug / console_error: Read the relevant source code. Diagnose, fix if straightforward, commit.
   - question: Research the codebase and write a clear answer.
   - improvement / change_request: Analyze feasibility, write findings + approach.
   - strategy_review: Audit the trading strategy or signal logic for the specified tool. Check agent prompts, scoring weights, confluence logic. Report findings.
   - prompt_review: Read the agent prompt code. Check for coherence vs concatenation. Fix if needed.
   - claude_md_update: Research topic, update CLAUDE.md, commit.

4. If you made code changes and committed, deploy:
   - Run: cd /Users/stuarta/tarakta && git push origin main && fly deploy --depot=false --remote-only --app tarakta-expanded
   - If deploy fails, note the error in your comment.
   - Do NOT deploy if you only wrote an answer/analysis.

5. Write findings as a comment:
   execute_sql: INSERT INTO review_comments (request_id, author, content, commit_hash) VALUES ('{id}', 'claude', '{findings}', '{commit_hash_or_null}')

6. Update status:
   - Success: UPDATE review_requests SET status = 'resolved', resolved_at = now() WHERE id = '{id}'
   - Too vague: UPDATE review_requests SET status = 'needs_info' WHERE id = '{id}'

7. Only process ONE request per run. Never batch.
```

### Required Permissions (.claude/settings.local.json)

```json
{
  "permissions": {
    "allow": [
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(git push:*)",
      "Bash(git status:*)",
      "Bash(git diff:*)",
      "Bash(git log:*)",
      "Bash(npm run build:*)",
      "Bash(curl:*)",
      "Bash(fly deploy:*)",
      "Bash(fly status:*)",
      "mcp__firecrawl-mcp__firecrawl_browser_create",
      "mcp__firecrawl-mcp__firecrawl_browser_execute",
      "mcp__firecrawl-mcp__firecrawl_browser_delete",
      "mcp__firecrawl-mcp__firecrawl_browser_list"
    ]
  }
}
```

---

## 10. Key Differences from Stitch

| Aspect | Stitch | Tarakta |
|--------|--------|---------|
| Framework | React (Vite) | FastAPI + Jinja2 |
| Auth | Supabase JWT | Basic Auth + session cookies |
| user_id type | uuid (FK auth.users) | text (dashboard username) |
| RLS | Enabled (auth.uid()) | Disabled (server-side auth) |
| Widget impl | React component (JSX) | Jinja2 partial + vanilla JS |
| Panel impl | Radix Dialog | CSS transform slide-over |
| Deploy | `fly deploy` | `fly deploy --depot=false --remote-only --app tarakta-expanded` |
| "Feature" type | `feature` | `improvement` (more trading-relevant) |
| "Learn Screenshot" type | `learn_screenshot` | `strategy_review` (audit trading logic) |
| Design tokens | Tailwind + shadcn | CSS variables (--teal, --magenta, --bg-deep) |

---

## 11. Resolved_at Rule

This rule must be enforced in the PATCH endpoint and the CRON task:

- Status -> `resolved` or `closed`: set `resolved_at = now()` if null
- Status -> `pending` (re-open): set `resolved_at = null`
- Other status changes: don't touch resolved_at

---

## 12. Testing Checklist

After implementation, verify:

- [ ] Widget button appears bottom-left on all dashboard pages
- [ ] Badge shows correct pending + needs_info count
- [ ] Click opens submit form
- [ ] Tool dropdown populates, selecting a tool shows endpoint dropdown
- [ ] Type dropdown shows all 8 types
- [ ] Priority pills toggle correctly (default: medium)
- [ ] Submit creates record in DB
- [ ] "View All" opens the slide-over panel
- [ ] Filter chips work and show counts
- [ ] Expanding a card shows full details + comments
- [ ] Adding a comment works
- [ ] Status changes work (resolve, close, re-open)
- [ ] Re-opening clears resolved_at
- [ ] Screenshot upload works (file picker + clipboard paste)
- [ ] CRON task picks up pending request, processes, and writes comment
- [ ] CRON task deploys after code fixes
- [ ] Double-trigger doesn't process same request twice (atomic claim)
