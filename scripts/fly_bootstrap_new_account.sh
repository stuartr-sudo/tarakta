#!/usr/bin/env bash
# Bootstrap the MM bot on a FRESH Fly.io account.
#
# User prereqs (once, ~3 minutes):
#   1) Create the account in a browser: https://fly.io/app/sign-up
#      (Claude cannot create accounts — this step is yours.)
#   2) In a terminal:  fly auth login
#   3) Optional but recommended: put your Anthropic API key in `env.local`
#      at the repo root (gitignored; the local bot does NOT read this file):
#          ANTHROPIC_API_KEY=sk-ant-...
#      This powers the committee on Fly via the SDK. Do NOT put the key in
#      `.env` — that would silently switch the LOCAL bot from the free
#      subscription CLI to metered API billing.
#
# Then:  ./scripts/fly_bootstrap_new_account.sh [app-name] [region] [instance-id]
#
# Defaults: app tarakta-mm2 (the old name is taken by the old account),
# region sin, INSTANCE_ID tarakta-mm.
#
# INSTANCE_ID=tarakta-mm makes the new deployment ADOPT the old book:
# engine_state has scanning_active:false (so it comes up PAUSED — no new
# entries) and it will restore + manage the 3 stale open positions.
# ONLY safe if the old Fly app is stopped first — two live bots on one
# instance_id will fight over positions (CLAUDE.md gotcha). Stop the old
# app's scanning via its own dashboard Stop button at
# https://tarakta-mm.fly.dev/mm (app-level login, no Fly account needed).
# If the old app cannot be stopped at all, pass a fresh instance id
# (e.g. tarakta-mm-v2) and accept that the old book stays orphaned.
set -euo pipefail
cd "$(dirname "$0")/.."

APP="${1:-tarakta-mm2}"
REGION="${2:-sin}"
INSTANCE="${3:-tarakta-mm}"

fly auth whoami >/dev/null 2>&1 || { echo "ERROR: run 'fly auth login' first"; exit 1; }

echo "App: $APP   Region: $REGION   INSTANCE_ID: $INSTANCE"
if [ "$INSTANCE" = "tarakta-mm" ]; then
  echo "REMINDER: old app must be stopped (https://tarakta-mm.fly.dev/mm -> Stop)"
  read -r -p "Old app stopped or unreachable? Type yes to continue: " ok
  [ "$ok" = "yes" ] || exit 1
fi

fly apps create "$APP" 2>/dev/null || echo "(app exists — continuing)"

# Secrets: values are piped straight from local files into Fly — never printed.
{
  grep -hE '^(BINANCE_API_KEY|BINANCE_API_SECRET|SUPABASE_URL|SUPABASE_KEY|DASHBOARD_USERNAME|DASHBOARD_PASSWORD_HASH|VIEWER_USERNAME|VIEWER_PASSWORD_HASH|SESSION_SECRET)=' .env
  if [ -f env.local ]; then grep -hE '^ANTHROPIC_API_KEY=' env.local || true; fi
} | fly secrets import --app "$APP"

fly secrets set --app "$APP" \
  TRADING_MODE=paper \
  "INSTANCE_ID=$INSTANCE" \
  MM_INITIAL_BALANCE=100000 \
  MM_COMMITTEE_ENABLED=true \
  MM_COMMITTEE_MODE=shadow \
  MM_SANITY_AGENT_ENABLED=false

fly deploy --app "$APP" --config fly.toml --regions "$REGION" --depot=false --remote-only

echo
echo "Deployed. Dashboard: https://${APP}.fly.dev"
echo "Engine starts PAUSED (scanning_active:false in engine_state). Committee is"
echo "in SHADOW mode via the Anthropic SDK if the API key secret was provided."
echo "Resume scanning from the dashboard only when you intend it to trade."
