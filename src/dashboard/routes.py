from __future__ import annotations

import os

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from src.config import Settings
from src.dashboard.auth import login_required
from src.data.repository import Repository

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def create_router(config: Settings, repo: Repository) -> APIRouter:
    router = APIRouter()
    templates = Jinja2Templates(directory=TEMPLATE_DIR)

    def _get_repo_for_request(request: Request) -> Repository:
        requested = request.query_params.get("instance")
        if not requested or requested == config.instance_id:
            return repo
        repos: dict = getattr(request.app.state, "repos", {})
        if requested in repos:
            return repos[requested]
        other_repo = Repository(repo.db, instance_id=requested)
        repos[requested] = other_repo
        return other_repo

    async def _base_context(request: Request) -> dict:
        db_offline = False
        try:
            state = await repo.get_engine_state()
        except Exception:
            state = None
            db_offline = True
        return {
            "request": request,
            "state": state,
            "config": config,
            "role": request.session.get("role", "viewer"),
            "db_offline": db_offline,
        }

    # --- Auth ---

    @router.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return templates.TemplateResponse(request, "login.html", context={"request": request, "error": None})

    @router.post("/login")
    async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
        from src.dashboard.auth import verify_password
        if username == config.dashboard_username and verify_password(password, config.dashboard_password_hash):
            request.session["authenticated"] = True
            request.session["role"] = "admin"
            request.session["username"] = username
            return RedirectResponse(url="/", status_code=303)
        if config.viewer_password_hash and username == config.viewer_username and verify_password(password, config.viewer_password_hash):
            request.session["authenticated"] = True
            request.session["role"] = "viewer"
            request.session["username"] = username
            return RedirectResponse(url="/", status_code=303)
        return templates.TemplateResponse(request, "login.html", context={"request": request, "error": "Invalid credentials"})

    @router.get("/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse(url="/login", status_code=303)

    # --- MM Dashboard (home page) ---

    @router.get("/", response_class=HTMLResponse)
    @router.get("/mm", response_class=HTMLResponse)
    @login_required
    async def mm_engine_page(request: Request):
        ctx = await _base_context(request)
        active_repo = _get_repo_for_request(request)

        mm_trades = []
        mm_stats = {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl": 0}
        try:
            all_trades = await active_repo.get_trades(per_page=200)
            mm_trades = [t for t in all_trades if t.get("strategy") == "mm_method"]
            mm_trades.sort(key=lambda t: t.get("created_at", ""), reverse=True)

            closed = [t for t in mm_trades if t.get("status") == "closed" and t.get("exit_reason") != "orphan_cleanup"]
            open_mm = [t for t in mm_trades if t.get("status") == "open"]
            if closed:
                wins = sum(1 for t in closed if (t.get("pnl_usd") or 0) > 0)
                losses = len(closed) - wins
                pnls = [float(t.get("pnl_usd") or 0) for t in closed]
                mm_stats = {
                    "total": len(closed),
                    "wins": wins,
                    "losses": losses,
                    "win_rate": wins / len(closed) if closed else 0,
                    "total_pnl": round(sum(pnls), 2),
                    "avg_pnl": round(sum(pnls) / len(closed), 2),
                }
        except Exception:
            open_mm = []
            ctx["db_offline"] = True

        mm_initial = getattr(config, "mm_initial_balance", 10000.0)
        mm_balance = mm_initial + mm_stats["total_pnl"]
        mm_drawdown = max(0, (mm_initial - mm_balance) / mm_initial) if mm_initial > 0 else 0
        ctx["mm_trades"] = mm_trades[:50]
        ctx["mm_open_trades"] = open_mm
        ctx["mm_stats"] = mm_stats
        ctx["mm_snapshot"] = {
            "balance_usd": mm_balance,
            "total_pnl_usd": mm_stats["total_pnl"],
            "drawdown_pct": mm_drawdown,
        }
        # Resolve current scanning state so the button/badge render correctly
        # on first paint (no "Start" flicker while JS fetches /api/mm/status).
        mm_engine_inst = getattr(request.app.state, "mm_engine", None)
        if mm_engine_inst is not None:
            ctx["mm_scanning_active"] = bool(getattr(mm_engine_inst, "_scanning_active", True))
        else:
            # Fall back to whatever the DB last persisted; default True.
            try:
                state = ctx.get("state") or {}
                overrides = (state or {}).get("config_overrides") or {}
                mm_settings = (overrides or {}).get("mm_engine_settings") or {}
                ctx["mm_scanning_active"] = bool(mm_settings.get("scanning_active", True))
            except Exception:
                ctx["mm_scanning_active"] = True
        return templates.TemplateResponse(request, "mm.html", context=ctx)

    # --- MM Status ---

    @router.get("/mm-status", response_class=HTMLResponse)
    @login_required
    async def mm_status_page(request: Request):
        ctx = await _base_context(request)

        mm_engine_inst = getattr(request.app.state, "mm_engine", None)
        last_funnel = None
        positions = {}
        cycle_count = 0
        scanning_active = False
        correlation_provider = None

        if mm_engine_inst is not None:
            last_funnel = getattr(mm_engine_inst, "last_funnel", None)
            positions = getattr(mm_engine_inst, "positions", {})
            cycle_count = getattr(mm_engine_inst, "cycle_count", 0)
            scanning_active = bool(getattr(mm_engine_inst, "_scanning_active", False))
            # Grab correlation provider info
            data_feeds = getattr(mm_engine_inst, "data_feeds", None)
            if data_feeds is not None:
                correlation_provider = getattr(data_feeds, "correlation", None)

        # Determine bot status label
        if not scanning_active:
            bot_status = "paused"
        elif last_funnel is None:
            bot_status = "idle"
        else:
            bot_status = "running"

        # Build per-pair signal table from last_funnel if available
        pair_rows = []
        top_rejections = []
        if last_funnel:
            rejects = last_funnel.get("rejects") or {}
            # Sort by count descending for top-5 rejection breakdown
            top_rejections = sorted(rejects.items(), key=lambda x: x[1], reverse=True)[:5]

        # Correlation provider status
        corr_status = {
            "enabled": False,
            "provider_name": "Stub (unavailable)",
            "last_fetched": None,
            "dxy_direction": None,
            "confidence": None,
        }
        if correlation_provider is not None:
            provider_class = type(correlation_provider).__name__
            corr_status["enabled"] = not provider_class.startswith("Stub")
            corr_status["provider_name"] = provider_class
            # YFinanceCorrelationProvider exposes _cache_time
            cache_time = getattr(correlation_provider, "_cache_time", None)
            if cache_time is not None:
                corr_status["last_fetched"] = cache_time.strftime("%H:%M:%S UTC")
            cached_signal = getattr(correlation_provider, "_cache", {}).get("signal")
            if cached_signal is not None:
                corr_status["dxy_direction"] = getattr(cached_signal, "dxy_direction", None)
                corr_status["confidence"] = getattr(cached_signal, "confidence", None)

        # Open positions list for template
        open_positions = []
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        for sym, pos in positions.items():
            entry_time = getattr(pos, "entry_time", None)
            time_open_mins = None
            if entry_time:
                try:
                    time_open_mins = int((now - entry_time).total_seconds() / 60)
                except Exception:
                    pass
            open_positions.append({
                "symbol": sym,
                "direction": getattr(pos, "direction", ""),
                "entry_price": getattr(pos, "entry_price", 0),
                "stop_loss": getattr(pos, "stop_loss", 0),
                "current_level": getattr(pos, "current_level", 0),
                "partial_closed_pct": round(getattr(pos, "partial_closed_pct", 0) * 100, 0),
                "time_open_mins": time_open_mins,
                "formation_type": getattr(pos, "formation_type", ""),
                "confluence_grade": getattr(pos, "confluence_grade", ""),
                "target_l1": getattr(pos, "target_l1", 0),
                "target_l2": getattr(pos, "target_l2", 0),
                "target_l3": getattr(pos, "target_l3", 0),
            })

        ctx.update({
            "bot_status": bot_status,
            "cycle_count": cycle_count,
            "scanning_active": scanning_active,
            "last_funnel": last_funnel,
            "pair_rows": pair_rows,
            "open_positions": open_positions,
            "top_rejections": top_rejections,
            "corr_status": corr_status,
        })
        return templates.TemplateResponse(request, "mm_status.html", context=ctx)

    # --- MM Settings ---

    @router.get("/mm/settings", response_class=HTMLResponse)
    @login_required
    async def mm_settings_page(request: Request):
        ctx = await _base_context(request)
        active_repo = _get_repo_for_request(request)

        mm_settings = {}
        try:
            state = await active_repo.get_engine_state()
            if state:
                overrides = state.get("config_overrides", {}) or {}
                mm_settings = overrides.get("mm_engine_settings", {}) or {}
        except Exception:
            pass

        ctx["mm"] = {
            "risk_pct": mm_settings.get("mm_risk_pct", getattr(config, "mm_risk_per_trade_pct", 1.0)),
            "max_positions": mm_settings.get("mm_max_positions", getattr(config, "mm_max_positions", 3)),
            "leverage": mm_settings.get("mm_leverage", 10),
            "min_rr": mm_settings.get("mm_min_rr", 1.5),
            "min_confluence": mm_settings.get("mm_min_confluence", 40),
            "min_formation_quality": mm_settings.get("mm_min_formation_quality", 0.4),
            "scan_interval": mm_settings.get("mm_scan_interval", getattr(config, "mm_scan_interval_minutes", 5)),
            "cooldown_hours": mm_settings.get("mm_cooldown_hours", 4),
            "max_sl_pct": mm_settings.get("mm_max_sl_pct", 5.0),
            "initial_balance": getattr(config, "mm_initial_balance", 10000.0),
        }
        ctx["instance_id"] = config.instance_id
        return templates.TemplateResponse(request, "mm_settings.html", context=ctx)

    return router
