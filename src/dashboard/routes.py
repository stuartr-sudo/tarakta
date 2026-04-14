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
        return templates.TemplateResponse(request, "mm.html", context=ctx)

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
