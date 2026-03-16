from __future__ import annotations

import os

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from src.config import Settings
from src.dashboard.auth import admin_required, login_required, verify_password
from src.data.repository import Repository

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

# Map market type -> TradingView exchange prefix
_TV_PREFIX_MAP = {
    "crypto": "BINANCE",
    "stocks": "NASDAQ",
    "commodities": "COMEX",
}


def create_router(config: Settings, repo: Repository) -> APIRouter:
    router = APIRouter()
    templates = Jinja2Templates(directory=TEMPLATE_DIR)

    # Build list of enabled markets for nav display
    enabled_markets = [name for name, mc in config.markets.items() if mc.enabled]

    def _get_repo_for_request(request: Request) -> Repository:
        """Return a Repository scoped to the requested instance_id.

        If ``?instance=<id>`` is in the query string and a matching repo exists
        in ``app.state.repos``, use it.  Otherwise fall back to the primary repo.
        For instances that don't have a pre-built repo (they run on a different
        deployment), create one on the fly sharing the same DB connection.
        """
        requested = request.query_params.get("instance")
        if not requested or requested == config.instance_id:
            return repo

        repos: dict = getattr(request.app.state, "repos", {})
        if requested in repos:
            return repos[requested]

        # Create a view-only repo for the other instance (shares DB connection)
        other_repo = Repository(repo.db, instance_id=requested)
        repos[requested] = other_repo
        return other_repo

    def _instance_id_for_request(request: Request) -> str:
        return request.query_params.get("instance") or config.instance_id

    async def _base_context(request: Request) -> dict:
        """Common context for all authenticated pages (mode banner, nav)."""
        active_repo = _get_repo_for_request(request)
        active_instance = _instance_id_for_request(request)
        try:
            state = await active_repo.get_engine_state()
        except Exception:
            state = None
        role = request.session.get("role", "viewer")

        # Fetch list of all known instances for tab switcher
        all_instances = []
        try:
            all_instances = await repo.get_all_instances()
        except Exception:
            pass
        # Always include the current instance even if DB is empty
        instance_ids = [i.get("instance_id") for i in all_instances]
        if config.instance_id not in instance_ids:
            all_instances.insert(0, {"instance_id": config.instance_id})

        return {
            "request": request,
            "state": state,
            "config": config,
            "role": role,
            "enabled_markets": enabled_markets,
            "db_offline": state is None,
            "active_instance": active_instance,
            "all_instances": all_instances,
        }

    @router.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return templates.TemplateResponse("login.html", {"request": request, "error": None})

    @router.post("/login")
    async def login_submit(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ):
        # Check admin credentials
        if username == config.dashboard_username and verify_password(
            password, config.dashboard_password_hash
        ):
            request.session["authenticated"] = True
            request.session["role"] = "admin"
            return RedirectResponse(url="/", status_code=303)

        # Check viewer credentials
        if (
            config.viewer_password_hash
            and username == config.viewer_username
            and verify_password(password, config.viewer_password_hash)
        ):
            request.session["authenticated"] = True
            request.session["role"] = "viewer"
            return RedirectResponse(url="/", status_code=303)

        return templates.TemplateResponse(
            "login.html", {"request": request, "error": "Invalid credentials"}
        )

    @router.get("/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse(url="/login", status_code=303)

    @router.get("/", response_class=HTMLResponse)
    @login_required
    async def dashboard_home(request: Request):
        ctx = await _base_context(request)
        active_repo = _get_repo_for_request(request)
        try:
            ctx["snapshot"] = await active_repo.get_latest_snapshot() or {}
            ctx["trades"] = await active_repo.get_open_trades(mode=config.trading_mode)
            ctx["signals"] = await active_repo.get_recent_signals(limit=10)
            ctx["stats"] = await active_repo.get_trade_stats(mode=config.trading_mode)
        except Exception:
            ctx["snapshot"] = {}
            ctx["trades"] = []
            ctx["signals"] = []
            ctx["stats"] = {}
            ctx["db_offline"] = True

        # Override snapshot values with fresh DB-computed ground truth
        stats = ctx["stats"]
        ctx["snapshot"]["daily_pnl_usd"] = stats.get("daily_pnl") or 0
        ctx["snapshot"]["total_pnl_usd"] = stats.get("total_pnl") or 0
        # Equity = initial balance + total realized P&L (from DB, not in-memory state)
        ctx["snapshot"]["balance_usd"] = config.initial_balance + (stats.get("total_pnl") or 0)

        # Main bot settings (margin_pct, leverage) from saved state
        state = ctx.get("state") or {}
        overrides = state.get("config_overrides") or {}
        main_settings = overrides.get("main_bot_settings", {}) if isinstance(overrides, dict) else {}
        ctx["main_margin_pct"] = main_settings.get("margin_pct", config.max_position_pct)
        ctx["main_leverage"] = main_settings.get("leverage", config.leverage)

        return templates.TemplateResponse("dashboard.html", ctx)

    @router.get("/trades", response_class=HTMLResponse)
    @login_required
    async def trades_page(request: Request):
        ctx = await _base_context(request)
        active_repo = _get_repo_for_request(request)
        status = request.query_params.get("status", "all")
        page = int(request.query_params.get("page", 1))
        try:
            ctx["trades"] = await active_repo.get_trades(status=status, mode=config.trading_mode, page=page, per_page=25)
            ctx["stats"] = await active_repo.get_trade_stats(mode=config.trading_mode)
        except Exception:
            ctx["trades"] = []
            ctx["stats"] = {}
            ctx["db_offline"] = True
        ctx["current_status"] = status
        ctx["current_page"] = page
        return templates.TemplateResponse("trades.html", ctx)

    @router.get("/signals", response_class=HTMLResponse)
    @login_required
    async def signals_page(request: Request):
        ctx = await _base_context(request)
        active_repo = _get_repo_for_request(request)
        page = int(request.query_params.get("page", 1))
        try:
            ctx["signals"] = await active_repo.get_signals(page=page, per_page=50)
        except Exception:
            ctx["signals"] = []
            ctx["db_offline"] = True
        ctx["current_page"] = page
        return templates.TemplateResponse("signals.html", ctx)

    @router.get("/analytics", response_class=HTMLResponse)
    @login_required
    async def analytics_page(request: Request):
        ctx = await _base_context(request)
        return templates.TemplateResponse("analytics.html", ctx)

    @router.get("/chart", response_class=HTMLResponse)
    @login_required
    async def chart_page(request: Request):
        ctx = await _base_context(request)
        symbol = request.query_params.get("symbol", "")
        market = request.query_params.get("market", "crypto")
        # Collect open position symbols for quick-pick buttons
        state = ctx.get("state") or {}
        open_positions = state.get("open_positions", {})
        ctx["open_symbols"] = list(open_positions.keys())
        ctx["initial_symbol"] = symbol
        ctx["tv_exchange_prefix"] = _TV_PREFIX_MAP.get(market, "BINANCE")
        return templates.TemplateResponse("chart.html", ctx)

    @router.get("/settings", response_class=HTMLResponse)
    @login_required
    async def settings_page(request: Request):
        ctx = await _base_context(request)
        state = ctx.get("state") or {}
        ctx["agent1_enabled"] = bool(config.agent_api_key and config.agent_enabled)
        ctx["refiner_agent_enabled"] = state.get(
            "refiner_agent_enabled",
            getattr(config, "refiner_agent_enabled", False),
        )
        ctx["position_agent_enabled"] = state.get(
            "position_agent_enabled",
            getattr(config, "position_agent_enabled", False),
        )
        # Agent models always start at config default on restart (cost-safe)
        overrides = state.get("config_overrides") or {}
        if not isinstance(overrides, dict):
            overrides = {}
        ctx["agent1_model"] = config.agent_model
        ctx["agent2_model"] = config.agent_model
        ctx["agent3_model"] = getattr(config, "position_agent_model", "gemini-3-flash-preview")
        ctx["available_agent_models"] = ["gemini-3-pro-preview", "gemini-3-flash-preview"]
        # Leverage & margin (same source as dashboard)
        main_settings = overrides.get("main_bot_settings", {}) or {}
        ctx["main_leverage"] = main_settings.get("leverage", config.leverage)
        ctx["main_margin_pct"] = main_settings.get("margin_pct", config.max_position_pct)
        return templates.TemplateResponse("settings.html", ctx)

    @router.post("/settings/toggle-mode")
    @admin_required
    async def toggle_mode(request: Request):
        state = await repo.get_engine_state()
        if state:
            new_mode = "paper" if state.get("mode") == "live" else "live"
            state["mode"] = new_mode
            await repo.upsert_engine_state(state)
        return RedirectResponse(url="/settings", status_code=303)

    @router.post("/settings/toggle-refiner-agent")
    @admin_required
    async def toggle_refiner_agent(request: Request):
        state = await repo.get_engine_state()
        if state:
            current = state.get(
                "refiner_agent_enabled",
                getattr(config, "refiner_agent_enabled", False),
            )
            state["refiner_agent_enabled"] = not current
            await repo.upsert_engine_state(state)
        else:
            await repo.upsert_engine_state({"refiner_agent_enabled": True})
        return RedirectResponse(url="/settings", status_code=303)

    @router.post("/settings/toggle-position-agent")
    @admin_required
    async def toggle_position_agent(request: Request):
        state = await repo.get_engine_state()
        if state:
            current = state.get(
                "position_agent_enabled",
                getattr(config, "position_agent_enabled", False),
            )
            state["position_agent_enabled"] = not current
            await repo.upsert_engine_state(state)
        else:
            await repo.upsert_engine_state({"position_agent_enabled": True})
        return RedirectResponse(url="/settings", status_code=303)

    return router
