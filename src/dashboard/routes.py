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

    async def _base_context(request: Request) -> dict:
        """Common context for all authenticated pages (mode banner, nav)."""
        state = await repo.get_engine_state()
        role = request.session.get("role", "viewer")
        return {
            "request": request,
            "state": state,
            "config": config,
            "role": role,
            "enabled_markets": enabled_markets,
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
        ctx["snapshot"] = await repo.get_latest_snapshot() or {}
        ctx["trades"] = await repo.get_open_trades(mode=config.trading_mode)
        ctx["signals"] = await repo.get_recent_signals(limit=10)
        ctx["stats"] = await repo.get_trade_stats(mode=config.trading_mode)

        # Override snapshot P&L with fresh DB-computed values
        # so the dashboard shows ground truth, not stale in-memory values
        stats = ctx["stats"]
        ctx["snapshot"]["daily_pnl_usd"] = stats.get("daily_pnl", 0)
        ctx["snapshot"]["total_pnl_usd"] = stats.get("total_pnl", 0)

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
        status = request.query_params.get("status", "all")
        page = int(request.query_params.get("page", 1))
        ctx["trades"] = await repo.get_trades(status=status, mode=config.trading_mode, page=page, per_page=25)
        ctx["stats"] = await repo.get_trade_stats(mode=config.trading_mode)
        ctx["current_status"] = status
        ctx["current_page"] = page
        return templates.TemplateResponse("trades.html", ctx)

    @router.get("/signals", response_class=HTMLResponse)
    @login_required
    async def signals_page(request: Request):
        ctx = await _base_context(request)
        page = int(request.query_params.get("page", 1))
        ctx["signals"] = await repo.get_signals(page=page, per_page=50)
        ctx["current_page"] = page
        return templates.TemplateResponse("signals.html", ctx)

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
        ctx["llm_enabled"] = state.get("llm_enabled", config.llm_enabled)
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

    @router.post("/settings/toggle-llm")
    @admin_required
    async def toggle_llm(request: Request):
        state = await repo.get_engine_state()
        if state:
            current = state.get("llm_enabled", config.llm_enabled)
            state["llm_enabled"] = not current
            await repo.upsert_engine_state(state)
        else:
            await repo.upsert_engine_state({"llm_enabled": True})
        return RedirectResponse(url="/settings", status_code=303)

    return router
