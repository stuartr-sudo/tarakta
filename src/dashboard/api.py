"""MM Engine API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.dashboard.auth import admin_required, login_required
from src.data.repository import Repository
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_router(repo: Repository) -> APIRouter:
    router = APIRouter()

    def _get_mm_engine(request: Request):
        return getattr(request.app.state, "mm_engine", None)

    # --- MM Status ---

    @router.get("/mm/status")
    @login_required
    async def get_mm_status(request: Request):
        mm = _get_mm_engine(request)
        if not mm:
            return {"available": False, "scanning_active": False}
        status = await mm.get_status()
        status["available"] = True
        return status

    # --- MM Start/Stop ---

    @router.post("/mm/begin")
    @admin_required
    async def begin_mm_scanning(request: Request):
        mm = _get_mm_engine(request)
        if not mm:
            return JSONResponse({"success": False, "error": "MM Engine not available"}, status_code=503)
        mm.begin_scanning()
        try:
            state = await repo.get_engine_state()
            if state:
                overrides = state.get("config_overrides", {}) or {}
                if not isinstance(overrides, dict):
                    overrides = {}
                mm_settings = overrides.get("mm_engine_settings", {})
                if not isinstance(mm_settings, dict):
                    mm_settings = {}
                mm_settings["scanning_active"] = True
                overrides["mm_engine_settings"] = mm_settings
                state["config_overrides"] = overrides
                await repo.upsert_engine_state(state)
        except Exception as e:
            logger.warning("begin_mm_persist_failed", error=str(e))
        return {"success": True, "message": "MM Engine scanning started"}

    @router.post("/mm/stop")
    @admin_required
    async def stop_mm_scanning(request: Request):
        mm = _get_mm_engine(request)
        if not mm:
            return JSONResponse({"success": False, "error": "MM Engine not available"}, status_code=503)
        mm.stop_scanning()
        try:
            state = await repo.get_engine_state()
            if state:
                overrides = state.get("config_overrides", {}) or {}
                if not isinstance(overrides, dict):
                    overrides = {}
                mm_settings = overrides.get("mm_engine_settings", {})
                if not isinstance(mm_settings, dict):
                    mm_settings = {}
                mm_settings["scanning_active"] = False
                overrides["mm_engine_settings"] = mm_settings
                state["config_overrides"] = overrides
                await repo.upsert_engine_state(state)
        except Exception as e:
            logger.warning("stop_mm_persist_failed", error=str(e))
        return {"success": True, "message": "MM Engine scanning paused"}

    # --- MM Settings ---

    _MM_SETTINGS_VALIDATION = {
        "mm_risk_pct":              (float, 0.1, 10.0),
        "mm_max_positions":         (int,   1,   50),
        "mm_leverage":              (int,   1,   100),
        "mm_min_rr":                (float, 0.5, 10.0),
        "mm_min_confluence":        (float, 5.0, 100.0),
        "mm_min_formation_quality": (float, 0.05, 1.0),
        "mm_scan_interval":         (float, 1.0, 60.0),
        "mm_cooldown_hours":        (float, 0.0, 48.0),
        "mm_max_sl_pct":            (float, 1.0, 15.0),
    }

    @router.post("/mm/settings")
    @login_required
    async def save_mm_settings(request: Request):
        body = await request.json()
        parsed = {}
        for key, val in body.items():
            if key not in _MM_SETTINGS_VALIDATION:
                continue
            typ, lo, hi = _MM_SETTINGS_VALIDATION[key]
            try:
                v = typ(val)
                v = max(lo, min(hi, v))
                parsed[key] = v
            except (ValueError, TypeError):
                continue

        if not parsed:
            return JSONResponse({"error": "No valid settings"}, status_code=400)

        try:
            state = await repo.get_engine_state() or {}
            overrides = state.get("config_overrides", {}) or {}
            mm_settings = overrides.get("mm_engine_settings", {}) or {}
            mm_settings.update(parsed)
            overrides["mm_engine_settings"] = mm_settings
            state["config_overrides"] = overrides
            await repo.upsert_engine_state(state)
        except Exception as e:
            logger.warning("mm_settings_save_failed", error=str(e))
            return JSONResponse({"error": str(e)}, status_code=500)

        mm = _get_mm_engine(request)
        if mm:
            if "mm_max_positions" in parsed:
                mm.max_positions = parsed["mm_max_positions"]
            if "mm_scan_interval" in parsed:
                mm.scan_interval = parsed["mm_scan_interval"] * 60
            if "mm_cooldown_hours" in parsed:
                mm._cooldown_hours = parsed["mm_cooldown_hours"]

        logger.info("mm_settings_saved", settings=parsed)
        return {"success": True, "saved": parsed}

    return router
