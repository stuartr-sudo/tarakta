from __future__ import annotations

from fastapi import APIRouter, Request

from src.dashboard.auth import login_required
from src.data.repository import Repository


def create_router(repo: Repository) -> APIRouter:
    router = APIRouter()

    @router.get("/portfolio")
    @login_required
    async def get_portfolio(request: Request):
        snapshot = await repo.get_latest_snapshot()
        history = await repo.get_snapshot_history(hours=168)
        return {
            "current": snapshot,
            "history": history,
        }

    @router.get("/trades/open")
    @login_required
    async def get_open_trades(request: Request):
        trades = await repo.get_open_trades()
        return {"trades": trades}

    @router.get("/stats")
    @login_required
    async def get_stats(request: Request):
        return await repo.get_trade_stats()

    @router.get("/signals/recent")
    @login_required
    async def get_recent_signals(request: Request):
        signals = await repo.get_recent_signals(limit=20)
        return {"signals": signals}

    return router
