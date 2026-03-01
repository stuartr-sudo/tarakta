from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from src.config import Settings
from src.data.repository import Repository


def create_dashboard_app(config: Settings, repo: Repository, exchange=None) -> FastAPI:
    """Create the FastAPI dashboard application."""
    app = FastAPI(title="Tarakta", docs_url=None, redoc_url=None)

    app.add_middleware(SessionMiddleware, secret_key=config.session_secret)

    # Static files
    import os

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Store config, repo, and exchange in app state
    app.state.config = config
    app.state.repo = repo
    app.state.exchange = exchange

    # Register routes
    from src.dashboard.routes import create_router as create_page_router
    from src.dashboard.api import create_router as create_api_router

    app.include_router(create_page_router(config, repo))
    app.include_router(
        create_api_router(
            repo,
            exchange,
            api_key=config.kraken_api_key,
            api_secret=config.kraken_api_secret,
        ),
        prefix="/api",
    )

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "tarakta"}

    return app
