from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from src.config import Settings
from src.data.repository import Repository


def create_dashboard_app(config: Settings, repo: Repository, exchange=None, engine=None, engines: dict | None = None) -> FastAPI:
    """Create the FastAPI dashboard application."""
    app = FastAPI(title="Tarakta", docs_url=None, redoc_url=None)

    app.add_middleware(SessionMiddleware, secret_key=config.session_secret)

    # Static files
    import os

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Store config, repo, exchange, and engine in app state
    app.state.config = config
    app.state.repo = repo
    app.state.exchange = exchange
    app.state.engine = engine
    app.state.engines = engines or {}

    # Register routes
    from src.dashboard.routes import create_router as create_page_router
    from src.dashboard.api import create_router as create_api_router

    app.include_router(create_page_router(config, repo))

    # Resolve API credentials for dashboard exchange client
    # Use crypto market config if available, fallback to legacy flat config
    crypto_market = config.markets.get("crypto")
    dash_api_key = (crypto_market.api_key if crypto_market else "") or config.binance_api_key
    dash_api_secret = (crypto_market.api_secret if crypto_market else "") or config.binance_api_secret
    dash_account_type = (crypto_market.account_type if crypto_market else "") or config.account_type

    app.include_router(
        create_api_router(
            repo,
            exchange,
            exchange_name=config.exchange_name,
            api_key=dash_api_key,
            api_secret=dash_api_secret,
            account_type=dash_account_type,
            engine=engine,
            engines=engines or {},
        ),
        prefix="/api",
    )

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "tarakta"}

    # Suppress favicon 404
    from fastapi.responses import Response

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    return app
