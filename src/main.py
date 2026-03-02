from __future__ import annotations

import asyncio
import os
import signal
import threading

import uvicorn

from src.config import Settings
from src.data.candles import CandleManager
from src.data.db import Database
from src.data.repository import Repository
from src.engine.core import TradingEngine
from src.exchange.client import create_exchange
from src.exchange.paper import PaperExchange
from src.utils.logging import get_logger, setup_logging


async def main() -> None:
    config = Settings()
    setup_logging(log_level=config.log_level, log_format=config.log_format)
    logger = get_logger("tarakta")

    logger.info("tarakta_starting", mode=config.trading_mode)

    # Database
    db = Database(config.supabase_url, config.supabase_key)
    repo = Repository(db)

    # Exchange
    if config.exchange_name == "binance":
        api_key, api_secret = config.binance_api_key, config.binance_api_secret
    else:
        api_key, api_secret = config.kraken_api_key, config.kraken_api_secret
    live_exchange = create_exchange(config.exchange_name, api_key, api_secret)

    if config.trading_mode == "paper":
        exchange = PaperExchange(initial_balance=config.initial_balance, live_exchange=live_exchange)
        logger.info("paper_mode_active", balance=config.initial_balance)
    else:
        exchange = live_exchange
        logger.info("live_mode_active")

    # Candle manager
    candle_manager = CandleManager(exchange=live_exchange, repo=repo)

    # Trading engine
    engine = TradingEngine(
        config=config,
        exchange=exchange,
        repo=repo,
        candle_manager=candle_manager,
    )

    # Dashboard
    from src.dashboard.app import create_dashboard_app

    dashboard_app = create_dashboard_app(config, repo, exchange)
    port = int(os.getenv("PORT", config.port))

    dashboard_thread = threading.Thread(
        target=uvicorn.run,
        args=(dashboard_app,),
        kwargs={"host": "0.0.0.0", "port": port, "log_level": "warning"},
        daemon=True,
    )
    dashboard_thread.start()
    logger.info("dashboard_started", port=port)

    # Graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(engine.shutdown()))

    # Run engine
    try:
        await engine.run()
    except Exception as e:
        logger.critical("engine_fatal", error=str(e), exc_info=True)
        await repo.log_error("engine", "critical", str(e))
        raise
    finally:
        await live_exchange.close()
        logger.info("tarakta_stopped")


if __name__ == "__main__":
    asyncio.run(main())
