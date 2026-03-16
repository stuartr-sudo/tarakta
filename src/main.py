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


def _create_market_exchange(market_name: str, market_config, config: Settings, logger):
    """Create exchange + paper wrapper for a single market."""
    connector_name = market_config.connector

    # Build connector kwargs based on connector type
    if connector_name.startswith("binance"):
        live_exchange = create_exchange(
            config.exchange_name,
            market_config.api_key or config.binance_api_key,
            market_config.api_secret or config.binance_api_secret,
            account_type=market_config.account_type,
            leverage=market_config.leverage,
            margin_mode=market_config.margin_mode,
        )
    else:
        # Non-crypto connectors (yfinance, alpaca, etc.)
        # Ensure connectors module is imported to register them
        import src.exchange.connectors  # noqa: F401
        from src.exchange.factory import create_exchange as factory_create

        live_exchange = factory_create(
            connector_name,
            symbol_universe=market_config.symbol_universe,
            api_key=market_config.api_key,
            api_secret=market_config.api_secret,
        )

    # Always paper trade non-crypto markets initially, and crypto if config says paper
    is_paper = config.trading_mode == "paper" or not connector_name.startswith("binance")
    if is_paper:
        exchange = PaperExchange(
            initial_balance=market_config.initial_balance,
            live_exchange=live_exchange,
            account_type=market_config.account_type,
            leverage=market_config.leverage,
        )
        logger.info(
            "market_paper_mode",
            market=market_name,
            connector=connector_name,
            balance=market_config.initial_balance,
        )
    else:
        exchange = live_exchange
        logger.info("market_live_mode", market=market_name, connector=connector_name)

    return live_exchange, exchange


async def main() -> None:
    config = Settings()
    setup_logging(log_level=config.log_level, log_format=config.log_format)
    logger = get_logger("tarakta")

    logger.info("tarakta_starting", mode=config.trading_mode, markets=list(config.markets.keys()))

    # Database (scoped to this instance)
    db = Database(config.supabase_url, config.supabase_key)
    repo = Repository(db, instance_id=config.instance_id)

    # Create engines for each enabled market
    engines: dict[str, TradingEngine] = {}
    live_exchanges = []
    primary_exchange = None  # For dashboard (first market)

    for market_name, market_config in config.markets.items():
        if not market_config.enabled:
            continue

        try:
            live_exchange, exchange = _create_market_exchange(market_name, market_config, config, logger)
            live_exchanges.append(live_exchange)

            if primary_exchange is None:
                primary_exchange = exchange

            candle_manager = CandleManager(exchange=live_exchange, repo=repo)

            engine = TradingEngine(
                config=config,
                exchange=exchange,
                repo=repo,
                candle_manager=candle_manager,
            )
            engine._market_name = market_name

            engines[market_name] = engine
            logger.info("market_engine_created", market=market_name, connector=market_config.connector)

        except Exception as e:
            logger.error("market_engine_failed", market=market_name, error=str(e), exc_info=True)
            continue

    if not engines:
        logger.critical("no_engines_created", hint="Check your market configuration")
        return

    # Dashboard (uses primary market's exchange for live data)
    from src.dashboard.app import create_dashboard_app

    dashboard_app = create_dashboard_app(config, repo, primary_exchange, engine=list(engines.values())[0], engines=engines)
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

    async def shutdown_all():
        for engine in engines.values():
            await engine.shutdown()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown_all()))

    # Run all market engines concurrently (with restart-on-crash)
    async def run_engine(name: str, engine: TradingEngine):
        backoff = 30
        max_backoff = 600  # 10 min cap
        while True:
            try:
                await engine.run()
                # Engine exited cleanly (shutdown requested)
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.critical("engine_fatal", market=name, error=str(e), exc_info=True)
                try:
                    await repo.log_error(f"engine_{name}", "critical", str(e))
                except Exception:
                    pass
                logger.info("engine_restart_backoff", market=name, seconds=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    try:
        await asyncio.gather(*[run_engine(name, eng) for name, eng in engines.items()])
    finally:
        for live_ex in live_exchanges:
            try:
                await live_ex.close()
            except Exception:
                pass
        logger.info("tarakta_stopped")


if __name__ == "__main__":
    asyncio.run(main())
