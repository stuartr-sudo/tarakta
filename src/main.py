"""Tarakta — Market Makers Method trading bot.

Runs the MM Method engine: a standalone algorithmic trading system
that detects M/W formations, tracks levels, and manages positions
with zero LLM calls.
"""
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
from src.exchange.client import create_exchange
from src.exchange.paper import PaperExchange
from src.strategy.inverse_mirror import InverseMirrorEngine
from src.strategy.mm_engine import MMEngine
from src.utils.logging import get_logger, setup_logging


async def main() -> None:
    config = Settings()
    setup_logging(log_level=config.log_level, log_format=config.log_format)
    logger = get_logger("tarakta")

    logger.info("tarakta_starting", mode=config.trading_mode, bot_role=config.bot_role)

    # Database
    db = Database(config.supabase_url, config.supabase_key)
    repo = Repository(db, instance_id=config.instance_id)

    # Exchange — use first market config for credentials/leverage
    market_name = list(config.markets.keys())[0]
    market_config = config.markets[market_name]

    live_exchange = create_exchange(
        config.exchange_name,
        market_config.api_key or config.binance_api_key,
        market_config.api_secret or config.binance_api_secret,
        account_type=market_config.account_type,
        leverage=market_config.leverage,
        margin_mode=market_config.margin_mode,
    )

    mm_exchange = PaperExchange(
        initial_balance=config.mm_initial_balance,
        live_exchange=live_exchange,
        account_type=market_config.account_type,
        leverage=market_config.leverage,
    )
    logger.info("mm_paper_exchange_created", balance=config.mm_initial_balance)

    candle_manager = CandleManager(exchange=live_exchange, repo=repo)

    if config.bot_role == "inverse_mirror":
        source_repo = Repository(db, instance_id=config.inverse_source_instance_id)
        runtime_engine = InverseMirrorEngine(
            exchange=mm_exchange,
            repo=repo,
            source_repo=source_repo,
            config=config,
        )
        logger.info(
            "inverse_mirror_created",
            source_instance=config.inverse_source_instance_id,
            source_strategy=config.inverse_source_strategy,
            inverse_strategy=config.inverse_strategy_tag,
            poll_interval=config.inverse_poll_interval_seconds,
        )
    else:
        runtime_engine = MMEngine(
            exchange=mm_exchange,
            repo=repo,
            candle_manager=candle_manager,
            config=config,
            scan_interval_minutes=config.mm_scan_interval_minutes,
        )
        logger.info("mm_engine_created", scan_interval=config.mm_scan_interval_minutes, balance=config.mm_initial_balance)

    # Dashboard
    from src.dashboard.app import create_dashboard_app

    dashboard_app = create_dashboard_app(config, repo, mm_exchange)
    if isinstance(runtime_engine, MMEngine):
        dashboard_app.state.mm_engine = runtime_engine
    else:
        dashboard_app.state.inverse_mirror_engine = runtime_engine
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
        loop.add_signal_handler(sig, lambda: asyncio.create_task(runtime_engine.shutdown()))

    # Run MM engine with restart-on-crash
    backoff = 30
    max_backoff = 600
    while True:
        try:
            await runtime_engine.run()
            break
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.critical("runtime_engine_fatal", role=config.bot_role, error=str(e), exc_info=True)
            try:
                await repo.log_error(config.bot_role, "critical", str(e))
            except Exception:
                pass
            logger.info("runtime_engine_restart_backoff", role=config.bot_role, seconds=backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    # Cleanup
    try:
        await live_exchange.close()
    except Exception:
        pass
    logger.info("tarakta_stopped")


if __name__ == "__main__":
    asyncio.run(main())
