from __future__ import annotations

import httpx
from supabase import create_client, Client, ClientOptions

from src.utils.logging import get_logger

logger = get_logger(__name__)


class Database:
    """Supabase client wrapper."""

    def __init__(self, url: str, key: str) -> None:
        # Force HTTP/1.1 to avoid HTTP/2 stream errors (ConnectionTerminated,
        # StreamIDTooLowError) when the engine and dashboard share a client
        # across threads.
        options = ClientOptions(httpx_client=httpx.Client(http2=False))
        self.client: Client = create_client(url, key, options)
        logger.info("database_connected", url=url[:30] + "...")

    def table(self, name: str):
        return self.client.table(name)
