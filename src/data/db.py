from __future__ import annotations

from supabase import create_client, Client

from src.utils.logging import get_logger

logger = get_logger(__name__)


class Database:
    """Supabase client wrapper."""

    def __init__(self, url: str, key: str) -> None:
        self.client: Client = create_client(url, key)
        logger.info("database_connected", url=url[:30] + "...")

    def table(self, name: str):
        return self.client.table(name)
