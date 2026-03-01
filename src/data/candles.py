from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.data.repository import Repository
from src.exchange.client import KrakenClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


class CandleManager:
    """Manages candle data with DB caching to minimize API calls."""

    def __init__(self, exchange: KrakenClient, repo: Repository) -> None:
        self.exchange = exchange
        self.repo = repo

    async def get_candles(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> pd.DataFrame:
        """Fetch candles with cache-first strategy."""
        # Try cache
        cached = await self.repo.get_cached_candles(symbol, timeframe, limit=limit)

        if cached and len(cached) >= limit * 0.9:
            df = self._rows_to_df(cached)
            # Skip refresh if cache is fresh enough for this timeframe
            last_ts = df.index[-1]
            now = datetime.now(timezone.utc)
            age_seconds = (now - last_ts).total_seconds()
            tf_secs = TF_SECONDS.get(timeframe, 900)
            if age_seconds < tf_secs:
                return df.tail(limit)
            # Fetch only new candles since last cached
            since_ms = int(last_ts.timestamp() * 1000) + 1
            try:
                new_candles = await self.exchange.fetch_candles(
                    symbol, timeframe, limit=50, since=since_ms
                )
                if not new_candles.empty:
                    await self._cache_df(symbol, timeframe, new_candles)
                    df = pd.concat([df, new_candles]).drop_duplicates()
            except Exception as e:
                logger.warning("candle_refresh_failed", symbol=symbol, tf=timeframe, error=str(e))
            return df.tail(limit)

        # Full fetch
        try:
            df = await self.exchange.fetch_candles(symbol, timeframe, limit=limit)
            if not df.empty:
                await self._cache_df(symbol, timeframe, df)
            return df
        except Exception as e:
            logger.error("candle_fetch_failed", symbol=symbol, tf=timeframe, error=str(e))
            # Fall back to whatever cache we have
            if cached:
                return self._rows_to_df(cached)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    async def _cache_df(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        rows = []
        for ts, row in df.iterrows():
            rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )
        await self.repo.upsert_candles(symbol, timeframe, rows)

    def _rows_to_df(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
