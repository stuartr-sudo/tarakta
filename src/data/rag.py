"""Trade Knowledge RAG — Hybrid retrieval for past trade intelligence.

Ports the doubleclicker RAG system (Supabase pgvector + FTS + RRF)
to Python so Agent 1 and Agent 2 can learn from historical trades.

Requires:
  - pgvector extension enabled on Supabase
  - knowledge_sources + knowledge_chunks tables (see migrations/)
  - OPENAI_API_KEY env var for text-embedding-3-small (1536d)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import httpx

from src.data.db import Database
from src.utils.logging import get_logger

logger = get_logger(__name__)

# RAG scoping constants
RAG_USER = "tarakta"
RAG_DOMAIN = "trades"
SOURCE_TYPE = "trade_result"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


class TradeRAG:
    """Hybrid RAG for trade history — ingest closed trades, retrieve similar setups."""

    def __init__(self, db: Database, openai_api_key: str) -> None:
        self.db = db
        self._openai_key = openai_api_key
        self._http: httpx.AsyncClient | None = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    # ── Embedding ────────────────────────────────────────────────────

    async def _embed(self, text: str) -> list[float]:
        """Generate embedding via OpenAI text-embedding-3-small."""
        client = await self._get_http()
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self._openai_key}"},
            json={"model": EMBEDDING_MODEL, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed up to 200 texts."""
        if not texts:
            return []
        client = await self._get_http()
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self._openai_key}"},
            json={"model": EMBEDDING_MODEL, "input": texts[:200]},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        # Sort by index to maintain order
        data.sort(key=lambda x: x["index"])
        return [d["embedding"] for d in data]

    # ── Trade Ingestion ──────────────────────────────────────────────

    @staticmethod
    def _format_trade(trade: dict[str, Any]) -> str:
        """Convert a closed trade into searchable text content."""
        symbol = trade.get("symbol", "UNKNOWN")
        direction = trade.get("direction", "unknown")
        pnl = trade.get("pnl_usd", 0) or 0
        pnl_pct = trade.get("pnl_percent", 0) or 0
        outcome = "WIN" if pnl > 0 else "LOSS"
        entry_price = trade.get("entry_price", 0) or 0
        exit_price = trade.get("exit_price", 0) or 0
        exit_reason = trade.get("exit_reason", "unknown")
        score = trade.get("confluence_score", 0) or 0
        sl = trade.get("stop_loss", 0) or 0
        tp = trade.get("take_profit", 0) or 0
        reasons = trade.get("signal_reasons", "") or ""

        # Hold duration
        hold_str = ""
        entry_time = trade.get("entry_time")
        exit_time = trade.get("exit_time")
        if entry_time and exit_time:
            try:
                et = datetime.fromisoformat(str(entry_time).replace("Z", "+00:00"))
                xt = datetime.fromisoformat(str(exit_time).replace("Z", "+00:00"))
                hold_mins = int((xt - et).total_seconds() / 60)
                hold_str = f"Hold duration: {hold_mins} minutes."
            except Exception:
                pass

        # Agent analysis
        agent_reasoning = ""
        a2_reasoning = trade.get("last_agent2_reasoning", "") or ""
        if a2_reasoning:
            agent_reasoning = f"Agent 2 reasoning: {a2_reasoning}"

        return (
            f"{symbol} {direction.upper()} trade — {outcome}.\n"
            f"Entry: {entry_price:.6g}, Exit: {exit_price:.6g}. "
            f"PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%). "
            f"Exit reason: {exit_reason}. "
            f"Confluence score: {score:.0f}. "
            f"SL: {sl:.6g}, TP: {tp:.6g}. "
            f"{hold_str}\n"
            f"Signal reasons: {reasons}\n"
            f"{agent_reasoning}"
        ).strip()

    async def ingest_trade(self, trade: dict[str, Any]) -> bool:
        """Ingest a single closed trade as a knowledge chunk.

        Returns True if ingested, False if skipped or failed.
        """
        trade_id = trade.get("id", "")
        if not trade_id:
            return False

        content = self._format_trade(trade)
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        source_url = f"trade://{trade_id}"
        symbol = trade.get("symbol", "UNKNOWN")
        direction = trade.get("direction", "unknown")
        pnl = trade.get("pnl_usd", 0) or 0

        try:
            # Upsert knowledge_source
            source_data = {
                "user_name": RAG_USER,
                "domain": RAG_DOMAIN,
                "source_type": SOURCE_TYPE,
                "source_url": source_url,
                "title": f"{symbol} {direction} {'WIN' if pnl > 0 else 'LOSS'}",
                "description": f"Trade {trade_id}: {symbol} {direction} PnL ${pnl:+.2f}",
                "status": "processing",
                "content_hash": content_hash,
                "metafields": {
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "direction": direction,
                    "pnl_usd": pnl,
                    "exit_reason": trade.get("exit_reason", ""),
                    "confluence_score": trade.get("confluence_score", 0),
                },
            }

            result = await asyncio.to_thread(
                lambda: self.db.table("knowledge_sources")
                .upsert(source_data, on_conflict="user_name,domain,source_type,source_url")
                .select()
                .execute()
            )
            if not result.data:
                return False
            source_id = result.data[0]["id"]

            # Generate embedding
            embedding = await self._embed(content)

            # Upsert chunk
            chunk_data = {
                "user_name": RAG_USER,
                "domain": RAG_DOMAIN,
                "source_id": source_id,
                "chunk_index": 0,
                "chunk_hash": content_hash,
                "content": content,
                "embedding": json.dumps(embedding),
                "url": source_url,
                "heading": f"{symbol} {direction}",
                "metadata": {
                    "symbol": symbol,
                    "direction": direction,
                    "pnl_usd": pnl,
                    "exit_reason": trade.get("exit_reason", ""),
                },
            }

            await asyncio.to_thread(
                lambda: self.db.table("knowledge_chunks")
                .upsert(chunk_data, on_conflict="source_id,chunk_index")
                .execute()
            )

            # Mark source ready
            await asyncio.to_thread(
                lambda: self.db.table("knowledge_sources")
                .update({
                    "status": "ready",
                    "last_ingested_at": datetime.now(timezone.utc).isoformat(),
                })
                .eq("id", source_id)
                .execute()
            )

            logger.info(
                "rag_trade_ingested",
                trade_id=trade_id,
                symbol=symbol,
                direction=direction,
                pnl=round(pnl, 2),
            )
            return True

        except Exception as e:
            logger.warning("rag_ingest_failed", trade_id=trade_id, error=str(e)[:100])
            return False

    async def backfill_trades(self, limit: int = 200) -> int:
        """Ingest the most recent closed trades that aren't already in RAG.

        Returns count of newly ingested trades.
        """
        try:
            result = await asyncio.to_thread(
                lambda: self.db.table("trades")
                .select("*")
                .eq("status", "closed")
                .order("exit_time", desc=True)
                .limit(limit)
                .execute()
            )
            trades = result.data or []
        except Exception as e:
            logger.warning("rag_backfill_fetch_failed", error=str(e)[:100])
            return 0

        # Check which trade IDs are already ingested
        trade_ids = [t["id"] for t in trades if t.get("id")]
        existing_urls = set()
        try:
            result = await asyncio.to_thread(
                lambda: self.db.table("knowledge_sources")
                .select("source_url")
                .eq("user_name", RAG_USER)
                .eq("domain", RAG_DOMAIN)
                .eq("source_type", SOURCE_TYPE)
                .execute()
            )
            existing_urls = {r["source_url"] for r in (result.data or [])}
        except Exception:
            pass  # If check fails, we'll upsert (idempotent)

        ingested = 0
        for trade in trades:
            tid = trade.get("id", "")
            if f"trade://{tid}" in existing_urls:
                continue
            if await self.ingest_trade(trade):
                ingested += 1

        logger.info("rag_backfill_complete", total=len(trades), ingested=ingested)
        return ingested

    # ── Retrieval ────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        min_similarity: float = 0.15,
    ) -> list[dict[str, Any]]:
        """Hybrid search (dense + lexical + RRF) for similar trade knowledge.

        Returns list of dicts with keys: content, heading, rrf_score, similarity.
        """
        try:
            query_embedding = await self._embed(query)

            result = await asyncio.to_thread(
                lambda: self.db.client.rpc(
                    "rag_hybrid_search",
                    {
                        "p_user_name": RAG_USER,
                        "p_domain": RAG_DOMAIN,
                        "p_query": query,
                        "p_query_embedding": json.dumps(query_embedding),
                        "p_k": k,
                        "p_rrf_k": 60,
                        "p_min_similarity": min_similarity,
                    },
                ).execute()
            )

            return [
                {
                    "content": r.get("content", ""),
                    "heading": r.get("heading", ""),
                    "rrf_score": r.get("rrf_score", 0),
                    "similarity": r.get("similarity", 0),
                    "metadata": r.get("metadata", {}),
                }
                for r in (result.data or [])
            ]

        except Exception as e:
            logger.warning("rag_retrieve_failed", query=query[:60], error=str(e)[:100])
            return []

    async def retrieve_for_symbol(
        self,
        symbol: str,
        direction: str | None = None,
        setup_context: str = "",
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve past trade knowledge relevant to a specific symbol/setup.

        Builds a semantic query from the symbol, direction, and setup context.
        """
        parts = [symbol]
        if direction:
            parts.append(direction.upper())
        if setup_context:
            parts.append(setup_context)
        query = " ".join(parts)
        return await self.retrieve(query, k=k)

    def format_context(self, results: list[dict[str, Any]], max_results: int = 5) -> str:
        """Format RAG results into a text block for agent prompts."""
        if not results:
            return "  No similar past trades found in knowledge base."

        lines = []
        for i, r in enumerate(results[:max_results], 1):
            sim = r.get("similarity", 0)
            sim_str = f" (relevance: {sim * 100:.0f}%)" if sim else ""
            lines.append(f"  [{i}] {r['content']}{sim_str}")

        return "\n".join(lines)

    async def close(self) -> None:
        """Clean up HTTP client."""
        if self._http and not self._http.is_closed:
            await self._http.aclose()
