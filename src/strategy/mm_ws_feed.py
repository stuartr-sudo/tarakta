"""Binance USDS-M futures websocket feed — mark price + liquidation events.

Why this exists: stop-losses were enforced once per 5-minute scan cycle at
whatever price the poll observed. Measured result (41 stop-outs, May–Jul
2026): average fill −1.39R instead of −1.0R — ~$3k/quarter of pure
execution slippage (docs/STATUS_2026-07-19.md §2, docs/DATAFEEDS_2026-07-31.md).
``markPrice@1s`` gives a ~1-second-fresh price so the engine's fast-stop
loop can act within seconds of the level being crossed. This is execution
realism (a real trader's stop-market order rests on the exchange); no
strategy rule changes.

Streams (free, keyless, verified live 2026-07-31):
  - ``<symbol>@markPrice@1s`` — latest mark price per subscribed symbol
  - ``!forceOrder@arr``       — liquidation orders, market-wide (snapshot
    stream: at most one order per symbol per second, so cascade totals are
    undercounted — treat as event detection, not exact volume)

Uses aiohttp (already a project dependency). The subscribed symbol set is
dynamic: ``ensure_symbols`` updates the desired set and the connection is
rebuilt on change (symbol-universe changes are rare — a reconnect per
change is simpler and more robust than live SUBSCRIBE frames).
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from datetime import datetime, timezone

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Routed base per Binance's websocket restructuring: market-data streams live
# under /market; unrouted connections silently push nothing (verified live
# 2026-07-31 — the unrouted legacy path timed out on upgrade from two
# independent networks, the routed one streamed immediately).
WS_BASE = "wss://fstream.binance.com/market/stream"

# Reconnect backoff schedule (seconds). Binance also closes every
# connection at the 24h mark — that close is expected and reconnects
# on the first step.
_BACKOFF_STEPS = (1.0, 2.0, 5.0, 10.0, 30.0)
# A connection that stayed up this long resets the backoff ladder.
_HEALTHY_CONNECTION_S = 60.0


def normalize_symbol(symbol: str) -> str:
    """CCXT 'BTC/USDT:USDT' (or 'BTCUSDT') → binance stream 'btcusdt'."""
    return (
        symbol.replace("/", "").replace(":USDT", "").replace(":USD", "").lower()
    )


class BinanceWsFeed:
    """Maintains live mark prices + recent liquidation events.

    Consumers use :meth:`get_price` (returns ``(price, age_seconds)`` —
    callers decide their own staleness tolerance) and
    :meth:`liquidation_stats`. Run :meth:`run` as a background task; call
    :meth:`stop` to shut down.
    """

    LIQ_BUFFER_MAX = 2000

    def __init__(self, symbols: list[str] | None = None) -> None:
        self._desired: set[str] = {normalize_symbol(s) for s in (symbols or [])}
        self._resubscribe = asyncio.Event()
        self._running = True
        # norm_symbol -> (price, monotonic_ts)
        self._prices: dict[str, tuple[float, float]] = {}
        # (norm_symbol, side, notional_usd, monotonic_ts)
        self._liqs: deque[tuple[str, str, float, float]] = deque(
            maxlen=self.LIQ_BUFFER_MAX
        )
        self.connected = False
        self.last_message_at: datetime | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_symbols(self, symbols: list[str]) -> None:
        """Extend the subscribed set; reconnects only when it changes.

        Symbols are only added, never dropped mid-run — an open position's
        stream must survive the symbol falling out of the scan universe.
        """
        wanted = {normalize_symbol(s) for s in symbols}
        new = wanted - self._desired
        if new:
            self._desired |= new
            self._resubscribe.set()
            logger.info("mm_ws_symbols_added", added=sorted(new),
                        total=len(self._desired))

    def get_price(self, symbol: str) -> tuple[float, float] | None:
        """Return ``(mark_price, age_seconds)`` or None if never seen."""
        entry = self._prices.get(normalize_symbol(symbol))
        if entry is None:
            return None
        price, ts = entry
        return price, max(0.0, time.monotonic() - ts)

    def liquidation_stats(self, symbol: str, window_s: float = 300.0) -> dict:
        """Recent liquidation summary for a symbol.

        forceOrder side semantics: a SELL liquidation order means LONGS were
        liquidated (their positions force-sold), and vice versa.
        """
        norm = normalize_symbol(symbol)
        cutoff = time.monotonic() - window_s
        events = 0
        long_liq_usd = 0.0
        short_liq_usd = 0.0
        for sym, side, notional, ts in self._liqs:
            if sym != norm or ts < cutoff:
                continue
            events += 1
            if side == "SELL":
                long_liq_usd += notional
            else:
                short_liq_usd += notional
        return {
            "events": events,
            "long_liq_usd": round(long_liq_usd, 2),
            "short_liq_usd": round(short_liq_usd, 2),
        }

    def stop(self) -> None:
        self._running = False
        self._resubscribe.set()  # break out of any wait

    # ------------------------------------------------------------------
    # Message handling (pure — unit-testable without a connection)
    # ------------------------------------------------------------------

    def _handle_message(self, payload: dict) -> None:
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return
        event = data.get("e")
        if event == "markPriceUpdate":
            sym = str(data.get("s", "")).lower()
            try:
                price = float(data.get("p"))
            except (TypeError, ValueError):
                return
            if sym and price > 0:
                self._prices[sym] = (price, time.monotonic())
                self.last_message_at = datetime.now(timezone.utc)
        elif event == "forceOrder":
            order = data.get("o") or {}
            sym = str(order.get("s", "")).lower()
            side = str(order.get("S", "")).upper()
            try:
                qty = float(order.get("q") or 0)
                avg_price = float(order.get("ap") or 0)
            except (TypeError, ValueError):
                return
            if sym and side and qty > 0 and avg_price > 0:
                self._liqs.append((sym, side, qty * avg_price, time.monotonic()))
                self.last_message_at = datetime.now(timezone.utc)

    def _stream_url(self) -> str | None:
        if not self._desired:
            return None
        streams = [f"{s}@markPrice@1s" for s in sorted(self._desired)]
        streams.append("!forceOrder@arr")
        return f"{WS_BASE}?streams={'/'.join(streams)}"

    @staticmethod
    def _backoff(attempt: int) -> float:
        return _BACKOFF_STEPS[min(attempt, len(_BACKOFF_STEPS) - 1)]

    @staticmethod
    def _ssl_context():
        """Explicit CA bundle via certifi, falling back to system defaults.

        The Mac's framework Python has NO default CA path
        (ssl.get_default_verify_paths().cafile is None), so aiohttp's default
        context trusts nothing and every TLS handshake fails. ccxt/httpx work
        because they pass certifi explicitly — do the same. Never disable
        verification.
        """
        import ssl
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            return ssl.create_default_context()

    # ------------------------------------------------------------------
    # Connection loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        import aiohttp

        attempt = 0
        while self._running:
            url = self._stream_url()
            if url is None:
                # Nothing to subscribe to yet — wait for ensure_symbols.
                self._resubscribe.clear()
                try:
                    await asyncio.wait_for(self._resubscribe.wait(), timeout=30.0)
                except asyncio.TimeoutError:
                    pass
                continue
            self._resubscribe.clear()
            connected_at = time.monotonic()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        url, heartbeat=30.0, max_msg_size=2 ** 20,
                        ssl=self._ssl_context(),
                    ) as ws:
                        self.connected = True
                        logger.info("mm_ws_connected",
                                    symbols=len(self._desired))
                        while self._running and not self._resubscribe.is_set():
                            msg = await ws.receive(timeout=60.0)
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    self._handle_message(json.loads(msg.data))
                                except (json.JSONDecodeError, TypeError):
                                    continue
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.CLOSE,
                                aiohttp.WSMsgType.CLOSING,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("mm_ws_error", error=str(e)[:200])
            finally:
                self.connected = False
            if not self._running:
                break
            if self._resubscribe.is_set():
                # Symbol set changed — reconnect immediately with new URL.
                continue
            if time.monotonic() - connected_at >= _HEALTHY_CONNECTION_S:
                attempt = 0
            delay = self._backoff(attempt)
            attempt += 1
            logger.info("mm_ws_reconnect_wait", delay_s=delay, attempt=attempt)
            try:
                await asyncio.wait_for(self._resubscribe.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
        logger.info("mm_ws_stopped")
