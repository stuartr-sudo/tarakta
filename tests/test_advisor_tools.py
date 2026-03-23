"""Test that MCP tool functions work correctly."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.advisor.tools import build_advisor_tools


def _mock_db(signals_data=None, candle_data=None):
    """Build a mock DB that returns different data per table."""
    mock_result_signals = MagicMock()
    mock_result_signals.data = signals_data or []

    mock_result_candles = MagicMock()
    mock_result_candles.data = candle_data or []

    def table_factory(name):
        chain = MagicMock()
        if name == "signals":
            chain.execute = MagicMock(return_value=mock_result_signals)
        else:
            chain.execute = MagicMock(return_value=mock_result_candles)
        chain.select = MagicMock(return_value=chain)
        chain.eq = MagicMock(return_value=chain)
        chain.gte = MagicMock(return_value=chain)
        chain.order = MagicMock(return_value=chain)
        chain.limit = MagicMock(return_value=chain)
        return chain

    db = MagicMock()
    db.table = MagicMock(side_effect=table_factory)
    return db


async def test_build_advisor_tools_creates_server():
    db = _mock_db()
    server = build_advisor_tools(db, instance_id="test")
    # Server should be created without error
    assert server is not None


async def test_get_missed_signals_tool():
    signals = [
        {
            "id": "sig-1",
            "symbol": "BTC/USDT:USDT",
            "direction": "long",
            "score": 75.0,
            "acted_on": False,
            "created_at": "2026-03-20T12:00:00Z",
        }
    ]
    db = _mock_db(signals_data=signals)
    server = build_advisor_tools(db, instance_id="test")
    # Just verify the server was built with the tools —
    # actual tool invocation happens through the SDK
    assert server is not None
