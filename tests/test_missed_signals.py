from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.advisor.missed_signals import fetch_missed_signals


def _mock_db(data: list) -> MagicMock:
    """Build a mock Supabase client with chainable query."""
    mock_result = MagicMock()
    mock_result.data = data
    chain = MagicMock()
    chain.execute = MagicMock(return_value=mock_result)
    chain.select = MagicMock(return_value=chain)
    chain.eq = MagicMock(return_value=chain)
    chain.gte = MagicMock(return_value=chain)
    chain.order = MagicMock(return_value=chain)
    chain.limit = MagicMock(return_value=chain)
    mock_db = MagicMock()
    mock_db.table = MagicMock(return_value=chain)
    return mock_db


@pytest.mark.asyncio
async def test_fetch_missed_signals_returns_high_score_unacted():
    db = _mock_db(
        [
            {
                "id": "sig-1",
                "symbol": "ETH/USDT:USDT",
                "direction": "long",
                "score": 72.0,
                "reasons": {"sweep": True, "displacement": True},
                "components": {"sweep_level": 3200.0},
                "current_price": 3250.0,
                "acted_on": False,
                "created_at": "2026-03-20T12:00:00Z",
            },
        ]
    )
    signals = await fetch_missed_signals(
        db, instance_id="main", min_score=60.0, limit=50, days_back=7
    )
    assert len(signals) == 1
    assert signals[0]["symbol"] == "ETH/USDT:USDT"
    assert signals[0]["acted_on"] is False


@pytest.mark.asyncio
async def test_fetch_missed_signals_empty_result():
    db = _mock_db([])
    signals = await fetch_missed_signals(db, instance_id="main")
    assert signals == []
