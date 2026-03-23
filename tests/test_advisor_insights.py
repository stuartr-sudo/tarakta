"""Tests for advisor insights data layer and formatting."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.advisor.insights import (
    format_insights_for_agent,
    get_recent_insights,
    store_insight,
)
from src.advisor.runner import _parse_structured_output


def _mock_db(data=None):
    """Build a mock Supabase client."""
    mock_result = MagicMock()
    mock_result.data = data or []
    chain = MagicMock()
    chain.execute = MagicMock(return_value=mock_result)
    chain.select = MagicMock(return_value=chain)
    chain.eq = MagicMock(return_value=chain)
    chain.order = MagicMock(return_value=chain)
    chain.limit = MagicMock(return_value=chain)
    chain.upsert = MagicMock(return_value=chain)
    db = MagicMock()
    db.table = MagicMock(return_value=chain)
    return db


async def test_store_insight_calls_upsert():
    db = _mock_db()
    data = {
        "signals_analyzed": 10,
        "simulated_winners": 6,
        "simulated_losers": 4,
        "win_rate_pct": 60.0,
        "top_missed": [{"symbol": "ETH/USDT:USDT", "pnl_pct": 3.5}],
        "patterns": {"common_components": "sweep+displacement"},
        "recommendations": ["Lower min score to 62"],
        "full_analysis": "Analysis text...",
        "cost_usd": 0.05,
    }
    await store_insight(db, "main", data)
    db.table.assert_called_with("advisor_insights")


async def test_get_recent_insights_parses_json_fields():
    raw_row = {
        "id": "abc",
        "instance_id": "main",
        "run_date": "2026-03-22",
        "signals_analyzed": 8,
        "simulated_winners": 5,
        "simulated_losers": 3,
        "win_rate_pct": 62.5,
        "top_missed": json.dumps([{"symbol": "BTC/USDT:USDT", "pnl_pct": 4.2}]),
        "patterns": json.dumps({"common_components": "sweep+OB"}),
        "recommendations": json.dumps(["Widen entry zone tolerance"]),
        "full_analysis": "Full text here",
        "cost_usd": 0.08,
    }
    db = _mock_db(data=[raw_row])
    insights = await get_recent_insights(db, "main", limit=1)
    assert len(insights) == 1
    # JSON fields should be parsed to Python objects
    assert isinstance(insights[0]["top_missed"], list)
    assert isinstance(insights[0]["patterns"], dict)
    assert isinstance(insights[0]["recommendations"], list)
    assert insights[0]["recommendations"][0] == "Widen entry zone tolerance"


async def test_get_recent_insights_empty():
    db = _mock_db(data=[])
    insights = await get_recent_insights(db, "main")
    assert insights == []


def test_format_insights_for_agent_with_data():
    insights = [
        {
            "run_date": "2026-03-22",
            "signals_analyzed": 10,
            "simulated_winners": 7,
            "simulated_losers": 3,
            "win_rate_pct": 70.0,
            "patterns": {"common_components": "sweep+displacement", "avg_winner_score": 68.0},
            "recommendations": [
                "Lower minimum score threshold from 70 to 62",
                "Widen entry zone by 0.3%",
            ],
        }
    ]
    result = format_insights_for_agent(insights)
    assert "Advisor Insights" in result
    assert "70.0%" in result
    assert "10 missed signals" in result
    assert "Lower minimum score" in result


def test_format_insights_for_agent_empty():
    assert format_insights_for_agent([]) == ""


def test_format_insights_for_agent_zero_analyzed():
    insights = [{"signals_analyzed": 0, "simulated_winners": 0}]
    assert format_insights_for_agent(insights) == ""


def test_parse_structured_output_extracts_json():
    analysis = """Here is my analysis.

Summary: 5 signals analyzed, 3 winners.

```json
{
  "signals_analyzed": 5,
  "simulated_winners": 3,
  "simulated_losers": 2,
  "win_rate_pct": 60.0,
  "top_missed": [],
  "patterns": {},
  "recommendations": ["Lower threshold"]
}
```"""
    result = _parse_structured_output(analysis)
    assert result is not None
    assert result["signals_analyzed"] == 5
    assert result["win_rate_pct"] == 60.0
    assert result["recommendations"] == ["Lower threshold"]


def test_parse_structured_output_no_json():
    analysis = "Just plain text analysis with no JSON block."
    result = _parse_structured_output(analysis)
    assert result is None


def test_parse_structured_output_invalid_json():
    analysis = """Some text
```json
{invalid json here}
```"""
    result = _parse_structured_output(analysis)
    assert result is None
