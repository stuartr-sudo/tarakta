"""One-shot E2E: run MMCommittee.review() against the real repo + CLI backend."""
import asyncio
import sys

sys.path.insert(0, "/Users/stuarta/tarakta")

from src.config import Settings
from src.data.db import Database
from src.data.repository import Repository
from src.strategy.mm_committee import MMCommittee


async def main() -> None:
    config = Settings()
    print("api_key_present:", bool(config.anthropic_api_key))
    print("committee_enabled:", config.mm_committee_enabled,
          "mode:", config.mm_committee_mode)
    print("instance:", config.instance_id)
    db = Database(config.supabase_url, config.supabase_key)
    repo = Repository(db, instance_id=config.instance_id)
    committee = MMCommittee(config=config, repo=repo)
    client = committee._get_client()
    print("backend:", type(client).__name__, getattr(client, "cli_path", ""))

    ctx = {
        "symbol": "BTC/USDT:USDT",
        "direction": "long",
        "formation_type": "W",
        "formation_variant": "multi_session",
        "formation_timeframe": "1h",
        "entry_price": 60000.0,
        "sl_ref": 59400.0,
        "grade": "B",
        "score_pct": 62.0,
        "retest_met": 3,
        "htf_trend_4h": "bullish",
        "htf_4h_strength": 0.62,
        "htf_4h_accel": False,
        "htf_trend_1d": "bullish",
        "price_vs_50ema_pct": 0.4,
        "price_vs_200ema_pct": 2.1,
        "counter_trend": False,
        "session_name": "us",
        "minutes_in": 95,
        "weekly_phase": "midweek_reversal",
        "dow": "Friday",
        "recent_trades": [],
        "cycle_count": 0,
        "note": "e2e_cli_backend_test_2026-07-18",
    }
    verdict = await committee.review(ctx)
    if verdict is None:
        print("verdict: None (committee disabled)")
    else:
        print("verdict:", verdict.decision)
        print("reason:", verdict.reason[:220])


asyncio.run(main())
