"""Wrapper around replay_scan.py that DISABLES 15M-dependent setup paths.

The MM engine has a fallback chain when no 1H W/M formation is detected:
  1. three_hits (1H)
  2. 200_ema_rejection (1H + 4H + 15M)  ← 15M
  3. board_meeting (1H)
  4. brinks (15M)                       ← 15M
  5. nyc_reversal (1H)
  6. stophunt (1H)
  7. scalp_vwap_rsi (15M)               ← 15M
  8. ribbon (15M)                       ← 15M

This wrapper makes the candle manager return an empty DataFrame for the
"15m" timeframe, which causes all 15M-dependent fallbacks to skip
(they all guard on `candles_15m.empty`). The 1H W/M detector and the
1H-only fallbacks (three_hits, board_meeting, nyc_reversal, stophunt)
still fire normally.

Tests the hypothesis that 15M-driven setups are introducing noise that
hurts strategy performance more than they help.

Run: python3 scripts/replay_scan_no_15m.py --symbols ... --days 30 --pnl
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

import scripts.replay_scan as rs  # noqa: E402

_orig_get_candles = rs.ReplayCandleManager.get_candles


async def _get_candles_no_15m(self, symbol: str, tf: str, limit: int = 500) -> pd.DataFrame:
    if tf == "15m":
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return await _orig_get_candles(self, symbol, tf, limit)


rs.ReplayCandleManager.get_candles = _get_candles_no_15m  # type: ignore[method-assign]

print("# 15M timeframe DISABLED — Brinks, scalps, 200ema_rejection paths skip\n", flush=True)

if __name__ == "__main__":
    raise SystemExit(rs.main())
