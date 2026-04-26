"""Wrapper around replay_scan.py that disables the SL-to-breakeven move
after TP1 hits — i.e. "let winners run."

Default simulator behavior:
  - 30% closes at TP1 (+1R partial)
  - SL moves to breakeven (+ fee buffer) on remaining 70%
  - If price reverses: remaining 70% exits at BE
  - Net realized: +0.3R per "win"

This wrapper:
  - 30% still closes at TP1 (+1R partial)
  - SL stays at original placement
  - Remaining 70% rides until TP2/TP3 OR original SL hits
  - Best case: TP3 = +2.0R realized
  - Worst case: original SL hit on remainder = -0.4R realized
  - Mid case: TP2 partial = +1.1R realized

Tests whether the BE-after-TP1 rule is structurally capping winners and
producing the avg-win 0.3R / avg-loss 1.0R asymmetry that prevents
positive expectancy.

Run: python3 scripts/replay_scan_let_run.py --symbols ... --days 30 --pnl
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.replay_scan as rs  # noqa: E402

rs.MOVE_SL_TO_BE = False
print("# SL-to-BE-after-TP1 DISABLED — winners ride to TP2/TP3 or original SL\n", flush=True)

if __name__ == "__main__":
    raise SystemExit(rs.main())
