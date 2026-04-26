"""Wrapper around replay_scan.py that simulates HYPOTHETICAL PERFECT ENTRIES.

For every detected formation, this overrides `entry_price = current_price`
with `entry_price = peak2_wick_price` — i.e. assumes the bot filled at the
exact retest wick the course teaches as the correct entry, instead of the
live tick at scan time.

This is a thought experiment, not realistic. The market may not have
re-touched peak2 wick after detection. But the result isolates a single
question: *if entry quality were perfect, would the strategy be
profitable?* If yes, the bottleneck is entry timing/positioning. If no,
the problem is downstream (TP ladder, signal selection, symbol fit, etc.).

Run: python3 scripts/replay_scan_perfect_entry.py --symbols ... --days 30 --pnl
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.strategy.mm_engine as eng  # noqa: E402

_orig_init = eng.MMEngine.__init__


def _init_with_perfect_entry(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self._hypothetical_perfect_entry = True


eng.MMEngine.__init__ = _init_with_perfect_entry  # type: ignore[method-assign]

print("# HYPOTHETICAL PERFECT ENTRY enabled (entry_price = peak2_wick_price)\n", flush=True)

from scripts.replay_scan import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
