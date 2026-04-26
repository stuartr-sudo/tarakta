"""Wrapper that reduces SWING_WINDOW from 5 to 3.

Tests whether faster swing confirmation (3 forward bars instead of 5)
admits earlier entries — i.e. closer to peak2 wick rather than 5 hours
after the actual swing low forms.

Trade-off: faster confirmation = potentially more false positives
(price not actually a swing low, just a brief dip). Backtest answers
whether the trade-off is favorable.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.strategy.mm_formations as fm  # noqa: E402

fm.SWING_WINDOW = 3
print("# SWING_WINDOW reduced 5 → 3 (faster swing confirmation)\n", flush=True)

# Re-import after patch so the formation detector picks up new value
import importlib  # noqa: E402
importlib.reload(fm)
fm.SWING_WINDOW = 3

import scripts.replay_scan as rs  # noqa: E402

# Also enable 3/5 gates and cap=10 for a fair comparison
import src.strategy.mm_engine as eng  # noqa: E402
_orig_init = eng.MMEngine.__init__


def _init_with_gates(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self._gate_threshold = 3


eng.MMEngine.__init__ = _init_with_gates  # type: ignore[method-assign]

if __name__ == "__main__":
    raise SystemExit(rs.main())
