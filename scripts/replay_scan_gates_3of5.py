"""Run replay with the 5-gate framework, threshold = 3 of 5.

Gates:
  1. Valid M/W formation (auto-pass at this stage)
  2. HTF aligned (auto-pass; real HTF veto fires downstream)
  3. Course-specific variant (not bare "standard")
  4. Hammer/engulfing at peak2 (formation.confirmed)
  5. At LOD/LOW or HOD/HOW (formation.at_key_level)
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.strategy.mm_engine as eng  # noqa: E402

_orig_init = eng.MMEngine.__init__


def _init_with_gates(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self._gate_threshold = 3


eng.MMEngine.__init__ = _init_with_gates  # type: ignore[method-assign]
print("# Gate framework: 3 of 5 required\n", flush=True)

from scripts.replay_scan import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
