"""Wrapper around replay_scan.py that enables the experimental Lesson-7
quality gates on the MM engine before scanning.

Gates applied (after formation_found stage):
  1. Reject `variant="standard"` — require multi_session promotion or
     a course-specific variant (board_meeting, brinks, three_hits, etc.)
  2. Reject `not confirmed` for standard/multi_session variants —
     require hammer/engulfing close at peak2.

Hypothesis: humans pass on 80% of "valid" setups based on these two
filters. Forcing them as gates should radically tighten the funnel
toward the rare quality setups course teaches.

Run: python3 scripts/replay_scan_quality_gates.py --symbols ... --days 30 --pnl
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Patch BEFORE replay machinery imports the engine — we monkey-patch the
# MMEngine class so every instance picks up the flag.
import src.strategy.mm_engine as eng  # noqa: E402

_orig_init = eng.MMEngine.__init__


def _init_with_gates(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self._quality_gates_enabled = True


eng.MMEngine.__init__ = _init_with_gates  # type: ignore[method-assign]

print("# Lesson-7 quality gates ENABLED (variant!=standard, hammer at peak2)\n", flush=True)

from scripts.replay_scan import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
