"""Wrapper around replay_scan.py that zeroes out dead-factor weights so we
can see what grade distribution looks like under a smaller (more honest)
denominator.

Dead factors (per the 30-day × 10-symbol replay run on 2026-04-25):
  - stopping_volume_candle (15 pts) — fires 0.3%
  - ema_alignment           (8 pts) — never appeared in factor hit list
  - oi_behavior             (8 pts) — never appeared in factor hit list
  - correlation_confirmed   (4 pts) — never appeared in factor hit list

Total dead weight: 35 of 127 (28% of denominator).

Setting these to 0 shrinks AVAILABLE_MAX from 127 to ~92, so the same
absolute factor points convert to a higher percentage. This is a rough
proxy for Option A (conditional-max) — it answers: "If we stopped
penalising setups for missing factors that essentially never fire under
any condition, what grade do they get?"

Run: python3 scripts/replay_scan_dead_zeroed.py --symbols BTC,ETH,BNB,SOL,DOGE,LINK,AVAX,NEAR,APT,XRP --days 30
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.strategy.mm_confluence as conf  # noqa: E402

DEAD_FACTORS = [
    "stopping_volume_candle",
    "ema_alignment",
    "oi_behavior",
    "correlation_confirmed",
]

orig_max = conf.MAX_POSSIBLE
orig_avail = conf.AVAILABLE_MAX

for k in DEAD_FACTORS:
    if k in conf.WEIGHTS:
        conf.WEIGHTS[k] = 0.0

conf.MAX_POSSIBLE = sum(conf.WEIGHTS.values())
conf.AVAILABLE_MAX = conf.MAX_POSSIBLE - sum(
    conf.WEIGHTS[k] for k in conf.STUBBED_FACTORS if k in conf.WEIGHTS
)

print(
    f"# Dead-factor weights zeroed: {DEAD_FACTORS}\n"
    f"# MAX_POSSIBLE: {orig_max} → {conf.MAX_POSSIBLE}\n"
    f"# AVAILABLE_MAX: {orig_avail} → {conf.AVAILABLE_MAX}\n",
    flush=True,
)

# Import AFTER patching so any module-level reads of MAX/AVAILABLE see new values.
from scripts.replay_scan import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
