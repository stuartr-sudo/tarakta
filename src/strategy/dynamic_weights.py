"""Dynamic confluence weight optimizer.

Adjusts confluence component weights based on historical per-component
win rates. Components that contribute to winning trades get more weight;
those that don't get less. Uses exponential decay so recent trades matter
more than old ones.

Safe by design:
- Weights always sum to 100 (95 scoring + 5 rr_bonus)
- Per-component bounds prevent any single factor from dominating or vanishing
- Max step per recalculation prevents wild swings
- Minimum sample size before adjusting any component
- Disabled by default (config.dynamic_weights_enabled = False)
"""
from __future__ import annotations

import math
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Recalculate weights every N closed trades
RECALC_INTERVAL = 10

# Minimum trades where a component contributed before we adjust its weight
MIN_COMPONENT_TRADES = 15

# Maximum weight change per recalculation (points)
MAX_STEP = 2.0

# Exponential decay half-life in trades
DECAY_HALF_LIFE = 30

# Sum of all scoring weights (excludes rr_bonus which stays fixed at 5)
TARGET_SUM = 95

# Per-component weight bounds (min, max)
WEIGHT_BOUNDS: dict[str, tuple[int, int]] = {
    "htf_trend": (12, 28),
    "market_structure": (8, 22),
    "order_block": (8, 22),
    "volume": (8, 22),
    "fvg": (5, 18),
    "liquidity_sweep": (5, 18),
    "premium_discount": (5, 18),
}

# Components that participate in dynamic weighting
SCORING_COMPONENTS = list(WEIGHT_BOUNDS.keys())


class DynamicWeightOptimizer:
    """Adjusts confluence component weights based on historical per-component win rates."""

    def __init__(self, default_weights: dict[str, int]) -> None:
        # Store defaults for the 7 scoring components (excluding rr_bonus)
        self._defaults: dict[str, float] = {
            k: float(v) for k, v in default_weights.items() if k in WEIGHT_BOUNDS
        }
        # Current weights (start at defaults)
        self._weights: dict[str, float] = dict(self._defaults)

        # Per-component tracking: list of (decay_weight, contributed, won)
        # We store aggregated weighted sums for efficiency
        self._comp_weighted_total: dict[str, float] = {k: 0.0 for k in SCORING_COMPONENTS}
        self._comp_weighted_wins: dict[str, float] = {k: 0.0 for k in SCORING_COMPONENTS}

        self._total_trades = 0
        self._decay_factor = math.log(2) / DECAY_HALF_LIFE

    def record_outcome(self, components: dict[str, float], is_win: bool) -> None:
        """Record which components contributed and whether the trade won.

        Args:
            components: Dict mapping component name to its score for this trade.
                        A score > 0 means the component contributed.
            is_win: Whether the trade was profitable.
        """
        if not components:
            return

        # Apply decay to all existing data before adding new observation
        decay = math.exp(-self._decay_factor)
        for k in SCORING_COMPONENTS:
            self._comp_weighted_total[k] *= decay
            self._comp_weighted_wins[k] *= decay

        # Record new observation (weight=1.0 for the newest trade)
        for k in SCORING_COMPONENTS:
            score = components.get(k, 0.0)
            if score > 0:
                self._comp_weighted_total[k] += 1.0
                if is_win:
                    self._comp_weighted_wins[k] += 1.0

        self._total_trades += 1

        logger.debug(
            "dynamic_weights_outcome",
            total_trades=self._total_trades,
            is_win=is_win,
            contributing=[k for k in SCORING_COMPONENTS if components.get(k, 0) > 0],
        )

        # Recalculate every N trades
        if self._total_trades % RECALC_INTERVAL == 0:
            self._recalculate()

    def _recalculate(self) -> None:
        """Recompute weights from per-component win rates."""
        old_weights = dict(self._weights)

        # Step 1: Compute raw weights based on win rates
        raw: dict[str, float] = {}
        for k in SCORING_COMPONENTS:
            total = self._comp_weighted_total[k]
            if total < MIN_COMPONENT_TRADES:
                # Not enough data — keep default
                raw[k] = self._defaults[k]
                continue

            win_rate = self._comp_weighted_wins[k] / total  # 0.0 to 1.0
            # Scale: win_rate=0.5 keeps default, 1.0 = +50%, 0.0 = -50%
            raw[k] = self._defaults[k] * (0.5 + win_rate)

        # Step 2: Clamp to per-component bounds
        for k in SCORING_COMPONENTS:
            lo, hi = WEIGHT_BOUNDS[k]
            raw[k] = max(lo, min(hi, raw[k]))

        # Step 3: Normalize to TARGET_SUM
        raw_sum = sum(raw.values())
        if raw_sum > 0:
            for k in raw:
                raw[k] = raw[k] / raw_sum * TARGET_SUM

        # Step 4: Apply MAX_STEP limit per component
        for k in SCORING_COMPONENTS:
            delta = raw[k] - self._weights[k]
            clamped_delta = max(-MAX_STEP, min(MAX_STEP, delta))
            raw[k] = self._weights[k] + clamped_delta

        # Step 5: Re-normalize after step clamping
        raw_sum = sum(raw.values())
        if raw_sum > 0:
            for k in raw:
                raw[k] = raw[k] / raw_sum * TARGET_SUM

        # Step 6: Final clamp to bounds
        for k in SCORING_COMPONENTS:
            lo, hi = WEIGHT_BOUNDS[k]
            raw[k] = max(lo, min(hi, raw[k]))

        self._weights = raw

        # Log changes
        changes = {
            k: round(self._weights[k] - old_weights[k], 2)
            for k in SCORING_COMPONENTS
            if abs(self._weights[k] - old_weights[k]) > 0.01
        }
        if changes:
            win_rates = {}
            for k in SCORING_COMPONENTS:
                total = self._comp_weighted_total[k]
                if total >= MIN_COMPONENT_TRADES:
                    win_rates[k] = round(self._comp_weighted_wins[k] / total, 3)
            logger.info(
                "weights_recalculated",
                total_trades=self._total_trades,
                changes=changes,
                new_weights={k: round(v, 1) for k, v in self._weights.items()},
                win_rates=win_rates,
            )

    def get_weights(self) -> dict[str, int]:
        """Return current weights as integers, including rr_bonus.

        Weights always sum to 100 (95 scoring + 5 rr_bonus).
        """
        result = {k: round(v) for k, v in self._weights.items()}
        result["rr_bonus"] = 5
        return result

    def get_status(self) -> dict[str, Any]:
        """Return optimizer status for dashboard/logging."""
        component_stats = {}
        for k in SCORING_COMPONENTS:
            total = self._comp_weighted_total[k]
            stats: dict[str, Any] = {
                "weight": round(self._weights[k], 1),
                "default": self._defaults[k],
                "samples": round(total, 1),
            }
            if total >= 3:
                stats["win_rate"] = round(self._comp_weighted_wins[k] / total, 3)
            component_stats[k] = stats

        return {
            "total_trades": self._total_trades,
            "weights": {k: round(v, 1) for k, v in self._weights.items()},
            "weight_sum": round(sum(self._weights.values()) + 5, 1),
            "components": component_stats,
        }

    def to_state(self) -> dict[str, Any]:
        """Serialize state for persistence (Supabase engine_state)."""
        return {
            "weights": {k: round(v, 2) for k, v in self._weights.items()},
            "comp_weighted_total": {k: round(v, 4) for k, v in self._comp_weighted_total.items()},
            "comp_weighted_wins": {k: round(v, 4) for k, v in self._comp_weighted_wins.items()},
            "total_trades": self._total_trades,
        }

    def from_state(self, state: dict[str, Any]) -> None:
        """Restore state from persistence."""
        if not state:
            return

        saved_weights = state.get("weights", {})
        for k in SCORING_COMPONENTS:
            if k in saved_weights:
                lo, hi = WEIGHT_BOUNDS[k]
                self._weights[k] = max(lo, min(hi, float(saved_weights[k])))

        saved_total = state.get("comp_weighted_total", {})
        saved_wins = state.get("comp_weighted_wins", {})
        for k in SCORING_COMPONENTS:
            if k in saved_total:
                self._comp_weighted_total[k] = float(saved_total[k])
            if k in saved_wins:
                self._comp_weighted_wins[k] = float(saved_wins[k])

        self._total_trades = int(state.get("total_trades", 0))

        logger.info(
            "dynamic_weights_loaded",
            total_trades=self._total_trades,
            weights={k: round(v, 1) for k, v in self._weights.items()},
        )
