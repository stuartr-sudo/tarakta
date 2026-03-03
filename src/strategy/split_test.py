"""A/B split test manager for LLM trade analyst.

Assigns each signal to either "control" (existing system) or "llm"
(Claude-filtered) group using deterministic hashing for reproducibility.
Tracks per-group statistics for comparison.
"""
from __future__ import annotations

import hashlib
from typing import Any

from src.config import Settings
from src.exchange.models import SignalCandidate
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SplitTestManager:
    """Manages A/B test group assignment and statistics."""

    def __init__(self, config: Settings) -> None:
        self._ratio = config.llm_split_ratio  # fraction assigned to "llm" group
        self._enabled = config.llm_enabled

    def assign_group(self, signal: SignalCandidate) -> str:
        """Assign a signal to a test group.

        Uses deterministic hashing of symbol + timestamp so the same signal
        always gets the same group (useful for debugging/replay).

        Returns:
            "control" or "llm"
        """
        if not self._enabled:
            return "control"

        # Deterministic hash based on symbol + entry time
        seed = f"{signal.symbol}:{signal.timestamp.isoformat()}"
        digest = hashlib.sha256(seed.encode()).hexdigest()
        # Convert first 8 hex chars to a float in [0, 1)
        hash_value = int(digest[:8], 16) / 0xFFFFFFFF

        group = "llm" if hash_value < self._ratio else "control"

        logger.debug(
            "split_test_assigned",
            symbol=signal.symbol,
            group=group,
            hash_value=round(hash_value, 4),
        )
        return group

    @staticmethod
    def compute_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute per-group comparison statistics from closed trades.

        Args:
            trades: List of trade dicts with at least:
                - test_group: "control" or "llm"
                - pnl_usd: float
                - pnl_percent: float
                - status: "closed"

        Returns:
            Dict with per-group stats and comparison metrics.
        """
        groups: dict[str, list[dict]] = {"control": [], "llm": []}
        for t in trades:
            group = t.get("test_group", "control")
            if group in groups:
                groups[group].append(t)

        result: dict[str, Any] = {}
        for group, group_trades in groups.items():
            if not group_trades:
                result[group] = {
                    "trade_count": 0,
                    "win_rate": 0.0,
                    "avg_pnl_pct": 0.0,
                    "total_pnl_usd": 0.0,
                    "avg_pnl_usd": 0.0,
                }
                continue

            wins = sum(1 for t in group_trades if (t.get("pnl_usd") or 0) > 0)
            pnls_usd = [t.get("pnl_usd") or 0.0 for t in group_trades]
            pnls_pct = [t.get("pnl_percent") or 0.0 for t in group_trades]
            count = len(group_trades)

            result[group] = {
                "trade_count": count,
                "win_rate": round(wins / count * 100, 1) if count else 0.0,
                "avg_pnl_pct": round(sum(pnls_pct) / count, 2) if count else 0.0,
                "total_pnl_usd": round(sum(pnls_usd), 2),
                "avg_pnl_usd": round(sum(pnls_usd) / count, 2) if count else 0.0,
            }

        # Basic significance test (z-test on win rates)
        ctrl = result.get("control", {})
        llm = result.get("llm", {})
        n_ctrl = ctrl.get("trade_count", 0)
        n_llm = llm.get("trade_count", 0)

        if n_ctrl >= 5 and n_llm >= 5:
            p_ctrl = ctrl["win_rate"] / 100
            p_llm = llm["win_rate"] / 100
            p_pool = (p_ctrl * n_ctrl + p_llm * n_llm) / (n_ctrl + n_llm)
            se = (p_pool * (1 - p_pool) * (1 / n_ctrl + 1 / n_llm)) ** 0.5 if p_pool > 0 and p_pool < 1 else 0
            z_score = (p_llm - p_ctrl) / se if se > 0 else 0
            result["comparison"] = {
                "z_score": round(z_score, 2),
                "significant_at_95": abs(z_score) >= 1.96,
                "llm_win_rate_delta": round(llm["win_rate"] - ctrl["win_rate"], 1),
                "llm_avg_pnl_delta": round(llm["avg_pnl_usd"] - ctrl["avg_pnl_usd"], 2),
            }
        else:
            min_needed = max(5 - n_ctrl, 5 - n_llm, 0)
            result["comparison"] = {
                "z_score": None,
                "significant_at_95": None,
                "message": f"Need at least 5 trades per group for comparison (need ~{min_needed} more)",
            }

        return result
