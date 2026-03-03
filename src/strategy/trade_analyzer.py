"""Post-trade analysis: structured outcome logging + XGBoost pattern recognition.

After every closed trade, logs the full context (entry conditions, market
state, holding time, exit reason, P&L) and over time builds a lightweight
classifier that identifies which signal patterns tend to win or lose.

If a pre-trained XGBoost model (model.xgb) is available, it is used for
scoring instead of the simple pattern lookup. Falls back to pattern memory
if the model is unavailable.
"""
from __future__ import annotations

import os
import statistics
from datetime import datetime, timezone
from typing import Any

from src.data.repository import Repository
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Minimum closed trades before the classifier starts producing scores
MIN_TRADES_FOR_LEARNING = 10

# Known signal reasons for XGBoost feature extraction (must match train_model.py)
KNOWN_REASONS = [
    "ob_bullish", "ob_bearish", "fvg_bullish", "fvg_bearish",
    "bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish",
    "displacement_bullish", "displacement_bearish",
    "premium_zone", "discount_zone",
    "liquidity_sweep_high", "liquidity_sweep_low",
    "rvol_elevated", "rvol_very_high",
    "volume_increasing", "sentiment_positive", "sentiment_negative",
]


class TradeAnalyzer:
    """Analyzes closed trades, logs structured context, builds pattern memory.

    Uses XGBoost model if available, otherwise falls back to pattern lookup.
    """

    def __init__(self, repo: Repository) -> None:
        self.repo = repo
        # In-memory pattern cache: maps pattern keys to outcome lists
        self._pattern_outcomes: dict[str, list[float]] = {}
        self._total_analyzed: int = 0

        # XGBoost model (loaded lazily)
        self._xgb_model = None
        self._xgb_feature_names: list[str] | None = None
        self._load_xgb_model()

    def _load_xgb_model(self) -> None:
        """Try to load a pre-trained XGBoost model from disk."""
        model_path = os.path.join(os.path.dirname(__file__), "..", "..", "model.xgb")
        features_path = os.path.join(os.path.dirname(__file__), "..", "..", "model_features.txt")

        if not os.path.exists(model_path):
            logger.info("xgb_model_not_found", path=model_path)
            return

        try:
            import xgboost as xgb
            self._xgb_model = xgb.Booster()
            self._xgb_model.load_model(model_path)

            # Load feature names
            if os.path.exists(features_path):
                with open(features_path) as f:
                    self._xgb_feature_names = [line.strip() for line in f if line.strip()]
            else:
                # Default feature names matching train_model.py
                self._xgb_feature_names = (
                    ["confluence_score", "direction", "hour_of_day"]
                    + [f"reason_{r}" for r in KNOWN_REASONS]
                )

            logger.info("xgb_model_loaded", features=len(self._xgb_feature_names))
        except Exception as e:
            logger.warning("xgb_model_load_failed", error=str(e))
            self._xgb_model = None

    async def analyze_closed_trade(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl_usd: float,
        pnl_percent: float,
        exit_reason: str,
        confluence_score: float,
        holding_seconds: float,
        signal_reasons: list[str] | None = None,
    ) -> dict[str, Any]:
        """Log structured outcome for a closed trade and update pattern memory.

        Returns the analysis dict (also persisted to DB via trade_analyses).
        """
        is_win = pnl_usd > 0
        holding_hours = holding_seconds / 3600

        analysis = {
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_usd": round(pnl_usd, 4),
            "pnl_percent": round(pnl_percent, 4),
            "exit_reason": exit_reason,
            "confluence_score": confluence_score,
            "holding_hours": round(holding_hours, 2),
            "is_win": is_win,
            "signal_reasons": signal_reasons or [],
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Update pattern memory
        self._update_patterns(analysis)

        # Persist analysis to DB (stored in the trade record's details)
        try:
            await self.repo.update_trade(trade_id, {"analysis": analysis})
        except Exception as e:
            logger.warning("trade_analysis_persist_failed", trade_id=trade_id, error=str(e))

        logger.info(
            "trade_analyzed",
            symbol=symbol,
            direction=direction,
            pnl_usd=round(pnl_usd, 2),
            exit_reason=exit_reason,
            confluence_score=confluence_score,
            holding_hours=round(holding_hours, 1),
            is_win=is_win,
        )

        self._total_analyzed += 1
        return analysis

    def _update_patterns(self, analysis: dict) -> None:
        """Extract pattern keys from trade context and record outcomes."""
        pnl_pct = analysis["pnl_percent"]

        # Pattern 1: exit_reason (e.g., "sl_hit", "tp_hit", "trailing_stop")
        self._record_pattern(f"exit:{analysis['exit_reason']}", pnl_pct)

        # Pattern 2: direction
        self._record_pattern(f"dir:{analysis['direction']}", pnl_pct)

        # Pattern 3: confluence score bucket (60-70, 70-80, 80-90, 90+)
        score = analysis["confluence_score"]
        bucket = int(score // 10) * 10
        self._record_pattern(f"score:{bucket}-{bucket+10}", pnl_pct)

        # Pattern 4: holding time bucket
        hours = analysis["holding_hours"]
        if hours < 1:
            time_bucket = "<1h"
        elif hours < 4:
            time_bucket = "1-4h"
        elif hours < 24:
            time_bucket = "4-24h"
        else:
            time_bucket = ">24h"
        self._record_pattern(f"hold:{time_bucket}", pnl_pct)

        # Pattern 5: each signal reason
        for reason in analysis.get("signal_reasons", []):
            # Normalize: take first 40 chars, lowercase
            key = reason[:40].lower().strip()
            if key:
                self._record_pattern(f"reason:{key}", pnl_pct)

    def _record_pattern(self, key: str, pnl_pct: float) -> None:
        if key not in self._pattern_outcomes:
            self._pattern_outcomes[key] = []
        self._pattern_outcomes[key].append(pnl_pct)

    def get_pattern_score(
        self, signal_reasons: list[str], direction: str, confluence_score: float
    ) -> float | None:
        """Score a potential trade using XGBoost or pattern fallback.

        Returns a modifier between -1.0 and +1.0 (negative = patterns suggest
        this is a losing setup, positive = winning). Returns None if not
        enough data.
        """
        # Try XGBoost first
        if self._xgb_model is not None and self._xgb_feature_names is not None:
            try:
                return self._xgb_score(signal_reasons, direction, confluence_score)
            except Exception as e:
                logger.debug("xgb_score_failed_using_fallback", error=str(e))

        # Fallback to pattern lookup
        return self._pattern_score(signal_reasons, direction, confluence_score)

    def _xgb_score(
        self, signal_reasons: list[str], direction: str, confluence_score: float
    ) -> float:
        """Score using pre-trained XGBoost model. Returns modifier in [-1.0, +1.0]."""
        import xgboost as xgb
        import numpy as np

        # Build feature vector matching the training schema
        dir_val = 1.0 if direction in ("long", "bullish") else 0.0
        hour = float(datetime.now(timezone.utc).hour)

        features: dict[str, float] = {
            "confluence_score": confluence_score,
            "direction": dir_val,
            "hour_of_day": hour,
        }

        reasons_lower = [r[:40].lower().strip() for r in signal_reasons]
        for reason_key in KNOWN_REASONS:
            features[f"reason_{reason_key}"] = 1.0 if any(reason_key in r for r in reasons_lower) else 0.0

        # Build array in feature name order
        row = [features.get(f, 0.0) for f in self._xgb_feature_names]
        dmatrix = xgb.DMatrix(np.array([row], dtype=np.float32), feature_names=self._xgb_feature_names)

        # Predict win probability
        prob = float(self._xgb_model.predict(dmatrix)[0])

        # Map probability [0, 1] to modifier [-1, +1]
        # 0.5 probability = neutral (0.0), 1.0 = +1.0, 0.0 = -1.0
        modifier = (prob - 0.5) * 2.0
        return max(-1.0, min(1.0, modifier))

    def _pattern_score(
        self, signal_reasons: list[str], direction: str, confluence_score: float
    ) -> float | None:
        """Fallback pattern-based scoring."""
        if self._total_analyzed < MIN_TRADES_FOR_LEARNING:
            return None

        scores: list[float] = []

        # Check direction pattern
        dir_key = f"dir:{direction}"
        dir_score = self._pattern_win_rate(dir_key)
        if dir_score is not None:
            scores.append(dir_score)

        # Check score bucket
        bucket = int(confluence_score // 10) * 10
        score_key = f"score:{bucket}-{bucket+10}"
        bucket_score = self._pattern_win_rate(score_key)
        if bucket_score is not None:
            scores.append(bucket_score)

        # Check signal reasons
        for reason in signal_reasons:
            key = f"reason:{reason[:40].lower().strip()}"
            reason_score = self._pattern_win_rate(key)
            if reason_score is not None:
                scores.append(reason_score)

        if not scores:
            return None

        # Average of all pattern signals, mapped to -1..+1
        avg_win_rate = statistics.mean(scores)
        # Center around 0.5 (50% win rate = neutral)
        return (avg_win_rate - 0.5) * 2

    def _pattern_win_rate(self, key: str) -> float | None:
        """Get win rate for a pattern. Returns None if <3 samples."""
        outcomes = self._pattern_outcomes.get(key, [])
        if len(outcomes) < 3:
            return None
        wins = sum(1 for o in outcomes if o > 0)
        return wins / len(outcomes)

    def get_pattern_report(self) -> dict[str, Any]:
        """Generate a summary report of all pattern performances."""
        report: dict[str, Any] = {
            "total_analyzed": self._total_analyzed,
            "xgb_model_loaded": self._xgb_model is not None,
            "patterns": {},
        }

        for key, outcomes in sorted(self._pattern_outcomes.items()):
            if len(outcomes) < 2:
                continue
            wins = sum(1 for o in outcomes if o > 0)
            report["patterns"][key] = {
                "count": len(outcomes),
                "win_rate": round(wins / len(outcomes), 3),
                "avg_pnl_pct": round(statistics.mean(outcomes), 3),
                "median_pnl_pct": round(statistics.median(outcomes), 3),
            }

        return report

    async def load_history(self) -> None:
        """Load historical closed trades to rebuild pattern memory on startup."""
        try:
            trades = await self.repo.get_trades(status="closed", per_page=500)
            for t in trades:
                analysis = t.get("analysis")
                if analysis and isinstance(analysis, dict):
                    self._update_patterns(analysis)
                    self._total_analyzed += 1
                elif t.get("status") == "closed":
                    # Build a minimal analysis from trade data
                    pnl_pct = float(t.get("pnl_percent", 0))
                    minimal = {
                        "exit_reason": t.get("exit_reason", "unknown"),
                        "direction": t.get("direction", "long"),
                        "confluence_score": float(t.get("confluence_score", 0)),
                        "holding_hours": 0,
                        "pnl_percent": pnl_pct,
                        "signal_reasons": t.get("signal_reasons") or [],
                    }
                    # Compute holding hours if we have times
                    entry_time = t.get("entry_time")
                    exit_time = t.get("exit_time")
                    if entry_time and exit_time:
                        try:
                            et = datetime.fromisoformat(entry_time)
                            xt = datetime.fromisoformat(exit_time)
                            minimal["holding_hours"] = (xt - et).total_seconds() / 3600
                        except (ValueError, TypeError):
                            pass
                    self._update_patterns(minimal)
                    self._total_analyzed += 1

            logger.info("trade_analyzer_loaded", trades_loaded=self._total_analyzed,
                        patterns=len(self._pattern_outcomes),
                        xgb_model=self._xgb_model is not None)
        except Exception as e:
            logger.warning("trade_analyzer_load_failed", error=str(e))
