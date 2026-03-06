"""Offline XGBoost training script.

Queries closed trades from Supabase, extracts features, trains an
XGBClassifier to predict win/loss, and saves the model to model.xgb.

Usage:
    python train_model.py
"""
from __future__ import annotations

import asyncio
import os
import sys

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.config import Settings
from src.data.db import Database
from src.data.repository import Repository


# Common signal reasons to one-hot encode
KNOWN_REASONS = [
    "ob_bullish", "ob_bearish", "fvg_bullish", "fvg_bearish",
    "bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish",
    "displacement_bullish", "displacement_bearish",
    "premium_zone", "discount_zone",
    "liquidity_sweep_high", "liquidity_sweep_low",
    "rvol_elevated", "rvol_very_high",
    "volume_increasing", "sentiment_positive", "sentiment_negative",
]


def extract_features(trade: dict) -> dict[str, float] | None:
    """Extract feature dict from a closed trade record."""
    confluence_score = float(trade.get("confluence_score", 0))
    direction = trade.get("direction", "long")
    entry_time = trade.get("entry_time", "")
    signal_reasons = trade.get("signal_reasons") or []
    pnl_usd = float(trade.get("pnl_usd", 0))

    if confluence_score <= 0:
        return None

    # Direction: 1 for long, 0 for short
    dir_val = 1.0 if direction == "long" else 0.0

    # Time of day (hour) from entry_time
    hour = 12.0  # default
    if entry_time:
        try:
            from datetime import datetime
            if isinstance(entry_time, str):
                dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            else:
                dt = entry_time
            hour = float(dt.hour)
        except (ValueError, TypeError, AttributeError):
            pass

    features: dict[str, float] = {
        "confluence_score": confluence_score,
        "direction": dir_val,
        "hour_of_day": hour,
    }

    # One-hot encode signal reasons
    reasons_lower = [r[:40].lower().strip() for r in signal_reasons]
    for reason_key in KNOWN_REASONS:
        features[f"reason_{reason_key}"] = 1.0 if any(reason_key in r for r in reasons_lower) else 0.0

    # Target
    features["is_win"] = 1.0 if pnl_usd > 0 else 0.0

    return features


async def load_trades() -> list[dict]:
    """Load closed trades from Supabase."""
    config = Settings()
    db = Database(config.supabase_url, config.supabase_key)
    repo = Repository(db)
    trades = await repo.get_trades(status="closed", per_page=1000)
    return trades


def main():
    print("Loading closed trades from Supabase...")
    trades = asyncio.run(load_trades())
    print(f"Found {len(trades)} closed trades")

    if len(trades) < 20:
        print("Not enough trades to train a meaningful model (need at least 20). Exiting.")
        sys.exit(1)

    # Extract features
    rows = []
    for t in trades:
        feat = extract_features(t)
        if feat is not None:
            rows.append(feat)

    if len(rows) < 20:
        print(f"Only {len(rows)} valid feature rows. Need at least 20. Exiting.")
        sys.exit(1)

    print(f"Extracted features from {len(rows)} trades")

    # Build arrays
    feature_names = [k for k in rows[0].keys() if k != "is_win"]
    X = np.array([[row[f] for f in feature_names] for row in rows], dtype=np.float32)
    y = np.array([row["is_win"] for row in rows], dtype=np.float32)

    wins = int(y.sum())
    losses = len(y) - wins
    print(f"Win/Loss split: {wins}W / {losses}L ({wins / len(y) * 100:.1f}% win rate)")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if wins >= 2 and losses >= 2 else None
    )

    # Train XGBClassifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Loss", "Win"]))

    # Feature importance
    print("\nTop Features by Importance:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "model.xgb")
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    # Also save feature names for inference
    names_path = os.path.join(os.path.dirname(__file__), "model_features.txt")
    with open(names_path, "w") as f:
        f.write("\n".join(feature_names))
    print(f"Feature names saved to {names_path}")


if __name__ == "__main__":
    main()
