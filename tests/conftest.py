from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def bullish_candles() -> pd.DataFrame:
    """Generate a bullish trending OHLCV DataFrame (200 candles)."""
    np.random.seed(42)
    n = 200
    base = 100.0
    prices = [base]
    for i in range(1, n):
        # Upward drift with noise
        change = np.random.normal(0.3, 1.5)
        prices.append(max(prices[-1] + change, 1.0))

    data = {
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
    }

    for i, price in enumerate(prices):
        noise = abs(np.random.normal(0, 0.5))
        o = price - noise * 0.5
        c = price + noise * 0.5
        h = max(o, c) + abs(np.random.normal(0, 0.3))
        l = min(o, c) - abs(np.random.normal(0, 0.3))
        data["open"].append(o)
        data["high"].append(h)
        data["low"].append(max(l, 0.01))
        data["close"].append(c)
        data["volume"].append(np.random.uniform(1000, 10000))

    timestamps = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(data, index=timestamps)


@pytest.fixture
def bearish_candles() -> pd.DataFrame:
    """Generate a bearish trending OHLCV DataFrame (200 candles)."""
    np.random.seed(42)
    n = 200
    base = 200.0
    prices = [base]
    for i in range(1, n):
        change = np.random.normal(-0.3, 1.5)
        prices.append(max(prices[-1] + change, 1.0))

    data = {"open": [], "high": [], "low": [], "close": [], "volume": []}
    for price in prices:
        noise = abs(np.random.normal(0, 0.5))
        o = price + noise * 0.5
        c = price - noise * 0.5
        h = max(o, c) + abs(np.random.normal(0, 0.3))
        l = min(o, c) - abs(np.random.normal(0, 0.3))
        data["open"].append(o)
        data["high"].append(h)
        data["low"].append(max(l, 0.01))
        data["close"].append(max(c, 0.01))
        data["volume"].append(np.random.uniform(1000, 10000))

    timestamps = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(data, index=timestamps)


@pytest.fixture
def ranging_candles() -> pd.DataFrame:
    """Generate a ranging/sideways OHLCV DataFrame (200 candles)."""
    np.random.seed(42)
    n = 200
    base = 150.0

    data = {"open": [], "high": [], "low": [], "close": [], "volume": []}
    for i in range(n):
        price = base + np.sin(i / 10) * 5 + np.random.normal(0, 1)
        noise = abs(np.random.normal(0, 0.5))
        o = price - noise * 0.3
        c = price + noise * 0.3
        h = max(o, c) + abs(np.random.normal(0, 0.3))
        l = min(o, c) - abs(np.random.normal(0, 0.3))
        data["open"].append(o)
        data["high"].append(h)
        data["low"].append(max(l, 0.01))
        data["close"].append(c)
        data["volume"].append(np.random.uniform(1000, 10000))

    timestamps = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(data, index=timestamps)
