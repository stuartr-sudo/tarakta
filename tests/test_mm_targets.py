"""Tests for MM Method target identification — VectorScanner recovery_pct.

B7: Vector 50% recovery rule (lesson 13).
  "If price has recovered >50% of a vector candle's body → expect full recovery."
  A >50% recovered vector gets a priority boost when assigned to target levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.mm_targets import VectorCandle, VectorScanner, TargetAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vector_ohlcv(
    n: int = 30,
    vector_pos: int = 10,
    body_low: float = 200.0,
    body_high: float = 210.0,
    recovery_into_body: float = 0.0,
    direction: str = "bullish",
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with a single vector candle.

    The baseline candles trade well ABOVE the vector body zone so they do not
    accidentally trigger the "recovered" check (which looks for candles that
    consolidate through the body zone). After the vector candle, price is set
    above body_high so the body remains unrecovered unless recovery_into_body > 0.

    Args:
        n: Total candle count.
        vector_pos: Index where the vector candle sits.
        body_low / body_high: Body bounds of the vector candle.
        recovery_into_body: How far INTO the body subsequent candles penetrate
            as a fraction of body_size (0=none, 0.6=60%).
        direction: "bullish" (close > open) or "bearish".
    """
    body_size = body_high - body_low

    # Baseline price far ABOVE the body_high so candles don't accidentally
    # fall inside [body_low, body_high] and mark the vector as "recovered".
    base_above = body_high + 20.0  # well above the body

    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")

    opens = np.full(n, base_above, dtype=float)
    closes = np.full(n, base_above, dtype=float)
    highs = np.full(n, base_above + 1.0, dtype=float)
    lows = np.full(n, base_above - 1.0, dtype=float)
    vols = np.ones(n) * 100.0  # baseline volume

    # Vector candle — volume 3x baseline (vector_200)
    if direction == "bullish":
        opens[vector_pos] = body_low
        closes[vector_pos] = body_high
    else:
        opens[vector_pos] = body_high
        closes[vector_pos] = body_low
    highs[vector_pos] = body_high + 1.0
    lows[vector_pos] = body_low - 1.0
    vols[vector_pos] = 300.0  # 3x average (well above VECTOR_200_THRESHOLD=2.0)

    # After vector candle, ensure no accidental body recovery in the default case.
    # All post-vector candles stay at base_above (above body_high).

    # Recovery candles — explicitly penetrate into the body.
    # The FIRST recovery candle touches the body but must NOT satisfy the full
    # body-consolidation condition (so the vector stays unrecovered overall).
    if recovery_into_body > 0 and vector_pos + 1 < n:
        penetration = body_size * recovery_into_body
        if direction == "bullish":
            # Price dips DOWN into the body from above: wick dips in, body stays above
            recovery_low = body_high - penetration
            opens[vector_pos + 1] = base_above  # body above the zone
            closes[vector_pos + 1] = base_above
            highs[vector_pos + 1] = base_above + 1.0
            lows[vector_pos + 1] = recovery_low  # wick dips into body
            # Ensure it doesn't satisfy "min(o,c) <= body_high AND max(o,c) >= body_low"
            # (that would mark it recovered). Body is base_above > body_high so it's fine.
        else:
            # Price rises UP into the body from below: wick pokes up, body stays below
            recovery_high = body_low + penetration
            opens[vector_pos + 1] = base_above - 40.0  # below body_low
            closes[vector_pos + 1] = base_above - 40.0
            lows[vector_pos + 1] = base_above - 41.0
            highs[vector_pos + 1] = recovery_high  # wick touches body

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


# ---------------------------------------------------------------------------
# VectorCandle dataclass
# ---------------------------------------------------------------------------


def test_vector_candle_has_recovery_pct_field():
    """VectorCandle dataclass must have a recovery_pct field defaulting to 0.0."""
    vc = VectorCandle()
    assert hasattr(vc, "recovery_pct"), "VectorCandle missing recovery_pct field"
    assert vc.recovery_pct == 0.0


def test_vector_candle_recovery_pct_settable():
    """recovery_pct must be settable."""
    vc = VectorCandle(recovery_pct=0.75)
    assert vc.recovery_pct == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# B7 — VectorScanner.scan() recovery_pct calculation
# ---------------------------------------------------------------------------


class TestVectorScannerRecovery:
    """B7 (lesson 13): VectorScanner must calculate recovery_pct for each
    unrecovered vector candle.
    """

    def test_no_recovery_gives_zero_pct(self):
        """No subsequent candles penetrate the body → recovery_pct == 0.0."""
        scanner = VectorScanner()
        # vector at pos 10, no recovery candles (recovery_into_body=0)
        ohlc = _make_vector_ohlcv(n=30, vector_pos=10, recovery_into_body=0.0)
        current_price = 230.0  # above body_high=210, baseline is at 230
        vectors = scanner.scan(ohlc, current_price)

        # Should find at least one vector
        assert len(vectors) >= 1, "Expected at least one vector candle"
        # The one we built should have recovery_pct == 0
        our_vec = min(vectors, key=lambda v: abs(v.midpoint - 105.0))
        assert our_vec.recovery_pct == pytest.approx(0.0, abs=0.05)

    def test_60_pct_recovery_recorded(self):
        """60% body recovery → recovery_pct ≈ 0.6."""
        scanner = VectorScanner()
        ohlc = _make_vector_ohlcv(
            n=30, vector_pos=10,
            body_low=100.0, body_high=110.0,
            recovery_into_body=0.6,  # 60% of 10-unit body = 6 units
            direction="bullish",
        )
        current_price = 105.0
        vectors = scanner.scan(ohlc, current_price)
        assert len(vectors) >= 1

        # Find the vector with midpoint near 205 (body 200-210)
        our_vec = min(vectors, key=lambda v: abs(v.midpoint - 205.0))
        assert our_vec.recovery_pct > 0.5, (
            f"Expected recovery_pct > 0.5 for 60% recovery, got {our_vec.recovery_pct}"
        )

    def test_20_pct_recovery_below_threshold(self):
        """20% body recovery → recovery_pct < 0.5 (no priority boost)."""
        scanner = VectorScanner()
        ohlc = _make_vector_ohlcv(
            n=30, vector_pos=10,
            body_low=100.0, body_high=110.0,
            recovery_into_body=0.2,
            direction="bullish",
        )
        current_price = 105.0
        vectors = scanner.scan(ohlc, current_price)
        assert len(vectors) >= 1

        our_vec = min(vectors, key=lambda v: abs(v.midpoint - 105.0))
        assert our_vec.recovery_pct < 0.5, (
            f"Expected recovery_pct < 0.5 for 20% recovery, got {our_vec.recovery_pct}"
        )

    def test_recovery_pct_capped_at_one(self):
        """recovery_pct must never exceed 1.0."""
        scanner = VectorScanner()
        # Over-recovery: 120% — should be capped at 1.0
        ohlc = _make_vector_ohlcv(
            n=30, vector_pos=10,
            body_low=100.0, body_high=110.0,
            recovery_into_body=1.2,
            direction="bullish",
        )
        current_price = 105.0
        vectors = scanner.scan(ohlc, current_price)
        for v in vectors:
            assert v.recovery_pct <= 1.0, f"recovery_pct {v.recovery_pct} exceeds 1.0"


# ---------------------------------------------------------------------------
# B7 — TargetAnalyzer priority boost for >50% recovered vectors
# ---------------------------------------------------------------------------


class TestVectorRecoveryPriorityBoost:
    """B7: Vectors with recovery_pct > 0.5 must get a lower priority number
    (= higher priority) when assigned to target levels.
    """

    def _make_vector_with_recovery(self, recovery_pct: float) -> VectorCandle:
        """Create a VectorCandle with the given recovery_pct."""
        return VectorCandle(
            index=5,
            timestamp="2025-01-01T00:00:00",
            high=112.0,
            low=98.0,
            close=110.0,
            volume_ratio=2.5,
            vector_type="vector_200",
            direction="bullish",
            midpoint=105.0,
            recovered=False,
            recovery_pct=recovery_pct,
        )

    def test_60_pct_recovery_gets_lower_priority_number(self):
        """A vector with 60% recovery should have priority 1 (boosted from base 2)."""
        analyzer = TargetAnalyzer()
        v_high = self._make_vector_with_recovery(0.60)
        v_low = self._make_vector_with_recovery(0.20)

        # Use _targets_for_level directly with a simple entry above the vector midpoint
        # L1 range: dist <= 3% → entry ~102 puts midpoint=105 at ~2.9% away
        targets_high = analyzer._targets_for_level(
            level=1,
            direction="bullish",
            entry_price=102.0,
            ema_values=None,
            how=None,
            low=None,
            liquidation_levels=None,
            vectors=[v_high],
        )
        targets_low = analyzer._targets_for_level(
            level=1,
            direction="bullish",
            entry_price=102.0,
            ema_values=None,
            how=None,
            low=None,
            liquidation_levels=None,
            vectors=[v_low],
        )

        assert len(targets_high) >= 1, "Expected at least one target for high-recovery vector"
        assert len(targets_low) >= 1, "Expected at least one target for low-recovery vector"

        high_recovery_priority = targets_high[0].priority
        low_recovery_priority = targets_low[0].priority

        assert high_recovery_priority < low_recovery_priority, (
            f">50% recovered vector should have lower priority number (higher priority): "
            f"got {high_recovery_priority} vs {low_recovery_priority}"
        )

    def test_20_pct_recovery_uses_base_priority(self):
        """A vector with 20% recovery uses base priority (2 for L1 vectors)."""
        analyzer = TargetAnalyzer()
        v = self._make_vector_with_recovery(0.20)

        targets = analyzer._targets_for_level(
            level=1,
            direction="bullish",
            entry_price=102.0,
            ema_values=None,
            how=None,
            low=None,
            liquidation_levels=None,
            vectors=[v],
        )

        assert len(targets) >= 1
        # Base priority for L1 vector is 2; no boost expected
        assert targets[0].priority == 2, (
            f"Expected base priority=2 for <50% recovery, got {targets[0].priority}"
        )

    def test_60_pct_recovery_description_mentions_boost(self):
        """Description of a >50% recovered vector target must note the boost."""
        analyzer = TargetAnalyzer()
        v = self._make_vector_with_recovery(0.60)

        targets = analyzer._targets_for_level(
            level=1,
            direction="bullish",
            entry_price=102.0,
            ema_values=None,
            how=None,
            low=None,
            liquidation_levels=None,
            vectors=[v],
        )

        assert len(targets) >= 1
        assert "boost" in targets[0].description.lower() or "recover" in targets[0].description.lower(), (
            f"Expected boost/recovery mention in description: {targets[0].description}"
        )
