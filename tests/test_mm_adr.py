"""Tests for the MM ADR indicator (src.strategy.mm_adr)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.mm_adr import ADRAnalyzer, ADRState, AT_FIFTY_PCT_TOLERANCE, DEFAULT_ADR_PERIOD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_1h_ohlc(
    daily_highs: list[float],
    daily_lows: list[float],
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Build a 1H OHLCV DataFrame with known daily ranges.

    Each element in daily_highs/daily_lows corresponds to one day.
    Within each day all 24 hourly candles share the same high and low so
    that the daily resample produces exactly the values we specified.
    """
    assert len(daily_highs) == len(daily_lows)
    records = []
    base = pd.Timestamp(start, tz="UTC")
    for day_idx, (h, l) in enumerate(zip(daily_highs, daily_lows)):
        mid = (h + l) / 2.0
        for hour in range(24):
            ts = base + pd.Timedelta(days=day_idx, hours=hour)
            records.append({
                "timestamp": ts,
                "open": mid,
                "high": h,
                "low": l,
                "close": mid,
                "volume": 1000.0,
            })
    df = pd.DataFrame(records)
    df = df.set_index("timestamp")
    return df


def _uniform_daily_ranges(daily_range: float, n_days: int, base_price: float = 100.0) -> tuple[list[float], list[float]]:
    """Build symmetric high/low lists for n_days with a fixed daily range."""
    highs = [base_price + daily_range / 2.0] * n_days
    lows = [base_price - daily_range / 2.0] * n_days
    return highs, lows


# ---------------------------------------------------------------------------
# ADR calculation
# ---------------------------------------------------------------------------

class TestADRCalculation:
    """Verify ADR values match known daily ranges."""

    def test_uniform_daily_ranges(self):
        """14 identical daily ranges → ADR = that range exactly."""
        # 15 days so we have 14 complete + 1 partial current day
        highs, lows = _uniform_daily_ranges(daily_range=1000.0, n_days=15, base_price=50000.0)
        df = _make_1h_ohlc(highs, lows)
        current_price = 50000.0

        state = ADRAnalyzer(period=14).calculate(df, current_price)

        assert state is not None
        assert state.adr_value == pytest.approx(1000.0, abs=0.01)

    def test_varying_daily_ranges(self):
        """ADR = mean of last 14 complete daily ranges."""
        # Build 15 days: ranges 100, 200, 100, 200, ... alternating
        # Mean of 14 alternating = 150.0
        highs = []
        lows = []
        base = 50000.0
        for i in range(15):
            rng = 100.0 if i % 2 == 0 else 200.0
            highs.append(base + rng / 2.0)
            lows.append(base - rng / 2.0)

        df = _make_1h_ohlc(highs, lows)
        state = ADRAnalyzer(period=14).calculate(df, base)

        assert state is not None
        # Days used are index 0..13 (14 complete days), day 14 is partial
        # Ranges alternate 100, 200 → mean = 150
        assert state.adr_value == pytest.approx(150.0, abs=0.01)

    def test_adr_pct_is_ratio_of_current_price(self):
        """adr_pct = adr_value / current_price."""
        highs, lows = _uniform_daily_ranges(daily_range=2000.0, n_days=15, base_price=100000.0)
        df = _make_1h_ohlc(highs, lows)
        current_price = 100000.0

        state = ADRAnalyzer(period=14).calculate(df, current_price)

        assert state is not None
        expected_pct = state.adr_value / current_price
        assert state.adr_pct == pytest.approx(expected_pct, abs=1e-6)

    def test_period_14_default(self):
        assert ADRAnalyzer().period == DEFAULT_ADR_PERIOD


# ---------------------------------------------------------------------------
# 50% line calculation
# ---------------------------------------------------------------------------

class TestFiftyPctLine:
    """Verify fifty_pct_line = current day's low + 0.5 * ADR."""

    def test_fifty_pct_line_value(self):
        """50% line = current day low + 0.5 * ADR."""
        # 15 days: first 14 have range 1000 (ADR=1000), last (current) has low=49000
        highs = [50500.0] * 14 + [50000.0]  # last day high doesn't matter much
        lows = [49500.0] * 14 + [49000.0]   # last day low = 49000

        df = _make_1h_ohlc(highs, lows)
        current_price = 49500.0

        state = ADRAnalyzer(period=14).calculate(df, current_price)

        assert state is not None
        # ADR = 1000.0, current day low = 49000
        # 50% line = 49000 + 500 = 49500
        assert state.adr_value == pytest.approx(1000.0, abs=0.01)
        assert state.fifty_pct_line == pytest.approx(49500.0, abs=0.01)

    def test_fifty_pct_line_with_smaller_range(self):
        """50% line arithmetic with a small range."""
        # 15 days, range = 200, current day low = 100.0
        highs = [101.0] * 14 + [100.5]
        lows = [99.0] * 14 + [100.0]
        df = _make_1h_ohlc(highs, lows)
        current_price = 100.0

        state = ADRAnalyzer(period=14).calculate(df, current_price)

        assert state is not None
        # ADR = 2.0 (highs-lows = 101 - 99 = 2), current day low = 100
        # 50% line = 100 + 1.0 = 101.0
        assert state.fifty_pct_line == pytest.approx(101.0, abs=0.01)


# ---------------------------------------------------------------------------
# at_fifty_pct proximity check
# ---------------------------------------------------------------------------

class TestAtFiftyPct:
    """Verify at_fifty_pct is True iff price within 0.3% of 50% line."""

    def _make_state_with_line(self, fifty_pct_line: float, current_price: float) -> ADRState | None:
        """Build a real ADR state where we can predict the fifty_pct_line."""
        # Design: ADR=1000, current day low = fifty_pct_line - 500
        adr = 1000.0
        current_day_low = fifty_pct_line - adr * 0.5
        n_complete = 14
        highs = [current_day_low + 1000.0 + 500.0] * n_complete + [current_day_low + adr]
        lows = [current_day_low + 500.0] * n_complete + [current_day_low]
        df = _make_1h_ohlc(highs, lows)
        return ADRAnalyzer(period=14).calculate(df, current_price)

    def test_at_line_exact(self):
        """Price exactly at 50% line → at_fifty_pct=True."""
        fifty_pct = 50000.0
        state = self._make_state_with_line(fifty_pct, current_price=fifty_pct)
        assert state is not None
        assert state.fifty_pct_line == pytest.approx(fifty_pct, abs=1.0)
        assert state.at_fifty_pct is True

    def test_at_line_within_tolerance(self):
        """Price 0.2% away from 50% line → at_fifty_pct=True (tolerance=0.3%)."""
        fifty_pct = 50000.0
        # 0.2% above line
        current_price = fifty_pct * (1 + 0.002)
        state = self._make_state_with_line(fifty_pct, current_price=current_price)
        assert state is not None
        assert state.at_fifty_pct is True

    def test_outside_tolerance(self):
        """Price 0.5% away from 50% line → at_fifty_pct=False."""
        fifty_pct = 50000.0
        # 0.5% above line (> 0.3% tolerance)
        current_price = fifty_pct * (1 + 0.005)
        state = self._make_state_with_line(fifty_pct, current_price=current_price)
        assert state is not None
        assert state.at_fifty_pct is False

    def test_tolerance_boundary(self):
        """Price far enough outside 50% line to be strictly False.

        The tolerance is 0.3% of current_price. We use 0.4% to be clearly
        outside — avoids floating-point ambiguity at the exact boundary.
        """
        fifty_pct = 50000.0
        # 0.4% above line: abs(50200 - 50000) / 50200 ≈ 0.398% > 0.3%
        current_price = fifty_pct * (1 + 0.004)
        state = self._make_state_with_line(fifty_pct, current_price=current_price)
        assert state is not None
        assert state.at_fifty_pct is False

    def test_below_line_within_tolerance(self):
        """Price 0.2% below 50% line → at_fifty_pct=True."""
        fifty_pct = 50000.0
        current_price = fifty_pct * (1 - 0.002)
        state = self._make_state_with_line(fifty_pct, current_price=current_price)
        assert state is not None
        assert state.at_fifty_pct is True


# ---------------------------------------------------------------------------
# Insufficient data → None
# ---------------------------------------------------------------------------

class TestInsufficientData:
    def test_none_when_empty_df(self):
        df = pd.DataFrame()
        assert ADRAnalyzer().calculate(df, 50000.0) is None

    def test_none_when_none_df(self):
        assert ADRAnalyzer().calculate(None, 50000.0) is None

    def test_none_when_invalid_price(self):
        highs, lows = _uniform_daily_ranges(daily_range=1000.0, n_days=15)
        df = _make_1h_ohlc(highs, lows)
        assert ADRAnalyzer().calculate(df, 0.0) is None
        assert ADRAnalyzer().calculate(df, -1.0) is None

    def test_none_when_too_few_days(self):
        """Only 10 daily candles — need at least period+1=15."""
        highs, lows = _uniform_daily_ranges(daily_range=1000.0, n_days=10, base_price=50000.0)
        df = _make_1h_ohlc(highs, lows)
        state = ADRAnalyzer(period=14).calculate(df, 50000.0)
        assert state is None

    def test_none_when_exactly_period_days(self):
        """14 daily candles — need 15 (period+1). Should return None."""
        highs, lows = _uniform_daily_ranges(daily_range=1000.0, n_days=14, base_price=50000.0)
        df = _make_1h_ohlc(highs, lows)
        state = ADRAnalyzer(period=14).calculate(df, 50000.0)
        assert state is None

    def test_not_none_when_period_plus_one_days(self):
        """15 daily candles → period+1 → just enough."""
        highs, lows = _uniform_daily_ranges(daily_range=1000.0, n_days=15, base_price=50000.0)
        df = _make_1h_ohlc(highs, lows)
        state = ADRAnalyzer(period=14).calculate(df, 50000.0)
        assert state is not None


# ---------------------------------------------------------------------------
# Integration: calculate() returns correct ADRState fields
# ---------------------------------------------------------------------------

class TestCalculateIntegration:
    def test_state_fields_populated(self):
        highs, lows = _uniform_daily_ranges(daily_range=1000.0, n_days=20, base_price=50000.0)
        df = _make_1h_ohlc(highs, lows)
        state = ADRAnalyzer(period=14).calculate(df, 50000.0)

        assert isinstance(state, ADRState)
        assert isinstance(state.adr_value, float)
        assert isinstance(state.adr_pct, float)
        assert isinstance(state.fifty_pct_line, float)
        assert isinstance(state.at_fifty_pct, bool)

    def test_adr_value_positive(self):
        highs, lows = _uniform_daily_ranges(daily_range=500.0, n_days=20, base_price=50000.0)
        df = _make_1h_ohlc(highs, lows)
        state = ADRAnalyzer(period=14).calculate(df, 50000.0)
        assert state is not None
        assert state.adr_value > 0

    def test_extra_days_still_uses_period(self):
        """30 days of data → ADR still uses last 14 complete days.

        We have 30 days total. The last row is the partial "current" day.
        complete_days = days[-(14+1):-1] = days[14..28] (15 elements → last 14
        complete + 1 being sliced off as current).

        Layout: 15 days range=2000, 15 days range=1000.
        The 15 complete days selected (indices 14..28) are all from the narrow
        block (range=1000) since indices 15..29, excluding index 29 (current).
        → ADR = 1000.
        """
        # 30 days: first 15 wide (range=2000), last 15 narrow (range=1000)
        highs = [50000.0 + 1000.0] * 15 + [50000.0 + 500.0] * 15
        lows = [50000.0 - 1000.0] * 15 + [50000.0 - 500.0] * 15
        df = _make_1h_ohlc(highs, lows)

        state = ADRAnalyzer(period=14).calculate(df, 50000.0)
        assert state is not None
        # complete_days = days[14..28] inclusive — all from narrow block
        assert state.adr_value == pytest.approx(1000.0, abs=1.0)

    def test_timestamp_column_works(self):
        """DataFrame with 'timestamp' column (not DatetimeIndex) is handled."""
        highs, lows = _uniform_daily_ranges(daily_range=1000.0, n_days=15, base_price=50000.0)
        df = _make_1h_ohlc(highs, lows)
        # Reset index to make timestamp a column (unix ms).
        # The index has dtype datetime64[us, UTC], so astype(int64) gives
        # microseconds. Divide by 1000 to get milliseconds.
        df = df.reset_index()
        df["timestamp"] = df["timestamp"].astype(np.int64) // 10 ** 3  # us → ms
        state = ADRAnalyzer(period=14).calculate(df, 50000.0)
        assert state is not None
        assert state.adr_value == pytest.approx(1000.0, abs=1.0)
