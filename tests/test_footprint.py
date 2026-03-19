"""Tests for the FootprintAnalyzer order flow confirmation gate."""
import pytest

from src.strategy.footprint import FootprintAnalyzer
from src.exchange.models import FootprintResult


@pytest.fixture
def analyzer():
    return FootprintAnalyzer(
        min_delta_pct=0.10,
        absorption_threshold=0.30,
        min_confidence=0.40,
    )


def _make_trades(buy_pct: float, n: int = 1000, base_price: float = 100.0, sweep_level: float = 99.0) -> list[dict]:
    """Generate mock trades with a given buy/sell ratio.

    buy_pct: 0.0 = all sells, 1.0 = all buys, 0.5 = balanced.
    """
    trades = []
    n_buy = int(n * buy_pct)
    for i in range(n):
        is_buy = i < n_buy
        price = base_price + (i % 10) * 0.01
        trades.append({
            "price": price,
            "amount": 1.0,
            "cost": price * 1.0,
            "side": "buy" if is_buy else "sell",
            "timestamp": 1000000 + i,
        })
    return trades


def _make_absorption_trades(sweep_level: float, is_bullish: bool, n: int = 1000) -> list[dict]:
    """Generate trades with high volume concentrated at the sweep level."""
    trades = []
    for i in range(n):
        if i < 500:
            # Heavy volume at sweep level
            price = sweep_level + (i % 3 - 1) * 0.001
            # If bullish sweep: lots of buying at the low (absorption of selling)
            side = "buy" if is_bullish else "sell"
        else:
            # Price moves away from sweep
            offset = (i - 500) * 0.01
            price = (sweep_level + offset) if is_bullish else (sweep_level - offset)
            side = "buy" if is_bullish else "sell"
        trades.append({
            "price": price,
            "amount": 1.0,
            "cost": price * 1.0,
            "side": side,
            "timestamp": 1000000 + i,
        })
    return trades


def _make_accelerating_trades(is_bullish: bool, n: int = 1000) -> list[dict]:
    """Generate trades where delta accelerates in the reversal direction.

    Overall net delta favours the reversal (60/40 split), and the second
    half is more skewed than the first (acceleration).
    """
    trades = []
    mid = n // 2
    for i in range(n):
        if i < mid:
            # First half: slightly against reversal (40% reversal, 60% against)
            side = "sell" if is_bullish else "buy"
            if i % 5 < 2:  # 40% reversal direction
                side = "buy" if is_bullish else "sell"
        else:
            # Second half: heavily with reversal (90% reversal, 10% against)
            side = "buy" if is_bullish else "sell"
            if i % 10 == 0:
                side = "sell" if is_bullish else "buy"  # 10% against
        trades.append({
            "price": 100.0,
            "amount": 1.0,
            "cost": 100.0,
            "side": side,
            "timestamp": 1000000 + i,
        })
    return trades


class TestDeltaCalculation:
    def test_strong_buying_positive_delta(self, analyzer):
        trades = _make_trades(buy_pct=0.80)
        buy_vol, sell_vol = analyzer._aggregate_volume(trades)
        delta = buy_vol - sell_vol
        assert delta > 0
        assert buy_vol > sell_vol * 3

    def test_strong_selling_negative_delta(self, analyzer):
        trades = _make_trades(buy_pct=0.20)
        buy_vol, sell_vol = analyzer._aggregate_volume(trades)
        delta = buy_vol - sell_vol
        assert delta < 0
        assert sell_vol > buy_vol * 3

    def test_balanced_near_zero_delta(self, analyzer):
        trades = _make_trades(buy_pct=0.50)
        buy_vol, sell_vol = analyzer._aggregate_volume(trades)
        delta_pct = (buy_vol - sell_vol) / (buy_vol + sell_vol)
        assert abs(delta_pct) < 0.05


class TestAbsorption:
    def test_absorption_at_bullish_sweep(self, analyzer):
        trades = _make_absorption_trades(sweep_level=99.0, is_bullish=True)
        score = analyzer._calculate_absorption(
            trades, sweep_level=99.0, current_price=100.0, is_bullish=True,
        )
        assert score > 0.0

    def test_no_absorption_low_volume(self, analyzer):
        # Trades spread out, nothing concentrated at sweep
        trades = _make_trades(buy_pct=0.5, base_price=105.0)
        score = analyzer._calculate_absorption(
            trades, sweep_level=99.0, current_price=100.0, is_bullish=True,
        )
        assert score < 0.3

    def test_absorption_fails_when_price_continues(self, analyzer):
        # Price is still below sweep level = didn't reverse
        trades = _make_absorption_trades(sweep_level=100.0, is_bullish=True)
        score = analyzer._calculate_absorption(
            trades, sweep_level=100.0, current_price=99.5, is_bullish=True,
        )
        assert score <= 0.2


class TestCumulativeDelta:
    def test_accelerating_buying_confirms_bullish(self, analyzer):
        trades = _make_accelerating_trades(is_bullish=True)
        assert analyzer._evaluate_cumulative_delta(trades, is_bullish=True) is True

    def test_accelerating_selling_confirms_bearish(self, analyzer):
        trades = _make_accelerating_trades(is_bullish=False)
        assert analyzer._evaluate_cumulative_delta(trades, is_bullish=False) is True

    def test_decelerating_buying_fails_bullish(self, analyzer):
        # Reverse: buying decelerating (strong first half, weak second)
        trades = _make_accelerating_trades(is_bullish=True)
        trades.reverse()  # Now first half is strong buying, second is weak
        assert analyzer._evaluate_cumulative_delta(trades, is_bullish=True) is False

    def test_too_few_trades_returns_false(self, analyzer):
        trades = [{"price": 100, "amount": 1, "cost": 100, "side": "buy", "timestamp": 1}] * 5
        assert analyzer._evaluate_cumulative_delta(trades, is_bullish=True) is False


class TestPassFail:
    def test_strong_bullish_reversal_passes(self, analyzer):
        """Strong buying after a swing_low sweep should pass."""
        trades = _make_accelerating_trades(is_bullish=True, n=1000)
        result = analyzer._analyze_sync(trades, "swing_low", 99.0, 100.5)
        assert result.passed is True
        assert result.confidence >= 0.40
        assert result.delta > 0

    def test_continuation_selling_fails_bullish(self, analyzer):
        """Heavy selling after a swing_low sweep = continuation, not reversal."""
        trades = _make_trades(buy_pct=0.15)  # 85% selling
        result = analyzer._analyze_sync(trades, "swing_low", 99.0, 100.0)
        assert result.passed is False
        assert result.delta < 0

    def test_weak_signal_below_threshold_fails(self, analyzer):
        """Near-balanced trades shouldn't pass."""
        trades = _make_trades(buy_pct=0.52)  # barely bullish
        result = analyzer._analyze_sync(trades, "swing_low", 99.0, 100.0)
        assert result.passed is False

    def test_strong_bearish_reversal_passes(self, analyzer):
        """Strong selling after a swing_high sweep should pass."""
        trades = _make_accelerating_trades(is_bullish=False, n=1000)
        result = analyzer._analyze_sync(trades, "swing_high", 101.0, 99.5)
        assert result.passed is True
        assert result.delta < 0

    def test_neutral_on_empty_trades(self, analyzer):
        result = analyzer._analyze_sync([], "swing_low", 99.0, 100.0)
        assert result.passed is True  # fail open
        assert "NEUTRAL" in result.reasons[0]


# Add a sync helper for testing (avoids needing async)
FootprintAnalyzer._analyze_sync = lambda self, trades, sweep_dir, sweep_lvl, cur_price: self._analyze_sync_impl(trades, sweep_dir, sweep_lvl, cur_price)


def _analyze_sync_impl(self, trades, sweep_direction, sweep_level, current_price):
    """Synchronous version of analyze for unit tests (no exchange calls)."""
    is_bullish = "low" in sweep_direction

    if not trades:
        return self._neutral("no_trades_returned")

    buy_vol, sell_vol = self._aggregate_volume(trades)
    total_vol = buy_vol + sell_vol
    if total_vol == 0:
        return self._neutral("zero_volume")

    delta = buy_vol - sell_vol
    delta_pct = delta / total_vol
    absorption = self._calculate_absorption(trades, sweep_level, current_price, is_bullish)
    cum_delta_confirms = self._evaluate_cumulative_delta(trades, is_bullish)

    delta_direction_ok = (delta_pct > 0 and is_bullish) or (delta_pct < 0 and not is_bullish)
    delta_strong = abs(delta_pct) >= self.min_delta_pct

    if delta_direction_ok and delta_strong:
        delta_score = 1.0
    elif delta_direction_ok:
        delta_score = abs(delta_pct) / self.min_delta_pct
    else:
        delta_score = 0.0

    if absorption >= self.absorption_threshold:
        absorption_score_weight = min(1.0, absorption / 0.8)
    else:
        absorption_score_weight = absorption / self.absorption_threshold * 0.5

    cum_delta_score = 1.0 if cum_delta_confirms else 0.0
    oi_score = 1.0  # neutral in tests

    confidence = 0.35 * delta_score + 0.25 * absorption_score_weight + 0.25 * cum_delta_score + 0.15 * oi_score
    passed = confidence >= self.min_confidence and delta_direction_ok

    reasons = [f"{'PASS' if passed else 'FAIL'} (confidence={confidence:.2f})"]

    return FootprintResult(
        passed=passed,
        confidence=round(confidence, 4),
        delta=round(delta, 4),
        delta_pct=round(delta_pct, 6),
        absorption_score=round(absorption, 4),
        cumulative_delta_confirms=cum_delta_confirms,
        oi_change_pct=0.0,
        oi_confirms=True,
        total_volume=round(total_vol, 4),
        buy_volume=round(buy_vol, 4),
        sell_volume=round(sell_vol, 4),
        reasons=reasons,
    )


FootprintAnalyzer._analyze_sync_impl = _analyze_sync_impl
