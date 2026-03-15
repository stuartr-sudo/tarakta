import pytest

from src.exchange.models import LeverageProfile
from src.strategy.leverage import LeverageAnalyzer
from src.strategy.scanner import AltcoinScanner


@pytest.fixture
def analyzer():
    return LeverageAnalyzer()


class TestCrowdingDetection:
    def test_positive_funding_detects_long_crowding(self, analyzer):
        """Strong positive funding = longs are crowded."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=0.001,  # 0.1% — very positive
            long_short_ratio=1.5,
            sweep_direction=None,
            in_kill_zone=False,
            in_post_kill_zone=False,
        )
        assert profile.crowded_side == "long"
        assert profile.funding_bias == "long_pay"
        assert profile.crowding_intensity > 0.5

    def test_negative_funding_detects_short_crowding(self, analyzer):
        """Strong negative funding = shorts are crowded."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=-0.0008,
            long_short_ratio=0.6,
            sweep_direction=None,
            in_kill_zone=False,
            in_post_kill_zone=False,
        )
        assert profile.crowded_side == "short"
        assert profile.funding_bias == "short_pay"
        assert profile.crowding_intensity > 0.5

    def test_neutral_funding_no_crowding(self, analyzer):
        """Tiny funding + balanced ratio = no crowding."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=0.00005,
            long_short_ratio=1.0,
            sweep_direction=None,
            in_kill_zone=False,
            in_post_kill_zone=False,
        )
        assert profile.crowded_side is None
        assert profile.crowding_intensity < 0.2

    def test_skewed_ratio_only(self, analyzer):
        """Neutral funding but very skewed L/S ratio → crowding detected."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=0.0001,  # Below moderate threshold
            long_short_ratio=2.5,  # Very long-heavy
            sweep_direction=None,
            in_kill_zone=False,
            in_post_kill_zone=False,
        )
        assert profile.crowded_side == "long"

    def test_no_ls_ratio_uses_funding_only(self, analyzer):
        """When L/S ratio unavailable, crowding based on funding alone."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=0.0006,
            long_short_ratio=None,
            sweep_direction=None,
            in_kill_zone=False,
            in_post_kill_zone=False,
        )
        assert profile.crowded_side == "long"
        assert profile.crowding_intensity > 0.5


class TestLiquidationLevels:
    def test_liquidation_levels_calculated(self, analyzer):
        """Verify liquidation levels exist for all leverage tiers."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=0.0,
            long_short_ratio=1.0,
            sweep_direction=None,
            in_kill_zone=False,
            in_post_kill_zone=False,
        )
        assert len(profile.liquidation_clusters) == 10  # 5 tiers × 2 sides

        # Check 10x long liquidation is ~9.6% below
        long_10x = [c for c in profile.liquidation_clusters if c["leverage"] == "10x" and c["side"] == "long"][0]
        assert 89.0 < long_10x["price"] < 91.0  # ~$90.4

        # Check 10x short liquidation is ~9.6% above
        short_10x = [c for c in profile.liquidation_clusters if c["leverage"] == "10x" and c["side"] == "short"][0]
        assert 109.0 < short_10x["price"] < 111.0  # ~$109.6

    def test_nearest_levels(self, analyzer):
        """Nearest long liq below and short liq above current price."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=0.0,
            long_short_ratio=1.0,
            sweep_direction=None,
            in_kill_zone=False,
            in_post_kill_zone=False,
        )
        # Nearest long liq = 100x long at ~$99.6 (1% below with 0.4% maint)
        assert profile.nearest_long_liq > 98.0
        assert profile.nearest_long_liq < 100.0
        # Nearest short liq = 100x short at ~$100.6
        assert profile.nearest_short_liq > 100.0
        assert profile.nearest_short_liq < 102.0


class TestSweepAlignment:
    def test_bullish_sweep_long_crowded_aligns(self, analyzer):
        """Bullish sweep (swept lows) + longs crowded = MMs grabbed long stops → aligned."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=0.001,
            long_short_ratio=1.5,
            sweep_direction="swing_low",
            in_kill_zone=False,
            in_post_kill_zone=False,
        )
        assert profile.sweep_aligns_with_crowding is True

    def test_bearish_sweep_short_crowded_aligns(self, analyzer):
        """Bearish sweep (swept highs) + shorts crowded = MMs grabbed short stops → aligned."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=-0.001,
            long_short_ratio=0.5,
            sweep_direction="swing_high",
            in_kill_zone=False,
            in_post_kill_zone=False,
        )
        assert profile.sweep_aligns_with_crowding is True

    def test_sweep_wrong_side_no_alignment(self, analyzer):
        """Bullish sweep + shorts crowded = grabbed the wrong side → no alignment."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=-0.001,
            long_short_ratio=0.5,
            sweep_direction="swing_low",
            in_kill_zone=False,
            in_post_kill_zone=False,
        )
        assert profile.sweep_aligns_with_crowding is False

    def test_no_sweep_no_alignment(self, analyzer):
        """No sweep direction → no alignment possible."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=0.001,
            long_short_ratio=1.5,
            sweep_direction=None,
            in_kill_zone=False,
            in_post_kill_zone=False,
        )
        assert profile.sweep_aligns_with_crowding is False

    def test_no_crowding_no_alignment(self, analyzer):
        """Sweep present but no crowding → no alignment."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=0.00005,
            long_short_ratio=1.0,
            sweep_direction="swing_low",
            in_kill_zone=False,
            in_post_kill_zone=False,
        )
        assert profile.sweep_aligns_with_crowding is False


class TestJudasSwing:
    def test_kill_zone_extreme_crowding_high_probability(self, analyzer):
        """Kill zone + extreme crowding = high Judas swing probability."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=0.001,
            long_short_ratio=2.0,
            sweep_direction=None,
            in_kill_zone=True,
            in_post_kill_zone=False,
        )
        assert profile.judas_swing_probability > 0.7

    def test_post_kill_zone_moderate_crowding(self, analyzer):
        """Post-kill zone = manipulation likely done, lower probability."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=0.0005,
            long_short_ratio=1.5,
            sweep_direction=None,
            in_kill_zone=False,
            in_post_kill_zone=True,
        )
        assert 0.2 < profile.judas_swing_probability < 0.7

    def test_no_session_low_probability(self, analyzer):
        """No session context = low Judas swing probability."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=0.0005,
            long_short_ratio=1.5,
            sweep_direction=None,
            in_kill_zone=False,
            in_post_kill_zone=False,
        )
        assert profile.judas_swing_probability < 0.3

    def test_no_crowding_zero_probability(self, analyzer):
        """No crowding = zero Judas swing probability."""
        profile = analyzer.analyze(
            current_price=100.0,
            open_interest_usd=1_000_000,
            funding_rate=0.00005,
            long_short_ratio=1.0,
            sweep_direction=None,
            in_kill_zone=True,
            in_post_kill_zone=False,
        )
        assert profile.judas_swing_probability == 0.0


class TestScoringIntegration:
    def test_aligned_sweep_gets_bonus(self):
        """Aligned sweep with crowding gets 5+ bonus points."""
        profile = LeverageProfile(
            open_interest_usd=1_000_000,
            funding_rate=0.001,
            long_short_ratio=1.5,
            crowded_side="long",
            crowding_intensity=0.7,
            funding_bias="long_pay",
            sweep_aligns_with_crowding=True,
            judas_swing_probability=0.3,
        )
        score = AltcoinScanner._score_leverage(profile)
        assert score >= 5.0
        assert score <= 10.0

    def test_extreme_crowding_max_bonus(self):
        """Extreme crowding + Judas swing = max 10 pts."""
        profile = LeverageProfile(
            open_interest_usd=5_000_000,
            funding_rate=0.002,
            long_short_ratio=2.5,
            crowded_side="long",
            crowding_intensity=0.9,
            funding_bias="long_pay",
            sweep_aligns_with_crowding=True,
            judas_swing_probability=0.8,
        )
        score = AltcoinScanner._score_leverage(profile)
        assert score == 10.0

    def test_no_alignment_zero_bonus(self):
        """No sweep alignment = 0 bonus."""
        profile = LeverageProfile(
            open_interest_usd=1_000_000,
            funding_rate=0.001,
            long_short_ratio=1.5,
            crowded_side="long",
            crowding_intensity=0.9,
            funding_bias="long_pay",
            sweep_aligns_with_crowding=False,
            judas_swing_probability=0.8,
        )
        score = AltcoinScanner._score_leverage(profile)
        assert score == 0.0

    def test_moderate_crowding_partial_bonus(self):
        """Moderate crowding = base 5 + 2 = 7 pts."""
        profile = LeverageProfile(
            open_interest_usd=1_000_000,
            funding_rate=0.0004,
            long_short_ratio=1.3,
            crowded_side="long",
            crowding_intensity=0.6,
            funding_bias="long_pay",
            sweep_aligns_with_crowding=True,
            judas_swing_probability=0.3,
        )
        score = AltcoinScanner._score_leverage(profile)
        assert score == 7.0
