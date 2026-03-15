"""Tests for PullbackPlan — formal pending order contract for WAIT_PULLBACK entries."""
from datetime import datetime, timedelta, timezone

import pytest

from src.exchange.models import PullbackPlan, SignalCandidate


# ── PullbackPlan Dataclass Method Tests ──


class TestPullbackPlanIsExpired:
    def test_not_expired_when_future(self):
        plan = PullbackPlan(
            zone_low=100.0, zone_high=105.0,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        assert not plan.is_expired

    def test_expired_when_past(self):
        plan = PullbackPlan(
            zone_low=100.0, zone_high=105.0,
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        assert plan.is_expired


class TestPullbackPlanPriceInZone:
    def _make_plan(self, low=100.0, high=105.0, tol_bps=2.0):
        return PullbackPlan(
            zone_low=low, zone_high=high,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            zone_tolerance_bps=tol_bps,
        )

    def test_price_inside_zone(self):
        plan = self._make_plan()
        assert plan.price_in_zone(102.0)

    def test_price_at_zone_low(self):
        plan = self._make_plan()
        assert plan.price_in_zone(100.0)

    def test_price_at_zone_high(self):
        plan = self._make_plan()
        assert plan.price_in_zone(105.0)

    def test_price_below_zone_rejected(self):
        plan = self._make_plan()
        assert not plan.price_in_zone(99.0)

    def test_price_above_zone_rejected(self):
        plan = self._make_plan()
        assert not plan.price_in_zone(106.0)

    def test_tolerance_allows_slightly_outside(self):
        """2 bps tolerance on zone_high=105 = ~$0.021 tolerance."""
        plan = self._make_plan()
        # Price slightly below zone_low within tolerance
        tolerance = 105.0 * (2.0 / 10000)  # ~0.021
        assert plan.price_in_zone(100.0 - tolerance + 0.001)

    def test_tolerance_does_not_save_far_outside(self):
        plan = self._make_plan()
        assert not plan.price_in_zone(99.5)


class TestPullbackPlanSlippageOk:
    def _make_plan(self, direction="bullish", max_chase=3.0):
        return PullbackPlan(
            zone_low=100.0, zone_high=105.0,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            max_chase_bps=max_chase,
            direction=direction,
        )

    def test_long_price_below_zone_high_always_ok(self):
        plan = self._make_plan("bullish")
        assert plan.slippage_ok(104.0)

    def test_long_price_slightly_above_zone_high_ok(self):
        """3 bps above zone_high=105 = ~$0.0315."""
        plan = self._make_plan("bullish")
        assert plan.slippage_ok(105.02)

    def test_long_price_far_above_zone_high_rejected(self):
        plan = self._make_plan("bullish")
        # 50 bps above = $0.525 — way beyond 3 bps
        assert not plan.slippage_ok(105.6)

    def test_short_price_above_zone_low_always_ok(self):
        plan = self._make_plan("bearish")
        assert plan.slippage_ok(101.0)

    def test_short_price_slightly_below_zone_low_ok(self):
        plan = self._make_plan("bearish")
        assert plan.slippage_ok(99.98)

    def test_short_price_far_below_zone_low_rejected(self):
        plan = self._make_plan("bearish")
        assert not plan.slippage_ok(99.5)


class TestPullbackPlanInvalidationHit:
    def _make_plan(self, direction="bullish", invalidation=98.0):
        return PullbackPlan(
            zone_low=100.0, zone_high=105.0,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            invalidation_level=invalidation,
            direction=direction,
        )

    def test_no_invalidation_when_level_zero(self):
        plan = self._make_plan(invalidation=0.0)
        assert not plan.invalidation_hit(50.0)

    def test_bullish_invalidation_when_price_below(self):
        plan = self._make_plan("bullish", 98.0)
        assert plan.invalidation_hit(97.5)

    def test_bullish_no_invalidation_when_price_above(self):
        plan = self._make_plan("bullish", 98.0)
        assert not plan.invalidation_hit(99.0)

    def test_bearish_invalidation_when_price_above(self):
        plan = self._make_plan("bearish", 107.0)
        assert plan.invalidation_hit(108.0)

    def test_bearish_no_invalidation_when_price_below(self):
        plan = self._make_plan("bearish", 107.0)
        assert not plan.invalidation_hit(106.0)


class TestPullbackPlanComputeLimitPrice:
    def test_bullish_lower_quarter(self):
        plan = PullbackPlan(
            zone_low=100.0, zone_high=104.0,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            direction="bullish",
        )
        # Lower quarter: 100 + 0.25 * 4 = 101
        assert plan.compute_limit_price() == pytest.approx(101.0)

    def test_bearish_upper_quarter(self):
        plan = PullbackPlan(
            zone_low=100.0, zone_high=104.0,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            direction="bearish",
        )
        # Upper quarter: 104 - 0.25 * 4 = 103
        assert plan.compute_limit_price() == pytest.approx(103.0)

    def test_long_alias(self):
        plan = PullbackPlan(
            zone_low=100.0, zone_high=108.0,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            direction="long",
        )
        assert plan.compute_limit_price() == pytest.approx(102.0)


class TestPullbackPlanZoneStr:
    def test_zone_str(self):
        plan = PullbackPlan(
            zone_low=100.123456, zone_high=105.654321,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        assert plan.zone_str() == "100.123456-105.654321"


class TestPullbackPlanAgeSeconds:
    def test_age_positive(self):
        plan = PullbackPlan(
            zone_low=100.0, zone_high=105.0,
            created_at=datetime.now(timezone.utc) - timedelta(seconds=120),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        assert plan.age_seconds >= 119.0  # At least 119 seconds


# ── SignalCandidate Integration ──


class TestSignalCandidatePullbackPlan:
    def test_default_plan_is_none(self):
        sig = SignalCandidate(score=50.0, direction="bullish", symbol="BTCUSDT")
        assert sig.pullback_plan is None

    def test_plan_can_be_attached(self):
        plan = PullbackPlan(
            zone_low=100.0, zone_high=105.0,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            direction="bullish",
        )
        sig = SignalCandidate(score=50.0, direction="bullish", symbol="BTCUSDT")
        sig.pullback_plan = plan
        assert sig.pullback_plan is not None
        assert sig.pullback_plan.zone_low == 100.0


# ── Config Defaults ──


class TestPullbackPlanConfigDefaults:
    def test_config_defaults(self):
        from src.config import Settings

        defaults = Settings.model_fields
        assert defaults["pullback_zone_tolerance_bps"].default == 2.0
        assert defaults["pullback_max_chase_bps"].default == 3.0
        assert defaults["pullback_valid_candles"].default == 6
        assert defaults["pullback_use_limit_in_zone"].default is True


# ── Zone Enforcement Gate Logic ──


class TestZoneEnforcementGate:
    """Tests that simulate the gate logic used in core.py and orders.py."""

    def _make_plan(self, direction="bullish", zone_low=100.0, zone_high=105.0,
                   invalidation=98.0, max_chase=3.0, expired=False):
        now = datetime.now(timezone.utc)
        return PullbackPlan(
            zone_low=zone_low, zone_high=zone_high,
            created_at=now - timedelta(minutes=10),
            expires_at=now - timedelta(minutes=1) if expired else now + timedelta(minutes=20),
            invalidation_level=invalidation,
            max_chase_bps=max_chase,
            direction=direction,
        )

    def _gate_check(self, plan, live_price):
        """Replicate the gate logic from core.py/orders.py."""
        if plan.is_expired:
            return "plan_expired"
        if plan.invalidation_hit(live_price):
            return "invalidation_hit"
        if not plan.price_in_zone(live_price):
            return "price_outside_zone"
        if not plan.slippage_ok(live_price):
            return "slippage_exceeded"
        return None  # Passed

    def test_passes_when_price_in_zone(self):
        plan = self._make_plan()
        assert self._gate_check(plan, 102.0) is None

    def test_rejects_expired_plan(self):
        plan = self._make_plan(expired=True)
        assert self._gate_check(plan, 102.0) == "plan_expired"

    def test_rejects_invalidation_hit(self):
        plan = self._make_plan(invalidation=98.0)
        assert self._gate_check(plan, 97.0) == "invalidation_hit"

    def test_rejects_price_outside_zone(self):
        plan = self._make_plan()
        assert self._gate_check(plan, 110.0) == "price_outside_zone"

    def test_rejects_slippage_exceeded(self):
        """Price slightly above zone_high beyond max_chase_bps.

        The gate checks price_in_zone (with tolerance) before slippage_ok.
        To hit slippage_exceeded, the price must pass price_in_zone (within
        tolerance) but fail the stricter slippage check.

        With zone_high=105, tolerance=2bps (~$0.021), and max_chase=0.5bps:
        - 105.015 is within zone tolerance (passes price_in_zone)
        - But 0.015/105 = 1.43 bps > 0.5 bps max_chase (fails slippage_ok)
        """
        plan = self._make_plan(max_chase=0.5)  # Very tight 0.5 bps chase
        # 105.015 is within 2 bps tolerance but beyond 0.5 bps max_chase
        assert self._gate_check(plan, 105.015) == "slippage_exceeded"

    def test_bearish_in_zone_passes(self):
        plan = self._make_plan(direction="bearish", invalidation=107.0)
        assert self._gate_check(plan, 103.0) is None

    def test_bearish_invalidation_above(self):
        plan = self._make_plan(direction="bearish", invalidation=107.0)
        assert self._gate_check(plan, 108.0) == "invalidation_hit"


# ── ADJUST_ZONE Sync ──


class TestAdjustZoneSync:
    """Verify that ADJUST_ZONE updates PullbackPlan correctly."""

    def test_zone_update_preserves_expiry(self):
        now = datetime.now(timezone.utc)
        original_expiry = now + timedelta(minutes=20)
        plan = PullbackPlan(
            zone_low=100.0, zone_high=105.0,
            created_at=now, expires_at=original_expiry,
            direction="bullish",
        )
        plan.limit_price = plan.compute_limit_price()

        # Simulate ADJUST_ZONE
        plan.zone_high = 106.0
        plan.zone_low = 101.0
        plan.limit_price = plan.compute_limit_price()
        plan.zone_updates += 1

        # Expiry should NOT have changed
        assert plan.expires_at == original_expiry
        assert plan.zone_updates == 1
        # New limit price for bullish: 101 + 0.25 * 5 = 102.25
        assert plan.limit_price == pytest.approx(102.25)

    def test_multiple_zone_updates_increment(self):
        now = datetime.now(timezone.utc)
        plan = PullbackPlan(
            zone_low=100.0, zone_high=105.0,
            created_at=now, expires_at=now + timedelta(minutes=20),
            direction="bearish",
        )
        plan.zone_updates += 1
        plan.zone_updates += 1
        plan.zone_updates += 1
        assert plan.zone_updates == 3


# ── State Serialization ──


class TestPullbackPlanSerialization:
    """Verify PullbackPlan can round-trip through get_state/restore_state format."""

    def test_round_trip(self):
        now = datetime.now(timezone.utc)
        plan = PullbackPlan(
            zone_low=100.5, zone_high=105.5,
            created_at=now, expires_at=now + timedelta(minutes=30),
            invalidation_level=98.0,
            max_chase_bps=3.0,
            zone_tolerance_bps=2.0,
            valid_for_candles=6,
            direction="bullish",
            limit_price=101.75,
            original_suggested_entry=102.0,
            zone_updates=1,
        )
        # Serialize (as done in entry_refiner.get_state)
        serialized = {
            "zone_low": plan.zone_low,
            "zone_high": plan.zone_high,
            "created_at": plan.created_at.isoformat(),
            "expires_at": plan.expires_at.isoformat(),
            "invalidation_level": plan.invalidation_level,
            "max_chase_bps": plan.max_chase_bps,
            "zone_tolerance_bps": plan.zone_tolerance_bps,
            "valid_for_candles": plan.valid_for_candles,
            "direction": plan.direction,
            "limit_price": plan.limit_price,
            "original_suggested_entry": plan.original_suggested_entry,
            "zone_updates": plan.zone_updates,
        }
        # Deserialize (as done in entry_refiner.restore_state)
        restored = PullbackPlan(
            zone_low=serialized["zone_low"],
            zone_high=serialized["zone_high"],
            created_at=datetime.fromisoformat(serialized["created_at"]),
            expires_at=datetime.fromisoformat(serialized["expires_at"]),
            invalidation_level=serialized.get("invalidation_level", 0.0),
            max_chase_bps=serialized.get("max_chase_bps", 3.0),
            zone_tolerance_bps=serialized.get("zone_tolerance_bps", 2.0),
            valid_for_candles=serialized.get("valid_for_candles", 6),
            direction=serialized.get("direction", ""),
            limit_price=serialized.get("limit_price", 0.0),
            original_suggested_entry=serialized.get("original_suggested_entry", 0.0),
            zone_updates=serialized.get("zone_updates", 0),
        )
        assert restored.zone_low == plan.zone_low
        assert restored.zone_high == plan.zone_high
        assert restored.invalidation_level == plan.invalidation_level
        assert restored.direction == plan.direction
        assert restored.limit_price == plan.limit_price
        assert restored.zone_updates == plan.zone_updates
        assert not restored.is_expired


# ── Enter Now Unchanged ──


class TestEnterNowUnchanged:
    """ENTRY_CONFIRMED signals (no PullbackPlan) must be completely unaffected."""

    def test_signal_without_plan(self):
        sig = SignalCandidate(
            score=65.0, direction="bullish", symbol="ETHUSDT",
            entry_price=2000.0,
        )
        assert sig.pullback_plan is None
        # Gate logic should skip plan checks when plan is None
        plan = getattr(sig, "pullback_plan", None)
        assert plan is None
