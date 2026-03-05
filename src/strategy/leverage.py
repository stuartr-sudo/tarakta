"""Leverage intelligence for the Trade Travel Chill strategy.

Analyzes open interest, funding rates, and long/short ratios to:
1. Determine which side (longs or shorts) is crowded
2. Estimate liquidation cluster levels at common leverage tiers
3. Score alignment between detected sweeps and leverage crowding
4. Predict Judas swing probability based on funding + session timing

This data CONFIRMS existing sweep detection — it does not replace it.
Funding rate tells us WHO got swept (longs or shorts).
OI tells us the magnitude of position building.
Liquidation levels tell us WHERE the next sweep target is.
"""
from __future__ import annotations

from src.exchange.models import LeverageProfile
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Common leverage tiers for liquidation estimation
LEVERAGE_TIERS = [5, 10, 25, 50, 100]

# Funding rate thresholds (per 8h period)
FUNDING_EXTREME = 0.0005   # 0.05% — very crowded
FUNDING_MODERATE = 0.0002  # 0.02% — moderately crowded

# Long/short ratio thresholds
LS_EXTREME = 2.0   # 2:1 ratio
LS_MODERATE = 1.3   # 1.3:1 ratio

# Binance approximate maintenance margin rate
MAINTENANCE_MARGIN = 0.004  # 0.4%


class LeverageAnalyzer:
    """Analyzes leverage data to detect crowding and estimate liquidation levels."""

    def analyze(
        self,
        current_price: float,
        open_interest_usd: float,
        funding_rate: float,
        long_short_ratio: float | None,
        sweep_direction: str | None,
        in_kill_zone: bool,
        in_post_kill_zone: bool,
    ) -> LeverageProfile:
        """Produce a full leverage profile for a symbol.

        Args:
            current_price: Current market price
            open_interest_usd: Total open interest in USD
            funding_rate: Current 8h funding rate (positive = longs pay)
            long_short_ratio: Top trader L/S ratio (>1 = more longs), or None
            sweep_direction: "bullish" (swept lows) or "bearish" (swept highs)
            in_kill_zone: Whether we're in a London/NY kill zone
            in_post_kill_zone: Whether we're in a post-kill-zone window
        """
        crowded_side, crowding_intensity, funding_bias = self._detect_crowding(
            funding_rate, long_short_ratio,
        )

        liquidation_clusters = self._estimate_liquidation_levels(current_price)

        nearest_long_liq, nearest_short_liq = self._nearest_liquidation_levels(
            liquidation_clusters, current_price,
        )

        sweep_aligns = self._check_sweep_alignment(sweep_direction, crowded_side)

        judas_prob = self._judas_swing_probability(
            crowded_side, crowding_intensity, in_kill_zone, in_post_kill_zone,
        )

        profile = LeverageProfile(
            open_interest_usd=open_interest_usd,
            funding_rate=funding_rate,
            long_short_ratio=long_short_ratio,
            crowded_side=crowded_side,
            crowding_intensity=crowding_intensity,
            funding_bias=funding_bias,
            liquidation_clusters=liquidation_clusters,
            nearest_long_liq=nearest_long_liq,
            nearest_short_liq=nearest_short_liq,
            sweep_aligns_with_crowding=sweep_aligns,
            judas_swing_probability=judas_prob,
        )

        logger.info(
            "leverage_analyzed",
            funding=f"{funding_rate:.6f}",
            oi_usd=f"{open_interest_usd:,.0f}",
            crowded=crowded_side,
            intensity=f"{crowding_intensity:.2f}",
            sweep_aligns=sweep_aligns,
            judas_prob=f"{judas_prob:.2f}",
        )

        return profile

    def _detect_crowding(
        self, funding_rate: float, long_short_ratio: float | None,
    ) -> tuple[str | None, float, str | None]:
        """Detect which side is crowded and how extreme.

        Returns:
            (crowded_side, crowding_intensity, funding_bias)
        """
        # Funding component: how far from neutral (0.0-1.0)
        abs_funding = abs(funding_rate)
        funding_component = min(1.0, abs_funding / FUNDING_EXTREME)

        # Ratio component: how far from balanced 1.0 (0.0-1.0)
        ratio_component = 0.0
        if long_short_ratio is not None:
            deviation = abs(long_short_ratio - 1.0)
            ratio_component = min(1.0, deviation / (LS_EXTREME - 1.0))

        # Combined intensity (funding weighted higher — it's the market-wide signal)
        if long_short_ratio is not None:
            intensity = funding_component * 0.6 + ratio_component * 0.4
        else:
            intensity = funding_component  # Only funding available

        # Determine bias direction
        funding_bias = None
        if funding_rate > FUNDING_MODERATE:
            funding_bias = "long_pay"
        elif funding_rate < -FUNDING_MODERATE:
            funding_bias = "short_pay"

        # Determine crowded side — need at least moderate signal
        crowded_side = None
        if funding_rate > FUNDING_MODERATE:
            # Longs are paying — market is long-biased
            if long_short_ratio is None or long_short_ratio >= 1.0:
                crowded_side = "long"
        elif funding_rate < -FUNDING_MODERATE:
            # Shorts are paying — market is short-biased
            if long_short_ratio is None or long_short_ratio <= 1.0:
                crowded_side = "short"
        elif long_short_ratio is not None:
            # Neutral funding but skewed ratio
            if long_short_ratio >= LS_MODERATE:
                crowded_side = "long"
            elif long_short_ratio <= 1 / LS_MODERATE:  # ~0.77
                crowded_side = "short"

        return crowded_side, intensity, funding_bias

    def _estimate_liquidation_levels(self, current_price: float) -> list[dict]:
        """Estimate liquidation cluster levels at common leverage tiers.

        Assumes positions were opened near current price (reasonable for
        detecting nearby clusters). Real levels vary by individual entry.
        """
        clusters = []
        for lev in LEVERAGE_TIERS:
            # Long liquidation: price drops enough to wipe margin
            long_liq = current_price * (1 - 1 / lev + MAINTENANCE_MARGIN)
            # Short liquidation: price rises enough to wipe margin
            short_liq = current_price * (1 + 1 / lev - MAINTENANCE_MARGIN)

            clusters.append({
                "price": round(long_liq, 8),
                "leverage": f"{lev}x",
                "side": "long",
            })
            clusters.append({
                "price": round(short_liq, 8),
                "leverage": f"{lev}x",
                "side": "short",
            })

        return clusters

    def _nearest_liquidation_levels(
        self, clusters: list[dict], current_price: float,
    ) -> tuple[float, float]:
        """Find the nearest long liq level below and short liq level above."""
        nearest_long = 0.0
        nearest_short = 0.0

        for c in clusters:
            if c["side"] == "long" and c["price"] < current_price:
                if nearest_long == 0.0 or c["price"] > nearest_long:
                    nearest_long = c["price"]
            elif c["side"] == "short" and c["price"] > current_price:
                if nearest_short == 0.0 or c["price"] < nearest_short:
                    nearest_short = c["price"]

        return nearest_long, nearest_short

    def _check_sweep_alignment(
        self, sweep_direction: str | None, crowded_side: str | None,
    ) -> bool:
        """Check if the sweep grabbed the crowded side's liquidity.

        - Bullish sweep (swept lows = grabbed long stop losses below)
          + longs crowded = MMs cleared overleveraged longs → bullish move incoming
        - Bearish sweep (swept highs = grabbed short stop losses above)
          + shorts crowded = MMs cleared overleveraged shorts → bearish move incoming
        """
        if not sweep_direction or not crowded_side:
            return False

        if sweep_direction == "bullish" and crowded_side == "long":
            return True
        if sweep_direction == "bearish" and crowded_side == "short":
            return True

        return False

    def _judas_swing_probability(
        self,
        crowded_side: str | None,
        crowding_intensity: float,
        in_kill_zone: bool,
        in_post_kill_zone: bool,
    ) -> float:
        """Estimate probability of a Judas swing (fake-out) based on leverage + timing.

        Kill zone + extreme crowding = high probability of initial fake-out
        in the direction of the crowded side, followed by reversal.

        Post-kill zone = the Judas swing likely already happened.
        """
        if not crowded_side:
            return 0.0

        if in_kill_zone and crowding_intensity > 0.4:
            # Active manipulation window + crowded market = prime Judas swing setup
            return min(1.0, 0.4 + crowding_intensity * 0.6)

        if in_post_kill_zone and crowding_intensity > 0.3:
            # Manipulation likely done — real move underway
            return min(1.0, 0.2 + crowding_intensity * 0.4)

        # No session context — mild signal from crowding alone
        return crowding_intensity * 0.2
