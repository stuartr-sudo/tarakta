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
            sweep_direction: "swing_low" (sell-side taken) or "swing_high" (buy-side taken)
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
        # Extreme funding (>5x moderate) overrides mildly contradictory L/S ratio
        # because funding is the actual market-wide cost being paid.
        extreme_funding = abs_funding > FUNDING_EXTREME * 2
        crowded_side = None
        if funding_rate > FUNDING_MODERATE:
            # Longs are paying — market is long-biased
            if long_short_ratio is None or long_short_ratio >= 1.0 or extreme_funding:
                crowded_side = "long"
        elif funding_rate < -FUNDING_MODERATE:
            # Shorts are paying — market is short-biased
            if long_short_ratio is None or long_short_ratio <= 1.0 or extreme_funding:
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

        - swing_low sweep (swept lows = grabbed long stop losses below)
          + longs crowded = MMs cleared overleveraged longs → bullish move incoming
        - swing_high sweep (swept highs = grabbed short stop losses above)
          + shorts crowded = MMs cleared overleveraged shorts → bearish move incoming
        """
        if not sweep_direction or not crowded_side:
            return False

        if sweep_direction == "swing_low" and crowded_side == "long":
            return True
        if sweep_direction == "swing_high" and crowded_side == "short":
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

    # ------------------------------------------------------------------
    # Sweep flip probability — used by configurable bot's "smart flip"
    # ------------------------------------------------------------------

    def compute_sweep_flip_probability(
        self,
        signal_direction: str,
        current_price: float,
        open_interest_usd: float,
        funding_rate: float,
        long_short_ratio: float | None,
        in_kill_zone: bool,
        in_post_kill_zone: bool,
    ) -> float:
        """Compute the probability that flipping the signal is the correct play.

        Uses a 6-factor weighted model to estimate the likelihood that the
        market will sweep liquidity in the signal's direction BEFORE the
        expected move — meaning the opposite trade (flipped) is profitable.

        A high score (close to 1.0) means "definitely flip"; a low score
        (close to 0.0) means "trade the signal direction as-is".

        Factors and weights:
            1. Funding bias       (0.25) — crowding via funding costs
            2. L/S ratio          (0.20) — top-trader positioning skew
            3. Crowding intensity  (0.20) — combined funding + ratio signal
            4. Liquidation prox    (0.15) — nearby liq clusters as sweep magnets
            5. Kill zone timing    (0.10) — session-based manipulation window
            6. OI magnitude        (0.10) — fuel available for the sweep
        """
        # --- Factor 1: Funding bias (weight 0.25) ---
        # If signal is bullish and longs are paying (funding > 0), the long
        # side is crowded → MMs will sweep long stops first → flip is correct.
        # Mirror for bearish signal + negative funding.
        funding_score = 0.0
        if signal_direction == "bullish" and funding_rate > 0:
            # Longs crowded, signal says go long → sweep likely before move up
            funding_score = min(1.0, abs(funding_rate) / FUNDING_EXTREME)
        elif signal_direction == "bearish" and funding_rate < 0:
            # Shorts crowded, signal says go short → sweep likely before move down
            funding_score = min(1.0, abs(funding_rate) / FUNDING_EXTREME)

        # --- Factor 2: L/S ratio imbalance (weight 0.20) ---
        ls_score = 0.0
        if long_short_ratio is not None:
            if signal_direction == "bullish" and long_short_ratio > 1.0:
                # More longs than shorts → longs crowded, sweep down likely
                ls_score = min(1.0, (long_short_ratio - 1.0) / (LS_EXTREME - 1.0))
            elif signal_direction == "bearish" and long_short_ratio < 1.0:
                # More shorts → shorts crowded, sweep up likely
                inverse = 1.0 / long_short_ratio if long_short_ratio > 0 else LS_EXTREME
                ls_score = min(1.0, (inverse - 1.0) / (LS_EXTREME - 1.0))

        # --- Factor 3: Crowding intensity (weight 0.20) ---
        crowded_side, crowding_intensity, _ = self._detect_crowding(
            funding_rate, long_short_ratio,
        )
        crowding_score = 0.0
        if crowded_side:
            # Crowding matches signal direction → flip more likely correct
            if (signal_direction == "bullish" and crowded_side == "long") or \
               (signal_direction == "bearish" and crowded_side == "short"):
                crowding_score = crowding_intensity

        # --- Factor 4: Liquidation proximity (weight 0.15) ---
        clusters = self._estimate_liquidation_levels(current_price)
        nearest_long_liq, nearest_short_liq = self._nearest_liquidation_levels(
            clusters, current_price,
        )
        liq_score = 0.0
        if signal_direction == "bullish" and nearest_long_liq > 0:
            # Long liq levels below = sweep magnets for downward move
            distance_pct = (current_price - nearest_long_liq) / current_price
            liq_score = max(0.0, 1.0 - min(1.0, distance_pct / 0.05))
        elif signal_direction == "bearish" and nearest_short_liq > 0:
            # Short liq levels above = sweep magnets for upward move
            distance_pct = (nearest_short_liq - current_price) / current_price
            liq_score = max(0.0, 1.0 - min(1.0, distance_pct / 0.05))

        # --- Factor 5: Kill zone timing (weight 0.10) ---
        if in_kill_zone:
            timing_score = 1.0
        elif in_post_kill_zone:
            timing_score = 0.5
        else:
            timing_score = 0.1

        # --- Factor 6: OI magnitude (weight 0.10) ---
        # More OI = more fuel for MMs to trigger a sweep
        oi_score = min(1.0, open_interest_usd / 500_000_000)

        # --- Weighted combination ---
        probability = (
            funding_score * 0.25
            + ls_score * 0.20
            + crowding_score * 0.20
            + liq_score * 0.15
            + timing_score * 0.10
            + oi_score * 0.10
        )

        probability = max(0.0, min(1.0, probability))

        logger.info(
            "sweep_flip_probability",
            signal_direction=signal_direction,
            prob=f"{probability:.3f}",
            funding=f"{funding_score:.2f}",
            ls=f"{ls_score:.2f}",
            crowding=f"{crowding_score:.2f}",
            liq=f"{liq_score:.2f}",
            timing=f"{timing_score:.2f}",
            oi=f"{oi_score:.2f}",
        )

        return probability
