"""MM Method target identification and take-profit logic.

Combines multiple target sources to identify where price is likely headed
at each level of the 3-level cycle. The MM Method uses specific targeting
rules that differ from traditional TA.

Target sources (in priority order):
1. **EMA targets per level**: Level 1 → 50 EMA, Level 2 → 200 EMA, Level 3 → 800 EMA
2. **Unrecovered Vector candles**: PVSRA 200% or 150% candles whose imbalance
   hasn't been filled by subsequent consolidation
3. **HOW/LOW (High/Low of Week)**: Previous week's extremes as targets
4. **Liquidation level clusters**: Where leveraged positions will be liquidated
   (front-run by ~0.1-0.2%)
5. **Fixed Range Volume Profile (FRVP) crevices**: Low-volume zones where price
   moves quickly through → acts as a magnet

Take Profit Rules (from the course):
- Initial R:R calculation uses Level 1 target ONLY — beyond is bonus
- At Level 1 complete: take 25-33% profit
- At Level 2 complete: take 50% total
- At Level 3 SVC: take remaining
- Friday UK session: exit when peak formation appears
- If only 2 levels by Friday: can hold through weekend IF SL can move without ruining R:R
- Every SL movement point = also a partial profit point
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EMA periods used as targets per level (course lessons 12, 47, 48)
# Level 1 → 50 EMA (primary, lesson 12 "Level 1 should break the 50 EMA")
# Level 2 → 800 EMA (lesson 12: "Depending on where the 800 EMA is, that is
#                    often the level 2 target") — **200 was wrong before 2026-04**
# Level 3 → higher-TF 200/800 EMA (lesson 12: "a 200 or an 800 EMA on a higher
#                                   time frame as a Target")
LEVEL_EMA_TARGETS = {
    1: 50,    # Level 1 targets the 50 EMA
    2: 800,   # Level 2 targets the 800 EMA (corrected 2026-04 per course audit)
    3: 800,   # Level 3 targets higher-TF EMA — best same-TF proxy is still 800
}

# Fallback EMAs at each level (used when primary not in direction)
LEVEL_EMA_FALLBACKS = {
    1: 200,   # Lesson 48: "If the 200 isn't a good target because it's too close, look for a previous unrecovered Vector candle"
    2: 200,
    3: 200,
}

# Minimum volume ratio to qualify as a "Vector" candle (PVSRA)
VECTOR_200_THRESHOLD = 2.0  # 200% of 10-bar average
VECTOR_150_THRESHOLD = 1.5  # 150% of 10-bar average
VOLUME_LOOKBACK = 10

# How much to front-run liquidation levels (as fraction of price)
LIQUIDATION_FRONTRUN_PCT = 0.002  # 0.2%

# Partial profit framework
PROFIT_SCHEDULE = {
    1: 0.30,   # Take 30% at Level 1 (25-33% range)
    2: 0.50,   # Take 50% total by Level 2
    3: 1.00,   # Take remaining at Level 3
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VectorCandle:
    """An unrecovered Vector candle (PVSRA high-volume imbalance).

    Vector candles are price targets because the imbalance they created
    persists until consolidation fills the zone.
    """
    index: int = -1                # Index in the DataFrame
    timestamp: str = ""
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume_ratio: float = 0.0     # volume / avg volume
    vector_type: str = ""          # "vector_200" or "vector_150"
    direction: str = ""            # "bullish" (close > open) or "bearish"
    midpoint: float = 0.0         # 50% of the candle body (primary target)
    recovered: bool = False       # Has price returned to fill this zone?


@dataclass
class TargetLevel:
    """A single target level with source and priority."""
    price: float = 0.0
    source: str = ""               # "ema_50", "ema_200", "vector", "how", "low", etc.
    level_num: int = 0             # Which MM level this target is for (1, 2, 3)
    priority: int = 0             # Lower = higher priority
    description: str = ""
    distance_pct: float = 0.0     # Distance from current price as %


@dataclass
class PartialProfitPlan:
    """Partial profit schedule based on MM level progression."""
    entries: list[dict] = field(default_factory=list)
    # Each entry: {level: int, target_price: float, close_pct: float, description: str}


@dataclass
class TargetAnalysis:
    """Complete target analysis for a trade setup."""
    direction: str = ""            # "bullish" or "bearish"
    current_level: int = 0         # Current MM level (1-3)
    entry_price: float = 0.0

    # Targets per level
    level_1_targets: list[TargetLevel] = field(default_factory=list)
    level_2_targets: list[TargetLevel] = field(default_factory=list)
    level_3_targets: list[TargetLevel] = field(default_factory=list)

    # Primary targets (best for each level)
    primary_l1: TargetLevel | None = None
    primary_l2: TargetLevel | None = None
    primary_l3: TargetLevel | None = None

    # Unrecovered vectors
    unrecovered_vectors: list[VectorCandle] = field(default_factory=list)

    # Partial profit plan
    profit_plan: PartialProfitPlan | None = None

    # R:R using Level 1 target only (as per course: "beyond is bonus")
    risk_reward_l1: float = 0.0


# ---------------------------------------------------------------------------
# Vector Scanner
# ---------------------------------------------------------------------------

class VectorScanner:
    """Finds unrecovered Vector candles (PVSRA high-volume imbalances)."""

    def scan(
        self,
        ohlc: pd.DataFrame,
        current_price: float,
        lookback: int = 200,
    ) -> list[VectorCandle]:
        """Scan for unrecovered Vector candles.

        A Vector candle has volume >= 150% (or 200%) of the average of the
        prior 10 candles. It's "unrecovered" if price hasn't returned to
        consolidate through the candle's body zone.

        Args:
            ohlc: OHLCV DataFrame.
            current_price: Current market price (for recovery checking).
            lookback: How many candles back to scan.

        Returns:
            List of unrecovered VectorCandle objects, sorted by distance.
        """
        if ohlc is None or ohlc.empty or len(ohlc) < VOLUME_LOOKBACK + 2:
            return []

        df = ohlc.tail(lookback) if len(ohlc) > lookback else ohlc
        volumes = df["volume"].values
        closes = df["close"].values
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values

        vectors: list[VectorCandle] = []

        for i in range(VOLUME_LOOKBACK, len(df)):
            avg_vol = np.mean(volumes[i - VOLUME_LOOKBACK:i])
            if avg_vol <= 0:
                continue

            ratio = volumes[i] / avg_vol

            if ratio >= VECTOR_200_THRESHOLD:
                vtype = "vector_200"
            elif ratio >= VECTOR_150_THRESHOLD:
                vtype = "vector_150"
            else:
                continue

            # Direction
            direction = "bullish" if closes[i] > opens[i] else "bearish"

            # Midpoint of the body (primary target)
            body_high = max(opens[i], closes[i])
            body_low = min(opens[i], closes[i])
            midpoint = (body_high + body_low) / 2

            # Check if recovered: has price SINCE this candle consolidated
            # through the body zone? Look at candles after this one.
            recovered = False
            for j in range(i + 1, len(df)):
                # Price returned into the body zone = recovered
                if lows[j] <= body_high and highs[j] >= body_low:
                    # Check if it spent time there (not just a wick through)
                    if min(opens[j], closes[j]) <= body_high and max(opens[j], closes[j]) >= body_low:
                        recovered = True
                        break

            if not recovered:
                ts = ""
                if hasattr(df.index, '__getitem__'):
                    try:
                        ts = str(df.index[i])
                    except Exception:
                        pass

                vectors.append(VectorCandle(
                    index=i,
                    timestamp=ts,
                    high=float(highs[i]),
                    low=float(lows[i]),
                    close=float(closes[i]),
                    volume_ratio=round(ratio, 2),
                    vector_type=vtype,
                    direction=direction,
                    midpoint=round(midpoint, 8),
                    recovered=False,
                ))

        # Sort by distance from current price
        vectors.sort(key=lambda v: abs(v.midpoint - current_price))

        return vectors


# ---------------------------------------------------------------------------
# Target Analyzer
# ---------------------------------------------------------------------------

class TargetAnalyzer:
    """Identifies and prioritizes targets for MM Method trades.

    Combines EMA targets, unrecovered vectors, HOW/LOW, and liquidation
    levels into a prioritized target list for each level.
    """

    def __init__(self) -> None:
        self.vector_scanner = VectorScanner()

    def analyze(
        self,
        ohlc: pd.DataFrame,
        direction: str,
        entry_price: float,
        stop_loss: float,
        current_level: int = 1,
        ema_values: dict[int, float] | None = None,
        how: float | None = None,
        low: float | None = None,
        liquidation_levels: list[float] | None = None,
        htf_ema_values: dict[int, float] | None = None,
    ) -> TargetAnalysis:
        """Run full target analysis for a trade setup.

        Args:
            ohlc: OHLCV DataFrame (1H recommended).
            direction: Trade direction ("bullish" or "bearish").
            entry_price: Entry price.
            stop_loss: Stop loss price.
            current_level: Current MM level (1, 2, or 3).
            ema_values: Dict of EMA period -> current value {50: 95000, 200: 93000, ...}.
            how: High of Week.
            low: Low of Week.
            liquidation_levels: Known liquidation price levels.

        Returns:
            TargetAnalysis with targets per level and profit plan.
        """
        result = TargetAnalysis(
            direction=direction,
            current_level=current_level,
            entry_price=entry_price,
        )

        # 1. Find unrecovered vectors
        vectors = self.vector_scanner.scan(ohlc, entry_price)
        result.unrecovered_vectors = vectors

        # 2. Build targets for each level.
        # Level 3 is passed htf_ema_values (4H / daily 200 & 800) if available
        # so course lesson 12's "higher-TF 200/800 EMA as L3 target" is
        # actually honoured — previously L3 used same-TF 800, which is the
        # same as L2 and collapses in range markets.
        result.level_1_targets = self._targets_for_level(
            1, direction, entry_price, ema_values, how, low,
            liquidation_levels, vectors,
        )
        result.level_2_targets = self._targets_for_level(
            2, direction, entry_price, ema_values, how, low,
            liquidation_levels, vectors,
        )
        result.level_3_targets = self._targets_for_level(
            3, direction, entry_price, ema_values, how, low,
            liquidation_levels, vectors,
            htf_ema_values=htf_ema_values,
        )

        # 3. Pick primary targets (closest valid target for each level)
        result.primary_l1 = self._pick_primary(result.level_1_targets, direction, entry_price)
        result.primary_l2 = self._pick_primary(result.level_2_targets, direction, entry_price)
        result.primary_l3 = self._pick_primary(result.level_3_targets, direction, entry_price)

        # 4. Calculate R:R using Level 1 target only
        if result.primary_l1:
            risk = abs(entry_price - stop_loss)
            reward = abs(result.primary_l1.price - entry_price)
            result.risk_reward_l1 = round(reward / risk, 2) if risk > 0 else 0.0

        # 5. Build partial profit plan
        result.profit_plan = self._build_profit_plan(
            direction, entry_price, result.primary_l1, result.primary_l2, result.primary_l3,
        )

        logger.info(
            "target_analysis_complete",
            direction=direction,
            level=current_level,
            l1_target=result.primary_l1.price if result.primary_l1 else None,
            l2_target=result.primary_l2.price if result.primary_l2 else None,
            l3_target=result.primary_l3.price if result.primary_l3 else None,
            rr_l1=result.risk_reward_l1,
            unrecovered_vectors=len(vectors),
        )

        return result

    def _targets_for_level(
        self,
        level: int,
        direction: str,
        entry_price: float,
        ema_values: dict[int, float] | None,
        how: float | None,
        low: float | None,
        liquidation_levels: list[float] | None,
        vectors: list[VectorCandle],
        htf_ema_values: dict[int, float] | None = None,
    ) -> list[TargetLevel]:
        """Build target list for a specific level."""
        targets: list[TargetLevel] = []

        # --- Level 3 higher-TF EMA targets (course lesson 12) ---
        # "a 200 or an 800 EMA on a higher time frame as a Target". When
        # callers pass htf_ema_values ({200: 4H_200, 800: 4H_800, ...}), we
        # treat those as the PRIMARY L3 targets, pushing the same-TF 800 to
        # a secondary priority. This prevents L2 and L3 collapsing to the
        # same price (the bug that stopped RENDER out prematurely).
        if level == 3 and htf_ema_values:
            for htf_period in (800, 200):  # 800 first — it's the "big" L3
                if htf_period in htf_ema_values:
                    htf_price = htf_ema_values[htf_period]
                    if htf_price and self._is_valid_target(htf_price, direction, entry_price):
                        dist = abs(htf_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                        targets.append(TargetLevel(
                            price=htf_price,
                            source=f"htf_ema_{htf_period}",
                            level_num=level,
                            priority=1,  # top priority — beats same-TF 800
                            description=f"Higher-TF {htf_period} EMA (course L3 target)",
                            distance_pct=round(dist, 2),
                        ))

        # EMA target for this level
        ema_period = LEVEL_EMA_TARGETS.get(level)
        if ema_period and ema_values and ema_period in ema_values:
            ema_price = ema_values[ema_period]
            # Only valid if in the right direction
            if self._is_valid_target(ema_price, direction, entry_price):
                dist = abs(ema_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                # Demote the same-TF 800 at L3 when an htf target already
                # supplied a real higher-TF L3.
                priority = 2 if (level == 3 and htf_ema_values and any(p in htf_ema_values for p in (200, 800))) else 1
                targets.append(TargetLevel(
                    price=ema_price,
                    source=f"ema_{ema_period}",
                    level_num=level,
                    priority=priority,
                    description=f"{ema_period} EMA (Level {level} primary target)",
                    distance_pct=round(dist, 2),
                ))

        # Unrecovered vectors in the right direction
        for v in vectors:
            if self._is_valid_target(v.midpoint, direction, entry_price):
                dist = abs(v.midpoint - entry_price) / entry_price * 100 if entry_price > 0 else 0
                # Assign to appropriate level based on distance
                if level == 1 and dist <= 3:
                    targets.append(TargetLevel(
                        price=v.midpoint,
                        source="vector",
                        level_num=level,
                        priority=2,
                        description=f"Unrecovered {v.vector_type} ({v.direction})",
                        distance_pct=round(dist, 2),
                    ))
                elif level == 2 and 2 < dist <= 8:
                    targets.append(TargetLevel(
                        price=v.midpoint,
                        source="vector",
                        level_num=level,
                        priority=2,
                        description=f"Unrecovered {v.vector_type} ({v.direction})",
                        distance_pct=round(dist, 2),
                    ))
                elif level == 3 and dist > 5:
                    targets.append(TargetLevel(
                        price=v.midpoint,
                        source="vector",
                        level_num=level,
                        priority=3,
                        description=f"Unrecovered {v.vector_type} ({v.direction})",
                        distance_pct=round(dist, 2),
                    ))

        # HOW/LOW targets (primarily for Level 3)
        if how and self._is_valid_target(how, direction, entry_price):
            dist = abs(how - entry_price) / entry_price * 100 if entry_price > 0 else 0
            if level >= 2:
                targets.append(TargetLevel(
                    price=how,
                    source="how",
                    level_num=level,
                    priority=3 if level == 3 else 4,
                    description="Previous High of Week",
                    distance_pct=round(dist, 2),
                ))

        if low and self._is_valid_target(low, direction, entry_price):
            dist = abs(low - entry_price) / entry_price * 100 if entry_price > 0 else 0
            if level >= 2:
                targets.append(TargetLevel(
                    price=low,
                    source="low",
                    level_num=level,
                    priority=3 if level == 3 else 4,
                    description="Previous Low of Week",
                    distance_pct=round(dist, 2),
                ))

        # Liquidation levels (front-run)
        if liquidation_levels:
            for liq_price in liquidation_levels:
                # Front-run by LIQUIDATION_FRONTRUN_PCT
                if direction == "bullish":
                    frontrun_price = liq_price * (1 - LIQUIDATION_FRONTRUN_PCT)
                else:
                    frontrun_price = liq_price * (1 + LIQUIDATION_FRONTRUN_PCT)

                if self._is_valid_target(frontrun_price, direction, entry_price):
                    dist = abs(frontrun_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                    targets.append(TargetLevel(
                        price=round(frontrun_price, 8),
                        source="liquidation",
                        level_num=level,
                        priority=2,
                        description=f"Liquidation cluster at {liq_price:.2f} (front-run)",
                        distance_pct=round(dist, 2),
                    ))

        # Sort by priority then distance
        targets.sort(key=lambda t: (t.priority, t.distance_pct))

        return targets

    @staticmethod
    def _is_valid_target(price: float, direction: str, entry_price: float) -> bool:
        """Check if a target price is in the right direction from entry."""
        if price <= 0:
            return False
        if direction == "bullish":
            return price > entry_price
        else:
            return price < entry_price

    @staticmethod
    def _pick_primary(
        targets: list[TargetLevel],
        direction: str,
        entry_price: float,
    ) -> TargetLevel | None:
        """Pick the best primary target from a level's target list.

        Prefers: EMA > Vector > HOW/LOW > Liquidation
        Among same source type, picks closest.
        """
        if not targets:
            return None

        # Already sorted by priority then distance
        return targets[0]

    @staticmethod
    def _build_profit_plan(
        direction: str,
        entry_price: float,
        l1_target: TargetLevel | None,
        l2_target: TargetLevel | None,
        l3_target: TargetLevel | None,
    ) -> PartialProfitPlan:
        """Build the partial profit schedule per the MM Method.

        Course rules:
        - Level 1 complete: take 25-33% (we use 30%)
        - Level 2 complete: take 50% total (so 20% more)
        - Level 3 SVC: take remaining (50%)
        """
        entries = []

        if l1_target:
            entries.append({
                "level": 1,
                "target_price": l1_target.price,
                "close_pct": PROFIT_SCHEDULE[1],
                "description": f"Level 1 → {l1_target.source} at {l1_target.price:.2f}: close 30%",
            })

        if l2_target:
            entries.append({
                "level": 2,
                "target_price": l2_target.price,
                "close_pct": PROFIT_SCHEDULE[2],
                "description": f"Level 2 → {l2_target.source} at {l2_target.price:.2f}: close to 50% total",
            })

        if l3_target:
            entries.append({
                "level": 3,
                "target_price": l3_target.price,
                "close_pct": PROFIT_SCHEDULE[3],
                "description": f"Level 3 → {l3_target.source} at {l3_target.price:.2f}: close remaining",
            })

        return PartialProfitPlan(entries=entries)
