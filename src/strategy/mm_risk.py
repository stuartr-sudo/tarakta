"""MM Method risk management module.

Implements the position sizing, leverage calculation, and risk rules from
the Market Makers Method course.

Key rules:
- Risk per trade: 1% of total account (all exchanges combined)
- Position Size = Risk Amount / Stop Loss Distance %
- Leverage = capital freeing tool, NOT profit multiplier
- Dollar profit identical regardless of leverage (only ROI% changes)
- Probably don't need past 10x even with 2-3 simultaneous trades
- Large accounts: keep only 10% on exchange, use leverage to cover size
- Judge performance in batches of 10 trades, not individually
- Never tighten SL to improve R:R
- R:R calculated to Level 1 target ONLY — beyond is bonus
- Minimum R:R thresholds determine whether to take a trade

Win rate / R:R relationship:
  1:1 → need 6/10 wins
  2:1 → need 4/10 wins
  3:1 → need 3/10 wins
  4:1 → need 2.5/10 wins
  5:1 → need 2/10 wins
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Perpetual funding fee schedule
FUNDING_INTERVAL_HOURS = 8  # Perpetual funding every 8 hours
FUNDING_TIMES_UTC = [0, 8, 16]  # UTC hours when funding is charged

# Default risk per trade (as fraction of total account)
DEFAULT_RISK_PER_TRADE = 0.01  # 1%

# Minimum R:R thresholds
MIN_RR_ABSOLUTE = 1.4    # Below this: "don't get out of bed"
MIN_RR_STANDARD = 3.0    # Standard minimum for MM Method
MIN_RR_AGGRESSIVE = 2.0  # For aggressive entries on 2nd M/W peak

# Maximum recommended leverage
MAX_RECOMMENDED_LEVERAGE = 10

# Performance evaluation batch size
EVAL_BATCH_SIZE = 10

# Win rate required per R:R (for reference)
WIN_RATE_TABLE = {
    1.0: 0.60,   # 6/10
    2.0: 0.40,   # 4/10
    3.0: 0.30,   # 3/10
    4.0: 0.25,   # 2.5/10
    5.0: 0.20,   # 2/10
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    position_size_usd: float = 0.0  # Total position size in USD
    risk_amount_usd: float = 0.0    # Dollar amount at risk
    stop_loss_distance_pct: float = 0.0  # SL distance as % of entry
    leverage_needed: float = 1.0    # Leverage needed if using all margin
    margin_required_usd: float = 0.0  # Margin at 1x
    recommended_leverage: float = 1.0  # Practical recommendation
    is_viable: bool = True          # False if position too large for account


@dataclass
class RiskAssessment:
    """Risk assessment for a potential trade."""
    risk_reward: float = 0.0        # R:R ratio (using Level 1 target)
    meets_minimum: bool = False     # Does it meet MIN_RR_STANDARD?
    meets_absolute_minimum: bool = False  # Does it meet MIN_RR_ABSOLUTE?
    entry_type: str = ""            # "aggressive" or "conservative"
    required_win_rate: float = 0.0  # Win rate needed at this R:R
    recommendation: str = ""        # "take", "caution", "skip"
    reason: str = ""


@dataclass
class RefundZoneCheck:
    """Refund Zone validation result.

    The Refund Zone applies when entering on the 2nd peak of an M/W.
    If price CLOSES below the wick of the 2nd peak W (or above for M),
    the formation is invalidated → cut immediately for a tiny loss.
    """
    is_in_refund_zone: bool = False
    should_cut: bool = False
    loss_pct: float = 0.0           # Current loss as %
    peak_wick_price: float = 0.0    # The wick price that defines the zone
    invalidation_price: float = 0.0 # Close beyond this = cut


@dataclass
class BatchPerformance:
    """Performance evaluation over a batch of trades.

    MM Method says to judge in batches of 10, not individually.
    """
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_rr: float = 0.0
    total_r: float = 0.0          # Total R earned (wins * RR - losses * 1)
    is_profitable: bool = False
    expected_r_per_trade: float = 0.0
    assessment: str = ""


# ---------------------------------------------------------------------------
# Risk Calculator
# ---------------------------------------------------------------------------

class MMRiskCalculator:
    """Position sizing and risk management per MM Method rules."""

    def __init__(
        self,
        risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
        max_leverage: float = MAX_RECOMMENDED_LEVERAGE,
    ):
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage

    def calculate_position_size(
        self,
        account_balance_usd: float,
        entry_price: float,
        stop_loss_price: float,
        exchange_balance_usd: float | None = None,
    ) -> PositionSizeResult:
        """Calculate position size per the MM Method.

        Formula: Position Size = Risk Amount / Stop Loss Distance %

        Args:
            account_balance_usd: Total account balance across all exchanges.
            entry_price: Intended entry price.
            stop_loss_price: Stop loss price.
            exchange_balance_usd: Balance on the specific exchange (for leverage calc).
                                  Defaults to account_balance_usd if not specified.

        Returns:
            PositionSizeResult with size, leverage, and viability.
        """
        if account_balance_usd <= 0 or entry_price <= 0:
            return PositionSizeResult(is_viable=False)

        if exchange_balance_usd is None:
            exchange_balance_usd = account_balance_usd

        # Risk amount: 1% of total account
        risk_amount = account_balance_usd * self.risk_per_trade

        # SL distance as percentage
        sl_distance = abs(entry_price - stop_loss_price)
        sl_distance_pct = sl_distance / entry_price

        if sl_distance_pct <= 0:
            return PositionSizeResult(is_viable=False)

        # Position size
        position_size = risk_amount / sl_distance_pct

        # Leverage calculation
        # How much leverage needed if using entire exchange balance
        leverage_needed = position_size / exchange_balance_usd if exchange_balance_usd > 0 else float("inf")

        # Recommended leverage (cap at max)
        recommended_leverage = min(
            max(1.0, round(leverage_needed, 1)),
            self.max_leverage,
        )

        # Check viability
        is_viable = leverage_needed <= self.max_leverage

        # Margin required at 1x
        margin_required = position_size

        result = PositionSizeResult(
            position_size_usd=round(position_size, 2),
            risk_amount_usd=round(risk_amount, 2),
            stop_loss_distance_pct=round(sl_distance_pct * 100, 4),
            leverage_needed=round(leverage_needed, 2),
            margin_required_usd=round(margin_required, 2),
            recommended_leverage=recommended_leverage,
            is_viable=is_viable,
        )

        logger.info(
            "mm_position_size_calculated",
            account_balance=account_balance_usd,
            risk_amount=round(risk_amount, 2),
            position_size=round(position_size, 2),
            sl_distance_pct=round(sl_distance_pct * 100, 4),
            leverage_needed=round(leverage_needed, 2),
            is_viable=is_viable,
        )

        return result

    def assess_risk(
        self,
        entry_price: float,
        stop_loss: float,
        target_l1: float,
        entry_type: str = "conservative",
    ) -> RiskAssessment:
        """Assess whether a trade meets MM Method risk criteria.

        R:R is ALWAYS calculated to Level 1 target — beyond is bonus.

        Args:
            entry_price: Entry price.
            stop_loss: Stop loss price.
            target_l1: Level 1 target price.
            entry_type: "aggressive" (on 2nd M/W peak) or "conservative" (after Level 1).

        Returns:
            RiskAssessment with recommendation.
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(target_l1 - entry_price)

        if risk <= 0:
            return RiskAssessment(
                recommendation="skip",
                reason="Invalid: zero risk distance",
            )

        rr = round(reward / risk, 2)

        # Minimum R:R check
        if entry_type == "aggressive":
            min_rr = MIN_RR_AGGRESSIVE
        else:
            min_rr = MIN_RR_STANDARD

        meets_standard = rr >= min_rr
        meets_absolute = rr >= MIN_RR_ABSOLUTE

        # Win rate needed at this R:R
        required_win_rate = self._interpolate_win_rate(rr)

        # Recommendation
        if rr < MIN_RR_ABSOLUTE:
            recommendation = "skip"
            reason = f"R:R {rr:.1f} below absolute minimum {MIN_RR_ABSOLUTE:.1f} — don't get out of bed"
        elif rr < min_rr:
            recommendation = "caution"
            reason = f"R:R {rr:.1f} below standard {min_rr:.1f} for {entry_type} entry — marginal"
        elif rr >= 5.0:
            recommendation = "take"
            reason = f"R:R {rr:.1f} — excellent setup, only need {required_win_rate * 100:.0f}% win rate"
        elif rr >= min_rr:
            recommendation = "take"
            reason = f"R:R {rr:.1f} — meets {entry_type} minimum ({min_rr:.1f}), need {required_win_rate * 100:.0f}% win rate"
        else:
            recommendation = "skip"
            reason = f"R:R {rr:.1f} insufficient"

        return RiskAssessment(
            risk_reward=rr,
            meets_minimum=meets_standard,
            meets_absolute_minimum=meets_absolute,
            entry_type=entry_type,
            required_win_rate=round(required_win_rate, 2),
            recommendation=recommendation,
            reason=reason,
        )

    def check_refund_zone(
        self,
        entry_price: float,
        current_price: float,
        formation_type: str,
        peak2_wick_price: float,
    ) -> RefundZoneCheck:
        """Check if a trade entered on 2nd M/W peak is in the Refund Zone.

        The Refund Zone = the invalidation zone for aggressive entries.
        If price CLOSES beyond the 2nd peak's wick:
          - W formation: price closes below the low wick of 2nd peak → cut
          - M formation: price closes above the high wick of 2nd peak → cut

        Args:
            entry_price: Entry price.
            current_price: Current candle CLOSE (must be close, not just price).
            formation_type: "W" or "M".
            peak2_wick_price: The wick extreme of the 2nd peak.

        Returns:
            RefundZoneCheck with cut decision.
        """
        loss_pct = 0.0
        if entry_price > 0:
            loss_pct = ((current_price - entry_price) / entry_price) * 100

        if formation_type.upper() == "W":
            # W = bullish. 2nd peak wick is at the bottom.
            # If close goes below the wick → invalidated
            should_cut = current_price < peak2_wick_price
            invalidation_price = peak2_wick_price
        elif formation_type.upper() == "M":
            # M = bearish. 2nd peak wick is at the top.
            # If close goes above the wick → invalidated
            should_cut = current_price > peak2_wick_price
            invalidation_price = peak2_wick_price
        else:
            return RefundZoneCheck()

        return RefundZoneCheck(
            is_in_refund_zone=True,
            should_cut=should_cut,
            loss_pct=round(abs(loss_pct), 4),
            peak_wick_price=peak2_wick_price,
            invalidation_price=invalidation_price,
        )

    def evaluate_batch(
        self,
        trades: list[dict],
    ) -> BatchPerformance:
        """Evaluate trading performance over a batch of trades.

        MM Method says: judge in batches of 10, not individually.

        Args:
            trades: List of trade dicts with at least 'pnl_usd' and 'risk_usd' keys.
                    Optional: 'rr_achieved' (actual R:R of the trade).

        Returns:
            BatchPerformance assessment.
        """
        if not trades:
            return BatchPerformance()

        wins = 0
        losses = 0
        total_r = 0.0
        rr_values = []

        for trade in trades:
            pnl = trade.get("pnl_usd", 0) or 0
            risk = trade.get("risk_usd", 0) or 0
            rr_achieved = trade.get("rr_achieved")

            if pnl > 0:
                wins += 1
                if rr_achieved:
                    total_r += rr_achieved
                    rr_values.append(rr_achieved)
                elif risk > 0:
                    rr = pnl / risk
                    total_r += rr
                    rr_values.append(rr)
                else:
                    total_r += 1  # Assume 1:1 if no data
                    rr_values.append(1.0)
            else:
                losses += 1
                total_r -= 1  # Each loss costs 1R

        total = len(trades)
        win_rate = wins / total if total > 0 else 0
        avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0
        expected_r = total_r / total if total > 0 else 0

        # Assessment
        if total < EVAL_BATCH_SIZE:
            assessment = f"Incomplete batch ({total}/{EVAL_BATCH_SIZE}) — too early to judge"
        elif total_r > 0:
            assessment = f"Profitable: {total_r:.1f}R over {total} trades ({expected_r:.2f}R/trade)"
        elif total_r == 0:
            assessment = f"Breakeven: 0R over {total} trades"
        else:
            assessment = f"Losing: {total_r:.1f}R over {total} trades — review entries and SL placement"

        return BatchPerformance(
            total_trades=total,
            wins=wins,
            losses=losses,
            win_rate=round(win_rate, 2),
            avg_rr=round(avg_rr, 2),
            total_r=round(total_r, 2),
            is_profitable=total_r > 0,
            expected_r_per_trade=round(expected_r, 3),
            assessment=assessment,
        )

    def check_funding_fee_proximity(self, now: datetime | None = None) -> dict:
        """Check proximity to next perpetual funding fee.

        Course (Scalp Lesson 06): fees charged on entire position size.
        Be aware of funding timing when scalp trading.

        Returns: {minutes_to_next: float, is_near: bool, next_time: str}
        """
        if now is None:
            now = datetime.now(timezone.utc)

        current_hour = now.hour
        current_min = now.minute

        # Find next funding time
        for ft in FUNDING_TIMES_UTC:
            if ft > current_hour or (ft == current_hour and current_min == 0):
                minutes_to = (ft - current_hour) * 60 - current_min
                return {
                    "minutes_to_next": float(minutes_to),
                    "is_near": minutes_to <= 30,
                    "next_time": f"{ft:02d}:00 UTC",
                }

        # Wrap to next day
        minutes_to = (24 - current_hour + FUNDING_TIMES_UTC[0]) * 60 - current_min
        return {
            "minutes_to_next": float(minutes_to),
            "is_near": minutes_to <= 30,
            "next_time": f"{FUNDING_TIMES_UTC[0]:02d}:00 UTC",
        }

    @staticmethod
    def _interpolate_win_rate(rr: float) -> float:
        """Interpolate required win rate for a given R:R.

        Uses the MM Method win rate table:
          1:1 → 60%, 2:1 → 40%, 3:1 → 30%, 4:1 → 25%, 5:1 → 20%
        """
        if rr <= 0:
            return 1.0
        if rr >= 5.0:
            return 0.20

        # Linear interpolation between table entries
        breakpoints = sorted(WIN_RATE_TABLE.keys())
        for i in range(len(breakpoints) - 1):
            low_rr = breakpoints[i]
            high_rr = breakpoints[i + 1]
            if low_rr <= rr <= high_rr:
                low_wr = WIN_RATE_TABLE[low_rr]
                high_wr = WIN_RATE_TABLE[high_rr]
                frac = (rr - low_rr) / (high_rr - low_rr)
                return low_wr + (high_wr - low_wr) * frac

        return 0.50  # Fallback
