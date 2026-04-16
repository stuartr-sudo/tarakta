"""Market Makers Method confluence scoring module.

Scores the confluence of MM Method factors for trade entries. The best
entries have maximum confluence across multiple factors from the MM
Method strategy.

Confluence factors are grouped by weight:

HIGH weight (core setup requirements):
  - M/W at session changeover (gap time formation):      20 pts
  - M/W at HOW/LOW or HOD/LOD (key level alignment):     15 pts
  - 50 EMA break with volume (Level 1 confirmation):     15 pts
  - Stopping Volume Candle present (Level 3 completion):  15 pts

MEDIUM weight (confirming factors):
  - Unrecovered Vector zone alignment:                     8 pts
  - Liquidation level cluster alignment:                   8 pts
  - EMA alignment (proper 10/20/50/200/800 order):         8 pts
  - Fibonacci level alignment (38.2/50/61.8%):             6 pts
  - News event timing (midweek reversal catalyst):         6 pts

LOW weight (bonus confluence):
  - Open Interest behavior (breakout vs fakeout):          4 pts
  - Correlation confirmation (DXY, NASDAQ divergence):     4 pts
  - Moon cycle alignment (full = bottom, new = top):       2 pts

Entry requirements:
  - M/W formation detected
  - At least 2 of 4 retest conditions met
  - R:R >= 3:1 (below 1.4:1 = "don't get out of bed")
  - Score >= configurable minimum (default 40)

Grading:
  A >= 70%  |  B >= 55%  |  C >= 40%  |  F < 40%
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Confluence factor weights
# ---------------------------------------------------------------------------
WEIGHTS: dict[str, float] = {
    # HIGH weight — core setup
    "mw_session_changeover": 20.0,
    "mw_key_level": 15.0,
    "ema50_break_volume": 15.0,
    "stopping_volume_candle": 15.0,
    # MEDIUM weight — confirming
    "unrecovered_vector": 8.0,
    "liquidation_cluster": 8.0,
    "ema_alignment": 8.0,
    # Course lesson 29 names Open Interest specifically as a trapped-trader
    # detector ("can also be used to identify trapped Traders"). Trapped
    # traders are the setup — promoted from LOW(4) to MEDIUM(8) in 2026-04
    # course-faithful redesign.
    "oi_behavior": 8.0,
    # Course lesson 15: "if you don't see the false breakout in your weekend
    # box, also look for W's and M's." A formation entirely inside the
    # weekend trap box is a SPECIFIC high-probability setup the course
    # singles out — treat as MEDIUM confluence (not HIGH because it's
    # conditional on the weekend cycle context).
    "mw_inside_weekend_box": 8.0,
    "fib_alignment": 6.0,
    "news_event": 6.0,
    # C2: RSI confirmation — divergence at formation OR trend bias aligned
    # with trade direction. 6 pts, same tier as fib and news.
    "rsi_confirmation": 6.0,
    # LOW weight — bonus
    "correlation_confirmed": 4.0,
    # C3: ADR (Average Daily Range) 50% line confluence. Low weight because
    # the course instructor notes ADR is "more of a Forex tool" and she
    # doesn't use it much personally. 4 pts when price is at the 50% ADR line.
    "adr_confluence": 4.0,
    "moon_cycle": 2.0,
}

MAX_POSSIBLE: float = sum(WEIGHTS.values())  # 111 → 123 (weekend-box, OI promotion) → 129 (+rsi) → 133 (+adr)

# Factors that require external data feeds (currently stubbed).
# When calculating grade thresholds, we use AVAILABLE_MAX instead of
# MAX_POSSIBLE so the bot isn't penalized for missing data it can't get.
STUBBED_FACTORS: set[str] = {"liquidation_cluster", "news_event", "correlation_confirmed"}
AVAILABLE_MAX: float = MAX_POSSIBLE - sum(WEIGHTS[k] for k in STUBBED_FACTORS)  # 133 - 18 = 115

# Grade thresholds (percentage of max possible)
GRADE_A_THRESHOLD: float = 70.0
GRADE_B_THRESHOLD: float = 55.0
GRADE_C_THRESHOLD: float = 40.0

# R:R thresholds
DONT_GET_OUT_OF_BED_RR: float = 1.4
DEFAULT_MIN_RR: float = 3.0

# Retest condition minimum
MIN_RETEST_CONDITIONS: int = 2


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class MMContext:
    """Input context assembled by the caller with all MM Method factors.

    The caller is responsible for populating these fields from scanner
    output, EMA framework results, session data, and market state.
    """

    # M/W formation data
    formation: dict | None = None  # type, variant, at_key_level, session info

    # EMA framework state
    ema_state: dict | None = None  # alignment, price distances, break detected

    # Level cycle state (Level 1/2/3 progression)
    level_state: dict | None = None  # current_level, svc_detected, volume_degrading

    # Weekly cycle state
    cycle_state: dict | None = None  # phase, direction, how, low, hod, lod

    # Trade price levels (required for R:R calculation)
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0

    # HIGH weight factor flags
    at_session_changeover: bool = False
    at_how_low: bool = False
    at_hod_lod: bool = False

    # MEDIUM weight factor flags
    has_unrecovered_vector: bool = False
    has_liquidation_cluster: bool = False
    has_fib_alignment: bool = False
    has_news_event: bool = False
    # Course lesson 15: M/W formation entirely inside the weekend trap box
    # (no FMWB spike out of the box). Caller sets this when it has both a
    # valid weekend box AND the formation's peaks are inside it.
    mw_inside_weekend_box: bool = False

    # C2: RSI confirmation (None = data not available)
    rsi_confirmed: bool | None = None

    # C3: ADR 50% line confluence (None = data not available)
    adr_at_fifty_pct: bool | None = None

    # LOW weight factor flags (None = data not available)
    oi_increasing: bool | None = None
    correlation_confirmed: bool | None = None
    moon_phase_aligned: bool | None = None


@dataclass
class RetestConditions:
    """The 4 retest conditions for M/W entry validation.

    At least 2 of these must be met before taking a trade. These are
    checked during the "board meeting" after Level 1.

    Conditions:
      1. Price at 50 EMA
      2. Price at vector that created Level 1
      3. Higher low (W formation) or lower high (M formation)
      4. Heat map / liquidation cluster at entry level
    """

    at_50_ema: bool = False
    at_level1_vector: bool = False
    higher_low_or_lower_high: bool = False
    at_liquidity_cluster: bool = False

    @property
    def conditions_met(self) -> int:
        """Count how many of the 4 retest conditions are True."""
        return sum([
            self.at_50_ema,
            self.at_level1_vector,
            self.higher_low_or_lower_high,
            self.at_liquidity_cluster,
        ])

    @property
    def sufficient(self) -> bool:
        """True if at least MIN_RETEST_CONDITIONS are met."""
        return self.conditions_met >= MIN_RETEST_CONDITIONS


@dataclass
class ConfluenceScore:
    """Result of scoring all MM Method confluence factors.

    Contains the total score, individual factor breakdown, R:R data,
    and a letter grade for quick assessment.
    """

    total_score: float = 0.0
    max_possible: float = MAX_POSSIBLE
    score_pct: float = 0.0  # total_score / max_possible * 100
    factors: dict[str, float] = field(default_factory=dict)
    risk_reward: float = 0.0
    meets_min_rr: bool = False
    meets_min_score: bool = False
    retest_conditions_met: int = 0
    grade: str = "F"  # A / B / C / F


@dataclass
class EntryDecision:
    """Final go/no-go decision for an MM Method entry.

    Includes the reason for the decision and whether it qualifies as
    aggressive (fewer conditions met) or conservative (strong confluence).
    """

    should_enter: bool = False
    reason: str = ""
    score: ConfluenceScore = field(default_factory=ConfluenceScore)
    entry_type: str = "conservative"  # "aggressive" or "conservative"


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------
class MMConfluenceScorer:
    """Scores confluence of MM Method factors for trade entry decisions.

    The scorer evaluates all 12 confluence factors across HIGH, MEDIUM,
    and LOW weight tiers, validates the 4 retest conditions, and computes
    the risk-to-reward ratio to produce a final entry decision.

    Usage::

        scorer = MMConfluenceScorer(min_rr=3.0, min_score=40.0)
        ctx = MMContext(
            entry_price=95000.0,
            stop_loss=96500.0,
            target_price=90500.0,
            at_session_changeover=True,
            ...
        )
        score = scorer.score(ctx)
        decision = scorer.should_enter(score)
    """

    def __init__(self, min_rr: float = DEFAULT_MIN_RR, min_score: float = 40.0) -> None:
        self.min_rr = min_rr
        self.min_score = min_score
        self._weights: dict[str, float] = dict(WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, context: MMContext) -> ConfluenceScore:
        """Score all confluence factors from the provided context.

        Evaluates each of the 12 factors, computes the R:R, checks retest
        conditions, and assigns a letter grade.

        Args:
            context: Fully populated MMContext with formation data, price
                     levels, and factor flags.

        Returns:
            ConfluenceScore with full breakdown.
        """
        factors: dict[str, float] = {}

        # --- HIGH weight factors ---
        factors["mw_session_changeover"] = self._score_session_changeover(context)
        factors["mw_key_level"] = self._score_key_level(context)
        factors["ema50_break_volume"] = self._score_ema50_break(context)
        factors["stopping_volume_candle"] = self._score_stopping_volume(context)

        # --- MEDIUM weight factors ---
        factors["unrecovered_vector"] = self._score_unrecovered_vector(context)
        factors["liquidation_cluster"] = self._score_liquidation_cluster(context)
        factors["ema_alignment"] = self._score_ema_alignment(context)
        factors["mw_inside_weekend_box"] = self._score_mw_inside_weekend_box(context)
        factors["fib_alignment"] = self._score_fib_alignment(context)
        factors["news_event"] = self._score_news_event(context)

        # --- MEDIUM weight factors (continued) ---
        factors["rsi_confirmation"] = self._score_rsi_confirmation(context)

        # --- LOW weight factors ---
        factors["oi_behavior"] = self._score_oi_behavior(context)
        factors["correlation_confirmed"] = self._score_correlation(context)
        factors["adr_confluence"] = self._score_adr_confluence(context)
        factors["moon_cycle"] = self._score_moon_cycle(context)

        total = sum(factors.values())
        # Use AVAILABLE_MAX (excludes stubbed data feeds) so the bot isn't
        # penalized for missing external data it can't access yet.
        effective_max = AVAILABLE_MAX if AVAILABLE_MAX > 0 else MAX_POSSIBLE
        score_pct = (total / effective_max * 100.0) if effective_max > 0 else 0.0

        # R:R
        rr = self.calculate_rr(context.entry_price, context.stop_loss, context.target_price)

        # Retest conditions
        retest = self.check_retest_conditions(context)

        # Grade
        grade = self._compute_grade(score_pct)

        result = ConfluenceScore(
            total_score=round(total, 2),
            max_possible=effective_max,
            score_pct=round(score_pct, 2),
            factors=factors,
            risk_reward=round(rr, 2),
            meets_min_rr=rr >= self.min_rr,
            meets_min_score=total >= self.min_score,
            retest_conditions_met=retest.conditions_met,
            grade=grade,
        )

        logger.info(
            "MM confluence score: %.1f/%.0f (%.1f%%) grade=%s R:R=%.2f retest=%d/4",
            result.total_score,
            result.max_possible,
            result.score_pct,
            result.grade,
            result.risk_reward,
            result.retest_conditions_met,
        )
        logger.debug("Factor breakdown: %s", factors)

        return result

    def calculate_rr(self, entry: float, stop_loss: float, target: float) -> float:
        """Calculate risk-to-reward ratio.

        For M (short) formations: entry < stop_loss (stop above), target < entry.
        For W (long) formations: entry > stop_loss (stop below), target > entry.

        The calculation uses absolute distances so it works for both
        long and short setups.

        Args:
            entry:     Entry price (2nd peak of M/W).
            stop_loss: Stop loss price (above 1st peak wick for M,
                       below 1st peak wick for W).
            target:    Target price (first unrecovered vector candle).

        Returns:
            Risk-to-reward ratio as a float. Returns 0.0 if risk is zero
            or prices are invalid.
        """
        risk = abs(entry - stop_loss)
        reward = abs(target - entry)

        if risk <= 0:
            logger.warning("R:R calculation: risk is zero (entry=%.2f, sl=%.2f)", entry, stop_loss)
            return 0.0

        rr = reward / risk
        logger.debug(
            "R:R calculation: entry=%.2f sl=%.2f target=%.2f risk=%.2f reward=%.2f R:R=%.2f",
            entry, stop_loss, target, risk, reward, rr,
        )
        return rr

    def check_retest_conditions(self, context: MMContext) -> RetestConditions:
        """Check the 4 retest conditions for M/W entry validation.

        The conditions are derived from the context's EMA state, level
        state, formation data, and liquidation cluster presence.

        Args:
            context: MMContext with ema_state, level_state, formation,
                     and liquidation cluster data.

        Returns:
            RetestConditions with each condition evaluated.
        """
        # Condition 1: Price at 50 EMA
        at_50_ema = False
        if context.ema_state and context.ema_state.get("at_50_ema"):
            at_50_ema = True

        # Condition 2: Price at vector that created Level 1
        at_level1_vector = False
        if context.level_state and context.level_state.get("at_level1_vector"):
            at_level1_vector = True
        elif context.has_unrecovered_vector:
            # If we know there's an unrecovered vector and the level state
            # confirms price is near it, this counts
            at_level1_vector = True

        # Condition 3: Higher low (W) or lower high (M)
        higher_low_or_lower_high = False
        if context.formation:
            form_type = context.formation.get("type", "")
            has_hl_lh = context.formation.get("higher_low_or_lower_high", False)
            if has_hl_lh:
                higher_low_or_lower_high = True
            elif form_type == "W" and context.formation.get("higher_low"):
                higher_low_or_lower_high = True
            elif form_type == "M" and context.formation.get("lower_high"):
                higher_low_or_lower_high = True

        # Condition 4: Heat map / liquidation cluster at entry level
        at_liquidity_cluster = context.has_liquidation_cluster

        retest = RetestConditions(
            at_50_ema=at_50_ema,
            at_level1_vector=at_level1_vector,
            higher_low_or_lower_high=higher_low_or_lower_high,
            at_liquidity_cluster=at_liquidity_cluster,
        )

        logger.debug(
            "Retest conditions: 50ema=%s vector=%s hl_lh=%s liq=%s => %d/4 (sufficient=%s)",
            at_50_ema, at_level1_vector, higher_low_or_lower_high,
            at_liquidity_cluster, retest.conditions_met, retest.sufficient,
        )

        return retest

    def should_enter(self, score: ConfluenceScore) -> EntryDecision:
        """Make the final go/no-go entry decision.

        Entry requires ALL of:
          1. Score >= min_score
          2. R:R >= min_rr
          3. At least 2 of 4 retest conditions met

        The entry is classified as "aggressive" if exactly 2 retest
        conditions are met or the score is below grade B. "Conservative"
        entries have 3+ retest conditions and grade B or above.

        Args:
            score: ConfluenceScore from the score() method.

        Returns:
            EntryDecision with should_enter, reason, and entry_type.
        """
        reasons: list[str] = []

        # Check M/W formation presence (implied by retest conditions > 0
        # or a non-zero formation-related score)
        has_formation = (
            score.factors.get("mw_session_changeover", 0) > 0
            or score.factors.get("mw_key_level", 0) > 0
            or score.retest_conditions_met > 0
        )

        if not has_formation:
            return EntryDecision(
                should_enter=False,
                reason="No M/W formation detected — MM Method requires a formation",
                score=score,
                entry_type="conservative",
            )

        # R:R gate
        if score.risk_reward <= DONT_GET_OUT_OF_BED_RR:
            return EntryDecision(
                should_enter=False,
                reason=(
                    f"R:R {score.risk_reward:.2f} is at or below {DONT_GET_OUT_OF_BED_RR} "
                    f"— don't get out of bed"
                ),
                score=score,
                entry_type="conservative",
            )

        if not score.meets_min_rr:
            reasons.append(
                f"R:R {score.risk_reward:.2f} below minimum {self.min_rr:.1f}"
            )

        # Score gate
        if not score.meets_min_score:
            reasons.append(
                f"Score {score.total_score:.1f} below minimum {self.min_score:.1f}"
            )

        # Retest conditions gate
        if score.retest_conditions_met < MIN_RETEST_CONDITIONS:
            reasons.append(
                f"Only {score.retest_conditions_met}/4 retest conditions met "
                f"(need >= {MIN_RETEST_CONDITIONS})"
            )

        if reasons:
            combined = "; ".join(reasons)
            logger.info("MM entry rejected: %s", combined)
            return EntryDecision(
                should_enter=False,
                reason=combined,
                score=score,
                entry_type="conservative",
            )

        # Determine entry type
        is_aggressive = (
            score.retest_conditions_met == MIN_RETEST_CONDITIONS
            or score.grade not in ("A", "B")
        )
        entry_type = "aggressive" if is_aggressive else "conservative"

        reason = (
            f"ENTER ({entry_type}): grade {score.grade} "
            f"score {score.total_score:.1f}/{score.max_possible:.0f} "
            f"({score.score_pct:.1f}%) R:R {score.risk_reward:.2f} "
            f"retest {score.retest_conditions_met}/4"
        )
        logger.info("MM entry approved: %s", reason)

        return EntryDecision(
            should_enter=True,
            reason=reason,
            score=score,
            entry_type=entry_type,
        )

    # ------------------------------------------------------------------
    # Factor scoring (private)
    # ------------------------------------------------------------------

    def _score_session_changeover(self, ctx: MMContext) -> float:
        """M/W at session changeover (gap time formation, especially multi-session).

        The highest-probability MM setups form during session handovers
        (gap times) when one market maker session ends and another begins.
        Multi-session formations (e.g., Asia gap into UK open) are ideal.
        """
        if not ctx.at_session_changeover:
            return 0.0

        weight = self._weights["mw_session_changeover"]

        # Check for multi-session formation (bonus — gets full weight)
        if ctx.formation and ctx.formation.get("multi_session"):
            logger.debug("Session changeover: multi-session formation — full %s pts", weight)
            return weight

        # Single-session changeover still scores, but at 75% weight
        reduced = round(weight * 0.75, 2)
        logger.debug("Session changeover: single-session — %s pts", reduced)
        return reduced

    def _score_key_level(self, ctx: MMContext) -> float:
        """M/W at HOW/LOW or HOD/LOD (key level alignment).

        Formations at the high/low of the week or high/low of the day
        carry extra weight because these are the levels MMs target.
        HOW/LOW is stronger than HOD/LOD.
        """
        weight = self._weights["mw_key_level"]

        if ctx.at_how_low:
            logger.debug("Key level: at HOW/LOW — full %s pts", weight)
            return weight
        if ctx.at_hod_lod:
            reduced = round(weight * 0.75, 2)
            logger.debug("Key level: at HOD/LOD — %s pts", reduced)
            return reduced

        return 0.0

    def _score_ema50_break(self, ctx: MMContext) -> float:
        """50 EMA break with volume (Level 1 confirmation).

        Level 1 is confirmed when price breaks the 50 EMA with
        above-average volume. Without volume, the break is suspect.
        """
        if not ctx.ema_state:
            return 0.0

        broke_50 = ctx.ema_state.get("broke_50", False)
        volume_confirmed = ctx.ema_state.get("volume_confirmed", False)

        if not broke_50:
            return 0.0

        weight = self._weights["ema50_break_volume"]

        if volume_confirmed:
            logger.debug("50 EMA break: volume confirmed — full %s pts", weight)
            return weight

        # Break without volume is weaker — 50% credit
        reduced = round(weight * 0.5, 2)
        logger.debug("50 EMA break: NO volume confirmation — %s pts", reduced)
        return reduced

    def _score_stopping_volume(self, ctx: MMContext) -> float:
        """Stopping Volume Candle present (Level 3 completion).

        A stopping volume candle (high volume with small body, often a
        hammer/doji) signals exhaustion and Level 3 completion. This
        marks the transition point where MMs are finishing their move.
        """
        if not ctx.level_state:
            return 0.0

        svc = ctx.level_state.get("svc_detected", False)
        if not svc:
            return 0.0

        weight = self._weights["stopping_volume_candle"]

        # Extra confidence if volume is degrading into the SVC
        if ctx.level_state.get("volume_degrading", False):
            logger.debug("Stopping volume candle: with degrading volume — full %s pts", weight)
            return weight

        # SVC without degrading volume — still strong, 85% credit
        reduced = round(weight * 0.85, 2)
        logger.debug("Stopping volume candle: no degrading volume — %s pts", reduced)
        return reduced

    def _score_unrecovered_vector(self, ctx: MMContext) -> float:
        """Unrecovered Vector zone alignment.

        Vector candles (PVSRA 200%+ volume) that haven't been revisited
        act as magnets. Price aligned with an unrecovered vector zone
        gives extra confluence.
        """
        if not ctx.has_unrecovered_vector:
            return 0.0

        weight = self._weights["unrecovered_vector"]
        logger.debug("Unrecovered vector alignment — %s pts", weight)
        return weight

    def _score_liquidation_cluster(self, ctx: MMContext) -> float:
        """Liquidation level cluster alignment.

        When the entry is near a cluster of liquidation levels (from
        the heatmap), MMs are likely targeting that area for liquidity.
        """
        if not ctx.has_liquidation_cluster:
            return 0.0

        weight = self._weights["liquidation_cluster"]
        logger.debug("Liquidation cluster alignment — %s pts", weight)
        return weight

    def _score_ema_alignment(self, ctx: MMContext) -> float:
        """EMA alignment (proper 10/20/50/200/800 order).

        Perfect EMA stacking (all EMAs in order for the trend direction)
        confirms a clean trend. Mixed alignment reduces the score.
        """
        if not ctx.ema_state:
            return 0.0

        alignment = ctx.ema_state.get("alignment", "mixed")
        weight = self._weights["ema_alignment"]

        if alignment in ("bullish", "bearish"):
            logger.debug("EMA alignment: %s — full %s pts", alignment, weight)
            return weight

        # Mixed alignment — partial credit if at least mostly aligned
        if alignment == "partial":
            reduced = round(weight * 0.5, 2)
            logger.debug("EMA alignment: partial — %s pts", reduced)
            return reduced

        return 0.0

    def _score_fib_alignment(self, ctx: MMContext) -> float:
        """Fibonacci level alignment (38.2/50/61.8% in board meeting).

        During the board meeting (retracement after Level 1), price
        should retrace to a key Fibonacci level. The 61.8% is ideal,
        50% is good, 38.2% is acceptable.
        """
        if not ctx.has_fib_alignment:
            return 0.0

        weight = self._weights["fib_alignment"]
        logger.debug("Fibonacci alignment — %s pts", weight)
        return weight

    def _score_mw_inside_weekend_box(self, ctx: MMContext) -> float:
        """Course lesson 15: "if you don't see the false breakout in your
        weekend box, also look for W's and M's." A formation entirely
        inside the weekend trap box — when FMWB hasn't fired — is one of
        the course's named high-probability setups. Scored on an
        explicit flag supplied by the caller (engine looks at formation
        peaks + weekend box bounds).
        """
        if not getattr(ctx, "mw_inside_weekend_box", False):
            return 0.0
        weight = self._weights["mw_inside_weekend_box"]
        logger.debug("M/W inside weekend box — %s pts", weight)
        return weight

    def _score_news_event(self, ctx: MMContext) -> float:
        """News event timing (midweek reversal catalyst).

        MM Method often uses midweek news events (FOMC, CPI, etc.) as
        catalysts for reversals. If a qualifying news event aligns with
        the formation timing, it's additional confluence.
        """
        if not ctx.has_news_event:
            return 0.0

        weight = self._weights["news_event"]
        logger.debug("News event timing — %s pts", weight)
        return weight

    def _score_rsi_confirmation(self, ctx: MMContext) -> float:
        """RSI confirmation (C2) — divergence at formation OR trend bias aligned.

        Scores 6.0 when rsi_confirmed is True, 0.0 when False or None.
        Caller sets rsi_confirmed=True if:
          - RSI divergence is detected at the M/W formation, OR
          - RSI trend_bias aligns with the trade direction.
        """
        if ctx.rsi_confirmed is None or not ctx.rsi_confirmed:
            return 0.0

        weight = self._weights["rsi_confirmation"]
        logger.debug("RSI confirmation — %s pts", weight)
        return weight

    def _score_oi_behavior(self, ctx: MMContext) -> float:
        """Open Interest behavior (breakout vs fakeout signal).

        Rising OI on a move = real conviction. Falling OI on a move =
        likely a fakeout (positions closing, not opening). None means
        OI data was unavailable.
        """
        if ctx.oi_increasing is None:
            return 0.0

        weight = self._weights["oi_behavior"]

        if ctx.oi_increasing:
            logger.debug("OI increasing (conviction) — %s pts", weight)
            return weight

        return 0.0

    def _score_correlation(self, ctx: MMContext) -> float:
        """Correlation confirmation (DXY, NASDAQ divergence).

        When BTC diverges from correlated assets (e.g., DXY inverse,
        NASDAQ positive), it can signal hidden strength or weakness
        that confirms the MM setup direction.
        """
        if ctx.correlation_confirmed is None:
            return 0.0

        weight = self._weights["correlation_confirmed"]

        if ctx.correlation_confirmed:
            logger.debug("Correlation confirmed — %s pts", weight)
            return weight

        return 0.0

    def _score_adr_confluence(self, ctx: MMContext) -> float:
        """ADR 50% line confluence (C3) — price near the midpoint of the day's range.

        Scores 4.0 when adr_at_fifty_pct is True, 0.0 when False or None.
        The 50% ADR line = current day's low + 0.5 * ADR14. Price at this
        level is considered "cheap" for longs and "expensive" for shorts.
        None means ADR data was unavailable.
        """
        if ctx.adr_at_fifty_pct is None or not ctx.adr_at_fifty_pct:
            return 0.0

        weight = self._weights["adr_confluence"]
        logger.debug("ADR 50%% line confluence — %s pts", weight)
        return weight

    def _score_moon_cycle(self, ctx: MMContext) -> float:
        """Moon cycle alignment (full moon = bottom, new moon = top, +/-3 days).

        Statistical quirk in crypto: full moons tend to align with local
        bottoms, new moons with local tops. Low weight because it's a
        soft signal, but worth tracking as additional confluence.
        """
        if ctx.moon_phase_aligned is None:
            return 0.0

        weight = self._weights["moon_cycle"]

        if ctx.moon_phase_aligned:
            logger.debug("Moon cycle aligned — %s pts", weight)
            return weight

        return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_grade(score_pct: float) -> str:
        """Map score percentage to a letter grade.

        A >= 70% | B >= 55% | C >= 40% | F < 40%
        """
        if score_pct >= GRADE_A_THRESHOLD:
            return "A"
        if score_pct >= GRADE_B_THRESHOLD:
            return "B"
        if score_pct >= GRADE_C_THRESHOLD:
            return "C"
        return "F"
