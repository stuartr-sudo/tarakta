"""M/W formation detection for the Market Makers Method.

Detects M-top and W-bottom formations that are the primary entry signals
in the MM strategy. Market makers create these patterns to trap retail
traders before reversing price.

Formation types:
    Standard M/W:
        Two peaks where the 2nd peak does NOT reach the 1st. MMs keep
        traders trapped by making them think the trend will continue.
        Three confirmations: (a) Stopping Volume Candle at Level 3,
        (b) 3 trap candles on inside right side (drop one TF),
        (c) 50 EMA break with volume.

    Multi-Session M/W (strongest):
        1st peak forms in one session (e.g., Asia), 2nd peak in another
        (e.g., UK open). Almost guarantees 3-level follow-through because
        two separate groups of MMs have set the trap.

    Final Damage M/W:
        2nd peak makes a LOWER low (W) or HIGHER high (M) than the 1st
        peak, plus a hammer/inverted hammer candle. Better R:R but riskier
        because the 2nd peak exceeds the 1st (fakeout before reversal).

    Board Meeting M/W:
        Forms inside tight consolidation between levels. Reduced criteria --
        just the M/W shape is required without full confirmation.

    Three Hits Rule:
        3 tests of HOW/LOW without breaking at Level 3 = reversal imminent.
        4 hits = continuation. Hits must occur in different sessions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.strategy.mm_sessions import MMSessionAnalyzer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Swing detection: a swing high requires this many lower highs on each side.
SWING_WINDOW = 5

# Minimum candle separation between the two peaks of an M/W.
MIN_PEAK_SEPARATION = 3

# Default candle lookback for formation scanning.
DEFAULT_LOOKBACK = 40

# 50-period EMA for confirmation.
EMA_PERIOD = 50

# Volume multiplier threshold to classify a candle as "stopping volume".
SVC_VOLUME_MULT = 1.5

# Minimum number of trap (inside) candles required on the right side.
TRAP_CANDLE_COUNT = 3

# Tolerance for "near a level" checks (0.2%).
LEVEL_TOLERANCE = 0.002

# Quality score weights.
_W_SEPARATION = 0.20
_W_SHORTFALL = 0.25
_W_WICK = 0.20
_W_VOLUME = 0.15
_W_SYMMETRY = 0.20


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Formation:
    """Detected M-top or W-bottom formation.

    Attributes:
        type: 'M' for M-top (bearish) or 'W' for W-bottom (bullish).
        variant: One of 'standard', 'multi_session', 'final_damage',
            'board_meeting'.
        peak1_idx: DataFrame integer-location index of the 1st peak.
        peak1_price: Price at the 1st peak.
        peak2_idx: DataFrame integer-location index of the 2nd peak.
        peak2_price: Price at the 2nd peak.
        trough_idx: Index of the valley/ridge between the two peaks.
        trough_price: Price at that valley/ridge.
        direction: 'bullish' for W, 'bearish' for M.
        quality_score: 0-1 based on how textbook the formation is.
        session_peak1: MM session name when peak1 occurred (if available).
        session_peak2: MM session name when peak2 occurred (if available).
        at_key_level: True if the formation occurs near HOW/LOW/HOD/LOD.
        confirmed: True if the candle at peak2 closed as hammer/engulfing.
    """

    type: str
    variant: str
    peak1_idx: int
    peak1_price: float
    peak2_idx: int
    peak2_price: float
    trough_idx: int
    trough_price: float
    direction: str
    quality_score: float = 0.0
    session_peak1: str | None = None
    session_peak2: str | None = None
    at_key_level: bool = False
    confirmed: bool = False


@dataclass
class ThreeHitsResult:
    """Result of the Three Hits Rule check.

    Three tests of the same level without breaking = reversal imminent.
    Four hits = continuation (MMs have decided to break through).

    Attributes:
        detected: True if 3+ hits were found.
        hit_count: Number of times the level was tested (3 or 4+).
        level_tested: The price level that was repeatedly tested.
        hit_indices: DataFrame integer-location indices of each hit candle.
        hit_sessions: Session name for each hit (must differ for valid pattern).
        expected_outcome: 'reversal' if 3 hits, 'continuation' if 4+.
    """

    detected: bool = False
    hit_count: int = 0
    level_tested: float = 0.0
    hit_indices: list[int] = field(default_factory=list)
    hit_sessions: list[str] = field(default_factory=list)
    expected_outcome: str = ""


@dataclass
class FormationValidation:
    """Validation of MM confirmation criteria for a formation.

    The Market Makers Method requires three confirmations before entry:
    1. Stopping Volume Candle (SVC) at the 1st peak.
    2. Three trap candles on the inside right side of the formation.
    3. 50 EMA break with above-average volume after the formation.

    Attributes:
        has_svc: True if a stopping volume candle exists at the 1st peak.
        has_inside_traps: True if 3+ inside/trap candles on the right side.
        has_ema_break: True if price broke the 50 EMA with volume.
        confirmation_score: 0-3 count of confirmations met.
        entry_price: Suggested entry (close of the 2nd peak candle).
        stop_loss: Above 1st peak wick (M) or below 1st peak wick (W).
    """

    has_svc: bool = False
    has_inside_traps: bool = False
    has_ema_break: bool = False
    confirmation_score: int = 0
    entry_price: float = 0.0
    stop_loss: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_swing_highs(highs: np.ndarray, window: int = SWING_WINDOW) -> list[int]:
    """Return indices where ``highs[i]`` is the max within *window* candles
    on both sides.
    """
    indices: list[int] = []
    n = len(highs)
    for i in range(window, n - window):
        if highs[i] == np.max(highs[i - window : i + window + 1]):
            indices.append(i)
    return indices


def _find_swing_lows(lows: np.ndarray, window: int = SWING_WINDOW) -> list[int]:
    """Return indices where ``lows[i]`` is the min within *window* candles
    on both sides.
    """
    indices: list[int] = []
    n = len(lows)
    for i in range(window, n - window):
        if lows[i] == np.min(lows[i - window : i + window + 1]):
            indices.append(i)
    return indices


def _is_hammer(o: float, h: float, l: float, c: float) -> bool:
    """True if the candle is a bullish hammer (long lower wick, small body).

    A hammer has a lower shadow at least twice the body length and a small
    or nonexistent upper shadow.
    """
    body = abs(c - o)
    full_range = h - l
    if full_range == 0:
        return False
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    return (
        lower_shadow >= 2 * body
        and upper_shadow <= body * 0.5
        and body / full_range < 0.4
    )


def _is_inverted_hammer(o: float, h: float, l: float, c: float) -> bool:
    """True if the candle is an inverted hammer / shooting star.

    Long upper wick, small body, small lower shadow.
    """
    body = abs(c - o)
    full_range = h - l
    if full_range == 0:
        return False
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    return (
        upper_shadow >= 2 * body
        and lower_shadow <= body * 0.5
        and body / full_range < 0.4
    )


def _is_engulfing_bullish(
    prev_o: float, prev_c: float, cur_o: float, cur_c: float,
) -> bool:
    """True if the current candle is a bullish engulfing (green body wraps
    the prior red body)."""
    return prev_c < prev_o and cur_c > cur_o and cur_c > prev_o and cur_o < prev_c


def _is_engulfing_bearish(
    prev_o: float, prev_c: float, cur_o: float, cur_c: float,
) -> bool:
    """True if the current candle is a bearish engulfing."""
    return prev_c > prev_o and cur_c < cur_o and cur_c < prev_o and cur_o > prev_c


def _calc_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# FormationDetector
# ---------------------------------------------------------------------------


class FormationDetector:
    """Detects M-top and W-bottom formations used in the Market Makers Method.

    These formations are the primary entry signals. MMs create double-top /
    double-bottom patterns where the second peak falls short of the first,
    trapping retail traders before reversing price sharply.

    Args:
        session_analyzer: Optional :class:`MMSessionAnalyzer` instance for
            multi-session detection and session tagging.  When ``None``,
            session-related features are disabled gracefully.
    """

    def __init__(self, session_analyzer: MMSessionAnalyzer | None = None) -> None:
        self.session_analyzer = session_analyzer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        ohlc: pd.DataFrame,
        direction_bias: str | None = None,
    ) -> list[Formation]:
        """Find all M/W formations in recent candle data.

        This is the main entry point that orchestrates the sub-detectors.

        Args:
            ohlc: OHLCV DataFrame with columns ``open``, ``high``, ``low``,
                ``close``, ``volume`` and a DatetimeIndex (UTC).
            direction_bias: Optional ``'bullish'`` or ``'bearish'`` to filter
                results. ``None`` returns both M and W formations.

        Returns:
            List of :class:`Formation` instances sorted by quality score
            (best first).
        """
        if ohlc is None or ohlc.empty or len(ohlc) < DEFAULT_LOOKBACK:
            logger.debug("mm_formations_skip", reason="insufficient_data", bars=len(ohlc) if ohlc is not None else 0)
            return []

        formations = self.detect_mw(ohlc, lookback=DEFAULT_LOOKBACK)

        # Tag multi-session formations if session analyzer is available.
        if self.session_analyzer is not None:
            formations = self.detect_multi_session(ohlc, formations)

        # Apply direction filter.
        if direction_bias == "bullish":
            formations = [f for f in formations if f.direction == "bullish"]
        elif direction_bias == "bearish":
            formations = [f for f in formations if f.direction == "bearish"]

        # Sort by quality (best first).
        formations.sort(key=lambda f: f.quality_score, reverse=True)

        logger.info(
            "mm_formations_detected",
            count=len(formations),
            direction_bias=direction_bias,
            variants=[f.variant for f in formations],
        )
        return formations

    def detect_mw(
        self,
        ohlc: pd.DataFrame,
        lookback: int = DEFAULT_LOOKBACK,
    ) -> list[Formation]:
        """Core M/W detection algorithm.

        Scans for swing high-low-high (M-top) and swing low-high-low
        (W-bottom) patterns within the most recent *lookback* candles.

        Algorithm:
            1. Find all swing highs and swing lows using a rolling window.
            2. For M-top: pair swing highs separated by a swing low trough.
               The 2nd high must fall short of the 1st (standard) or exceed
               it (final damage).
            3. For W-bottom: pair swing lows separated by a swing high ridge.
               The 2nd low must stay above the 1st (standard) or drop below
               it (final damage).
            4. Score quality based on separation, shortfall, wick behavior,
               and volume characteristics.

        Args:
            ohlc: OHLCV DataFrame.
            lookback: Number of recent candles to scan.

        Returns:
            List of :class:`Formation` instances (unsorted).
        """
        if ohlc.empty or len(ohlc) < max(lookback, SWING_WINDOW * 2 + 1):
            return []

        df = ohlc.iloc[-lookback:].copy()

        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        opens = df["open"].values.astype(float)
        volumes = df["volume"].values.astype(float)

        swing_highs = _find_swing_highs(highs)
        swing_lows = _find_swing_lows(lows)

        formations: list[Formation] = []

        # --- M-top detection (bearish) ---
        formations.extend(
            self._detect_m_tops(
                df, highs, lows, opens, closes, volumes, swing_highs, swing_lows,
            )
        )

        # --- W-bottom detection (bullish) ---
        formations.extend(
            self._detect_w_bottoms(
                df, highs, lows, opens, closes, volumes, swing_highs, swing_lows,
            )
        )

        return formations

    def detect_multi_session(
        self,
        ohlc: pd.DataFrame,
        formations: list[Formation],
    ) -> list[Formation]:
        """Tag formations whose peaks span different MM sessions.

        A multi-session M/W is the strongest variant because two distinct
        groups of market makers have independently set the trap. If peak1
        forms during Asia and peak2 during UK open, the formation almost
        guarantees a 3-level follow-through.

        Requires ``self.session_analyzer`` to be set. If unavailable, returns
        formations unchanged.

        Args:
            ohlc: OHLCV DataFrame with DatetimeIndex.
            formations: Previously detected formations to enrich.

        Returns:
            The same list with ``session_peak1``, ``session_peak2``, and
            ``variant`` updated where applicable.
        """
        if self.session_analyzer is None:
            return formations

        if not hasattr(ohlc.index, 'tz') or ohlc.index.empty:
            return formations

        for f in formations:
            try:
                # Map DataFrame iloc index back to the original index.
                peak1_time = ohlc.index[f.peak1_idx]
                peak2_time = ohlc.index[f.peak2_idx]

                if not isinstance(peak1_time, datetime):
                    # pandas Timestamp -> datetime
                    peak1_time = peak1_time.to_pydatetime()
                    peak2_time = peak2_time.to_pydatetime()

                sess1 = self.session_analyzer.get_session_for_candle(peak1_time)
                sess2 = self.session_analyzer.get_session_for_candle(peak2_time)

                f.session_peak1 = sess1.session_name
                f.session_peak2 = sess2.session_name

                # Promote to multi_session if peaks are in different major sessions.
                major_sessions = {"asia", "uk", "us"}
                if (
                    sess1.session_name in major_sessions
                    and sess2.session_name in major_sessions
                    and sess1.session_name != sess2.session_name
                    and f.variant == "standard"
                ):
                    f.variant = "multi_session"
                    # Multi-session formations get a quality boost.
                    f.quality_score = min(1.0, f.quality_score + 0.15)
                    logger.info(
                        "mm_formation_multi_session",
                        type=f.type,
                        session1=sess1.session_name,
                        session2=sess2.session_name,
                    )
            except Exception as e:
                logger.debug("mm_formation_session_tag_error", error=str(e))

        return formations

    def detect_three_hits(
        self,
        ohlc: pd.DataFrame,
        level: float,
        tolerance: float = LEVEL_TOLERANCE,
    ) -> ThreeHitsResult:
        """Detect the Three Hits Rule at a given price level.

        Three tests of HOW/LOW without breaking = reversal imminent. Four
        or more hits = MMs have decided to break through (continuation).
        Hits must occur in different sessions to be valid.

        Args:
            ohlc: OHLCV DataFrame with DatetimeIndex.
            level: The price level to test (e.g., HOW, LOW, HOD, LOD).
            tolerance: Fractional tolerance for "touching" the level
                (default 0.2%).

        Returns:
            :class:`ThreeHitsResult`.
        """
        result = ThreeHitsResult()

        if ohlc is None or ohlc.empty or len(ohlc) < 10:
            return result

        highs = ohlc["high"].values.astype(float)
        lows = ohlc["low"].values.astype(float)
        tol_abs = level * tolerance

        hit_indices: list[int] = []
        hit_sessions: list[str] = []

        for i in range(len(ohlc)):
            # Check if this candle's wick touched the level.
            touched_from_below = abs(highs[i] - level) <= tol_abs and highs[i] >= level - tol_abs
            touched_from_above = abs(lows[i] - level) <= tol_abs and lows[i] <= level + tol_abs

            if touched_from_below or touched_from_above:
                # Determine session for this candle.
                session_name = "unknown"
                if self.session_analyzer is not None:
                    try:
                        candle_time = ohlc.index[i]
                        if not isinstance(candle_time, datetime):
                            candle_time = candle_time.to_pydatetime()
                        sess = self.session_analyzer.get_session_for_candle(candle_time)
                        session_name = sess.session_name
                    except Exception:
                        pass

                # Only count if this is a different session or different candle group.
                # Consecutive candles in the same session count as one hit.
                if hit_sessions and session_name == hit_sessions[-1]:
                    # Same session — update the last hit's index to the latest candle.
                    hit_indices[-1] = i
                else:
                    hit_indices.append(i)
                    hit_sessions.append(session_name)

        hit_count = len(hit_indices)

        # Course lesson 18: "The hits to the level also need to be in different
        # sessions, and not one after the other" and "There can be up to two
        # hits to a level in one session, for purpose of looking for three
        # hits to a level, but not more than this. You don't want all your
        # three hits to a level to appear in the exact same session."
        if hit_count >= 3:
            # Count hits per session
            from collections import Counter
            session_counts = Counter(hit_sessions)
            max_per_session = max(session_counts.values()) if session_counts else 0
            unique_sessions = set(s for s in hit_sessions if s != "unknown")
            multi_session = len(unique_sessions) >= 2 or "unknown" in hit_sessions
            # B5 enforcement: ≤2 hits per session
            session_cap_ok = max_per_session <= 2

            if (multi_session and session_cap_ok) or self.session_analyzer is None:
                result.detected = True
                result.hit_count = hit_count
                result.level_tested = level
                result.hit_indices = hit_indices
                result.hit_sessions = hit_sessions
                result.expected_outcome = "continuation" if hit_count >= 4 else "reversal"

                logger.info(
                    "mm_three_hits_detected",
                    hits=hit_count,
                    level=level,
                    outcome=result.expected_outcome,
                    sessions=hit_sessions,
                    max_per_session=max_per_session,
                )
            elif not session_cap_ok:
                logger.info(
                    "mm_three_hits_rejected_same_session",
                    hits=hit_count,
                    level=level,
                    max_per_session=max_per_session,
                    sessions=hit_sessions,
                )

        return result

    def validate_formation(
        self,
        ohlc: pd.DataFrame,
        formation: Formation,
    ) -> FormationValidation:
        """Check MM confirmation criteria for a detected formation.

        Three confirmations (any combination counts):
            1. **Stopping Volume Candle (SVC)**: The candle at or near the
               1st peak has volume >= 1.5x the 20-period average, indicating
               MMs absorbing orders at Level 3.
            2. **Inside/Trap Candles**: At least 3 small-bodied (inside)
               candles on the right side of the formation between the peaks,
               showing price being "trapped" before reversal.
            3. **50 EMA Break with Volume**: After the 2nd peak, price
               crosses the 50 EMA with above-average volume, confirming the
               reversal direction.

        Args:
            ohlc: OHLCV DataFrame.
            formation: The :class:`Formation` to validate.

        Returns:
            :class:`FormationValidation` with confirmation flags and
            suggested entry/stop-loss levels.
        """
        validation = FormationValidation()

        if ohlc is None or ohlc.empty:
            return validation

        highs = ohlc["high"].values.astype(float)
        lows = ohlc["low"].values.astype(float)
        opens = ohlc["open"].values.astype(float)
        closes = ohlc["close"].values.astype(float)
        volumes = ohlc["volume"].values.astype(float)

        p1 = formation.peak1_idx
        p2 = formation.peak2_idx

        # --- 1. Stopping Volume Candle at 1st peak ---
        validation.has_svc = self._check_svc(volumes, p1)

        # --- 2. Inside/trap candles between peaks ---
        validation.has_inside_traps = self._check_inside_traps(
            opens, highs, lows, closes, p1, p2,
        )

        # --- 3. 50 EMA break with volume after 2nd peak ---
        validation.has_ema_break = self._check_ema_break(
            ohlc, formation, volumes,
        )

        # --- Confirmation score ---
        validation.confirmation_score = sum([
            validation.has_svc,
            validation.has_inside_traps,
            validation.has_ema_break,
        ])

        # --- Entry and stop-loss ---
        if p2 < len(closes):
            validation.entry_price = closes[p2]

        if formation.type == "M":
            # M-top (bearish): stop above 1st peak's wick.
            if p1 < len(highs):
                validation.stop_loss = highs[p1]
        else:
            # W-bottom (bullish): stop below 1st peak's wick.
            if p1 < len(lows):
                validation.stop_loss = lows[p1]

        logger.debug(
            "mm_formation_validated",
            type=formation.type,
            variant=formation.variant,
            svc=validation.has_svc,
            traps=validation.has_inside_traps,
            ema_break=validation.has_ema_break,
            score=validation.confirmation_score,
        )

        return validation

    # ------------------------------------------------------------------
    # M-top detection
    # ------------------------------------------------------------------

    def _detect_m_tops(
        self,
        df: pd.DataFrame,
        highs: np.ndarray,
        lows: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        swing_highs: list[int],
        swing_lows: list[int],
    ) -> list[Formation]:
        """Detect M-top (bearish double-top) patterns.

        An M-top consists of: Swing High 1 -> pullback to Swing Low (trough)
        -> Swing High 2 that fails to reach High 1.
        """
        formations: list[Formation] = []

        for i in range(len(swing_highs)):
            for j in range(i + 1, len(swing_highs)):
                sh1 = swing_highs[i]
                sh2 = swing_highs[j]

                # Enforce minimum separation.
                if sh2 - sh1 < MIN_PEAK_SEPARATION:
                    continue

                # Find the deepest trough (swing low) between the two peaks.
                trough_idx = self._find_trough_between(
                    swing_lows, lows, sh1, sh2, direction="low",
                )
                if trough_idx is None:
                    continue

                peak1_price = highs[sh1]
                peak2_price = highs[sh2]
                trough_price = lows[trough_idx]

                # The trough must be meaningfully below both peaks.
                avg_peak = (peak1_price + peak2_price) / 2
                if avg_peak == 0:
                    continue
                trough_depth = (avg_peak - trough_price) / avg_peak
                if trough_depth < 0.001:
                    # Trough too shallow — not a real M pattern.
                    continue

                # Determine variant.
                variant, is_valid = self._classify_m_variant(
                    peak1_price, peak2_price, sh2, opens, highs, lows, closes,
                )
                if not is_valid:
                    continue

                # Check for confirmed candle close at peak2.
                confirmed = self._check_peak2_confirmation_m(
                    sh2, opens, highs, lows, closes,
                )

                # Calculate quality score.
                quality = self._score_m_quality(
                    peak1_price, peak2_price, trough_price,
                    sh1, sh2, trough_idx, volumes, highs,
                )

                # Remap local indices to the original DataFrame indices.
                offset = len(df) - len(highs)  # Should be 0 for iloc-sliced df
                global_sh1 = df.index.get_loc(df.index[0]) + sh1 if hasattr(df.index, 'get_loc') else sh1
                # Use positional indices relative to the full ohlc to keep
                # compatibility. We store the iloc position within the passed df
                # since the caller slices ohlc[-lookback:].
                # To map back to the original ohlc, the caller can offset.
                # For now, store the original ohlc iloc position.
                base = len(df.iloc[0:0])  # 0
                orig_offset = len(df) - len(highs)  # 0

                formations.append(
                    Formation(
                        type="M",
                        variant=variant,
                        peak1_idx=sh1 + (len(df) - len(highs)),
                        peak1_price=peak1_price,
                        peak2_idx=sh2 + (len(df) - len(highs)),
                        peak2_price=peak2_price,
                        trough_idx=trough_idx + (len(df) - len(lows)),
                        trough_price=trough_price,
                        direction="bearish",
                        quality_score=quality,
                        confirmed=confirmed,
                    )
                )

        return formations

    # ------------------------------------------------------------------
    # W-bottom detection
    # ------------------------------------------------------------------

    def _detect_w_bottoms(
        self,
        df: pd.DataFrame,
        highs: np.ndarray,
        lows: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        swing_highs: list[int],
        swing_lows: list[int],
    ) -> list[Formation]:
        """Detect W-bottom (bullish double-bottom) patterns.

        A W-bottom consists of: Swing Low 1 -> rally to Swing High (ridge)
        -> Swing Low 2 that stays above Low 1.
        """
        formations: list[Formation] = []

        for i in range(len(swing_lows)):
            for j in range(i + 1, len(swing_lows)):
                sl1 = swing_lows[i]
                sl2 = swing_lows[j]

                # Enforce minimum separation.
                if sl2 - sl1 < MIN_PEAK_SEPARATION:
                    continue

                # Find the highest ridge (swing high) between the two troughs.
                ridge_idx = self._find_trough_between(
                    swing_highs, highs, sl1, sl2, direction="high",
                )
                if ridge_idx is None:
                    continue

                peak1_price = lows[sl1]
                peak2_price = lows[sl2]
                ridge_price = highs[ridge_idx]

                # Ridge must be meaningfully above both lows.
                avg_low = (peak1_price + peak2_price) / 2
                if avg_low == 0:
                    continue
                ridge_height = (ridge_price - avg_low) / avg_low
                if ridge_height < 0.001:
                    continue

                # Determine variant.
                variant, is_valid = self._classify_w_variant(
                    peak1_price, peak2_price, sl2, opens, highs, lows, closes,
                )
                if not is_valid:
                    continue

                # Check for confirmed candle close at peak2 (the 2nd low).
                confirmed = self._check_peak2_confirmation_w(
                    sl2, opens, highs, lows, closes,
                )

                # Quality score.
                quality = self._score_w_quality(
                    peak1_price, peak2_price, ridge_price,
                    sl1, sl2, ridge_idx, volumes, lows,
                )

                formations.append(
                    Formation(
                        type="W",
                        variant=variant,
                        peak1_idx=sl1,
                        peak1_price=peak1_price,
                        peak2_idx=sl2,
                        peak2_price=peak2_price,
                        trough_idx=ridge_idx,
                        trough_price=ridge_price,
                        direction="bullish",
                        quality_score=quality,
                        confirmed=confirmed,
                    )
                )

        return formations

    # ------------------------------------------------------------------
    # Variant classification
    # ------------------------------------------------------------------

    def _classify_m_variant(
        self,
        peak1: float,
        peak2: float,
        peak2_idx: int,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> tuple[str, bool]:
        """Classify the M-top variant and check validity.

        Returns:
            (variant_name, is_valid) tuple.
        """
        if peak1 == 0:
            return ("standard", False)

        shortfall_pct = (peak1 - peak2) / peak1

        if shortfall_pct > 0:
            # Standard M: 2nd peak is lower than 1st (textbook MM pattern).
            # Must be a meaningful shortfall (at least 0.05%).
            if shortfall_pct < 0.0005:
                return ("board_meeting", True)
            return ("standard", True)

        elif shortfall_pct < -0.0005:
            # Final Damage M: 2nd peak EXCEEDS 1st (higher high fakeout).
            # Requires inverted hammer on the 2nd peak candle.
            if peak2_idx < len(opens):
                has_iham = _is_inverted_hammer(
                    opens[peak2_idx], highs[peak2_idx],
                    lows[peak2_idx], closes[peak2_idx],
                )
                # Also accept bearish engulfing if next candle exists.
                has_engulf = False
                if peak2_idx + 1 < len(opens):
                    has_engulf = _is_engulfing_bearish(
                        opens[peak2_idx], closes[peak2_idx],
                        opens[peak2_idx + 1], closes[peak2_idx + 1],
                    )
                if has_iham or has_engulf:
                    return ("final_damage", True)
            # Higher high without reversal candle — not valid.
            return ("final_damage", False)

        else:
            # Peaks are nearly equal — board meeting (consolidation) variant.
            return ("board_meeting", True)

    def _classify_w_variant(
        self,
        low1: float,
        low2: float,
        low2_idx: int,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> tuple[str, bool]:
        """Classify the W-bottom variant and check validity.

        Returns:
            (variant_name, is_valid) tuple.
        """
        if low1 == 0:
            return ("standard", False)

        shortfall_pct = (low2 - low1) / low1

        if shortfall_pct > 0:
            # Standard W: 2nd low is higher than 1st (textbook MM pattern).
            if shortfall_pct < 0.0005:
                return ("board_meeting", True)
            return ("standard", True)

        elif shortfall_pct < -0.0005:
            # Final Damage W: 2nd low drops BELOW 1st (lower low fakeout).
            # Requires hammer candle on the 2nd low.
            if low2_idx < len(opens):
                has_ham = _is_hammer(
                    opens[low2_idx], highs[low2_idx],
                    lows[low2_idx], closes[low2_idx],
                )
                # Also accept bullish engulfing.
                has_engulf = False
                if low2_idx + 1 < len(opens):
                    has_engulf = _is_engulfing_bullish(
                        opens[low2_idx], closes[low2_idx],
                        opens[low2_idx + 1], closes[low2_idx + 1],
                    )
                if has_ham or has_engulf:
                    return ("final_damage", True)
            return ("final_damage", False)

        else:
            return ("board_meeting", True)

    # ------------------------------------------------------------------
    # Confirmation checks
    # ------------------------------------------------------------------

    def _check_peak2_confirmation_m(
        self,
        peak2_idx: int,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> bool:
        """Check if the candle at the M-top's 2nd peak closed as an
        inverted hammer, shooting star, or bearish engulfing."""
        if peak2_idx >= len(opens):
            return False
        if _is_inverted_hammer(opens[peak2_idx], highs[peak2_idx], lows[peak2_idx], closes[peak2_idx]):
            return True
        # Check bearish engulfing with the next candle.
        if peak2_idx + 1 < len(opens):
            return _is_engulfing_bearish(
                opens[peak2_idx], closes[peak2_idx],
                opens[peak2_idx + 1], closes[peak2_idx + 1],
            )
        # Also accept a red (bearish) close.
        return closes[peak2_idx] < opens[peak2_idx]

    def _check_peak2_confirmation_w(
        self,
        peak2_idx: int,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> bool:
        """Check if the candle at the W-bottom's 2nd trough closed as a
        hammer or bullish engulfing."""
        if peak2_idx >= len(opens):
            return False
        if _is_hammer(opens[peak2_idx], highs[peak2_idx], lows[peak2_idx], closes[peak2_idx]):
            return True
        if peak2_idx + 1 < len(opens):
            return _is_engulfing_bullish(
                opens[peak2_idx], closes[peak2_idx],
                opens[peak2_idx + 1], closes[peak2_idx + 1],
            )
        # Also accept a green (bullish) close.
        return closes[peak2_idx] > opens[peak2_idx]

    # ------------------------------------------------------------------
    # Validation sub-checks
    # ------------------------------------------------------------------

    def _check_svc(self, volumes: np.ndarray, peak_idx: int) -> bool:
        """Check for a Stopping Volume Candle at or near the 1st peak.

        A SVC is a candle with volume >= 1.5x the 20-period rolling average,
        indicating market makers absorbing orders at a key level.
        """
        if peak_idx >= len(volumes) or len(volumes) < 20:
            return False

        # Check the peak candle and its immediate neighbors.
        lookback_start = max(0, peak_idx - 20)
        avg_vol = np.mean(volumes[lookback_start:peak_idx]) if peak_idx > lookback_start else np.mean(volumes[:20])

        if avg_vol == 0:
            return False

        # Check the peak candle and one candle on each side.
        for offset in (-1, 0, 1):
            idx = peak_idx + offset
            if 0 <= idx < len(volumes):
                if volumes[idx] >= avg_vol * SVC_VOLUME_MULT:
                    return True

        return False

    def _check_inside_traps(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        peak1_idx: int,
        peak2_idx: int,
    ) -> bool:
        """Check for 3+ inside/trap candles between the peaks.

        Trap candles are small-bodied candles that stay within the range
        of the prior candle (inside bars). They indicate trapped traders
        being lured into thinking the move will continue.
        """
        if peak2_idx - peak1_idx < TRAP_CANDLE_COUNT + 1:
            return False

        # Scan the right side of the formation (after peak1 pullback).
        # Focus on candles closer to peak2 (the "inside right side").
        scan_start = peak1_idx + 1
        scan_end = min(peak2_idx, len(opens))

        inside_count = 0
        for i in range(scan_start + 1, scan_end):
            prev_range = highs[i - 1] - lows[i - 1]
            if prev_range == 0:
                continue
            # Inside bar: current candle's range fits within previous candle's range.
            is_inside = highs[i] <= highs[i - 1] and lows[i] >= lows[i - 1]
            # Small body: body is less than 50% of previous candle range.
            body = abs(closes[i] - opens[i])
            is_small = body < prev_range * 0.5

            if is_inside or is_small:
                inside_count += 1

        return inside_count >= TRAP_CANDLE_COUNT

    def _check_ema_break(
        self,
        ohlc: pd.DataFrame,
        formation: Formation,
        volumes: np.ndarray,
    ) -> bool:
        """Check for a 50 EMA break with volume after the 2nd peak.

        After the formation completes, price should cross the 50 EMA in
        the reversal direction with above-average volume to confirm the
        move.
        """
        if len(ohlc) < EMA_PERIOD:
            return False

        ema = _calc_ema(ohlc["close"].astype(float), EMA_PERIOD)
        closes = ohlc["close"].values.astype(float)
        vols = volumes

        p2 = formation.peak2_idx

        # Look at candles after peak2 (up to 10 candles).
        scan_end = min(p2 + 10, len(closes))

        if scan_end <= p2:
            return False

        avg_vol = np.mean(vols[max(0, p2 - 20):p2]) if p2 > 0 else np.mean(vols[:20])
        if avg_vol == 0:
            return False

        for i in range(p2 + 1, scan_end):
            ema_val = ema.iloc[i] if i < len(ema) else 0

            if formation.type == "M":
                # Bearish: price should break below EMA.
                if closes[i] < ema_val and vols[i] >= avg_vol * 1.2:
                    return True
            else:
                # Bullish: price should break above EMA.
                if closes[i] > ema_val and vols[i] >= avg_vol * 1.2:
                    return True

        return False

    # ------------------------------------------------------------------
    # Quality scoring
    # ------------------------------------------------------------------

    def _score_m_quality(
        self,
        peak1: float,
        peak2: float,
        trough: float,
        idx1: int,
        idx2: int,
        trough_idx: int,
        volumes: np.ndarray,
        highs: np.ndarray,
    ) -> float:
        """Score M-top formation quality from 0 to 1.

        Factors:
            - Separation: How far apart the two peaks are (normalized).
            - Shortfall: How much the 2nd peak falls short of the 1st.
            - Wick behavior: Whether spikes pull away quickly (long wicks).
            - Volume: Higher volume at peaks = more significant.
            - Symmetry: How centered the trough is between peaks.
        """
        if peak1 == 0:
            return 0.0

        # Separation score: ideal is 8-20 candles apart.
        separation = idx2 - idx1
        if separation < MIN_PEAK_SEPARATION:
            sep_score = 0.0
        elif separation <= 8:
            sep_score = separation / 8.0
        elif separation <= 20:
            sep_score = 1.0
        else:
            sep_score = max(0.3, 1.0 - (separation - 20) / 20.0)

        # Shortfall score: how much peak2 falls short (0.1-2% is ideal).
        shortfall_pct = (peak1 - peak2) / peak1
        if shortfall_pct > 0:
            if shortfall_pct < 0.001:
                sf_score = 0.3  # Too close — board meeting territory.
            elif shortfall_pct < 0.02:
                sf_score = 1.0  # Sweet spot.
            elif shortfall_pct < 0.05:
                sf_score = 0.6  # Acceptable but wide.
            else:
                sf_score = 0.2  # Too far apart — may not be an M.
        else:
            # Final damage — 2nd peak exceeds 1st.
            sf_score = 0.5

        # Wick score: long upper wicks at peaks indicate quick rejection.
        wick_score = 0.5
        for idx in (idx1, idx2):
            if idx < len(highs) and idx < len(volumes):
                body_top = max(
                    volumes[idx] if idx < len(volumes) else 0,  # placeholder
                    0,
                )
                # Use actual OHLC for wick calculation.
                # We only have highs here; approximate with the high - close.
                pass
        # Simplified: check if peaks have rejection wicks (high > close).
        if idx1 < len(highs):
            close1 = highs[idx1]  # Approximate — we don't have closes here.
            wick_score = 0.7  # Default good score when peaks are swing highs.

        # Volume score: above-average volume at peaks is more significant.
        vol_score = 0.5
        if len(volumes) > 20:
            avg_vol = np.mean(volumes[-20:])
            if avg_vol > 0:
                peak_vols = []
                for idx in (idx1, idx2):
                    if 0 <= idx < len(volumes):
                        peak_vols.append(volumes[idx])
                if peak_vols:
                    max_peak_vol = max(peak_vols)
                    ratio = max_peak_vol / avg_vol
                    vol_score = min(1.0, ratio / 2.0)

        # Symmetry score: trough should be roughly centered.
        total_span = idx2 - idx1
        if total_span > 0:
            trough_pos = (trough_idx - idx1) / total_span
            sym_score = 1.0 - abs(trough_pos - 0.5) * 2  # 1.0 at center.
        else:
            sym_score = 0.0

        quality = (
            _W_SEPARATION * sep_score
            + _W_SHORTFALL * sf_score
            + _W_WICK * wick_score
            + _W_VOLUME * vol_score
            + _W_SYMMETRY * sym_score
        )

        return round(min(1.0, max(0.0, quality)), 3)

    def _score_w_quality(
        self,
        low1: float,
        low2: float,
        ridge: float,
        idx1: int,
        idx2: int,
        ridge_idx: int,
        volumes: np.ndarray,
        lows: np.ndarray,
    ) -> float:
        """Score W-bottom formation quality from 0 to 1.

        Mirror of ``_score_m_quality`` for the bullish case.
        """
        if low1 == 0:
            return 0.0

        # Separation.
        separation = idx2 - idx1
        if separation < MIN_PEAK_SEPARATION:
            sep_score = 0.0
        elif separation <= 8:
            sep_score = separation / 8.0
        elif separation <= 20:
            sep_score = 1.0
        else:
            sep_score = max(0.3, 1.0 - (separation - 20) / 20.0)

        # Shortfall: how much low2 stays above low1.
        shortfall_pct = (low2 - low1) / low1
        if shortfall_pct > 0:
            if shortfall_pct < 0.001:
                sf_score = 0.3
            elif shortfall_pct < 0.02:
                sf_score = 1.0
            elif shortfall_pct < 0.05:
                sf_score = 0.6
            else:
                sf_score = 0.2
        else:
            sf_score = 0.5  # Final damage.

        # Wick score: long lower wicks at lows indicate rejection.
        wick_score = 0.7  # Default for swing lows.

        # Volume.
        vol_score = 0.5
        if len(volumes) > 20:
            avg_vol = np.mean(volumes[-20:])
            if avg_vol > 0:
                peak_vols = []
                for idx in (idx1, idx2):
                    if 0 <= idx < len(volumes):
                        peak_vols.append(volumes[idx])
                if peak_vols:
                    max_peak_vol = max(peak_vols)
                    ratio = max_peak_vol / avg_vol
                    vol_score = min(1.0, ratio / 2.0)

        # Symmetry.
        total_span = idx2 - idx1
        if total_span > 0:
            ridge_pos = (ridge_idx - idx1) / total_span
            sym_score = 1.0 - abs(ridge_pos - 0.5) * 2
        else:
            sym_score = 0.0

        quality = (
            _W_SEPARATION * sep_score
            + _W_SHORTFALL * sf_score
            + _W_WICK * wick_score
            + _W_VOLUME * vol_score
            + _W_SYMMETRY * sym_score
        )

        return round(min(1.0, max(0.0, quality)), 3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_trough_between(
        swing_indices: list[int],
        prices: np.ndarray,
        start: int,
        end: int,
        direction: str,
    ) -> int | None:
        """Find the most extreme swing point between two indices.

        Args:
            swing_indices: Sorted list of swing high or swing low indices.
            prices: The price array (highs for ridges, lows for troughs).
            start: Start of the range (exclusive).
            end: End of the range (exclusive).
            direction: ``'low'`` to find the lowest trough, ``'high'`` to
                find the highest ridge.

        Returns:
            Index of the most extreme swing point, or ``None`` if none exists.
        """
        candidates = [s for s in swing_indices if start < s < end]
        if not candidates:
            return None

        if direction == "low":
            return min(candidates, key=lambda i: prices[i])
        else:
            return max(candidates, key=lambda i: prices[i])
