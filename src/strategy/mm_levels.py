"""MM Method level counting module.

Counts the 3-level cycle that Market Makers run after an M or W formation.
Levels are identified by abnormally high volume bursts (PVSRA vectors).

Key concepts from the MM Method:
- After an M (double top) or W (double bottom) formation, MM runs price
  through 3 levels over approximately 3 days.
- Level 1: Must break the 50 EMA with volume. Takes out 100x leverage traders.
- Level 2: Runs toward the 200 EMA. Takes out 50x leverage traders.
- Level 3: Trend acceleration, EMAs fan out. Takes out 25x leverage traders.
- A 4th level (Extended Rise/Drop) almost always brings the correction.
- Between levels: "Board Meetings" = consolidation where MM accumulates
  contracts and prepares the next push.
- Levels are distinguished by abnormally high volume compared to surrounding
  candles, classified using the PVSRA (Price Volume Support Resistance Analysis)
  system.

PVSRA Vector classification:
- Vector 200%: volume >= 200% of the 10-period average -> strongest signal
- Vector 150%: volume >= 150% of the 10-period average -> moderate signal
- Color coding:
    Bullish (close > open): green (200%), blue (150%)
    Bearish (close < open): red (200%), magenta (150%)
    Normal: gray

Stopping Volume Candle (SVC):
- Small body + large wick in prior trend direction + very high volume + at Level 3
- Signals trend exhaustion: big players are offloading, creating the wick
- Confirmed when price fails to return into the wick zone
- Volume degradation (200% vectors becoming 150% vectors) = guaranteed pause
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

if TYPE_CHECKING:
    pass  # EMAFramework forward reference handled via string annotation

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PVSRA_LOOKBACK = 10        # candles for average volume baseline
VECTOR_200_THRESHOLD = 2.0  # 200% of average
VECTOR_150_THRESHOLD = 1.5  # 150% of average
CLUSTER_GAP_MAX = 3         # max candles between vectors to still form a cluster
BOARD_MEETING_MIN = 3       # minimum candles for a board meeting
SVC_BODY_RATIO_MAX = 0.35   # body must be <= 35% of total range for SVC
SVC_VOLUME_MIN = 2.0        # minimum volume ratio for SVC detection
SVC_WICK_RETURN_CANDLES = 5 # candles to check for wick return after SVC
EMA_FLAT_THRESHOLD = 0.001  # EMA slope threshold for "flat" classification


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LevelInfo:
    """A single MM level identified by a cluster of vector candles.

    Levels 1-3 form the standard MM cycle. Level 4 is the Extended
    Rise/Drop that almost always precedes a major correction.
    """
    level_number: int           # 1-4
    start_idx: int              # first candle index (absolute in DataFrame)
    end_idx: int                # last candle index
    direction: str              # 'bullish' or 'bearish'
    magnitude: float            # price move magnitude as percentage
    volume_confirmed: bool      # True if cluster contains vector candles
    vector_type: str            # dominant vector: 'vector_200' or 'vector_150'


@dataclass
class BoardMeetingInfo:
    """Consolidation period between MM levels.

    During board meetings, MM accumulates contracts at favorable prices.
    Price action is range-bound, volume is relatively low, and EMAs
    tend to flatten out.
    """
    start_idx: int
    end_idx: int
    duration_candles: int
    contains_stop_hunt: bool    # brief wick outside range to grab liquidity
    ema_flattening: bool        # True if EMAs are converging/flat


@dataclass
class SVCResult:
    """Stopping Volume Candle detection result.

    An SVC at Level 3 is a high-probability reversal signal. The candle
    shows a small body (indecision) with a large wick in the trend
    direction (absorption) and abnormally high volume (big players
    offloading). Confirmation requires price NOT returning into the wick.
    """
    detected: bool
    candle_idx: int
    wick_direction: str         # 'up' or 'down' — direction of the large wick
    body_ratio: float           # body / total range (small = good SVC)
    volume_ratio: float         # vs average volume
    price_returned_to_wick: bool  # False = confirmed SVC (bullish signal)


@dataclass
class LevelAnalysis:
    """Complete MM level analysis output.

    Provides the current level count, all identified levels, board
    meetings between them, SVC detection, and volume health signals.
    """
    current_level: int                          # 0-4 (0 = no levels detected)
    levels: list[LevelInfo] = field(default_factory=list)
    board_meetings: list[BoardMeetingInfo] = field(default_factory=list)
    svc: SVCResult | None = None
    direction: str = ""                         # 'bullish' or 'bearish'
    is_extended: bool = False                   # True if level 4 detected
    volume_degrading: bool = False              # True if vectors weakening


# ---------------------------------------------------------------------------
# LevelTracker
# ---------------------------------------------------------------------------

class LevelTracker:
    """Counts MM Method levels from PVSRA-classified price action.

    The tracker scans OHLCV data for clusters of high-volume vector
    candles separated by low-volume consolidation (board meetings).
    Each cluster represents one MM level in the 3-level cycle.

    Usage::

        tracker = LevelTracker()
        analysis = tracker.analyze(ohlcv_df, direction="bearish")
        if analysis.current_level >= 3 and analysis.svc and analysis.svc.detected:
            # Level 3 + SVC = high probability reversal
            ...
    """

    def __init__(self, ema_framework: "EMAFramework" = None) -> None:
        """Initialize the level tracker.

        Args:
            ema_framework: Optional EMA framework for enhanced EMA-based
                checks (board meeting flattening, level/EMA alignment).
                If None, EMA calculations are done internally with pandas.
        """
        self.ema_framework = ema_framework

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self, ohlc: pd.DataFrame, direction: str | None = None
    ) -> LevelAnalysis:
        """Analyze price action for MM Method levels.

        This is the main entry point. It classifies PVSRA vectors,
        detects the trend direction (if not supplied), counts levels,
        identifies board meetings between them, and checks for a
        Stopping Volume Candle.

        Args:
            ohlc: DataFrame with columns: open, high, low, close, volume.
                  Must have at least ~30 rows for meaningful analysis.
            direction: Force direction ('bullish' or 'bearish'). If None,
                inferred from price action over the analysis window.

        Returns:
            LevelAnalysis with all findings.
        """
        if ohlc is None or ohlc.empty or len(ohlc) < PVSRA_LOOKBACK + 5:
            logger.debug("mm_levels.analyze: insufficient data")
            return LevelAnalysis(current_level=0)

        # Ensure numeric types
        ohlc = ohlc.copy()
        for col in ("open", "high", "low", "close", "volume"):
            if col in ohlc.columns:
                ohlc[col] = pd.to_numeric(ohlc[col], errors="coerce")

        # Step 1: PVSRA classification
        classified = self.classify_pvsra(ohlc)

        # Step 2: Infer direction if not provided
        if direction is None:
            direction = self._infer_direction(classified)

        # Step 3: Count levels
        levels = self.count_levels(classified, direction)

        # Step 4: Detect board meetings between levels
        board_meetings: list[BoardMeetingInfo] = []
        for i in range(len(levels) - 1):
            bm_start = levels[i].end_idx + 1
            bm_end = levels[i + 1].start_idx - 1
            if bm_end > bm_start:
                bm = self.detect_board_meeting(classified, bm_start, bm_end)
                if bm is not None:
                    board_meetings.append(bm)

        # Step 5: Detect stopping volume candle
        svc = self.detect_stopping_volume(classified)

        # Step 6: Determine volume degradation
        volume_degrading = self._check_volume_degradation(levels)

        current_level = len(levels)
        is_extended = current_level >= 4

        analysis = LevelAnalysis(
            current_level=current_level,
            levels=levels,
            board_meetings=board_meetings,
            svc=svc,
            direction=direction,
            is_extended=is_extended,
            volume_degrading=volume_degrading,
        )

        logger.info(
            "mm_levels.analyze",
            current_level=current_level,
            direction=direction,
            num_board_meetings=len(board_meetings),
            svc_detected=svc.detected if svc else False,
            volume_degrading=volume_degrading,
            is_extended=is_extended,
        )

        return analysis

    def classify_pvsra(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """Add PVSRA classification columns to the DataFrame.

        Columns added:
        - pvsra_type: 'vector_200', 'vector_150', or 'normal'
        - pvsra_color: 'green', 'blue', 'red', 'magenta', or 'gray'
        - is_vector: bool

        PVSRA compares each candle's volume against the rolling average
        of the prior 10 candles. A "vector" candle has abnormally high
        volume indicating institutional activity.

        Args:
            ohlc: DataFrame with open, high, low, close, volume columns.

        Returns:
            DataFrame with PVSRA columns added.
        """
        df = ohlc.copy()
        volume = df["volume"].astype(float)

        # Rolling average of prior N candles (shift(1) so current candle
        # is not included in its own baseline)
        avg_vol = volume.shift(1).rolling(window=PVSRA_LOOKBACK, min_periods=3).mean()

        # Volume ratio: current candle volume / average
        vol_ratio = volume / avg_vol.replace(0, np.nan)

        # Classify vector type
        pvsra_type = pd.Series("normal", index=df.index)
        pvsra_type[vol_ratio >= VECTOR_200_THRESHOLD] = "vector_200"
        pvsra_type[
            (vol_ratio >= VECTOR_150_THRESHOLD) & (vol_ratio < VECTOR_200_THRESHOLD)
        ] = "vector_150"

        # Determine candle direction
        is_bullish = df["close"] > df["open"]

        # Assign PVSRA colors
        pvsra_color = pd.Series("gray", index=df.index)
        # Bullish vectors
        pvsra_color[(pvsra_type == "vector_200") & is_bullish] = "green"
        pvsra_color[(pvsra_type == "vector_150") & is_bullish] = "blue"
        # Bearish vectors
        pvsra_color[(pvsra_type == "vector_200") & ~is_bullish] = "red"
        pvsra_color[(pvsra_type == "vector_150") & ~is_bullish] = "magenta"

        df["pvsra_type"] = pvsra_type
        df["pvsra_color"] = pvsra_color
        df["is_vector"] = (pvsra_type == "vector_200") | (pvsra_type == "vector_150")
        df["vol_ratio"] = vol_ratio

        return df

    def count_levels(
        self, ohlc: pd.DataFrame, direction: str
    ) -> list[LevelInfo]:
        """Identify individual MM levels from vector candle clusters.

        A level is defined as a cluster of vector candles (possibly with
        small gaps of up to CLUSTER_GAP_MAX normal candles) that move
        price in the specified direction. Each cluster = one level push.

        The algorithm:
        1. Find all vector candle indices
        2. Group them into clusters (consecutive or near-consecutive)
        3. Filter clusters that move in the target direction
        4. Assign level numbers 1, 2, 3, (4)

        Args:
            ohlc: PVSRA-classified DataFrame (must have is_vector, pvsra_type).
            direction: 'bullish' or 'bearish'.

        Returns:
            List of LevelInfo, ordered by level number.
        """
        if "is_vector" not in ohlc.columns:
            logger.warning("count_levels called without PVSRA classification")
            return []

        # Get indices of all vector candles
        vector_mask = ohlc["is_vector"].values
        vector_indices = np.where(vector_mask)[0]

        if len(vector_indices) == 0:
            return []

        # Group vector candles into clusters
        clusters = self._cluster_vectors(vector_indices)

        # Build LevelInfo for each qualifying cluster
        levels: list[LevelInfo] = []
        level_num = 0

        for cluster_indices in clusters:
            if len(cluster_indices) == 0:
                continue

            start_idx = int(cluster_indices[0])
            end_idx = int(cluster_indices[-1])

            # Calculate the price move across the cluster
            # Use the range from first candle's open to last candle's close
            # to capture the full directional push
            cluster_open = float(ohlc.iloc[start_idx]["open"])
            cluster_close = float(ohlc.iloc[end_idx]["close"])
            cluster_high = float(ohlc.iloc[start_idx:end_idx + 1]["high"].max())
            cluster_low = float(ohlc.iloc[start_idx:end_idx + 1]["low"].min())

            # Determine cluster direction
            if cluster_close > cluster_open:
                cluster_dir = "bullish"
                magnitude = (
                    (cluster_high - cluster_open) / cluster_open * 100
                    if cluster_open > 0
                    else 0.0
                )
            else:
                cluster_dir = "bearish"
                magnitude = (
                    (cluster_open - cluster_low) / cluster_open * 100
                    if cluster_open > 0
                    else 0.0
                )

            # Only count clusters that move in the target direction
            if cluster_dir != direction:
                continue

            # Determine dominant vector type in this cluster
            cluster_slice = ohlc.iloc[start_idx:end_idx + 1]
            n_200 = (cluster_slice["pvsra_type"] == "vector_200").sum()
            n_150 = (cluster_slice["pvsra_type"] == "vector_150").sum()
            dominant_vector = "vector_200" if n_200 >= n_150 else "vector_150"

            level_num += 1
            levels.append(
                LevelInfo(
                    level_number=level_num,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    direction=direction,
                    magnitude=round(magnitude, 4),
                    volume_confirmed=True,
                    vector_type=dominant_vector,
                )
            )

            # Cap at 4 levels (Extended Rise/Drop)
            if level_num >= 4:
                break

        return levels

    def detect_board_meeting(
        self,
        ohlc: pd.DataFrame,
        start_idx: int,
        end_idx: int | None = None,
    ) -> BoardMeetingInfo | None:
        """Detect a board meeting (consolidation) between levels.

        Board meetings are where MM accumulates contracts before the
        next push. Characteristics:
        - Low volume (few or no vector candles)
        - Price range contracts
        - EMAs flatten / converge
        - May contain brief stop hunts (wicks outside range)

        Args:
            ohlc: PVSRA-classified DataFrame.
            start_idx: Start of the candidate board meeting zone.
            end_idx: End of the zone (if None, scans forward up to 20 candles).

        Returns:
            BoardMeetingInfo if consolidation detected, else None.
        """
        if end_idx is None:
            end_idx = min(start_idx + 20, len(ohlc) - 1)

        if end_idx <= start_idx:
            return None

        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(ohlc) - 1, end_idx)

        duration = end_idx - start_idx + 1
        if duration < BOARD_MEETING_MIN:
            return None

        segment = ohlc.iloc[start_idx:end_idx + 1]

        # Board meetings should have low vector density
        n_vectors = segment["is_vector"].sum() if "is_vector" in segment.columns else 0
        vector_density = n_vectors / duration
        if vector_density > 0.4:
            # Too many vectors — this is a level push, not a board meeting
            return None

        # Detect stop hunts: wicks that exceed the body range
        highs = segment["high"].astype(float)
        lows = segment["low"].astype(float)
        opens = segment["open"].astype(float)
        closes = segment["close"].astype(float)

        body_high = pd.concat([opens, closes], axis=1).max(axis=1)
        body_low = pd.concat([opens, closes], axis=1).min(axis=1)
        range_high = body_high.max()
        range_low = body_low.min()

        # A stop hunt = wick pokes outside the body range
        contains_stop_hunt = bool(
            (highs > range_high * 1.001).any() or (lows < range_low * 0.999).any()
        )

        # Check EMA flattening
        ema_flattening = self._check_ema_flattening(ohlc, start_idx, end_idx)

        return BoardMeetingInfo(
            start_idx=start_idx,
            end_idx=end_idx,
            duration_candles=duration,
            contains_stop_hunt=contains_stop_hunt,
            ema_flattening=ema_flattening,
        )

    def detect_stopping_volume(self, ohlc: pd.DataFrame) -> SVCResult | None:
        """Find a Stopping Volume Candle in recent price action.

        An SVC signals trend exhaustion. It is characterized by:
        1. Small body relative to total range (body_ratio <= SVC_BODY_RATIO_MAX)
        2. Large wick in the prior trend direction (absorption)
        3. Very high volume (>= SVC_VOLUME_MIN times average)
        4. Ideally appears at or near Level 3

        Confirmation: price must FAIL to return into the wick zone in
        subsequent candles. If price re-enters the wick, the SVC is
        invalidated.

        Args:
            ohlc: PVSRA-classified DataFrame.

        Returns:
            SVCResult if a candidate SVC found, else None.
        """
        if ohlc is None or len(ohlc) < PVSRA_LOOKBACK + SVC_WICK_RETURN_CANDLES:
            return None

        # We scan the recent portion of the dataframe (last 30 candles)
        # looking for SVC candidates
        scan_start = max(PVSRA_LOOKBACK, len(ohlc) - 30)
        best_svc: SVCResult | None = None
        best_vol_ratio = 0.0

        for i in range(scan_start, len(ohlc)):
            row = ohlc.iloc[i]
            o, h, l, c = (
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
            )
            total_range = h - l
            if total_range <= 0:
                continue

            body = abs(c - o)
            body_ratio = body / total_range

            # Must have small body
            if body_ratio > SVC_BODY_RATIO_MAX:
                continue

            # Volume check
            vol_ratio = float(row.get("vol_ratio", 0.0))
            if pd.isna(vol_ratio) or vol_ratio < SVC_VOLUME_MIN:
                continue

            # Determine wick direction (which wick is larger)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            wick_direction = "up" if upper_wick > lower_wick else "down"

            # The dominant wick should be substantial (> 40% of range)
            dominant_wick = max(upper_wick, lower_wick)
            if dominant_wick / total_range < 0.40:
                continue

            # Check if price returned into the wick zone
            price_returned = self._check_wick_return(
                ohlc, i, wick_direction, h, l, o, c
            )

            # Keep the strongest SVC candidate
            if vol_ratio > best_vol_ratio:
                best_vol_ratio = vol_ratio
                best_svc = SVCResult(
                    detected=True,
                    candle_idx=i,
                    wick_direction=wick_direction,
                    body_ratio=round(body_ratio, 4),
                    volume_ratio=round(vol_ratio, 4),
                    price_returned_to_wick=price_returned,
                )

        if best_svc is not None:
            logger.info(
                "mm_levels.svc_detected",
                candle_idx=best_svc.candle_idx,
                wick_direction=best_svc.wick_direction,
                body_ratio=best_svc.body_ratio,
                volume_ratio=best_svc.volume_ratio,
                confirmed=not best_svc.price_returned_to_wick,
            )

        return best_svc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer_direction(self, ohlc: pd.DataFrame) -> str:
        """Infer the dominant trend direction from recent price action.

        Uses a simple comparison of the 20-period EMA slope over the
        most recent candles.
        """
        close = ohlc["close"].astype(float)
        if len(close) < 20:
            # Fallback: compare first and last close
            return "bullish" if close.iloc[-1] > close.iloc[0] else "bearish"

        ema20 = close.ewm(span=20, adjust=False).mean()
        recent_slope = ema20.iloc[-1] - ema20.iloc[-5] if len(ema20) >= 5 else 0
        return "bullish" if recent_slope > 0 else "bearish"

    def _cluster_vectors(self, vector_indices: np.ndarray) -> list[list[int]]:
        """Group vector candle indices into clusters.

        Two vectors belong to the same cluster if they are separated by
        at most CLUSTER_GAP_MAX normal candles. This accounts for brief
        pauses within a single level push.
        """
        if len(vector_indices) == 0:
            return []

        clusters: list[list[int]] = []
        current_cluster: list[int] = [int(vector_indices[0])]

        for i in range(1, len(vector_indices)):
            gap = int(vector_indices[i]) - int(vector_indices[i - 1])
            if gap <= CLUSTER_GAP_MAX + 1:
                # Close enough — same cluster
                current_cluster.append(int(vector_indices[i]))
            else:
                # Gap too large — start a new cluster
                clusters.append(current_cluster)
                current_cluster = [int(vector_indices[i])]

        # Don't forget the last cluster
        clusters.append(current_cluster)

        return clusters

    def _check_volume_degradation(self, levels: list[LevelInfo]) -> bool:
        """Check if volume is degrading across levels.

        Volume degradation occurs when earlier levels had 200% vectors
        but later levels only have 150% vectors. This indicates MM is
        running out of fuel and a pause/reversal is likely.

        A guaranteed pause signal: Level 1 = vector_200, Level 3 = vector_150.
        """
        if len(levels) < 2:
            return False

        # Compare the first level's vector type to the last level's
        first_type = levels[0].vector_type
        last_type = levels[-1].vector_type

        if first_type == "vector_200" and last_type == "vector_150":
            return True

        # Also check for progressive degradation across all levels
        type_scores = {"vector_200": 2, "vector_150": 1}
        scores = [type_scores.get(lv.vector_type, 0) for lv in levels]
        if len(scores) >= 3:
            # Monotonically non-increasing = degrading
            return all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)) and (
                scores[0] > scores[-1]
            )

        return False

    def _check_ema_flattening(
        self, ohlc: pd.DataFrame, start_idx: int, end_idx: int
    ) -> bool:
        """Check if EMAs are flattening in a given segment.

        Uses the 50 EMA. If the slope (normalized change per candle) is
        below the threshold, EMAs are considered flat — characteristic of
        a board meeting.
        """
        close = ohlc["close"].astype(float)

        if len(close) < 50:
            return False

        ema50 = close.ewm(span=50, adjust=False).mean()

        # Get EMA values at the segment boundaries
        if end_idx >= len(ema50) or start_idx >= len(ema50):
            return False

        ema_start = ema50.iloc[start_idx]
        ema_end = ema50.iloc[end_idx]
        duration = end_idx - start_idx

        if duration == 0 or ema_start == 0:
            return False

        # Normalized slope: percentage change per candle
        slope = abs(ema_end - ema_start) / ema_start / duration
        return slope < EMA_FLAT_THRESHOLD

    def _check_wick_return(
        self,
        ohlc: pd.DataFrame,
        svc_idx: int,
        wick_direction: str,
        h: float,
        l: float,
        o: float,
        c: float,
    ) -> bool:
        """Check if price returned into the SVC wick zone.

        For a confirmed SVC, price should NOT return into the wick.
        - If wick is 'up' (upper wick is dominant): check if subsequent
          candles reach back up into the upper wick zone.
        - If wick is 'down' (lower wick is dominant): check if subsequent
          candles reach back down into the lower wick zone.

        Returns:
            True if price returned to wick (SVC invalidated).
            False if price did not return (SVC confirmed).
        """
        end_check = min(svc_idx + SVC_WICK_RETURN_CANDLES + 1, len(ohlc))

        if svc_idx + 1 >= len(ohlc):
            # No subsequent candles to check — cannot confirm yet
            return False

        subsequent = ohlc.iloc[svc_idx + 1 : end_check]

        if wick_direction == "up":
            # The wick zone is from max(open, close) to high
            wick_bottom = max(o, c)
            # Price returns if any subsequent candle's high enters the wick
            return bool((subsequent["high"].astype(float) > wick_bottom).any())
        else:
            # The wick zone is from low to min(open, close)
            wick_top = min(o, c)
            # Price returns if any subsequent candle's low enters the wick
            return bool((subsequent["low"].astype(float) < wick_top).any())
