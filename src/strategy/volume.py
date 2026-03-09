from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

RVOL_LOOKBACK = 20  # candles for average volume baseline


@dataclass
class VolumeProfile:
    """Volume analysis results for a single timeframe."""

    relative_volume: float  # current volume / average volume (1.0 = normal)
    volume_trend: str  # "increasing", "decreasing", "flat"
    displacement_detected: bool  # large candle + high volume = institutional move
    displacement_direction: str | None  # "bullish" or "bearish"
    displacement_strength: float  # 0-1 scale
    displacement_candle_idx: int | None  # absolute index in DataFrame of the displacement candle
    volume_at_ob: float  # volume ratio at the nearest order block (if any)
    high_volume_nodes: list[float]  # price levels with concentrated volume


@dataclass
class VolumeResult:
    """Aggregated volume analysis across timeframes."""

    profiles: dict[str, VolumeProfile]  # keyed by timeframe
    overall_volume_score: float  # 0-10 score for confluence
    reasons: list[str] = field(default_factory=list)


class VolumeAnalyzer:
    """Analyzes volume for confirmation of SMC setups.

    Key concepts:
    - Relative Volume (RVOL): Compares current volume to the 20-period average.
      RVOL > 1.5 = above-average interest. RVOL > 2.5 = institutional activity.
    - Displacement: Large-body candles with high volume indicate genuine
      institutional commitment. A candle whose body is > 1.5x ATR and volume
      is > 1.5x average is a displacement candle.
    - Volume at Order Blocks: OBs formed with high volume are more significant.
    - Volume Trend: Rising volume in the direction of the trend confirms momentum.
    """

    def __init__(self, rvol_lookback: int = RVOL_LOOKBACK) -> None:
        self.rvol_lookback = rvol_lookback

    def analyze(self, ohlc: pd.DataFrame) -> VolumeProfile:
        """Analyze volume characteristics for a single timeframe."""
        if ohlc.empty or len(ohlc) < self.rvol_lookback + 5:
            return self._empty_profile()

        volume = ohlc["volume"].astype(float)
        close = ohlc["close"].astype(float)
        open_ = ohlc["open"].astype(float)
        high = ohlc["high"].astype(float)
        low = ohlc["low"].astype(float)

        # --- Relative Volume ---
        avg_vol = volume.rolling(self.rvol_lookback).mean()
        current_vol = float(volume.iloc[-1])
        avg_current = float(avg_vol.iloc[-1]) if pd.notna(avg_vol.iloc[-1]) else 1.0
        rvol = current_vol / avg_current if avg_current > 0 else 0.0

        # --- Volume Trend (last 5 candles vs prior 5) ---
        if len(volume) >= 10:
            recent_avg = float(volume.iloc[-5:].mean())
            prior_avg = float(volume.iloc[-10:-5].mean())
            if prior_avg > 0:
                vol_change = (recent_avg - prior_avg) / prior_avg
                if vol_change > 0.2:
                    vol_trend = "increasing"
                elif vol_change < -0.2:
                    vol_trend = "decreasing"
                else:
                    vol_trend = "flat"
            else:
                vol_trend = "flat"
        else:
            vol_trend = "flat"

        # --- Displacement Detection ---
        # A displacement candle has: body > 1.5x ATR AND volume > 1.5x average
        atr = self._atr(high, low, close, period=14)
        displacement = False
        disp_direction = None
        disp_strength = 0.0
        disp_candle_idx: int | None = None

        if len(ohlc) >= 5 and atr is not None:
            # Check last 8 candles for displacement (extended for pullback detection)
            for i in range(-8, 0):
                idx = len(ohlc) + i
                if idx < 0:
                    continue
                body = abs(float(close.iloc[idx]) - float(open_.iloc[idx]))
                candle_vol = float(volume.iloc[idx])
                atr_val = float(atr.iloc[idx]) if pd.notna(atr.iloc[idx]) else 0
                avg_v = float(avg_vol.iloc[idx]) if pd.notna(avg_vol.iloc[idx]) else 1

                if atr_val > 0 and avg_v > 0:
                    body_ratio = body / atr_val
                    vol_ratio = candle_vol / avg_v

                    if body_ratio > 1.5 and vol_ratio > 1.5:
                        # --- Volume Decay Check ---
                        # After displacement, check if the next 2-3 candles
                        # sustained volume.  If volume drops >50% immediately,
                        # the displacement was a one-off liquidation grab.
                        decay_failed = False
                        following_candles = min(3, len(ohlc) - idx - 1)
                        if following_candles >= 2:
                            follow_vols = [
                                float(volume.iloc[idx + k])
                                for k in range(1, following_candles + 1)
                                if idx + k < len(ohlc)
                            ]
                            if follow_vols:
                                avg_follow = sum(follow_vols) / len(follow_vols)
                                if avg_follow < candle_vol * 0.5:
                                    decay_failed = True  # Volume collapsed

                        if decay_failed:
                            # Skip this displacement — volume didn't sustain
                            continue

                        displacement = True
                        disp_direction = "bullish" if float(close.iloc[idx]) > float(open_.iloc[idx]) else "bearish"
                        # Strength: how extreme the move was (0-1 scale)
                        disp_strength = min(1.0, (body_ratio - 1.0) * 0.3 + (vol_ratio - 1.0) * 0.3)
                        disp_candle_idx = idx
                        break  # Use most recent displacement

        # --- High Volume Nodes (simple: find price levels with volume spikes) ---
        hvn = self._high_volume_nodes(ohlc)

        return VolumeProfile(
            relative_volume=round(rvol, 2),
            volume_trend=vol_trend,
            displacement_detected=displacement,
            displacement_direction=disp_direction,
            displacement_strength=round(disp_strength, 3),
            displacement_candle_idx=disp_candle_idx,
            volume_at_ob=0.0,  # filled externally when OB data is available
            high_volume_nodes=hvn,
        )

    def score_volume(
        self,
        profiles: dict[str, VolumeProfile],
        direction: str | None,
    ) -> VolumeResult:
        """Score volume analysis across multiple timeframes.

        Returns up to 10 points for the confluence engine:
        - Displacement confirmation: 0-5 pts
        - Relative volume confirmation: 0-3 pts
        - Volume trend alignment: 0-2 pts
        """
        if direction is None:
            return VolumeResult(profiles=profiles, overall_volume_score=0)

        score = 0.0
        reasons: list[str] = []

        # --- Displacement (0-7 pts) ---
        for tf in ["1h"]:
            p = profiles.get(tf)
            if not p or not p.displacement_detected:
                continue
            if p.displacement_direction == direction:
                pts = 4 + p.displacement_strength * 3  # 4-7 pts
                score += pts
                reasons.append(f"Displacement {direction} on {tf} (strength {p.displacement_strength:.0%})")
                break  # Only count once

        # --- Relative Volume (0-4 pts) ---
        # Check entry timeframes for elevated volume
        best_rvol = 0.0
        best_rvol_tf = ""
        for tf in ["1h"]:
            p = profiles.get(tf)
            if p and p.relative_volume > best_rvol:
                best_rvol = p.relative_volume
                best_rvol_tf = tf

        if best_rvol >= 2.5:
            score += 4
            reasons.append(f"Very high RVOL {best_rvol:.1f}x on {best_rvol_tf}")
        elif best_rvol >= 1.5:
            score += 2
            reasons.append(f"Elevated RVOL {best_rvol:.1f}x on {best_rvol_tf}")
        elif best_rvol >= 1.2:
            score += 1
            reasons.append(f"RVOL {best_rvol:.1f}x on {best_rvol_tf}")

        # --- Volume Trend (0-4 pts) ---
        # Increasing volume on HTF confirms institutional participation
        for tf in ["1h", "4h"]:
            p = profiles.get(tf)
            if not p:
                continue
            if p.volume_trend == "increasing":
                score += 2
                reasons.append(f"Volume increasing on {tf}")

        score = min(score, 10)  # cap at weight allocation

        return VolumeResult(
            profiles=profiles,
            overall_volume_score=round(score, 1),
            reasons=reasons,
        )

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series | None:
        """Average True Range."""
        if len(close) < period + 1:
            return None
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean()

    def _high_volume_nodes(self, ohlc: pd.DataFrame, bins: int = 20) -> list[float]:
        """Find price levels where volume is concentrated (volume profile)."""
        if len(ohlc) < 20:
            return []

        try:
            prices = ohlc["close"].astype(float).values
            volumes = ohlc["volume"].astype(float).values

            price_min, price_max = np.min(prices), np.max(prices)
            if price_max == price_min:
                return []

            edges = np.linspace(price_min, price_max, bins + 1)
            vol_bins = np.zeros(bins)

            for p, v in zip(prices, volumes):
                bin_idx = int((p - price_min) / (price_max - price_min) * (bins - 1))
                bin_idx = max(0, min(bin_idx, bins - 1))
                vol_bins[bin_idx] += v

            # Find bins above 1.5x average
            avg_bin_vol = np.mean(vol_bins)
            if avg_bin_vol <= 0:
                return []

            hvn = []
            for i, v in enumerate(vol_bins):
                if v > avg_bin_vol * 1.5:
                    mid_price = (edges[i] + edges[i + 1]) / 2
                    hvn.append(round(float(mid_price), 6))

            return hvn[:5]  # top 5
        except Exception:
            return []

    def _empty_profile(self) -> VolumeProfile:
        return VolumeProfile(
            relative_volume=0.0,
            volume_trend="flat",
            displacement_detected=False,
            displacement_direction=None,
            displacement_strength=0.0,
            displacement_candle_idx=None,
            volume_at_ob=0.0,
            high_volume_nodes=[],
        )
