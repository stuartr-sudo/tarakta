"""Moon phase calculation for MM Method confluence (lesson 37).

Course rules:
  - Every 29.5 days the moon completes a full cycle.
  - New Moon = local TOP signal; Full Moon = local BOTTOM signal.
  - ±3 day buffer either side.
  - Other quarter/crescent phases are warning signals (lower weight).

Implementation: a self-contained ephemeris approximation. No external
astronomical libraries needed — the algorithm is accurate to < 1 day
for the relevant 20th/21st-century window, which is well within the
±3 day buffer the course uses.

Reference: Standard "synodic month" algorithm from Jean Meeus's
*Astronomical Algorithms*, simplified for trading-context precision.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math


# Synodic month (New Moon to New Moon) in days.
SYNODIC_MONTH_DAYS = 29.530588853

# Reference New Moon: 2000-01-06 18:14 UTC (well-documented).
_REF_NEW_MOON_JD = 2451549.75971  # Julian Day Number

# Course buffer — within ±3 days of a primary phase, the signal is active.
PHASE_BUFFER_DAYS = 3.0


@dataclass
class MoonPhaseInfo:
    """Current moon phase relative to trading signals."""

    phase_name: str  # "new_moon", "full_moon", "first_quarter", "third_quarter",
                    # "waxing_crescent", "waxing_gibbous", "waning_gibbous", "waning_crescent"
    illumination_pct: float  # 0 (new) to 100 (full)
    days_since_new: float    # 0 to ~29.53
    is_primary: bool          # True if within ±3d of new or full moon
    signal: str               # "local_top" (near new moon), "local_bottom" (near full),
                             # "warning_down", "warning_up", or "neutral"
    signal_strength: float   # 0..1 — 1 at exact phase, tapers to 0 at buffer edge
    buffer_days_remaining: float  # Days until buffer exit (or since entry if past)


def _julian_day(dt: datetime) -> float:
    """Convert a datetime to a Julian Day Number."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    y = dt.year
    m = dt.month
    d = dt.day + (dt.hour + (dt.minute + dt.second / 60.0) / 60.0) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5
    return jd


def compute_moon_phase(now: datetime | None = None) -> MoonPhaseInfo:
    """Compute moon phase info for `now` (UTC default).

    Args:
        now: datetime (UTC aware preferred). Defaults to current UTC time.

    Returns:
        MoonPhaseInfo with phase name, illumination, signal, and buffer window.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    jd = _julian_day(now)
    days_since_ref = jd - _REF_NEW_MOON_JD
    synodic_cycles = days_since_ref / SYNODIC_MONTH_DAYS
    days_since_new = (synodic_cycles - math.floor(synodic_cycles)) * SYNODIC_MONTH_DAYS

    # Illumination: approximately (1 - cos(angle)) / 2 * 100
    angle_rad = 2 * math.pi * days_since_new / SYNODIC_MONTH_DAYS
    illumination_pct = round((1 - math.cos(angle_rad)) * 50, 2)

    # Map days into the 8 phases using standard boundaries
    # 0 → new moon
    # 7.38 → first quarter
    # 14.77 → full moon
    # 22.15 → last quarter
    # (8ths of the cycle)
    q = SYNODIC_MONTH_DAYS / 4.0  # ~7.38 days
    if days_since_new < q / 2 or days_since_new > (SYNODIC_MONTH_DAYS - q / 2):
        phase_name = "new_moon"
    elif days_since_new < q:
        phase_name = "waxing_crescent"
    elif days_since_new < q + q / 2:
        phase_name = "first_quarter"
    elif days_since_new < 2 * q:
        phase_name = "waxing_gibbous"
    elif days_since_new < 2 * q + q / 2:
        phase_name = "full_moon"
    elif days_since_new < 3 * q:
        phase_name = "waning_gibbous"
    elif days_since_new < 3 * q + q / 2:
        phase_name = "third_quarter"
    else:
        phase_name = "waning_crescent"

    # Distance to the nearest PRIMARY phase (new or full)
    dist_to_new = min(days_since_new, SYNODIC_MONTH_DAYS - days_since_new)
    dist_to_full = abs(days_since_new - SYNODIC_MONTH_DAYS / 2)
    nearest_primary_dist = min(dist_to_new, dist_to_full)
    is_primary = nearest_primary_dist <= PHASE_BUFFER_DAYS

    # Signal per course: new moon = local top, full moon = local bottom
    if is_primary and dist_to_new <= dist_to_full:
        signal = "local_top"
        signal_strength = round(1 - (dist_to_new / PHASE_BUFFER_DAYS), 3)
    elif is_primary and dist_to_full < dist_to_new:
        signal = "local_bottom"
        signal_strength = round(1 - (dist_to_full / PHASE_BUFFER_DAYS), 3)
    elif phase_name in ("waxing_crescent", "first_quarter", "waxing_gibbous"):
        signal = "warning_up"
        signal_strength = 0.3
    elif phase_name in ("waning_gibbous", "third_quarter", "waning_crescent"):
        signal = "warning_down"
        signal_strength = 0.3
    else:
        signal = "neutral"
        signal_strength = 0.0

    buffer_remaining = round(max(0.0, PHASE_BUFFER_DAYS - nearest_primary_dist), 3)

    return MoonPhaseInfo(
        phase_name=phase_name,
        illumination_pct=illumination_pct,
        days_since_new=round(days_since_new, 3),
        is_primary=is_primary,
        signal=signal,
        signal_strength=signal_strength,
        buffer_days_remaining=buffer_remaining,
    )


def moon_signal_aligns_with_direction(info: MoonPhaseInfo, direction: str) -> bool:
    """Return True if the moon signal supports the requested trade direction.

    Course lesson 37: new moon near local top supports shorts; full moon near
    local bottom supports longs. Warning phases are weaker cues.
    """
    if not info.is_primary:
        return False
    if direction == "short":
        return info.signal == "local_top"
    if direction == "long":
        return info.signal == "local_bottom"
    return False
