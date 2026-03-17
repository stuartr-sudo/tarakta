"""Direction format utilities.

The codebase uses multiple direction formats:
  - Scanner/refiner: 'bullish'/'bearish'
  - Agents: 'LONG'/'SHORT'
  - Sweep direction: 'swing_low' (= bullish/LONG) / 'swing_high' (= bearish/SHORT)

These functions provide safe conversion between ALL formats.
"""

_LONG_VARIANTS = ("bullish", "long", "swing_low")
_SHORT_VARIANTS = ("bearish", "short", "swing_high")


def to_long_short(direction: str | None) -> str:
    """Convert any direction format to LONG/SHORT."""
    if not direction:
        return ""
    d = direction.strip().lower()
    if d in _LONG_VARIANTS:
        return "LONG"
    if d in _SHORT_VARIANTS:
        return "SHORT"
    return ""


def to_bullish_bearish(direction: str | None) -> str:
    """Convert any direction format to bullish/bearish."""
    if not direction:
        return ""
    d = direction.strip().lower()
    if d in _LONG_VARIANTS:
        return "bullish"
    if d in _SHORT_VARIANTS:
        return "bearish"
    return ""


def is_long(direction: str | None) -> bool:
    """Check if direction is long/bullish in any format."""
    if not direction:
        return False
    return direction.strip().lower() in _LONG_VARIANTS
