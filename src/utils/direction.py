"""Direction format utilities.

The codebase uses 'bullish'/'bearish' in scanner/refiner and 'LONG'/'SHORT' in agents.
These functions provide safe conversion between formats.
"""


def to_long_short(direction: str | None) -> str:
    """Convert any direction format to LONG/SHORT."""
    if not direction:
        return ""
    d = direction.strip().lower()
    if d in ("bullish", "long"):
        return "LONG"
    if d in ("bearish", "short"):
        return "SHORT"
    return ""


def to_bullish_bearish(direction: str | None) -> str:
    """Convert any direction format to bullish/bearish."""
    if not direction:
        return ""
    d = direction.strip().lower()
    if d in ("bullish", "long"):
        return "bullish"
    if d in ("bearish", "short"):
        return "bearish"
    return ""


def is_long(direction: str | None) -> bool:
    """Check if direction is long/bullish in any format."""
    if not direction:
        return False
    return direction.strip().lower() in ("bullish", "long")
