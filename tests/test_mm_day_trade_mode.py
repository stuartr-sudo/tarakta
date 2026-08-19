"""Day-trade mode (config.mm_day_trade_mode) — course Lesson 16 [45:30-46:00],
Lesson 5 [06:00-06:30], Lesson 10 [45:00]. See config.py for the rule text."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from src.strategy.mm_engine import MMEngine
from src.strategy.mm_sessions import MMSessionAnalyzer as SessionAnalyzer

# 2026-08-18 is a Tuesday. 5pm New York (EDT, UTC-4) == 21:00 UTC.
DEAD_GAP_UTC = datetime(2026, 8, 18, 21, 30, tzinfo=timezone.utc)
US_SESSION_UTC = datetime(2026, 8, 18, 18, 0, tzinfo=timezone.utc)


def _engine(day_trade: bool) -> MMEngine:
    eng = MMEngine(exchange=None, repo=None, candle_manager=None, config=None)
    eng.day_trade_mode = day_trade
    eng.session_analyzer = SessionAnalyzer()
    return eng


def test_session_analyzer_dead_zone_anchor():
    sa = SessionAnalyzer()
    assert sa.is_dead_zone(DEAD_GAP_UTC)
    assert not sa.is_dead_zone(US_SESSION_UTC)


def test_config_default_is_on_and_engine_reads_it():
    from src.config import Settings  # noqa: WPS433 — local import keeps test light
    assert Settings.model_fields["mm_day_trade_mode"].default is True
    eng = MMEngine(exchange=None, repo=None, candle_manager=None,
                   config=SimpleNamespace(mm_day_trade_mode=False))
    assert eng.day_trade_mode is False


def _eod_rule(eng: MMEngine, pos, now: datetime) -> str | None:
    """Mirror of the engine's day_trade_eod predicate (kept in lockstep with
    _manage_position so the rule is unit-testable without an exchange)."""
    formation_tf = str(getattr(pos, "formation_timeframe", "1h") or "1h").lower()
    if not (eng.day_trade_mode and formation_tf in {"1h", "15m"}
            and eng.session_analyzer.is_dead_zone(now)):
        return None
    if pos.current_level >= 1 and pos.sl_moved_to_breakeven:
        return "hold"
    return "day_trade_eod"


def test_eod_exit_fires_for_1h_position_in_dead_gap():
    pos = SimpleNamespace(formation_timeframe="1h", current_level=0,
                          sl_moved_to_breakeven=False)
    assert _eod_rule(_engine(True), pos, DEAD_GAP_UTC) == "day_trade_eod"
    assert _eod_rule(_engine(True), pos, US_SESSION_UTC) is None     # not yet EOD
    assert _eod_rule(_engine(False), pos, DEAD_GAP_UTC) is None      # mode off


def test_eod_exit_spares_4h_swing_and_stop_in_profit():
    swing = SimpleNamespace(formation_timeframe="4h", current_level=0,
                            sl_moved_to_breakeven=False)
    assert _eod_rule(_engine(True), swing, DEAD_GAP_UTC) is None
    held = SimpleNamespace(formation_timeframe="1h", current_level=1,
                           sl_moved_to_breakeven=True)
    assert _eod_rule(_engine(True), held, DEAD_GAP_UTC) == "hold"


def test_engine_source_wires_the_rule():
    """Guard against the rule being silently dropped from _manage_position
    (the unused-config failure class from CLAUDE.md)."""
    import inspect
    src = inspect.getsource(MMEngine._manage_position)
    assert '"day_trade_eod"' in src
    assert "is_dead_zone(now)" in src
    cyc = inspect.getsource(MMEngine._cycle)
    # Dead gap still blocks NEW entries, but management runs first in day-trade mode.
    assert cyc.index("await self._manage_position(symbol)") < cyc.index(
        'if session.session_name == "dead_zone":\n            # Day-trade mode'
    )
    scan = inspect.getsource(MMEngine._analyze_pair)
    assert "mm_day_trade_1h_preferred" in scan
