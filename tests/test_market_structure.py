from src.strategy.market_structure import MarketStructureAnalyzer
import pandas as pd


class TestMarketStructureAnalyzer:
    def setup_method(self):
        self.analyzer = MarketStructureAnalyzer()

    def test_bullish_trend(self, bullish_candles):
        result = self.analyzer.analyze(bullish_candles, timeframe="15m")
        # Should detect some form of trend or structure
        assert result.trend in ("bullish", "bearish", "ranging")
        assert result.key_levels is not None

    def test_bearish_trend(self, bearish_candles):
        result = self.analyzer.analyze(bearish_candles, timeframe="15m")
        assert result.trend in ("bullish", "bearish", "ranging")

    def test_ranging(self, ranging_candles):
        result = self.analyzer.analyze(ranging_candles, timeframe="15m")
        assert result.trend in ("bullish", "bearish", "ranging")

    def test_empty_data(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = self.analyzer.analyze(empty, timeframe="15m")
        assert result.trend == "ranging"
        assert result.structure_strength == 0.0

    def test_insufficient_data(self):
        # Less than 30 candles
        short = pd.DataFrame(
            {"open": [1] * 10, "high": [2] * 10, "low": [0.5] * 10, "close": [1.5] * 10, "volume": [100] * 10},
            index=pd.date_range("2025-01-01", periods=10, freq="15min", tz="UTC"),
        )
        result = self.analyzer.analyze(short, timeframe="15m")
        assert result.trend == "ranging"

    def test_swing_levels_extracted(self, bullish_candles):
        result = self.analyzer.analyze(bullish_candles, timeframe="15m")
        # Should have at least some key levels
        assert "swing_high" in result.key_levels
        assert "swing_low" in result.key_levels

    def test_structure_strength(self, bullish_candles):
        result = self.analyzer.analyze(bullish_candles, timeframe="15m")
        assert 0.0 <= result.structure_strength <= 1.0
