"""Exchange connectors — register all connectors on import."""

# Import to trigger registration
from src.exchange.connectors.yfinance_connector import YFinanceStocksConnector, YFinanceCommoditiesConnector  # noqa: F401
