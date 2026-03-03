from __future__ import annotations

from datetime import datetime, timezone

from src.exchange.models import Position, TakeProfitTier
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EngineState:
    """In-memory engine state with serialization for crash recovery."""

    def __init__(
        self,
        mode: str = "paper",
        status: str = "running",
        initial_balance: float = 100.0,
    ) -> None:
        self.mode = mode
        self.status = status  # "running", "paused", "circuit_break", "shutdown"
        self.current_balance = initial_balance
        self.daily_start_balance = initial_balance
        self.peak_balance = initial_balance
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.open_positions: dict[str, Position] = {}
        self.last_scan_time: datetime | None = None
        self.cycle_count: int = 0
        self.errors_consecutive: int = 0

    @classmethod
    def from_db(cls, data: dict, mode: str = "paper") -> EngineState:
        """Restore state from DB record."""
        state = cls(
            mode=data.get("mode", mode),
            status=data.get("status", "running"),
            initial_balance=float(data.get("current_balance", 100.0)),
        )
        state.current_balance = float(data.get("current_balance", 100.0))
        state.daily_start_balance = float(data.get("daily_start_bal", 100.0))
        state.peak_balance = float(data.get("peak_balance", 100.0))
        state.daily_pnl = float(data.get("daily_pnl_usd", 0))
        state.total_pnl = float(data.get("total_pnl_usd", 0))
        state.cycle_count = int(data.get("cycle_count", 0))

        if data.get("last_scan_time"):
            try:
                state.last_scan_time = datetime.fromisoformat(str(data["last_scan_time"]))
            except (ValueError, TypeError):
                state.last_scan_time = None

        # Restore open positions
        positions_data = data.get("open_positions", {})
        if isinstance(positions_data, dict):
            for sym, pos_data in positions_data.items():
                try:
                    entry_time = datetime.now(timezone.utc)
                    if pos_data.get("entry_time"):
                        entry_time = datetime.fromisoformat(str(pos_data["entry_time"]))

                    # Restore TP tiers if present
                    tp_tiers = None
                    tp_tiers_data = pos_data.get("tp_tiers")
                    if tp_tiers_data and isinstance(tp_tiers_data, list):
                        tp_tiers = []
                        for td in tp_tiers_data:
                            fill_time = None
                            if td.get("fill_time"):
                                try:
                                    fill_time = datetime.fromisoformat(str(td["fill_time"]))
                                except (ValueError, TypeError):
                                    pass
                            tp_tiers.append(TakeProfitTier(
                                level=int(td["level"]),
                                price=float(td["price"]) if td.get("price") is not None else None,
                                pct=float(td.get("pct", 0.33)),
                                quantity=float(td.get("quantity", 0)),
                                filled=bool(td.get("filled", False)),
                                fill_price=float(td.get("fill_price", 0)),
                                fill_time=fill_time,
                            ))

                    state.open_positions[sym] = Position(
                        trade_id=pos_data.get("trade_id", ""),
                        symbol=sym,
                        entry_price=float(pos_data.get("entry_price", 0)),
                        quantity=float(pos_data.get("quantity", 0)),
                        stop_loss=float(pos_data.get("stop_loss", 0)),
                        take_profit=float(pos_data.get("take_profit")) if pos_data.get("take_profit") else None,
                        high_water_mark=float(pos_data.get("high_water_mark", pos_data.get("entry_price", 0))),
                        entry_time=entry_time,
                        cost_usd=float(pos_data.get("cost_usd", 0)),
                        direction=pos_data.get("direction", "long"),
                        leverage=int(pos_data.get("leverage", 1) or 1),
                        margin_used=float(pos_data.get("margin_used", 0) or 0),
                        liquidation_price=float(pos_data.get("liquidation_price", 0) or 0),
                        tp_tiers=tp_tiers,
                        original_quantity=float(pos_data.get("original_quantity", 0) or 0),
                        original_stop_loss=float(pos_data.get("original_stop_loss", 0) or 0),
                        current_tier=int(pos_data.get("current_tier", 0) or 0),
                    )
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning("position_restore_failed", symbol=sym, error=str(e))

        logger.info(
            "state_restored",
            mode=state.mode,
            balance=state.current_balance,
            open_positions=len(state.open_positions),
            cycle_count=state.cycle_count,
        )
        return state
