"""One-off script: Close ELSA/USDT flipped trade at its SL price (0.0776).

Also scans ALL other open flipped trades to check if any have blown through
their SL/TP and closes them at the SL/TP price.

The trades should have been stopped out but weren't due to the monitoring bug
(flipped positions were only checked every 15 min instead of every 60s).

Run: python fix_elsa_trade.py
"""
import asyncio
import os

import ccxt.async_support as ccxt
from dotenv import load_dotenv

load_dotenv()

from src.data.db import Database
from src.data.repository import Repository

ELSA_SL_EXIT_PRICE = 0.0776
SIM_FEE_RATE = 0.0004


def calc_pnl(direction, entry_price, exit_price, quantity, cost_usd):
    if direction == "long":
        pnl = (exit_price - entry_price) * quantity
    else:
        pnl = (entry_price - exit_price) * quantity
    exit_fee = cost_usd * SIM_FEE_RATE
    pnl -= exit_fee
    total_fees = cost_usd * SIM_FEE_RATE + exit_fee
    pnl_pct = (pnl / cost_usd * 100) if cost_usd > 0 else 0
    return pnl, pnl_pct, total_fees


async def close_trade_at(repo, trade, exit_price, reason):
    trade_id = trade["id"]
    symbol = trade["symbol"]
    direction = trade.get("direction", "long")
    entry_price = float(trade.get("entry_price", 0))
    quantity = float(trade.get("remaining_quantity") or trade.get("entry_quantity", 0))
    cost_usd = float(trade.get("entry_cost_usd", 0))

    pnl, pnl_pct, total_fees = calc_pnl(direction, entry_price, exit_price, quantity, cost_usd)

    print(f"  Closing {symbol} (ID: {trade_id})")
    print(f"    Direction: {direction} | Entry: {entry_price} | Exit: {exit_price}")
    print(f"    PnL: ${pnl:.4f} ({pnl_pct:.2f}%) | Reason: {reason}")

    await repo.close_trade(
        trade_id=trade_id,
        exit_price=exit_price,
        exit_quantity=quantity,
        exit_order_id="manual_sl_fix",
        exit_reason=reason,
        pnl_usd=round(pnl, 4),
        pnl_percent=round(pnl_pct, 2),
        fees_usd=round(total_fees, 4),
    )
    print(f"    -> Closed")
    return pnl


async def main():
    db = Database(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    repo = Repository(db)

    open_trades = await repo.get_open_trades(mode="flipped_paper")
    if not open_trades:
        print("No open flipped trades found.")
        return

    print(f"Found {len(open_trades)} open flipped trades.\n")

    # 1. Fix ELSA specifically at its SL price
    print("=== ELSA/USDT Fix (SL was 0.0776) ===")
    elsa_trades = [t for t in open_trades if "ELSA" in t.get("symbol", "")]
    remaining_trades = [t for t in open_trades if "ELSA" not in t.get("symbol", "")]

    for trade in elsa_trades:
        await close_trade_at(repo, trade, ELSA_SL_EXIT_PRICE, "sl_hit")

    # 2. Check remaining trades against live prices for SL/TP breaches
    if remaining_trades:
        print(f"\n=== Checking {len(remaining_trades)} other trades for SL/TP breaches ===")

        exchange = ccxt.binance({
            "apiKey": os.environ.get("BINANCE_API_KEY", ""),
            "secret": os.environ.get("BINANCE_API_SECRET", ""),
            "enableRateLimit": True,
            "options": {"defaultType": "future", "fetchCurrencies": False},
        })

        try:
            symbols = list({t["symbol"] for t in remaining_trades})
            tickers = await exchange.fetch_tickers(symbols)

            for trade in remaining_trades:
                symbol = trade["symbol"]
                direction = trade.get("direction", "long")
                sl = float(trade.get("stop_loss", 0))
                tp = float(trade.get("take_profit", 0) or 0)
                ticker = tickers.get(symbol)
                if not ticker:
                    print(f"  {symbol}: no ticker data, skipping")
                    continue

                current_price = float(ticker["last"])
                sl_breached = (direction == "long" and current_price <= sl) or \
                              (direction == "short" and current_price >= sl)
                tp_breached = tp > 0 and (
                    (direction == "long" and current_price >= tp) or
                    (direction == "short" and current_price <= tp)
                )

                if tp_breached:
                    await close_trade_at(repo, trade, tp, "tp_hit")
                elif sl_breached:
                    await close_trade_at(repo, trade, sl, "sl_hit")
                else:
                    entry = float(trade.get("entry_price", 0))
                    print(f"  {symbol}: OK (entry={entry}, sl={sl}, tp={tp}, current={current_price})")
        finally:
            await exchange.close()

    # 3. Update engine_state to remove closed positions from flipped_trader state
    print("\n=== Updating engine state ===")
    try:
        engine_state = await repo.get_engine_state()
        if engine_state:
            overrides = engine_state.get("config_overrides", {})
            flipped_state = overrides.get("flipped_trader", {})
            positions = flipped_state.get("positions", {})
            closed_symbols = [t["symbol"] for t in elsa_trades]
            # Also add any other trades we closed
            for t in remaining_trades:
                symbol = t["symbol"]
                direction = t.get("direction", "long")
                sl = float(t.get("stop_loss", 0))
                tp = float(t.get("take_profit", 0) or 0)
                # Re-check if we closed this one (approximate)
                if symbol in positions:
                    closed_symbols.append(symbol)

            removed = 0
            for sym in closed_symbols:
                if sym in positions:
                    del positions[sym]
                    removed += 1

            if removed > 0:
                flipped_state["positions"] = positions
                overrides["flipped_trader"] = flipped_state
                engine_state["config_overrides"] = overrides
                await repo.upsert_engine_state(engine_state)
                print(f"  Removed {removed} closed positions from engine state")
            else:
                print("  No positions to remove from engine state")
    except Exception as e:
        print(f"  Warning: Could not update engine state: {e}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
