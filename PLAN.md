# Margin & Futures Trading Support — Implementation Plan

## Overview

Add Binance **Margin** (spot-based shorting) and **USDM Futures** (perpetual contracts) support alongside existing spot trading. The account type is selectable via `ACCOUNT_TYPE` env var: `"spot"` (default), `"margin"`, or `"futures"`.

## Architecture Approach

**One exchange client per account type** — extend the factory pattern already in `create_exchange()`. The engine, risk manager, and portfolio tracker become account-type-aware.

---

## Step 1: Config — New Fields

**File: `src/config.py`**

Add:
```
account_type: Literal["spot", "margin", "futures"] = "spot"
leverage: int = 1                    # 1x–20x (futures only)
margin_mode: Literal["isolated", "cross"] = "isolated"  # futures only
```

The `account_type` controls:
- Which ccxt `defaultType` is used (`"spot"`, `"margin"`, `"future"`)
- Whether bearish/short signals are allowed
- How position sizing accounts for leverage
- How balance/equity is calculated

---

## Step 2: Exchange Client — BinanceFuturesClient

**File: `src/exchange/client.py`**

### New Class: `BinanceFuturesClient`
- Same interface as `BinanceClient` (fetch_candles, fetch_ticker, place_market_order, get_balance, get_tradeable_pairs, close)
- Uses `ccxt.binance` with `"defaultType": "future"`, `"defaultMarginMode": "isolated"`
- On init: set leverage via `exchange.fapiPrivatePostLeverage(symbol, leverage)`
- `get_tradeable_pairs()`: filter on `market.get("linear")` (USDT-margined) instead of `market.get("spot")`
- `place_market_order()`: same signature, ccxt handles futures orders transparently
- New method: `set_leverage(symbol, leverage)` — called before first trade on each symbol
- New method: `get_position_risk(symbol)` — returns margin ratio, liquidation price, unrealized PnL from exchange
- Properties: `taker_fee_rate = 0.0004` (0.04%), `min_order_usd = 5.0`, `exchange_name = "binance_futures"`

### New Class: `BinanceMarginClient`
- Same interface as `BinanceClient`
- Uses `ccxt.binance` with `"defaultType": "margin"`
- `place_market_order()` for shorts: uses `exchange.create_order(symbol, "market", "sell", qty, params={"type": "margin", "sideEffectType": "MARGIN_BUY"})` — this auto-borrows
- For closing shorts: `sideEffectType: "AUTO_REPAY"` — auto-repays the loan
- New method: `get_margin_info()` — returns available margin, used margin, margin level
- Properties: `taker_fee_rate = 0.001` (same as spot), `min_order_usd = 10.0`, `exchange_name = "binance_margin"`

### Update `create_exchange()` factory
Add `account_type` parameter:
```python
def create_exchange(name, api_key, api_secret, account_type="spot"):
    if name == "binance":
        if account_type == "futures":
            return BinanceFuturesClient(api_key, api_secret)
        elif account_type == "margin":
            return BinanceMarginClient(api_key, api_secret)
        return BinanceClient(api_key, api_secret)
    ...
```

---

## Step 3: Risk Manager — Allow Shorts, Leverage-Aware Sizing

**File: `src/risk/manager.py`**

### Remove spot-only short rejection
The `validate_trade()` method currently rejects all bearish signals with "Spot accounts cannot short." Change to:
```python
if signal.direction == "bearish" and self._account_type == "spot":
    return TradeValidation(allowed=False, reason="Spot accounts cannot short.")
```

### Add `_account_type` and `_leverage` to `__init__`
```python
self._account_type = getattr(exchange, "account_type", "spot") if exchange else "spot"
self._leverage = config.leverage if hasattr(config, "leverage") else 1
```

### Leverage-aware position sizing in `calculate_position_size()`
For futures with leverage:
- The cost to the trader is `notional_value / leverage`
- Risk per trade stays the same (based on SL distance)
- But the effective position size is multiplied by leverage
- Cap the notional value so that liquidation price is always beyond the stop-loss

Add a liquidation price check:
```python
if self._leverage > 1:
    # Ensure SL triggers before liquidation
    liq_distance = entry_price / self._leverage  # simplified
    if sl_distance > liq_distance * 0.8:  # 80% safety buffer
        # Reduce position or reject
```

### Add max leverage exposure check
Total notional exposure across all positions must not exceed `max_exposure_pct * equity * leverage`.

---

## Step 4: Paper Exchange — Futures Simulation

**File: `src/exchange/paper.py`**

### Add leverage-aware paper trading
- `__init__`: accept `account_type` and `leverage` parameters
- For futures paper mode:
  - Long entry: deduct `cost / leverage` from balance (margin used)
  - Short entry: deduct `cost / leverage` from balance (margin used)
  - Track `margin_used` per position
  - PnL calculation includes leverage multiplier on the notional value
- Add `get_position_risk()` stub that returns simulated margin ratio

---

## Step 5: Portfolio Tracker — Margin-Aware Equity

**File: `src/risk/portfolio.py`**

### Update `record_entry()`
For futures: deduct `cost_usd / leverage` (margin) instead of full `cost_usd`

### Update `record_exit()`
For futures: return margin + PnL (which is calculated on full notional)

### Update `get_equity()`
For futures: equity = cash + sum(margin_per_position) + unrealized_pnl_on_full_notional

### Add `Position.leverage` field
Track leverage per position for mixed-leverage scenarios.

---

## Step 6: Engine Core — Wire It Up

**File: `src/engine/core.py`**

### Primary tick changes
- The signal processing loop already handles direction-based entry (`is_long = signal.direction == "bullish"`)
- Remove/modify the spot-only guard — this is in `risk_manager.validate_trade()` (Step 3)
- Pass `account_type` context when logging rejections

### Position monitor changes
- For futures: add liquidation price monitoring alongside SL/TP
- If price approaches liquidation (within 2%), force-close the position before exchange liquidates it
- This protects against slippage past SL on volatile moves

### Reconciliation changes
- For futures: margin-based balance restoration on reconciliation
- Already handles direction="short" at line 587 — just needs leverage-aware balance math

---

## Step 7: Main Entry Point

**File: `src/main.py`**

Update exchange creation:
```python
live_exchange = create_exchange(
    config.exchange_name, api_key, api_secret,
    account_type=config.account_type,
)
```

For futures paper mode, pass leverage:
```python
if config.trading_mode == "paper":
    exchange = PaperExchange(
        initial_balance=config.initial_balance,
        live_exchange=live_exchange,
        account_type=config.account_type,
        leverage=config.leverage,
    )
```

---

## Step 8: Dashboard API — Futures/Margin Tickers

**File: `src/dashboard/api.py`**

### Update `_DashboardExchange`
- Accept `account_type` parameter
- Use correct `defaultType` ("future" for futures, "margin" for margin)
- For futures: ticker symbols are the same USDT-margined perps (`BTC/USDT:USDT`)
- The unrealized PnL endpoint already handles short direction — just needs correct ticker source

### Update `create_router()` signature
Add `account_type` parameter, pass to `_DashboardExchange`.

---

## Step 9: Dashboard App

**File: `src/dashboard/app.py`**

Pass `config.account_type` to `create_api_router()`.

---

## Step 10: Exchange Models

**File: `src/exchange/models.py`**

Add to `Position`:
```python
leverage: int = 1
margin_used: float = 0.0      # actual margin locked
liquidation_price: float = 0.0
```

---

## Implementation Order

1. Config (trivial — 3 new fields)
2. Exchange models (add leverage/margin fields to Position)
3. BinanceFuturesClient + BinanceMarginClient (biggest piece — new classes)
4. Update `create_exchange()` factory
5. Risk manager — remove spot-only guard, add leverage sizing
6. Paper exchange — leverage simulation
7. Portfolio tracker — margin-aware equity
8. Engine core — liquidation monitoring
9. Main entry point — pass account_type
10. Dashboard API/app — correct defaultType
11. Set Fly secrets and deploy
12. Test with paper mode first, then live

## Fly Secrets for Futures
```
ACCOUNT_TYPE=futures
LEVERAGE=3
MARGIN_MODE=isolated
```

## Risk Guardrails
- Max leverage capped at 10x in config validation
- Liquidation buffer: SL must be at least 20% closer to entry than liquidation price
- Circuit breaker still applies on equity (including unrealized P&L)
- Daily drawdown limit applies to margin equity, not just cash
