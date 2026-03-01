-- Fix numeric overflow for meme coins with huge volumes and tiny prices
-- DECIMAL(20, 8) overflows when volume > 10^12 or price has > 8 decimal places

-- Candle cache: volume can be in the trillions for meme coins
ALTER TABLE candle_cache ALTER COLUMN volume TYPE DECIMAL(30, 8);

-- Candle cache: prices can be extremely small (e.g., MOG at 0.0000001)
ALTER TABLE candle_cache ALTER COLUMN open TYPE DECIMAL(30, 12);
ALTER TABLE candle_cache ALTER COLUMN high TYPE DECIMAL(30, 12);
ALTER TABLE candle_cache ALTER COLUMN low TYPE DECIMAL(30, 12);
ALTER TABLE candle_cache ALTER COLUMN close TYPE DECIMAL(30, 12);

-- Trades: prices and quantities for meme coins
ALTER TABLE trades ALTER COLUMN entry_price TYPE DECIMAL(30, 12);
ALTER TABLE trades ALTER COLUMN entry_quantity TYPE DECIMAL(30, 12);
ALTER TABLE trades ALTER COLUMN exit_price TYPE DECIMAL(30, 12);
ALTER TABLE trades ALTER COLUMN exit_quantity TYPE DECIMAL(30, 12);
ALTER TABLE trades ALTER COLUMN stop_loss TYPE DECIMAL(30, 12);
ALTER TABLE trades ALTER COLUMN take_profit TYPE DECIMAL(30, 12);

-- Signals: current_price for meme coins
ALTER TABLE signals ALTER COLUMN current_price TYPE DECIMAL(30, 12);
