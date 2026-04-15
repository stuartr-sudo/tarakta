# Missed Trade Analyzer — Spec

**Date:** 2026-03-23
**Status:** Draft

## Problem

The Tarakta trading bot is too conservative with entries. It detects good signals (sweep + displacement scoring 60+) but then filters them out through downstream gates:

1. **Entry Refiner** skips trades when no 5m pullback into OTE zone (50-79% Fib) occurs within 30 minutes
2. **Pullback Analyzer** requires 20-78% retracement — shallow pullbacks (< 20%) in strong trends get classified as "waiting" and never trigger
3. **Consensus Monitor** penalizes trades that disagree with portfolio bias or BTC trend
4. **Sentiment Filter** blocks on strong contra-sentiment

The user suspects many profitable setups are being missed, particularly when:
- The pullback has already happened before the scanner detects the signal
- The move is strong enough that it only retraces 10-18% (below the 20% minimum)
- V-shaped moves that never give a deep pullback but would have been profitable

## Solution

A Claude Agent SDK-powered CLI tool that:

1. Queries Supabase `signals` table for signals where `acted_on = false` and `score >= threshold`
2. For each missed signal, fetches historical 1H candles from the signal time
3. Simulates the trade outcome (would SL or TP have been hit?)
4. Identifies patterns in which missed trades would have been profitable
5. Recommends specific parameter changes with confidence ratings

## Non-Goals

- No changes to the live trading bot in this phase
- No new database tables
- Not running continuously — this is an on-demand analysis tool
- Not making automatic parameter changes (advisory only)

## Architecture

```
src/advisor/
    __init__.py
    __main__.py              # Module entry point
    missed_trades.py         # Agent runner (main entry)
    missed_signals.py        # Supabase query for unacted signals
    outcome_simulator.py     # Walk-forward trade simulation
    tools.py                 # MCP tools wrapping the above for the agent
```

The agent uses 3 custom MCP tools:
- `get_missed_signals` — queries Supabase
- `simulate_outcome` — walks forward through candles
- `get_trade_stats` — baseline performance metrics

## Dependencies

- `claude-agent-sdk` (new dependency, local analysis only)
- `ANTHROPIC_API_KEY` environment variable (separate from bot's `OPENAI_API_KEY`)
- Existing: `src/data/db.py`, `src/data/candles.py`, `src/config.py`

## Usage

```bash
# Basic analysis (last 7 days, signals scoring 55+)
python -m src.advisor.missed_trades

# Custom range
python -m src.advisor.missed_trades --days 14 --min-score 60

# Different instance
python -m src.advisor.missed_trades --instance-id expanded
```

## Future Phases (Not This Plan)

- **Phase 2: Entry Strategy Advisor** — Uses missed trade data to suggest specific parameter tweaks
- **Phase 3: Live Context Layer** — Real-time macro context agent that feeds into Agent 2
