# Trading Bot v2 — Claude Code Instructions

## Project Overview

Automated cryptocurrency trading bot for **Hyperliquid** (perpetual DEX). Event-driven architecture with a native desktop GUI, hot-reloadable multi-strategy execution, paper trading, backtesting, and sentiment analysis.

**Stack**: Python 3.11+ | FastAPI | pywebview | SQLite | WebSocket | Hyperliquid API | Claude AI

Approximately 9,400 lines of code across 60 Python files.

## Architecture

### Three Threads
- **Main thread**: pywebview window (native GUI)
- **Thread 1**: FastAPI/uvicorn server on `127.0.0.1:8089`
- **Thread 2**: asyncio event loop with the trading engine

### Event Flow
```
WebSocket Hyperliquid → Engine._dispatch()
    ├─ _on_mids()    → prices     → strategy.on_tick()
    ├─ _on_fills()   → executions → strategy.on_fill()
    ├─ _on_book()    → order book → strategy.on_book()
    └─ timer (60s)                → strategy.on_timer()
```

### Order Routing
```
Strategy → StrategyAPI → OrderManager
    ├─ paper_mode=true  → PaperExchange (simulator)
    └─ paper_mode=false → RestClient → Hyperliquid (EIP-712)
```

## Key Files

### Core
- `main.py`: entry point, creates the 3 threads
- `trading_bot/engine.py`: main engine (lifecycle, dispatch, state persistence)
- `trading_bot/config.py`: configuration dataclasses, loads `bot_config.json`
- `trading_bot/types.py`: data types (Order, Fill, Position, Candle, Decimal...)
- `trading_bot/db.py`: SQLite layer with WAL mode

### Exchange
- `trading_bot/exchange/rest.py`: Hyperliquid REST client (rate limit 1200 req/min)
- `trading_bot/exchange/ws.py`: WebSocket client (real-time subscriptions)
- `trading_bot/exchange/signing.py`: EIP-712 order signing
- `trading_bot/exchange/order_manager.py`: paper/live routing + OID-to-strategy mapping
- `trading_bot/exchange/paper_exchange.py`: full simulator with fees

### Strategies
- `trading_bot/strategy/api.py`: **StrategyAPI** — primary interface used by strategies
- `trading_bot/strategy/loader.py`: dynamic loading + hot-reload (5s)
- `trading_bot/strategy/base.py`: strategy protocol (interface)
- `trading_bot/strategy/indicators.py`: 50+ technical indicators
- `trading_bot/strategies/template.py`: `TemplateStrategy` base class
- `trading_bot/strategies/btc_inside_bar_breakout_1h.py`: BTC inside bar breakout (EMA21+ATR, TP 4.5%/SL 2.5%)
- `trading_bot/strategies/btc_momentum_score_1h.py`: BTC momentum score composite (SMA20+RSI+MACD+vol, TP 6%/SL 2.5%)
- `trading_bot/strategies/eth_breakout_relaxed_1h.py`: ETH breakout lb=35 (anti-wick 60%, TP 3.5%/SL 1.8%)
- `trading_bot/strategies/sol_breakout_normal_1h.py`: SOL breakout lb=14 (anti-wick 40%, TP 4%/SL 0.9%)
- `trading_bot/strategies/sol_breakout_safe_1h.py`: SOL breakout lb=15 (TP 6%/SL 1%)
- `trading_bot/strategies/sol_breakout_aggressive_1h.py`: SOL breakout lb=10 (lev 7x, TP 8%/SL 1%)

### Web
- `trading_bot/web/app.py`: FastAPI factory (API key auth, CORS, routes)
- `trading_bot/web/routes/*.py`: REST endpoints
- `trading_bot/web/services/*.py`: business logic

### Other
- `trading_bot/risk/risk_manager.py`: loss limits, circuit breaker, leverage controls
- `trading_bot/data/data_manager.py`: sentiment (Claude AI + Fear & Greed)
- `trading_bot/backtest/engine.py`: backtesting engine with Monte Carlo

## Code Conventions

### Required Patterns
- **Async/await** in the engine and exchange (dedicated asyncio loop)
- **Dataclasses** for data types (`@dataclass` in `types.py`)
- **Protocol** for the strategy interface (`base.py`)
- **Custom Decimal** (`types.Decimal`) for monetary precision — do NOT use `float` for prices/order sizes
- **Parameterized SQL queries** only — never use string concatenation

### Naming
- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Engine callbacks: `_on_mids()`, `_on_fills()`, `_on_book()`
- API routes: `/api/` prefix

### Configuration
- **Config**: `config/bot_config.json` (JSON)
- **Secrets**: `.env` only (never in code, never committed)
- **Required env vars**: `TB_PRIVATE_KEY`, `TB_WALLET_ADDRESS`
- **Optional env vars**: `ANTHROPIC_API_KEY`, `TB_WEB_API_KEY`

### Database
- SQLite WAL mode (`data/trading_bot.db`)
- Tables: `candles`, `trades`, `strategy_state`, `funding_rates`, `order_strategy_map`, `backtest_history`

## Development Rules

### Strategies
- Inherit from `TemplateStrategy` (`trading_bot/strategies/template.py`)
- Use **only** `StrategyAPI` to interact with the engine
- Do NOT access `Engine`, `RestClient`, or `DB` directly from a strategy
- Use `api.save_state()` / `api.load_state()` for state persistence
- Manage cooldowns and internal state within the strategy

### Risk Management
- Every order passes through `risk_manager.check_order()` — do not bypass
- Circuit breaker: >7% movement in 15 minutes on a given asset
- Limits: daily_loss_pct=6%, emergency_close_pct=5%, max_leverage=10x

### Web Security
- API key required on all endpoints (`X-API-Key` header)
- CORS disabled (allow_origins=[])
- Escape frontend outputs (`TB.utils.esc()`)
- Validate strategy filenames (regex + `Path.is_relative_to()`)
- Sanitize ANSI codes before SSE streaming

### Paper Trading
- `PaperExchange` simulates real fees (maker 0.015%, taker 0.045%)
- All strategies must be tested in paper mode before live deployment
- Paper state is persisted in the database and restored on restart

## Useful Commands

```bash
# Start the bot
python main.py

# Backtest
python -m trading_bot.backtest.runner --strategy btc_sniper_1h --coin BTC --start 2025-01-01 --end 2025-03-01

# Fetch historical candles
./scripts/fetch_candles.sh
```

## Important Notes

- The bot starts in the OFF state; it must be started via the GUI
- Strategies with `paper_mode: true` use `PaperExchange`, not Hyperliquid
- Sentiment carries a weight of 0.3 and can block a trade if score < -0.7
- Hyperliquid rate limit: 1200 req/min with exponential backoff
- The complete security audit is available at `docs/SECURITY_AUDIT.md` (13 vulnerabilities remediated)
