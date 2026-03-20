# Trading Bot v2

An automated cryptocurrency trading bot for perpetual contracts on **Hyperliquid** (decentralized exchange). Features a native desktop interface, concurrent multi-strategy execution with hot-reload capability, paper trading, backtesting, and AI-driven sentiment analysis via Claude.

## Features

- **Multi-strategy execution**: Simultaneous operation of independent strategies across distinct assets
- **Hot-reload**: Strategy modifications take effect without restart (polling interval: 5 seconds)
- **Paper trading**: Full-fidelity simulator with realistic fee modeling (maker 0.015%, taker 0.045%)
- **Backtesting**: Historical replay engine with Monte Carlo simulation
- **Sentiment analysis**: Claude AI evaluates market sentiment via the Fear & Greed Index
- **Risk management**: Daily loss limits, circuit breaker, and leverage controls
- **50+ technical indicators**: RSI, MACD, Bollinger Bands, Ichimoku, Supertrend, and others
- **Desktop interface**: Native GUI via pywebview backed by a local FastAPI server
- **Persistence**: SQLite (WAL mode) for trades, candles, and strategy state

## Prerequisites

- Python 3.11+
- Hyperliquid account with a private key (Ethereum wallet)
- Anthropic API key (optional, for sentiment analysis)

## Installation

```bash
# Clone the repository
git clone <repo-url> trading-bot-v2
cd trading-bot-v2

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your credentials
```

## Configuration

### Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `TB_PRIVATE_KEY` | Yes | Ethereum private key (format `0x...`) for order signing |
| `TB_WALLET_ADDRESS` | Yes | Hyperliquid wallet address |
| `ANTHROPIC_API_KEY` | No | Claude API key for sentiment analysis |
| `TB_WEB_API_KEY` | No | API key for the web interface (auto-generated if absent) |

### Bot Configuration (`config/bot_config.json`)

```jsonc
{
  "exchange": {
    "rest_url": "https://api.hyperliquid.xyz",  // Hyperliquid REST URL
    "ws_url": "wss://api.hyperliquid.xyz/ws",   // WebSocket URL
    "is_testnet": false,                         // Set true for testnet
    "vault_address": null                        // Vault address (optional)
  },
  "risk": {
    "daily_loss_pct": 6.0,        // Halt trading after -6% daily loss
    "emergency_close_pct": 5.0,   // Emergency close at -5%
    "max_leverage": 10,           // Maximum leverage per order
    "max_position_pct": 700.0     // Maximum position size (% of account)
  },
  "strategies": {
    "dir": "./trading_bot/strategies",
    "reload_interval_sec": 5,
    "active": [
      {
        "file": "btc_inside_bar_breakout_1h.py",
        "role": "primary",
        "coins": ["BTC"],
        "paper_mode": true
      },
      {
        "file": "eth_breakout_relaxed_1h.py",
        "role": "primary",
        "coins": ["ETH"],
        "paper_mode": true
      },
      {
        "file": "sol_breakout_normal_1h.py",
        "role": "primary",
        "coins": ["SOL"],
        "paper_mode": true
      }
    ]
  },
  "mode": {
    "paper_trading": false,                      // Global paper trading mode
    "paper_initial_balance": 1000.0              // Starting balance for paper strategies
  },
  "sentiment": {
    "enabled": true,                             // Enable sentiment analysis
    "claude_model": "claude-haiku-4-5-20251001", // Claude model identifier
    "max_tokens_per_hour": 50000,                // Hourly token budget
    "cache_ttl_sec": 900,                        // Sentiment cache TTL (15 min)
    "weight": 0.3,                               // Sentiment weight in decisions
    "hard_block_threshold": -0.7                  // Block threshold (strongly negative)
  }
}
```

## Startup

```bash
# Direct startup
python main.py

# Or via the startup script
./scripts/start.sh
```

A native desktop window opens displaying the dashboard. The bot starts in the **OFF** state — use the interface to start it.

## Architecture

### Three Threads

```
Main Thread (pywebview)        Thread 1 (uvicorn)       Thread 2 (asyncio)
+------------------+          +------------------+     +------------------+
|  webview.start() |  HTTP    |  FastAPI          |     |  Engine          |
|  Native window   | <--------|  REST + WS        | <---|  Trading loop    |
|  localhost:8089   |          |  /api/*           |     |  Strategies      |
+------------------+          +------------------+     +------------------+
```

### Event Flow

```
WebSocket Hyperliquid → Engine dispatch
    ├─ on_mids()   → price updates    → strategy.on_tick()
    ├─ on_fills()  → trade executions → strategy.on_fill()
    ├─ on_book()   → order book       → strategy.on_book()
    └─ on_timer()  → every 60 seconds → strategy.on_timer()
```

### Order Routing

```
Strategy → StrategyAPI → OrderManager
    ├─ Paper mode → PaperExchange (local simulator)
    └─ Live mode  → RestClient → Hyperliquid API (EIP-712 signed)
```

## Project Structure

```
trading-bot-v2/
├── main.py                             # Entry point (3 threads)
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variable template
├── config/
│   └── bot_config.json                 # Main configuration
│
├── trading_bot/                        # Core package
│   ├── __init__.py
│   ├── engine.py                       # Trading engine (lifecycle, dispatch)
│   ├── config.py                       # Configuration dataclasses
│   ├── types.py                        # Data types (Order, Fill, Position...)
│   ├── db.py                           # SQLite layer (WAL mode)
│   ├── decimal_utils.py                # Fixed-precision arithmetic
│   ├── logging_config.py               # Logging configuration
│   │
│   ├── exchange/                       # Exchange integration
│   │   ├── rest.py                     # Hyperliquid REST client
│   │   ├── ws.py                       # WebSocket client (prices, fills, orders)
│   │   ├── signing.py                  # EIP-712 order signing
│   │   ├── order_manager.py            # Order routing (paper/live)
│   │   └── paper_exchange.py           # Paper trading simulator
│   │
│   ├── strategy/                       # Strategy framework
│   │   ├── api.py                      # StrategyAPI (primary interface)
│   │   ├── loader.py                   # Dynamic loading + hot-reload
│   │   ├── base.py                     # Strategy protocol
│   │   └── indicators.py              # 50+ technical indicators
│   │
│   ├── strategies/                     # Strategy implementations
│   │   ├── template.py                 # TemplateStrategy base class
│   │   ├── btc_inside_bar_breakout_1h.py  # BTC inside bar breakout (1h)
│   │   ├── btc_momentum_score_1h.py       # BTC momentum score composite (1h)
│   │   ├── eth_breakout_relaxed_1h.py     # ETH breakout relaxed (1h)
│   │   ├── sol_breakout_normal_1h.py      # SOL breakout normal (1h)
│   │   ├── sol_breakout_safe_1h.py        # SOL breakout safe (1h)
│   │   └── sol_breakout_aggressive_1h.py  # SOL breakout aggressive (1h)
│   │
│   ├── risk/
│   │   └── risk_manager.py             # Daily limits, circuit breaker
│   │
│   ├── data/
│   │   └── data_manager.py             # Sentiment (Claude AI + Fear & Greed)
│   │
│   ├── tools/                          # Utilities
│   │   ├── candle_fetcher.py           # Historical candle fetching (Binance)
│   │   ├── funding_fetcher.py          # Funding rate history
│   │   ├── regime_analyzer.py          # Market regime analysis
│   │   └── signal_scanner.py           # Signal detection
│   │
│   ├── backtest/                       # Backtesting engine
│   │   ├── engine.py                   # Backtest execution
│   │   ├── monte_carlo.py              # Monte Carlo simulation
│   │   └── runner.py                   # CLI runner
│   │
│   ├── report/
│   │   └── dashboard.py               # Real-time dashboard metrics
│   │
│   └── web/                            # Web interface (FastAPI)
│       ├── app.py                      # App factory (auth, CORS, routes)
│       ├── routes/                     # API endpoints
│       │   ├── bot.py                  # Bot start/stop/status
│       │   ├── account.py              # Account, positions, orders
│       │   ├── strategies.py           # Strategy management
│       │   ├── market.py               # Market data, sentiment
│       │   ├── trades.py               # Trade history
│       │   ├── backtest.py             # Backtest execution
│       │   ├── settings.py             # Configuration
│       │   └── logs.py                 # Log streaming (SSE)
│       ├── services/                   # Business logic
│       │   ├── market_data.py          # Market data aggregation
│       │   ├── metrics.py              # Metrics computation
│       │   ├── digest.py               # Daily digest
│       │   ├── alerts.py               # Alert system
│       │   └── backtest_service.py     # Backtest orchestration
│       └── static/                     # Frontend assets
│           └── js/pages/              # JS components
│
├── scripts/
│   ├── start.sh                        # Startup script
│   ├── backtest.sh                     # Backtest launcher
│   ├── fetch_candles.sh                # Candle fetching
│   ├── grid_search.sh                  # Hyperparameter search
│   └── run_backtests.py                # Python backtest runner
│
├── docs/
│   ├── gui.md                          # Interface documentation
│   └── SECURITY_AUDIT.md              # Security audit (13 vulnerabilities remediated)
│
├── data/                               # SQLite database (gitignored)
└── logs/                               # Execution logs (gitignored)
```

## API Endpoints

### Bot (`/api/bot`)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/status` | Bot state (running, uptime, WS connection) |
| POST | `/start` | Start the bot |
| POST | `/stop` | Stop the bot |

### Account (`/api/account`)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/value` | Account value |
| GET | `/positions` | Open positions |
| GET | `/open-orders` | Pending orders |

### Strategies (`/api/strategies`)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/list` | Active strategies |
| GET | `/{name}/code` | Strategy source code |
| PUT | `/{name}/disabled` | Enable/disable a strategy |

### Market (`/api/market`)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/mids` | Mid prices for all assets |
| GET | `/book/{coin}` | Order book |
| GET | `/sentiment` | Sentiment analysis |

### Trades (`/api/trades`)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/list` | Trade history |
| GET | `/export` | CSV export |

### Backtest (`/api/backtest`)

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/run` | Execute a backtest |

### Settings (`/api/settings`)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Current configuration |
| PUT | `/mode` | Toggle paper/live mode |
| PUT | `/` | Update configuration |

## Included Strategies

### BTC

| Strategy | Logic | TP/SL | Sizing | Backtest |
|----------|-------|-------|--------|----------|
| `btc_inside_bar_breakout_1h` | Inside bar + breakout direction (EMA21 trend, ATR compression < 20%, vol >= 0.8, heures 8-20 UTC) | 4.5% / 2.5% | 20%, lev 5x | Sharpe +1.90 (3Y), WR 56%, 114 trades, DD 11.1% |
| `btc_momentum_score_1h` | Score composite (close > SMA20, RSI > 50, MACD histo > 0, vol > 1.2). Long si score < 1 → >= 3, short si > 3 → <= 1 | 6% / 2.5% | 35%, lev 5x | Sharpe +0.65, positif 10/10 fenêtres |

### ETH

| Strategy | Logic | TP/SL | Sizing | Backtest |
|----------|-------|-------|--------|----------|
| `eth_breakout_relaxed_1h` | Breakout haut/bas sur 35 bougies (SMA50 trend, vol >= 4.5, anti-wick 60%, max hold 36h) | 3.5% / 1.8% | 20%, lev 5x | Sharpe +2.45 (3Y), WR 58%, 113 trades, DD 8.1% |

### SOL

| Strategy | Logic | TP/SL | Sizing | Backtest |
|----------|-------|-------|--------|----------|
| `sol_breakout_normal_1h` | Breakout haut/bas sur 14 bougies (SMA50 trend, vol >= 2.5, anti-wick 40%) | 4% / 0.9% | 30%, lev 5x | Sharpe +2.60 (3Y), WR 32%, 320 trades, DD 14.9% |
| `sol_breakout_safe_1h` | Breakout haut/bas sur 15 bougies (SMA50 trend, vol >= 2.5, cooldown 6h) | 6% / 1% | 30%, lev 5x | Sharpe +1.65, positif 10/10 fenêtres |
| `sol_breakout_aggressive_1h` | Breakout haut/bas sur 10 bougies (SMA50 trend, vol >= 2.5, cooldown 2h) | 8% / 1% | 50%, lev 7x | Sharpe +1.54, positif 10/10 fenêtres |

### Creating a Strategy

1. Copy `trading_bot/strategies/template.py`
2. Implement `on_init()`, `on_tick()`, `on_fill()`, `on_timer()`
3. Add an entry under `strategies.active` in `config/bot_config.json`
4. The strategy will be loaded automatically via hot-reload

## Backtesting

```bash
# Via script
./scripts/backtest.sh btc_sniper_1h BTC 2025-01-01 2025-03-01

# Via Python
python -m trading_bot.backtest.runner \
  --strategy btc_sniper_1h \
  --coin BTC \
  --start 2025-01-01 \
  --end 2025-03-01
```

## Database

SQLite with WAL mode. Principal tables:

| Table | Contents |
|-------|----------|
| `candles` | Historical OHLCV data |
| `trades` | Executed trade history |
| `strategy_state` | Persisted strategy state (JSON) |
| `funding_rates` | Funding rate history |
| `order_strategy_map` | Order-to-strategy mapping |
| `backtest_history` | Backtest results |

## Security

- API key authentication on all endpoints
- CORS disabled (localhost only)
- Path traversal protection on strategy loading
- XSS escaping on the frontend
- ANSI code sanitization on log streaming
- Comprehensive audit: 13 vulnerabilities identified and remediated (see `docs/SECURITY_AUDIT.md`)

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `httpx` | >= 0.27 | HTTP client (Hyperliquid REST API) |
| `websockets` | >= 13.0 | Real-time streaming |
| `msgpack` | >= 1.0 | Binary serialization (signing) |
| `eth-account` | >= 0.13 | Ethereum account management |
| `pycryptodome` | >= 3.20 | Cryptographic functions |
| `anthropic` | >= 0.40 | Claude API (sentiment) |
| `fastapi` | >= 0.115 | Web API framework |
| `uvicorn` | >= 0.32 | ASGI server |
| `pywebview` | >= 5.0 | Native desktop window |
| `python-dotenv` | >= 1.0 | `.env` file loading |
