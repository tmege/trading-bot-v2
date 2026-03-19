# Desktop GUI

## Architecture

The application uses 3 threads:

```
Main Thread (pywebview)        Thread 1 (uvicorn)       Thread 2 (asyncio)
+------------------+          +------------------+     +------------------+
|  webview.start() |  HTTP    |  FastAPI          |     |  Engine          |
|  Native window   | <--------|  REST + WS        | <---|  Trading loop    |
|  localhost:8089   |          |  /api/*           |     |  Strategies      |
+------------------+          +------------------+     +------------------+
```

- **Thread 2** (engine): asyncio event loop running the trading engine (WebSocket to Hyperliquid, strategies, order management)
- **Thread 1** (server): uvicorn serves FastAPI on `127.0.0.1:8089` (REST endpoints + WebSocket)
- **Main thread** (window): pywebview opens a native macOS window pointing to `http://127.0.0.1:8089`

Closing the window triggers a graceful shutdown of the engine and server.

## Usage

```bash
python main.py
```

A native desktop window opens with the dashboard. The terminal still shows the ANSI dashboard for debugging.

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/bot/status` | Bot running state, uptime, WS connection |
| POST | `/api/bot/stop` | Trigger graceful shutdown |
| GET | `/api/config` | Bot configuration (secrets excluded) |
| GET | `/api/strategies` | Strategy list with status, trades, win rate, position |
| GET | `/api/trades?limit=50` | Recent trades from SQLite |
| GET | `/api/positions` | Open positions with uPnL |
| GET | `/api/market/fear-greed` | Fear & Greed index (0-100 + label) |
| GET | `/api/market/candles?coin=BTC&interval=1h` | Historical candles |
| GET | `/api/logs?n=50` | Last N log lines |
| GET | `/api/account` | Account value, daily PnL, open orders count |
| GET | `/api/mids` | Current mid prices (REST fallback) |
| WS | `/ws/live` | Real-time mid price stream |

## Dashboard Sections

| Section | Data Source | Refresh |
|---------|-----------|---------|
| Status badge | `/api/bot/status` | 2s poll |
| Account / PnL | `/api/account` | 2s poll |
| Fear & Greed | `/api/market/fear-greed` | 10s poll |
| Mid Prices | `/ws/live` | Real-time (~300ms) |
| Strategies | `/api/strategies` | 2s poll |
| Positions | `/api/positions` | 2s poll |
| Logs | `/api/logs` | 3s poll |

## Dependencies

- `fastapi>=0.115` — REST + WebSocket server
- `uvicorn>=0.32` — ASGI server
- `pywebview>=5.0` — Native desktop window
