import asyncio
import json
import logging
import os
import re
import secrets
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from trading_bot.web.routes import bot, account, strategies, market, trades, backtest, settings, logs

log = logging.getLogger(__name__)

# --- V-01: API Key Authentication ---
_API_KEY: str = ""
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key_header: str | None = Depends(_API_KEY_HEADER),
    api_key: str | None = Query(default=None, alias="api_key"),
):
    """Check X-API-Key header first, fall back to ?api_key= query param (for SSE/EventSource)."""
    key = api_key_header or api_key or ""
    if not _API_KEY or not key or not secrets.compare_digest(key, _API_KEY):
        log.warning("Authentication failed — invalid or missing API key")
        raise HTTPException(status_code=401, detail="Unauthorized")


# --- V-13: Connection limits (H-03: thread-safe) ---
_conn_lock = threading.Lock()
_ws_connections = 0
_MAX_WS = 10
_sse_connections = 0
_MAX_SSE = 5

ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07')


# --- M-12: Security headers middleware ---
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # Skip CSP if already set by the route handler (e.g. index with nonce)
        if "Content-Security-Policy" not in response.headers:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' https://unpkg.com https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data:; "
                "connect-src 'self' ws://127.0.0.1:*; "
                "frame-ancestors 'none'"
            )
        return response


def create_app(engine, stop_event=None) -> FastAPI:
    global _API_KEY

    # Generate API key: use env var if set, otherwise generate a random one
    _API_KEY = os.environ.get("TB_WEB_API_KEY", "") or secrets.token_urlsafe(32)
    log.info("Web API key configured (length=%d)", len(_API_KEY))

    app = FastAPI(title="Trading Bot v2", docs_url=None, redoc_url=None)

    # --- M-12: Security headers ---
    app.add_middleware(SecurityHeadersMiddleware)

    # --- V-06: CORS middleware — block all cross-origin requests ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    static_dir = Path(__file__).parent / "static"

    bot.init(engine, stop_event)
    account.init(engine)
    strategies.init(engine)
    market.init(engine)
    trades.init(engine)
    backtest.init(engine)
    settings.init(engine)
    logs.init(engine)

    # All routers require API key authentication
    auth_dep = [Depends(verify_api_key)]
    app.include_router(bot.router, dependencies=auth_dep)
    app.include_router(account.router, dependencies=auth_dep)
    app.include_router(strategies.router, dependencies=auth_dep)
    app.include_router(market.router, dependencies=auth_dep)
    app.include_router(trades.router, dependencies=auth_dep)
    app.include_router(backtest.router, dependencies=auth_dep)
    app.include_router(settings.router, dependencies=auth_dep)
    app.include_router(logs.router, dependencies=auth_dep)

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = static_dir / "index.html"
        html = html_path.read_text(encoding="utf-8")
        # M-07: Inject API key safely with JSON encoding + CSP nonce to prevent XSS
        nonce = secrets.token_urlsafe(16)
        safe_key = json.dumps(_API_KEY).replace("<", "\\u003c").replace(">", "\\u003e")
        inject = f'<script nonce="{nonce}">window.__TB_API_KEY__={safe_key};</script>'
        html = html.replace("</head>", inject + "\n</head>")
        response = HTMLResponse(content=html)
        response.headers["Content-Security-Policy"] = (
            f"default-src 'self'; "
            f"script-src 'self' 'nonce-{nonce}' https://unpkg.com https://cdn.jsdelivr.net; "
            f"style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            f"img-src 'self' data:; "
            f"connect-src 'self' ws://127.0.0.1:*; "
            f"frame-ancestors 'none'"
        )
        return response

    # Enhanced WebSocket with fills via DB poll
    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket):
        # V-01: Verify API key from query param for WebSocket
        ws_key = websocket.query_params.get("key", "")
        if not _API_KEY or not ws_key or not secrets.compare_digest(ws_key, _API_KEY):
            await websocket.close(code=4001, reason="Unauthorized")
            return

        # H-03: Thread-safe connection limit
        global _ws_connections
        with _conn_lock:
            if _ws_connections >= _MAX_WS:
                await websocket.close(code=4002, reason="Too many connections")
                return
            _ws_connections += 1

        await websocket.accept()
        prev_mids: dict[str, float] = {}
        last_trade_id = _get_last_trade_id(engine)
        last_status_time = 0.0

        try:
            while True:
                now = time.time()

                # Mids (~300ms if changed) — filter out Hyperliquid index symbols (@1, @10, etc.)
                current = {k: v for k, v in engine._mid_prices.items() if not k.startswith("@")}
                if current != prev_mids:
                    await websocket.send_json({"type": "mids", "data": current})
                    prev_mids = dict(current)

                # Fills (poll DB every 2s)
                if engine.db and now - last_status_time >= 2:
                    new_trades = _get_new_trades(engine, last_trade_id)
                    for t in new_trades:
                        await websocket.send_json({"type": "fill", "data": t})
                        last_trade_id = max(last_trade_id, t["id"])

                # Status heartbeat every 2s
                if now - last_status_time >= 2:
                    ws_ok = False
                    if engine.ws:
                        ws_ok = getattr(engine.ws, "connected", engine._running)
                    await websocket.send_json({
                        "type": "status",
                        "data": {
                            "running": engine._running,
                            "connected": ws_ok,
                        },
                    })
                    last_status_time = now

                await asyncio.sleep(0.3)
        except WebSocketDisconnect:
            pass
        except Exception:
            log.debug("WebSocket error in ws_live")
        finally:
            with _conn_lock:
                _ws_connections -= 1

    return app


def _get_last_trade_id(engine) -> int:
    if not engine.db:
        return 0
    try:
        row = engine.db.fetchone("SELECT MAX(id) as max_id FROM trades")
        return int(row["max_id"]) if row and row["max_id"] else 0
    except Exception:
        return 0


def _get_new_trades(engine, last_id: int) -> list[dict]:
    if not engine.db:
        return []
    try:
        rows = engine.db.fetchall(
            "SELECT id, coin, side, price, size, fee, closed_pnl, strategy, time_ms "
            "FROM trades WHERE id > ? ORDER BY id LIMIT 20",
            (last_id,),
        )
        return [dict(r) for r in rows]
    except Exception:
        return []
