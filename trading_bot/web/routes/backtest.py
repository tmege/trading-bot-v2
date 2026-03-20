import asyncio
import logging
import re
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])

_engine = None


def init(engine):
    global _engine
    _engine = engine


@router.get("/coins")
async def get_coins():
    from trading_bot.web.services.backtest_service import get_available_coins
    if not _engine or not _engine.db:
        return []
    return get_available_coins(_engine.db)


@router.post("/run")
async def run_backtest(request: Request):
    from trading_bot.web.services.backtest_service import start_run

    if not _engine or not _engine.config:
        return {"error": "not initialized"}

    body = await request.json()
    strategy_file = body.get("strategy", "")
    coins = body.get("coins", [])
    initial_balance = 100.0  # Fixed at $100 — results shown as %
    interval_ms = body.get("interval_ms", 3_600_000)
    start_date = body.get("start_date", "")  # "YYYY-MM-DD" or empty
    end_date = body.get("end_date", "")      # "YYYY-MM-DD" or empty

    if not strategy_file or not coins:
        return {"error": "strategy and coins required"}

    # V-02/V-05: Strict filename validation — only alphanumeric, underscores, hyphens
    if not re.match(r'^[a-zA-Z0-9_-]+\.py$', strategy_file):
        return {"error": "Invalid strategy filename"}

    # V-02: Validate coins format
    for c in coins:
        if not isinstance(c, str) or not re.match(r'^[A-Z]{2,10}$', c):
            return {"error": f"Invalid coin format: {c}"}

    # Validate date format
    start_ms = 0
    end_ms = 0
    if start_date:
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', start_date):
            return {"error": "Invalid start_date format (YYYY-MM-DD)"}
        from datetime import datetime, timezone
        start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    if end_date:
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', end_date):
            return {"error": "Invalid end_date format (YYYY-MM-DD)"}
        from datetime import datetime, timezone
        end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    strategies_dir = _engine.config.strategies.dir
    db_path = _engine.config.database.path

    run_id = start_run(
        strategy_file=strategy_file,
        coins=coins,
        db_path=db_path,
        strategies_dir=strategies_dir,
        initial_balance=initial_balance,
        max_leverage=50,
        interval_ms=interval_ms,
        start_ms=start_ms,
        end_ms=end_ms,
    )

    return {"run_id": run_id}


@router.get("/progress/{run_id}")
async def get_progress(run_id: str):
    from trading_bot.web.services.backtest_service import get_run
    import json
    import trading_bot.web.app as _app_mod

    # V-13: SSE connection limit
    if _app_mod._sse_connections >= _app_mod._MAX_SSE:
        return StreamingResponse(
            _error_stream("Too many SSE connections"),
            media_type="text/event-stream",
        )

    # V-02: Validate run_id format (hex, max 12 chars)
    if not re.match(r'^[a-f0-9]{1,12}$', run_id):
        return StreamingResponse(
            _error_stream("Invalid run ID"),
            media_type="text/event-stream",
        )

    run = get_run(run_id)
    if not run:
        return StreamingResponse(
            _error_stream("Run not found"),
            media_type="text/event-stream",
        )

    async def event_generator():
        _app_mod._sse_connections += 1
        try:
            while True:
                try:
                    msg = run.queue.get(timeout=0.5)
                    yield f"data: {json.dumps(msg)}\n\n"
                    if msg.get("type") in ("complete", "error"):
                        break
                except Exception:
                    if run.status in ("complete", "error"):
                        break
                    yield ": keepalive\n\n"
                    await asyncio.sleep(0.3)
        finally:
            _app_mod._sse_connections -= 1

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history")
async def get_history(limit: int = Query(default=500, le=1000)):
    from trading_bot.web.services.backtest_service import get_history
    if not _engine or not _engine.db:
        return []
    return get_history(_engine.db, limit)


@router.get("/latest/{strategy}")
async def get_latest_result(strategy: str):
    """Return the latest backtest result for a strategy (full result_json)."""
    from trading_bot.web.services.backtest_service import get_latest_result
    if not _engine or not _engine.db:
        return None
    # V-02: Validate strategy filename
    if not re.match(r'^[a-zA-Z0-9_-]+\.py$', strategy):
        return {"error": "Invalid strategy filename"}
    return get_latest_result(_engine.db, strategy)


@router.delete("/history")
async def delete_history():
    from trading_bot.web.services.backtest_service import clear_history
    if not _engine or not _engine.db:
        return {"deleted": 0}
    count = clear_history(_engine.db)
    return {"deleted": count}


async def _error_stream(msg: str):
    import json
    yield f"data: {json.dumps({'type': 'error', 'message': msg})}\n\n"
