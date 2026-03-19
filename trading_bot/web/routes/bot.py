import asyncio
import logging
import time

from fastapi import APIRouter

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/bot", tags=["bot"])

_engine = None
_stop_event = None
_start_time = time.time()


def init(engine, stop_event):
    global _engine, _stop_event
    _engine = engine
    _stop_event = stop_event


@router.get("/status")
async def bot_status():
    ws_ok = False
    if _engine and _engine.ws:
        ws_ok = getattr(_engine.ws, "connected", _engine._running)
    paper = False
    if _engine and _engine.config:
        paper = _engine.config.mode.paper_trading
    return {
        "running": _engine._running if _engine else False,
        "uptime": int(time.time() - _start_time),
        "connected": ws_ok,
        "paper_trading": paper,
    }


@router.post("/stop")
async def bot_stop():
    if not _engine or not _engine._running:
        return {"status": "already_stopped"}
    try:
        if _engine._loop and _engine._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                _engine.stop(close_db=False), _engine._loop
            )
            future.result(timeout=15)
        else:
            await _engine.stop(close_db=False)
    except Exception:
        log.exception("Error stopping engine")
    return {"status": "stopped"}


@router.post("/start")
async def bot_start():
    if _engine and _engine._running:
        return {"status": "already_running"}
    if not _engine:
        return {"error": "no engine"}
    try:
        if _engine._loop and _engine._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                _engine.start(), _engine._loop
            )
            future.result(timeout=30)
        else:
            await _engine.start()
    except Exception:
        log.exception("Error starting engine")
        return {"error": "failed to start"}
    return {"status": "started"}
