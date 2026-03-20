import asyncio
import logging
import os
import re
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/logs", tags=["logs"])

# V-11: Strip ANSI escape sequences
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[()][A-Z0-9]')

_engine = None


def init(engine):
    global _engine
    _engine = engine


def _get_log_path() -> Path:
    log_dir = "logs"
    if _engine and _engine.config:
        log_dir = _engine.config.logging.dir
    return Path(log_dir) / "bot.log"


def _strip_ansi(line: str) -> str:
    """V-11: Remove ANSI escape sequences to prevent terminal injection."""
    return _ANSI_RE.sub('', line)


@router.get("")
async def get_logs(n: int = Query(default=50, le=500)):
    log_path = _get_log_path()
    if not log_path.exists():
        return []
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            read_size = min(file_size, n * 500)
            f.seek(max(0, file_size - read_size))
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()
        return [_strip_ansi(l) for l in lines[-n:]]
    except Exception:
        return []


@router.get("/stream")
async def stream_logs():
    import trading_bot.web.app as _app_mod

    # H-03: Thread-safe SSE connection limit
    with _app_mod._conn_lock:
        if _app_mod._sse_connections >= _app_mod._MAX_SSE:
            async def _too_many():
                yield "data: Too many SSE connections\n\n"
            return StreamingResponse(_too_many(), media_type="text/event-stream")

    log_path = _get_log_path()

    async def event_generator():
        with _app_mod._conn_lock:
            _app_mod._sse_connections += 1
        try:
            last_size = 0
            if log_path.exists():
                last_size = os.path.getsize(log_path)

            while True:
                try:
                    if not log_path.exists():
                        await asyncio.sleep(0.3)
                        continue

                    current_size = os.path.getsize(log_path)
                    if current_size > last_size:
                        with open(log_path, "rb") as f:
                            f.seek(last_size)
                            new_data = f.read().decode("utf-8", errors="replace")
                        last_size = current_size

                        for line in new_data.splitlines():
                            if line.strip():
                                clean = _strip_ansi(line).replace("\n", "\\n")
                                yield f"data: {clean}\n\n"
                    elif current_size < last_size:
                        last_size = 0

                    await asyncio.sleep(0.3)
                except asyncio.CancelledError:
                    break
                except Exception:
                    await asyncio.sleep(1)
        finally:
            with _app_mod._conn_lock:
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
