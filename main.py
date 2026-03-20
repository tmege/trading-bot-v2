import asyncio
import logging
import signal
import threading
import time

from trading_bot.engine import Engine

log = logging.getLogger(__name__)


def _run_engine(engine: Engine, stop_event: threading.Event, ready_event: threading.Event):
    """Run the engine event loop. Engine starts in OFF state — user starts via GUI."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    engine._loop = loop

    async def _run():
        ready_event.set()

        # Keep the loop alive until window close (stop_event)
        # Engine starts OFF — user clicks "Start Bot" in GUI
        while not stop_event.is_set():
            await asyncio.sleep(0.5)

        # Window closed — full shutdown
        if engine._running:
            await engine.stop(close_db=True)

    try:
        loop.run_until_complete(_run())
    except Exception:
        log.exception("Engine thread error")
    finally:
        loop.close()


def main():
    engine = Engine()
    engine.create()

    stop_event = threading.Event()
    ready_event = threading.Event()

    # B-01: Handle SIGTERM/SIGHUP for graceful shutdown
    def _signal_handler(signum, frame):
        log.info("Received signal %d — shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)

    # 1. Start engine in daemon thread
    engine_thread = threading.Thread(
        target=_run_engine,
        args=(engine, stop_event, ready_event),
        daemon=True,
    )
    engine_thread.start()

    # Wait for engine to initialize (up to 30s)
    ready_event.wait(timeout=30)
    log.info("Event loop ready — starting web server (bot OFF, start via GUI)")

    # 2. Start FastAPI in daemon thread
    import uvicorn
    from trading_bot.web.app import create_app

    app = create_app(engine, stop_event)
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=8089, log_level="warning"),
        daemon=True,
    )
    server_thread.start()
    time.sleep(0.5)

    # 3. Open native window (main thread, blocks)
    import webview

    window = webview.create_window(
        "Trading Bot v2",
        "http://127.0.0.1:8089",
        width=1400,
        height=900,
        min_size=(900, 600),
        text_select=True,
    )
    log.info("Opening desktop window on http://127.0.0.1:8089")

    try:
        webview.start()
    except KeyboardInterrupt:
        pass

    # 4. Window closed -> stop engine
    log.info("Window closed — stopping bot")
    stop_event.set()
    engine_thread.join(timeout=10)
    log.info("Bot stopped cleanly")


if __name__ == "__main__":
    main()
