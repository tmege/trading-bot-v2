import asyncio
import json
import logging
import time
from typing import Callable

import websockets

from trading_bot.types import (
    Book, BookLevel, Candle, Decimal, Fill, Mid, Order, Side, TIF,
)

log = logging.getLogger(__name__)

MidsCallback = Callable[[list[Mid]], None]
BookCallback = Callable[[Book], None]
CandleCallback = Callable[[str, Candle], None]
OrderUpdateCallback = Callable[[list[Order]], None]
FillCallback = Callable[[list[Fill]], None]


class WebSocketClient:
    def __init__(self, url: str = "wss://api.hyperliquid.xyz/ws", user_address: str = ""):
        self.url = url
        self.user_address = user_address
        self._ws = None
        self._subscriptions: list[dict] = []
        self._running = False
        self._task: asyncio.Task | None = None

        self.on_mids: MidsCallback | None = None
        self.on_book: BookCallback | None = None
        self.on_candle: CandleCallback | None = None
        self.on_order_update: OrderUpdateCallback | None = None
        self.on_fill: FillCallback | None = None

    async def connect(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def disconnect(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def subscribe_all_mids(self) -> None:
        self._add_subscription({"type": "allMids"})

    def subscribe_l2_book(self, coin: str) -> None:
        self._add_subscription({"type": "l2Book", "coin": coin})

    def subscribe_candle(self, coin: str, interval: str) -> None:
        self._add_subscription({"type": "candle", "coin": coin, "interval": interval})

    def subscribe_order_updates(self) -> None:
        if self.user_address:
            self._add_subscription({"type": "orderUpdates", "user": self.user_address})

    def subscribe_user_fills(self) -> None:
        if self.user_address:
            self._add_subscription({"type": "userFills", "user": self.user_address})

    def _add_subscription(self, sub: dict) -> None:
        if sub not in self._subscriptions:
            self._subscriptions.append(sub)
        if self._ws:
            asyncio.create_task(self._send_subscribe(sub))

    async def _send_subscribe(self, sub: dict) -> None:
        if self._ws:
            msg = json.dumps({"method": "subscribe", "subscription": sub})
            await self._ws.send(msg)

    async def _resubscribe_all(self) -> None:
        for sub in self._subscriptions:
            await self._send_subscribe(sub)

    async def _run_loop(self) -> None:
        backoff = 0.1
        max_backoff = 60.0

        while self._running:
            try:
                async with websockets.connect(self.url) as ws:
                    self._ws = ws
                    backoff = 0.1
                    log.info("WebSocket connected")
                    await self._resubscribe_all()

                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            self._dispatch(msg)
                        except json.JSONDecodeError:
                            log.warning("WS: invalid JSON received")
                        except Exception:
                            log.exception("WS: error processing message")

            except asyncio.CancelledError:
                break
            except Exception:
                log.warning(f"WS disconnected, reconnecting in {backoff:.1f}s")
                self._ws = None
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

        self._ws = None
        log.info("WebSocket loop ended")

    def _dispatch(self, msg: dict) -> None:
        channel = msg.get("channel", "")
        data = msg.get("data", {})

        if channel == "allMids":
            self._handle_mids(data)
        elif channel == "l2Book":
            self._handle_book(data)
        elif channel == "candle":
            self._handle_candle(data)
        elif channel == "orderUpdates":
            self._handle_order_updates(data)
        elif channel == "userFills":
            self._handle_fills(data)

    def _handle_mids(self, data: dict) -> None:
        if not self.on_mids:
            return
        mids_raw = data.get("mids", {})
        mids = [
            Mid(coin=coin, mid=Decimal.from_float(float(val)))
            for coin, val in mids_raw.items()
        ]
        self.on_mids(mids)

    def _handle_book(self, data: dict) -> None:
        if not self.on_book:
            return
        coin = data.get("coin", "")
        levels = data.get("levels", [[], []])
        bids = [
            BookLevel(
                px=Decimal.from_str(lv.get("px", "0")),
                sz=Decimal.from_str(lv.get("sz", "0")),
                n_orders=int(lv.get("n", 0)),
            )
            for lv in levels[0][:20]
        ]
        asks = [
            BookLevel(
                px=Decimal.from_str(lv.get("px", "0")),
                sz=Decimal.from_str(lv.get("sz", "0")),
                n_orders=int(lv.get("n", 0)),
            )
            for lv in levels[1][:20]
        ]
        book = Book(coin=coin, bids=bids, asks=asks, timestamp_ms=int(time.time() * 1000))
        self.on_book(book)

    def _handle_candle(self, data: dict) -> None:
        if not self.on_candle:
            return
        if isinstance(data, dict):
            data = [data]
        for c in data:
            coin = c.get("s", "").replace("-USD", "").replace("USDT", "")
            candle = Candle(
                time_open=c.get("t", 0),
                time_close=c.get("T", 0),
                open=float(c.get("o", 0)),
                high=float(c.get("h", 0)),
                low=float(c.get("l", 0)),
                close=float(c.get("c", 0)),
                volume=float(c.get("v", 0)),
                n_trades=int(c.get("n", 0)),
            )
            self.on_candle(coin, candle)

    def _handle_order_updates(self, data: list | dict) -> None:
        if not self.on_order_update:
            return
        if isinstance(data, dict):
            data = [data]
        orders = []
        for o in data:
            order = o.get("order", o)
            side_str = order.get("side", "").lower()
            orders.append(Order(
                oid=int(order.get("oid", 0)),
                asset=int(order.get("asset", 0)),
                coin=order.get("coin", ""),
                side=Side.BUY if side_str in ("b", "buy") else Side.SELL,
                limit_px=Decimal.from_str(str(order.get("limitPx", "0"))),
                sz=Decimal.from_str(str(order.get("sz", "0"))),
                orig_sz=Decimal.from_str(str(order.get("origSz", order.get("sz", "0")))),
                timestamp_ms=int(order.get("timestamp", 0)),
            ))
        self.on_order_update(orders)

    def _handle_fills(self, data: list | dict) -> None:
        if not self.on_fill:
            return
        if not data:
            return
        if isinstance(data, dict):
            data = [data]
        fills = [
            Fill(
                coin=f.get("coin", ""),
                px=Decimal.from_str(str(f.get("px", "0"))),
                sz=Decimal.from_str(str(f.get("sz", "0"))),
                side=Side.BUY if f.get("side", "").lower() in ("b", "buy") else Side.SELL,
                time_ms=int(f.get("time", 0)),
                closed_pnl=Decimal.from_str(str(f.get("closedPnl", "0"))),
                fee=Decimal.from_str(str(f.get("fee", "0"))),
                oid=int(f.get("oid", 0)),
                tid=int(f.get("tid", 0)),
                crossed=f.get("crossed", False),
                hash=f.get("hash", ""),
            )
            for f in data
        ]
        self.on_fill(fills)
