import logging
import time
import threading

import httpx

from trading_bot.types import (
    Account, AssetCtx, AssetMeta, Book, BookLevel, Candle, Decimal,
    Fill, Mid, Order, OrderRequest, Position, Side, TIF,
)
from trading_bot.exchange.signing import Signer

log = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, max_per_minute: int = 1200):
        self._max = max_per_minute
        self._count = 0
        self._window_start = time.monotonic()
        self._lock = threading.Lock()
        self._backoff_ms = 100

    def acquire(self) -> None:
        # M-06: Compute sleep time inside lock, sleep outside
        sleep_s = 0.0
        with self._lock:
            now = time.monotonic()
            if now - self._window_start >= 60.0:
                self._count = 0
                self._window_start = now
                self._backoff_ms = 100

            if self._count >= self._max:
                sleep_s = self._backoff_ms / 1000.0
                self._backoff_ms = min(self._backoff_ms * 2, 2000)
                self._count = 0
                self._window_start = time.monotonic()

            self._count += 1

        if sleep_s > 0:
            time.sleep(sleep_s)


class RestClient:
    def __init__(
        self,
        base_url: str = "https://api.hyperliquid.xyz",
        signer: Signer | None = None,
        rate_limit: int = 1200,
    ):
        self.base_url = base_url.rstrip("/")
        self.signer = signer
        self._client = httpx.Client(timeout=30.0, verify=True)
        self._limiter = RateLimiter(rate_limit)

    def close(self) -> None:
        self._client.close()

    # --- /info endpoints (no auth) ---

    def _info_post(self, payload: dict, timeout: float | None = None) -> dict | list:
        self._limiter.acquire()
        resp = self._client.post(
            f"{self.base_url}/info", json=payload, timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_meta(self) -> list[AssetMeta]:
        data = self._info_post({"type": "meta"})
        universe = data.get("universe", []) if isinstance(data, dict) else []
        return [
            AssetMeta(
                name=a["name"],
                asset_id=i,
                sz_decimals=a.get("szDecimals", 8),
            )
            for i, a in enumerate(universe)
        ]

    def get_all_mids(self) -> dict[str, float]:
        data = self._info_post({"type": "allMids"})
        return {coin: float(mid) for coin, mid in data.items()} if isinstance(data, dict) else {}

    def get_asset_ctxs(self) -> list[AssetCtx]:
        data = self._info_post({"type": "metaAndAssetCtxs"})
        if not isinstance(data, list) or len(data) < 2:
            return []
        ctxs_raw = data[1]
        universe = data[0].get("universe", [])
        result = []
        for i, ctx in enumerate(ctxs_raw):
            coin = universe[i]["name"] if i < len(universe) else f"UNKNOWN_{i}"
            result.append(AssetCtx(
                coin=coin,
                funding_rate=float(ctx.get("funding") or 0),
                premium=float(ctx.get("premium") or 0),
                open_interest=float(ctx.get("openInterest") or 0),
                mark_px=float(ctx.get("markPx") or 0),
                day_ntl_vlm=float(ctx.get("dayNtlVlm") or 0),
                valid=True,
            ))
        return result

    def get_l2_book(self, coin: str) -> Book:
        data = self._info_post({"type": "l2Book", "coin": coin})
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
        return Book(coin=coin, bids=bids, asks=asks, timestamp_ms=int(time.time() * 1000))

    def get_candles(
        self, coin: str, interval: str, start_ms: int, end_ms: int
    ) -> list[Candle]:
        data = self._info_post({
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": interval, "startTime": start_ms, "endTime": end_ms},
        })
        if not isinstance(data, list):
            return []
        return [
            Candle(
                time_open=c["t"],
                time_close=c["T"],
                open=float(c["o"]),
                high=float(c["h"]),
                low=float(c["l"]),
                close=float(c["c"]),
                volume=float(c["v"]),
                n_trades=int(c.get("n", 0)),
            )
            for c in data
        ]

    def get_account(self, address: str, timeout: float | None = None) -> Account:
        data = self._info_post({"type": "clearinghouseState", "user": address}, timeout=timeout)
        margin = data.get("marginSummary", {})
        positions_raw = data.get("assetPositions", [])
        positions = []
        for p in positions_raw:
            pos = p.get("position", p)
            sz = float(pos.get("szi", 0))
            if sz == 0:
                continue
            positions.append(Position(
                coin=pos.get("coin", ""),
                size=Decimal.from_float(sz),
                entry_px=Decimal.from_float(float(pos.get("entryPx", 0))),
                unrealized_pnl=Decimal.from_float(float(pos.get("unrealizedPnl", 0))),
                leverage=int(float(pos.get("leverage", {}).get("value", 1))),
                is_cross=pos.get("leverage", {}).get("type", "") == "cross",
                liquidation_px=Decimal.from_float(float(pos.get("liquidationPx", 0) or 0)),
                margin_used=Decimal.from_float(float(pos.get("marginUsed", 0))),
            ))
        return Account(
            account_value=Decimal.from_float(float(margin.get("accountValue", 0))),
            total_margin_used=Decimal.from_float(float(margin.get("totalMarginUsed", 0))),
            total_unrealized_pnl=Decimal.from_float(float(margin.get("totalNtlPos", 0))),
            withdrawable=Decimal.from_float(float(margin.get("withdrawable", 0))),
            positions=positions,
        )

    def get_open_orders(self, address: str) -> list[Order]:
        data = self._info_post({"type": "openOrders", "user": address})
        if not isinstance(data, list):
            return []
        return [
            Order(
                oid=o["oid"],
                asset=0,
                coin=o.get("coin", ""),
                side=Side.BUY if o.get("side", "").lower() == "b" or o.get("side", "").lower() == "buy" else Side.SELL,
                limit_px=Decimal.from_str(str(o.get("limitPx", "0"))),
                sz=Decimal.from_str(str(o.get("sz", "0"))),
                orig_sz=Decimal.from_str(str(o.get("origSz", o.get("sz", "0")))),
                timestamp_ms=int(o.get("timestamp", 0)),
                cloid=o.get("cloid", ""),
            )
            for o in data
        ]

    def get_user_fills(self, address: str) -> list[Fill]:
        data = self._info_post({"type": "userFills", "user": address})
        if not isinstance(data, list):
            return []
        return [
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

    # --- /exchange endpoints (auth required) ---

    def _exchange_post(
        self, action: dict, vault_address: str | None = None
    ) -> dict:
        if not self.signer:
            raise RuntimeError("Signer required for /exchange requests")

        self._limiter.acquire()
        nonce = self.signer.get_nonce()
        signature = self.signer.sign(action, vault_address=vault_address, nonce=nonce)

        payload = {
            "action": action,
            "nonce": nonce,
            "signature": signature,
            "vaultAddress": vault_address,
        }

        resp = self._client.post(f"{self.base_url}/exchange", json=payload)
        resp.raise_for_status()
        result = resp.json()
        if not isinstance(result, dict):
            # H-04: Don't log raw response content (may contain sensitive data)
            log.error("Exchange API returned non-dict response (type=%s)", type(result).__name__)
        return result

    def place_order(
        self, req: OrderRequest, vault_address: str | None = None
    ) -> dict:
        order_wire = self._build_order_wire(req)
        action = {
            "type": "order",
            "orders": [order_wire],
            "grouping": self._grouping_str(req.grouping),
        }
        return self._exchange_post(action, vault_address)

    def place_orders(
        self, reqs: list[OrderRequest], vault_address: str | None = None
    ) -> dict:
        wires = [self._build_order_wire(r) for r in reqs]
        grouping = self._grouping_str(reqs[0].grouping if reqs else Grouping.NA)
        action = {
            "type": "order",
            "orders": wires,
            "grouping": grouping,
        }
        return self._exchange_post(action, vault_address)

    def cancel_order(
        self, asset: int, oid: int, vault_address: str | None = None
    ) -> dict:
        action = {
            "type": "cancel",
            "cancels": [{"a": asset, "o": oid}],
        }
        return self._exchange_post(action, vault_address)

    def cancel_orders(
        self, cancels: list[tuple[int, int]], vault_address: str | None = None
    ) -> dict:
        action = {
            "type": "cancel",
            "cancels": [{"a": a, "o": o} for a, o in cancels],
        }
        return self._exchange_post(action, vault_address)

    def update_leverage(
        self, asset: int, leverage: int, is_cross: bool = False,
        vault_address: str | None = None
    ) -> dict:
        action = {
            "type": "updateLeverage",
            "asset": asset,
            "isCross": is_cross,
            "leverage": leverage,
        }
        return self._exchange_post(action, vault_address)

    def _build_order_wire(self, req: OrderRequest) -> dict:
        wire: dict = {
            "a": req.asset,
            "b": req.side == Side.BUY,
            "p": req.price.to_str(),
            "s": req.size.to_str(),
            "r": req.reduce_only,
            "t": self._build_order_type_wire(req),
        }
        if req.cloid:
            wire["c"] = req.cloid
        return wire

    def _build_order_type_wire(self, req: OrderRequest) -> dict:
        if req.order_type == OrderType.TRIGGER:
            tp_sl = "tp" if req.tpsl == TPSL.TP else "sl"
            return {
                "trigger": {
                    "triggerPx": req.trigger_px.to_str() if req.trigger_px else "0",
                    "isMarket": req.is_market,
                    "tpsl": tp_sl,
                }
            }
        tif_map = {TIF.GTC: "Gtc", TIF.IOC: "Ioc", TIF.ALO: "Alo"}
        return {"limit": {"tif": tif_map.get(req.tif, "Gtc")}}

    @staticmethod
    def _grouping_str(g) -> str:
        from trading_bot.types import Grouping
        return {Grouping.NA: "na", Grouping.NORMAL_TPSL: "normalTpsl", Grouping.POS_TPSL: "positionTpsl"}.get(g, "na")


from trading_bot.types import Grouping, OrderType, TPSL
