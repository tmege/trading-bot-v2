import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from trading_bot.types import (
    Decimal, Fill, Order, OrderRequest, OrderType, Position, Side, TPSL, TIF,
)

log = logging.getLogger(__name__)

MAKER_FEE = 0.00015
TAKER_FEE = 0.00045

FillCallback = Callable[[Fill], None]


@dataclass
class PaperPosition:
    coin: str
    size: float = 0.0
    entry_px: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class PaperOrder:
    oid: int
    coin: str
    side: Side
    price: float
    size: float
    order_type: OrderType = OrderType.LIMIT
    trigger_px: float = 0.0
    tpsl: TPSL | None = None
    reduce_only: bool = False
    tif: TIF = TIF.GTC
    placed_at_ms: int = 0
    placed_at_idx: int = -1


class PaperExchange:
    def __init__(self, name: str, initial_balance: float = 500.0):
        self.name = name
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self._positions: dict[str, PaperPosition] = {}
        self._orders: list[PaperOrder] = []
        self._leverages: dict[str, int] = {}
        self._next_oid = 1000000
        self._next_tid = 2000000
        self._fills: list[Fill] = []
        self.on_fill: FillCallback | None = None

    @property
    def equity(self) -> float:
        total_unreal = sum(p.unrealized_pnl for p in self._positions.values())
        return self.balance + total_unreal

    def set_leverage(self, coin: str, leverage: int) -> None:
        self._leverages[coin] = leverage

    def get_position(self, coin: str) -> Position | None:
        pp = self._positions.get(coin)
        if not pp or pp.size == 0:
            return None
        return Position(
            coin=coin,
            size=Decimal.from_float(pp.size),
            entry_px=Decimal.from_float(pp.entry_px),
            unrealized_pnl=Decimal.from_float(pp.unrealized_pnl),
            realized_pnl=Decimal.from_float(pp.realized_pnl),
            leverage=self._leverages.get(coin, 1),
        )

    def get_open_orders(self, coin: str | None = None) -> list[Order]:
        orders = self._orders if coin is None else [o for o in self._orders if o.coin == coin]
        return [
            Order(
                oid=o.oid, asset=0, coin=o.coin, side=o.side,
                limit_px=Decimal.from_float(o.price),
                sz=Decimal.from_float(o.size),
                orig_sz=Decimal.from_float(o.size),
                timestamp_ms=o.placed_at_ms,
                reduce_only=o.reduce_only,
                tif=o.tif,
            )
            for o in orders
        ]

    def place_order(self, req: OrderRequest) -> int:
        oid = self._next_oid
        self._next_oid += 1

        price = req.price.to_float()
        trigger = req.trigger_px.to_float() if req.trigger_px else 0.0

        order = PaperOrder(
            oid=oid,
            coin=req.coin,
            side=req.side,
            price=price if req.order_type == OrderType.LIMIT else trigger,
            size=req.size.to_float(),
            order_type=req.order_type,
            trigger_px=trigger,
            tpsl=req.tpsl,
            reduce_only=req.reduce_only,
            tif=req.tif,
            placed_at_ms=int(time.time() * 1000),
        )
        self._orders.append(order)
        return oid

    def cancel_order(self, oid: int) -> bool:
        for i, o in enumerate(self._orders):
            if o.oid == oid:
                self._orders.pop(i)
                return True
        return False

    def cancel_all(self, coin: str) -> int:
        before = len(self._orders)
        self._orders = [o for o in self._orders if o.coin != coin]
        return before - len(self._orders)

    def feed_mid(self, coin: str, price: float) -> list[Fill]:
        fills = []
        to_remove = []

        for order in self._orders:
            if order.coin != coin:
                continue

            filled = self._check_fill(order, price)
            if filled:
                fill = self._execute_fill(order, price)
                fills.append(fill)
                to_remove.append(order.oid)

        self._orders = [o for o in self._orders if o.oid not in to_remove]

        self._update_unrealized(coin, price)

        for fill in fills:
            if self.on_fill:
                self.on_fill(fill)

        return fills

    def _check_fill(self, order: PaperOrder, price: float) -> bool:
        if order.order_type == OrderType.LIMIT:
            if order.side == Side.BUY:
                return price <= order.price
            return price >= order.price

        if order.tpsl == TPSL.TP:
            pos = self._positions.get(order.coin)
            if pos and pos.size > 0:
                return price >= order.trigger_px
            elif pos and pos.size < 0:
                return price <= order.trigger_px
        elif order.tpsl == TPSL.SL:
            pos = self._positions.get(order.coin)
            if pos and pos.size > 0:
                return price <= order.trigger_px
            elif pos and pos.size < 0:
                return price >= order.trigger_px

        return False

    def _execute_fill(self, order: PaperOrder, price: float) -> Fill:
        is_taker = order.order_type == OrderType.TRIGGER or order.tif == TIF.IOC
        fee_rate = TAKER_FEE if is_taker else MAKER_FEE

        fill_px = order.price if order.order_type == OrderType.LIMIT else price
        notional = fill_px * order.size
        fee = notional * fee_rate

        pos = self._positions.get(order.coin)
        if not pos:
            pos = PaperPosition(coin=order.coin)
            self._positions[order.coin] = pos

        closed_pnl = 0.0
        signed_size = order.size if order.side == Side.BUY else -order.size

        if pos.size != 0 and (
            (pos.size > 0 and order.side == Side.SELL) or
            (pos.size < 0 and order.side == Side.BUY)
        ):
            close_size = min(abs(signed_size), abs(pos.size))
            if pos.size > 0:
                closed_pnl = (fill_px - pos.entry_px) * close_size
            else:
                closed_pnl = (pos.entry_px - fill_px) * close_size

            remaining_pos = abs(pos.size) - close_size
            if remaining_pos <= 1e-12:
                remaining_add = abs(signed_size) - close_size
                if remaining_add > 1e-12:
                    pos.entry_px = fill_px
                    pos.size = remaining_add * (1 if order.side == Side.BUY else -1)
                else:
                    pos.size = 0.0
                    pos.entry_px = 0.0
            else:
                pos.size = remaining_pos * (1 if pos.size > 0 else -1)

            pos.realized_pnl += closed_pnl
            self.balance += closed_pnl
        else:
            if pos.size == 0:
                pos.entry_px = fill_px
                pos.size = signed_size
            else:
                total = abs(pos.size) + order.size
                pos.entry_px = (pos.entry_px * abs(pos.size) + fill_px * order.size) / total
                pos.size += signed_size

        self.balance -= fee

        tid = self._next_tid
        self._next_tid += 1

        return Fill(
            coin=order.coin,
            px=Decimal.from_float(fill_px),
            sz=Decimal.from_float(order.size),
            side=order.side,
            time_ms=int(time.time() * 1000),
            closed_pnl=Decimal.from_float(closed_pnl),
            fee=Decimal.from_float(fee),
            oid=order.oid,
            tid=tid,
            crossed=is_taker,
        )

    def _update_unrealized(self, coin: str, price: float) -> None:
        pos = self._positions.get(coin)
        if not pos or pos.size == 0:
            return
        if pos.size > 0:
            pos.unrealized_pnl = (price - pos.entry_px) * abs(pos.size)
        else:
            pos.unrealized_pnl = (pos.entry_px - price) * abs(pos.size)

    def get_account_value(self) -> float:
        return self.equity

    def get_daily_pnl(self) -> float:
        return sum(
            f.closed_pnl.to_float() - f.fee.to_float()
            for f in self._fills
        )

    def reset(self) -> None:
        self._positions.clear()
        self._orders.clear()
        self._fills.clear()
        self.balance = self.initial_balance

    def to_dict(self) -> dict:
        positions = {}
        for coin, pp in self._positions.items():
            if pp.size != 0:
                positions[coin] = {
                    "size": pp.size,
                    "entry_px": pp.entry_px,
                    "realized_pnl": pp.realized_pnl,
                    "unrealized_pnl": pp.unrealized_pnl,
                }
        orders = []
        for o in self._orders:
            orders.append({
                "oid": o.oid, "coin": o.coin, "side": o.side.value,
                "price": o.price, "size": o.size,
                "order_type": o.order_type.value, "trigger_px": o.trigger_px,
                "tpsl": o.tpsl.value if o.tpsl else None,
                "reduce_only": o.reduce_only, "tif": o.tif.value,
                "placed_at_ms": o.placed_at_ms,
            })
        return {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "positions": positions,
            "orders": orders,
            "leverages": self._leverages,
            "next_oid": self._next_oid,
            "next_tid": self._next_tid,
        }

    def from_dict(self, d: dict) -> None:
        self.balance = d.get("balance", self.initial_balance)
        self.initial_balance = d.get("initial_balance", self.initial_balance)
        self._next_oid = d.get("next_oid", self._next_oid)
        self._next_tid = d.get("next_tid", self._next_tid)
        self._leverages = {k: int(v) for k, v in d.get("leverages", {}).items()}

        for coin, pdata in d.get("positions", {}).items():
            self._positions[coin] = PaperPosition(
                coin=coin,
                size=pdata["size"],
                entry_px=pdata["entry_px"],
                realized_pnl=pdata.get("realized_pnl", 0.0),
                unrealized_pnl=pdata.get("unrealized_pnl", 0.0),
            )

        for odata in d.get("orders", []):
            tpsl_val = odata.get("tpsl")
            self._orders.append(PaperOrder(
                oid=odata["oid"], coin=odata["coin"],
                side=Side(odata["side"]),
                price=odata["price"], size=odata["size"],
                order_type=OrderType(odata["order_type"]),
                trigger_px=odata.get("trigger_px", 0.0),
                tpsl=TPSL(tpsl_val) if tpsl_val is not None else None,
                reduce_only=odata.get("reduce_only", False),
                tif=TIF(odata.get("tif", "Gtc")),
                placed_at_ms=odata.get("placed_at_ms", 0),
            ))

        log.info(f"Paper exchange '{self.name}' state restored: balance={self.balance:.2f}, positions={len(self._positions)}, orders={len(self._orders)}")
