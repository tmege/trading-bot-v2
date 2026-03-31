import logging
import time
import threading
from typing import Callable

from trading_bot.types import (
    Decimal, Fill, Order, OrderRequest, OrderType, Position, Side, TPSL, TIF, Grouping,
)
from trading_bot.exchange.rest import RestClient
from trading_bot.exchange.paper_exchange import PaperExchange
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.db import Database

log = logging.getLogger(__name__)

FillDispatch = Callable[[str, Fill], None]


class OrderManager:
    def __init__(
        self,
        rest: RestClient | None,
        db: Database,
        wallet_address: str = "",
        global_paper: PaperExchange | None = None,
        is_global_paper: bool = False,
        vault_address: str | None = None,
        risk_manager: RiskManager | None = None,
    ):
        self._rest = rest
        self._db = db
        self._wallet = wallet_address
        self._global_paper = global_paper
        self._is_global_paper = is_global_paper
        self._vault_address = vault_address
        self._risk_manager = risk_manager

        self._strategy_papers: dict[str, PaperExchange] = {}
        self._strategy_coins: dict[str, list[str]] = {}
        self._coin_strategy: dict[str, str] = {}

        self._on_fill_dispatch: FillDispatch | None = None
        self._reconcile_thread: threading.Thread | None = None
        self._running = False

    def set_fill_dispatch(self, callback: FillDispatch) -> None:
        self._on_fill_dispatch = callback

    def register_strategy(
        self, name: str, coins: list[str],
        paper: PaperExchange | None = None,
    ) -> None:
        self._strategy_coins[name] = coins
        for coin in coins:
            self._coin_strategy[coin] = name

        if paper:
            self._strategy_papers[name] = paper
            paper.on_fill = lambda fill, s=name: self._dispatch_fill(s, fill)

        if self._global_paper:
            self._global_paper.on_fill = self._dispatch_global_paper_fill

    def _dispatch_fill(self, strategy: str, fill: Fill) -> None:
        self._db.insert_trade(
            oid=fill.oid, tid=fill.tid, coin=fill.coin,
            side=fill.side.value, price=fill.px.to_float(),
            size=fill.sz.to_float(), fee=fill.fee.to_float(),
            closed_pnl=fill.closed_pnl.to_float(),
            strategy=strategy, time_ms=fill.time_ms,
            hash_val=fill.hash,
        )
        if self._on_fill_dispatch:
            self._on_fill_dispatch(strategy, fill)

    def _dispatch_global_paper_fill(self, fill: Fill) -> None:
        strategy = self._coin_strategy.get(fill.coin, "")
        if strategy:
            self._dispatch_fill(strategy, fill)

    def _get_exchange(self, strategy: str) -> PaperExchange | None:
        if strategy in self._strategy_papers:
            return self._strategy_papers[strategy]
        if self._is_global_paper and self._global_paper:
            return self._global_paper
        return None

    def _get_vault(self, strategy: str) -> str | None:
        return self._vault_address

    def set_leverage(self, strategy: str, coin: str, leverage: int) -> None:
        paper = self._get_exchange(strategy)
        if paper:
            paper.set_leverage(coin, leverage)
            return
        if self._rest:
            try:
                asset = 0
                vault = self._get_vault(strategy)
                self._rest.update_leverage(asset, leverage, vault_address=vault)
            except Exception:
                log.exception(f"Failed to set leverage {leverage}x for {coin}")

    def place_order(
        self, strategy: str, req: OrderRequest,
        vault_override: str | None = None,
    ) -> int:
        # Risk check — skip for reduce_only (position closing must always work)
        if self._risk_manager and not req.reduce_only:
            try:
                account_value = self.get_account_value(strategy)
                allowed, reason = self._risk_manager.check_order(req, account_value, strategy)
                if not allowed:
                    log.warning("Order blocked by risk manager [%s]: %s %s %s — %s",
                                strategy, req.side, req.size, req.coin, reason)
                    return 0
            except Exception:
                log.exception("Risk check failed — allowing order (fail-open)")

        paper = self._get_exchange(strategy)

        if paper:
            oid = paper.place_order(req)
            self._db.map_order(oid, strategy, req.coin, int(time.time() * 1000))
            return oid

        if not self._rest:
            raise RuntimeError("No REST client for live trading")

        vault = vault_override or self._get_vault(strategy)
        result = self._rest.place_order(req, vault_address=vault)

        if not isinstance(result, dict):
            log.error(f"Order API returned non-dict: {result}")
            return 0

        response = result.get("response", {})
        if not isinstance(response, dict):
            log.error(f"Order API response is not a dict: {response}")
            return 0

        data = response.get("data", {})
        if not isinstance(data, dict):
            log.error(f"Order API data is not a dict: {data}")
            return 0

        statuses = data.get("statuses", [])
        oid = 0
        if statuses:
            status = statuses[0]
            if "resting" in status:
                oid = status["resting"].get("oid", 0)
            elif "filled" in status:
                oid = status["filled"].get("oid", 0)
            elif "error" in status:
                log.error(f"Order rejected: {status['error']}")
                return 0

        if oid:
            self._db.map_order(oid, strategy, req.coin, int(time.time() * 1000))

        return oid

    def cancel_order(self, strategy: str, coin: str, oid: int) -> bool:
        paper = self._get_exchange(strategy)
        if paper:
            return paper.cancel_order(oid)

        if not self._rest:
            return False

        try:
            asset = 0
            vault = self._get_vault(strategy)
            self._rest.cancel_order(asset, oid, vault_address=vault)
            return True
        except Exception:
            log.exception(f"Failed to cancel order {oid}")
            return False

    def cancel_all(self, strategy: str, coin: str) -> int:
        paper = self._get_exchange(strategy)
        if paper:
            return paper.cancel_all(coin)

        if not self._rest:
            return 0

        try:
            orders = self._rest.get_open_orders(self._wallet)
            to_cancel = [(0, o.oid) for o in orders if o.coin == coin]
            if to_cancel:
                vault = self._get_vault(strategy)
                self._rest.cancel_orders(to_cancel, vault_address=vault)
            return len(to_cancel)
        except Exception:
            log.exception(f"Failed to cancel all orders for {coin}")
            return 0

    def get_position(self, strategy: str, coin: str) -> Position | None:
        paper = self._get_exchange(strategy)
        if paper:
            return paper.get_position(coin)

        if not self._rest:
            return None

        try:
            account = self._rest.get_account(self._wallet)
            for pos in account.positions:
                if pos.coin == coin:
                    return pos
        except Exception:
            log.exception(f"Failed to get position for {coin}")
        return None

    def get_open_orders(self, strategy: str, coin: str) -> list[Order]:
        paper = self._get_exchange(strategy)
        if paper:
            return paper.get_open_orders(coin)

        if not self._rest:
            return []

        try:
            all_orders = self._rest.get_open_orders(self._wallet)
            return [o for o in all_orders if o.coin == coin]
        except Exception:
            log.exception(f"Failed to get open orders for {coin}")
            return []

    def get_account_value(self, strategy: str) -> float:
        paper = self._get_exchange(strategy)
        if paper:
            return paper.get_account_value()

        if not self._rest:
            return 0.0

        try:
            account = self._rest.get_account(self._wallet)
            return account.account_value.to_float()
        except Exception:
            log.exception("Failed to get account value")
            return 0.0

    def feed_mid(self, coin: str, price: float) -> None:
        if self._global_paper:
            self._global_paper.feed_mid(coin, price)
        for paper in self._strategy_papers.values():
            paper.feed_mid(coin, price)

    def handle_live_fill(self, fill: Fill) -> None:
        if not fill.coin or fill.oid == 0:
            return

        strategy = self._db.get_order_strategy(fill.oid)

        if not strategy and fill.closed_pnl.to_float() != 0:
            strategy = self._coin_strategy.get(fill.coin)
            if strategy:
                log.info(f"Trigger OID mismatch fallback: fill {fill.oid} → {strategy}")

        if strategy:
            self._dispatch_fill(strategy, fill)
        else:
            log.warning(f"Fill {fill.oid} on {fill.coin} has no mapped strategy")

    def handle_live_order_update(self, orders: list[Order]) -> None:
        pass

    def start_reconciliation(self, interval_sec: float = 30.0) -> None:
        self._running = True
        self._reconcile_thread = threading.Thread(
            target=self._reconcile_loop,
            args=(interval_sec,),
            daemon=True,
        )
        self._reconcile_thread.start()

    def stop_reconciliation(self) -> None:
        self._running = False

    def _reconcile_loop(self, interval_sec: float) -> None:
        while self._running:
            try:
                self._do_reconcile()
            except Exception:
                log.exception("Reconciliation error")
            time.sleep(interval_sec)

    def _do_reconcile(self) -> None:
        if not self._rest or self._is_global_paper:
            return
        self._db.cleanup_old_orders(max_age_ms=86400000)
