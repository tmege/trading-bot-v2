import asyncio
import json
import logging
import time

from trading_bot.config import BotConfig, load_config
from trading_bot.db import Database
from trading_bot.logging_config import setup_logging
from trading_bot.types import Book, Decimal, Fill, Mid, Order, OrderRequest, Side, TIF
from trading_bot.exchange.signing import Signer
from trading_bot.exchange.rest import RestClient
from trading_bot.exchange.ws import WebSocketClient
from trading_bot.exchange.paper_exchange import PaperExchange
from trading_bot.exchange.order_manager import OrderManager
from trading_bot.strategy.api import StrategyAPI
from trading_bot.strategy.loader import StrategyLoader, StrategyInfo
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.data.data_manager import DataManager
from trading_bot.report.dashboard import Dashboard

log = logging.getLogger(__name__)


class Engine:
    def __init__(self):
        self.config: BotConfig | None = None
        self.db: Database | None = None
        self.signer: Signer | None = None
        self.rest: RestClient | None = None
        self.ws: WebSocketClient | None = None
        self.order_manager: OrderManager | None = None
        self.risk_manager: RiskManager | None = None
        self.data_manager: DataManager | None = None
        self.loader: StrategyLoader | None = None

        self._global_paper: PaperExchange | None = None
        self._mid_prices: dict[str, float] = {}
        self._asset_ctxs: dict = {}
        self._strategies: list[StrategyInfo] = []
        self._timer_task: asyncio.Task | None = None
        self._reload_task: asyncio.Task | None = None
        self._running = False
        self._last_emergency_check: float = 0.0
        self._dashboard: Dashboard | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def create(self, config_path: str = "config/bot_config.json") -> None:
        self.config = load_config(config_path)
        setup_logging(self.config.logging.dir, self.config.logging.level)
        log.info("Engine creating...")

        self.db = Database(self.config.database.path)
        self.db.open()

        log.info("Engine created")

    async def start(self) -> None:
        if not self.config:
            raise RuntimeError("Call create() first")

        # Re-open DB if closed (after full stop)
        if not self.db or not self.db.conn:
            self.db = Database(self.config.database.path)
            self.db.open()

        # Reload config to pick up any changes
        self.config = load_config("config/bot_config.json")

        log.info("Engine starting...")
        cfg = self.config
        self._running = True
        self._loop = asyncio.get_event_loop()

        # 1. Signer — C-02: try/finally ensures key is always wiped
        if cfg.private_key and not cfg.mode.paper_trading:
            try:
                self.signer = Signer(cfg.private_key, cfg.exchange.is_testnet)
            finally:
                cfg.private_key = ""

        # 2. REST client
        self.rest = RestClient(
            base_url=cfg.exchange.rest_url,
            signer=self.signer,
            rate_limit=1200,
        )

        # 3. Fetch metadata
        try:
            metas = self.rest.get_meta()
            log.info(f"Loaded {len(metas)} asset metas")
            ctxs = self.rest.get_asset_ctxs()
            self._asset_ctxs = {c.coin: c for c in ctxs}
        except Exception:
            log.warning("Failed to fetch asset metadata")

        # 4. Paper exchanges
        if cfg.mode.paper_trading:
            self._global_paper = PaperExchange("global", cfg.mode.paper_initial_balance)

        # 5. Risk manager
        self.risk_manager = RiskManager(cfg.risk)

        # 6. Order manager
        self.order_manager = OrderManager(
            rest=self.rest,
            db=self.db,
            wallet_address=cfg.wallet_address,
            global_paper=self._global_paper,
            is_global_paper=cfg.mode.paper_trading,
            vault_address=cfg.exchange.vault_address,
            risk_manager=self.risk_manager,
        )
        self.order_manager.set_fill_dispatch(self._on_strategy_fill)

        # 7. Data manager
        self.data_manager = DataManager(cfg.sentiment)

        # 8. Load strategies
        self.loader = StrategyLoader(cfg.strategies.dir)
        self._load_strategies()

        # 8b. Restore engine state from DB
        self._restore_engine_state()

        # 9. Reconcile positions
        await self._reconcile_positions()

        # 10. WebSocket
        address = self.signer.address if self.signer else cfg.wallet_address
        self.ws = WebSocketClient(cfg.exchange.ws_url, user_address=address)
        self.ws.on_mids = self._on_mids
        self.ws.on_book = self._on_book
        self.ws.on_fill = self._on_fills
        self.ws.on_order_update = self._on_order_updates

        self.ws.subscribe_all_mids()
        if address:
            self.ws.subscribe_order_updates()
            self.ws.subscribe_user_fills()

        for info in self._strategies:
            for coin in info.coins:
                self.ws.subscribe_candle(coin, "1h")

        await self.ws.connect()

        # 11. Reconciliation thread
        if not cfg.mode.paper_trading:
            self.order_manager.start_reconciliation(30.0)

        # 12. Timer
        self._timer_task = asyncio.create_task(self._timer_loop())
        self._reload_task = asyncio.create_task(self._reload_loop())

        # 13. Dashboard
        self._dashboard = Dashboard(
            mid_prices=self._mid_prices,
            strategies=self._strategies,
            order_manager=self.order_manager,
            data_manager=self.data_manager,
        )
        self._dashboard.start()

        log.info("Engine started")

    async def stop(self, close_db: bool = True) -> None:
        log.info("Engine stopping...")
        self._running = False

        if self._timer_task:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
            self._timer_task = None

        if self._reload_task:
            self._reload_task.cancel()
            try:
                await self._reload_task
            except asyncio.CancelledError:
                pass
            self._reload_task = None

        if self.ws:
            await self.ws.disconnect()
            self.ws = None

        if self._dashboard:
            self._dashboard.stop()
            self._dashboard = None

        if self.order_manager:
            self.order_manager.stop_reconciliation()

        # Save engine state before shutdown
        self._save_engine_state()

        for info in self._strategies:
            if info.instance and not info.errored and not info.disabled:
                try:
                    info.instance.on_shutdown()
                except Exception:
                    log.exception(f"Error in {info.name}.on_shutdown()")

        if self.data_manager:
            self.data_manager.close()
            self.data_manager = None

        if self.rest:
            self.rest.close()
            self.rest = None

        # Clear strategy/order state for clean restart
        self._strategies.clear()
        self._mid_prices.clear()
        self._asset_ctxs.clear()
        self.order_manager = None
        self.risk_manager = None
        self.loader = None
        self.signer = None
        self._global_paper = None

        if close_db and self.db:
            self.db.close()

        log.info("Engine stopped")

    # --- Strategy loading ---

    def _load_strategies(self) -> None:
        if not self.config or not self.loader or not self.db or not self.order_manager:
            return

        for entry in self.config.strategies.active:
            try:
                is_paper = entry.paper_mode if entry.paper_mode is not None else self.config.mode.paper_trading
                paper_ex = None

                if is_paper and not self.config.mode.paper_trading:
                    paper_ex = PaperExchange(
                        f"paper_{entry.file}",
                        self.config.mode.paper_initial_balance,
                    )

                api = StrategyAPI(
                    strategy_name=entry.file.replace(".py", ""),
                    coin=entry.coins[0] if entry.coins else "",
                    config=self.config,
                    db=self.db,
                    order_manager=self.order_manager,
                    rest_client=self.rest,
                    mid_prices=self._mid_prices,
                    asset_ctxs=self._asset_ctxs,
                    data_manager=self.data_manager,
                )

                info = self.loader.load_strategy(entry.file, entry.coins, api)
                self._strategies.append(info)

                self.order_manager.register_strategy(
                    info.name, entry.coins, paper=paper_ex,
                )

            except Exception:
                log.exception(f"Failed to load strategy {entry.file}")

    # --- State persistence ---

    def _save_engine_state(self) -> None:
        if not self.db:
            return
        try:
            state = {}

            # Risk manager state
            if self.risk_manager:
                state["risk_manager"] = self.risk_manager.to_dict()

            # Paper exchange states
            paper_states = {}
            if self._global_paper:
                paper_states["__global__"] = self._global_paper.to_dict()
            if self.order_manager:
                for name, paper in self.order_manager._strategy_papers.items():
                    paper_states[name] = paper.to_dict()
            if paper_states:
                state["paper_exchanges"] = paper_states

            # Disabled strategies
            disabled = [info.name for info in self._strategies if info.disabled]
            if disabled:
                state["disabled_strategies"] = disabled

            self.db.save_state("__engine__", "state", json.dumps(state))
            log.info("Engine state saved")
        except Exception:
            log.exception("Failed to save engine state")

    def _restore_engine_state(self) -> None:
        if not self.db:
            return
        try:
            raw = self.db.load_state("__engine__", "state")
            if not raw:
                return

            state = json.loads(raw)

            # Risk manager
            if self.risk_manager and "risk_manager" in state:
                self.risk_manager.from_dict(state["risk_manager"])

            # Paper exchanges
            paper_states = state.get("paper_exchanges", {})
            if self._global_paper and "__global__" in paper_states:
                self._global_paper.from_dict(paper_states["__global__"])
            if self.order_manager:
                for name, paper in self.order_manager._strategy_papers.items():
                    if name in paper_states:
                        paper.from_dict(paper_states[name])

            # Disabled strategies
            for name in state.get("disabled_strategies", []):
                for info in self._strategies:
                    if info.name == name:
                        info.disabled = True
                        log.info(f"Strategy '{name}' restored as disabled")

            log.info("Engine state restored")
        except Exception:
            log.exception("Failed to restore engine state")

    # --- Reconciliation ---

    async def _reconcile_positions(self) -> None:
        if not self.rest or not self.config or self.config.mode.paper_trading:
            return

        try:
            address = self.config.wallet_address
            if not address:
                return

            account = self.rest.get_account(address)
            open_orders = self.rest.get_open_orders(address)

            coin_to_strategy = {}
            for info in self._strategies:
                for coin in info.coins:
                    coin_to_strategy[coin] = info.name

            for pos in account.positions:
                if pos.coin not in coin_to_strategy:
                    log.warning(f"Orphaned position on {pos.coin}")

            for order in open_orders:
                strategy = self.db.get_order_strategy(order.oid)
                if not strategy:
                    log.warning(f"Orphaned order OID={order.oid} on {order.coin}")

        except Exception:
            log.exception("Position reconciliation failed")

    # --- WS Callbacks ---

    def _on_mids(self, mids: list[Mid]) -> None:
        for mid in mids:
            price = mid.mid.to_float()
            self._mid_prices[mid.coin] = price

            if self.order_manager:
                self.order_manager.feed_mid(mid.coin, price)

            if self.risk_manager:
                self.risk_manager.update_price(mid.coin, price)

        # Throttled emergency close check (every 5s instead of every 60s)
        now = time.time()
        if now - self._last_emergency_check >= 5.0 and self._loop:
            self._last_emergency_check = now
            self._loop.create_task(self._check_emergency_close())

        for info in self._strategies:
            if info.errored or info.disabled or not info.instance:
                continue
            for coin in info.coins:
                price = self._mid_prices.get(coin)
                if price is not None:
                    try:
                        info.instance.on_tick(coin, price)
                    except Exception:
                        log.exception(f"Strategy {info.name} error in on_tick — suspended")
                        info.errored = True

    def _on_book(self, book: Book) -> None:
        for info in self._strategies:
            if info.errored or info.disabled or not info.instance:
                continue
            if book.coin in info.coins:
                try:
                    info.instance.on_book(book)
                except Exception:
                    log.exception(f"Strategy {info.name} error in on_book — suspended")
                    info.errored = True

    def _on_fills(self, fills: list[Fill]) -> None:
        if self.order_manager:
            for fill in fills:
                self.order_manager.handle_live_fill(fill)

    def _on_order_updates(self, orders: list[Order]) -> None:
        if self.order_manager:
            self.order_manager.handle_live_order_update(orders)

    def _on_strategy_fill(self, strategy: str, fill: Fill) -> None:
        info = self.loader.get(strategy) if self.loader else None
        if info and info.instance and not info.errored and not info.disabled:
            try:
                info.instance.on_fill(fill)
            except Exception:
                log.exception(f"Strategy {info.name} error in on_fill — suspended")
                info.errored = True

        if self.risk_manager:
            self.risk_manager.record_trade(
                fill.closed_pnl.to_float(), fill.fee.to_float()
            )

    # --- Timer ---

    def _get_total_unrealized_pnl(self) -> float:
        """Sum unrealized PnL across all strategy positions."""
        total = 0.0
        if not self.order_manager:
            return total
        for info in self._strategies:
            if info.errored or info.disabled:
                continue
            for coin in info.coins:
                try:
                    pos = self.order_manager.get_position(info.name, coin)
                    if pos and pos.unrealized_pnl:
                        total += pos.unrealized_pnl.to_float() if hasattr(pos.unrealized_pnl, 'to_float') else float(pos.unrealized_pnl)
                except Exception:
                    pass
        return total

    async def _check_emergency_close(self) -> None:
        """Check if daily losses (realized + unrealized) exceed emergency threshold."""
        if not self.risk_manager or not self.config:
            return

        try:
            account_value = 0.0
            if self._global_paper:
                account_value = self._global_paper.equity
            elif self.rest and self.config.wallet_address:
                acc = self.rest.get_account(self.config.wallet_address)
                account_value = acc.account_value.to_float()

            if account_value <= 0:
                return

            realized_pnl = self.risk_manager._daily_pnl
            unrealized_pnl = self._get_total_unrealized_pnl()
            total_pnl = realized_pnl + unrealized_pnl

            if self.risk_manager.check_emergency_close(account_value, total_pnl):
                log.critical(
                    "EMERGENCY CLOSE: daily PnL $%.2f (realized $%.2f + unrealized $%.2f) exceeds %.1f%% of $%.2f",
                    total_pnl, realized_pnl, unrealized_pnl,
                    self.config.risk.emergency_close_pct, account_value,
                )
                await self._emergency_close_all()
        except Exception:
            log.exception("Error checking emergency close")

    async def _emergency_close_all(self) -> None:
        """Close all positions and suspend all strategies."""
        log.critical("EMERGENCY: Closing all positions")

        # 1. Suspend all strategies immediately
        for info in self._strategies:
            info.errored = True

        # 2. Pause risk manager
        if self.risk_manager:
            self.risk_manager.pause()

        # 3. Close paper positions
        if self.order_manager:
            for info in self._strategies:
                try:
                    paper = self.order_manager._get_exchange(info.name)
                    if paper:
                        for coin in info.coins:
                            paper.cancel_all(coin)
                            pos = paper._positions.get(coin)
                            if pos and pos.size != 0:
                                mid = self._mid_prices.get(coin, 0)
                                if mid > 0:
                                    if pos.size > 0:
                                        pnl = (mid - pos.entry_px) * abs(pos.size)
                                    else:
                                        pnl = (pos.entry_px - mid) * abs(pos.size)
                                    paper.balance += pnl
                                    pos.realized_pnl += pnl
                                pos.size = 0.0
                                pos.entry_px = 0.0
                                pos.unrealized_pnl = 0.0
                                log.critical("Emergency closed paper position: %s", coin)
                except Exception:
                    log.exception(f"Failed to emergency close paper positions for {info.name}")

        # 4. Close live positions
        if self.rest and self.config and not self.config.mode.paper_trading:
            try:
                wallet = self.config.wallet_address
                if wallet:
                    open_orders = self.rest.get_open_orders(wallet)
                    if open_orders:
                        cancels = [(o.asset, o.oid) for o in open_orders]
                        self.rest.cancel_orders(cancels, self.config.exchange.vault_address)
                        log.critical("Cancelled %d open orders", len(cancels))

                    account = self.rest.get_account(wallet)
                    for pos in account.positions:
                        sz = pos.size.to_float()
                        if abs(sz) == 0:
                            continue
                        side = Side.SELL if sz > 0 else Side.BUY
                        mid = self._mid_prices.get(pos.coin, pos.entry_px.to_float())
                        slippage = 0.01
                        close_px = mid * (1 - slippage) if side == Side.SELL else mid * (1 + slippage)
                        req = OrderRequest(
                            asset=pos.asset,
                            coin=pos.coin,
                            side=side,
                            price=Decimal.from_float(close_px),
                            size=Decimal.from_float(abs(sz)),
                            reduce_only=True,
                            tif=TIF.IOC,
                        )
                        self.rest.place_order(req, self.config.exchange.vault_address)
                        log.critical("Emergency closed live position: %s (%.4f)", pos.coin, sz)
            except Exception:
                log.exception("Failed to emergency close live positions")

        self._save_engine_state()
        log.critical("Emergency close complete — all strategies suspended")

    def _refresh_asset_ctxs(self) -> None:
        """Refresh asset contexts (volumes, funding, etc.) from REST API."""
        try:
            if self.rest:
                ctxs = self.rest.get_asset_ctxs()
                self._asset_ctxs = {c.coin: c for c in ctxs}
        except Exception:
            log.debug("Failed to refresh asset contexts")

    async def _timer_loop(self) -> None:
        while self._running:
            await asyncio.sleep(60)

            # Emergency close check
            await self._check_emergency_close()

            # Refresh asset contexts every cycle (for volume data)
            self._refresh_asset_ctxs()

            for info in self._strategies:
                if info.errored or info.disabled or not info.instance:
                    continue
                try:
                    info.instance.on_timer()
                except Exception:
                    log.exception(f"Strategy {info.name} error in on_timer — suspended")
                    info.errored = True

    async def _reload_loop(self) -> None:
        interval = self.config.strategies.reload_interval_sec if self.config else 5
        while self._running:
            await asyncio.sleep(interval)
            if self.loader:
                reloaded = self.loader.check_reload()
                for name in reloaded:
                    log.info(f"Strategy reloaded: {name}")
