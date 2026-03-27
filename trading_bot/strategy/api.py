import dataclasses
import logging
import time
from collections import OrderedDict

from trading_bot.types import (
    Book, Candle, Decimal, Fill, Order, OrderRequest, OrderType, Position,
    Side, TIF, TPSL, Grouping,
)
from trading_bot.strategy.indicators import Indicators, compute_indicators
from trading_bot.config import BotConfig
from trading_bot.db import Database

log = logging.getLogger(__name__)

MAX_CACHE_SLOTS = 8
CACHE_TTL = 30.0
MAX_CANDLES = 300


class CandleCache:
    def __init__(self):
        self._cache: OrderedDict[tuple[str, str], tuple[float, list[Candle]]] = OrderedDict()

    def get(self, coin: str, interval: str) -> list[Candle] | None:
        key = (coin, interval)
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, candles = entry
        if time.time() - ts > CACHE_TTL:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return candles

    def put(self, coin: str, interval: str, candles: list[Candle]) -> None:
        key = (coin, interval)
        self._cache[key] = (time.time(), candles[-MAX_CANDLES:])
        while len(self._cache) > MAX_CACHE_SLOTS:
            self._cache.popitem(last=False)


class StrategyAPI:
    def __init__(
        self,
        strategy_name: str,
        coin: str,
        config: BotConfig,
        db: Database,
        order_manager: "OrderManager | None" = None,
        rest_client: "RestClient | None" = None,
        mid_prices: dict[str, float] | None = None,
        asset_ctxs: dict[str, "AssetCtx"] | None = None,
        data_manager: "DataManager | None" = None,
    ):
        self.strategy_name = strategy_name
        self.coin = coin
        self.config = config
        self._db = db
        self._order_manager = order_manager
        self._rest = rest_client
        self._mid_prices = mid_prices or {}
        self._asset_ctxs = asset_ctxs or {}
        self._data_manager = data_manager
        self._candle_cache = CandleCache()
        self._bt_time: float | None = None
        self._bt_candles: list[Candle] | None = None
        self._bt_funding_rates: list[tuple[int, float]] | None = None

    # --- Time ---

    def time(self) -> float:
        if self._bt_time is not None:
            return self._bt_time
        return time.time()

    # --- Orders ---

    def place_limit(
        self, coin: str, side: str, price: float, size: float,
        tif: str = "alo", reduce_only: bool = False,
    ) -> int:
        if not self._order_manager:
            return 0
        req = OrderRequest(
            asset=0,
            coin=coin,
            side=Side.BUY if side.lower() == "buy" else Side.SELL,
            price=Decimal.from_float(price),
            size=Decimal.from_float(size),
            reduce_only=reduce_only,
            order_type=OrderType.LIMIT,
            tif={"gtc": TIF.GTC, "ioc": TIF.IOC, "alo": TIF.ALO}.get(tif.lower(), TIF.ALO),
        )
        return self._order_manager.place_order(self.strategy_name, req)

    def place_trigger(
        self, coin: str, side: str, price: float, size: float,
        trigger_px: float, tpsl: str = "sl",
    ) -> int:
        if not self._order_manager:
            return 0
        req = OrderRequest(
            asset=0,
            coin=coin,
            side=Side.BUY if side.lower() == "buy" else Side.SELL,
            price=Decimal.from_float(price),
            size=Decimal.from_float(size),
            reduce_only=True,
            order_type=OrderType.TRIGGER,
            trigger_px=Decimal.from_float(trigger_px),
            tpsl=TPSL.TP if tpsl.lower() == "tp" else TPSL.SL,
            is_market=True,
            tif=TIF.GTC,
        )
        return self._order_manager.place_order(self.strategy_name, req)

    def cancel(self, coin: str, oid: int) -> bool:
        if not self._order_manager:
            return False
        return self._order_manager.cancel_order(self.strategy_name, coin, oid)

    def cancel_all(self, coin: str) -> int:
        if not self._order_manager:
            return 0
        return self._order_manager.cancel_all(self.strategy_name, coin)

    def set_leverage(self, coin: str, leverage: int) -> None:
        if self._order_manager:
            self._order_manager.set_leverage(self.strategy_name, coin, leverage)

    # --- Position ---

    def get_position(self, coin: str) -> Position | None:
        if not self._order_manager:
            return None
        return self._order_manager.get_position(self.strategy_name, coin)

    def get_mid_price(self, coin: str) -> float | None:
        return self._mid_prices.get(coin)

    def get_open_orders(self, coin: str) -> list[Order]:
        if not self._order_manager:
            return []
        return self._order_manager.get_open_orders(self.strategy_name, coin)

    # --- Candles & Indicators ---

    def get_candles(
        self, coin: str, interval: str, count: int,
        live_price: float | None = None,
    ) -> list[Candle]:
        if self._bt_candles is not None:
            candles = self._bt_candles[-count:]
            if live_price is not None and candles:
                last = dataclasses.replace(candles[-1], close=live_price)
                candles = candles[:-1] + [last]
            return candles

        cached = self._candle_cache.get(coin, interval)
        if cached:
            candles = cached[-count:]
        else:
            candles = self._fetch_candles(coin, interval, count)
            if candles:
                self._candle_cache.put(coin, interval, candles)
            candles = candles[-count:]

        if live_price is not None and candles:
            last = dataclasses.replace(candles[-1], close=live_price)
            candles = candles[:-1] + [last]

        return candles

    def _fetch_candles(self, coin: str, interval: str, count: int) -> list[Candle]:
        if self._rest:
            now_ms = int(time.time() * 1000)
            interval_ms = self._interval_to_ms(interval)
            start_ms = now_ms - interval_ms * count
            try:
                return self._rest.get_candles(coin, interval, start_ms, now_ms)
            except Exception:
                log.exception(f"Failed to fetch candles for {coin}/{interval}")

        rows = self._db.get_candles(coin, interval, count)
        return [
            Candle(
                time_open=r["time_open"], time_close=r["time_open"],
                open=r["open"], high=r["high"], low=r["low"],
                close=r["close"], volume=r["volume"],
                n_trades=r["n_trades"] or 0,
            )
            for r in rows
        ]

    def get_indicators(
        self, coin: str, interval: str, count: int = 200,
        live_price: float | None = None,
    ) -> Indicators | None:
        candles = self.get_candles(coin, interval, max(count, 200), live_price)
        if not candles:
            return None
        funding = self._get_current_funding(coin)
        return compute_indicators(candles, funding_rate=funding)

    # --- Account ---

    def get_account_value(self) -> float:
        if not self._order_manager:
            return 0.0
        return self._order_manager.get_account_value(self.strategy_name)

    def get_daily_pnl(self) -> float:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        day_start_ms = int(midnight.timestamp() * 1000)
        return self._db.get_daily_pnl(self.strategy_name, day_start_ms)

    # --- State persistence ---

    def save_state(self, key: str, value: str) -> None:
        self._db.save_state(self.strategy_name, key, value)

    def load_state(self, key: str) -> str | None:
        return self._db.load_state(self.strategy_name, key)

    # --- Logging ---

    def log(self, level: int, msg: str) -> None:
        logging.getLogger(f"strategy.{self.strategy_name}").log(level, msg)

    # --- Funding & Sentiment ---

    def get_funding_rate(self, coin: str) -> float | None:
        if self._bt_funding_rates is not None:
            return self._get_bt_funding(coin)
        ctx = self._asset_ctxs.get(coin)
        if ctx:
            return ctx.funding_rate
        return None

    def get_open_interest(self, coin: str) -> float | None:
        if self._bt_time is not None:
            return None
        ctx = self._asset_ctxs.get(coin)
        return ctx.open_interest if ctx else None

    def get_sentiment(self, coin: str) -> float:
        if not self._data_manager:
            return 0.0
        return self._data_manager.get_sentiment(coin)

    def get_fear_greed(self) -> float:
        if not self._data_manager:
            return 0.0
        return self._data_manager.get_fear_greed()

    # --- Kelly sizing ---

    def kelly_size(
        self, win_rate: float, avg_win_pct: float,
        avg_loss_pct: float, fraction: float = 0.5,
    ) -> float:
        if avg_loss_pct <= 0:
            return 0.0
        b = avg_win_pct / avg_loss_pct
        q = 1.0 - win_rate
        f_star = (win_rate * b - q) / b
        if f_star <= 0:
            return 0.0
        return max(0.05, min(0.50, f_star * fraction))

    # --- Internal helpers ---

    def _get_current_funding(self, coin: str) -> float:
        if self._bt_funding_rates is not None:
            rate = self._get_bt_funding(coin)
            return rate if rate is not None else 0.0
        ctx = self._asset_ctxs.get(coin)
        return ctx.funding_rate if ctx else 0.0

    def _get_bt_funding(self, coin: str) -> float | None:
        if not self._bt_funding_rates or self._bt_time is None:
            return None
        target_ms = int(self._bt_time * 1000)
        rates = self._bt_funding_rates
        lo, hi = 0, len(rates) - 1
        result = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if rates[mid][0] <= target_ms:
                result = rates[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        unit = interval[-1]
        val = int(interval[:-1])
        multipliers = {"m": 60000, "h": 3600000, "d": 86400000}
        return val * multipliers.get(unit, 3600000)
