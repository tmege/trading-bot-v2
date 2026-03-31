import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone

from trading_bot.types import OrderRequest
from trading_bot.config import RiskConfig

log = logging.getLogger(__name__)

CIRCUIT_BREAKER_WINDOW = 900
CIRCUIT_BREAKER_MOVE = 0.07
CIRCUIT_BREAKER_MAX_MOVE = 5.0  # >500% = data anomaly, ignore
MAX_TRACKED_COINS = 200


class RiskManager:
    def __init__(self, config: RiskConfig):
        self._config = config
        self._lock = threading.RLock()
        self._paused = False
        self._daily_pnl: dict[str, float] = {}
        self._daily_fees: dict[str, float] = {}
        self._daily_trades: dict[str, int] = {}
        self._last_reset_day = -1
        self._price_history: dict[str, deque] = {}
        self._circuit_breaker_coins: set[str] = set()
        self._cb_last_log: dict[str, float] = {}

        self._positions_notional: dict[str, float] = {}

    @property
    def circuit_breaker_active(self) -> bool:
        return bool(self._circuit_breaker_coins)

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        with self._lock:
            self._paused = True
        log.warning("Risk manager: PAUSED")

    def unpause(self) -> None:
        with self._lock:
            self._paused = False
        log.info("Risk manager: UNPAUSED")

    def check_order(
        self,
        req: OrderRequest,
        account_value: float,
        strategy: str,
    ) -> tuple[bool, str]:
        with self._lock:
            self._maybe_reset_daily()

            if self._paused:
                return False, "trading paused"

            if self._config.daily_loss_pct > 0:
                daily_pnl = self._daily_pnl.get(strategy, 0.0)
                max_loss = account_value * self._config.daily_loss_pct / 100.0
                if abs(daily_pnl) > max_loss and daily_pnl < 0:
                    return False, f"daily loss limit {daily_pnl:.2f} > {max_loss:.2f} [{strategy}]"

            notional = req.price.to_float() * req.size.to_float()

            if account_value > 0:
                order_leverage = notional / account_value
                if order_leverage > self._config.max_leverage:
                    return False, f"leverage {order_leverage:.1f}x > max {self._config.max_leverage}x"

            if account_value > 0:
                max_pos = account_value * self._config.max_position_pct / 100.0
                if notional > max_pos:
                    return False, f"position {notional:.2f} > max {max_pos:.2f}"

            if req.coin in self._circuit_breaker_coins and not req.reduce_only:
                return False, f"circuit breaker active for {req.coin}"

            return True, ""

    def check_emergency_close(self, account_value: float, daily_pnl: float) -> bool:
        if self._config.emergency_close_pct <= 0:
            return False
        threshold = account_value * self._config.emergency_close_pct / 100.0
        return daily_pnl < -threshold

    def update_price(self, coin: str, price: float) -> None:
        now = time.time()
        if coin not in self._price_history:
            if len(self._price_history) >= MAX_TRACKED_COINS:
                return
            self._price_history[coin] = deque()

        history = self._price_history[coin]
        history.append((now, price))

        while history and now - history[0][0] > CIRCUIT_BREAKER_WINDOW:
            history.popleft()

        if len(history) >= 2:
            oldest_price = history[0][1]
            if oldest_price > 0:
                move = abs(price - oldest_price) / oldest_price
                if move > CIRCUIT_BREAKER_MAX_MOVE:
                    # >500% in 15min = data anomaly, ignore silently
                    return
                if move > CIRCUIT_BREAKER_MOVE:
                    if coin not in self._circuit_breaker_coins or now - self._cb_last_log.get(coin, 0) > 300:
                        log.warning(
                            f"Circuit breaker triggered: {coin} moved {move*100:.1f}% in 15min"
                        )
                        self._cb_last_log[coin] = now
                    self._circuit_breaker_coins.add(coin)
                else:
                    self._circuit_breaker_coins.discard(coin)

    def update_position_notional(self, coin: str, notional: float) -> None:
        self._positions_notional[coin] = notional

    def get_total_exposure(self) -> float:
        return sum(abs(v) for v in self._positions_notional.values())

    def record_trade(self, strategy: str, pnl: float, fee: float) -> None:
        with self._lock:
            self._daily_pnl[strategy] = self._daily_pnl.get(strategy, 0.0) + pnl
            self._daily_fees[strategy] = self._daily_fees.get(strategy, 0.0) + fee
            self._daily_trades[strategy] = self._daily_trades.get(strategy, 0) + 1

    def get_strategy_daily_pnl(self, strategy: str) -> float:
        return self._daily_pnl.get(strategy, 0.0)

    def get_total_daily_pnl(self) -> float:
        return sum(self._daily_pnl.values())

    def get_total_daily_fees(self) -> float:
        return sum(self._daily_fees.values())

    def get_total_daily_trades(self) -> int:
        return sum(self._daily_trades.values())

    def _maybe_reset_daily(self) -> None:
        now = datetime.now(timezone.utc)
        day = now.timetuple().tm_yday
        if day != self._last_reset_day:
            self._daily_pnl.clear()
            self._daily_fees.clear()
            self._daily_trades.clear()
            if self._paused:
                self._paused = False
                log.warning("Risk manager: auto-UNPAUSED on daily reset")
            self._last_reset_day = day
            log.info("Daily risk counters reset")

    def to_dict(self) -> dict:
        return {
            "daily_pnl": dict(self._daily_pnl),
            "daily_fees": dict(self._daily_fees),
            "daily_trades": dict(self._daily_trades),
            "last_reset_day": self._last_reset_day,
            "circuit_breaker_coins": sorted(self._circuit_breaker_coins),
            "paused": self._paused,
        }

    def from_dict(self, d: dict) -> None:
        if not isinstance(d, dict):
            log.warning("Risk manager from_dict: expected dict, got %s — ignored", type(d).__name__)
            return
        raw_pnl = d.get("daily_pnl", {})
        raw_fees = d.get("daily_fees", {})
        raw_trades = d.get("daily_trades", {})

        # Backward compat: old format stored scalar values
        if isinstance(raw_pnl, dict):
            self._daily_pnl = raw_pnl
        else:
            self._daily_pnl = {"__legacy__": float(raw_pnl)} if raw_pnl else {}

        if isinstance(raw_fees, dict):
            self._daily_fees = raw_fees
        else:
            self._daily_fees = {"__legacy__": float(raw_fees)} if raw_fees else {}

        if isinstance(raw_trades, dict):
            self._daily_trades = raw_trades
        else:
            self._daily_trades = {"__legacy__": int(raw_trades)} if raw_trades else {}

        self._last_reset_day = d.get("last_reset_day", -1)
        cb_coins = d.get("circuit_breaker_coins")
        if cb_coins is not None:
            self._circuit_breaker_coins = set(cb_coins)
        else:
            self._circuit_breaker_coins = set()
        self._paused = d.get("paused", False)
        total_pnl = self.get_total_daily_pnl()
        total_fees = self.get_total_daily_fees()
        total_trades = self.get_total_daily_trades()
        log.info(f"Risk manager state restored: pnl={total_pnl:.4f}, fees={total_fees:.4f}, trades={total_trades}")
