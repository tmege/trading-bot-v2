import logging
import time
from collections import deque
from datetime import datetime, timezone

from trading_bot.types import OrderRequest
from trading_bot.config import RiskConfig

log = logging.getLogger(__name__)

CIRCUIT_BREAKER_WINDOW = 900
CIRCUIT_BREAKER_MOVE = 0.07


class RiskManager:
    def __init__(self, config: RiskConfig):
        self._config = config
        self._paused = False
        self._daily_pnl = 0.0
        self._daily_fees = 0.0
        self._daily_trades = 0
        self._last_reset_day = -1
        self._price_history: dict[str, deque] = {}
        self._circuit_breaker_active = False

        self._positions_notional: dict[str, float] = {}

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True
        log.warning("Risk manager: PAUSED")

    def unpause(self) -> None:
        self._paused = False
        log.info("Risk manager: UNPAUSED")

    def check_order(
        self,
        req: OrderRequest,
        account_value: float,
        daily_pnl: float,
    ) -> tuple[bool, str]:
        self._maybe_reset_daily()

        if self._paused:
            return False, "trading paused"

        if self._config.daily_loss_pct > 0:
            max_loss = account_value * self._config.daily_loss_pct / 100.0
            if abs(daily_pnl) > max_loss and daily_pnl < 0:
                return False, f"daily loss limit {daily_pnl:.2f} > {max_loss:.2f}"

        notional = req.price.to_float() * req.size.to_float()

        if account_value > 0:
            order_leverage = notional / account_value
            if order_leverage > self._config.max_leverage:
                return False, f"leverage {order_leverage:.1f}x > max {self._config.max_leverage}x"

        if account_value > 0:
            max_pos = account_value * self._config.max_position_pct / 100.0
            if notional > max_pos:
                return False, f"position {notional:.2f} > max {max_pos:.2f}"

        if self._circuit_breaker_active and not req.reduce_only:
            return False, "circuit breaker active"

        return True, ""

    def check_emergency_close(self, account_value: float, daily_pnl: float) -> bool:
        if self._config.emergency_close_pct <= 0:
            return False
        threshold = account_value * self._config.emergency_close_pct / 100.0
        return daily_pnl < -threshold

    def update_price(self, coin: str, price: float) -> None:
        now = time.time()
        if coin not in self._price_history:
            self._price_history[coin] = deque()

        history = self._price_history[coin]
        history.append((now, price))

        while history and now - history[0][0] > CIRCUIT_BREAKER_WINDOW:
            history.popleft()

        if len(history) >= 2:
            oldest_price = history[0][1]
            if oldest_price > 0:
                move = abs(price - oldest_price) / oldest_price
                if move > CIRCUIT_BREAKER_MOVE:
                    if not self._circuit_breaker_active:
                        log.warning(
                            f"Circuit breaker triggered: {coin} moved {move*100:.1f}% in 15min"
                        )
                        self._circuit_breaker_active = True
                else:
                    self._circuit_breaker_active = False

    def update_position_notional(self, coin: str, notional: float) -> None:
        self._positions_notional[coin] = notional

    def get_total_exposure(self) -> float:
        return sum(abs(v) for v in self._positions_notional.values())

    def record_trade(self, pnl: float, fee: float) -> None:
        self._daily_pnl += pnl
        self._daily_fees += fee
        self._daily_trades += 1

    def _maybe_reset_daily(self) -> None:
        now = datetime.now(timezone.utc)
        day = now.timetuple().tm_yday
        if day != self._last_reset_day:
            self._daily_pnl = 0.0
            self._daily_fees = 0.0
            self._daily_trades = 0
            self._last_reset_day = day
            log.info("Daily risk counters reset")

    def to_dict(self) -> dict:
        return {
            "daily_pnl": self._daily_pnl,
            "daily_fees": self._daily_fees,
            "daily_trades": self._daily_trades,
            "last_reset_day": self._last_reset_day,
            "circuit_breaker_active": self._circuit_breaker_active,
            "paused": self._paused,
        }

    def from_dict(self, d: dict) -> None:
        self._daily_pnl = d.get("daily_pnl", 0.0)
        self._daily_fees = d.get("daily_fees", 0.0)
        self._daily_trades = d.get("daily_trades", 0)
        self._last_reset_day = d.get("last_reset_day", -1)
        self._circuit_breaker_active = d.get("circuit_breaker_active", False)
        self._paused = d.get("paused", False)
        log.info(f"Risk manager state restored: pnl={self._daily_pnl:.4f}, fees={self._daily_fees:.4f}, trades={self._daily_trades}")
