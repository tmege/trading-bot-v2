import logging
from dataclasses import dataclass, field

from trading_bot.types import (
    Candle, Decimal, Fill, OrderRequest, OrderType, Position, Side, TPSL, TIF,
)
from trading_bot.exchange.paper_exchange import PaperExchange, PaperOrder
from trading_bot.strategy.api import StrategyAPI
from trading_bot.strategy.loader import StrategyLoader
from trading_bot.db import Database
from trading_bot.config import BotConfig

log = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    coin: str = "BTC"
    strategy_path: str = ""
    initial_balance: float = 100.0
    max_leverage: int = 10
    maker_fee: float = 0.00015
    taker_fee: float = 0.00045
    slippage_bps: float = 1.0
    strategy_interval_ms: int = 3_600_000
    grid_tp: float = 0.0
    grid_sl: float = 0.0


@dataclass
class BacktestTrade:
    time_ms: int = 0
    side: str = ""
    price: float = 0.0
    size: float = 0.0
    pnl: float = 0.0
    fee: float = 0.0
    balance_after: float = 0.0


@dataclass
class BacktestResult:
    start_balance: float = 0.0
    end_balance: float = 0.0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    monte_carlo: dict = field(default_factory=dict)


class BacktestEngine:
    def __init__(self, bt_config: BacktestConfig, db: Database):
        self.bt_config = bt_config
        self._db = db
        self._exchange = PaperExchange("backtest", bt_config.initial_balance)
        self._exchange._maker_fee = bt_config.maker_fee
        self._exchange._taker_fee = bt_config.taker_fee

        self._trades: list[BacktestTrade] = []
        self._equity_curve: list[dict] = []
        self._current_idx = 0
        self._max_dd = 0.0

    def run(self, strategy_instance, candles_5m: list[Candle]) -> BacktestResult:
        if not candles_5m:
            return BacktestResult(start_balance=self.bt_config.initial_balance)

        interval_ms = self.bt_config.strategy_interval_ms
        bars_per_tf = interval_ms // 300_000

        bt_config = self.bt_config
        config = BotConfig()
        config.risk.max_leverage = bt_config.max_leverage

        api = StrategyAPI(
            strategy_name="backtest",
            coin=bt_config.coin,
            config=config,
            db=self._db,
        )

        funding_rates = self._load_funding_rates(bt_config.coin)
        api._bt_funding_rates = funding_rates

        self._exchange.on_fill = lambda fill: self._on_fill(fill, strategy_instance)

        strategy_instance.coin = bt_config.coin
        strategy_instance.errored = False

        if hasattr(strategy_instance, "grid_tp") and bt_config.grid_tp > 0:
            strategy_instance.grid_tp = bt_config.grid_tp
        if hasattr(strategy_instance, "grid_sl") and bt_config.grid_sl > 0:
            strategy_instance.grid_sl = bt_config.grid_sl

        # Wrap API for backtest
        api._bt_time = candles_5m[0].time_open / 1000.0
        api._order_manager = _BacktestOrderManager(self._exchange, self._db)

        strategy_instance.on_init(api)

        tf_buffer: list[Candle] = []
        tf_history: list[Candle] = []
        peak_equity = bt_config.initial_balance
        max_dd = 0.0
        coin = bt_config.coin

        # Local references for hot loop
        exchange = self._exchange
        orders = exchange._orders
        check_limit = self._check_fills_limit
        check_trigger = self._check_fills_trigger

        for idx, candle in enumerate(candles_5m):
            self._current_idx = idx
            api._bt_time = candle.time_open / 1000.0

            # Pass 1 & 2: check fills (limit then trigger)
            if orders:
                check_limit(candle, idx)
                check_trigger(candle, idx)

            # Pass 3: TF aggregation + on_tick
            tf_buffer.append(candle)
            if len(tf_buffer) >= bars_per_tf:
                tf_candle = _aggregate_tf_fast(tf_buffer)
                tf_buffer = []

                # Incremental: just append the new TF candle
                tf_history.append(tf_candle)
                api._bt_candles = tf_history

                mid = tf_candle.close
                exchange._update_unrealized(coin, mid)

                try:
                    strategy_instance.on_tick(coin, mid)
                except Exception:
                    log.exception("Backtest strategy error in on_tick")
                    break

                # Set placed_at_idx for new orders
                for order in exchange._orders:
                    if order.placed_at_idx < 0:
                        order.placed_at_idx = idx

                # Track equity & DD only at TF boundaries
                # (unrealized PnL is only fresh here anyway)
                equity = exchange.equity
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
                self._equity_curve.append({
                    "time_ms": candle.time_open,
                    "equity": equity,
                    "drawdown": dd,
                })

            # Re-bind in case orders list was replaced by fill checking
            orders = exchange._orders

        self._max_dd = max_dd
        return self._compile_result()

    def _check_fills_limit(self, candle: Candle, idx: int) -> None:
        to_remove: set[int] = set()
        for order in self._exchange._orders:
            if order.order_type != OrderType.LIMIT:
                continue
            if order.placed_at_idx == idx or order.placed_at_idx < 0:
                continue
            if order.coin != self.bt_config.coin:
                continue

            filled = False
            if order.side == Side.BUY and candle.low <= order.price:
                filled = True
            elif order.side == Side.SELL and candle.high >= order.price:
                filled = True

            if filled:
                fill = self._exchange._execute_fill(order, order.price)
                to_remove.add(order.oid)
                self._record_trade(fill, candle.time_open)

        if to_remove:
            self._exchange._orders = [
                o for o in self._exchange._orders if o.oid not in to_remove
            ]

    def _check_fills_trigger(self, candle: Candle, idx: int) -> None:
        to_remove: set[int] = set()
        for order in self._exchange._orders:
            if order.order_type != OrderType.TRIGGER:
                continue
            if order.placed_at_idx == idx or order.placed_at_idx < 0:
                continue
            if order.coin != self.bt_config.coin:
                continue

            pos = self._exchange._positions.get(order.coin)
            if not pos:
                continue

            filled = False
            fill_price = order.trigger_px

            if order.tpsl == TPSL.TP:
                if pos.size > 0 and candle.high >= order.trigger_px:
                    filled = True
                elif pos.size < 0 and candle.low <= order.trigger_px:
                    filled = True
            elif order.tpsl == TPSL.SL:
                if pos.size > 0 and candle.low <= order.trigger_px:
                    filled = True
                elif pos.size < 0 and candle.high >= order.trigger_px:
                    filled = True

            if filled:
                slippage = fill_price * self.bt_config.slippage_bps / 10000
                if (order.tpsl == TPSL.SL and pos.size > 0) or (order.tpsl == TPSL.TP and pos.size < 0):
                    fill_price -= slippage
                else:
                    fill_price += slippage

                fill = self._exchange._execute_fill(order, fill_price)
                to_remove.add(order.oid)
                self._record_trade(fill, candle.time_open)

        if to_remove:
            self._exchange._orders = [
                o for o in self._exchange._orders if o.oid not in to_remove
            ]

    def _record_trade(self, fill: Fill, time_ms: int) -> None:
        pnl = fill.closed_pnl.to_float()
        fee = fill.fee.to_float()
        self._trades.append(BacktestTrade(
            time_ms=time_ms,
            side=fill.side.value,
            price=fill.px.to_float(),
            size=fill.sz.to_float(),
            pnl=pnl,
            fee=fee,
            balance_after=self._exchange.balance,
        ))

    def _on_fill(self, fill: Fill, strategy) -> None:
        try:
            strategy.on_fill(fill)
        except Exception:
            log.exception("Backtest strategy error in on_fill")

    def _load_funding_rates(self, coin: str) -> list[tuple[int, float]]:
        rows = self._db.fetchall(
            "SELECT time_ms, rate FROM funding_rates WHERE coin=? ORDER BY time_ms",
            (coin,)
        )
        return [(r["time_ms"], r["rate"]) for r in rows]

    def _compile_result(self) -> BacktestResult:
        cfg = self.bt_config
        result = BacktestResult(
            start_balance=cfg.initial_balance,
            end_balance=self._exchange.balance,
            trades=self._trades,
            equity_curve=self._equity_curve,
        )

        exit_trades = [t for t in self._trades if t.pnl != 0]
        result.total_trades = len(exit_trades)
        result.total_fees = sum(t.fee for t in self._trades)
        result.total_pnl = result.end_balance - cfg.initial_balance
        result.return_pct = result.total_pnl / cfg.initial_balance * 100 if cfg.initial_balance > 0 else 0

        wins = [t for t in exit_trades if t.pnl > 0]
        losses = [t for t in exit_trades if t.pnl <= 0]
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(exit_trades) if exit_trades else 0

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        result.avg_win = gross_profit / len(wins) if wins else 0
        result.avg_loss = gross_loss / len(losses) if losses else 0
        result.max_win = max((t.pnl for t in wins), default=0)
        result.max_loss = min((t.pnl for t in losses), default=0)

        # Use pre-computed max DD instead of iterating equity curve
        result.max_drawdown_pct = self._max_dd * 100

        # Sharpe & Sortino
        if len(exit_trades) >= 2:
            returns = [t.pnl / cfg.initial_balance for t in exit_trades]
            mean_r = sum(returns) / len(returns)
            var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            std = var ** 0.5

            result.sharpe_ratio = (252 ** 0.5) * mean_r / std if std > 0 else 0

            downside = [r for r in returns if r < 0]
            if downside:
                down_var = sum(r ** 2 for r in downside) / len(downside)
                down_std = down_var ** 0.5
                result.sortino_ratio = (252 ** 0.5) * mean_r / down_std if down_std > 0 else 0

        return result


def _aggregate_tf_fast(buffer: list[Candle]) -> Candle:
    """Aggregate 5m candles into a TF candle — standalone function to avoid method overhead."""
    hi = buffer[0].high
    lo = buffer[0].low
    vol = 0.0
    trades = 0
    for c in buffer:
        if c.high > hi:
            hi = c.high
        if c.low < lo:
            lo = c.low
        vol += c.volume
        trades += c.n_trades
    return Candle(
        time_open=buffer[0].time_open,
        time_close=buffer[-1].time_close,
        open=buffer[0].open,
        high=hi,
        low=lo,
        close=buffer[-1].close,
        volume=vol,
        n_trades=trades,
    )


class _BacktestOrderManager:
    def __init__(self, exchange: PaperExchange, db: Database):
        self._exchange = exchange
        self._db = db

    def place_order(self, strategy: str, req: OrderRequest) -> int:
        return self._exchange.place_order(req)

    def cancel_order(self, strategy: str, coin: str, oid: int) -> bool:
        return self._exchange.cancel_order(oid)

    def cancel_all(self, strategy: str, coin: str) -> int:
        return self._exchange.cancel_all(coin)

    def get_position(self, strategy: str, coin: str) -> Position | None:
        return self._exchange.get_position(coin)

    def get_open_orders(self, strategy: str, coin: str) -> list:
        return self._exchange.get_open_orders(coin)

    def get_account_value(self, strategy: str) -> float:
        return self._exchange.get_account_value()
