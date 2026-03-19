import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from trading_bot.backtest.engine import BacktestConfig, BacktestEngine
from trading_bot.backtest.monte_carlo import run_monte_carlo
from trading_bot.db import Database
from trading_bot.strategy.loader import StrategyLoader
from trading_bot.types import Candle

log = logging.getLogger(__name__)


def run_backtest(
    coin: str,
    strategy_path: str,
    db_path: str = "./data/trading_bot.db",
    initial_balance: float = 100.0,
    max_leverage: int = 10,
    grid_tp: float = 0.0,
    grid_sl: float = 0.0,
    interval_ms: int = 3_600_000,
    output_path: str | None = None,
) -> dict:
    db = Database(db_path)
    db.open()

    try:
        candles_5m = _load_candles(db, coin)
        if not candles_5m:
            log.error(f"No 5m candles for {coin}")
            return {"error": "no candles"}

        log.info(f"Loaded {len(candles_5m)} 5m candles for {coin}")

        bt_config = BacktestConfig(
            coin=coin,
            strategy_path=strategy_path,
            initial_balance=initial_balance,
            max_leverage=max_leverage,
            grid_tp=grid_tp,
            grid_sl=grid_sl,
            strategy_interval_ms=interval_ms,
        )

        strategy_dir = str(Path(strategy_path).parent)
        strategy_file = Path(strategy_path).name
        loader = StrategyLoader(strategy_dir)

        # Load strategy directly
        instance = loader._load_module(str(Path(strategy_path).resolve()), strategy_file.replace(".py", ""))

        engine = BacktestEngine(bt_config, db)
        result = engine.run(instance, candles_5m)

        # Monte Carlo
        exit_pnls = [t.pnl for t in result.trades if t.pnl != 0]
        if len(exit_pnls) >= 5:
            mc = run_monte_carlo(exit_pnls, initial_balance)
            result.monte_carlo = mc.to_dict()

        output = _result_to_dict(result)

        if output_path:
            resolved = Path(output_path).resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved, "w") as f:
                json.dump(output, f, indent=2)
            log.info(f"Results saved to {resolved}")

        return output

    finally:
        db.close()


def _load_candles(db: Database, coin: str) -> list[Candle]:
    rows = db.fetchall(
        "SELECT * FROM candles WHERE coin=? AND interval='5m' ORDER BY time_open",
        (coin,)
    )
    return [
        Candle(
            time_open=r["time_open"],
            time_close=r["time_open"] + 300000,
            open=r["open"],
            high=r["high"],
            low=r["low"],
            close=r["close"],
            volume=r["volume"],
            n_trades=r["n_trades"] or 0,
        )
        for r in rows
    ]


def _result_to_dict(result) -> dict:
    return {
        "start_balance": result.start_balance,
        "end_balance": round(result.end_balance, 4),
        "total_pnl": round(result.total_pnl, 4),
        "total_fees": round(result.total_fees, 6),
        "return_pct": round(result.return_pct, 2),
        "total_trades": result.total_trades,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "win_rate": round(result.win_rate, 4),
        "profit_factor": round(result.profit_factor, 4) if result.profit_factor != float('inf') else "inf",
        "avg_win": round(result.avg_win, 6),
        "avg_loss": round(result.avg_loss, 6),
        "max_win": round(result.max_win, 6),
        "max_loss": round(result.max_loss, 6),
        "max_drawdown_pct": round(result.max_drawdown_pct, 2),
        "sharpe_ratio": round(result.sharpe_ratio, 4),
        "sortino_ratio": round(result.sortino_ratio, 4),
        "n_trades_detail": len(result.trades),
        "monte_carlo": result.monte_carlo if result.monte_carlo else None,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backtest runner")
    parser.add_argument("coin", help="Coin symbol (e.g., BTC)")
    parser.add_argument("strategy", help="Path to strategy file")
    parser.add_argument("--db", default="./data/trading_bot.db", help="DB path")
    parser.add_argument("--balance", type=float, default=100.0, help="Initial balance")
    parser.add_argument("--leverage", type=int, default=10, help="Max leverage")
    parser.add_argument("--tp", type=float, default=0.0, help="Grid TP override (%)")
    parser.add_argument("--sl", type=float, default=0.0, help="Grid SL override (%)")
    parser.add_argument("--interval", type=int, default=3600000, help="Strategy interval ms")
    parser.add_argument("--output", default=None, help="Output JSON path")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    result = run_backtest(
        coin=args.coin,
        strategy_path=args.strategy,
        db_path=args.db,
        initial_balance=args.balance,
        max_leverage=args.leverage,
        grid_tp=args.tp / 100.0 if args.tp > 0 else 0.0,
        grid_sl=args.sl / 100.0 if args.sl > 0 else 0.0,
        interval_ms=args.interval,
        output_path=args.output,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
