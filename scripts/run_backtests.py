"""Run comprehensive backtests of all strategies over multiple time periods."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from trading_bot.db import Database
from trading_bot.backtest.engine import BacktestConfig, BacktestEngine
from trading_bot.strategy.loader import StrategyLoader
from trading_bot.types import Candle
from pathlib import Path

DB_PATH = "./data/trading_bot.db"
STRATEGIES_DIR = "./trading_bot/strategies"
INITIAL_BALANCE = 100.0
MAX_LEVERAGE = 50

STRATEGIES = [
    {"file": "btc_sniper_1h.py", "coin": "BTC"},
    {"file": "sol_range_breakout_1h.py", "coin": "SOL"},
]

# Time periods to test (from today backwards)
now = datetime.now(timezone.utc)
PERIODS = [
    ("1 month", now - timedelta(days=30), now),
    ("3 months", now - timedelta(days=90), now),
    ("6 months", now - timedelta(days=180), now),
    ("Full data", None, None),
]


def load_candles(db, coin, start_ms=0, end_ms=0):
    query = "SELECT * FROM candles WHERE coin=? AND interval='5m'"
    params = [coin]
    if start_ms > 0:
        query += " AND time_open >= ?"
        params.append(start_ms)
    if end_ms > 0:
        query += " AND time_open <= ?"
        params.append(end_ms)
    query += " ORDER BY time_open"
    rows = db.fetchall(query, tuple(params))
    return [
        Candle(
            time_open=r["time_open"],
            time_close=r["time_open"] + 300000,
            open=r["open"], high=r["high"],
            low=r["low"], close=r["close"],
            volume=r["volume"],
            n_trades=r["n_trades"] or 0,
        )
        for r in rows
    ]


def run_single(db, strategy_file, coin, start_ms=0, end_ms=0):
    candles = load_candles(db, coin, start_ms, end_ms)
    if len(candles) < 50:
        return None, len(candles)

    strategy_path = str(Path(STRATEGIES_DIR).resolve() / strategy_file)
    loader = StrategyLoader(STRATEGIES_DIR)
    instance = loader._load_module(strategy_path, Path(strategy_file).stem)

    config = BacktestConfig(
        coin=coin,
        strategy_path=strategy_path,
        initial_balance=INITIAL_BALANCE,
        max_leverage=MAX_LEVERAGE,
        strategy_interval_ms=3_600_000,
    )

    engine = BacktestEngine(config, db)
    result = engine.run(instance, candles)
    return result, len(candles)


def fmt_pf(v):
    if v == float("inf") or v >= 999:
        return "inf"
    return f"{v:.2f}"


def main():
    db = Database(DB_PATH)
    db.open()

    # Check available data
    print("=" * 80)
    print("DATA AVAILABILITY")
    print("=" * 80)
    for s in STRATEGIES:
        coin = s["coin"]
        row = db.fetchone(
            "SELECT COUNT(*) as cnt, MIN(time_open) as min_t, MAX(time_open) as max_t "
            "FROM candles WHERE coin=? AND interval='5m'",
            (coin,),
        )
        if row and row["cnt"] > 0:
            min_d = datetime.fromtimestamp(row["min_t"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            max_d = datetime.fromtimestamp(row["max_t"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"  {coin:6s}  {row['cnt']:>8,} candles  |  {min_d} -> {max_d}")
        else:
            print(f"  {coin:6s}  NO DATA")

    print()
    print("=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    header = f"{'Strategy':<28s} {'Period':<12s} {'Return%':>8s} {'Sharpe':>7s} {'MaxDD%':>7s} {'WinR%':>6s} {'Trades':>7s} {'PF':>6s} {'Verdict':<14s}"
    print(header)
    print("-" * len(header))

    for s in STRATEGIES:
        strategy_file = s["file"]
        coin = s["coin"]
        name = f"{strategy_file.replace('.py', '')} ({coin})"

        for period_name, start_dt, end_dt in PERIODS:
            start_ms = int(start_dt.timestamp() * 1000) if start_dt else 0
            end_ms = int(end_dt.timestamp() * 1000) if end_dt else 0

            try:
                result, n_candles = run_single(db, strategy_file, coin, start_ms, end_ms)
            except Exception as e:
                print(f"{name:<28s} {period_name:<12s}  ERROR: {e}")
                continue

            if result is None:
                print(f"{name:<28s} {period_name:<12s}  Insufficient data ({n_candles} candles)")
                continue

            r = result
            wr = r.win_rate * 100
            pf = fmt_pf(r.profit_factor)

            # Verdict
            if r.return_pct > 10 and r.sharpe_ratio > 1.0 and r.max_drawdown_pct < 15 and wr > 40 and r.profit_factor > 1.5:
                verdict = "DEPLOYABLE"
            elif r.return_pct > 5 and r.sharpe_ratio > 0.5 and r.profit_factor > 1.0:
                verdict = "A_OPTIMISER"
            elif r.return_pct > 0 and r.profit_factor > 0.8:
                verdict = "MARGINAL"
            elif r.return_pct > -5 or r.total_trades < 10:
                verdict = "INSUFFISANT"
            else:
                verdict = "ABANDON"

            print(
                f"{name:<28s} {period_name:<12s} "
                f"{r.return_pct:>7.2f}% "
                f"{r.sharpe_ratio:>7.2f} "
                f"{r.max_drawdown_pct:>6.2f}% "
                f"{wr:>5.1f}% "
                f"{r.total_trades:>7d} "
                f"{pf:>6s} "
                f"{verdict:<14s}"
            )

        print()  # Blank line between strategies

    db.close()
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
