import json
import logging
import queue as queue_mod
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from trading_bot.backtest.engine import BacktestConfig, BacktestEngine
from trading_bot.backtest.monte_carlo import run_monte_carlo
from trading_bot.db import Database
from trading_bot.strategy.loader import StrategyLoader
from trading_bot.types import Candle

log = logging.getLogger(__name__)


@dataclass
class BacktestRun:
    run_id: str
    strategy: str
    coins: list[str]
    status: str = "pending"
    progress: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    queue: queue_mod.Queue = field(default_factory=queue_mod.Queue)
    error: str = ""


_runs: dict[str, BacktestRun] = {}


def get_available_coins(db: Database) -> list[dict]:
    if not db:
        return []
    try:
        rows = db.fetchall(
            "SELECT coin, COUNT(*) as cnt, MIN(time_open) as min_t, MAX(time_open) as max_t "
            "FROM candles WHERE interval='5m' GROUP BY coin ORDER BY coin",
        )
        result = []
        for r in rows:
            span_ms = r["max_t"] - r["min_t"]
            years = span_ms / (365.25 * 86400 * 1000)
            result.append({
                "coin": r["coin"],
                "candle_count_5m": r["cnt"],
                "date_range": f"{_ms_to_date(r['min_t'])} — {_ms_to_date(r['max_t'])}",
                "min_date": _ms_to_date(r["min_t"]),
                "max_date": _ms_to_date(r["max_t"]),
                "years_available": round(years, 2),
            })
        return result
    except Exception:
        log.exception("Error fetching available coins")
        return []


def start_run(
    strategy_file: str,
    coins: list[str],
    db_path: str,
    strategies_dir: str,
    initial_balance: float = 100.0,
    max_leverage: int = 50,
    interval_ms: int = 3_600_000,
    start_ms: int = 0,
    end_ms: int = 0,
) -> str:
    run_id = uuid.uuid4().hex[:12]
    run = BacktestRun(
        run_id=run_id,
        strategy=strategy_file,
        coins=coins,
        status="running",
    )
    _runs[run_id] = run

    t = threading.Thread(
        target=_run_backtest_thread,
        args=(run, db_path, strategies_dir, initial_balance, max_leverage, interval_ms, start_ms, end_ms),
        daemon=True,
    )
    t.start()
    return run_id


def get_run(run_id: str) -> BacktestRun | None:
    return _runs.get(run_id)


def get_history(db: Database, limit: int = 500) -> list[dict]:
    if not db:
        return []
    try:
        rows = db.fetchall(
            "SELECT * FROM backtest_history ORDER BY timestamp_ms DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]
    except Exception:
        log.exception("Error fetching backtest history")
        return []


def clear_history(db: Database) -> int:
    if not db:
        return 0
    try:
        cursor = db.execute("DELETE FROM backtest_history")
        db.commit()
        return cursor.rowcount
    except Exception:
        log.exception("Error clearing backtest history")
        return 0


def _run_backtest_thread(
    run: BacktestRun,
    db_path: str,
    strategies_dir: str,
    initial_balance: float,
    max_leverage: int,
    interval_ms: int,
    start_ms: int = 0,
    end_ms: int = 0,
) -> None:
    db = Database(db_path)
    db.open()

    try:
        base_dir = Path(strategies_dir).resolve()
        strategy_path = str(base_dir / run.strategy)
        resolved = Path(strategy_path).resolve()
        if not resolved.is_relative_to(base_dir):
            run.error = "Path traversal blocked"
            run.status = "error"
            run.queue.put({"type": "error", "message": run.error})
            return

        for i, coin in enumerate(run.coins):
            try:
                run.queue.put({"type": "progress", "coin": coin, "pct": 0, "coin_idx": i, "total_coins": len(run.coins)})

                candles_5m = _load_candles(db, coin, start_ms, end_ms)
                if not candles_5m:
                    run.queue.put({"type": "coin_done", "coin": coin, "error": "No candles"})
                    continue

                bt_config = BacktestConfig(
                    coin=coin,
                    strategy_path=strategy_path,
                    initial_balance=initial_balance,
                    max_leverage=max_leverage,
                    strategy_interval_ms=interval_ms,
                )

                loader = StrategyLoader(strategies_dir)
                instance = loader._load_module(str(resolved), Path(run.strategy).stem)

                bt_engine = BacktestEngine(bt_config, db)

                monitor = threading.Thread(
                    target=_monitor_progress,
                    args=(run, bt_engine, coin, len(candles_5m), i, len(run.coins)),
                    daemon=True,
                )
                monitor.start()

                result = bt_engine.run(instance, candles_5m)

                exit_pnls = [t.pnl for t in result.trades if t.pnl != 0]
                mc = {}
                if len(exit_pnls) >= 5:
                    mc_result = run_monte_carlo(exit_pnls, initial_balance)
                    mc = mc_result.to_dict()

                result_dict = _result_to_dict(result)
                result_dict["monte_carlo"] = mc
                result_dict["verdict"] = _compute_verdict(result)

                wf = _walk_forward(candles_5m, instance.__class__, bt_config, db, loader, str(resolved))
                result_dict["walk_forward"] = wf

                run.results[coin] = result_dict

                _save_to_history(db, run.run_id, run.strategy, coin, result, result_dict)

                run.queue.put({"type": "coin_done", "coin": coin, "result": result_dict})

            except Exception as e:
                log.exception(f"Backtest error for {coin}")
                run.queue.put({"type": "coin_done", "coin": coin, "error": str(e)})

        run.status = "complete"
        run.queue.put({"type": "complete", "results": run.results})

    except Exception as e:
        log.exception("Backtest thread error")
        run.error = str(e)
        run.status = "error"
        run.queue.put({"type": "error", "message": str(e)})
    finally:
        db.close()


def _monitor_progress(run: BacktestRun, bt_engine: BacktestEngine, coin: str, total_candles: int, coin_idx: int, total_coins: int) -> None:
    while run.status == "running":
        try:
            current = len(bt_engine._equity_curve)
            pct = min(current / total_candles * 100, 100) if total_candles > 0 else 0
            run.queue.put({
                "type": "progress",
                "coin": coin,
                "pct": round(pct, 1),
                "coin_idx": coin_idx,
                "total_coins": total_coins,
            })
        except Exception:
            pass
        time.sleep(0.5)


def _walk_forward(candles_5m, strategy_cls, bt_config, db, loader, resolved):
    if len(candles_5m) < 100:
        return None

    try:
        split = int(len(candles_5m) * 0.7)
        is_candles = candles_5m[:split]
        oos_candles = candles_5m[split:]

        is_instance = strategy_cls()
        is_engine = BacktestEngine(bt_config, db)
        is_result = is_engine.run(is_instance, is_candles)

        oos_instance = strategy_cls()
        oos_engine = BacktestEngine(bt_config, db)
        oos_result = oos_engine.run(oos_instance, oos_candles)

        is_return = is_result.return_pct
        oos_return = oos_result.return_pct
        decay = ((oos_return - is_return) / is_return * 100) if is_return != 0 else 0

        return {
            "is_return": round(is_return, 2),
            "oos_return": round(oos_return, 2),
            "decay_pct": round(decay, 2),
            "overfit_alert": decay < -50,
        }
    except Exception:
        log.exception("Walk-forward analysis failed")
        return None


def _compute_verdict(result) -> str:
    r = result.return_pct
    s = result.sharpe_ratio
    dd = result.max_drawdown_pct
    wr = result.win_rate * 100
    pf = result.profit_factor
    trades = result.total_trades

    if r > 10 and s > 1.0 and dd < 15 and wr > 40 and pf > 1.5:
        return "DEPLOYABLE"
    if r > 5 and s > 0.5 and pf > 1.0:
        return "A_OPTIMISER"
    if r > 0 and pf > 0.8:
        return "MARGINAL"
    if r > -5 or trades < 10:
        return "INSUFFISANT"
    return "ABANDON"


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
        "win_rate": round(result.win_rate * 100, 2),
        "profit_factor": round(result.profit_factor, 4) if result.profit_factor != float("inf") else 999.0,
        "avg_win": round(result.avg_win, 6),
        "avg_loss": round(result.avg_loss, 6),
        "max_win": round(result.max_win, 6),
        "max_loss": round(result.max_loss, 6),
        "max_drawdown_pct": round(result.max_drawdown_pct, 2),
        "sharpe_ratio": round(result.sharpe_ratio, 4),
        "sortino_ratio": round(result.sortino_ratio, 4),
        "equity_curve": result.equity_curve[-500:] if len(result.equity_curve) > 500 else result.equity_curve,
        "trades": [
            {
                "time_ms": t.time_ms,
                "side": t.side,
                "price": t.price,
                "size": t.size,
                "pnl": round(t.pnl, 6),
                "fee": round(t.fee, 6),
                "balance_after": round(t.balance_after, 4),
            }
            for t in result.trades
        ],
    }


def _save_to_history(db, run_id, strategy, coin, result, result_dict):
    try:
        db.execute(
            "INSERT OR REPLACE INTO backtest_history "
            "(run_id, strategy, coin, timestamp_ms, return_pct, sharpe, max_dd, "
            "win_rate, total_trades, profit_factor, verdict, config_json, result_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"{run_id}_{coin}",
                strategy,
                coin,
                int(time.time() * 1000),
                result.return_pct,
                result.sharpe_ratio,
                result.max_drawdown_pct,
                result.win_rate * 100,
                result.total_trades,
                result.profit_factor if result.profit_factor != float("inf") else 999.0,
                result_dict.get("verdict", ""),
                "{}",
                json.dumps(result_dict, default=str),
            ),
        )
        db.commit()
    except Exception:
        log.exception("Error saving backtest to history")


def _load_candles(db: Database, coin: str, start_ms: int = 0, end_ms: int = 0) -> list[Candle]:
    query = "SELECT * FROM candles WHERE coin=? AND interval='5m'"
    params: list = [coin]

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
            open=r["open"],
            high=r["high"],
            low=r["low"],
            close=r["close"],
            volume=r["volume"],
            n_trades=r["n_trades"] or 0,
        )
        for r in rows
    ]


def _ms_to_date(ms: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
