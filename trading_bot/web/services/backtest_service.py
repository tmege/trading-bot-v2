"""
Backtest service — connects the GUI to the crypto_bot V2 backtesting engine.

Uses SweepBacktester._run_realistic() for Hyperliquid-realistic simulation:
  - Maker/taker fees, slippage on SL, entry offset ALO
  - Sizing compose with drawdown multiplier
  - Cooldown between trades, funding rate, max hold timeout
  - FeatureEngine for 40+ technical indicators
  - V2 strategies for signal generation
"""
import json
import logging
import os
import queue as queue_mod
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# H-05: Add crypto_bot to path with validation
_CRYPTO_BOT_DIR = str(Path(__file__).resolve().parents[3] / "crypto_bot")
if not os.path.isdir(_CRYPTO_BOT_DIR):
    raise ImportError(f"crypto_bot directory not found: {_CRYPTO_BOT_DIR}")
# Validate no stdlib shadow modules exist
for _shadow in ("os", "sys", "json", "re", "time", "logging", "pathlib", "io",
                "http", "socket", "subprocess", "threading", "asyncio", "httpx"):
    if os.path.exists(os.path.join(_CRYPTO_BOT_DIR, f"{_shadow}.py")):
        raise ImportError(f"SECURITY: crypto_bot contains stdlib shadow module: {_shadow}.py")
if _CRYPTO_BOT_DIR not in sys.path:
    sys.path.insert(0, _CRYPTO_BOT_DIR)

from exec_config import ExecConfig
from modules.feature_engine import FeatureEngine
from modules.strategies import V2_STRATEGY_REGISTRY
from sweep_runner import SweepBacktester

from trading_bot.db import Database

log = logging.getLogger(__name__)

INITIAL_EQUITY = 1000.0
_CONFIG_PATH = os.path.join(_CRYPTO_BOT_DIR, "config.yaml")


# ═══════════════════════════════════════════════════════════════
# Strategy mapping: live file → V2 class + params + exec config
# ═══════════════════════════════════════════════════════════════

STRATEGY_MAP = {
    "btc_inside_bar_breakout_1h.py": {
        "v2_class": "StratInsideBarBreakout",
        "v2_params": {
            "vol_min": 0.8,
            "trend_filter": True,
            "atr_filter": True,
            "sl_pct": 2.5,
            "tp_pct": 4.5,
        },
        "exec_config": ExecConfig(
            equity_pct=0.15, leverage=5,
            cooldown_bars=4, max_hold_bars=72,
        ),
        "signal_filter": "hours_8_20",
    },
    "sol_breakout_normal_1h.py": {
        "v2_class": "StratBreakoutRelaxed",
        "v2_params": {
            "lookback": 14,
            "vol_breakout_min": 2.5,
            "sl_pct": 0.9,
            "tp_pct": 4.0,
        },
        "exec_config": ExecConfig(
            equity_pct=0.15, leverage=5,
            cooldown_bars=4, max_hold_bars=48,
        ),
        "signal_filter": "anti_wick_40",
    },
    "eth_breakout_relaxed_1h.py": {
        "v2_class": "StratBreakoutRelaxed",
        "v2_params": {
            "lookback": 35,
            "vol_breakout_min": 4.5,
            "sl_pct": 1.8,
            "tp_pct": 3.5,
        },
        "exec_config": ExecConfig(
            equity_pct=0.20, leverage=5,
            cooldown_bars=4, max_hold_bars=36,
        ),
        "signal_filter": "anti_wick_60",
    },
    "xrp_mean_reversion_bb_1h.py": {
        "v2_class": "StratMeanReversionBB",
        "v2_params": {
            "rsi_oversold": 20,
            "rsi_overbought": 70,
            "bb_entry_low": 0.08,
            "bb_entry_high": 0.95,
            "sl_pct": 0.7,
            "tp_pct": 8.0,
        },
        "exec_config": ExecConfig(
            equity_pct=0.35, leverage=5,
            cooldown_bars=4, max_hold_bars=48,
        ),
        "signal_filter": "anti_wick_50",
    },
    "bnb_breakout_relaxed_1h.py": {
        "v2_class": "StratBreakoutRelaxed",
        "v2_params": {
            "lookback": 32,
            "vol_breakout_min": 0.8,
            "sl_pct": 0.3,
            "tp_pct": 4.0,
        },
        "exec_config": ExecConfig(
            equity_pct=0.35, leverage=5,
            cooldown_bars=3, max_hold_bars=48,
        ),
        "signal_filter": None,
    },
}


# ═══════════════════════════════════════════════════════════════
# Signal filters (same as crypto_bot sweeps)
# ═══════════════════════════════════════════════════════════════

def _filter_hours_8_20(signals, df):
    hours = df.index.hour
    mask = pd.Series(True, index=df.index)
    for h in list(range(0, 8)) + list(range(21, 24)):
        mask = mask & (hours != h)
    return signals.where(mask, 0)


def _filter_anti_wick(ratio):
    def _f(signals, df):
        body = (df["close"] - df["open"]).abs()
        total_range = df["high"] - df["low"]
        wick_ratio = 1 - body / total_range.replace(0, 1)
        return signals.where(wick_ratio < ratio, 0)
    return _f


SIGNAL_FILTERS = {
    "hours_8_20": _filter_hours_8_20,
    "anti_wick_40": _filter_anti_wick(0.40),
    "anti_wick_50": _filter_anti_wick(0.50),
    "anti_wick_60": _filter_anti_wick(0.60),
}


# ═══════════════════════════════════════════════════════════════
# Data types & state
# ═══════════════════════════════════════════════════════════════

# H-02: Limit concurrent backtests
_MAX_CONCURRENT_BACKTESTS = 2
_MAX_CANDLES_PER_QUERY = 200_000


@dataclass
class BacktestRun:
    run_id: str
    strategy: str
    coins: list[str]
    status: str = "pending"
    results: dict = field(default_factory=dict)
    queue: queue_mod.Queue = field(default_factory=lambda: queue_mod.Queue(maxsize=500))
    error: str = ""


# M-09: Thread-safe access to _runs
_runs_lock = threading.Lock()
_runs: dict[str, BacktestRun] = {}
_fe: FeatureEngine | None = None


def _get_fe() -> FeatureEngine:
    global _fe
    if _fe is None:
        _fe = FeatureEngine(_CONFIG_PATH)
    return _fe


# ═══════════════════════════════════════════════════════════════
# Public API (called by routes)
# ═══════════════════════════════════════════════════════════════

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
    strategies_dir: str = "",
    initial_balance: float = 1000.0,
    max_leverage: int = 50,
    interval_ms: int = 3_600_000,
    start_ms: int = 0,
    end_ms: int = 0,
) -> str:
    # H-02: Limit concurrent backtests
    with _runs_lock:
        running = sum(1 for r in _runs.values() if r.status == "running")
        if running >= _MAX_CONCURRENT_BACKTESTS:
            raise RuntimeError(f"Max {_MAX_CONCURRENT_BACKTESTS} concurrent backtests allowed")

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
        args=(run, db_path, strategy_file, start_ms, end_ms),
        daemon=True,
    )
    t.start()
    return run_id


def get_run(run_id: str) -> BacktestRun | None:
    with _runs_lock:
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


def get_latest_result(db: Database, strategy: str) -> dict | None:
    if not db:
        return None
    try:
        rows = db.fetchall(
            "SELECT coin, result_json, timestamp_ms FROM backtest_history "
            "WHERE strategy=? ORDER BY timestamp_ms DESC",
            (strategy,),
        )
        if not rows:
            return None
        seen = {}
        for r in rows:
            coin = r["coin"]
            if coin not in seen:
                try:
                    seen[coin] = json.loads(r["result_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
        if not seen:
            return None
        return {"strategy": strategy, "results": seen}
    except Exception:
        log.exception("Error fetching latest backtest result")
        return None


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


# ═══════════════════════════════════════════════════════════════
# Core backtest thread (crypto_bot V2 engine)
# ═══════════════════════════════════════════════════════════════

def _run_backtest_thread(
    run: BacktestRun,
    db_path: str,
    strategy_file: str,
    start_ms: int = 0,
    end_ms: int = 0,
) -> None:
    db = Database(db_path)
    db.open()

    try:
        mapping = STRATEGY_MAP.get(strategy_file)
        if not mapping:
            run.error = f"Strategy '{strategy_file}' not in STRATEGY_MAP"
            run.status = "error"
            run.queue.put({"type": "error", "message": run.error})
            return

        v2_cls = V2_STRATEGY_REGISTRY.get(mapping["v2_class"])
        if not v2_cls:
            run.error = f"V2 class '{mapping['v2_class']}' not found"
            run.status = "error"
            run.queue.put({"type": "error", "message": run.error})
            return

        bt = SweepBacktester(_CONFIG_PATH)
        fe = _get_fe()
        ec = mapping["exec_config"]
        sig_filter = SIGNAL_FILTERS.get(mapping["signal_filter"]) if mapping["signal_filter"] else None

        for i, coin in enumerate(run.coins):
            try:
                run.queue.put({
                    "type": "progress", "coin": coin, "pct": 0,
                    "coin_idx": i, "total_coins": len(run.coins),
                })

                # Load 5m candles from DB → DataFrame
                df_5m = _load_candles_df(db, coin, start_ms, end_ms)
                if df_5m is None or len(df_5m) < 100:
                    run.queue.put({"type": "coin_done", "coin": coin, "error": "Not enough candles"})
                    continue

                run.queue.put({
                    "type": "progress", "coin": coin, "pct": 20,
                    "coin_idx": i, "total_coins": len(run.coins),
                })

                # Resample to 1h + compute features
                df_1h = df_5m.resample("1h").agg({
                    "open": "first", "high": "max",
                    "low": "min", "close": "last", "volume": "sum",
                }).dropna(subset=["open"])

                df_1h = fe.compute_all(df_1h)

                run.queue.put({
                    "type": "progress", "coin": coin, "pct": 40,
                    "coin_idx": i, "total_coins": len(run.coins),
                })

                # Generate V2 signals
                strat = v2_cls(mapping["v2_params"])
                signals = strat.generate_signals(df_1h)

                # Apply signal filter
                if sig_filter is not None:
                    signals = sig_filter(signals, df_1h)

                run.queue.put({
                    "type": "progress", "coin": coin, "pct": 60,
                    "coin_idx": i, "total_coins": len(run.coins),
                })

                # Run realistic backtest
                metrics = bt.run(
                    df_1h, signals,
                    sl_pct=strat.sl_pct,
                    tp_pct=strat.tp_pct,
                    exec_config=ec,
                    initial_equity=INITIAL_EQUITY,
                )

                run.queue.put({
                    "type": "progress", "coin": coin, "pct": 80,
                    "coin_idx": i, "total_coins": len(run.coins),
                })

                # Walk-forward analysis (anchored expanding windows with gap)
                wf = _walk_forward_rolling(df_1h, v2_cls, mapping["v2_params"], bt, ec, sig_filter)

                # Monte Carlo
                trades_detail = metrics.get("trades_detail", [])
                mc = {}
                dollar_pnls = [t["net_pnl"] for t in trades_detail]
                if len(dollar_pnls) >= 5:
                    mc = SweepBacktester.monte_carlo(dollar_pnls, INITIAL_EQUITY) or {}

                # Convert to GUI format
                result_dict = _convert_to_gui(metrics, df_1h)
                result_dict["walk_forward"] = _sanitize_for_json(wf) if wf else None
                result_dict["monte_carlo"] = _sanitize_for_json(mc) if mc else {}
                result_dict["verdict"] = _compute_verdict(metrics)

                run.results[coin] = result_dict

                # Save to history
                _save_to_history(db, run.run_id, strategy_file, coin, metrics, result_dict)

                run.queue.put({
                    "type": "progress", "coin": coin, "pct": 100,
                    "coin_idx": i, "total_coins": len(run.coins),
                })
                run.queue.put({"type": "coin_done", "coin": coin, "result": result_dict})

            except Exception as e:
                log.exception("Backtest error for %s", coin)
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


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def _load_candles_df(
    db: Database, coin: str, start_ms: int = 0, end_ms: int = 0,
) -> pd.DataFrame | None:
    """Load 5m candles from SQLite → pandas DataFrame with DatetimeIndex."""
    query = "SELECT time_open, open, high, low, close, volume FROM candles WHERE coin=? AND interval='5m'"
    params: list = [coin]

    if start_ms > 0:
        query += " AND time_open >= ?"
        params.append(start_ms)
    if end_ms > 0:
        query += " AND time_open <= ?"
        params.append(end_ms)

    # H-02: Cap candles to prevent memory exhaustion
    query += f" ORDER BY time_open LIMIT {_MAX_CANDLES_PER_QUERY}"
    rows = db.fetchall(query, tuple(params))

    if not rows:
        return None

    data = []
    for r in rows:
        data.append({
            "time_open": r["time_open"],
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["volume"],
        })

    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["time_open"], unit="ms", utc=True)
    df = df.set_index("datetime")
    df = df.drop(columns=["time_open"])
    df = df[~df.index.duplicated(keep="first")]
    return df


# ═══════════════════════════════════════════════════════════════
# Walk-forward (V2 engine)
# ═══════════════════════════════════════════════════════════════

def _walk_forward_v2(df_1h, v2_cls, v2_params, bt, ec, sig_filter):
    """70/30 walk-forward using V2 engine."""
    if len(df_1h) < 200:
        return None
    try:
        split = int(len(df_1h) * 0.7)
        is_df = df_1h.iloc[:split]
        oos_df = df_1h.iloc[split:]

        results = {}
        for label, df_slice in [("is", is_df), ("oos", oos_df)]:
            strat = v2_cls(v2_params)
            signals = strat.generate_signals(df_slice)
            if sig_filter:
                signals = sig_filter(signals, df_slice)
            m = bt.run(
                df_slice, signals,
                sl_pct=strat.sl_pct, tp_pct=strat.tp_pct,
                exec_config=ec, initial_equity=INITIAL_EQUITY,
            )
            results[label] = m["total_return"] * 100  # ratio → %

        is_ret = results["is"]
        oos_ret = results["oos"]
        decay = ((oos_ret - is_ret) / is_ret * 100) if is_ret != 0 else 0

        return {
            "is_return": round(is_ret, 2),
            "oos_return": round(oos_ret, 2),
            "decay_pct": round(decay, 2),
            "overfit_alert": decay < -50,
        }
    except Exception:
        log.exception("Walk-forward analysis failed")
        return None


def _walk_forward_rolling(df_1h, v2_cls, v2_params, bt, ec, sig_filter):
    """Anchored expanding walk-forward with 2-week gap between IS and OOS.

    Windows:
      Window 1: train [0 : 50%],        gap 336 bars, test [50%+336 : 50%+336+step]
      Window 2: train [0 : 50%+step],   gap 336 bars, test [50%+2*step+336 : ...]
      ...
    Minimum 5 OOS windows, minimum 100 bars per OOS window.
    """
    total_bars = len(df_1h)
    if total_bars < 500:
        return _walk_forward_v2(df_1h, v2_cls, v2_params, bt, ec, sig_filter)

    try:
        gap_bars = 336  # 14 days x 24h
        min_windows = 5
        min_oos_bars = 100
        initial_train_end = int(total_bars * 0.50)

        # Auto-calculate step size to get at least min_windows OOS windows
        remaining = total_bars - initial_train_end
        step_bars = max(min_oos_bars, (remaining - gap_bars) // (min_windows + 1))

        windows = []
        w = 0

        while True:
            train_end = initial_train_end + w * step_bars
            oos_start = train_end + gap_bars
            oos_end = oos_start + step_bars

            # Bounds check
            if train_end > total_bars or oos_start >= total_bars:
                break
            if oos_end > total_bars:
                oos_end = total_bars
            if oos_end - oos_start < min_oos_bars:
                break

            is_df = df_1h.iloc[:train_end]
            oos_df = df_1h.iloc[oos_start:oos_end]

            # Run IS backtest
            strat_is = v2_cls(v2_params)
            sig_is = strat_is.generate_signals(is_df)
            if sig_filter:
                sig_is = sig_filter(sig_is, is_df)
            m_is = bt.run(
                is_df, sig_is,
                sl_pct=strat_is.sl_pct, tp_pct=strat_is.tp_pct,
                exec_config=ec, initial_equity=INITIAL_EQUITY,
            )

            # Run OOS backtest
            strat_oos = v2_cls(v2_params)
            sig_oos = strat_oos.generate_signals(oos_df)
            if sig_filter:
                sig_oos = sig_filter(sig_oos, oos_df)
            m_oos = bt.run(
                oos_df, sig_oos,
                sl_pct=strat_oos.sl_pct, tp_pct=strat_oos.tp_pct,
                exec_config=ec, initial_equity=INITIAL_EQUITY,
            )

            is_ret = m_is["total_return"] * 100
            oos_ret = m_oos["total_return"] * 100
            is_sharpe = m_is["sharpe_ratio"]
            oos_sharpe = m_oos["sharpe_ratio"]

            windows.append({
                "window": w + 1,
                "train_bars": train_end,
                "oos_bars": oos_end - oos_start,
                "is_return": round(is_ret, 2),
                "oos_return": round(oos_ret, 2),
                "is_sharpe": round(is_sharpe, 3),
                "oos_sharpe": round(oos_sharpe, 3),
                "oos_profitable": oos_ret > 0,
            })

            w += 1

        if not windows:
            return _walk_forward_v2(df_1h, v2_cls, v2_params, bt, ec, sig_filter)

        # Aggregate metrics
        is_sharpes = [win["is_sharpe"] for win in windows]
        oos_sharpes = [win["oos_sharpe"] for win in windows]
        is_returns = [win["is_return"] for win in windows]
        oos_returns = [win["oos_return"] for win in windows]

        mean_is_sharpe = float(np.mean(is_sharpes)) if is_sharpes else 0
        mean_oos_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0
        mean_is_return = float(np.mean(is_returns)) if is_returns else 0
        mean_oos_return = float(np.mean(oos_returns)) if oos_returns else 0

        overfit_score = (mean_oos_sharpe / mean_is_sharpe) if abs(mean_is_sharpe) > 0.01 else 0
        decay_pct = ((mean_oos_return - mean_is_return) / abs(mean_is_return) * 100) if abs(mean_is_return) > 0.01 else 0
        oos_consistency = sum(1 for w in windows if w["oos_profitable"]) / len(windows)

        aggregate = {
            "n_windows": len(windows),
            "gap_bars": gap_bars,
            "mean_is_return": round(mean_is_return, 2),
            "mean_oos_return": round(mean_oos_return, 2),
            "mean_is_sharpe": round(mean_is_sharpe, 3),
            "mean_oos_sharpe": round(mean_oos_sharpe, 3),
            "overfit_score": round(overfit_score, 3),
            "overfit_alert": overfit_score < 0.5,
            "decay_pct": round(decay_pct, 2),
            "oos_consistency": round(oos_consistency, 3),
        }

        # Legacy format for GUI backward compatibility
        legacy = {
            "is_return": round(mean_is_return, 2),
            "oos_return": round(mean_oos_return, 2),
            "decay_pct": round(decay_pct, 2),
            "overfit_alert": overfit_score < 0.5,
        }

        return {
            "windows": windows,
            "aggregate": aggregate,
            **legacy,
        }

    except Exception:
        log.exception("Walk-forward rolling analysis failed")
        return _walk_forward_v2(df_1h, v2_cls, v2_params, bt, ec, sig_filter)


# ═══════════════════════════════════════════════════════════════
# Result conversion (crypto_bot → GUI format)
# ═══════════════════════════════════════════════════════════════

def _convert_to_gui(metrics: dict, df_1h: pd.DataFrame) -> dict:
    """Convert SweepBacktester metrics to the format the GUI frontend expects."""
    initial = metrics.get("initial_equity", INITIAL_EQUITY)
    final = metrics.get("final_equity", initial)
    trades_detail = metrics.get("trades_detail", [])

    # Win/loss breakdown
    wins = [t for t in trades_detail if t["net_pnl"] > 0]
    losses = [t for t in trades_detail if t["net_pnl"] <= 0]

    avg_win = np.mean([t["net_pnl"] for t in wins]) if wins else 0.0
    avg_loss = abs(np.mean([t["net_pnl"] for t in losses])) if losses else 0.0
    max_win = max((t["net_pnl"] for t in wins), default=0.0)
    max_loss = min((t["net_pnl"] for t in losses), default=0.0)

    # Sortino ratio
    sortino = 0.0
    if len(trades_detail) >= 2:
        pnls = [t["pnl_pct"] for t in trades_detail]
        total_days = (df_1h.index[-1] - df_1h.index[0]).total_seconds() / 86400
        total_days = max(total_days, 1)
        trades_per_year = len(pnls) / (total_days / 365.25)
        downside = [p for p in pnls if p < 0]
        if downside:
            mean_pnl = np.mean(pnls)
            down_std = np.std(downside, ddof=1)
            if down_std > 1e-6:
                sortino = mean_pnl / down_std * np.sqrt(trades_per_year)
                sortino = max(-10.0, min(10.0, sortino))

    # Equity curve for chart (from trade exits)
    equity_curve = [{"time_ms": int(df_1h.index[0].timestamp() * 1000), "equity": initial}]
    for t in trades_detail:
        exit_time = t.get("exit_time")
        if exit_time is not None:
            ts_ms = int(exit_time.timestamp() * 1000) if hasattr(exit_time, "timestamp") else 0
            equity_curve.append({"time_ms": ts_ms, "equity": t["equity_after"]})

    # Limit equity curve size
    if len(equity_curve) > 500:
        equity_curve = equity_curve[-500:]

    # Trade journal
    gui_trades = []
    for t in trades_detail:
        exit_time = t.get("exit_time")
        ts_ms = int(exit_time.timestamp() * 1000) if exit_time is not None and hasattr(exit_time, "timestamp") else 0
        gui_trades.append({
            "time_ms": ts_ms,
            "side": "buy" if t["side"] == 1 else "sell",
            "price": round(t.get("entry_price", 0), 6),
            "size": round(t.get("notional", 0) / max(t.get("entry_price", 1), 1e-9), 6),
            "pnl": round(t["net_pnl"], 6),
            "fee": round(t.get("entry_fee", 0) + t.get("exit_fee", 0), 6),
            "balance_after": round(t["equity_after"], 4),
        })

    return _sanitize_for_json({
        "start_balance": initial,
        "end_balance": round(final, 4),
        "total_pnl": round(final - initial, 4),
        "total_fees": round(metrics.get("total_fees", 0) + metrics.get("total_funding", 0), 6),
        "return_pct": round(metrics["total_return"] * 100, 2),
        "total_trades": metrics["nb_trades"],
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(metrics["win_rate"] * 100, 2),
        "profit_factor": round(metrics["profit_factor"], 4) if metrics["profit_factor"] != float("inf") else 999.0,
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "max_win": round(max_win, 6),
        "max_loss": round(max_loss, 6),
        "max_drawdown_pct": round(metrics["max_drawdown"] * 100, 2),
        "sharpe_ratio": round(metrics["sharpe_ratio"], 4),
        "sortino_ratio": round(sortino, 4),
        "equity_curve": equity_curve,
        "trades": gui_trades,
    })


# ═══════════════════════════════════════════════════════════════
# Verdict & history
# ═══════════════════════════════════════════════════════════════

def _compute_verdict(metrics: dict) -> str:
    r = metrics["total_return"] * 100  # ratio → %
    s = metrics["sharpe_ratio"]
    dd = metrics["max_drawdown"] * 100  # ratio → %
    wr = metrics["win_rate"] * 100  # ratio → %
    pf = metrics["profit_factor"]
    trades = metrics["nb_trades"]

    if r > 10 and s > 1.0 and dd < 15 and wr > 40 and pf > 1.5:
        return "DEPLOYABLE"
    if r > 5 and s > 0.5 and pf > 1.0:
        return "A_OPTIMISER"
    if r > 0 and pf > 0.8:
        return "MARGINAL"
    if r > -5 or trades < 10:
        return "INSUFFISANT"
    return "ABANDON"


def _save_to_history(db, run_id, strategy, coin, metrics, result_dict):
    try:
        pf = metrics["profit_factor"]
        if pf == float("inf"):
            pf = 999.0

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
                metrics["total_return"] * 100,
                metrics["sharpe_ratio"],
                metrics["max_drawdown"] * 100,
                metrics["win_rate"] * 100,
                metrics["nb_trades"],
                pf,
                result_dict.get("verdict", ""),
                "{}",
                json.dumps(result_dict, default=str),
            ),
        )
        db.commit()
    except Exception:
        log.exception("Error saving backtest to history")


def _sanitize_for_json(obj):
    """Recursively convert numpy/pandas types to native Python for json.dumps()."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return int(obj.timestamp() * 1000)
    return obj


def _ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
