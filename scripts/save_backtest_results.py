#!/usr/bin/env python3
"""
Sauvegarde les resultats du backtest multi-periodes 5 strategies dans la DB.
"""
import sys
import os
import time
import json
import uuid

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "crypto_bot"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exec_config import ExecConfig
from modules.feature_engine import FeatureEngine
from modules.strategies import V2_STRATEGY_REGISTRY
from sweep_runner import SweepBacktester
from trading_bot.db import Database

INITIAL_EQUITY = 1000.0
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trading_bot.db")


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

STRATEGIES = [
    {"name": "BTC", "coin": "BTC", "v2_class": "StratInsideBarBreakout",
     "v2_params": {"vol_min": 0.8, "trend_filter": True, "atr_filter": True, "sl_pct": 2.5, "tp_pct": 4.5},
     "exec_config": ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=72),
     "signal_filter": "hours_8_20", "strategy_file": "btc_inside_bar_breakout_1h.py"},
    {"name": "SOL", "coin": "SOL", "v2_class": "StratBreakoutRelaxed",
     "v2_params": {"lookback": 14, "vol_breakout_min": 2.5, "sl_pct": 0.9, "tp_pct": 4.0},
     "exec_config": ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=48),
     "signal_filter": "anti_wick_40", "strategy_file": "sol_breakout_normal_1h.py"},
    {"name": "ETH", "coin": "ETH", "v2_class": "StratBreakoutRelaxed",
     "v2_params": {"lookback": 35, "vol_breakout_min": 4.5, "sl_pct": 1.8, "tp_pct": 3.5},
     "exec_config": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=36),
     "signal_filter": "anti_wick_60", "strategy_file": "eth_breakout_relaxed_1h.py"},
    {"name": "XRP", "coin": "XRP", "v2_class": "StratMeanReversionBB",
     "v2_params": {"rsi_oversold": 20, "rsi_overbought": 70, "bb_entry_low": 0.08, "bb_entry_high": 0.95, "sl_pct": 0.7, "tp_pct": 8.0},
     "exec_config": ExecConfig(equity_pct=0.35, leverage=5, cooldown_bars=4, max_hold_bars=48),
     "signal_filter": "anti_wick_50", "strategy_file": "xrp_mean_reversion_bb_1h.py"},
    {"name": "BNB", "coin": "BNB", "v2_class": "StratBreakoutRelaxed",
     "v2_params": {"lookback": 32, "vol_breakout_min": 0.8, "sl_pct": 0.3, "tp_pct": 4.0},
     "exec_config": ExecConfig(equity_pct=0.35, leverage=5, cooldown_bars=3, max_hold_bars=48),
     "signal_filter": None, "strategy_file": "bnb_breakout_relaxed_1h.py"},
]

WINDOWS = [
    ("6M_2023H1",  "2023-01-01", "2023-07-01"),
    ("6M_2023H2",  "2023-07-01", "2024-01-01"),
    ("6M_2024H1",  "2024-01-01", "2024-07-01"),
    ("6M_2024H2",  "2024-07-01", "2025-01-01"),
    ("6M_2025H1",  "2025-01-01", "2025-07-01"),
    ("6M_2025H2",  "2025-07-01", "2026-01-01"),
    ("1Y_2023",     "2023-01-01", "2024-01-01"),
    ("1Y_2024",     "2024-01-01", "2025-01-01"),
    ("1Y_2025",     "2025-01-01", "2026-01-01"),
    ("1Y_mid23-24", "2023-07-01", "2024-07-01"),
    ("1Y_mid24-25", "2024-07-01", "2025-07-01"),
    ("18M_23-mid24", "2023-01-01", "2024-07-01"),
    ("18M_mid23-25", "2023-07-01", "2025-01-01"),
    ("18M_24-mid25", "2024-01-01", "2025-07-01"),
    ("18M_mid24-26", "2024-07-01", "2026-01-01"),
    ("2Y_2023-25",  "2023-01-01", "2025-01-01"),
    ("2Y_mid23-mid25", "2023-07-01", "2025-07-01"),
    ("2Y_2024-26",  "2024-01-01", "2026-01-01"),
    ("3Y_2023-26",  "2023-01-01", "2026-01-01"),
]


def load_all_candles(coin):
    db = Database(DB_PATH)
    db.open()
    rows = db.fetchall(
        "SELECT time_open, open, high, low, close, volume FROM candles "
        "WHERE coin=? AND interval='5m' ORDER BY time_open", (coin,))
    db.close()
    if not rows:
        return None
    data = [dict(r) for r in rows]
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["time_open"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["time_open"])
    df = df[~df.index.duplicated(keep="first")]
    return df


def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "crypto_bot", "config.yaml")
    fe = FeatureEngine(config_path)
    bt = SweepBacktester(config_path)

    db = Database(DB_PATH)
    db.open()

    print("Chargement des donnees...")
    raw_5m = {}
    for s in STRATEGIES:
        coin = s["coin"]
        if coin not in raw_5m:
            raw_5m[coin] = load_all_candles(coin)
            print("  %s: OK" % coin)

    saved = 0

    for wname, start, end in WINDOWS:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")

        for s in STRATEGIES:
            coin = s["coin"]
            if raw_5m[coin] is None:
                continue

            df_5m = raw_5m[coin].loc[start_ts:end_ts]
            if len(df_5m) < 100:
                continue

            df_1h = df_5m.resample("1h").agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last", "volume": "sum",
            }).dropna(subset=["open"])

            if len(df_1h) < 50:
                continue

            df_1h = fe.compute_all(df_1h)

            v2_cls = V2_STRATEGY_REGISTRY[s["v2_class"]]
            strat = v2_cls(s["v2_params"])
            signals = strat.generate_signals(df_1h)

            sig_filter = SIGNAL_FILTERS.get(s["signal_filter"]) if s["signal_filter"] else None
            if sig_filter:
                signals = sig_filter(signals, df_1h)

            metrics = bt.run(
                df_1h, signals,
                sl_pct=strat.sl_pct, tp_pct=strat.tp_pct,
                exec_config=s["exec_config"],
                initial_equity=INITIAL_EQUITY,
            )

            pnl = metrics.get("dollar_pnl", 0)
            final = metrics.get("final_equity", INITIAL_EQUITY)
            pf = metrics["profit_factor"]
            if pf == float("inf"):
                pf = 999.0

            # Verdict
            r = metrics["total_return"] * 100
            sr = metrics["sharpe_ratio"]
            dd = metrics["max_drawdown"] * 100
            wr = metrics["win_rate"] * 100
            trades = metrics["nb_trades"]

            if r > 10 and sr > 1.0 and dd < 15 and wr > 40 and pf > 1.5:
                verdict = "DEPLOYABLE"
            elif r > 5 and sr > 0.5 and pf > 1.0:
                verdict = "A_OPTIMISER"
            elif r > 0 and pf > 0.8:
                verdict = "MARGINAL"
            elif r > -5 or trades < 10:
                verdict = "INSUFFISANT"
            else:
                verdict = "ABANDON"

            run_id = "%s_%s_%s" % (wname, s["strategy_file"].replace(".py", ""), coin)

            result_json = json.dumps({
                "window": wname,
                "period": "%s -> %s" % (start, end),
                "start_balance": INITIAL_EQUITY,
                "end_balance": round(final, 4),
                "total_pnl": round(pnl, 4),
                "return_pct": round(r, 2),
                "total_trades": trades,
                "winning_trades": int(wr / 100 * trades),
                "losing_trades": trades - int(wr / 100 * trades),
                "win_rate": round(wr, 2),
                "profit_factor": round(pf, 4),
                "max_drawdown_pct": round(dd, 2),
                "sharpe_ratio": round(sr, 4),
                "verdict": verdict,
            })

            db.execute(
                "INSERT OR REPLACE INTO backtest_history "
                "(run_id, strategy, coin, timestamp_ms, return_pct, sharpe, max_dd, "
                "win_rate, total_trades, profit_factor, verdict, config_json, result_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, s["strategy_file"], coin, int(time.time() * 1000),
                 round(r, 4), round(sr, 4), round(dd, 4), round(wr, 4),
                 trades, round(pf, 4), verdict, json.dumps({"window": wname}), result_json))
            saved += 1

    db.commit()
    db.close()
    print("\n%d resultats sauvegardes dans la DB." % saved)


if __name__ == "__main__":
    main()
