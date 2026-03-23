"""Backtest all 6 uniform coins individually + portfolio combos.

Usage: .venv/bin/python scripts/backtest_6coins.py
"""
import logging
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "crypto_bot"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exec_config import ExecConfig
from modules.feature_engine import FeatureEngine
from modules.strategies import V2_STRATEGY_REGISTRY
from sweep_runner import SweepBacktester

from trading_bot.db import Database

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INITIAL_EQUITY = 1000.0
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "crypto_bot", "config.yaml")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trading_bot.db")

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]

V2_PARAMS = {
    "lookback": 32,
    "vol_breakout_min": 0.8,
    "sl_pct": 0.3,
    "tp_pct": 4.0,
}
EC = ExecConfig(
    equity_pct=0.35, leverage=5,
    cooldown_bars=3, max_hold_bars=48,
)


def load_candles_df(db, coin):
    rows = db.fetchall(
        "SELECT time_open, open, high, low, close, volume FROM candles "
        "WHERE coin=? AND interval='5m' ORDER BY time_open",
        (coin,),
    )
    if not rows:
        return None
    data = [
        {"time_open": r["time_open"], "open": r["open"], "high": r["high"],
         "low": r["low"], "close": r["close"], "volume": r["volume"]}
        for r in rows
    ]
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["time_open"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["time_open"])
    df = df[~df.index.duplicated(keep="first")]
    return df


def run_coin_backtest(db, coin, fe, bt):
    df_5m = load_candles_df(db, coin)
    if df_5m is None or len(df_5m) < 100:
        return None

    df_1h = df_5m.resample("1h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["open"])

    df_1h = fe.compute_all(df_1h)

    v2_cls = V2_STRATEGY_REGISTRY["StratBreakoutRelaxed"]
    strat = v2_cls(V2_PARAMS)
    signals = strat.generate_signals(df_1h)

    metrics = bt.run(
        df_1h, signals,
        sl_pct=strat.sl_pct, tp_pct=strat.tp_pct,
        exec_config=EC, initial_equity=INITIAL_EQUITY,
    )

    return metrics


def main():
    db = Database(DB_PATH)
    db.open()

    try:
        fe = FeatureEngine(CONFIG_PATH)
        bt = SweepBacktester(CONFIG_PATH)

        results = {}
        for coin in COINS:
            print(f"  Backtesting {coin}...", end="", flush=True)
            m = run_coin_backtest(db, coin, fe, bt)
            if m:
                results[coin] = m
                print(f"  done ({m['nb_trades']} trades, {m['total_return']*100:+.1f}%)")
            else:
                print("  FAILED")

        # === Individual results ===
        print("\n" + "=" * 75)
        print("  RESULTATS INDIVIDUELS — Breakout Uniform 1h (params identiques)")
        print("=" * 75)
        print(f"  {'Coin':<6} {'Return':>9} {'Trades':>7} {'WR':>6} {'PF':>7} {'Sharpe':>8} {'MaxDD':>8} {'Verdict'}")
        print("-" * 75)

        for coin in COINS:
            m = results.get(coin)
            if not m:
                continue
            r = m["total_return"] * 100
            s = m["sharpe_ratio"]
            dd = m["max_drawdown"] * 100
            wr = m["win_rate"] * 100
            pf = m["profit_factor"]
            if pf == float("inf"):
                pf = 999.0
            trades = m["nb_trades"]

            if r > 10 and s > 1.0 and dd < 15 and wr > 40 and pf > 1.5:
                verdict = "DEPLOY"
            elif r > 5 and s > 0.5 and pf > 1.0:
                verdict = "OPTIM"
            elif r > 0 and pf > 0.8:
                verdict = "MARGIN"
            elif r > -5 or trades < 10:
                verdict = "INSUF"
            else:
                verdict = "ABANDON"

            print(f"  {coin:<6} {r:>+8.1f}% {trades:>7} {wr:>5.1f}% {pf:>7.2f} {s:>8.3f} {dd:>7.2f}% {verdict}")

        # === Portfolio combinations (4 of 6) ===
        print("\n" + "=" * 75)
        print("  COMBOS 4/6 — Return total & Sharpe moyen (ponderation egale)")
        print("=" * 75)
        print(f"  {'Combo':<28} {'Avg Ret':>9} {'Avg Sharpe':>11} {'Avg DD':>8} {'Sum Trades':>11}")
        print("-" * 75)

        combos_4 = []
        for combo in combinations(COINS, 4):
            coins_in = list(combo)
            rets = [results[c]["total_return"] * 100 for c in coins_in if c in results]
            sharpes = [results[c]["sharpe_ratio"] for c in coins_in if c in results]
            dds = [results[c]["max_drawdown"] * 100 for c in coins_in if c in results]
            trades = [results[c]["nb_trades"] for c in coins_in if c in results]

            if len(rets) < 4:
                continue

            avg_ret = np.mean(rets)
            avg_sharpe = np.mean(sharpes)
            avg_dd = np.mean(dds)
            sum_trades = sum(trades)

            combos_4.append({
                "coins": coins_in,
                "avg_ret": avg_ret,
                "avg_sharpe": avg_sharpe,
                "avg_dd": avg_dd,
                "sum_trades": sum_trades,
            })

        combos_4.sort(key=lambda x: x["avg_ret"], reverse=True)

        for c in combos_4:
            label = "+".join(c["coins"])
            print(f"  {label:<28} {c['avg_ret']:>+8.1f}% {c['avg_sharpe']:>11.3f} {c['avg_dd']:>7.2f}% {c['sum_trades']:>11}")

        # === 6/6 baseline ===
        print("-" * 75)
        all_rets = [results[c]["total_return"] * 100 for c in COINS if c in results]
        all_sharpes = [results[c]["sharpe_ratio"] for c in COINS if c in results]
        all_dds = [results[c]["max_drawdown"] * 100 for c in COINS if c in results]
        all_trades = [results[c]["nb_trades"] for c in COINS if c in results]

        print(f"  {'6/6 BASELINE':<28} {np.mean(all_rets):>+8.1f}% {np.mean(all_sharpes):>11.3f} {np.mean(all_dds):>7.2f}% {sum(all_trades):>11}")
        print("=" * 75)

        # === Combos 5/6 ===
        print("\n" + "=" * 75)
        print("  COMBOS 5/6 — Quel coin retirer?")
        print("=" * 75)
        print(f"  {'Retire':<8} {'Combo':<30} {'Avg Ret':>9} {'Avg Sharpe':>11} {'Avg DD':>8}")
        print("-" * 75)

        for remove in COINS:
            coins_in = [c for c in COINS if c != remove]
            rets = [results[c]["total_return"] * 100 for c in coins_in if c in results]
            sharpes = [results[c]["sharpe_ratio"] for c in coins_in if c in results]
            dds = [results[c]["max_drawdown"] * 100 for c in coins_in if c in results]

            if len(rets) < 5:
                continue

            label = "+".join(coins_in)
            print(f"  -{remove:<6} {label:<30} {np.mean(rets):>+8.1f}% {np.mean(sharpes):>11.3f} {np.mean(dds):>7.2f}%")

        print("=" * 75)

    finally:
        db.close()


if __name__ == "__main__":
    main()
