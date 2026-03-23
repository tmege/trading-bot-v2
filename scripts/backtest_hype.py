"""Standalone backtest for HYPE using the crypto_bot V2 engine.

Usage: .venv/bin/python scripts/backtest_hype.py
"""
import logging
import os
import sys

import numpy as np
import pandas as pd

# Add crypto_bot to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "crypto_bot"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exec_config import ExecConfig
from modules.feature_engine import FeatureEngine
from modules.strategies import V2_STRATEGY_REGISTRY
from sweep_runner import SweepBacktester

from trading_bot.db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INITIAL_EQUITY = 1000.0
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "crypto_bot", "config.yaml")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trading_bot.db")


def load_candles_df(db, coin):
    rows = db.fetchall(
        "SELECT time_open, open, high, low, close, volume FROM candles "
        "WHERE coin=? AND interval='5m' ORDER BY time_open",
        (coin,),
    )
    if not rows:
        return None

    data = [
        {
            "time_open": r["time_open"],
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["volume"],
        }
        for r in rows
    ]

    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["time_open"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["time_open"])
    df = df[~df.index.duplicated(keep="first")]
    return df


def main():
    db = Database(DB_PATH)
    db.open()

    try:
        # Load 5m candles
        log.info("Loading HYPE 5m candles...")
        df_5m = load_candles_df(db, "HYPE")
        if df_5m is None or len(df_5m) < 100:
            log.error("Not enough candles for HYPE")
            return

        log.info(f"Loaded {len(df_5m)} 5m candles: {df_5m.index[0]} → {df_5m.index[-1]}")

        # Resample to 1h
        df_1h = df_5m.resample("1h").agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum",
        }).dropna(subset=["open"])

        log.info(f"Resampled to {len(df_1h)} 1h candles")

        # Compute features
        fe = FeatureEngine(CONFIG_PATH)
        df_1h = fe.compute_all(df_1h)
        log.info(f"Features computed ({len(df_1h.columns)} columns)")

        # Strategy: BreakoutRelaxed with uniform params
        v2_params = {
            "lookback": 32,
            "vol_breakout_min": 0.8,
            "sl_pct": 0.3,
            "tp_pct": 4.0,
        }
        ec = ExecConfig(
            equity_pct=0.35, leverage=5,
            cooldown_bars=3, max_hold_bars=48,
        )

        v2_cls = V2_STRATEGY_REGISTRY["StratBreakoutRelaxed"]
        strat = v2_cls(v2_params)
        signals = strat.generate_signals(df_1h)

        n_long = (signals == 1).sum()
        n_short = (signals == -1).sum()
        log.info(f"Signals: {n_long} long, {n_short} short, {(signals == 0).sum()} neutral")

        # Run backtest
        bt = SweepBacktester(CONFIG_PATH)
        metrics = bt.run(
            df_1h, signals,
            sl_pct=strat.sl_pct, tp_pct=strat.tp_pct,
            exec_config=ec, initial_equity=INITIAL_EQUITY,
        )

        # Display results
        print("\n" + "=" * 60)
        print("  BACKTEST RESULTS — HYPE Breakout Uniform 1h")
        print("=" * 60)
        print(f"  Period        : {df_1h.index[0].strftime('%Y-%m-%d')} → {df_1h.index[-1].strftime('%Y-%m-%d')}")
        print(f"  1h bars       : {len(df_1h)}")
        print(f"  Initial equity: ${INITIAL_EQUITY:.2f}")
        print(f"  Final equity  : ${metrics.get('final_equity', 0):.2f}")
        print("-" * 60)
        print(f"  Total return  : {metrics['total_return'] * 100:+.2f}%")
        print(f"  Nb trades     : {metrics['nb_trades']}")
        print(f"  Win rate      : {metrics['win_rate'] * 100:.1f}%")
        print(f"  Profit factor : {metrics['profit_factor']:.2f}")
        print(f"  Sharpe ratio  : {metrics['sharpe_ratio']:.3f}")
        print(f"  Max drawdown  : {metrics['max_drawdown'] * 100:.2f}%")

        total_fees = metrics.get("total_fees", 0)
        total_funding = metrics.get("total_funding", 0)
        if total_fees or total_funding:
            print(f"  Total fees    : ${total_fees:.4f}")
            print(f"  Total funding : ${total_funding:.4f}")

        trades_detail = metrics.get("trades_detail", [])
        if trades_detail:
            wins = [t for t in trades_detail if t["net_pnl"] > 0]
            losses = [t for t in trades_detail if t["net_pnl"] <= 0]
            avg_win = np.mean([t["net_pnl"] for t in wins]) if wins else 0
            avg_loss = abs(np.mean([t["net_pnl"] for t in losses])) if losses else 0
            print(f"  Wins/Losses   : {len(wins)}/{len(losses)}")
            print(f"  Avg win       : ${avg_win:.4f}")
            print(f"  Avg loss      : ${avg_loss:.4f}")

            # Show exits breakdown
            exits = {}
            for t in trades_detail:
                r = t.get("exit_reason", "?")
                exits[r] = exits.get(r, 0) + 1
            print(f"  Exit reasons  : {exits}")

        print("-" * 60)

        # Verdict
        r = metrics["total_return"] * 100
        s = metrics["sharpe_ratio"]
        dd = metrics["max_drawdown"] * 100
        wr = metrics["win_rate"] * 100
        pf = metrics["profit_factor"]
        trades = metrics["nb_trades"]

        if r > 10 and s > 1.0 and dd < 15 and wr > 40 and pf > 1.5:
            verdict = "DEPLOYABLE"
        elif r > 5 and s > 0.5 and pf > 1.0:
            verdict = "A_OPTIMISER"
        elif r > 0 and pf > 0.8:
            verdict = "MARGINAL"
        elif r > -5 or trades < 10:
            verdict = "INSUFFISANT"
        else:
            verdict = "ABANDON"

        print(f"  VERDICT       : {verdict}")
        print("=" * 60)

    finally:
        db.close()


if __name__ == "__main__":
    main()
