#!/usr/bin/env python3
"""
ETH BreakoutRelaxed — test de paramètres resserrés + block-shuffle.

On compare les params actuels (lookback=35, vol_min=4.5) avec des variantes
plus strictes pour voir si l'alpha kill monte au-dessus de 60%.

Variantes testées:
  - Current  : lookback=35, vol_min=4.5  (trop large, alpha kill 38%)
  - Tight A  : lookback=20, vol_min=5.0
  - Tight B  : lookback=15, vol_min=5.0
  - Tight C  : lookback=15, vol_min=6.0, compression=True
"""
import sys
import os
import time

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
START = "2023-01-01"
END = "2026-01-01"
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trading_bot.db")

BLOCK_SIZE = 168
BLOCK_SHUFFLE_ITERS = 5
SHUFFLE_SEED = 42

EXEC_CONFIG = ExecConfig(
    equity_pct=0.20, leverage=5,
    cooldown_bars=4, max_hold_bars=36,
)

# ── Signal filter ──

def _filter_anti_wick_60(signals, df):
    body = (df["close"] - df["open"]).abs()
    total_range = df["high"] - df["low"]
    wick_ratio = 1 - body / total_range.replace(0, 1)
    return signals.where(wick_ratio < 0.60, 0)

# ── Parameter variants ──

VARIANTS = [
    {
        "label": "Current (lb=35,vol=4.5)",
        "v2_params": {
            "lookback": 35, "vol_breakout_min": 4.5,
            "sl_pct": 1.8, "tp_pct": 3.5,
        },
    },
    {
        "label": "Tight A (lb=20,vol=5.0)",
        "v2_params": {
            "lookback": 20, "vol_breakout_min": 5.0,
            "sl_pct": 1.8, "tp_pct": 3.5,
        },
    },
    {
        "label": "Tight B (lb=15,vol=5.0)",
        "v2_params": {
            "lookback": 15, "vol_breakout_min": 5.0,
            "sl_pct": 1.8, "tp_pct": 3.5,
        },
    },
    {
        "label": "Tight C (lb=15,vol=6.0,compr)",
        "v2_params": {
            "lookback": 15, "vol_breakout_min": 6.0,
            "use_compression": True,
            "sl_pct": 1.8, "tp_pct": 3.5,
        },
    },
    {
        "label": "Tight D (lb=20,vol=6.0)",
        "v2_params": {
            "lookback": 20, "vol_breakout_min": 6.0,
            "sl_pct": 1.8, "tp_pct": 3.5,
        },
    },
]


def load_candles_from_db(coin):
    db = Database(DB_PATH)
    db.open()
    start_ms = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(END, tz="UTC").timestamp() * 1000)

    rows = db.fetchall(
        "SELECT time_open, open, high, low, close, volume FROM candles "
        "WHERE coin=? AND interval='5m' AND time_open >= ? AND time_open <= ? "
        "ORDER BY time_open",
        (coin, start_ms, end_ms),
    )
    db.close()
    if not rows:
        return None
    data = [dict(r) for r in rows]
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["time_open"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["time_open"])
    df = df[~df.index.duplicated(keep="first")]
    return df


def block_shuffle_df(df_1h, fe, block_size, rng):
    n = len(df_1h)
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    values = df_1h[ohlcv_cols].values
    n_blocks = n // block_size
    remainder = n % block_size
    blocks = [values[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    if remainder > 0:
        blocks.append(values[n_blocks * block_size:])
    block_indices = list(range(len(blocks)))
    rng.shuffle(block_indices)
    shuffled = np.concatenate([blocks[i] for i in block_indices], axis=0)
    df_shuffled = df_1h.copy()
    df_shuffled[ohlcv_cols] = shuffled
    return fe.compute_all(df_shuffled)


def run_backtest(df_1h, v2_params, bt):
    v2_cls = V2_STRATEGY_REGISTRY["StratBreakoutRelaxed"]
    strat = v2_cls(v2_params)
    signals = strat.generate_signals(df_1h)
    signals = _filter_anti_wick_60(signals, df_1h)
    metrics = bt.run(
        df_1h, signals,
        sl_pct=strat.sl_pct, tp_pct=strat.tp_pct,
        exec_config=EXEC_CONFIG,
        initial_equity=INITIAL_EQUITY,
    )
    return metrics


def main():
    t0 = time.time()
    config_path = os.path.join(os.path.dirname(__file__), "..", "crypto_bot", "config.yaml")
    fe = FeatureEngine(config_path)
    bt = SweepBacktester(config_path)

    print("=" * 130)
    print("  ETH BreakoutRelaxed — RESSERREMENT PARAMETRES + BLOCK-SHUFFLE")
    print("  Block=%d bars | %d iter | anti_wick_60" % (BLOCK_SIZE, BLOCK_SHUFFLE_ITERS))
    print("=" * 130)

    # Load data
    df_5m = load_candles_from_db("ETH")
    if df_5m is None:
        print("  ERREUR: pas de donnees ETH")
        return
    df_1h = df_5m.resample("1h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["open"])
    df_1h = fe.compute_all(df_1h)
    print("  ETH: %s bougies 1h\n" % f"{len(df_1h):,}")

    # Test each variant
    print("  %-32s %7s %6s %7s %10s %10s %8s %8s" % (
        "Variante", "Return%", "Sharpe", "Trades", "BlkShuf%", "AlphaKill", "Status", "MaxDD%"))
    print("  " + "-" * 105)

    for v in VARIANTS:
        # Normal backtest
        metrics = run_backtest(df_1h, v["v2_params"], bt)
        normal_ret = metrics["total_return"] * 100
        sharpe = metrics["sharpe_ratio"]
        trades = metrics["nb_trades"]
        maxdd = metrics["max_drawdown"] * 100

        # Block-shuffle (N iterations)
        bshuf_rets = []
        for i in range(BLOCK_SHUFFLE_ITERS):
            rng = np.random.RandomState(SHUFFLE_SEED + i)
            df_shuf = block_shuffle_df(df_1h, fe, BLOCK_SIZE, rng)
            m = run_backtest(df_shuf, v["v2_params"], bt)
            bshuf_rets.append(m["total_return"] * 100)

        bshuf_median = float(np.median(bshuf_rets))
        if abs(normal_ret) > 0.01:
            alpha_kill = (normal_ret - bshuf_median) / abs(normal_ret) * 100
        else:
            alpha_kill = 100.0

        status = "PASS" if (alpha_kill > 50 or abs(bshuf_median) < 20) else "FAIL"

        print("  %-32s %+6.1f%% %+5.2f %7d %+9.1f%% %+8.1f%% %8s %7.1f%%" % (
            v["label"], normal_ret, sharpe, trades,
            bshuf_median, alpha_kill, status, maxdd))

    elapsed = time.time() - t0
    print("\n  Temps: %.1fs" % elapsed)


if __name__ == "__main__":
    main()
