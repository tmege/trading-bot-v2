#!/usr/bin/env python3
"""
Audit Lookahead Bias — 3 tests formels pour prouver l'absence de biais.

Tests:
  1. BLOCK-SHUFFLE: Découpe en blocs de 168 bars (1 semaine), shuffle l'ordre des blocs,
     recompute features, backtest. Préserve la structure locale des prix.
     Médiane sur 5 itérations pour réduire le bruit.
  2. SHIFT: Décale les signaux de +1 bar (simule un lookahead) → doit montrer inflation
  3. COMPARE: shift_inflation = (shifted - normal) / normal — négatif = suspect

Critères PASS/FAIL:
  - Block-shuffle: alpha_destroyed > 50% (la majorité de l'alpha disparaît)
     OU |block_shuffle_median| < 20%
  - Shift inflation > -10% (pour stratégies rentables)
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

SHUFFLE_SEED = 42
BLOCK_SIZE = 168          # 1 week of 1h bars
BLOCK_SHUFFLE_ITERS = 5   # median over N shuffles
SHIFT_INFLATION_THRESHOLD_PCT = -10.0


# ── Signal filters (same as backtest_5strat.py) ──

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
    {
        "name": "BTC InsideBarBreakout",
        "coin": "BTC",
        "v2_class": "StratInsideBarBreakout",
        "v2_params": {
            "vol_min": 0.8, "trend_filter": True,
            "atr_filter": True, "sl_pct": 2.5, "tp_pct": 4.5,
        },
        "exec_config": ExecConfig(
            equity_pct=0.15, leverage=5,
            cooldown_bars=4, max_hold_bars=72,
        ),
        "signal_filter": "hours_8_20",
    },
    {
        "name": "SOL BreakoutNormal",
        "coin": "SOL",
        "v2_class": "StratBreakoutRelaxed",
        "v2_params": {
            "lookback": 14, "vol_breakout_min": 2.5,
            "sl_pct": 0.9, "tp_pct": 4.0,
        },
        "exec_config": ExecConfig(
            equity_pct=0.15, leverage=5,
            cooldown_bars=4, max_hold_bars=48,
        ),
        "signal_filter": "anti_wick_40",
    },
    {
        "name": "ETH BreakoutRelaxed",
        "coin": "ETH",
        "v2_class": "StratBreakoutRelaxed",
        "v2_params": {
            "lookback": 35, "vol_breakout_min": 4.5,
            "sl_pct": 1.8, "tp_pct": 3.5,
        },
        "exec_config": ExecConfig(
            equity_pct=0.20, leverage=5,
            cooldown_bars=4, max_hold_bars=36,
        ),
        "signal_filter": "anti_wick_60",
    },
    {
        "name": "XRP MeanReversionBB",
        "coin": "XRP",
        "v2_class": "StratMeanReversionBB",
        "v2_params": {
            "rsi_oversold": 20, "rsi_overbought": 70,
            "bb_entry_low": 0.08, "bb_entry_high": 0.95,
            "sl_pct": 0.7, "tp_pct": 8.0,
        },
        "exec_config": ExecConfig(
            equity_pct=0.35, leverage=5,
            cooldown_bars=4, max_hold_bars=48,
        ),
        "signal_filter": "anti_wick_50",
    },
    {
        "name": "BNB BreakoutRelaxed",
        "coin": "BNB",
        "v2_class": "StratBreakoutRelaxed",
        "v2_params": {
            "lookback": 32, "vol_breakout_min": 0.8,
            "sl_pct": 0.3, "tp_pct": 4.0,
        },
        "exec_config": ExecConfig(
            equity_pct=0.35, leverage=5,
            cooldown_bars=3, max_hold_bars=48,
        ),
        "signal_filter": None,
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


def run_normal(df_1h, strat_cfg, fe, bt):
    """Run backtest with normal signals. Returns (return_pct, metrics)."""
    v2_cls = V2_STRATEGY_REGISTRY[strat_cfg["v2_class"]]
    strat = v2_cls(strat_cfg["v2_params"])
    signals = strat.generate_signals(df_1h)

    sig_filter = SIGNAL_FILTERS.get(strat_cfg["signal_filter"]) if strat_cfg["signal_filter"] else None
    if sig_filter:
        signals = sig_filter(signals, df_1h)

    metrics = bt.run(
        df_1h, signals,
        sl_pct=strat.sl_pct, tp_pct=strat.tp_pct,
        exec_config=strat_cfg["exec_config"],
        initial_equity=INITIAL_EQUITY,
    )
    return metrics["total_return"] * 100, metrics


def _block_shuffle_df(df_1h, fe, block_size, rng):
    """Shuffle blocks of consecutive candles, recompute features.

    Splits OHLCV into blocks of `block_size` bars, shuffles block order,
    reassembles with original DatetimeIndex, then recomputes all features.
    Local price dynamics within each block are preserved.
    """
    n = len(df_1h)
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    values = df_1h[ohlcv_cols].values

    # Split into blocks
    n_blocks = n // block_size
    remainder = n % block_size

    blocks = [values[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    if remainder > 0:
        blocks.append(values[n_blocks * block_size:])

    # Shuffle block order
    block_indices = list(range(len(blocks)))
    rng.shuffle(block_indices)
    shuffled = np.concatenate([blocks[i] for i in block_indices], axis=0)

    # Rebuild DataFrame with original index
    df_shuffled = df_1h.copy()
    df_shuffled[ohlcv_cols] = shuffled

    return fe.compute_all(df_shuffled)


def run_block_shuffled(df_1h, strat_cfg, fe, bt, block_size=BLOCK_SIZE,
                       n_iter=BLOCK_SHUFFLE_ITERS, seed=SHUFFLE_SEED):
    """Block-shuffle: preserve local structure, destroy long-range trends.

    Runs n_iter shuffles and returns the median return.
    """
    returns = []
    for i in range(n_iter):
        rng = np.random.RandomState(seed + i)
        df_recomputed = _block_shuffle_df(df_1h, fe, block_size, rng)

        v2_cls = V2_STRATEGY_REGISTRY[strat_cfg["v2_class"]]
        strat = v2_cls(strat_cfg["v2_params"])
        signals = strat.generate_signals(df_recomputed)

        sig_filter = SIGNAL_FILTERS.get(strat_cfg["signal_filter"]) if strat_cfg["signal_filter"] else None
        if sig_filter:
            signals = sig_filter(signals, df_recomputed)

        metrics = bt.run(
            df_recomputed, signals,
            sl_pct=strat.sl_pct, tp_pct=strat.tp_pct,
            exec_config=strat_cfg["exec_config"],
            initial_equity=INITIAL_EQUITY,
        )
        returns.append(metrics["total_return"] * 100)

    median_ret = float(np.median(returns))
    return median_ret, returns


def run_shifted(df_1h, strat_cfg, fe, bt):
    """Shift signals by -1 bar to simulate lookahead (signal acts on same bar).

    The engine enters at bar[i+1].open when signal[i]!=0.
    Shifting signals by -1 means signal[i] becomes signal at bar[i-1],
    so entry would be at bar[i].open = same bar as the original signal.
    This simulates a 1-bar lookahead — should inflate returns.
    """
    v2_cls = V2_STRATEGY_REGISTRY[strat_cfg["v2_class"]]
    strat = v2_cls(strat_cfg["v2_params"])
    signals = strat.generate_signals(df_1h)

    sig_filter = SIGNAL_FILTERS.get(strat_cfg["signal_filter"]) if strat_cfg["signal_filter"] else None
    if sig_filter:
        signals = sig_filter(signals, df_1h)

    # Shift signals forward by 1 (simulate lookahead)
    signals_shifted = signals.shift(-1).fillna(0).astype(int)

    metrics = bt.run(
        df_1h, signals_shifted,
        sl_pct=strat.sl_pct, tp_pct=strat.tp_pct,
        exec_config=strat_cfg["exec_config"],
        initial_equity=INITIAL_EQUITY,
    )
    return metrics["total_return"] * 100, metrics


def main():
    t0 = time.time()
    config_path = os.path.join(os.path.dirname(__file__), "..", "crypto_bot", "config.yaml")
    fe = FeatureEngine(config_path)
    bt = SweepBacktester(config_path)

    print("=" * 120)
    print("  AUDIT LOOKAHEAD BIAS — Block-Shuffle + Shift")
    print("  Block size=%d bars (1 week) | %d iterations/strat | Shift inflation seuil=%.0f%%" % (
        BLOCK_SIZE, BLOCK_SHUFFLE_ITERS, SHIFT_INFLATION_THRESHOLD_PCT))
    print("=" * 120)

    # ── Load data ──
    print("\n-- Chargement des donnees --")
    data_1h = {}
    for s in STRATEGIES:
        coin = s["coin"]
        if coin in data_1h:
            continue
        df_5m = load_candles_from_db(coin)
        if df_5m is None:
            print("  ERREUR: pas de donnees pour %s" % coin)
            continue
        df_1h = df_5m.resample("1h").agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum",
        }).dropna(subset=["open"])
        df_1h = fe.compute_all(df_1h)
        print("  %s: %s bougies 1h" % (coin, f"{len(df_1h):,}"))
        data_1h[coin] = df_1h

    # ── Run tests ──
    results = []
    all_pass = True

    for s in STRATEGIES:
        coin = s["coin"]
        if coin not in data_1h:
            results.append({
                "name": s["name"], "coin": coin,
                "normal": None, "shuffle": None, "shifted": None,
                "error": "NO DATA",
            })
            continue

        df = data_1h[coin]
        print("\n  Testing: %s (%s)..." % (s["name"], coin))

        # Test 1: Normal
        normal_ret, _ = run_normal(df, s, fe, bt)
        print("    Normal       : %+.2f%%" % normal_ret)

        # Test 2: Block-shuffle (median of N iterations)
        bshuf_median, bshuf_all = run_block_shuffled(df, s, fe, bt)
        print("    BlockShuffle : %+.2f%% (median of %d: %s)" % (
            bshuf_median, len(bshuf_all),
            ", ".join("%+.1f" % r for r in bshuf_all)))

        # Test 3: Shifted (lookahead)
        shifted_ret, _ = run_shifted(df, s, fe, bt)
        print("    Shifted      : %+.2f%%" % shifted_ret)

        # Alpha destroyed %
        if abs(normal_ret) > 0.01:
            alpha_destroyed = (normal_ret - bshuf_median) / abs(normal_ret) * 100
        else:
            alpha_destroyed = 100.0  # no alpha to destroy

        # Shift inflation
        if abs(normal_ret) > 0.01:
            shift_inflation = (shifted_ret - normal_ret) / abs(normal_ret) * 100
        else:
            shift_inflation = 0.0

        # PASS/FAIL — block-shuffle
        # PASS if >50% of alpha destroyed OR block-shuffle median is small (<20%)
        bshuf_pass = (alpha_destroyed > 50) or (abs(bshuf_median) < 20)
        shift_pass = shift_inflation > SHIFT_INFLATION_THRESHOLD_PCT if normal_ret > 0 else True

        if not bshuf_pass or not shift_pass:
            all_pass = False

        results.append({
            "name": s["name"],
            "coin": coin,
            "normal": normal_ret,
            "bshuf_median": bshuf_median,
            "bshuf_all": bshuf_all,
            "alpha_destroyed": alpha_destroyed,
            "shifted": shifted_ret,
            "shift_inflation": shift_inflation,
            "bshuf_pass": bshuf_pass,
            "shift_pass": shift_pass,
        })

    # ── Report ──
    print("\n" + "=" * 130)
    print("  RESULTATS AUDIT LOOKAHEAD (Block-Shuffle %d bars, %d iter)" % (BLOCK_SIZE, BLOCK_SHUFFLE_ITERS))
    print("=" * 130)
    print()
    print("  %-24s %5s %9s %10s %10s %9s %12s %8s %8s %8s" % (
        "Strategie", "Coin", "Normal%", "BlkShuf%", "AlphaKill", "Shifted%", "ShiftInflat",
        "BShPASS", "ShftPASS", "GLOBAL"))
    print("  " + "-" * 120)

    for r in results:
        if r.get("error"):
            print("  %-24s %5s  -- %s --" % (r["name"], r["coin"], r["error"]))
            continue

        bsh_status = "PASS" if r["bshuf_pass"] else "FAIL"
        shft_status = "PASS" if r["shift_pass"] else "FAIL"
        global_status = "PASS" if (r["bshuf_pass"] and r["shift_pass"]) else "FAIL"

        print("  %-24s %5s %+8.2f%% %+9.2f%% %+8.1f%% %+8.2f%% %+10.1f%% %8s %8s %8s" % (
            r["name"], r["coin"],
            r["normal"], r["bshuf_median"], r["alpha_destroyed"],
            r["shifted"], r["shift_inflation"],
            bsh_status, shft_status, global_status,
        ))

    # Detail: all block-shuffle iterations
    print()
    print("  Block-shuffle detail (toutes iterations) :")
    for r in results:
        if r.get("error"):
            continue
        iters_str = " | ".join("%+.1f%%" % v for v in r["bshuf_all"])
        print("    %-24s : %s  -> median %+.1f%%" % (r["name"], iters_str, r["bshuf_median"]))

    print()
    print("  " + "=" * 70)
    if all_pass:
        print("  VERDICT GLOBAL : PASS — Aucun lookahead bias detecte")
    else:
        failing = [r["name"] for r in results if not r.get("error") and (not r["bshuf_pass"] or not r["shift_pass"])]
        if failing:
            print("  VERDICT GLOBAL : FAIL — Investiguer: %s" % ", ".join(failing))
        else:
            print("  VERDICT GLOBAL : PASS")
    print("  " + "=" * 70)

    elapsed = time.time() - t0
    print("\n  Temps total : %.1fs" % elapsed)


if __name__ == "__main__":
    main()
