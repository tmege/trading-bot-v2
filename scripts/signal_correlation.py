#!/usr/bin/env python3
"""
Matrice de corrélation des signaux — toutes les 5 stratégies.

Mesure la corrélation entre les signaux d'entrée (1/-1/0) des stratégies
sur la même période. Si les signaux sont trop corrélés, le portfolio n'est
pas aussi diversifié qu'il paraît.

Output:
  1. Matrice de corrélation (Pearson) entre signaux
  2. Overlap % : fraction des bars où 2+ stratégies sont actives simultanément
  3. Simultaneous entries : nombre de fois où N stratégies entrent en même temps
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

START = "2023-01-01"
END = "2026-01-01"
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trading_bot.db")


# ── Signal filters ──

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
        "name": "BTC InsideBar",
        "coin": "BTC",
        "v2_class": "StratInsideBarBreakout",
        "v2_params": {
            "vol_min": 0.8, "trend_filter": True,
            "atr_filter": True, "sl_pct": 2.5, "tp_pct": 4.5,
        },
        "signal_filter": "hours_8_20",
    },
    {
        "name": "SOL Breakout",
        "coin": "SOL",
        "v2_class": "StratBreakoutRelaxed",
        "v2_params": {
            "lookback": 14, "vol_breakout_min": 2.5,
            "sl_pct": 0.9, "tp_pct": 4.0,
        },
        "signal_filter": "anti_wick_40",
    },
    {
        "name": "ETH Breakout",
        "coin": "ETH",
        "v2_class": "StratBreakoutRelaxed",
        "v2_params": {
            "lookback": 35, "vol_breakout_min": 4.5,
            "sl_pct": 1.8, "tp_pct": 3.5,
        },
        "signal_filter": "anti_wick_60",
    },
    {
        "name": "XRP MeanRev",
        "coin": "XRP",
        "v2_class": "StratMeanReversionBB",
        "v2_params": {
            "rsi_oversold": 20, "rsi_overbought": 70,
            "bb_entry_low": 0.08, "bb_entry_high": 0.95,
            "sl_pct": 0.7, "tp_pct": 8.0,
        },
        "signal_filter": "anti_wick_50",
    },
    {
        "name": "BNB Breakout",
        "coin": "BNB",
        "v2_class": "StratBreakoutRelaxed",
        "v2_params": {
            "lookback": 32, "vol_breakout_min": 0.8,
            "sl_pct": 0.3, "tp_pct": 4.0,
        },
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


def main():
    t0 = time.time()
    config_path = os.path.join(os.path.dirname(__file__), "..", "crypto_bot", "config.yaml")
    fe = FeatureEngine(config_path)

    print("=" * 100)
    print("  MATRICE DE CORRELATION DES SIGNAUX — 5 strategies")
    print("=" * 100)

    # Load and prepare data
    print("\n-- Chargement et generation des signaux --")
    data_1h = {}
    for s in STRATEGIES:
        coin = s["coin"]
        if coin in data_1h:
            continue
        df_5m = load_candles_from_db(coin)
        if df_5m is None:
            continue
        df_1h = df_5m.resample("1h").agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum",
        }).dropna(subset=["open"])
        df_1h = fe.compute_all(df_1h)
        data_1h[coin] = df_1h

    # Generate signals for each strategy
    signal_series = {}
    for s in STRATEGIES:
        coin = s["coin"]
        if coin not in data_1h:
            continue
        df = data_1h[coin]
        v2_cls = V2_STRATEGY_REGISTRY[s["v2_class"]]
        strat = v2_cls(s["v2_params"])
        signals = strat.generate_signals(df)

        sig_filter = SIGNAL_FILTERS.get(s["signal_filter"]) if s["signal_filter"] else None
        if sig_filter:
            signals = sig_filter(signals, df)

        # Convert to binary active signal (1 if any signal, 0 otherwise)
        active = (signals != 0).astype(int)
        n_signals = active.sum()
        print("  %-16s (%s): %d signaux / %d bars (%.2f%%)" % (
            s["name"], coin, n_signals, len(df), n_signals / len(df) * 100))

        signal_series[s["name"]] = {
            "raw": signals,
            "active": active,
            "coin": coin,
        }

    strat_names = list(signal_series.keys())
    n_strats = len(strat_names)

    # ═══════════════════════════════════════════════════════════════
    # TABLE 1: Correlation matrix (active signals on common dates)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TABLE 1: CORRELATION DES SIGNAUX ACTIFS (Pearson)")
    print("  Note: strategies sur coins differents — correlation via timestamps communs")
    print("=" * 100)

    # Build matrix of active signals aligned on common index
    # Since strategies trade different coins, we align by timestamp
    common_idx = None
    for name in strat_names:
        idx = signal_series[name]["active"].index
        if common_idx is None:
            common_idx = idx
        else:
            common_idx = common_idx.intersection(idx)

    # Build DataFrame of active signals on common timestamps
    active_df = pd.DataFrame(index=common_idx)
    for name in strat_names:
        active_df[name] = signal_series[name]["active"].reindex(common_idx).fillna(0)

    corr_matrix = active_df.corr()

    # Print matrix
    header = "  %-16s" + " %14s" * n_strats
    print(header % tuple([""] + strat_names))
    print("  " + "-" * (16 + 15 * n_strats))

    for i, name_i in enumerate(strat_names):
        row_vals = []
        for j, name_j in enumerate(strat_names):
            v = corr_matrix.loc[name_i, name_j]
            if i == j:
                row_vals.append("     1.000")
            else:
                row_vals.append("    %+.3f" % v)
        print(("  %-16s" + " %14s" * n_strats) % tuple([name_i] + row_vals))

    # ═══════════════════════════════════════════════════════════════
    # TABLE 2: Simultaneous activity
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TABLE 2: ACTIVITE SIMULTANEE")
    print("=" * 100)

    # Count how many strategies are active at each bar
    active_count = active_df.sum(axis=1)
    total_bars = len(active_count)

    print()
    print("  Bars totales (common): %d" % total_bars)
    for n_active in range(n_strats + 1):
        count = (active_count == n_active).sum()
        pct = count / total_bars * 100
        bar = "#" * int(pct)
        print("  %d strat actives : %6d bars (%5.2f%%) %s" % (n_active, count, pct, bar))

    # Overlap: bars where 2+ strategies active
    overlap_bars = (active_count >= 2).sum()
    overlap_pct = overlap_bars / total_bars * 100
    print("\n  Overlap (2+ actives) : %d bars (%.2f%%)" % (overlap_bars, overlap_pct))

    # ═══════════════════════════════════════════════════════════════
    # TABLE 3: Pairwise overlap
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TABLE 3: OVERLAP PAR PAIRE (% des bars ou les 2 sont actives)")
    print("=" * 100)
    print()

    header = "  %-16s" + " %14s" * n_strats
    print(header % tuple([""] + strat_names))
    print("  " + "-" * (16 + 15 * n_strats))

    for i, name_i in enumerate(strat_names):
        row_vals = []
        for j, name_j in enumerate(strat_names):
            if i == j:
                row_vals.append("       ---")
            else:
                both_active = ((active_df[name_i] == 1) & (active_df[name_j] == 1)).sum()
                either_active = ((active_df[name_i] == 1) | (active_df[name_j] == 1)).sum()
                jaccard = both_active / either_active * 100 if either_active > 0 else 0
                row_vals.append("    %5.2f%%" % jaccard)
        print(("  %-16s" + " %14s" * n_strats) % tuple([name_i] + row_vals))

    # ═══════════════════════════════════════════════════════════════
    # Directional correlation (raw signals: +1, -1, 0)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TABLE 4: CORRELATION DIRECTIONNELLE (signaux bruts +1/-1/0)")
    print("=" * 100)

    raw_df = pd.DataFrame(index=common_idx)
    for name in strat_names:
        raw_df[name] = signal_series[name]["raw"].reindex(common_idx).fillna(0)

    dir_corr = raw_df.corr()

    print()
    header = "  %-16s" + " %14s" * n_strats
    print(header % tuple([""] + strat_names))
    print("  " + "-" * (16 + 15 * n_strats))

    for i, name_i in enumerate(strat_names):
        row_vals = []
        for j, name_j in enumerate(strat_names):
            v = dir_corr.loc[name_i, name_j]
            if i == j:
                row_vals.append("     1.000")
            else:
                row_vals.append("    %+.3f" % v)
        print(("  %-16s" + " %14s" * n_strats) % tuple([name_i] + row_vals))

    # Verdict
    print("\n  " + "=" * 60)
    max_corr = 0
    max_pair = ("", "")
    for i in range(n_strats):
        for j in range(i + 1, n_strats):
            v = abs(dir_corr.iloc[i, j])
            if v > max_corr:
                max_corr = v
                max_pair = (strat_names[i], strat_names[j])

    if max_corr < 0.3:
        print("  VERDICT: DIVERSIFIE — Correlation max %.3f (%s / %s)" % (max_corr, max_pair[0], max_pair[1]))
    elif max_corr < 0.6:
        print("  VERDICT: MODEREMENT CORRELE — Max %.3f (%s / %s)" % (max_corr, max_pair[0], max_pair[1]))
    else:
        print("  VERDICT: TROP CORRELE — Max %.3f (%s / %s) — revoir allocation" % (max_corr, max_pair[0], max_pair[1]))
    print("  " + "=" * 60)

    elapsed = time.time() - t0
    print("\n  Temps total : %.1fs" % elapsed)


if __name__ == "__main__":
    main()
