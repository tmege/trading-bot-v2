#!/usr/bin/env python3
"""
Walk-Forward Rolling — 4 stratégies PASS (BTC, SOL, XRP, BNB).

Fenêtres ancrées expansives avec gap de 2 semaines :
  Window 1: train [0 : 50%],        gap 336 bars, test [50%+336 : 50%+336+step]
  Window 2: train [0 : 50%+step],   gap 336 bars, test [50%+2*step+336 : ...]
  ...

Métriques clés :
  - overfit_score = mean(oos_sharpe) / mean(is_sharpe) — alerte si < 0.5
  - decay_pct = (mean_oos - mean_is) / mean_is × 100
  - oos_consistency = fraction de fenêtres OOS > 0
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

GAP_BARS = 336       # 14 days x 24h
MIN_WINDOWS = 5
MIN_OOS_BARS = 100


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

# Only the 4 PASS strategies (excluding ETH)
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


def walk_forward_rolling(df_1h, strat_cfg, bt):
    """Run anchored expanding walk-forward with gap."""
    v2_cls = V2_STRATEGY_REGISTRY[strat_cfg["v2_class"]]
    ec = strat_cfg["exec_config"]
    sig_filter = SIGNAL_FILTERS.get(strat_cfg["signal_filter"]) if strat_cfg["signal_filter"] else None

    total_bars = len(df_1h)
    initial_train_end = int(total_bars * 0.50)
    remaining = total_bars - initial_train_end
    step_bars = max(MIN_OOS_BARS, (remaining - GAP_BARS) // (MIN_WINDOWS + 1))

    windows = []
    w = 0

    while True:
        train_end = initial_train_end + w * step_bars
        oos_start = train_end + GAP_BARS
        oos_end = oos_start + step_bars

        if train_end > total_bars or oos_start >= total_bars:
            break
        if oos_end > total_bars:
            oos_end = total_bars
        if oos_end - oos_start < MIN_OOS_BARS:
            break

        is_df = df_1h.iloc[:train_end]
        oos_df = df_1h.iloc[oos_start:oos_end]

        # IS backtest
        strat_is = v2_cls(strat_cfg["v2_params"])
        sig_is = strat_is.generate_signals(is_df)
        if sig_filter:
            sig_is = sig_filter(sig_is, is_df)
        m_is = bt.run(is_df, sig_is, sl_pct=strat_is.sl_pct, tp_pct=strat_is.tp_pct,
                      exec_config=ec, initial_equity=INITIAL_EQUITY)

        # OOS backtest
        strat_oos = v2_cls(strat_cfg["v2_params"])
        sig_oos = strat_oos.generate_signals(oos_df)
        if sig_filter:
            sig_oos = sig_filter(sig_oos, oos_df)
        m_oos = bt.run(oos_df, sig_oos, sl_pct=strat_oos.sl_pct, tp_pct=strat_oos.tp_pct,
                       exec_config=ec, initial_equity=INITIAL_EQUITY)

        is_ret = m_is["total_return"] * 100
        oos_ret = m_oos["total_return"] * 100
        is_sharpe = m_is["sharpe_ratio"]
        oos_sharpe = m_oos["sharpe_ratio"]

        windows.append({
            "window": w + 1,
            "train_bars": train_end,
            "oos_bars": oos_end - oos_start,
            "oos_range": "%s — %s" % (
                oos_df.index[0].strftime("%Y-%m-%d"),
                oos_df.index[-1].strftime("%Y-%m-%d")),
            "is_return": is_ret,
            "oos_return": oos_ret,
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "is_trades": m_is["nb_trades"],
            "oos_trades": m_oos["nb_trades"],
        })
        w += 1

    return windows


def main():
    t0 = time.time()
    config_path = os.path.join(os.path.dirname(__file__), "..", "crypto_bot", "config.yaml")
    fe = FeatureEngine(config_path)
    bt = SweepBacktester(config_path)

    print("=" * 130)
    print("  WALK-FORWARD ROLLING — 4 strategies PASS")
    print("  Gap=%d bars (2 sem) | Min %d fenetres OOS | Min %d bars/fenetre" % (
        GAP_BARS, MIN_WINDOWS, MIN_OOS_BARS))
    print("=" * 130)

    # Load data
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

    # Run walk-forward for each strategy
    all_results = []

    for s in STRATEGIES:
        coin = s["coin"]
        if coin not in data_1h:
            continue

        df = data_1h[coin]
        print("\n" + "=" * 130)
        print("  %s (%s)" % (s["name"], coin))
        print("=" * 130)

        windows = walk_forward_rolling(df, s, bt)

        if not windows:
            print("  Pas assez de donnees pour walk-forward")
            continue

        # Detail table
        print()
        print("  %3s %11s %-23s %8s %8s %8s %8s %5s %5s" % (
            "Win", "TrainBars", "OOS Range", "IS_Ret%", "OOS_Ret%", "IS_SR", "OOS_SR", "IS_T", "OOS_T"))
        print("  " + "-" * 100)

        for w in windows:
            oos_marker = " +" if w["oos_return"] > 0 else " -"
            print("  %3d %11d %-23s %+7.1f%% %+7.1f%%%s %+7.2f %+7.2f %5d %5d" % (
                w["window"], w["train_bars"], w["oos_range"],
                w["is_return"], w["oos_return"], oos_marker,
                w["is_sharpe"], w["oos_sharpe"],
                w["is_trades"], w["oos_trades"]))

        # Aggregate metrics
        is_rets = [w["is_return"] for w in windows]
        oos_rets = [w["oos_return"] for w in windows]
        is_sharpes = [w["is_sharpe"] for w in windows]
        oos_sharpes = [w["oos_sharpe"] for w in windows]

        mean_is_ret = float(np.mean(is_rets))
        mean_oos_ret = float(np.mean(oos_rets))
        mean_is_sr = float(np.mean(is_sharpes))
        mean_oos_sr = float(np.mean(oos_sharpes))

        overfit_score = mean_oos_sr / mean_is_sr if abs(mean_is_sr) > 0.01 else 0
        decay_pct = (mean_oos_ret - mean_is_ret) / abs(mean_is_ret) * 100 if abs(mean_is_ret) > 0.01 else 0
        consistency = sum(1 for w in windows if w["oos_return"] > 0) / len(windows)

        print()
        print("  Aggregate (%d fenetres) :" % len(windows))
        print("    Mean IS return   : %+.2f%%" % mean_is_ret)
        print("    Mean OOS return  : %+.2f%%" % mean_oos_ret)
        print("    Mean IS Sharpe   : %+.3f" % mean_is_sr)
        print("    Mean OOS Sharpe  : %+.3f" % mean_oos_sr)
        print("    Overfit score    : %.3f%s" % (
            overfit_score, "  *** ALERTE OVERFIT ***" if overfit_score < 0.5 else "  OK"))
        print("    Decay            : %+.1f%%" % decay_pct)
        print("    OOS consistency  : %.0f%% (%d/%d fenetres profitables)" % (
            consistency * 100, sum(1 for w in windows if w["oos_return"] > 0), len(windows)))

        # Verdict
        if overfit_score >= 0.5 and consistency >= 0.6:
            verdict = "GENERALISE"
        elif overfit_score >= 0.3 and consistency >= 0.4:
            verdict = "ACCEPTABLE"
        elif overfit_score >= 0 and consistency >= 0.2:
            verdict = "FRAGILE"
        else:
            verdict = "OVERFIT"
        print("    VERDICT          : %s" % verdict)

        all_results.append({
            "name": s["name"], "coin": coin,
            "n_windows": len(windows),
            "mean_is_ret": mean_is_ret, "mean_oos_ret": mean_oos_ret,
            "overfit_score": overfit_score, "decay_pct": decay_pct,
            "consistency": consistency, "verdict": verdict,
        })

    # Summary table
    print("\n" + "=" * 130)
    print("  RESUME WALK-FORWARD")
    print("=" * 130)
    print()
    print("  %-24s %5s %5s %9s %9s %8s %8s %7s %12s" % (
        "Strategie", "Coin", "Wins", "IS_Ret%", "OOS_Ret%", "OvfScore", "Decay%", "Consist", "VERDICT"))
    print("  " + "-" * 100)

    for r in all_results:
        print("  %-24s %5s %5d %+8.1f%% %+8.1f%% %+7.3f %+7.1f%% %5.0f%% %12s" % (
            r["name"], r["coin"], r["n_windows"],
            r["mean_is_ret"], r["mean_oos_ret"],
            r["overfit_score"], r["decay_pct"],
            r["consistency"] * 100, r["verdict"]))

    elapsed = time.time() - t0
    print("\n  Temps total : %.1fs" % elapsed)


if __name__ == "__main__":
    main()
