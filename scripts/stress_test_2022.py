#!/usr/bin/env python3
"""
Stress Test Bear Market 2022 — 4 scénarios d'exécution sur les 5 stratégies.

Données disponibles pour 2022:
  - BTC, ETH, SOL, DOGE : full 2022 (données depuis 2019-2020)
  - BNB, XRP : données depuis Dec 2021 — utilise ce qui existe à partir du 2022-01-01

4 Scénarios:
  - Normal     : slippage 1 bps, funding 0.01%/8h, cooldown base
  - Stress A   : slippage 5 bps, funding 0.01%/8h, cooldown base
  - Stress B   : slippage 10 bps, funding 0.05%/8h, cooldown base
  - Stress C   : slippage 10 bps, funding 0.10%/8h, cooldown x2

Output: 3 tables
  1. Bear 2022 — tous scénarios
  2. Normal 2023-2025 vs Bear 2022
  3. Verdict: RESILIENT / VULNERABLE / CRITICAL

Pre-requisite: 2022 5m candles must be in the DB.
Run: python -m trading_bot.tools.candle_fetcher BTC ETH SOL XRP BNB --start 1640995200000 --end 1672531200000
(2022-01-01 to 2023-01-01 in ms)
"""
import sys
import os
import time
from dataclasses import replace

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

# ── Date ranges ──
BEAR_START = "2022-01-01"
BEAR_END = "2023-01-01"
NORMAL_START = "2023-01-01"
NORMAL_END = "2026-01-01"


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


# ── 4 Stress scenarios ──

SCENARIOS = {
    "Normal": lambda ec: ec,  # unchanged
    "Stress_A": lambda ec: replace(ec, slippage_sl_bps=5.0),
    "Stress_B": lambda ec: replace(ec, slippage_sl_bps=10.0, funding_rate_8h=0.0005),
    "Stress_C": lambda ec: replace(ec, slippage_sl_bps=10.0, funding_rate_8h=0.001,
                                   cooldown_bars=ec.cooldown_bars * 2),
}


def load_candles_from_db(coin, start_date, end_date):
    db = Database(DB_PATH)
    db.open()
    start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)

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


def check_and_fetch_2022_data():
    """Check if 2022 candles exist, offer to download if missing."""
    coins = list({s["coin"] for s in STRATEGIES})
    db = Database(DB_PATH)
    db.open()

    missing = []
    start_ms = int(pd.Timestamp(BEAR_START, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(BEAR_END, tz="UTC").timestamp() * 1000)

    for coin in coins:
        rows = db.fetchall(
            "SELECT COUNT(*) as cnt FROM candles "
            "WHERE coin=? AND interval='5m' AND time_open >= ? AND time_open <= ?",
            (coin, start_ms, end_ms),
        )
        count = rows[0]["cnt"] if rows else 0
        # 2022 = 365 days x 24h x 12 (5m bars/h) = ~105,120 bars expected
        if count < 50000:
            missing.append((coin, count))
    db.close()

    if missing:
        print("\n  DONNEES 2022 MANQUANTES:")
        for coin, cnt in missing:
            print("    %s: %d bougies 5m (besoin ~105,000)" % (coin, cnt))
        print("\n  Telechargement des bougies 5m 2022 depuis Binance...")
        print("  (Cela peut prendre quelques minutes)\n")

        from trading_bot.tools.candle_fetcher import fetch_candles
        import logging
        logging.basicConfig(level=logging.INFO)

        for coin, cnt in missing:
            print("  Fetching %s 2022 candles..." % coin)
            fetched = fetch_candles(coin, DB_PATH, start_ms=start_ms, end_ms=end_ms)
            print("    -> %d bougies telechargees" % fetched)

        print()
        return True

    return False


def run_backtest(df_1h, strat_cfg, bt, fe, exec_config):
    """Run a single backtest with given exec_config."""
    v2_cls = V2_STRATEGY_REGISTRY[strat_cfg["v2_class"]]
    strat = v2_cls(strat_cfg["v2_params"])
    signals = strat.generate_signals(df_1h)

    sig_filter = SIGNAL_FILTERS.get(strat_cfg["signal_filter"]) if strat_cfg["signal_filter"] else None
    if sig_filter:
        signals = sig_filter(signals, df_1h)

    metrics = bt.run(
        df_1h, signals,
        sl_pct=strat.sl_pct, tp_pct=strat.tp_pct,
        exec_config=exec_config,
        initial_equity=INITIAL_EQUITY,
    )
    return metrics


def main():
    t0 = time.time()
    config_path = os.path.join(os.path.dirname(__file__), "..", "crypto_bot", "config.yaml")
    fe = FeatureEngine(config_path)
    bt = SweepBacktester(config_path)

    print("=" * 120)
    print("  STRESS TEST BEAR MARKET 2022 — 4 scenarios x 5 strategies")
    print("=" * 120)

    # ── Check and fetch 2022 data if needed ──
    check_and_fetch_2022_data()

    # ── Load 2022 (bear) data ──
    print("\n-- Chargement donnees Bear 2022 (%s -> %s) --" % (BEAR_START, BEAR_END))
    bear_1h = {}
    for s in STRATEGIES:
        coin = s["coin"]
        if coin in bear_1h:
            continue
        df_5m = load_candles_from_db(coin, BEAR_START, BEAR_END)
        if df_5m is None or len(df_5m) < 1000:
            print("  %s: PAS ASSEZ DE DONNEES 2022 (%d bougies 5m)" % (
                coin, len(df_5m) if df_5m is not None else 0))
            continue
        df_1h = df_5m.resample("1h").agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum",
        }).dropna(subset=["open"])
        df_1h = fe.compute_all(df_1h)
        print("  %s: %s bougies 1h [%s -> %s]" % (
            coin, f"{len(df_1h):,}",
            df_1h.index[0].strftime("%Y-%m-%d"),
            df_1h.index[-1].strftime("%Y-%m-%d"),
        ))
        bear_1h[coin] = df_1h

    # ── Load 2023-2025 (normal) data ──
    print("\n-- Chargement donnees Normal 2023-2025 (%s -> %s) --" % (NORMAL_START, NORMAL_END))
    normal_1h = {}
    for s in STRATEGIES:
        coin = s["coin"]
        if coin in normal_1h:
            continue
        df_5m = load_candles_from_db(coin, NORMAL_START, NORMAL_END)
        if df_5m is None or len(df_5m) < 1000:
            print("  %s: PAS ASSEZ DE DONNEES 2023-2025" % coin)
            continue
        df_1h = df_5m.resample("1h").agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum",
        }).dropna(subset=["open"])
        df_1h = fe.compute_all(df_1h)
        print("  %s: %s bougies 1h [%s -> %s]" % (
            coin, f"{len(df_1h):,}",
            df_1h.index[0].strftime("%Y-%m-%d"),
            df_1h.index[-1].strftime("%Y-%m-%d"),
        ))
        normal_1h[coin] = df_1h

    # ═══════════════════════════════════════════════════════════════
    # TABLE 1: Bear 2022 — all scenarios
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("  TABLE 1: BEAR MARKET 2022 — TOUS SCENARIOS")
    print("=" * 120)
    print()
    print("  %-24s %5s %-10s %7s %7s %8s %6s" % (
        "Strategie", "Coin", "Scenario", "Return%", "Sharpe", "MaxDD%", "Trades"))
    print("  " + "-" * 78)

    # Store results for tables 2 and 3
    bear_results = {}  # (strat_name, scenario) -> metrics
    normal_results = {}  # strat_name -> metrics

    for s in STRATEGIES:
        coin = s["coin"]
        if coin not in bear_1h:
            for sc_name in SCENARIOS:
                print("  %-24s %5s %-10s   -- PAS DE DONNEES --" % (s["name"], coin, sc_name))
            continue

        df = bear_1h[coin]

        for sc_name, sc_fn in SCENARIOS.items():
            ec = sc_fn(s["exec_config"])
            metrics = run_backtest(df, s, bt, fe, ec)

            ret = metrics["total_return"] * 100
            sharpe = metrics["sharpe_ratio"]
            maxdd = metrics["max_drawdown"] * 100
            trades = metrics["nb_trades"]

            bear_results[(s["name"], sc_name)] = {
                "return": ret, "sharpe": sharpe, "maxdd": maxdd, "trades": trades,
            }

            print("  %-24s %5s %-10s %+6.1f%% %+6.2f %7.1f%% %6d" % (
                s["name"], coin, sc_name, ret, sharpe, maxdd, trades))

    # ═══════════════════════════════════════════════════════════════
    # Normal 2023-2025 results (for comparison)
    # ═══════════════════════════════════════════════════════════════
    for s in STRATEGIES:
        coin = s["coin"]
        if coin not in normal_1h:
            continue
        df = normal_1h[coin]
        metrics = run_backtest(df, s, bt, fe, s["exec_config"])
        normal_results[s["name"]] = {
            "return": metrics["total_return"] * 100,
            "sharpe": metrics["sharpe_ratio"],
            "maxdd": metrics["max_drawdown"] * 100,
            "trades": metrics["nb_trades"],
        }

    # ═══════════════════════════════════════════════════════════════
    # TABLE 2: Normal 2023-2025 vs Bear 2022
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("  TABLE 2: NORMAL 2023-2025 vs BEAR 2022 (scenario Normal)")
    print("=" * 120)
    print()
    print("  %-24s %5s | %8s %7s %7s | %8s %7s %7s | %8s" % (
        "Strategie", "Coin",
        "Norm_Ret", "Norm_SR", "Norm_DD",
        "Bear_Ret", "Bear_SR", "Bear_DD",
        "DeltaRet"))
    print("  " + "-" * 100)

    for s in STRATEGIES:
        name = s["name"]
        coin = s["coin"]
        nr = normal_results.get(name)
        br = bear_results.get((name, "Normal"))

        if not nr or not br:
            print("  %-24s %5s   -- DONNEES MANQUANTES --" % (name, coin))
            continue

        delta = br["return"] - nr["return"]
        print("  %-24s %5s | %+7.1f%% %+6.2f %6.1f%% | %+7.1f%% %+6.2f %6.1f%% | %+7.1f%%" % (
            name, coin,
            nr["return"], nr["sharpe"], nr["maxdd"],
            br["return"], br["sharpe"], br["maxdd"],
            delta))

    # ═══════════════════════════════════════════════════════════════
    # TABLE 3: Verdict
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("  TABLE 3: VERDICT PAR STRATEGIE")
    print("=" * 120)
    print()
    print("  %-24s %5s %9s %8s %12s" % (
        "Strategie", "Coin", "BearRet%", "BearDD%", "VERDICT"))
    print("  " + "-" * 65)

    for s in STRATEGIES:
        name = s["name"]
        coin = s["coin"]
        br = bear_results.get((name, "Normal"))

        if not br:
            print("  %-24s %5s     -- NO DATA --" % (name, coin))
            continue

        ret = br["return"]
        maxdd = br["maxdd"]

        if ret > -10 and maxdd < 20:
            verdict = "RESILIENT"
        elif ret >= -25 and maxdd < 35:
            verdict = "VULNERABLE"
        else:
            verdict = "CRITICAL"

        print("  %-24s %5s %+8.1f%% %7.1f%% %12s" % (
            name, coin, ret, maxdd, verdict))

    # ═══════════════════════════════════════════════════════════════
    # Worst-case summary (Stress C)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("  WORST CASE (Stress C) :")
    for s in STRATEGIES:
        br = bear_results.get((s["name"], "Stress_C"))
        if br:
            print("    %-24s : %+.1f%% (DD %.1f%%)" % (s["name"], br["return"], br["maxdd"]))
    print("-" * 60)

    elapsed = time.time() - t0
    print("\n  Temps total : %.1fs" % elapsed)


if __name__ == "__main__":
    main()
