#!/usr/bin/env python3
"""
Backtest portfolio rebalancé : 4 stratégies validées (ETH retirée).

Redistribution du capital $5,000 basée sur les scores de validation :
  - Overfit score (walk-forward)
  - OOS consistency
  - Bear market resilience (2022)

Comparaison : ancien portfolio 5 strats vs nouveau 4 strats rebalancé.
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

TOTAL_CAPITAL = 5000.0
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
}


# ══════════════════════════════════════════════════════════════════
# Validation scores (from walk-forward + stress test 2022)
# ══════════════════════════════════════════════════════════════════
# Overfit score : mean(oos_sharpe) / mean(is_sharpe) — higher = better
# OOS consistency : fraction of OOS windows profitable
# Bear resilience : 1.0 if RESILIENT (bear return > -10%), 0.5 if VULNERABLE

VALIDATION = {
    "SOL": {"overfit": 0.930, "consistency": 0.83, "bear": 1.0},
    "BNB": {"overfit": 0.896, "consistency": 0.67, "bear": 1.0},
    "XRP": {"overfit": 0.541, "consistency": 0.67, "bear": 1.0},
    "BTC": {"overfit": 0.438, "consistency": 0.83, "bear": 0.5},
}


def compute_weights():
    """Compute capital weights from validation composite scores."""
    raw = {}
    for coin, v in VALIDATION.items():
        # Composite = overfit * consistency * bear_factor
        raw[coin] = v["overfit"] * v["consistency"] * v["bear"]

    total = sum(raw.values())
    weights = {coin: score / total for coin, score in raw.items()}
    return weights, raw


# ══════════════════════════════════════════════════════════════════
# Strategy configurations
# ══════════════════════════════════════════════════════════════════

def build_strategies(weights):
    """Build strategy list with rebalanced capital."""
    return [
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
            "capital": round(TOTAL_CAPITAL * weights["SOL"], 2),
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
            "capital": round(TOTAL_CAPITAL * weights["BNB"], 2),
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
            "capital": round(TOTAL_CAPITAL * weights["XRP"], 2),
        },
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
            "capital": round(TOTAL_CAPITAL * weights["BTC"], 2),
        },
    ]


# ══════════════════════════════════════════════════════════════════
# Old portfolio (5 strats × $1000 equal-weight)
# ══════════════════════════════════════════════════════════════════

OLD_STRATEGIES = [
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
        "capital": 1000.0,
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
        "capital": 1000.0,
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
        "signal_filter": None,
        "capital": 1000.0,
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
        "capital": 1000.0,
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
        "capital": 1000.0,
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


def run_portfolio(strategies, data_1h, bt, label):
    """Run backtest for a portfolio of strategies. Returns results list."""
    results = []
    total_invested = sum(s["capital"] for s in strategies)
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0

    print("\n  %-24s %5s %8s %7s %6s %7s %10s %10s %7s %5s" % (
        "Strategie", "Coin", "Capital", "Return%", "Trades", "WR", "$PnL", "Final$", "MaxDD%", "SR"))
    print("  " + "-" * 102)

    for s in strategies:
        coin = s["coin"]
        if coin not in data_1h:
            print("  %-24s %5s  -- PAS DE DONNEES --" % (s["name"], coin))
            continue

        df = data_1h[coin]
        v2_cls = V2_STRATEGY_REGISTRY[s["v2_class"]]
        strat = v2_cls(s["v2_params"])
        signals = strat.generate_signals(df)

        sig_filter_name = s.get("signal_filter")
        sig_filter = SIGNAL_FILTERS.get(sig_filter_name) if sig_filter_name else None
        if sig_filter:
            signals = sig_filter(signals, df)

        capital = s["capital"]
        metrics = bt.run(
            df, signals,
            sl_pct=strat.sl_pct, tp_pct=strat.tp_pct,
            exec_config=s["exec_config"],
            initial_equity=capital,
        )

        pnl = metrics.get("dollar_pnl", 0)
        final = metrics.get("final_equity", capital)
        ret = metrics["total_return"] * 100
        wr = metrics["win_rate"] * 100
        dd = metrics["max_drawdown"] * 100
        nb = metrics["nb_trades"]
        sharpe = metrics["sharpe_ratio"]
        wins = int(wr / 100 * nb)

        total_pnl += pnl
        total_trades += nb
        total_wins += wins

        pf = metrics["profit_factor"]
        pf_str = "%.2f" % pf if pf < 100 else "INF"

        print("  %-24s %5s %8.0f %+6.1f%% %6d %5.0f%% %+10.2f %10.2f %6.1f%% %+4.2f" % (
            s["name"], coin, capital, ret, nb, wr, pnl, final, dd, sharpe))

        results.append({
            "name": s["name"], "coin": coin, "capital": capital,
            "pnl": pnl, "final": final, "return_pct": ret,
            "trades": nb, "sharpe": sharpe, "maxdd": dd, "win_rate": wr,
        })

    total_final = total_invested + total_pnl
    total_ret = total_pnl / total_invested * 100
    combined_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0

    print("  " + "-" * 102)
    print("  %-24s %5s %8.0f %+6.1f%% %6d %5.0f%% %+10.2f %10.2f" % (
        "TOTAL", "", total_invested, total_ret, total_trades, combined_wr,
        total_pnl, total_final))

    return {
        "label": label,
        "invested": total_invested,
        "pnl": total_pnl,
        "final": total_final,
        "return_pct": total_ret,
        "trades": total_trades,
        "win_rate": combined_wr,
        "details": results,
    }


def main():
    t0 = time.time()
    config_path = os.path.join(os.path.dirname(__file__), "..", "crypto_bot", "config.yaml")
    fe = FeatureEngine(config_path)
    bt = SweepBacktester(config_path)

    # Compute weights
    weights, raw_scores = compute_weights()

    print("=" * 110)
    print("  BACKTEST REBALANCE — 4 strategies validees, $%s total" % f"{TOTAL_CAPITAL:,.0f}")
    print("  Periode: %s -> %s" % (START, END))
    print("=" * 110)

    # Show allocation
    print("\n  ALLOCATION BASEE SUR LES SCORES DE VALIDATION")
    print("  %-24s %8s %8s %8s %8s %8s %8s" % (
        "Strategie", "Overfit", "Consist", "Bear", "Score", "Poids%", "Capital$"))
    print("  " + "-" * 78)

    for coin in ["SOL", "BNB", "XRP", "BTC"]:
        v = VALIDATION[coin]
        print("  %-24s %7.3f %7.0f%% %7.1f %8.3f %7.1f%% %8.0f" % (
            coin, v["overfit"], v["consistency"] * 100, v["bear"],
            raw_scores[coin], weights[coin] * 100,
            TOTAL_CAPITAL * weights[coin]))

    print("  " + "-" * 78)
    print("  %-24s %8s %8s %8s %8.3f %7.1f%% %8.0f" % (
        "TOTAL", "", "", "", sum(raw_scores.values()), 100.0, TOTAL_CAPITAL))

    # Load data
    print("\n-- Chargement des donnees --")
    data_1h = {}
    all_coins = set()
    for s in build_strategies(weights) + OLD_STRATEGIES:
        all_coins.add(s["coin"])

    for coin in sorted(all_coins):
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

    # ═══════════════════════════════════════════════════════════════
    # ANCIEN PORTFOLIO : 5 strats × $1,000 equal-weight
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  ANCIEN PORTFOLIO — 5 strats x $1,000 equal-weight")
    print("=" * 110)
    old = run_portfolio(OLD_STRATEGIES, data_1h, bt, "Ancien (5x$1000)")

    # ═══════════════════════════════════════════════════════════════
    # NOUVEAU PORTFOLIO : 4 strats, weighted by validation
    # ═══════════════════════════════════════════════════════════════
    new_strats = build_strategies(weights)
    print("\n" + "=" * 110)
    print("  NOUVEAU PORTFOLIO — 4 strats, weighted by validation score")
    print("=" * 110)
    new = run_portfolio(new_strats, data_1h, bt, "Nouveau (4 weighted)")

    # ═══════════════════════════════════════════════════════════════
    # COMPARAISON
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  COMPARAISON ANCIEN vs NOUVEAU")
    print("=" * 110)
    print()
    print("  %-28s %12s %12s %12s" % ("Metrique", "Ancien", "Nouveau", "Delta"))
    print("  " + "-" * 66)
    print("  %-28s %12.0f %12.0f %+11.0f" % ("Capital investi $", old["invested"], new["invested"], new["invested"] - old["invested"]))
    print("  %-28s %+11.2f %+11.2f %+11.2f" % ("PnL total $", old["pnl"], new["pnl"], new["pnl"] - old["pnl"]))
    print("  %-28s %12.2f %12.2f %+11.2f" % ("Capital final $", old["final"], new["final"], new["final"] - old["final"]))
    print("  %-28s %+10.1f%% %+10.1f%% %+10.1fpp" % (
        "Rendement %", old["return_pct"], new["return_pct"], new["return_pct"] - old["return_pct"]))
    print("  %-28s %12d %12d %+11d" % ("Trades total", old["trades"], new["trades"], new["trades"] - old["trades"]))
    print("  %-28s %10.1f%% %10.1f%% %+10.1fpp" % (
        "Win rate %", old["win_rate"], new["win_rate"], new["win_rate"] - old["win_rate"]))

    # Per-strategy contribution to new portfolio
    print("\n  CONTRIBUTION PAR STRATEGIE (nouveau) :")
    for r in new["details"]:
        contrib = r["pnl"] / new["pnl"] * 100 if new["pnl"] != 0 else 0
        print("    %-24s : $%+8.2f (%5.1f%% du PnL) — capital $%.0f, ret %+.1f%%" % (
            r["name"], r["pnl"], contrib, r["capital"], r["return_pct"]))

    elapsed = time.time() - t0
    print("\n" + "=" * 110)
    print("  Temps total : %.1fs" % elapsed)
    print("=" * 110)


if __name__ == "__main__":
    main()
