#!/usr/bin/env python3
"""
Backtest portfolio complet : 5 stratégies actives sur 3 ans (2023-01-01 → 2026-01-01).
Chaque stratégie tourne avec $1,000 isolés (comme en paper mode).
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


# ── 5 stratégies actives (même config que bot_config.json + STRATEGY_MAP) ──

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


def main():
    t0 = time.time()
    config_path = os.path.join(os.path.dirname(__file__), "..", "crypto_bot", "config.yaml")
    fe = FeatureEngine(config_path)
    bt = SweepBacktester(config_path)

    print("=" * 110)
    print("  BACKTEST 5 STRATÉGIES — %s → %s ($1,000/strat, $5,000 total)" % (START, END))
    print("=" * 110)

    # ── Load & prepare data ──
    print("\n-- Chargement des données (DB SQLite) --")
    data_1h = {}
    for s in STRATEGIES:
        coin = s["coin"]
        if coin in data_1h:
            continue
        df_5m = load_candles_from_db(coin)
        if df_5m is None:
            print("  ERREUR: pas de données pour %s" % coin)
            continue
        df_1h = df_5m.resample("1h").agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum",
        }).dropna(subset=["open"])
        df_1h = fe.compute_all(df_1h)
        print("  %s: %s bougies 1h [%s → %s]" % (
            coin, f"{len(df_1h):,}",
            df_1h.index[0].strftime("%Y-%m-%d"),
            df_1h.index[-1].strftime("%Y-%m-%d"),
        ))
        data_1h[coin] = df_1h

    # ── Individual results ──
    print("\n" + "=" * 110)
    print("  PERFORMANCE INDIVIDUELLE (chaque strat sur $1,000 isolés)")
    print("=" * 110)
    print("  %-22s %5s %6s %5s %7s %10s %10s %7s %5s" % (
        "Stratégie", "Coin", "Trades", "WR", "Sharpe", "$PnL", "Final$", "MaxDD", "PF"))
    print("  " + "-" * 85)

    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    total_losses = 0
    all_trade_pnls = []
    strat_results = []

    for s in STRATEGIES:
        coin = s["coin"]
        if coin not in data_1h:
            print("  %-22s %5s  -- PAS DE DONNÉES --" % (s["name"], coin))
            continue

        df = data_1h[coin]
        v2_cls = V2_STRATEGY_REGISTRY[s["v2_class"]]
        strat = v2_cls(s["v2_params"])
        signals = strat.generate_signals(df)

        sig_filter = SIGNAL_FILTERS.get(s["signal_filter"]) if s["signal_filter"] else None
        if sig_filter:
            signals = sig_filter(signals, df)

        metrics = bt.run(
            df, signals,
            sl_pct=strat.sl_pct, tp_pct=strat.tp_pct,
            exec_config=s["exec_config"],
            initial_equity=INITIAL_EQUITY,
        )

        pnl = metrics.get("dollar_pnl", 0)
        final = metrics.get("final_equity", INITIAL_EQUITY)
        wr = metrics["win_rate"] * 100
        dd = metrics["max_drawdown"] * 100
        pf = metrics["profit_factor"]
        nb = metrics["nb_trades"]
        wins = int(wr / 100 * nb)
        losses = nb - wins

        total_pnl += pnl
        total_trades += nb
        total_wins += wins
        total_losses += losses

        if "trades_detail" in metrics:
            all_trade_pnls.extend([t["pnl_pct"] for t in metrics["trades_detail"]])

        pf_str = "%.2f" % pf if pf < 100 else "INF"
        print("  %-22s %5s %6d %4.0f%% %+7.2f %+10.2f %10.2f %6.1f%% %5s" % (
            s["name"], coin, nb, wr, metrics["sharpe_ratio"], pnl, final, dd, pf_str))

        strat_results.append({
            "name": s["name"], "coin": coin, "pnl": pnl, "final": final,
            "return_pct": metrics["total_return"] * 100, "trades": nb,
        })

    # ── Combined portfolio ──
    print("\n" + "=" * 110)
    print("  PORTFOLIO COMBINÉ — 5 stratégies × $1,000 = $5,000 investis")
    print("=" * 110)

    total_invested = INITIAL_EQUITY * len(STRATEGIES)
    total_final = total_invested + total_pnl
    total_return_on_invested = (total_pnl / total_invested) * 100
    combined_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0

    print("\n  Capital investi    : $%.0f (%d x $%.0f)" % (total_invested, len(STRATEGIES), INITIAL_EQUITY))
    print("  PnL total          : $%+.2f" % total_pnl)
    print("  Capital final      : $%.2f" % total_final)
    print("  Rendement réel     : %+.1f%% (sur $%.0f investis)" % (total_return_on_invested, total_invested))
    print("  Trades total       : %d (W:%d / L:%d)" % (total_trades, total_wins, total_losses))
    print("  Win rate global    : %.1f%%" % combined_wr)

    # Per-strategy breakdown
    print("\n  Détail par stratégie :")
    for r in strat_results:
        print("    %-22s : $%+.2f (%+.1f%%) — %d trades" % (
            r["name"], r["pnl"], r["return_pct"], r["trades"]))

    # ── Additive sum (comme backtest_portfolio_v2 le calcule — pour comparaison) ──
    additive_return = (total_pnl / INITIAL_EQUITY) * 100
    print("\n  ⚠  Rendement 'additif' (PnL total / $1000) : %+.1f%%" % additive_return)
    print("     C'est le chiffre trompeur — il divise le PnL de 5 strats par $1000 au lieu de $5000")

    elapsed = time.time() - t0
    print("\n" + "=" * 110)
    print("  Temps total : %.1fs" % elapsed)
    print("=" * 110)


if __name__ == "__main__":
    main()
