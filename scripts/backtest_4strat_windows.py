#!/usr/bin/env python3
"""
Backtest portfolio 4 strategies validees sur plusieurs fenetres temporelles.
Capital rebalance selon les scores de validation ($5,000 total).

Allocation :
  SOL  40.3%  $2,014  (overfit 0.93, consist 83%, bear resilient)
  BNB  31.3%  $1,566  (overfit 0.90, consist 67%, bear resilient)
  XRP  18.9%  $  946  (overfit 0.54, consist 67%, bear resilient)
  BTC   9.5%  $  474  (overfit 0.44, consist 83%, bear vulnerable)
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
}

# ── Validation-weighted allocation ──
VALIDATION = {
    "SOL": {"overfit": 0.930, "consistency": 0.83, "bear": 1.0},
    "BNB": {"overfit": 0.896, "consistency": 0.67, "bear": 1.0},
    "XRP": {"overfit": 0.541, "consistency": 0.67, "bear": 1.0},
    "BTC": {"overfit": 0.438, "consistency": 0.83, "bear": 0.5},
}

def _compute_weights():
    raw = {c: v["overfit"] * v["consistency"] * v["bear"] for c, v in VALIDATION.items()}
    total = sum(raw.values())
    return {c: s / total for c, s in raw.items()}

WEIGHTS = _compute_weights()

STRATEGIES = [
    {
        "name": "SOL", "coin": "SOL",
        "v2_class": "StratBreakoutRelaxed",
        "v2_params": {"lookback": 14, "vol_breakout_min": 2.5, "sl_pct": 0.9, "tp_pct": 4.0},
        "exec_config": ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=48),
        "signal_filter": "anti_wick_40",
        "capital": round(TOTAL_CAPITAL * WEIGHTS["SOL"], 2),
    },
    {
        "name": "BNB", "coin": "BNB",
        "v2_class": "StratBreakoutRelaxed",
        "v2_params": {"lookback": 32, "vol_breakout_min": 0.8, "sl_pct": 0.3, "tp_pct": 4.0},
        "exec_config": ExecConfig(equity_pct=0.35, leverage=5, cooldown_bars=3, max_hold_bars=48),
        "signal_filter": None,
        "capital": round(TOTAL_CAPITAL * WEIGHTS["BNB"], 2),
    },
    {
        "name": "XRP", "coin": "XRP",
        "v2_class": "StratMeanReversionBB",
        "v2_params": {"rsi_oversold": 20, "rsi_overbought": 70, "bb_entry_low": 0.08, "bb_entry_high": 0.95, "sl_pct": 0.7, "tp_pct": 8.0},
        "exec_config": ExecConfig(equity_pct=0.35, leverage=5, cooldown_bars=4, max_hold_bars=48),
        "signal_filter": "anti_wick_50",
        "capital": round(TOTAL_CAPITAL * WEIGHTS["XRP"], 2),
    },
    {
        "name": "BTC", "coin": "BTC",
        "v2_class": "StratInsideBarBreakout",
        "v2_params": {"vol_min": 0.8, "trend_filter": True, "atr_filter": True, "sl_pct": 2.5, "tp_pct": 4.5},
        "exec_config": ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=72),
        "signal_filter": "hours_8_20",
        "capital": round(TOTAL_CAPITAL * WEIGHTS["BTC"], 2),
    },
]

WINDOWS = [
    # 6 mois
    ("6M  2023-H1",  "2023-01-01", "2023-07-01"),
    ("6M  2023-H2",  "2023-07-01", "2024-01-01"),
    ("6M  2024-H1",  "2024-01-01", "2024-07-01"),
    ("6M  2024-H2",  "2024-07-01", "2025-01-01"),
    ("6M  2025-H1",  "2025-01-01", "2025-07-01"),
    ("6M  2025-H2",  "2025-07-01", "2026-01-01"),
    # 1 an
    ("1Y  2023",     "2023-01-01", "2024-01-01"),
    ("1Y  2024",     "2024-01-01", "2025-01-01"),
    ("1Y  2025",     "2025-01-01", "2026-01-01"),
    ("1Y  mid23-24", "2023-07-01", "2024-07-01"),
    ("1Y  mid24-25", "2024-07-01", "2025-07-01"),
    # 18 mois
    ("18M 23-mid24", "2023-01-01", "2024-07-01"),
    ("18M mid23-25", "2023-07-01", "2025-01-01"),
    ("18M 24-mid25", "2024-01-01", "2025-07-01"),
    ("18M mid24-26", "2024-07-01", "2026-01-01"),
    # 2 ans
    ("2Y  2023-25",  "2023-01-01", "2025-01-01"),
    ("2Y  mid23-mid25", "2023-07-01", "2025-07-01"),
    ("2Y  2024-26",  "2024-01-01", "2026-01-01"),
    # 3 ans
    ("3Y  2023-26",  "2023-01-01", "2026-01-01"),
]


def load_all_candles(coin):
    db = Database(DB_PATH)
    db.open()
    rows = db.fetchall(
        "SELECT time_open, open, high, low, close, volume FROM candles "
        "WHERE coin=? AND interval='5m' ORDER BY time_open",
        (coin,),
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

    print("=" * 140)
    print("  BACKTEST MULTI-PERIODES — 4 strategies validees, $5,000 rebalances")
    print("  Allocation : SOL %.0f%% ($%.0f) | BNB %.0f%% ($%.0f) | XRP %.0f%% ($%.0f) | BTC %.0f%% ($%.0f)" % (
        WEIGHTS["SOL"]*100, TOTAL_CAPITAL*WEIGHTS["SOL"],
        WEIGHTS["BNB"]*100, TOTAL_CAPITAL*WEIGHTS["BNB"],
        WEIGHTS["XRP"]*100, TOTAL_CAPITAL*WEIGHTS["XRP"],
        WEIGHTS["BTC"]*100, TOTAL_CAPITAL*WEIGHTS["BTC"],
    ))
    print("=" * 140)

    # Load all data once
    print("\n-- Chargement des donnees --")
    raw_5m = {}
    for s in STRATEGIES:
        coin = s["coin"]
        if coin not in raw_5m:
            raw_5m[coin] = load_all_candles(coin)
            if raw_5m[coin] is not None:
                print("  %s: %d bougies 5m [%s -> %s]" % (
                    coin, len(raw_5m[coin]),
                    raw_5m[coin].index[0].strftime("%Y-%m-%d"),
                    raw_5m[coin].index[-1].strftime("%Y-%m-%d"),
                ))

    # Header
    print("\n  %-16s | %6s | %7s | %8s | %8s | %6s | %s" % (
        "Fenetre", "Trades", "Return", "$/mois", "%/mois", "MaxDD", "Detail par strat"))
    print("  " + "-" * 120)

    all_6m_returns = []

    for wname, start, end in WINDOWS:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        months = (end_ts - start_ts).days / 30.44

        total_pnl = 0.0
        total_trades = 0
        detail_parts = []
        max_dd = 0.0
        skip = False
        total_invested = 0.0

        for s in STRATEGIES:
            coin = s["coin"]
            if raw_5m[coin] is None:
                skip = True
                break

            df_5m = raw_5m[coin].loc[start_ts:end_ts]
            if len(df_5m) < 100:
                skip = True
                break

            df_1h = df_5m.resample("1h").agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last", "volume": "sum",
            }).dropna(subset=["open"])

            if len(df_1h) < 50:
                skip = True
                break

            df_1h = fe.compute_all(df_1h)

            v2_cls = V2_STRATEGY_REGISTRY[s["v2_class"]]
            strat = v2_cls(s["v2_params"])
            signals = strat.generate_signals(df_1h)

            sig_filter = SIGNAL_FILTERS.get(s["signal_filter"]) if s["signal_filter"] else None
            if sig_filter:
                signals = sig_filter(signals, df_1h)

            capital = s["capital"]
            total_invested += capital

            metrics = bt.run(
                df_1h, signals,
                sl_pct=strat.sl_pct, tp_pct=strat.tp_pct,
                exec_config=s["exec_config"],
                initial_equity=capital,
            )

            pnl = metrics.get("dollar_pnl", 0)
            total_pnl += pnl
            total_trades += metrics["nb_trades"]
            dd = metrics["max_drawdown"] * 100
            if dd > max_dd:
                max_dd = dd

            ret_pct = metrics["total_return"] * 100
            detail_parts.append("%s:%+.0f%%" % (s["name"], ret_pct))

        if skip:
            print("  %-16s | %6s | %7s | %8s | %8s | %6s | DONNEES INSUFFISANTES" % (
                wname, "-", "-", "-", "-", "-"))
            continue

        total_return = (total_pnl / total_invested) * 100
        pnl_per_month = total_pnl / months if months > 0 else 0
        pct_per_month = total_return / months if months > 0 else 0

        if wname.startswith("6M"):
            all_6m_returns.append((wname.strip(), total_return, pct_per_month, max_dd))

        detail = "  ".join(detail_parts)
        print("  %-16s | %6d | %+6.1f%% | $%+7.0f | %+6.1f%% | %5.1f%% | %s" % (
            wname, total_trades, total_return, pnl_per_month, pct_per_month, max_dd, detail))

    # Summary for 6M windows
    if all_6m_returns:
        print("\n" + "=" * 140)
        print("  RESUME SEMESTRES (6M)")
        print("=" * 140)
        rets = [r[1] for r in all_6m_returns]
        monthly = [r[2] for r in all_6m_returns]
        dds = [r[3] for r in all_6m_returns]
        all_positive = all(r > 0 for r in rets)

        print("  Fenetres rentables : %d/%d %s" % (
            sum(1 for r in rets if r > 0), len(rets),
            "(TOUTES)" if all_positive else ""))
        print("  Pire semestre      : %s avec %+.1f%% (%+.1f%%/mois)" % (
            all_6m_returns[rets.index(min(rets))][0], min(rets), monthly[rets.index(min(rets))]))
        print("  Meilleur semestre  : %s avec %+.1f%% (%+.1f%%/mois)" % (
            all_6m_returns[rets.index(max(rets))][0], max(rets), monthly[rets.index(max(rets))]))
        print("  Rendement moyen    : %+.1f%%/semestre (%+.1f%%/mois)" % (
            np.mean(rets), np.mean(monthly)))
        print("  MaxDD moyen        : %.1f%%" % np.mean(dds))

    print("\n" + "=" * 140)
    elapsed = time.time() - t0
    print("  Temps total : %.1fs" % elapsed)
    print("=" * 140)


if __name__ == "__main__":
    main()
