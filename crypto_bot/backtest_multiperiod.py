#!/usr/bin/env python3
"""
backtest_multiperiod.py — Backtest du portfolio sur 6 periodes croissantes.

Periodes (toutes finissent au meme point) :
  6 mois  : 2025-07 -> 2026-01
  1 an    : 2025-01 -> 2026-01
  1.5 ans : 2024-07 -> 2026-01
  2 ans   : 2024-01 -> 2026-01
  2.5 ans : 2023-07 -> 2026-01
  3 ans   : 2023-01 -> 2026-01

Chaque asset est teste individuellement puis combine en portfolio.
Monte Carlo sur la periode 3 ans.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from exec_config import ExecConfig
from modules.feature_engine import FeatureEngine
from modules.strategies import V2_STRATEGY_REGISTRY
from sweep_runner import SweepBacktester

INITIAL_EQUITY = 1000.0

PERIODS = [
    ("6 mois",   "2025-07-01", "2026-01-01"),
    ("1 an",     "2025-01-01", "2026-01-01"),
    ("1.5 ans",  "2024-07-01", "2026-01-01"),
    ("2 ans",    "2024-01-01", "2026-01-01"),
    ("2.5 ans",  "2023-07-01", "2026-01-01"),
    ("3 ans",    "2023-01-01", "2026-01-01"),
]

# Signal filters
def filter_hours_8_20():
    blocked = list(range(0, 8)) + list(range(21, 24))
    def _filter(signals, df):
        hours = df.index.hour
        mask = pd.Series(True, index=df.index)
        for h in blocked:
            mask = mask & (hours != h)
        return signals.where(mask, 0)
    return _filter

def filter_anti_wick(max_wick_ratio=0.5):
    def _filter(signals, df):
        body = (df["close"] - df["open"]).abs()
        total_range = df["high"] - df["low"]
        wick_ratio = 1 - body / total_range.replace(0, 1)
        return signals.where(wick_ratio < max_wick_ratio, 0)
    return _filter

# Portfolio definition
PORTFOLIO = {
    "BTC": {
        "name": "BTC InsideBar + hours 8-20",
        "strat_class": "StratInsideBarBreakout",
        "params": {
            "vol_min": 1.5, "trend_filter": True,
            "atr_filter": True, "sl_pct": 1.5, "tp_pct": 3.0,
        },
        "ec": ExecConfig(
            equity_pct=0.20, leverage=5,
            cooldown_bars=4, max_hold_bars=72,
        ),
        "signal_filter": filter_hours_8_20(),
    },
    "SOL": {
        "name": "SOL Breakout + anti-wick 40%",
        "strat_class": "StratBreakoutRelaxed",
        "params": {
            "lookback": 15, "vol_breakout_min": 2.5,
            "use_compression": False, "sl_pct": 1.0, "tp_pct": 4.0,
        },
        "ec": ExecConfig(
            equity_pct=0.30, leverage=5,
            cooldown_bars=4, max_hold_bars=48,
        ),
        "signal_filter": filter_anti_wick(0.4),
    },
    "ETH": {
        "name": "ETH Breakout + anti-wick 60%",
        "strat_class": "StratBreakoutRelaxed",
        "params": {
            "lookback": 30, "vol_breakout_min": 4.0,
            "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0,
        },
        "ec": ExecConfig(
            equity_pct=0.20, leverage=5,
            cooldown_bars=4, max_hold_bars=48,
        ),
        "signal_filter": filter_anti_wick(0.6),
    },
}


def load_asset(symbol):
    fe = FeatureEngine()
    df_5m = pd.read_parquet("data/%s_USDT_5m_ohlcv.parquet" % symbol)
    df_5m = df_5m[~df_5m.index.duplicated(keep="first")]
    df_1h = df_5m.resample("1h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["open"])
    df_1h = fe.compute_all(df_1h)
    return df_1h


def slice_window(df, start, end):
    mask = (df.index >= pd.Timestamp(start, tz="UTC")) & \
           (df.index < pd.Timestamp(end, tz="UTC"))
    return df.loc[mask]


def run_asset(data, bt, asset, cfg, start, end):
    cls = V2_STRATEGY_REGISTRY[cfg["strat_class"]]
    strat = cls(cfg["params"])
    df_w = slice_window(data[asset], start, end)
    if len(df_w) < 50:
        return {"nb_trades": 0, "win_rate": 0, "sharpe_ratio": 0,
                "dollar_pnl": 0, "final_equity": INITIAL_EQUITY,
                "max_drawdown": 0, "trades_detail": []}

    signals = strat.generate_signals(df_w)
    if cfg.get("signal_filter") is not None:
        signals = cfg["signal_filter"](signals, df_w)

    return bt.run(df_w, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                  exec_config=cfg["ec"], initial_equity=INITIAL_EQUITY)


def main():
    t0 = time.time()

    print("=" * 110)
    print("  BACKTEST MULTI-PERIODE — Portfolio 3 assets sur $%d" % int(INITIAL_EQUITY))
    print("=" * 110)

    # Print configs
    print("\n  PORTFOLIO:")
    for asset in ["BTC", "SOL", "ETH"]:
        cfg = PORTFOLIO[asset]
        p = cfg["params"]
        ec = cfg["ec"]
        print("    %s: %s | eq=%.0f%% lev=%d | %s" % (
            asset, cfg["name"], ec.equity_pct * 100, ec.leverage,
            " ".join("%s=%s" % (k, v) for k, v in p.items())))

    # Load data
    print("\n-- Chargement des donnees --")
    data = {}
    for sym in ["BTC", "SOL", "ETH"]:
        data[sym] = load_asset(sym)
        print("  %s: %d barres [%s -> %s]" % (
            sym, len(data[sym]),
            data[sym].index[0].date(), data[sym].index[-1].date()))

    bt = SweepBacktester()

    # ── Per-asset results table ──
    print("\n" + "=" * 110)
    print("  RESULTATS PAR ASSET ET PAR PERIODE")
    print("=" * 110)

    for asset in ["BTC", "SOL", "ETH"]:
        cfg = PORTFOLIO[asset]
        print("\n  -- %s: %s --" % (asset, cfg["name"]))
        print("  %-10s %6s %5s %7s %9s %9s %7s %6s" %
              ("Periode", "Trades", "WR", "Sharpe", "$PnL", "Final$", "MaxDD", "PF"))
        print("  " + "-" * 70)

        for pname, start, end in PERIODS:
            m = run_asset(data, bt, asset, cfg, start, end)
            nt = m["nb_trades"]
            wr = m["win_rate"] * 100
            sr = m.get("sharpe_ratio", 0)
            pnl = m.get("dollar_pnl", 0)
            fe_val = m.get("final_equity", INITIAL_EQUITY)
            dd = m["max_drawdown"] * 100
            pf = m.get("profit_factor", 0)
            print("  %-10s %6d %4.0f%% %+7.2f %+9.2f %9.2f %6.1f%% %6.2f" %
                  (pname, nt, wr, sr, pnl, fe_val, dd, pf))

    # ── Combined portfolio ──
    print("\n" + "=" * 110)
    print("  PORTFOLIO COMBINE (3 assets sur $%d chacun = $%d total)" %
          (int(INITIAL_EQUITY), int(INITIAL_EQUITY * 3)))
    print("=" * 110)
    print("  %-10s %6s %5s %7s %9s %9s %7s  %s" %
          ("Periode", "Trades", "WR", "Sharpe", "$PnL", "Final$", "MaxDD", "Detail PnL"))
    print("  " + "-" * 105)

    all_3y_pnls = []

    for pname, start, end in PERIODS:
        total_pnl = 0
        total_trades = 0
        total_wins = 0
        max_dd = 0
        all_pnls = []
        asset_pnls = {}

        for asset in ["BTC", "SOL", "ETH"]:
            cfg = PORTFOLIO[asset]
            m = run_asset(data, bt, asset, cfg, start, end)
            pnl = m.get("dollar_pnl", 0)
            nt = m["nb_trades"]
            wr = m["win_rate"]
            dd = m["max_drawdown"]

            total_pnl += pnl
            total_trades += nt
            total_wins += int(wr * nt)
            max_dd = max(max_dd, dd)
            asset_pnls[asset] = pnl
            if "trades_detail" in m:
                all_pnls.extend([t["pnl_pct"] for t in m["trades_detail"]])

        # Portfolio Sharpe
        if len(all_pnls) > 1:
            pa = np.array(all_pnls)
            df_ref = slice_window(data["BTC"], start, end)
            days = (df_ref.index[-1] - df_ref.index[0]).total_seconds() / 86400
            tpy = len(pa) / max(days / 365.25, 0.01)
            sharpe = (pa.mean() / pa.std(ddof=1)) * np.sqrt(tpy)
            sharpe = max(-10.0, min(10.0, sharpe))
        else:
            sharpe = 0.0

        port_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        total_initial = INITIAL_EQUITY * 3
        detail = "BTC:%+.0f  SOL:%+.0f  ETH:%+.0f" % (
            asset_pnls.get("BTC", 0), asset_pnls.get("SOL", 0), asset_pnls.get("ETH", 0))

        print("  %-10s %6d %4.0f%% %+7.2f %+9.2f %9.2f %6.1f%%  %s" %
              (pname, total_trades, port_wr, sharpe, total_pnl,
               total_initial + total_pnl, max_dd * 100, detail))

        # Collect 3Y pnls for Monte Carlo
        if pname == "3 ans":
            all_3y_pnls = all_pnls

    # ── Stability analysis ──
    print("\n" + "=" * 110)
    print("  ANALYSE DE STABILITE — Sharpe par sous-periodes de 6 mois")
    print("=" * 110)

    sub_windows = [
        ("2023-H1", "2023-01-01", "2023-07-01"),
        ("2023-H2", "2023-07-01", "2024-01-01"),
        ("2024-H1", "2024-01-01", "2024-07-01"),
        ("2024-H2", "2024-07-01", "2025-01-01"),
        ("2025-H1", "2025-01-01", "2025-07-01"),
        ("2025-H2", "2025-07-01", "2026-01-01"),
    ]

    for asset in ["BTC", "SOL", "ETH"]:
        cfg = PORTFOLIO[asset]
        sharpes = []
        pnls_list = []
        for wname, start, end in sub_windows:
            m = run_asset(data, bt, asset, cfg, start, end)
            sr = m.get("sharpe_ratio", 0)
            pnl = m.get("dollar_pnl", 0)
            sharpes.append(sr)
            pnls_list.append(pnl)

        n_pos = sum(1 for s in sharpes if s > 0)
        sr_str = "  ".join("%+.2f" % s for s in sharpes)
        pnl_str = "  ".join("%+.0f" % p for p in pnls_list)
        print("  %s: Sharpe [%s]  Stability %d/6" % (asset, sr_str, n_pos))
        print("       $PnL  [%s]" % pnl_str)

    # ── Monte Carlo ──
    if len(all_3y_pnls) >= 10:
        print("\n" + "=" * 110)
        print("  MONTE CARLO — 3 ans, 2000 simulations, %d trades" % len(all_3y_pnls))
        print("=" * 110)

        pnls_arr = np.array(all_3y_pnls)
        n_trades = len(pnls_arr)
        np.random.seed(42)
        n_sims = 2000

        final_returns = []
        max_drawdowns = []

        for _ in range(n_sims):
            sample = np.random.choice(pnls_arr, size=n_trades, replace=True)
            equity = INITIAL_EQUITY * 3  # portfolio total
            peak = equity
            max_dd = 0.0
            for pnl_pct in sample:
                equity *= (1 + pnl_pct / 100)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
            ret = (equity - INITIAL_EQUITY * 3) / (INITIAL_EQUITY * 3) * 100
            final_returns.append(ret)
            max_drawdowns.append(max_dd * 100)

        fr = np.array(final_returns)
        md = np.array(max_drawdowns)

        print("\n  Input: %d trades, WR %.1f%%, mean %+.3f%%, std %.3f%%" %
              (n_trades, np.sum(pnls_arr > 0) / n_trades * 100,
               pnls_arr.mean(), pnls_arr.std()))

        print("\n  Final return ($%d initial):" % int(INITIAL_EQUITY * 3))
        for pct in [5, 10, 25, 50, 75, 90, 95]:
            val = np.percentile(fr, pct)
            dollar = INITIAL_EQUITY * 3 * (1 + val / 100)
            print("    P%-3d: %+8.1f%%  ($%.0f)" % (pct, val, dollar))

        print("\n  Max Drawdown:")
        for pct in [50, 75, 90, 95, 99]:
            print("    P%-3d: %.1f%%" % (pct, np.percentile(md, pct)))

        profitable = np.sum(fr > 0) / n_sims * 100
        double = np.sum(fr > 100) / n_sims * 100
        ruin_50 = np.sum(fr < -50) / n_sims * 100
        print("\n  Risk:")
        print("    P(profit > 0%%)   : %.1f%%" % profitable)
        print("    P(return > 100%%) : %.1f%%" % double)
        print("    P(loss > 50%%)    : %.1f%%" % ruin_50)

    elapsed = time.time() - t0
    print("\n" + "=" * 110)
    print("  Runtime: %.1fs" % elapsed)
    print("=" * 110)


if __name__ == "__main__":
    main()
