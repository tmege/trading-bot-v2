#!/usr/bin/env python3
"""
Backtest portfolio complet : BTC InsideBarBreakout + SOL BreakoutNormal + ETH BreakoutRelaxed.
Résultats par rapport à $1000 de capital initial.
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

WINDOWS = [
    ("2023-H1", "2023-01-01", "2023-07-01"),
    ("2023-H2", "2023-07-01", "2024-01-01"),
    ("2024-H1", "2024-01-01", "2024-07-01"),
    ("2024-H2", "2024-07-01", "2025-01-01"),
    ("2025-H1", "2025-01-01", "2025-07-01"),
    ("Full 2Y",  "2023-01-01", "2025-01-01"),
    ("Full 3Y",  "2023-01-01", "2026-01-01"),
]

INITIAL_EQUITY = 1000.0


def load_asset(symbol):
    fe = FeatureEngine()
    df_5m = pd.read_parquet("data/%s_USDT_5m_ohlcv.parquet" % symbol)
    df_5m = df_5m[~df_5m.index.duplicated(keep="first")]
    df_1h = df_5m.resample("1h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["open"])
    df_1h = fe.compute_all(df_1h)
    print("  %s/USDT: %s bougies 1h [%s -> %s]" %
          (symbol, f"{len(df_1h):,}", df_1h.index[0].date(), df_1h.index[-1].date()))
    return df_1h


def slice_window(df, start, end):
    mask = (df.index >= pd.Timestamp(start, tz="UTC")) & \
           (df.index < pd.Timestamp(end, tz="UTC"))
    return df.loc[mask]


# ── Strategy configs (matching live strategies) ──

STRATEGIES = [
    {
        "name": "BTC InsideBarBreakout",
        "asset": "BTC",
        "strat_class": "StratInsideBarBreakout",
        "params": {
            "vol_min": 1.5, "trend_filter": True,
            "atr_filter": True, "sl_pct": 1.5, "tp_pct": 3.0,
        },
        "exec_config": ExecConfig(
            equity_pct=0.20, leverage=5,
            cooldown_bars=4, max_hold_bars=72,
        ),
    },
    {
        "name": "SOL BreakoutNormal",
        "asset": "SOL",
        "strat_class": "StratBreakoutRelaxed",
        "params": {
            "lookback": 15, "vol_breakout_min": 3.0,
            "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0,
        },
        "exec_config": ExecConfig(
            equity_pct=0.30, leverage=5,
            cooldown_bars=4, max_hold_bars=48,
        ),
    },
    {
        "name": "ETH BreakoutRelaxed",
        "asset": "ETH",
        "strat_class": "StratBreakoutRelaxed",
        "params": {
            "lookback": 15, "vol_breakout_min": 3.0,
            "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0,
        },
        "exec_config": ExecConfig(
            equity_pct=0.20, leverage=5,
            cooldown_bars=4, max_hold_bars=48,
        ),
    },
]


def main():
    t0 = time.time()

    print("=" * 110)
    print("  BACKTEST PORTFOLIO V2 — BTC + SOL + ETH ($1,000 initial)")
    print("=" * 110)

    # Load all assets
    print("\n-- Chargement des données --")
    data = {}
    for s in STRATEGIES:
        asset = s["asset"]
        if asset not in data:
            data[asset] = load_asset(asset)

    bt = SweepBacktester()

    # ── Individual strategy performance ──
    print("\n" + "=" * 110)
    print("  PERFORMANCE INDIVIDUELLE (chaque strat sur $1,000)")
    print("=" * 110)

    for strat_cfg in STRATEGIES:
        cls = V2_STRATEGY_REGISTRY[strat_cfg["strat_class"]]
        strat = cls(strat_cfg["params"])
        df_asset = data[strat_cfg["asset"]]

        print("\n  -- %s (%s, equity=%.0f%%) --" %
              (strat_cfg["name"], strat_cfg["asset"],
               strat_cfg["exec_config"].equity_pct * 100))
        print("  %-12s %6s %5s %7s %9s %9s %7s" %
              ("Fenêtre", "Trades", "WR", "Sharpe", "$PnL", "Final$", "MaxDD"))
        print("  " + "-" * 65)

        for window_name, start, end in WINDOWS:
            df_w = slice_window(df_asset, start, end)
            if len(df_w) < 200:
                continue

            signals = strat.generate_signals(df_w)
            metrics = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct,
                             strat.max_hold, exec_config=strat_cfg["exec_config"],
                             initial_equity=INITIAL_EQUITY)
            metrics.pop("trades_detail", None)

            wr = metrics["win_rate"] * 100
            dd = metrics["max_drawdown"] * 100
            pnl = metrics.get("dollar_pnl", 0)
            final = metrics.get("final_equity", INITIAL_EQUITY)
            print("  %-12s %6d %4.0f%% %+7.2f %+9.2f %9.2f %6.1f%%" %
                  (window_name, metrics["nb_trades"], wr,
                   metrics["sharpe_ratio"], pnl, final, dd))

    # ── Combined portfolio ──
    print("\n" + "=" * 110)
    print("  PORTFOLIO COMBINÉ (3 strats, $1,000 total)")
    print("=" * 110)
    print("  %-12s %6s %5s %7s %9s %9s %7s  %s" %
          ("Fenêtre", "Trades", "WR", "Sharpe", "$PnL", "Final$", "MaxDD", "Détail PnL"))
    print("  " + "-" * 100)

    for window_name, start, end in WINDOWS:
        total_pnl = 0.0
        total_trades = 0
        total_wins = 0
        max_dd = 0.0
        all_pnls = []
        detail_parts = []

        for strat_cfg in STRATEGIES:
            cls = V2_STRATEGY_REGISTRY[strat_cfg["strat_class"]]
            strat = cls(strat_cfg["params"])
            df_w = slice_window(data[strat_cfg["asset"]], start, end)
            if len(df_w) < 200:
                continue

            signals = strat.generate_signals(df_w)
            metrics = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct,
                             strat.max_hold, exec_config=strat_cfg["exec_config"],
                             initial_equity=INITIAL_EQUITY)

            pnl = metrics.get("dollar_pnl", 0)
            total_pnl += pnl
            total_trades += metrics["nb_trades"]
            total_wins += int(metrics["win_rate"] * metrics["nb_trades"])
            max_dd = max(max_dd, metrics["max_drawdown"])

            if "trades_detail" in metrics:
                all_pnls.extend([t["pnl_pct"] for t in metrics["trades_detail"]])

            detail_parts.append("%s:%+.0f" % (strat_cfg["asset"], pnl))

        final_equity = INITIAL_EQUITY + total_pnl
        wr = total_wins / total_trades * 100 if total_trades > 0 else 0

        if len(all_pnls) > 1:
            pnls_arr = np.array(all_pnls)
            df_ref = slice_window(data["BTC"], start, end)
            total_days = (df_ref.index[-1] - df_ref.index[0]).total_seconds() / 86400
            trades_per_year = len(pnls_arr) / max(total_days / 365.25, 0.01)
            sharpe = (pnls_arr.mean() / pnls_arr.std(ddof=1)) * np.sqrt(trades_per_year)
            sharpe = max(-10.0, min(10.0, sharpe))
        else:
            sharpe = 0.0

        dd_pct = max_dd * 100
        detail = "  ".join(detail_parts)
        print("  %-12s %6d %4.0f%% %+7.2f %+9.2f %9.2f %6.1f%%  %s" %
              (window_name, total_trades, wr, sharpe, total_pnl, final_equity, dd_pct, detail))

    # ── Signal correlation ──
    print("\n" + "=" * 110)
    print("  ANALYSE CORRÉLATION DES SIGNAUX")
    print("=" * 110)

    df_3y_btc = slice_window(data["BTC"], "2023-01-01", "2026-01-01")
    df_3y_sol = slice_window(data["SOL"], "2023-01-01", "2026-01-01")
    df_3y_eth = slice_window(data["ETH"], "2023-01-01", "2026-01-01")

    btc_strat = V2_STRATEGY_REGISTRY["StratInsideBarBreakout"](STRATEGIES[0]["params"])
    sol_strat = V2_STRATEGY_REGISTRY["StratBreakoutRelaxed"](STRATEGIES[1]["params"])
    eth_strat = V2_STRATEGY_REGISTRY["StratBreakoutRelaxed"](STRATEGIES[2]["params"])

    sig_btc = btc_strat.generate_signals(df_3y_btc)
    sig_sol = sol_strat.generate_signals(df_3y_sol)
    sig_eth = eth_strat.generate_signals(df_3y_eth)

    # Align to common index
    common_idx = sig_btc.index.intersection(sig_sol.index).intersection(sig_eth.index)
    sb = sig_btc.reindex(common_idx).fillna(0)
    ss = sig_sol.reindex(common_idx).fillna(0)
    se = sig_eth.reindex(common_idx).fillna(0)

    active_btc = (sb != 0)
    active_sol = (ss != 0)
    active_eth = (se != 0)

    n_total = len(common_idx)
    n_btc = active_btc.sum()
    n_sol = active_sol.sum()
    n_eth = active_eth.sum()
    n_btc_sol = (active_btc & active_sol).sum()
    n_btc_eth = (active_btc & active_eth).sum()
    n_sol_eth = (active_sol & active_eth).sum()
    n_all3 = (active_btc & active_sol & active_eth).sum()

    # 4h window overlap
    active_btc_4h = active_btc.rolling(4, min_periods=1).max().astype(bool)
    active_sol_4h = active_sol.rolling(4, min_periods=1).max().astype(bool)
    active_eth_4h = active_eth.rolling(4, min_periods=1).max().astype(bool)
    n_all3_4h = (active_btc_4h & active_sol_4h & active_eth_4h).sum()

    print("\n  Signaux totaux sur 3Y :")
    print("    BTC: %d signaux" % n_btc)
    print("    SOL: %d signaux" % n_sol)
    print("    ETH: %d signaux" % n_eth)

    print("\n  Chevauchements (même bougie 1h) :")
    print("    BTC+SOL : %d (%.1f%%)" % (n_btc_sol, n_btc_sol / n_total * 100))
    print("    BTC+ETH : %d (%.1f%%)" % (n_btc_eth, n_btc_eth / n_total * 100))
    print("    SOL+ETH : %d (%.1f%%)" % (n_sol_eth, n_sol_eth / n_total * 100))
    print("    3 en même temps : %d (%.1f%%)" % (n_all3, n_all3 / n_total * 100))

    print("\n  Chevauchements (fenêtre 4h) :")
    print("    3 en même temps : %d bougies (%.1f%%)" %
          (n_all3_4h, n_all3_4h / n_total * 100))

    # Worst case simultaneous loss
    btc_max_loss = STRATEGIES[0]["exec_config"].equity_pct * STRATEGIES[0]["exec_config"].leverage * STRATEGIES[0]["params"]["sl_pct"] / 100
    sol_max_loss = STRATEGIES[1]["exec_config"].equity_pct * STRATEGIES[1]["exec_config"].leverage * STRATEGIES[1]["params"]["sl_pct"] / 100
    eth_max_loss = STRATEGIES[2]["exec_config"].equity_pct * STRATEGIES[2]["exec_config"].leverage * STRATEGIES[2]["params"]["sl_pct"] / 100
    total_max_loss = btc_max_loss + sol_max_loss + eth_max_loss

    print("\n  Perte max simultanée (pire cas) :")
    print("    BTC: %.1f%% × %dx × %.1f%% SL = %.1f%%" %
          (STRATEGIES[0]["exec_config"].equity_pct * 100,
           STRATEGIES[0]["exec_config"].leverage,
           STRATEGIES[0]["params"]["sl_pct"],
           btc_max_loss * 100))
    print("    SOL: %.1f%% × %dx × %.1f%% SL = %.1f%%" %
          (STRATEGIES[1]["exec_config"].equity_pct * 100,
           STRATEGIES[1]["exec_config"].leverage,
           STRATEGIES[1]["params"]["sl_pct"],
           sol_max_loss * 100))
    print("    ETH: %.1f%% × %dx × %.1f%% SL = %.1f%%" %
          (STRATEGIES[2]["exec_config"].equity_pct * 100,
           STRATEGIES[2]["exec_config"].leverage,
           STRATEGIES[2]["params"]["sl_pct"],
           eth_max_loss * 100))
    print("    TOTAL: %.1f%% (limite daily loss: 10%%)" % (total_max_loss * 100))

    # ── Monte Carlo ──
    print("\n" + "=" * 110)
    print("  MONTE CARLO — PORTFOLIO COMBINÉ (1000 sims)")
    print("=" * 110)

    # Collect all trade PnLs from 3Y
    all_trade_pnls = []
    for strat_cfg in STRATEGIES:
        cls = V2_STRATEGY_REGISTRY[strat_cfg["strat_class"]]
        strat = cls(strat_cfg["params"])
        df_w = slice_window(data[strat_cfg["asset"]], "2023-01-01", "2026-01-01")
        if len(df_w) < 200:
            continue
        signals = strat.generate_signals(df_w)
        metrics = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct,
                         strat.max_hold, exec_config=strat_cfg["exec_config"],
                         initial_equity=INITIAL_EQUITY)
        if "trades_detail" in metrics:
            all_trade_pnls.extend([t["pnl_pct"] for t in metrics["trades_detail"]])

    if len(all_trade_pnls) > 10:
        pnls_arr = np.array(all_trade_pnls)
        n_trades = len(pnls_arr)
        n_sims = 1000
        np.random.seed(42)

        final_returns = []
        max_drawdowns = []

        for _ in range(n_sims):
            sample = np.random.choice(pnls_arr, size=n_trades, replace=True)
            equity = INITIAL_EQUITY
            peak = equity
            max_dd = 0.0

            for pnl_pct in sample:
                equity *= (1 + pnl_pct / 100)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

            final_returns.append((equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100)
            max_drawdowns.append(max_dd * 100)

        fr = np.array(final_returns)
        md = np.array(max_drawdowns)

        print("\n  Rendement final ($1,000 initial) :")
        print("    Médian : %+.1f%% ($%.0f)" %
              (np.median(fr), INITIAL_EQUITY * (1 + np.median(fr) / 100)))
        print("    P5     : %+.1f%% ($%.0f)" %
              (np.percentile(fr, 5), INITIAL_EQUITY * (1 + np.percentile(fr, 5) / 100)))
        print("    P25    : %+.1f%% ($%.0f)" %
              (np.percentile(fr, 25), INITIAL_EQUITY * (1 + np.percentile(fr, 25) / 100)))
        print("    P75    : %+.1f%% ($%.0f)" %
              (np.percentile(fr, 75), INITIAL_EQUITY * (1 + np.percentile(fr, 75) / 100)))
        print("    P95    : %+.1f%% ($%.0f)" %
              (np.percentile(fr, 95), INITIAL_EQUITY * (1 + np.percentile(fr, 95) / 100)))

        print("\n  Max Drawdown :")
        print("    Médian : %.1f%%" % np.median(md))
        print("    P95    : %.1f%%" % np.percentile(md, 95))

        ruin_count = np.sum(fr < -80)
        print("\n  Probabilité de ruine (>80%% perte) : %.1f%%" %
              (ruin_count / n_sims * 100))

    elapsed = time.time() - t0
    print("\n" + "=" * 110)
    print("  Temps total : %.1fs" % elapsed)
    print("=" * 110)


if __name__ == "__main__":
    main()
