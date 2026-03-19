#!/usr/bin/env python3
"""Extract exact params of top BTC configs from sweep."""
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, ".")

from exec_config import ExecConfig
from modules.feature_engine import FeatureEngine
from modules.strategies import V2_STRATEGY_REGISTRY
from param_sweep import FULL_GRID, expand_grid
from sweep_runner import SweepBacktester

WINDOWS = [
    ("2023-H1", "2023-01-01", "2023-07-01"),
    ("2023-H2", "2023-07-01", "2024-01-01"),
    ("2024-H1", "2024-01-01", "2024-07-01"),
    ("2024-H2", "2024-07-01", "2025-01-01"),
    ("2025-H1", "2025-01-01", "2025-07-01"),
]

REALISTIC_EC = ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=72)

def load_btc():
    fe = FeatureEngine()
    df_5m = pd.read_parquet("data/BTC_USDT_5m_ohlcv.parquet")
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

def main():
    df_full = load_btc()
    bt = SweepBacktester()

    # Collect all results across windows
    perf = defaultdict(lambda: {"sharpes": {}, "params": None, "strat_name": None, "pnls": {}})

    for window_name, start, end in WINDOWS:
        df_w = slice_window(df_full, start, end)
        if len(df_w) < 200:
            continue

        for strat_name in ["StratBreakoutRelaxed", "StratInsideBarBreakout"]:
            param_grid = FULL_GRID.get(strat_name)
            if not param_grid:
                continue
            cls = V2_STRATEGY_REGISTRY[strat_name]
            combos = expand_grid(param_grid)

            for params in combos:
                strat = cls(params)
                freq = strat.signal_frequency(df_w)
                if freq["total_signaux"] < 3:
                    continue
                signals = strat.generate_signals(df_w)
                metrics = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                                 exec_config=REALISTIC_EC, initial_equity=1000.0)
                metrics.pop("trades_detail", None)

                key = "%s|%s" % (strat_name, str(sorted(params.items())))
                perf[key]["sharpes"][window_name] = metrics["sharpe_ratio"]
                perf[key]["pnls"][window_name] = metrics.get("dollar_pnl", 0)
                perf[key]["params"] = params
                perf[key]["strat_name"] = strat_name

    # Filter 5/5 stable
    stable_5 = []
    for key, data in perf.items():
        sharpes = list(data["sharpes"].values())
        if len(sharpes) >= 5 and all(s > 0 for s in sharpes):
            stable_5.append({
                "key": key,
                "strat_name": data["strat_name"],
                "params": data["params"],
                "avg_sharpe": np.mean(sharpes),
                "min_sharpe": np.min(sharpes),
                "sharpes": data["sharpes"],
                "pnls": data["pnls"],
            })
    stable_5.sort(key=lambda x: x["avg_sharpe"], reverse=True)

    print("=" * 120)
    print("  TOP BTC CONFIGS — STABLE 5/5")
    print("=" * 120)

    for i, c in enumerate(stable_5):
        print("\n  #%d — %s (avg Sharpe: %+.2f, min: %+.2f)" %
              (i+1, c["strat_name"], c["avg_sharpe"], c["min_sharpe"]))
        print("  Params: %s" % c["params"])
        print("  Sharpes: %s" % "  ".join("%s:%+.2f" % (w, s) for w, s in c["sharpes"].items()))
        print("  PnLs:   %s" % "  ".join("%s:$%+.0f" % (w, p) for w, p in c["pnls"].items()))

    # Full 3Y backtest for top 5
    print("\n" + "=" * 120)
    print("  FULL 3Y BACKTEST — TOP 5")
    print("=" * 120)

    df_3y = slice_window(df_full, "2023-01-01", "2026-01-01")
    for i, c in enumerate(stable_5[:5]):
        cls = V2_STRATEGY_REGISTRY[c["strat_name"]]
        strat = cls(c["params"])
        signals = strat.generate_signals(df_3y)
        metrics = bt.run(df_3y, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                         exec_config=REALISTIC_EC, initial_equity=1000.0)
        metrics.pop("trades_detail", None)

        # Also test at 20% equity (to cohabit with SOL + ETH)
        ec_20 = ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=72)
        metrics_20 = bt.run(df_3y, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                            exec_config=ec_20, initial_equity=1000.0)
        metrics_20.pop("trades_detail", None)

        print("\n  #%d — %s" % (i+1, c["strat_name"]))
        print("  Params: %s" % c["params"])
        print("  @30%% equity: $1000 -> $%.0f (%+.0f%%), Sharpe %+.2f, %d trades, WR %.0f%%, MaxDD %.1f%%" %
              (metrics.get("final_equity", 1000), metrics.get("dollar_pnl", 0) / 10,
               metrics["sharpe_ratio"], metrics["nb_trades"],
               metrics["win_rate"] * 100, metrics["max_drawdown"] * 100))
        print("  @20%% equity: $1000 -> $%.0f (%+.0f%%), Sharpe %+.2f, %d trades, WR %.0f%%, MaxDD %.1f%%" %
              (metrics_20.get("final_equity", 1000), metrics_20.get("dollar_pnl", 0) / 10,
               metrics_20["sharpe_ratio"], metrics_20["nb_trades"],
               metrics_20["win_rate"] * 100, metrics_20["max_drawdown"] * 100))


if __name__ == "__main__":
    main()
