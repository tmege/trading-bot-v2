#!/usr/bin/env python3
"""Quick 1-year portfolio backtest."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")

from exec_config import ExecConfig
from modules.feature_engine import FeatureEngine
from modules.strategies import V2_STRATEGY_REGISTRY
from sweep_runner import SweepBacktester

INITIAL_EQUITY = 1000.0

STRATEGIES = [
    {
        "name": "BTC InsideBarBreakout",
        "asset": "BTC",
        "strat_class": "StratInsideBarBreakout",
        "params": {"vol_min": 1.5, "trend_filter": True, "atr_filter": True, "sl_pct": 1.5, "tp_pct": 3.0},
        "exec_config": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=72),
    },
    {
        "name": "SOL BreakoutNormal",
        "asset": "SOL",
        "strat_class": "StratBreakoutRelaxed",
        "params": {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
        "exec_config": ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48),
    },
    {
        "name": "ETH BreakoutRelaxed",
        "asset": "ETH",
        "strat_class": "StratBreakoutRelaxed",
        "params": {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
        "exec_config": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48),
    },
]

WINDOWS = [
    ("2024-Q2",   "2024-04-01", "2024-07-01"),
    ("2024-Q3",   "2024-07-01", "2024-10-01"),
    ("2024-Q4",   "2024-10-01", "2025-01-01"),
    ("2025-Q1",   "2025-01-01", "2025-04-01"),
    ("1Y récent", "2025-03-19", "2026-03-19"),
    ("1Y glissant", "2024-03-19", "2025-03-19"),
]

def load_asset(symbol):
    fe = FeatureEngine()
    df_5m = pd.read_parquet("data/%s_USDT_5m_ohlcv.parquet" % symbol)
    df_5m = df_5m[~df_5m.index.duplicated(keep="first")]
    df_1h = df_5m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["open"])
    df_1h = fe.compute_all(df_1h)
    return df_1h

def slice_window(df, start, end):
    mask = (df.index >= pd.Timestamp(start, tz="UTC")) & (df.index < pd.Timestamp(end, tz="UTC"))
    return df.loc[mask]

def main():
    data = {}
    for s in STRATEGIES:
        if s["asset"] not in data:
            data[s["asset"]] = load_asset(s["asset"])

    bt = SweepBacktester()

    print("=" * 110)
    print("  BACKTEST 1 AN — PORTFOLIO BTC + SOL + ETH ($1,000)")
    print("=" * 110)

    # Individual
    print("\n  PERFORMANCE INDIVIDUELLE")
    print("  %-14s %-22s %6s %5s %7s %9s %9s %7s" %
          ("Fenêtre", "Stratégie", "Trades", "WR", "Sharpe", "$PnL", "Final$", "MaxDD"))
    print("  " + "-" * 95)

    for wname, start, end in WINDOWS:
        for sc in STRATEGIES:
            cls = V2_STRATEGY_REGISTRY[sc["strat_class"]]
            strat = cls(sc["params"])
            df_w = slice_window(data[sc["asset"]], start, end)
            if len(df_w) < 100:
                continue
            signals = strat.generate_signals(df_w)
            m = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                       exec_config=sc["exec_config"], initial_equity=INITIAL_EQUITY)
            m.pop("trades_detail", None)
            print("  %-14s %-22s %6d %4.0f%% %+7.2f %+9.2f %9.2f %6.1f%%" %
                  (wname, sc["name"], m["nb_trades"], m["win_rate"]*100,
                   m["sharpe_ratio"], m.get("dollar_pnl",0), m.get("final_equity",1000),
                   m["max_drawdown"]*100))
        print()

    # Combined portfolio
    print("\n  PORTFOLIO COMBINÉ")
    print("  %-14s %6s %5s %7s %9s %9s %7s  %s" %
          ("Fenêtre", "Trades", "WR", "Sharpe", "$PnL", "Final$", "MaxDD", "Détail"))
    print("  " + "-" * 100)

    for wname, start, end in WINDOWS:
        total_pnl = 0.0
        total_trades = 0
        total_wins = 0
        max_dd = 0.0
        all_pnls = []
        parts = []

        for sc in STRATEGIES:
            cls = V2_STRATEGY_REGISTRY[sc["strat_class"]]
            strat = cls(sc["params"])
            df_w = slice_window(data[sc["asset"]], start, end)
            if len(df_w) < 100:
                continue
            signals = strat.generate_signals(df_w)
            m = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                       exec_config=sc["exec_config"], initial_equity=INITIAL_EQUITY)
            pnl = m.get("dollar_pnl", 0)
            total_pnl += pnl
            total_trades += m["nb_trades"]
            total_wins += int(m["win_rate"] * m["nb_trades"])
            max_dd = max(max_dd, m["max_drawdown"])
            if "trades_detail" in m:
                all_pnls.extend([t["pnl_pct"] for t in m["trades_detail"]])
            parts.append("%s:%+.0f" % (sc["asset"], pnl))

        final = INITIAL_EQUITY + total_pnl
        wr = total_wins / total_trades * 100 if total_trades > 0 else 0

        if len(all_pnls) > 1:
            pa = np.array(all_pnls)
            df_ref = slice_window(data["BTC"], start, end)
            days = (df_ref.index[-1] - df_ref.index[0]).total_seconds() / 86400
            tpy = len(pa) / max(days / 365.25, 0.01)
            sharpe = (pa.mean() / pa.std(ddof=1)) * np.sqrt(tpy)
            sharpe = max(-10.0, min(10.0, sharpe))
        else:
            sharpe = 0.0

        print("  %-14s %6d %4.0f%% %+7.2f %+9.2f %9.2f %6.1f%%  %s" %
              (wname, total_trades, wr, sharpe, total_pnl, final, max_dd*100, "  ".join(parts)))

if __name__ == "__main__":
    main()
