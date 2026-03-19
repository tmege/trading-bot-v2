#!/usr/bin/env python3
"""
Test: entry offset plus large (0.5%, 1%, 1.5%) vs baseline (0.02%).
Long entre 0.5% plus bas, short 0.5% plus haut.
Simule en augmentant entry_offset dans ExecConfig.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")

from exec_config import ExecConfig
from modules.feature_engine import FeatureEngine
from modules.strategies import V2_STRATEGY_REGISTRY
from sweep_runner import SweepBacktester

INITIAL_EQUITY = 1000.0

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

WINDOWS = [
    ("2023-H1", "2023-01-01", "2023-07-01"),
    ("2023-H2", "2023-07-01", "2024-01-01"),
    ("2024-H1", "2024-01-01", "2024-07-01"),
    ("2024-H2", "2024-07-01", "2025-01-01"),
    ("2025-H1", "2025-01-01", "2025-07-01"),
    ("Full 3Y", "2023-01-01", "2026-01-01"),
]

STRATEGIES = [
    {"name": "BTC InsideBar", "asset": "BTC", "strat_class": "StratInsideBarBreakout",
     "params": {"vol_min": 1.5, "trend_filter": True, "atr_filter": True, "sl_pct": 1.5, "tp_pct": 3.0},
     "base_ec": {"equity_pct": 0.20, "leverage": 5, "cooldown_bars": 4, "max_hold_bars": 72}},
    {"name": "SOL Breakout", "asset": "SOL", "strat_class": "StratBreakoutRelaxed",
     "params": {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
     "base_ec": {"equity_pct": 0.30, "leverage": 5, "cooldown_bars": 4, "max_hold_bars": 48}},
    {"name": "ETH Breakout", "asset": "ETH", "strat_class": "StratBreakoutRelaxed",
     "params": {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 5.0},
     "base_ec": {"equity_pct": 0.20, "leverage": 5, "cooldown_bars": 4, "max_hold_bars": 48}},
]

OFFSETS = [0.0002, 0.002, 0.005, 0.010, 0.015]  # 0.02%, 0.2%, 0.5%, 1%, 1.5%

def main():
    data = {}
    for s in STRATEGIES:
        if s["asset"] not in data:
            data[s["asset"]] = load_asset(s["asset"])

    bt = SweepBacktester()

    print("=" * 130)
    print("  TEST ENTRY OFFSET — Long plus bas / Short plus haut")
    print("  Question: entrer 0.5%+ plus loin du signal améliore-t-il le PnL?")
    print("=" * 130)

    # Test par stratégie individuelle d'abord
    for sc in STRATEGIES:
        cls = V2_STRATEGY_REGISTRY[sc["strat_class"]]
        strat = cls(sc["params"])
        df_asset = data[sc["asset"]]

        print("\n  ── %s ──" % sc["name"])
        header = "  %-12s" % "Offset"
        for wname, _, _ in WINDOWS:
            header += " %10s" % wname
        header += " %9s %7s" % ("$PnL 3Y", "Trades")
        print(header)
        print("  " + "-" * 110)

        for offset in OFFSETS:
            ec = ExecConfig(
                entry_offset=offset,
                **sc["base_ec"]
            )
            row = "  %-12s" % ("%.2f%%" % (offset * 100))

            pnl_3y = 0
            trades_3y = 0
            for wname, start, end in WINDOWS:
                df_w = slice_window(df_asset, start, end)
                if len(df_w) < 100:
                    row += " %10s" % "N/A"
                    continue
                signals = strat.generate_signals(df_w)
                m = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                           exec_config=ec, initial_equity=INITIAL_EQUITY)
                m.pop("trades_detail", None)
                row += " %+10.2f" % m["sharpe_ratio"]
                if wname == "Full 3Y":
                    pnl_3y = m.get("dollar_pnl", 0)
                    trades_3y = m["nb_trades"]

            row += " %+9.0f %7d" % (pnl_3y, trades_3y)
            print(row)

    # Portfolio combined
    print("\n" + "=" * 130)
    print("  PORTFOLIO COMBINÉ — Tous les offsets")
    print("=" * 130)

    header = "  %-12s" % "Offset"
    for wname, _, _ in WINDOWS:
        header += " %10s" % wname
    header += " %9s %7s %7s" % ("$PnL 3Y", "Trades", "MaxDD")
    print(header)
    print("  " + "-" * 120)

    for offset in OFFSETS:
        row = "  %-12s" % ("%.2f%%" % (offset * 100))

        for wname, start, end in WINDOWS:
            total_pnl = 0
            total_trades = 0
            max_dd = 0
            all_pnls = []

            for sc in STRATEGIES:
                cls = V2_STRATEGY_REGISTRY[sc["strat_class"]]
                strat = cls(sc["params"])
                df_w = slice_window(data[sc["asset"]], start, end)
                if len(df_w) < 100:
                    continue
                ec = ExecConfig(entry_offset=offset, **sc["base_ec"])
                signals = strat.generate_signals(df_w)
                m = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                           exec_config=ec, initial_equity=INITIAL_EQUITY)
                total_pnl += m.get("dollar_pnl", 0)
                total_trades += m["nb_trades"]
                max_dd = max(max_dd, m["max_drawdown"])
                if "trades_detail" in m:
                    all_pnls.extend([t["pnl_pct"] for t in m["trades_detail"]])

            if len(all_pnls) > 1:
                pa = np.array(all_pnls)
                df_ref = slice_window(data["BTC"], start, end)
                days = (df_ref.index[-1] - df_ref.index[0]).total_seconds() / 86400
                tpy = len(pa) / max(days / 365.25, 0.01)
                sharpe = (pa.mean() / pa.std(ddof=1)) * np.sqrt(tpy)
                sharpe = max(-10.0, min(10.0, sharpe))
            else:
                sharpe = 0.0

            row += " %+10.2f" % sharpe

            if wname == "Full 3Y":
                pnl_3y = total_pnl
                trades_3y = total_trades
                dd_3y = max_dd * 100

        row += " %+9.0f %7d %6.1f%%" % (pnl_3y, trades_3y, dd_3y)
        print(row)

    print("\n  Note: offset = distance entre le signal et le prix d'entrée limit.")
    print("  0.02% = baseline actuelle (ALO classique)")
    print("  0.50% = entrer 0.5%% plus loin (long plus bas, short plus haut)")
    print("  Plus l'offset est grand, plus on entre à un meilleur prix MAIS plus de trades sont ratés.")


if __name__ == "__main__":
    main()
