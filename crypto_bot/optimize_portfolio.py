#!/usr/bin/env python3
"""
Exploration d'optimisations du portfolio V2.
Compare chaque optimisation vs la baseline.

Axes testés :
  1. BTC : relâcher les filtres pour plus de trades
  2. SOL : réduire le sizing ou ajuster SL/TP pour baisser MaxDD
  3. ETH : tester des variantes de paramètres
  4. Sizing global : redistribuer l'equity entre les 3 strats
  5. 4ème asset (DOGE) pour diversifier
  6. Exits améliorés : max_hold plus court/long
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
]

WINDOWS_FULL = WINDOWS + [("Full 3Y", "2023-01-01", "2026-01-01")]


def run_portfolio(data, bt, strategies, label=""):
    """Run a portfolio config across all windows, return summary."""
    results = {}

    for wname, start, end in WINDOWS_FULL:
        total_pnl = 0.0
        total_trades = 0
        total_wins = 0
        max_dd = 0.0
        all_pnls = []

        for sc in strategies:
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

        results[wname] = {
            "trades": total_trades, "wr": wr, "sharpe": sharpe,
            "pnl": total_pnl, "final": INITIAL_EQUITY + total_pnl,
            "maxdd": max_dd * 100,
        }

    return results


def print_comparison(baseline_results, opt_results, opt_name):
    """Print side-by-side comparison."""
    b3y = baseline_results.get("Full 3Y", {})
    o3y = opt_results.get("Full 3Y", {})

    # Stability: count windows with positive Sharpe
    b_stable = sum(1 for w in WINDOWS if baseline_results.get(w[0], {}).get("sharpe", 0) > 0)
    o_stable = sum(1 for w in WINDOWS if opt_results.get(w[0], {}).get("sharpe", 0) > 0)

    delta_sharpe = o3y.get("sharpe", 0) - b3y.get("sharpe", 0)
    delta_pnl = o3y.get("pnl", 0) - b3y.get("pnl", 0)
    delta_dd = o3y.get("maxdd", 0) - b3y.get("maxdd", 0)

    better = "MIEUX" if (delta_sharpe > 0.1 and delta_dd <= 2) else \
             "PIRE" if delta_sharpe < -0.1 else "~PAREIL"

    print("  %-35s  Sharpe %+.2f (%+.2f)  $PnL %+.0f (%+.0f)  DD %.1f%% (%+.1f%%)  Stable %d/5  → %s" %
          (opt_name,
           o3y.get("sharpe", 0), delta_sharpe,
           o3y.get("pnl", 0), delta_pnl,
           o3y.get("maxdd", 0), delta_dd,
           o_stable, better))


def main():
    t0 = time.time()

    print("=" * 120)
    print("  EXPLORATION D'OPTIMISATIONS — vs Baseline (Sharpe +1.99, $+2549, DD 21%)")
    print("=" * 120)

    print("\n-- Chargement --")
    data = {}
    for sym in ["BTC", "SOL", "ETH", "DOGE"]:
        try:
            data[sym] = load_asset(sym)
            print("  %s OK" % sym)
        except Exception as e:
            print("  %s SKIP (%s)" % (sym, e))

    bt = SweepBacktester()

    # ── BASELINE ──
    BASELINE = [
        {"name": "BTC", "asset": "BTC", "strat_class": "StratInsideBarBreakout",
         "params": {"vol_min": 1.5, "trend_filter": True, "atr_filter": True, "sl_pct": 1.5, "tp_pct": 3.0},
         "exec_config": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=72)},
        {"name": "SOL", "asset": "SOL", "strat_class": "StratBreakoutRelaxed",
         "params": {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
         "exec_config": ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48)},
        {"name": "ETH", "asset": "ETH", "strat_class": "StratBreakoutRelaxed",
         "params": {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
         "exec_config": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48)},
    ]

    baseline_r = run_portfolio(data, bt, BASELINE)

    print("\n" + "=" * 120)
    print("  BASELINE:")
    b3y = baseline_r["Full 3Y"]
    print("  Sharpe %+.2f  $PnL %+.0f  Final $%.0f  DD %.1f%%  Trades %d  WR %.0f%%" %
          (b3y["sharpe"], b3y["pnl"], b3y["final"], b3y["maxdd"], b3y["trades"], b3y["wr"]))
    print("=" * 120)

    # ═══════════════════════════════════════════════════════════
    # AXE 1 : BTC — Plus de trades
    # ═══════════════════════════════════════════════════════════
    print("\n  ── AXE 1 : BTC — Augmenter la fréquence ──")

    btc_variants = [
        ("BTC: vol_min=1.0 (plus large)", {"vol_min": 1.0, "trend_filter": True, "atr_filter": True, "sl_pct": 1.5, "tp_pct": 3.0}),
        ("BTC: sans trend filter", {"vol_min": 1.5, "trend_filter": False, "atr_filter": True, "sl_pct": 1.5, "tp_pct": 3.0}),
        ("BTC: sans ATR filter", {"vol_min": 1.5, "trend_filter": True, "atr_filter": False, "sl_pct": 1.5, "tp_pct": 3.0}),
        ("BTC: vol=1.0 + sans ATR", {"vol_min": 1.0, "trend_filter": True, "atr_filter": False, "sl_pct": 1.5, "tp_pct": 3.0}),
        ("BTC: tout relâché", {"vol_min": 0.8, "trend_filter": False, "atr_filter": False, "sl_pct": 1.5, "tp_pct": 3.0}),
        ("BTC: TP=5% (plus large)", {"vol_min": 1.5, "trend_filter": True, "atr_filter": True, "sl_pct": 1.5, "tp_pct": 5.0}),
        ("BTC: SL=2% TP=5%", {"vol_min": 1.5, "trend_filter": True, "atr_filter": True, "sl_pct": 2.0, "tp_pct": 5.0}),
        ("BTC: equity 30%", {"vol_min": 1.5, "trend_filter": True, "atr_filter": True, "sl_pct": 1.5, "tp_pct": 3.0}),
    ]

    for name, params in btc_variants:
        strats = list(BASELINE)
        ec = BASELINE[0]["exec_config"]
        if "equity 30%" in name:
            ec = ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=72)
        strats[0] = {**BASELINE[0], "params": params, "exec_config": ec}
        r = run_portfolio(data, bt, strats)
        print_comparison(baseline_r, r, name)

    # ═══════════════════════════════════════════════════════════
    # AXE 2 : SOL — Réduire le risque
    # ═══════════════════════════════════════════════════════════
    print("\n  ── AXE 2 : SOL — Réduire MaxDD ──")

    sol_variants = [
        ("SOL: equity 20% (réduit)", {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
         ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48)),
        ("SOL: SL=1.0% (tight)", {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.0, "tp_pct": 4.0},
         ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48)),
        ("SOL: cooldown 6h", {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
         ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=6, max_hold_bars=48)),
        ("SOL: max_hold 24h", {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
         ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=24)),
        ("SOL: vol_min=3.5 (strict)", {"lookback": 15, "vol_breakout_min": 3.5, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
         ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48)),
        ("SOL: lookback=20", {"lookback": 20, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
         ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48)),
        ("SOL: leverage 3x", {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
         ExecConfig(equity_pct=0.30, leverage=3, cooldown_bars=4, max_hold_bars=48)),
    ]

    for name, params, ec in sol_variants:
        strats = list(BASELINE)
        strats[1] = {**BASELINE[1], "params": params, "exec_config": ec}
        r = run_portfolio(data, bt, strats)
        print_comparison(baseline_r, r, name)

    # ═══════════════════════════════════════════════════════════
    # AXE 3 : ETH — Variantes
    # ═══════════════════════════════════════════════════════════
    print("\n  ── AXE 3 : ETH — Variantes de params ──")

    eth_variants = [
        ("ETH: equity 25%", {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
         ExecConfig(equity_pct=0.25, leverage=5, cooldown_bars=4, max_hold_bars=48)),
        ("ETH: TP=5%", {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 5.0},
         ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48)),
        ("ETH: lookback=10", {"lookback": 10, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
         ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48)),
        ("ETH: vol_min=2.5", {"lookback": 15, "vol_breakout_min": 2.5, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
         ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48)),
        ("ETH: max_hold 72h", {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
         ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=72)),
        ("ETH: compression ON", {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": True, "sl_pct": 1.5, "tp_pct": 4.0},
         ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48)),
    ]

    for name, params, ec in eth_variants:
        strats = list(BASELINE)
        strats[2] = {**BASELINE[2], "params": params, "exec_config": ec}
        r = run_portfolio(data, bt, strats)
        print_comparison(baseline_r, r, name)

    # ═══════════════════════════════════════════════════════════
    # AXE 4 : Sizing global
    # ═══════════════════════════════════════════════════════════
    print("\n  ── AXE 4 : Redistribution du sizing ──")

    sizing_variants = [
        ("Equal 20/20/20", 0.20, 0.20, 0.20),
        ("Equal 25/25/25", 0.25, 0.25, 0.25),
        ("BTC heavy 30/20/20", 0.30, 0.20, 0.20),
        ("SOL light 20/20/20", 0.20, 0.20, 0.20),
        ("Agressif 25/35/25", 0.25, 0.35, 0.25),
        ("Conservateur 15/20/15", 0.15, 0.20, 0.15),
    ]

    for name, btc_eq, sol_eq, eth_eq in sizing_variants:
        strats = [
            {**BASELINE[0], "exec_config": ExecConfig(equity_pct=btc_eq, leverage=5, cooldown_bars=4, max_hold_bars=72)},
            {**BASELINE[1], "exec_config": ExecConfig(equity_pct=sol_eq, leverage=5, cooldown_bars=4, max_hold_bars=48)},
            {**BASELINE[2], "exec_config": ExecConfig(equity_pct=eth_eq, leverage=5, cooldown_bars=4, max_hold_bars=48)},
        ]
        r = run_portfolio(data, bt, strats)
        print_comparison(baseline_r, r, name)

    # ═══════════════════════════════════════════════════════════
    # AXE 5 : 4ème asset (DOGE)
    # ═══════════════════════════════════════════════════════════
    if "DOGE" in data:
        print("\n  ── AXE 5 : Ajout de DOGE ──")

        doge_configs = [
            ("DOGE Breakout lb=10 vol=2.0",
             {"lookback": 10, "vol_breakout_min": 2.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0}),
            ("DOGE Breakout lb=15 vol=3.0",
             {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 5.0}),
            ("DOGE InsideBar",
             None),  # use InsideBarBreakout
        ]

        for doge_name, doge_params in doge_configs:
            strats = list(BASELINE)
            if doge_params is None:
                strats.append({
                    "name": "DOGE", "asset": "DOGE", "strat_class": "StratInsideBarBreakout",
                    "params": {"vol_min": 1.5, "trend_filter": True, "atr_filter": True, "sl_pct": 1.5, "tp_pct": 3.0},
                    "exec_config": ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=48),
                })
            else:
                strats.append({
                    "name": "DOGE", "asset": "DOGE", "strat_class": "StratBreakoutRelaxed",
                    "params": doge_params,
                    "exec_config": ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=48),
                })
            # Reduce others slightly to make room
            strats[0] = {**strats[0], "exec_config": ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=72)}
            strats[1] = {**strats[1], "exec_config": ExecConfig(equity_pct=0.25, leverage=5, cooldown_bars=4, max_hold_bars=48)}
            strats[2] = {**strats[2], "exec_config": ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=48)}

            r = run_portfolio(data, bt, strats)
            print_comparison(baseline_r, r, "+ " + doge_name)

    # ═══════════════════════════════════════════════════════════
    # AXE 6 : Leverage
    # ═══════════════════════════════════════════════════════════
    print("\n  ── AXE 6 : Leverage global ──")

    for lev in [3, 7, 10]:
        strats = [
            {**BASELINE[0], "exec_config": ExecConfig(equity_pct=0.20, leverage=lev, cooldown_bars=4, max_hold_bars=72)},
            {**BASELINE[1], "exec_config": ExecConfig(equity_pct=0.30, leverage=lev, cooldown_bars=4, max_hold_bars=48)},
            {**BASELINE[2], "exec_config": ExecConfig(equity_pct=0.20, leverage=lev, cooldown_bars=4, max_hold_bars=48)},
        ]
        r = run_portfolio(data, bt, strats)
        print_comparison(baseline_r, r, "Leverage global %dx" % lev)

    elapsed = time.time() - t0
    print("\n" + "=" * 120)
    print("  Temps total : %.1fs" % elapsed)
    print("=" * 120)


if __name__ == "__main__":
    main()
