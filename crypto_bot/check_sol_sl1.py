#!/usr/bin/env python3
"""
Check SOL StratBreakoutRelaxed: SL=1.0% TP=4.0% lookback=15 vol=2.5 no-compression
vs baseline SL=1.5% TP=4.0% vol=3.0

Prints detailed results per window.
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

WINDOWS_FULL = [
    ("2023-H1", "2023-01-01", "2023-07-01"),
    ("2023-H2", "2023-07-01", "2024-01-01"),
    ("2024-H1", "2024-01-01", "2024-07-01"),
    ("2024-H2", "2024-07-01", "2025-01-01"),
    ("2025-H1", "2025-01-01", "2025-07-01"),
    ("Full 3Y", "2023-01-01", "2026-01-01"),
]


def load_sol():
    fe = FeatureEngine()
    df_5m = pd.read_parquet("data/SOL_USDT_5m_ohlcv.parquet")
    df_5m = df_5m[~df_5m.index.duplicated(keep="first")]
    df_1h = df_5m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["open"])
    df_1h = fe.compute_all(df_1h)
    return df_1h


def slice_window(df, start, end):
    mask = (df.index >= pd.Timestamp(start, tz="UTC")) & (df.index < pd.Timestamp(end, tz="UTC"))
    return df.loc[mask]


def filter_anti_wick(max_wick_ratio=0.5):
    def _filter(signals, df):
        body = (df["close"] - df["open"]).abs()
        total_range = df["high"] - df["low"]
        wick_ratio = 1 - body / total_range.replace(0, 1)
        return signals.where(wick_ratio < max_wick_ratio, 0)
    return _filter


def run_config(df_sol, bt, params, ec, label, signal_filter=None):
    """Run a single config on all windows and print detailed results."""
    cls = V2_STRATEGY_REGISTRY["StratBreakoutRelaxed"]
    strat = cls(params)

    print(f"\n{'=' * 100}")
    print(f"  {label}")
    print(f"  Params: lookback={params['lookback']}  vol_breakout_min={params['vol_breakout_min']}  "
          f"use_compression={params['use_compression']}  SL={params['sl_pct']}%  TP={params['tp_pct']}%")
    if signal_filter:
        print(f"  Filter: anti_wick 40%")
    print(f"  ExecConfig: equity={ec.equity_pct*100:.0f}%  lev={ec.leverage}x  "
          f"cooldown={ec.cooldown_bars}  max_hold={ec.max_hold_bars}")
    print(f"{'=' * 100}")

    header = (f"  {'Window':<10} {'Sharpe':>8} {'$PnL':>9} {'MaxDD':>7} "
              f"{'Trades':>7} {'WinRate':>8} {'AvgWin':>8} {'AvgLoss':>8} "
              f"{'PF':>6} {'FinalEq':>10}")
    print(header)
    print(f"  {'-' * 95}")

    sharpes = []

    for wname, start, end in WINDOWS_FULL:
        df_w = slice_window(df_sol, start, end)
        if len(df_w) < 100:
            print(f"  {wname:<10} {'(insufficient data)':>30}")
            if wname != "Full 3Y":
                sharpes.append(0)
            continue

        signals = strat.generate_signals(df_w)
        if signal_filter is not None:
            signals = signal_filter(signals, df_w)

        m = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                   exec_config=ec, initial_equity=INITIAL_EQUITY)

        sr = m.get("sharpe_ratio", 0)
        pnl = m.get("dollar_pnl", 0)
        dd = m.get("max_drawdown", 0)
        nb = m.get("nb_trades", 0)
        wr = m.get("win_rate", 0)
        avg_win = m.get("avg_win", 0)
        avg_loss = m.get("avg_loss", 0)
        pf = m.get("profit_factor", 0)
        final_eq = m.get("final_equity", INITIAL_EQUITY)

        if wname != "Full 3Y":
            sharpes.append(sr)

        status = ""
        if wname != "Full 3Y":
            status = " OK" if sr > 0 else " --"

        print(f"  {wname:<10} {sr:+8.2f} ${pnl:+8.0f} {dd*100:6.1f}% "
              f"{nb:7d} {wr*100:7.1f}% {avg_win*100:7.2f}% {avg_loss*100:7.2f}% "
              f"{pf:6.2f} ${final_eq:9.0f}{status}")

    n_pos = sum(1 for s in sharpes if s > 0)
    avg_sr = np.mean(sharpes) if sharpes else 0
    print(f"  {'-' * 95}")
    print(f"  Stability: {n_pos}/5 positive windows  |  Avg Sharpe (per window): {avg_sr:+.3f}")

    return sharpes


def main():
    print("Loading SOL 5m data and computing indicators...")
    df_sol = load_sol()
    print(f"  SOL 1H: {len(df_sol)} bars from {df_sol.index[0]} to {df_sol.index[-1]}")

    bt = SweepBacktester()

    ec = ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48)

    # ── Config to test: SL=1.0, TP=4.0, lookback=15, vol=2.5, no compression ──
    test_params = {
        "lookback": 15,
        "vol_breakout_min": 2.5,
        "use_compression": False,
        "sl_pct": 1.0,
        "tp_pct": 4.0,
    }

    # ── Baseline: SL=1.5, TP=4.0, vol=3.0, lookback=15, no compression ──
    baseline_params = {
        "lookback": 15,
        "vol_breakout_min": 3.0,
        "use_compression": False,
        "sl_pct": 1.5,
        "tp_pct": 4.0,
    }

    # Run without filter
    s1 = run_config(df_sol, bt, test_params, ec, "TEST: SL=1.0% TP=4.0% vol=2.5 lookback=15 (no filter)")

    # Run with anti-wick 40%
    wick_filter = filter_anti_wick(0.4)
    s2 = run_config(df_sol, bt, test_params, ec, "TEST + ANTI-WICK 40%: SL=1.0% TP=4.0% vol=2.5 lookback=15", wick_filter)

    # Baseline without filter
    s3 = run_config(df_sol, bt, baseline_params, ec, "BASELINE: SL=1.5% TP=4.0% vol=3.0 lookback=15 (no filter)")

    # Baseline with anti-wick 40%
    s4 = run_config(df_sol, bt, baseline_params, ec, "BASELINE + ANTI-WICK 40%: SL=1.5% TP=4.0% vol=3.0 lookback=15", wick_filter)

    # ── Summary comparison ──
    print("\n" + "=" * 100)
    print("  SUMMARY COMPARISON")
    print("=" * 100)
    configs = [
        ("SL=1.0 TP=4.0 vol=2.5 (raw)", s1),
        ("SL=1.0 TP=4.0 vol=2.5 + wick40", s2),
        ("SL=1.5 TP=4.0 vol=3.0 (raw)", s3),
        ("SL=1.5 TP=4.0 vol=3.0 + wick40", s4),
    ]
    print(f"  {'Config':<40} {'Stab':>5} {'AvgSR':>8}  Per-window Sharpes")
    print(f"  {'-' * 90}")
    for name, sharpes in configs:
        n_pos = sum(1 for s in sharpes if s > 0)
        avg = np.mean(sharpes) if sharpes else 0
        sr_str = "  ".join(f"{s:+.2f}" for s in sharpes)
        print(f"  {name:<40} {n_pos:>3}/5 {avg:+8.3f}  [{sr_str}]")


if __name__ == "__main__":
    main()
