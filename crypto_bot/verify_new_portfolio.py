#!/usr/bin/env python3
"""
verify_new_portfolio.py — Backtest the NEW optimized portfolio vs OLD baseline.

NEW portfolio:
  BTC: StratInsideBarBreakout, vol_min=1.5, trend_filter=True, atr_filter=True,
       sl_pct=1.5, tp_pct=3.0, ExecConfig(eq=20%, lev=5, cd=4, mh=72), hours 8-20 UTC filter
  SOL: StratBreakoutRelaxed, lookback=15, vol_breakout_min=2.5, use_compression=False,
       sl_pct=1.0, tp_pct=4.0, ExecConfig(eq=30%, lev=5, cd=4, mh=48), anti-wick 40% filter
  ETH: StratBreakoutRelaxed, lookback=30, vol_breakout_min=4.0, use_compression=False,
       sl_pct=1.5, tp_pct=4.0, ExecConfig(eq=20%, lev=5, cd=4, mh=48), anti-wick 60% filter

OLD baseline:
  BTC: same params, NO filter
  SOL: lookback=15, vol_breakout_min=3.0, sl_pct=1.5, tp_pct=4.0, same EC, NO filter
  ETH: lookback=15, vol_breakout_min=3.0, sl_pct=1.5, tp_pct=5.0, same EC, NO filter

Output:
  1. Per-asset results for each window (2023-H1, H2, 2024-H1, H2, 2025-H1, Full 3Y)
  2. Portfolio combined Sharpe, $PnL, DD, trades per window
  3. Delta vs old baseline
  4. Monte Carlo simulation on the new portfolio
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

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

INITIAL_EQUITY = 1000.0

WINDOWS = [
    ("2023-H1", "2023-01-01", "2023-07-01"),
    ("2023-H2", "2023-07-01", "2024-01-01"),
    ("2024-H1", "2024-01-01", "2024-07-01"),
    ("2024-H2", "2024-07-01", "2025-01-01"),
    ("2025-H1", "2025-01-01", "2025-07-01"),
    ("Full 3Y",  "2023-01-01", "2026-01-01"),
]

# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_asset(symbol):
    fe = FeatureEngine()
    df_5m = pd.read_parquet("data/%s_USDT_5m_ohlcv.parquet" % symbol)
    df_5m = df_5m[~df_5m.index.duplicated(keep="first")]
    df_1h = df_5m.resample("1h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["open"])
    df_1h = fe.compute_all(df_1h)
    print("  %s/USDT: %s bars 1h [%s -> %s]" %
          (symbol, f"{len(df_1h):,}", df_1h.index[0].date(), df_1h.index[-1].date()))
    return df_1h


def slice_window(df, start, end):
    mask = (df.index >= pd.Timestamp(start, tz="UTC")) & \
           (df.index < pd.Timestamp(end, tz="UTC"))
    return df.loc[mask]


# ═══════════════════════════════════════════════════════════════
# Signal filters
# ═══════════════════════════════════════════════════════════════

def filter_hours_8_20():
    """Allow signals only during hours 8-20 UTC (block 0-7 and 21-23)."""
    blocked = list(range(0, 8)) + list(range(21, 24))
    def _filter(signals, df):
        hours = df.index.hour
        mask = pd.Series(True, index=df.index)
        for h in blocked:
            mask = mask & (hours != h)
        return signals.where(mask, 0)
    return _filter


def filter_anti_wick(max_wick_ratio=0.5):
    """Block signals where the wick ratio exceeds max_wick_ratio."""
    def _filter(signals, df):
        body = (df["close"] - df["open"]).abs()
        total_range = df["high"] - df["low"]
        wick_ratio = 1 - body / total_range.replace(0, 1)
        return signals.where(wick_ratio < max_wick_ratio, 0)
    return _filter


# ═══════════════════════════════════════════════════════════════
# Portfolio definitions
# ═══════════════════════════════════════════════════════════════

NEW_PORTFOLIO = {
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

OLD_PORTFOLIO = {
    "BTC": {
        "name": "BTC InsideBar (no filter)",
        "strat_class": "StratInsideBarBreakout",
        "params": {
            "vol_min": 1.5, "trend_filter": True,
            "atr_filter": True, "sl_pct": 1.5, "tp_pct": 3.0,
        },
        "ec": ExecConfig(
            equity_pct=0.20, leverage=5,
            cooldown_bars=4, max_hold_bars=72,
        ),
        "signal_filter": None,
    },
    "SOL": {
        "name": "SOL Breakout (no filter)",
        "strat_class": "StratBreakoutRelaxed",
        "params": {
            "lookback": 15, "vol_breakout_min": 3.0,
            "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0,
        },
        "ec": ExecConfig(
            equity_pct=0.30, leverage=5,
            cooldown_bars=4, max_hold_bars=48,
        ),
        "signal_filter": None,
    },
    "ETH": {
        "name": "ETH Breakout (no filter)",
        "strat_class": "StratBreakoutRelaxed",
        "params": {
            "lookback": 15, "vol_breakout_min": 3.0,
            "use_compression": False, "sl_pct": 1.5, "tp_pct": 5.0,
        },
        "ec": ExecConfig(
            equity_pct=0.20, leverage=5,
            cooldown_bars=4, max_hold_bars=48,
        ),
        "signal_filter": None,
    },
}


# ═══════════════════════════════════════════════════════════════
# Backtesting functions
# ═══════════════════════════════════════════════════════════════

def run_single_asset(data, bt, asset, cfg, window_start, window_end):
    """Run a single asset backtest for one window. Returns metrics dict."""
    cls = V2_STRATEGY_REGISTRY[cfg["strat_class"]]
    strat = cls(cfg["params"])
    df_w = slice_window(data[asset], window_start, window_end)

    if len(df_w) < 100:
        return {
            "nb_trades": 0, "win_rate": 0, "sharpe_ratio": 0,
            "dollar_pnl": 0, "final_equity": INITIAL_EQUITY,
            "max_drawdown": 0, "trades_detail": [],
        }

    signals = strat.generate_signals(df_w)
    if cfg.get("signal_filter") is not None:
        signals = cfg["signal_filter"](signals, df_w)

    metrics = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                     exec_config=cfg["ec"], initial_equity=INITIAL_EQUITY)
    return metrics


def compute_portfolio_sharpe(all_pnls, df_ref):
    """Compute portfolio Sharpe from combined trade PnLs."""
    if len(all_pnls) < 2:
        return 0.0
    pnls_arr = np.array(all_pnls)
    if len(df_ref) > 1:
        total_days = (df_ref.index[-1] - df_ref.index[0]).total_seconds() / 86400
        trades_per_year = len(pnls_arr) / max(total_days / 365.25, 0.01)
        sharpe = (pnls_arr.mean() / pnls_arr.std(ddof=1)) * np.sqrt(trades_per_year)
        sharpe = max(-10.0, min(10.0, sharpe))
    else:
        sharpe = 0.0
    return sharpe


def run_portfolio_all_windows(data, bt, portfolio):
    """Run full portfolio across all windows. Returns per-asset and combined results."""
    assets = ["BTC", "SOL", "ETH"]
    per_asset = {a: [] for a in assets}
    combined = []

    for wname, start, end in WINDOWS:
        total_pnl = 0.0
        total_trades = 0
        total_wins = 0
        max_dd = 0.0
        all_pnls = []
        asset_details = {}

        for asset in assets:
            cfg = portfolio[asset]
            m = run_single_asset(data, bt, asset, cfg, start, end)

            pnl = m.get("dollar_pnl", 0)
            nt = m["nb_trades"]
            wr = m["win_rate"]
            sr = m.get("sharpe_ratio", 0)
            dd = m["max_drawdown"]
            fe = m.get("final_equity", INITIAL_EQUITY)

            per_asset[asset].append({
                "window": wname,
                "trades": nt,
                "win_rate": wr,
                "sharpe": sr,
                "dollar_pnl": pnl,
                "final_equity": fe,
                "max_dd": dd,
            })

            total_pnl += pnl
            total_trades += nt
            total_wins += int(wr * nt)
            max_dd = max(max_dd, dd)
            if "trades_detail" in m:
                all_pnls.extend([t["pnl_pct"] for t in m["trades_detail"]])
            asset_details[asset] = pnl

        # Portfolio Sharpe
        df_ref = slice_window(data["BTC"], start, end)
        port_sharpe = compute_portfolio_sharpe(all_pnls, df_ref)
        port_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

        combined.append({
            "window": wname,
            "trades": total_trades,
            "win_rate": port_wr,
            "sharpe": port_sharpe,
            "dollar_pnl": total_pnl,
            "final_equity": INITIAL_EQUITY + total_pnl,
            "max_dd": max_dd,
            "btc_pnl": asset_details.get("BTC", 0),
            "sol_pnl": asset_details.get("SOL", 0),
            "eth_pnl": asset_details.get("ETH", 0),
        })

    return per_asset, combined


def collect_all_trades(data, bt, portfolio):
    """Collect all trade PnLs from the Full 3Y window for Monte Carlo."""
    all_pnls = []
    for asset in ["BTC", "SOL", "ETH"]:
        cfg = portfolio[asset]
        m = run_single_asset(data, bt, asset, cfg, "2023-01-01", "2026-01-01")
        if "trades_detail" in m:
            all_pnls.extend([t["pnl_pct"] for t in m["trades_detail"]])
    return all_pnls


# ═══════════════════════════════════════════════════════════════
# Display functions
# ═══════════════════════════════════════════════════════════════

def print_per_asset(label, per_asset, portfolio):
    """Print per-asset results table."""
    print("\n" + "=" * 110)
    print("  %s — PER-ASSET RESULTS ($%d initial)" % (label, INITIAL_EQUITY))
    print("=" * 110)

    for asset in ["BTC", "SOL", "ETH"]:
        cfg = portfolio[asset]
        print("\n  -- %s: %s --" % (asset, cfg["name"]))
        print("  %-12s %6s %5s %7s %9s %9s %7s" %
              ("Window", "Trades", "WR", "Sharpe", "$PnL", "Final$", "MaxDD"))
        print("  " + "-" * 65)

        for row in per_asset[asset]:
            wr_str = "%.0f%%" % (row["win_rate"] * 100)
            dd_str = "%.1f%%" % (row["max_dd"] * 100)
            print("  %-12s %6d %5s %+7.2f %+9.2f %9.2f %7s" %
                  (row["window"], row["trades"], wr_str,
                   row["sharpe"], row["dollar_pnl"],
                   row["final_equity"], dd_str))


def print_combined(label, combined):
    """Print combined portfolio table."""
    print("\n" + "=" * 110)
    print("  %s — COMBINED PORTFOLIO ($%d initial)" % (label, INITIAL_EQUITY))
    print("=" * 110)
    print("  %-12s %6s %5s %7s %9s %9s %7s  %s" %
          ("Window", "Trades", "WR", "Sharpe", "$PnL", "Final$", "MaxDD", "Detail PnL"))
    print("  " + "-" * 105)

    for row in combined:
        detail = "BTC:%+.0f  SOL:%+.0f  ETH:%+.0f" % (
            row["btc_pnl"], row["sol_pnl"], row["eth_pnl"])
        print("  %-12s %6d %4.0f%% %+7.2f %+9.2f %9.2f %6.1f%%  %s" %
              (row["window"], row["trades"], row["win_rate"],
               row["sharpe"], row["dollar_pnl"],
               row["final_equity"], row["max_dd"] * 100, detail))


def print_delta(new_combined, old_combined):
    """Print delta comparison between NEW and OLD portfolio."""
    print("\n" + "=" * 110)
    print("  DELTA: NEW vs OLD BASELINE")
    print("=" * 110)
    print("  %-12s %8s %8s %8s %8s %8s %8s" %
          ("Window", "dSharpe", "d$PnL", "dDD%", "dTrades", "dWR%",
           "Verdict"))
    print("  " + "-" * 75)

    for n_row, o_row in zip(new_combined, old_combined):
        d_sharpe = n_row["sharpe"] - o_row["sharpe"]
        d_pnl = n_row["dollar_pnl"] - o_row["dollar_pnl"]
        d_dd = (n_row["max_dd"] - o_row["max_dd"]) * 100
        d_trades = n_row["trades"] - o_row["trades"]
        d_wr = n_row["win_rate"] - o_row["win_rate"]

        # Verdict: better if Sharpe improved or PnL improved with acceptable DD
        if d_sharpe > 0.05 and d_pnl > 0:
            verdict = "BETTER"
        elif d_sharpe > 0.05 or d_pnl > 0:
            verdict = "mixed+"
        elif d_sharpe < -0.05 and d_pnl < 0:
            verdict = "WORSE"
        else:
            verdict = "~same"

        print("  %-12s %+8.2f %+8.2f %+8.1f %+8d %+7.1f  %s" %
              (n_row["window"], d_sharpe, d_pnl, d_dd, d_trades, d_wr, verdict))


def run_monte_carlo(all_trade_pnls, n_sims=2000):
    """Run Monte Carlo simulation on the combined trade PnLs."""
    print("\n" + "=" * 110)
    print("  MONTE CARLO — NEW PORTFOLIO (%d sims, %d trades)" %
          (n_sims, len(all_trade_pnls)))
    print("=" * 110)

    if len(all_trade_pnls) < 10:
        print("  Not enough trades for Monte Carlo simulation.")
        return

    pnls_arr = np.array(all_trade_pnls)
    n_trades = len(pnls_arr)
    np.random.seed(42)

    final_returns = []
    max_drawdowns = []
    sharpes_mc = []

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

        ret = (equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100
        final_returns.append(ret)
        max_drawdowns.append(max_dd * 100)

        # Approximate Sharpe from this path
        if sample.std() > 0:
            sr = (sample.mean() / sample.std()) * np.sqrt(252)
            sharpes_mc.append(max(-10.0, min(10.0, sr)))
        else:
            sharpes_mc.append(0.0)

    fr = np.array(final_returns)
    md = np.array(max_drawdowns)
    sr_mc = np.array(sharpes_mc)

    print("\n  Input trades distribution:")
    print("    Total trades : %d" % n_trades)
    print("    Win rate     : %.1f%%" % (np.sum(pnls_arr > 0) / n_trades * 100))
    print("    Mean PnL     : %+.3f%%" % pnls_arr.mean())
    print("    Median PnL   : %+.3f%%" % np.median(pnls_arr))
    print("    Std PnL      : %.3f%%" % pnls_arr.std())
    print("    Best trade   : %+.2f%%" % pnls_arr.max())
    print("    Worst trade  : %+.2f%%" % pnls_arr.min())

    print("\n  Final return ($%d initial):" % INITIAL_EQUITY)
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        val = np.percentile(fr, pct)
        print("    P%-3d: %+8.1f%%  ($%.0f)" %
              (pct, val, INITIAL_EQUITY * (1 + val / 100)))

    print("\n  Max Drawdown:")
    print("    Median : %.1f%%" % np.median(md))
    print("    P75    : %.1f%%" % np.percentile(md, 75))
    print("    P90    : %.1f%%" % np.percentile(md, 90))
    print("    P95    : %.1f%%" % np.percentile(md, 95))
    print("    P99    : %.1f%%" % np.percentile(md, 99))

    print("\n  Sharpe ratio:")
    print("    Median : %+.2f" % np.median(sr_mc))
    print("    P25    : %+.2f" % np.percentile(sr_mc, 25))
    print("    P75    : %+.2f" % np.percentile(sr_mc, 75))

    # Risk metrics
    ruin_80 = np.sum(fr < -80) / n_sims * 100
    ruin_50 = np.sum(fr < -50) / n_sims * 100
    profitable = np.sum(fr > 0) / n_sims * 100
    double = np.sum(fr > 100) / n_sims * 100

    print("\n  Risk metrics:")
    print("    P(profit > 0%%)   : %.1f%%" % profitable)
    print("    P(return > 100%%) : %.1f%%" % double)
    print("    P(loss > 50%%)    : %.1f%%" % ruin_50)
    print("    P(loss > 80%%)    : %.1f%%" % ruin_80)

    # Calmar ratio (median return / P95 drawdown)
    p95_dd = np.percentile(md, 95)
    if p95_dd > 0:
        calmar = np.median(fr) / p95_dd
        print("    Calmar (med ret / P95 DD) : %.2f" % calmar)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("=" * 110)
    print("  VERIFY NEW PORTFOLIO vs OLD BASELINE")
    print("  3 assets x 6 windows + Monte Carlo")
    print("=" * 110)

    # ── Print configs ──
    print("\n  NEW PORTFOLIO:")
    for asset in ["BTC", "SOL", "ETH"]:
        cfg = NEW_PORTFOLIO[asset]
        ec = cfg["ec"]
        p = cfg["params"]
        filt = "hours 8-20" if asset == "BTC" else (
            "anti-wick 40%%" if asset == "SOL" else "anti-wick 60%%")
        print("    %s: %s | %s | eq=%.0f%% lev=%d cd=%d mh=%d | filter=%s" % (
            asset, cfg["strat_class"],
            " ".join("%s=%s" % (k, v) for k, v in p.items()),
            ec.equity_pct * 100, ec.leverage, ec.cooldown_bars, ec.max_hold_bars,
            filt))

    print("\n  OLD BASELINE:")
    for asset in ["BTC", "SOL", "ETH"]:
        cfg = OLD_PORTFOLIO[asset]
        ec = cfg["ec"]
        p = cfg["params"]
        print("    %s: %s | %s | eq=%.0f%% lev=%d cd=%d mh=%d | filter=none" % (
            asset, cfg["strat_class"],
            " ".join("%s=%s" % (k, v) for k, v in p.items()),
            ec.equity_pct * 100, ec.leverage, ec.cooldown_bars, ec.max_hold_bars))

    # ── Load data ──
    print("\n-- Loading data --")
    data = {}
    for sym in ["BTC", "SOL", "ETH"]:
        data[sym] = load_asset(sym)

    bt = SweepBacktester()

    # ══════════════════════════════════════════════════════════
    # 1. NEW portfolio — per-asset + combined
    # ══════════════════════════════════════════════════════════
    new_per_asset, new_combined = run_portfolio_all_windows(data, bt, NEW_PORTFOLIO)
    print_per_asset("NEW PORTFOLIO", new_per_asset, NEW_PORTFOLIO)
    print_combined("NEW PORTFOLIO", new_combined)

    # ══════════════════════════════════════════════════════════
    # 2. OLD baseline — per-asset + combined
    # ══════════════════════════════════════════════════════════
    old_per_asset, old_combined = run_portfolio_all_windows(data, bt, OLD_PORTFOLIO)
    print_per_asset("OLD BASELINE", old_per_asset, OLD_PORTFOLIO)
    print_combined("OLD BASELINE", old_combined)

    # ══════════════════════════════════════════════════════════
    # 3. Delta comparison
    # ══════════════════════════════════════════════════════════
    print_delta(new_combined, old_combined)

    # Per-asset delta for Full 3Y
    print("\n  PER-ASSET DELTA (Full 3Y):")
    print("  %-6s %8s %8s %8s %8s %8s" %
          ("Asset", "dSharpe", "d$PnL", "dDD%", "dTrades", "dWR%"))
    print("  " + "-" * 50)
    for asset in ["BTC", "SOL", "ETH"]:
        # Find Full 3Y row
        n_3y = [r for r in new_per_asset[asset] if r["window"] == "Full 3Y"][0]
        o_3y = [r for r in old_per_asset[asset] if r["window"] == "Full 3Y"][0]
        d_sr = n_3y["sharpe"] - o_3y["sharpe"]
        d_pnl = n_3y["dollar_pnl"] - o_3y["dollar_pnl"]
        d_dd = (n_3y["max_dd"] - o_3y["max_dd"]) * 100
        d_tr = n_3y["trades"] - o_3y["trades"]
        d_wr = (n_3y["win_rate"] - o_3y["win_rate"]) * 100
        print("  %-6s %+8.2f %+8.2f %+8.1f %+8d %+7.1f%%" %
              (asset, d_sr, d_pnl, d_dd, d_tr, d_wr))

    # ══════════════════════════════════════════════════════════
    # 4. Monte Carlo on new portfolio
    # ══════════════════════════════════════════════════════════
    new_trades = collect_all_trades(data, bt, NEW_PORTFOLIO)
    run_monte_carlo(new_trades, n_sims=2000)

    # ── Timing ──
    elapsed = time.time() - t0
    print("\n" + "=" * 110)
    print("  Total runtime: %.1fs (%.1f min)" % (elapsed, elapsed / 60))
    print("=" * 110)


if __name__ == "__main__":
    main()
