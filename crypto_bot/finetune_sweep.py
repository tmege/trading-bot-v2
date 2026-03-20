#!/usr/bin/env python3
"""
finetune_sweep.py — Fine-tune 10,000+ variantes proches des configs optimales.

Approche : variations granulaires autour des 3 best configs actuelles.
Chaque parametre est teste avec des pas fins dans un range etroit.

Configs actuelles :
  BTC: InsideBarBreakout vol_min=1.5 trend=T atr=T SL=1.5% TP=3.0% (hours 8-20)
  SOL: BreakoutRelaxed lb=15 vol=2.5 comp=F SL=1.0% TP=4.0% (anti-wick 40%)
  ETH: BreakoutRelaxed lb=30 vol=4.0 comp=F SL=1.5% TP=4.0% (anti-wick 60%)

Total : ~12,000+ combinaisons
"""
import sys
import time
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from exec_config import ExecConfig
from modules.feature_engine import FeatureEngine
from modules.strategies import V2_STRATEGY_REGISTRY
from sweep_runner import SweepBacktester

INITIAL_EQUITY = 1000.0

WINDOWS = [
    ("2023-H1", "2023-01-01", "2023-07-01"),
    ("2023-H2", "2023-07-01", "2024-01-01"),
    ("2024-H1", "2024-01-01", "2024-07-01"),
    ("2024-H2", "2024-07-01", "2025-01-01"),
    ("2025-H1", "2025-01-01", "2025-07-01"),
]
WINDOWS_FULL = WINDOWS + [("Full 3Y", "2023-01-01", "2026-01-01")]


# ═══════════════════════════════════════════════════════════════
# Signal filters
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Data
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
    return df_1h

def slice_window(df, start, end):
    mask = (df.index >= pd.Timestamp(start, tz="UTC")) & \
           (df.index < pd.Timestamp(end, tz="UTC"))
    return df.loc[mask]


# ═══════════════════════════════════════════════════════════════
# Core runner
# ═══════════════════════════════════════════════════════════════

def run_eval(data, bt, asset, strat_class, params, ec, signal_filter=None):
    cls = V2_STRATEGY_REGISTRY[strat_class]
    strat = cls(params)
    df_asset = data[asset]

    sharpes = []
    r3y = {}
    for wname, start, end in WINDOWS_FULL:
        df_w = slice_window(df_asset, start, end)
        if len(df_w) < 100:
            if wname != "Full 3Y":
                sharpes.append(0)
            continue
        signals = strat.generate_signals(df_w)
        if signal_filter is not None:
            signals = signal_filter(signals, df_w)
        m = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                   exec_config=ec, initial_equity=INITIAL_EQUITY)
        m.pop("trades_detail", None)
        if wname == "Full 3Y":
            r3y = m
        else:
            sharpes.append(m.get("sharpe_ratio", 0))

    n_pos = sum(1 for s in sharpes if s > 0)
    avg_sr = np.mean(sharpes) if sharpes else 0

    return {
        "sharpe_3y": r3y.get("sharpe_ratio", 0),
        "avg_sharpe": round(avg_sr, 4),
        "dollar_pnl": r3y.get("dollar_pnl", 0),
        "max_drawdown": r3y.get("max_drawdown", 0),
        "nb_trades": r3y.get("nb_trades", 0),
        "win_rate": r3y.get("win_rate", 0),
        "profit_factor": r3y.get("profit_factor", 0),
        "stability": n_pos,
        "sharpes": sharpes,
    }


# ═══════════════════════════════════════════════════════════════
# Fine-tune grids — granulaires autour des best configs
# ═══════════════════════════════════════════════════════════════

# BTC InsideBarBreakout — current: vol=1.5, trend=T, atr=T, SL=1.5, TP=3.0
BTC_FINETUNE = {
    "strat_class": "StratInsideBarBreakout",
    "params_grid": {
        "vol_min":      [0.8, 1.0, 1.2, 1.3, 1.5, 1.7, 1.8, 2.0, 2.2, 2.5],
        "trend_filter": [True, False],
        "atr_filter":   [True, False],
        "sl_pct":       [0.8, 1.0, 1.2, 1.3, 1.5, 1.7, 2.0, 2.2, 2.5],
        "tp_pct":       [2.0, 2.5, 2.8, 3.0, 3.2, 3.5, 4.0, 4.5, 5.0],
    },
    "ec_base": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=72),
    "signal_filter": filter_hours_8_20(),
    "ec_variants": [
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=72),
        ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=72),
        ExecConfig(equity_pct=0.25, leverage=5, cooldown_bars=4, max_hold_bars=72),
        ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=3, max_hold_bars=72),
        ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=6, max_hold_bars=72),
        ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=96),
    ],
    "filter_variants": {
        "hours_8_20": filter_hours_8_20(),
        "no_filter": None,
    },
}

# SOL BreakoutRelaxed — current: lb=15, vol=2.5, comp=F, SL=1.0, TP=4.0
SOL_FINETUNE = {
    "strat_class": "StratBreakoutRelaxed",
    "params_grid": {
        "lookback":         [10, 12, 13, 14, 15, 16, 17, 18, 20],
        "vol_breakout_min": [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5],
        "use_compression":  [False],
        "sl_pct":           [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5],
        "tp_pct":           [3.0, 3.5, 3.8, 4.0, 4.2, 4.5, 5.0, 5.5, 6.0],
    },
    "ec_base": ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48),
    "signal_filter": filter_anti_wick(0.4),
    "ec_variants": [
        ExecConfig(equity_pct=0.25, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.35, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=3, max_hold_bars=48),
        ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=6, max_hold_bars=48),
        ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=36),
        ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=72),
    ],
    "filter_variants": {
        "anti_wick_35": filter_anti_wick(0.35),
        "anti_wick_40": filter_anti_wick(0.4),
        "anti_wick_45": filter_anti_wick(0.45),
        "anti_wick_50": filter_anti_wick(0.5),
        "no_filter": None,
    },
}

# ETH BreakoutRelaxed — current: lb=30, vol=4.0, comp=F, SL=1.5, TP=4.0
ETH_FINETUNE = {
    "strat_class": "StratBreakoutRelaxed",
    "params_grid": {
        "lookback":         [20, 22, 25, 27, 28, 30, 32, 35, 40],
        "vol_breakout_min": [2.5, 3.0, 3.5, 3.8, 4.0, 4.2, 4.5, 5.0],
        "use_compression":  [False],
        "sl_pct":           [1.0, 1.2, 1.3, 1.5, 1.7, 1.8, 2.0, 2.5],
        "tp_pct":           [3.0, 3.5, 3.8, 4.0, 4.2, 4.5, 5.0, 5.5, 6.0],
    },
    "ec_base": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48),
    "signal_filter": filter_anti_wick(0.6),
    "ec_variants": [
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.25, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=3, max_hold_bars=48),
        ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=6, max_hold_bars=48),
        ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=36),
        ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=72),
    ],
    "filter_variants": {
        "anti_wick_50": filter_anti_wick(0.5),
        "anti_wick_55": filter_anti_wick(0.55),
        "anti_wick_60": filter_anti_wick(0.6),
        "anti_wick_65": filter_anti_wick(0.65),
        "anti_wick_70": filter_anti_wick(0.7),
        "no_filter": None,
    },
}


def expand_grid(params_grid):
    keys = list(params_grid.keys())
    values = list(params_grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def run_finetune(data, bt, asset, config):
    """Run fine-tune sweep for one asset. Returns sorted results."""
    strat_class = config["strat_class"]
    combos = expand_grid(config["params_grid"])
    ec_base = config["ec_base"]
    base_filter = config["signal_filter"]

    print("\n  %s — %s — %d param combos" % (asset, strat_class, len(combos)))

    results = []
    t0 = time.time()

    for idx, params in enumerate(combos):
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(combos) - idx - 1) / rate
            print("    [%d/%d] %.0f/s — ETA %.0fs" % (idx + 1, len(combos), rate, eta))

        r = run_eval(data, bt, asset, strat_class, params, ec_base, base_filter)
        r["params"] = params
        r["ec"] = "base"
        r["filter"] = "base"
        results.append(r)

    elapsed = time.time() - t0
    print("    %d param combos en %.1fs (%.0f/s)" % (len(combos), elapsed, len(combos) / elapsed))

    # Phase 2: EC variants on top 10 param configs
    results.sort(key=lambda x: (x["stability"], x["sharpe_3y"]), reverse=True)
    top10 = results[:10]

    ec_variants = config.get("ec_variants", [])
    if ec_variants:
        print("    Phase 2: %d EC variants x top 10 configs" % len(ec_variants))
        t0 = time.time()
        ec_results = []
        for rank, top_cfg in enumerate(top10):
            for ec in ec_variants:
                r = run_eval(data, bt, asset, strat_class, top_cfg["params"], ec, base_filter)
                r["params"] = top_cfg["params"]
                r["ec"] = "eq%.0f_cd%d_mh%d" % (ec.equity_pct * 100, ec.cooldown_bars, ec.max_hold_bars)
                r["filter"] = "base"
                r["base_rank"] = rank + 1
                ec_results.append(r)
        elapsed = time.time() - t0
        print("    %d EC combos en %.1fs" % (len(ec_results), elapsed))
        results.extend(ec_results)

    # Phase 3: Filter variants on top 10 param configs
    filter_variants = config.get("filter_variants", {})
    if filter_variants:
        print("    Phase 3: %d filter variants x top 10 configs" % len(filter_variants))
        t0 = time.time()
        filt_results = []
        for rank, top_cfg in enumerate(top10):
            for fname, filt in filter_variants.items():
                r = run_eval(data, bt, asset, strat_class, top_cfg["params"], ec_base, filt)
                r["params"] = top_cfg["params"]
                r["ec"] = "base"
                r["filter"] = fname
                r["base_rank"] = rank + 1
                filt_results.append(r)
        elapsed = time.time() - t0
        print("    %d filter combos en %.1fs" % (len(filt_results), elapsed))
        results.extend(filt_results)

    # Sort all results
    results.sort(key=lambda x: (x["stability"], x["sharpe_3y"]), reverse=True)
    return results


def print_top(asset, results, n=20):
    print("\n  TOP %d %s:" % (n, asset))
    print("  %3s %4s %7s %7s %8s %6s %4s %4s %6s  %-8s %-14s  %s" %
          ("#", "Stab", "SR 3Y", "AvgSR", "$PnL", "DD", "Tr", "WR", "PF",
           "EC", "Filter", "Params"))
    print("  " + "-" * 130)

    for i, r in enumerate(results[:n]):
        p = r["params"]
        if "lookback" in p:
            p_str = "lb=%s vol=%s sl=%s tp=%s" % (
                p.get("lookback"), p.get("vol_breakout_min"),
                p.get("sl_pct"), p.get("tp_pct"))
        else:
            p_str = "vm=%s tf=%s af=%s sl=%s tp=%s" % (
                p.get("vol_min"), p.get("trend_filter"),
                p.get("atr_filter"), p.get("sl_pct"), p.get("tp_pct"))

        print("  %3d %4d/5 %+7.2f %+7.2f %+8.0f %5.1f%% %4d %3.0f%% %6.2f  %-8s %-14s  %s" %
              (i + 1, r["stability"], r["sharpe_3y"], r["avg_sharpe"],
               r["dollar_pnl"], r["max_drawdown"] * 100,
               r["nb_trades"], r["win_rate"] * 100, r["profit_factor"],
               r.get("ec", "base"), r.get("filter", "base"), p_str))


def print_current_baseline(data, bt):
    """Print current config performance for reference."""
    print("\n  BASELINE ACTUEL:")

    configs = {
        "BTC": {
            "strat_class": "StratInsideBarBreakout",
            "params": {"vol_min": 1.5, "trend_filter": True, "atr_filter": True,
                       "sl_pct": 1.5, "tp_pct": 3.0},
            "ec": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=72),
            "filter": filter_hours_8_20(),
        },
        "SOL": {
            "strat_class": "StratBreakoutRelaxed",
            "params": {"lookback": 15, "vol_breakout_min": 2.5, "use_compression": False,
                       "sl_pct": 1.0, "tp_pct": 4.0},
            "ec": ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48),
            "filter": filter_anti_wick(0.4),
        },
        "ETH": {
            "strat_class": "StratBreakoutRelaxed",
            "params": {"lookback": 30, "vol_breakout_min": 4.0, "use_compression": False,
                       "sl_pct": 1.5, "tp_pct": 4.0},
            "ec": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48),
            "filter": filter_anti_wick(0.6),
        },
    }

    baseline = {}
    for asset, cfg in configs.items():
        r = run_eval(data, bt, asset, cfg["strat_class"], cfg["params"],
                     cfg["ec"], cfg["filter"])
        baseline[asset] = r
        print("    %s: Stab %d/5  SR %+.2f  AvgSR %+.2f  $%+.0f  DD %.1f%%  "
              "Tr %d  WR %.0f%%  PF %.2f" %
              (asset, r["stability"], r["sharpe_3y"], r["avg_sharpe"],
               r["dollar_pnl"], r["max_drawdown"] * 100,
               r["nb_trades"], r["win_rate"] * 100, r["profit_factor"]))

    return baseline


def print_improvements(asset, results, baseline_r):
    """Find configs that beat the baseline."""
    base_sr = baseline_r["sharpe_3y"]
    base_pnl = baseline_r["dollar_pnl"]
    base_stab = baseline_r["stability"]

    improved = [r for r in results
                if (r["stability"] >= base_stab and r["sharpe_3y"] > base_sr + 0.05)
                or (r["stability"] >= base_stab and r["dollar_pnl"] > base_pnl * 1.1)]

    if improved:
        print("\n  %s — %d configs qui battent le baseline (Stab>=%d, SR>%+.2f ou $PnL>$%+.0f):" %
              (asset, len(improved), base_stab, base_sr + 0.05, base_pnl * 1.1))
        for i, r in enumerate(improved[:10]):
            d_sr = r["sharpe_3y"] - base_sr
            d_pnl = r["dollar_pnl"] - base_pnl
            p = r["params"]
            print("    #%d  SR %+.2f (d%+.2f)  $%+.0f (d%+.0f)  DD %.1f%%  "
                  "Tr %d  WR %.0f%%  EC=%s  Filt=%s  %s" %
                  (i + 1, r["sharpe_3y"], d_sr, r["dollar_pnl"], d_pnl,
                   r["max_drawdown"] * 100, r["nb_trades"],
                   r["win_rate"] * 100, r.get("ec", "base"),
                   r.get("filter", "base"), p))
    else:
        print("\n  %s — Aucune config ne bat significativement le baseline." % asset)
        print("    Le baseline est deja bien optimise !")


def main():
    t_start = time.time()

    print("=" * 130)
    print("  FINE-TUNE SWEEP — 10,000+ variantes proches des configs optimales")
    print("=" * 130)

    # Count combos
    btc_n = len(expand_grid(BTC_FINETUNE["params_grid"]))
    sol_n = len(expand_grid(SOL_FINETUNE["params_grid"]))
    eth_n = len(expand_grid(ETH_FINETUNE["params_grid"]))
    btc_extra = 10 * len(BTC_FINETUNE.get("ec_variants", [])) + 10 * len(BTC_FINETUNE.get("filter_variants", {}))
    sol_extra = 10 * len(SOL_FINETUNE.get("ec_variants", [])) + 10 * len(SOL_FINETUNE.get("filter_variants", {}))
    eth_extra = 10 * len(ETH_FINETUNE.get("ec_variants", [])) + 10 * len(ETH_FINETUNE.get("filter_variants", {}))
    total = btc_n + sol_n + eth_n + btc_extra + sol_extra + eth_extra

    print("\n  BTC: %d param combos + ~%d EC/filter variants" % (btc_n, btc_extra))
    print("  SOL: %d param combos + ~%d EC/filter variants" % (sol_n, sol_extra))
    print("  ETH: %d param combos + ~%d EC/filter variants" % (eth_n, eth_extra))
    print("  TOTAL: ~%d backtests" % total)

    # Load data
    print("\n-- Chargement des donnees --")
    data = {}
    for sym in ["BTC", "SOL", "ETH"]:
        data[sym] = load_asset(sym)
        print("  %s: %d barres [%s -> %s]" % (
            sym, len(data[sym]),
            data[sym].index[0].date(), data[sym].index[-1].date()))

    bt = SweepBacktester()

    # Baseline
    baseline = print_current_baseline(data, bt)

    # Run fine-tune for each asset
    all_results = {}
    for asset, config in [("BTC", BTC_FINETUNE), ("SOL", SOL_FINETUNE), ("ETH", ETH_FINETUNE)]:
        results = run_finetune(data, bt, asset, config)
        all_results[asset] = results
        print_top(asset, results, n=20)
        print_improvements(asset, results, baseline[asset])

    # ── Summary ──
    elapsed = time.time() - t_start
    total_run = sum(len(v) for v in all_results.values())

    print("\n" + "=" * 130)
    print("  RESUME FINAL")
    print("=" * 130)
    print("\n  Temps total : %.0fs (%.1f min)" % (elapsed, elapsed / 60))
    print("  Total backtests : %d" % total_run)

    for asset in ["BTC", "SOL", "ETH"]:
        results = all_results[asset]
        base_r = baseline[asset]
        best = results[0]
        n_better = sum(1 for r in results
                       if r["stability"] >= base_r["stability"]
                       and r["sharpe_3y"] > base_r["sharpe_3y"] + 0.05)

        print("\n  %s:" % asset)
        print("    Baseline : Stab %d/5  SR %+.2f  $%+.0f  DD %.1f%%" %
              (base_r["stability"], base_r["sharpe_3y"],
               base_r["dollar_pnl"], base_r["max_drawdown"] * 100))
        print("    Best     : Stab %d/5  SR %+.2f  $%+.0f  DD %.1f%%  (%s)" %
              (best["stability"], best["sharpe_3y"],
               best["dollar_pnl"], best["max_drawdown"] * 100, best["params"]))
        d_sr = best["sharpe_3y"] - base_r["sharpe_3y"]
        d_pnl = best["dollar_pnl"] - base_r["dollar_pnl"]
        print("    Delta    : SR %+.2f  $%+.0f" % (d_sr, d_pnl))
        print("    %d configs battent le baseline" % n_better)

    # Best portfolio combination from top 3 of each
    print("\n  -- MEILLEURS PORTFOLIOS --")
    btc_top3 = all_results["BTC"][:3]
    sol_top3 = all_results["SOL"][:3]
    eth_top3 = all_results["ETH"][:3]

    port_results = []
    for bi, br in enumerate(btc_top3):
        for si, sr_val in enumerate(sol_top3):
            for ei, er in enumerate(eth_top3):
                total_pnl = br["dollar_pnl"] + sr_val["dollar_pnl"] + er["dollar_pnl"]
                max_dd = max(br["max_drawdown"], sr_val["max_drawdown"], er["max_drawdown"])
                min_stab = min(br["stability"], sr_val["stability"], er["stability"])
                avg_sr = (br["sharpe_3y"] + sr_val["sharpe_3y"] + er["sharpe_3y"]) / 3
                port_results.append({
                    "btc_idx": bi, "sol_idx": si, "eth_idx": ei,
                    "total_pnl": total_pnl, "max_dd": max_dd,
                    "min_stab": min_stab, "avg_sharpe": avg_sr,
                    "btc_pnl": br["dollar_pnl"], "sol_pnl": sr_val["dollar_pnl"],
                    "eth_pnl": er["dollar_pnl"],
                })

    port_results.sort(key=lambda x: (x["min_stab"], x["avg_sharpe"]), reverse=True)
    base_total = sum(baseline[a]["dollar_pnl"] for a in ["BTC", "SOL", "ETH"])

    print("  %-5s %5s %7s %8s %6s  %s" %
          ("#", "Stab", "AvgSR", "$PnL", "DD", "Detail"))
    for i, p in enumerate(port_results[:5]):
        d_pnl = p["total_pnl"] - base_total
        print("  #%-4d %4d+ %+7.2f %+8.0f %5.1f%%  BTC:%+.0f SOL:%+.0f ETH:%+.0f  (d$%+.0f)" %
              (i + 1, p["min_stab"], p["avg_sharpe"], p["total_pnl"],
               p["max_dd"] * 100, p["btc_pnl"], p["sol_pnl"], p["eth_pnl"], d_pnl))

    print("\n" + "=" * 130)


if __name__ == "__main__":
    main()
