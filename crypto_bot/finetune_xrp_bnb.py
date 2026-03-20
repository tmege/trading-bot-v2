#!/usr/bin/env python3
"""
finetune_xrp_bnb.py — Fine-tune XRP + Deep dive BNB.

XRP: Fine-tune autour du best sweep
  MeanReversionBB: RSI 20/70, BB 0.10/0.98, SL 1.0%, TP 8.0% + anti_wick_50
  SR +1.72, $+1,210, DD 6.6%, 6/6 stable

BNB: Deep dive — le sweep a trouve:
  1. BreakoutRelaxed: lb=30, vol=1.0, SL=0.5, TP=4.0 — SR +1.59, $+1,133, DD 10.1%, 154 configs 5/5
  2. MomentumScore: thr=1/4, SL=1.5, TP=2.0 — SR +1.68, $+334, DD 4.7%
  3. StochReversal: os=30, ob=75, vol=1.5, SL=1.0, TP=5.0 — SR +1.10, $+677
  -> Fine-tune les 3 familles avec grids tres larges pour trouver mieux.
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

def chain_filters(*filters):
    def _filter(signals, df):
        s = signals
        for f in filters:
            s = f(s, df)
        return s
    return _filter


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


def expand_grid(params_grid):
    keys = list(params_grid.keys())
    values = list(params_grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


# ═══════════════════════════════════════════════════════════════
# XRP Fine-tune — MeanReversionBB autour de RSI 20/70 BB 0.10/0.98 SL=1.0 TP=8.0
# ═══════════════════════════════════════════════════════════════

XRP_MEANREV = {
    "strat_class": "StratMeanReversionBB",
    "params_grid": {
        "rsi_oversold":   [15, 18, 20, 22, 25, 28],
        "rsi_overbought": [65, 68, 70, 72, 75],
        "bb_entry_low":   [0.05, 0.08, 0.10, 0.12, 0.15],
        "bb_entry_high":  [0.92, 0.95, 0.97, 0.98, 1.0],
        "sl_pct":         [0.7, 0.8, 1.0, 1.2, 1.5],
        "tp_pct":         [6.0, 7.0, 8.0, 9.0, 10.0],
    },
    "ec_base": ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=48),
    "signal_filter": filter_anti_wick(0.5),
    "ec_variants": [
        ExecConfig(equity_pct=0.10, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.12, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.18, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=3, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=6, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=36),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=72),
    ],
    "filter_variants": {
        "anti_wick_40": filter_anti_wick(0.4),
        "anti_wick_45": filter_anti_wick(0.45),
        "anti_wick_50": filter_anti_wick(0.5),
        "anti_wick_55": filter_anti_wick(0.55),
        "anti_wick_60": filter_anti_wick(0.6),
        "wick50+hours": chain_filters(filter_anti_wick(0.5), filter_hours_8_20()),
        "no_filter": None,
    },
}


# ═══════════════════════════════════════════════════════════════
# BNB Deep Dive — 3 familles avec grids tres larges
# ═══════════════════════════════════════════════════════════════

# BNB BreakoutRelaxed — best: lb=30, vol=1.0, SL=0.5, TP=4.0, SR +1.59, $+1133
# 154 configs 5/5 stable — la plus prometteuse pour BNB
BNB_BREAKOUT = {
    "strat_class": "StratBreakoutRelaxed",
    "params_grid": {
        "lookback":         [20, 22, 25, 27, 28, 30, 32, 35, 40],
        "vol_breakout_min": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "use_compression":  [False],
        "sl_pct":           [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2],
        "tp_pct":           [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0],
    },
    "ec_base": ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=48),
    "signal_filter": None,
    "ec_variants": [
        ExecConfig(equity_pct=0.10, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.12, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.18, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=3, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=6, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=36),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=72),
    ],
    "filter_variants": {
        "anti_wick_30": filter_anti_wick(0.3),
        "anti_wick_35": filter_anti_wick(0.35),
        "anti_wick_40": filter_anti_wick(0.4),
        "anti_wick_50": filter_anti_wick(0.5),
        "hours_8_20": filter_hours_8_20(),
        "wick40+hours": chain_filters(filter_anti_wick(0.4), filter_hours_8_20()),
        "no_filter": None,
    },
}

# BNB MomentumScore — best: thr=1/4, SL=1.5, TP=2.0, SR +1.68, $+334
BNB_MOMENTUM = {
    "strat_class": "StratMomentumScore",
    "params_grid": {
        "threshold_low":  [1, 2, 3],
        "threshold_high": [3, 4, 5],
        "sl_pct":         [0.5, 0.8, 1.0, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0],
        "tp_pct":         [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    },
    "ec_base": ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=48),
    "signal_filter": None,
    "ec_variants": [
        ExecConfig(equity_pct=0.10, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=3, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=6, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=36),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=72),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=96),
    ],
    "filter_variants": {
        "hours_8_20": filter_hours_8_20(),
        "anti_wick_40": filter_anti_wick(0.4),
        "anti_wick_50": filter_anti_wick(0.5),
        "wick40+hours": chain_filters(filter_anti_wick(0.4), filter_hours_8_20()),
        "wick50+hours": chain_filters(filter_anti_wick(0.5), filter_hours_8_20()),
        "no_filter": None,
    },
}

# BNB StochReversal — best: os=30, ob=75, vol=1.5, SL=1.0, TP=5.0, SR +1.10, $+677
BNB_STOCH = {
    "strat_class": "StratStochReversal",
    "params_grid": {
        "oversold":   [20, 25, 28, 30, 32, 35],
        "overbought": [68, 70, 72, 75, 78, 80],
        "vol_min":    [0.8, 1.0, 1.2, 1.5, 2.0, 2.5],
        "sl_pct":     [0.5, 0.7, 0.8, 1.0, 1.2, 1.5],
        "tp_pct":     [3.0, 4.0, 5.0, 5.5, 6.0, 8.0],
    },
    "ec_base": ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=48),
    "signal_filter": None,
    "ec_variants": [
        ExecConfig(equity_pct=0.10, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=3, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=6, max_hold_bars=48),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=36),
        ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=72),
    ],
    "filter_variants": {
        "anti_wick_40": filter_anti_wick(0.4),
        "anti_wick_50": filter_anti_wick(0.5),
        "hours_8_20": filter_hours_8_20(),
        "wick50+hours": chain_filters(filter_anti_wick(0.5), filter_hours_8_20()),
        "no_filter": None,
    },
}


def run_finetune(data, bt, asset, config, label=""):
    strat_class = config["strat_class"]
    combos = expand_grid(config["params_grid"])
    ec_base = config["ec_base"]
    base_filter = config["signal_filter"]

    print("\n  %s %s — %s — %d param combos" % (asset, label, strat_class, len(combos)))

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
    print("    %d param combos en %.1fs (%.0f/s)" % (len(combos), elapsed, len(combos) / max(elapsed, 0.01)))

    results.sort(key=lambda x: (x["stability"], x["sharpe_3y"]), reverse=True)
    top10 = results[:10]

    # Phase 2: EC variants
    ec_variants = config.get("ec_variants", [])
    if ec_variants and top10[0]["stability"] >= 3:
        print("    Phase 2: %d EC variants x top 10" % len(ec_variants))
        t0 = time.time()
        for rank, top_cfg in enumerate(top10):
            for ec in ec_variants:
                r = run_eval(data, bt, asset, strat_class, top_cfg["params"], ec, base_filter)
                r["params"] = top_cfg["params"]
                r["ec"] = "eq%.0f_cd%d_mh%d" % (ec.equity_pct * 100, ec.cooldown_bars, ec.max_hold_bars)
                r["filter"] = "base"
                results.append(r)
        print("    %d EC combos en %.1fs" % (10 * len(ec_variants), time.time() - t0))

    # Phase 3: Filter variants
    filter_variants = config.get("filter_variants", {})
    if filter_variants and top10[0]["stability"] >= 3:
        print("    Phase 3: %d filter variants x top 10" % len(filter_variants))
        t0 = time.time()
        for rank, top_cfg in enumerate(top10):
            for fname, filt in filter_variants.items():
                r = run_eval(data, bt, asset, strat_class, top_cfg["params"], ec_base, filt)
                r["params"] = top_cfg["params"]
                r["ec"] = "base"
                r["filter"] = fname
                results.append(r)
        print("    %d filter combos en %.1fs" % (10 * len(filter_variants), time.time() - t0))

    results.sort(key=lambda x: (x["stability"], x["sharpe_3y"]), reverse=True)
    return results


def print_top(asset, label, results, n=15):
    print("\n  TOP %d %s %s:" % (n, asset, label))
    print("  %3s %4s %7s %7s %8s %6s %4s %4s %6s  %-14s %-14s  %s" %
          ("#", "Stab", "SR 3Y", "AvgSR", "$PnL", "DD", "Tr", "WR", "PF",
           "EC", "Filter", "Params"))
    print("  " + "-" * 150)

    for i, r in enumerate(results[:n]):
        p = r["params"]
        p_parts = []
        for k, v in p.items():
            if k in ("sl_pct", "tp_pct", "use_compression"):
                continue
            if isinstance(v, bool):
                if v:
                    p_parts.append(k[:3])
            elif isinstance(v, float):
                p_parts.append("%s=%.2f" % (k[:4], v))
            else:
                p_parts.append("%s=%s" % (k[:4], v))
        p_str = " ".join(p_parts) + " sl=%.1f tp=%.1f" % (p.get("sl_pct", 0), p.get("tp_pct", 0))

        print("  %3d %4d/5 %+7.2f %+7.2f %+8.0f %5.1f%% %4d %3.0f%% %6.2f  %-14s %-14s  %s" %
              (i + 1, r["stability"], r["sharpe_3y"], r["avg_sharpe"],
               r["dollar_pnl"], r["max_drawdown"] * 100,
               r["nb_trades"], r["win_rate"] * 100, r["profit_factor"],
               r.get("ec", "base"), r.get("filter", "base"), p_str))


def evaluate_portfolio(data, bt, xrp_cfg, bnb_cfg):
    print("\n" + "=" * 130)
    print("  EVALUATION PORTFOLIO — 3 vs 4 vs 5 assets")
    print("=" * 130)

    existing = {
        "BTC": {
            "strat_class": "StratInsideBarBreakout",
            "params": {"vol_min": 0.8, "trend_filter": True, "atr_filter": True,
                       "sl_pct": 2.5, "tp_pct": 4.5},
            "ec": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=72),
            "signal_filter": filter_hours_8_20(),
        },
        "SOL": {
            "strat_class": "StratBreakoutRelaxed",
            "params": {"lookback": 14, "vol_breakout_min": 2.5, "use_compression": False,
                       "sl_pct": 0.9, "tp_pct": 4.0},
            "ec": ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48),
            "signal_filter": filter_anti_wick(0.4),
        },
        "ETH": {
            "strat_class": "StratBreakoutRelaxed",
            "params": {"lookback": 35, "vol_breakout_min": 4.5, "use_compression": False,
                       "sl_pct": 1.8, "tp_pct": 3.5},
            "ec": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48),
            "signal_filter": filter_anti_wick(0.6),
        },
    }

    def run_asset(asset, cfg):
        cls = V2_STRATEGY_REGISTRY[cfg["strat_class"]]
        strat = cls(cfg["params"])
        df_w = slice_window(data[asset], "2023-01-01", "2026-01-01")
        if len(df_w) < 100:
            return None
        signals = strat.generate_signals(df_w)
        if cfg.get("signal_filter"):
            signals = cfg["signal_filter"](signals, df_w)
        m = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                   exec_config=cfg["ec"], initial_equity=INITIAL_EQUITY)
        return m

    asset_results = {}
    for a_name, cfg in existing.items():
        m = run_asset(a_name, cfg)
        if m:
            asset_results[a_name] = m
            print("  %s: $%+.0f  Tr %d  WR %.0f%%  DD %.1f%%  SR %.2f" %
                  (a_name, m["dollar_pnl"], m["nb_trades"], m["win_rate"] * 100,
                   m["max_drawdown"] * 100, m.get("sharpe_ratio", 0)))

    if xrp_cfg:
        m = run_asset("XRP", xrp_cfg)
        if m:
            asset_results["XRP"] = m
            print("  XRP: $%+.0f  Tr %d  WR %.0f%%  DD %.1f%%  SR %.2f" %
                  (m["dollar_pnl"], m["nb_trades"], m["win_rate"] * 100,
                   m["max_drawdown"] * 100, m.get("sharpe_ratio", 0)))

    if bnb_cfg:
        m = run_asset("BNB", bnb_cfg)
        if m:
            asset_results["BNB"] = m
            print("  BNB: $%+.0f  Tr %d  WR %.0f%%  DD %.1f%%  SR %.2f" %
                  (m["dollar_pnl"], m["nb_trades"], m["win_rate"] * 100,
                   m["max_drawdown"] * 100, m.get("sharpe_ratio", 0)))

    def portfolio_stats(asset_list, results):
        total_pnl = sum(results[a]["dollar_pnl"] for a in asset_list if a in results)
        total_trades = sum(results[a]["nb_trades"] for a in asset_list if a in results)
        max_dd = max(results[a]["max_drawdown"] for a in asset_list if a in results)
        all_pnls = []
        for a in asset_list:
            if a in results and "trades_detail" in results[a]:
                all_pnls.extend([t["pnl_pct"] for t in results[a]["trades_detail"]])
        if len(all_pnls) > 1:
            pa = np.array(all_pnls)
            df_ref = slice_window(data["BTC"], "2023-01-01", "2026-01-01")
            days = (df_ref.index[-1] - df_ref.index[0]).total_seconds() / 86400
            tpy = len(pa) / max(days / 365.25, 0.01)
            sharpe = (pa.mean() / pa.std(ddof=1)) * np.sqrt(tpy)
            sharpe = max(-10.0, min(10.0, sharpe))
        else:
            sharpe = 0.0
        return total_pnl, total_trades, max_dd, sharpe

    print("\n  --- COMPARAISON ---")
    combos = [
        ("3 assets (BTC+SOL+ETH)", ["BTC", "SOL", "ETH"]),
        ("4 assets (+XRP)", ["BTC", "SOL", "ETH", "XRP"]),
        ("4 assets (+BNB)", ["BTC", "SOL", "ETH", "BNB"]),
        ("5 assets (all)", ["BTC", "SOL", "ETH", "XRP", "BNB"]),
    ]

    for label, assets in combos:
        if all(a in asset_results for a in assets):
            pnl, trades, dd, sharpe = portfolio_stats(assets, asset_results)
            print("  %-25s  $%+8.0f  Tr %4d  DD %5.1f%%  Sharpe %+.2f" %
                  (label, pnl, trades, dd * 100, sharpe))

    # Stability per semester
    for asset, cfg in [("XRP", xrp_cfg), ("BNB", bnb_cfg)]:
        if not cfg:
            continue
        print("\n  Stabilite %s par semestre:" % asset)
        cls = V2_STRATEGY_REGISTRY[cfg["strat_class"]]
        strat = cls(cfg["params"])
        sub_windows = [
            ("2023-H1", "2023-01-01", "2023-07-01"),
            ("2023-H2", "2023-07-01", "2024-01-01"),
            ("2024-H1", "2024-01-01", "2024-07-01"),
            ("2024-H2", "2024-07-01", "2025-01-01"),
            ("2025-H1", "2025-01-01", "2025-07-01"),
            ("2025-H2", "2025-07-01", "2026-01-01"),
        ]
        sharpes = []
        pnls = []
        for wname, start, end in sub_windows:
            df_sw = slice_window(data[asset], start, end)
            if len(df_sw) < 50:
                sharpes.append(0)
                pnls.append(0)
                continue
            signals_sw = strat.generate_signals(df_sw)
            if cfg.get("signal_filter"):
                signals_sw = cfg["signal_filter"](signals_sw, df_sw)
            m_sw = bt.run(df_sw, signals_sw, strat.sl_pct, strat.tp_pct, strat.max_hold,
                          exec_config=cfg["ec"], initial_equity=INITIAL_EQUITY)
            sharpes.append(m_sw.get("sharpe_ratio", 0))
            pnls.append(m_sw.get("dollar_pnl", 0))

        n_pos = sum(1 for s in sharpes if s > 0)
        sr_str = "  ".join("%+.2f" % s for s in sharpes)
        pnl_str = "  ".join("%+.0f" % p for p in pnls)
        print("    Sharpe [%s]  Stability %d/6" % (sr_str, n_pos))
        print("    $PnL   [%s]" % pnl_str)


def main():
    t_start = time.time()

    print("=" * 130)
    print("  FINE-TUNE XRP + DEEP DIVE BNB")
    print("=" * 130)

    configs = [
        ("XRP MeanRevBB", XRP_MEANREV),
        ("BNB Breakout", BNB_BREAKOUT),
        ("BNB Momentum", BNB_MOMENTUM),
        ("BNB Stoch", BNB_STOCH),
    ]
    total = 0
    for label, cfg in configs:
        n = len(expand_grid(cfg["params_grid"]))
        extra = 10 * len(cfg.get("ec_variants", [])) + 10 * len(cfg.get("filter_variants", {}))
        total += n + extra
        print("  %s: %d combos + ~%d EC/filter variants" % (label, n, extra))
    print("  TOTAL: ~%d backtests" % total)

    # Load data
    print("\n-- Chargement des donnees --")
    data = {}
    for sym in ["BTC", "SOL", "ETH", "XRP", "BNB"]:
        data[sym] = load_asset(sym)
        print("  %s: %d barres [%s -> %s]" % (
            sym, len(data[sym]),
            data[sym].index[0].date(), data[sym].index[-1].date()))

    bt = SweepBacktester()

    # ── XRP Fine-tune ──
    print("\n" + "=" * 130)
    print("  PHASE 1: FINE-TUNE XRP")
    print("=" * 130)
    xrp_results = run_finetune(data, bt, "XRP", XRP_MEANREV, "MeanRevBB")
    print_top("XRP", "MeanRevBB", xrp_results, n=15)

    # ── BNB Deep Dive ──
    print("\n" + "=" * 130)
    print("  PHASE 2: DEEP DIVE BNB — 3 familles")
    print("=" * 130)

    bnb_breakout = run_finetune(data, bt, "BNB", BNB_BREAKOUT, "Breakout")
    print_top("BNB", "BreakoutRelaxed", bnb_breakout, n=15)

    bnb_momentum = run_finetune(data, bt, "BNB", BNB_MOMENTUM, "Momentum")
    print_top("BNB", "MomentumScore", bnb_momentum, n=10)

    bnb_stoch = run_finetune(data, bt, "BNB", BNB_STOCH, "Stoch")
    print_top("BNB", "StochReversal", bnb_stoch, n=10)

    # ── Pick best BNB ──
    print("\n" + "=" * 130)
    print("  COMPARAISON BNB — Meilleure famille")
    print("=" * 130)

    bnb_families = [
        ("BreakoutRelaxed", bnb_breakout, BNB_BREAKOUT),
        ("MomentumScore", bnb_momentum, BNB_MOMENTUM),
        ("StochReversal", bnb_stoch, BNB_STOCH),
    ]

    for fname, results, cfg in bnb_families:
        best = results[0]
        n5 = sum(1 for r in results if r["stability"] == 5)
        print("  %s: %d configs 5/5 stable" % (fname, n5))
        print("    Best: Stab %d/5  SR %+.2f  $%+.0f  DD %.1f%%  Tr %d  WR %.0f%%  PF %.2f" %
              (best["stability"], best["sharpe_3y"], best["dollar_pnl"],
               best["max_drawdown"] * 100, best["nb_trades"],
               best["win_rate"] * 100, best["profit_factor"]))
        print("    Params: %s  EC=%s  Filt=%s" %
              (best["params"], best.get("ec", "base"), best.get("filter", "base")))

    # Pick overall best BNB (stability first, then Sharpe)
    all_bnb = []
    for fname, results, cfg in bnb_families:
        for r in results:
            r["family"] = fname
            r["family_cfg"] = cfg
            all_bnb.append(r)
    all_bnb.sort(key=lambda x: (x["stability"], x["sharpe_3y"]), reverse=True)

    bnb_winner = all_bnb[0]
    bnb_strat_cfg = bnb_winner["family_cfg"]
    print("\n  -> BNB winner: %s" % bnb_winner["family"])
    print("     Stab %d/5  SR %+.2f  $%+.0f  DD %.1f%%  %s" %
          (bnb_winner["stability"], bnb_winner["sharpe_3y"], bnb_winner["dollar_pnl"],
           bnb_winner["max_drawdown"] * 100, bnb_winner["params"]))

    # ── Build configs for portfolio eval ──
    xrp_best = xrp_results[0]
    xrp_cfg = {
        "strat_class": XRP_MEANREV["strat_class"],
        "params": xrp_best["params"],
        "ec": XRP_MEANREV["ec_base"],
        "signal_filter": XRP_MEANREV["signal_filter"],
    }
    # Override EC/filter if winner used different ones
    if xrp_best.get("ec", "base") != "base":
        parts = xrp_best["ec"].split("_")
        if len(parts) == 3:
            eq = float(parts[0].replace("eq", "")) / 100
            cd = int(parts[1].replace("cd", ""))
            mh = int(parts[2].replace("mh", ""))
            xrp_cfg["ec"] = ExecConfig(equity_pct=eq, leverage=5, cooldown_bars=cd, max_hold_bars=mh)
    if xrp_best.get("filter", "base") != "base":
        fname = xrp_best["filter"]
        filt_map = XRP_MEANREV.get("filter_variants", {})
        if fname in filt_map:
            xrp_cfg["signal_filter"] = filt_map[fname]

    bnb_cfg = {
        "strat_class": bnb_strat_cfg["strat_class"],
        "params": bnb_winner["params"],
        "ec": bnb_strat_cfg["ec_base"],
        "signal_filter": bnb_strat_cfg["signal_filter"],
    }
    if bnb_winner.get("ec", "base") != "base":
        parts = bnb_winner["ec"].split("_")
        if len(parts) == 3:
            eq = float(parts[0].replace("eq", "")) / 100
            cd = int(parts[1].replace("cd", ""))
            mh = int(parts[2].replace("mh", ""))
            bnb_cfg["ec"] = ExecConfig(equity_pct=eq, leverage=5, cooldown_bars=cd, max_hold_bars=mh)
    if bnb_winner.get("filter", "base") != "base":
        fname = bnb_winner["filter"]
        filt_map = bnb_strat_cfg.get("filter_variants", {})
        if fname in filt_map:
            bnb_cfg["signal_filter"] = filt_map[fname]

    # Portfolio evaluation
    evaluate_portfolio(data, bt, xrp_cfg, bnb_cfg)

    # ── Final summary ──
    elapsed = time.time() - t_start

    print("\n" + "=" * 130)
    print("  RESUME FINAL")
    print("=" * 130)
    print("\n  Temps total : %.0fs (%.1f min)" % (elapsed, elapsed / 60))

    print("\n  XRP best: %s" % XRP_MEANREV["strat_class"])
    print("    Params: %s" % xrp_best["params"])
    print("    EC: %s  Filter: %s" % (xrp_best.get("ec", "base"), xrp_best.get("filter", "base")))
    print("    Stab %d/5  SR %+.2f  $%+.0f  DD %.1f%%  Tr %d  WR %.0f%%" %
          (xrp_best["stability"], xrp_best["sharpe_3y"], xrp_best["dollar_pnl"],
           xrp_best["max_drawdown"] * 100, xrp_best["nb_trades"],
           xrp_best["win_rate"] * 100))

    print("\n  BNB best: %s (%s)" % (bnb_strat_cfg["strat_class"], bnb_winner["family"]))
    print("    Params: %s" % bnb_winner["params"])
    print("    EC: %s  Filter: %s" % (bnb_winner.get("ec", "base"), bnb_winner.get("filter", "base")))
    print("    Stab %d/5  SR %+.2f  $%+.0f  DD %.1f%%  Tr %d  WR %.0f%%" %
          (bnb_winner["stability"], bnb_winner["sharpe_3y"], bnb_winner["dollar_pnl"],
           bnb_winner["max_drawdown"] * 100, bnb_winner["nb_trades"],
           bnb_winner["win_rate"] * 100))

    print("\n" + "=" * 130)


if __name__ == "__main__":
    main()
