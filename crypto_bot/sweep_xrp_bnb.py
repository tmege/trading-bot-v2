#!/usr/bin/env python3
"""
sweep_xrp_bnb.py — Sweep de strategies pour XRP et BNB.

Teste les 7 strategies V2 sur XRP et BNB avec les memes grids que le mega sweep.
Evalue ensuite si les ajouter au portfolio 3 assets existant est benefique.

Portfolio actuel:
  BTC: InsideBarBreakout vol=0.8 trend=T atr=T SL=2.5% TP=4.5% (hours 8-20)
  SOL: BreakoutRelaxed lb=14 vol=2.5 SL=0.9% TP=4.0% (anti-wick 40%)
  ETH: BreakoutRelaxed lb=35 vol=4.5 SL=1.8% TP=3.5% (anti-wick 60%)
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

def chain_filters(*filters):
    def _filter(signals, df):
        s = signals
        for f in filters:
            s = f(s, df)
        return s
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


def expand_grid(params_grid):
    keys = list(params_grid.keys())
    values = list(params_grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


# ═══════════════════════════════════════════════════════════════
# Strategy grids for XRP and BNB
# ═══════════════════════════════════════════════════════════════

# EC presets to test
EC_PRESETS = {
    "conservative": ExecConfig(equity_pct=0.15, leverage=5, cooldown_bars=4, max_hold_bars=48),
    "moderate":     ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48),
    "normal":       ExecConfig(equity_pct=0.25, leverage=5, cooldown_bars=4, max_hold_bars=48),
}

# All strategy grids — EXPANDED to 12,000+ combos per asset
STRATEGY_GRIDS = {
    "StratBreakoutRelaxed": {
        "lookback":         [6, 8, 10, 12, 14, 16, 18, 20, 25, 30],
        "vol_breakout_min": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        "use_compression":  [False],
        "sl_pct":           [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5],
        "tp_pct":           [1.5, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0],
    },
    "StratInsideBarBreakout": {
        "vol_min":      [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5],
        "trend_filter": [True, False],
        "atr_filter":   [True, False],
        "sl_pct":       [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0],
        "tp_pct":       [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    },
    "StratMomentumScore": {
        "threshold_low":  [1, 2, 3],
        "threshold_high": [3, 4, 5],
        "sl_pct":         [0.5, 0.8, 1.0, 1.5, 2.0, 2.5],
        "tp_pct":         [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    },
    "StratMeanReversionBB": {
        "rsi_oversold":   [20, 25, 30, 35, 40],
        "rsi_overbought": [60, 65, 70, 75, 80],
        "bb_entry_low":   [0.03, 0.05, 0.10],
        "bb_entry_high":  [0.90, 0.95, 0.98],
        "sl_pct":         [1.0, 1.5, 2.0, 2.5, 3.0],
        "tp_pct":         [3.0, 4.0, 5.0, 6.0, 8.0],
    },
    "StratStochReversal": {
        "oversold":   [10, 15, 20, 25, 30],
        "overbought": [70, 75, 80, 85, 90],
        "vol_min":    [0.5, 0.8, 1.0, 1.5, 2.0, 2.5],
        "sl_pct":     [0.5, 0.8, 1.0, 1.5, 2.0, 2.5],
        "tp_pct":     [3.0, 4.0, 5.0, 6.0, 8.0],
    },
    "StratRegimeAdaptive": {
        "rsi_bull":         [45, 50, 55, 60],
        "rsi_range_low":    [20, 25, 30, 35],
        "rsi_range_high":   [65, 70, 75, 80],
        "use_ranging_only": [True, False],
        "sl_pct":           [1.0, 1.5, 2.0, 2.5],
        "tp_pct":           [3.0, 4.0, 5.0, 6.0, 8.0],
    },
    "StratEmaCrossover": {
        "ema_fast":          [5, 9, 12, 15, 21],
        "ema_slow":          [21, 34, 50],
        "use_regime_filter": [True, False],
        "sl_buffer_pct":     [0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
        "tp_pct":            [2.0, 3.0, 5.0, 8.0, 10.0],
    },
}

SIGNAL_FILTERS = {
    "none": None,
    "anti_wick_40": filter_anti_wick(0.4),
    "anti_wick_50": filter_anti_wick(0.5),
    "anti_wick_60": filter_anti_wick(0.6),
    "hours_8_20": filter_hours_8_20(),
    "wick50+hours": chain_filters(filter_anti_wick(0.5), filter_hours_8_20()),
}


def run_full_sweep(data, bt, asset):
    """Run all strategy grids on an asset. Returns sorted results."""
    print("\n" + "=" * 130)
    print("  SWEEP %s — Toutes strategies" % asset)
    print("=" * 130)

    all_results = []
    ec = EC_PRESETS["moderate"]  # Start with moderate EC

    for strat_name, grid in STRATEGY_GRIDS.items():
        combos = expand_grid(grid)
        print("\n  %s — %s — %d combos" % (asset, strat_name, len(combos)))
        t0 = time.time()

        for idx, params in enumerate(combos):
            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (len(combos) - idx - 1) / rate
                print("    [%d/%d] %.0f/s — ETA %.0fs" % (idx + 1, len(combos), rate, eta))

            r = run_eval(data, bt, asset, strat_name, params, ec)
            r["params"] = params
            r["strat_class"] = strat_name
            r["ec"] = "moderate"
            r["filter"] = "none"
            all_results.append(r)

        elapsed = time.time() - t0
        # Stats
        strat_results = [r for r in all_results if r["strat_class"] == strat_name]
        n_stable5 = sum(1 for r in strat_results if r["stability"] == 5)
        n_stable4 = sum(1 for r in strat_results if r["stability"] >= 4)
        best = max(strat_results, key=lambda x: (x["stability"], x["sharpe_3y"]))
        print("    %d combos en %.1fs — %d stable 5/5, %d stable 4+/5" %
              (len(combos), elapsed, n_stable5, n_stable4))
        if best["stability"] >= 3:
            print("    Best: Stab %d/5  SR %+.2f  $%+.0f  DD %.1f%%  WR %.0f%%  %s" %
                  (best["stability"], best["sharpe_3y"], best["dollar_pnl"],
                   best["max_drawdown"] * 100, best["win_rate"] * 100, best["params"]))

    # Sort overall
    all_results.sort(key=lambda x: (x["stability"], x["sharpe_3y"]), reverse=True)

    # Phase 2: EC variants and filters on top 10
    top10 = all_results[:10]

    if top10 and top10[0]["stability"] >= 3:
        print("\n  Phase 2: EC variants + filtres sur top 10")
        phase2_results = []
        t0 = time.time()

        for rank, cfg in enumerate(top10):
            # EC variants
            for ec_name, ec_val in EC_PRESETS.items():
                r = run_eval(data, bt, asset, cfg["strat_class"], cfg["params"], ec_val)
                r["params"] = cfg["params"]
                r["strat_class"] = cfg["strat_class"]
                r["ec"] = ec_name
                r["filter"] = "none"
                r["base_rank"] = rank + 1
                phase2_results.append(r)

            # Signal filters
            for fname, filt in SIGNAL_FILTERS.items():
                if fname == "none":
                    continue
                r = run_eval(data, bt, asset, cfg["strat_class"], cfg["params"],
                             EC_PRESETS["moderate"], filt)
                r["params"] = cfg["params"]
                r["strat_class"] = cfg["strat_class"]
                r["ec"] = "moderate"
                r["filter"] = fname
                r["base_rank"] = rank + 1
                phase2_results.append(r)

        elapsed = time.time() - t0
        print("    %d combos en %.1fs" % (len(phase2_results), elapsed))
        all_results.extend(phase2_results)
        all_results.sort(key=lambda x: (x["stability"], x["sharpe_3y"]), reverse=True)

    return all_results


def print_top(asset, results, n=20):
    print("\n  TOP %d %s:" % (n, asset))
    print("  %3s %22s %4s %7s %7s %8s %6s %4s %4s %6s  %-10s %-12s  %s" %
          ("#", "Strategy", "Stab", "SR 3Y", "AvgSR", "$PnL", "DD", "Tr", "WR", "PF",
           "EC", "Filter", "Key Params"))
    print("  " + "-" * 150)

    for i, r in enumerate(results[:n]):
        p = r["params"]
        # Build short param string
        p_parts = []
        for k, v in p.items():
            if k in ("sl_pct", "tp_pct"):
                continue
            if isinstance(v, bool):
                if v:
                    p_parts.append(k[:3])
            elif isinstance(v, float):
                p_parts.append("%s=%.1f" % (k[:3], v))
            else:
                p_parts.append("%s=%s" % (k[:3], v))
        p_str = " ".join(p_parts) + " sl=%.1f tp=%.1f" % (p.get("sl_pct", 0), p.get("tp_pct", 0))

        print("  %3d %22s %4d/5 %+7.2f %+7.2f %+8.0f %5.1f%% %4d %3.0f%% %6.2f  %-10s %-12s  %s" %
              (i + 1, r["strat_class"][:22], r["stability"], r["sharpe_3y"], r["avg_sharpe"],
               r["dollar_pnl"], r["max_drawdown"] * 100,
               r["nb_trades"], r["win_rate"] * 100, r["profit_factor"],
               r.get("ec", ""), r.get("filter", ""), p_str))


def evaluate_portfolio_addition(data, bt, asset, best_config):
    """Test if adding this asset to the existing portfolio is beneficial."""
    print("\n" + "=" * 130)
    print("  EVALUATION PORTFOLIO — Ajout de %s" % asset)
    print("=" * 130)

    # Current portfolio configs
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

    # Run existing portfolio
    print("\n  -- Portfolio actuel (3 assets) --")
    port_3 = {"pnls": [], "trades": 0, "max_dd": 0, "asset_pnls": {}}

    for a_name, cfg in existing.items():
        if a_name not in data:
            continue
        cls = V2_STRATEGY_REGISTRY[cfg["strat_class"]]
        strat = cls(cfg["params"])
        df_w = slice_window(data[a_name], "2023-01-01", "2026-01-01")
        if len(df_w) < 100:
            continue
        signals = strat.generate_signals(df_w)
        if cfg.get("signal_filter"):
            signals = cfg["signal_filter"](signals, df_w)
        m = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                   exec_config=cfg["ec"], initial_equity=INITIAL_EQUITY)
        pnl = m.get("dollar_pnl", 0)
        port_3["asset_pnls"][a_name] = pnl
        port_3["trades"] += m["nb_trades"]
        port_3["max_dd"] = max(port_3["max_dd"], m["max_drawdown"])
        if "trades_detail" in m:
            port_3["pnls"].extend([t["pnl_pct"] for t in m["trades_detail"]])
        print("    %s: $%+.0f  Tr %d  WR %.0f%%  DD %.1f%%" %
              (a_name, pnl, m["nb_trades"], m["win_rate"] * 100, m["max_drawdown"] * 100))

    port_3_total = sum(port_3["asset_pnls"].values())
    print("    TOTAL: $%+.0f  Tr %d  DD %.1f%%" %
          (port_3_total, port_3["trades"], port_3["max_dd"] * 100))

    # Run with new asset added
    print("\n  -- Portfolio + %s (4 assets) --" % asset)
    new_cfg = best_config

    cls = V2_STRATEGY_REGISTRY[new_cfg["strat_class"]]
    strat = cls(new_cfg["params"])
    df_w = slice_window(data[asset], "2023-01-01", "2026-01-01")
    signals = strat.generate_signals(df_w)
    if new_cfg.get("signal_filter"):
        signals = new_cfg["signal_filter"](signals, df_w)
    m = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
               exec_config=new_cfg["ec"], initial_equity=INITIAL_EQUITY)

    new_pnl = m.get("dollar_pnl", 0)
    new_trades = m["nb_trades"]
    new_dd = m["max_drawdown"]
    all_pnls = port_3["pnls"][:]
    if "trades_detail" in m:
        all_pnls.extend([t["pnl_pct"] for t in m["trades_detail"]])

    print("    %s: $%+.0f  Tr %d  WR %.0f%%  DD %.1f%%" %
          (asset, new_pnl, new_trades, m["win_rate"] * 100, new_dd * 100))

    port_4_total = port_3_total + new_pnl
    port_4_trades = port_3["trades"] + new_trades
    port_4_dd = max(port_3["max_dd"], new_dd)

    # Compute portfolio Sharpe
    if len(all_pnls) > 1:
        pa = np.array(all_pnls)
        df_ref = slice_window(data["BTC"], "2023-01-01", "2026-01-01")
        days = (df_ref.index[-1] - df_ref.index[0]).total_seconds() / 86400
        tpy = len(pa) / max(days / 365.25, 0.01)
        sharpe = (pa.mean() / pa.std(ddof=1)) * np.sqrt(tpy)
        sharpe = max(-10.0, min(10.0, sharpe))
    else:
        sharpe = 0.0

    print("\n    PORTFOLIO 4 assets: $%+.0f  Tr %d  DD %.1f%%  Sharpe %.2f" %
          (port_4_total, port_4_trades, port_4_dd * 100, sharpe))
    print("    Delta vs 3 assets: $%+.0f  %+d trades  DD %+.1f%%" %
          (new_pnl, new_trades, (port_4_dd - port_3["max_dd"]) * 100))

    # Stability check on the new asset alone
    print("\n  -- Stabilite %s par semestre --" % asset)
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
        if new_cfg.get("signal_filter"):
            signals_sw = new_cfg["signal_filter"](signals_sw, df_sw)
        m_sw = bt.run(df_sw, signals_sw, strat.sl_pct, strat.tp_pct, strat.max_hold,
                      exec_config=new_cfg["ec"], initial_equity=INITIAL_EQUITY)
        sharpes.append(m_sw.get("sharpe_ratio", 0))
        pnls.append(m_sw.get("dollar_pnl", 0))

    n_pos = sum(1 for s in sharpes if s > 0)
    sr_str = "  ".join("%+.2f" % s for s in sharpes)
    pnl_str = "  ".join("%+.0f" % p for p in pnls)
    print("    Sharpe [%s]  Stability %d/6" % (sr_str, n_pos))
    print("    $PnL   [%s]" % pnl_str)

    verdict = "AJOUTER" if n_pos >= 4 and new_pnl > 100 else "REJETER" if n_pos < 3 or new_pnl < 0 else "A EVALUER"
    print("\n    VERDICT: %s" % verdict)

    return {
        "asset": asset,
        "config": new_cfg,
        "pnl": new_pnl,
        "trades": new_trades,
        "dd": new_dd,
        "stability": n_pos,
        "sharpes": sharpes,
        "verdict": verdict,
    }


def main():
    t_start = time.time()

    print("=" * 130)
    print("  SWEEP XRP + BNB — Recherche de strategies pour le portfolio")
    print("=" * 130)

    # Count total combos
    total = sum(len(expand_grid(g)) for g in STRATEGY_GRIDS.values())
    print("\n  %d combos par asset x 2 assets = %d backtests (+ phase 2)" % (total, total * 2))

    # Load data
    print("\n-- Chargement des donnees --")
    data = {}
    for sym in ["BTC", "SOL", "ETH", "XRP", "BNB"]:
        try:
            data[sym] = load_asset(sym)
            print("  %s: %d barres [%s -> %s]" % (
                sym, len(data[sym]),
                data[sym].index[0].date(), data[sym].index[-1].date()))
        except FileNotFoundError:
            print("  %s: FICHIER MANQUANT" % sym)

    bt = SweepBacktester()

    # Run sweep for each new asset
    best_configs = {}

    for asset in ["XRP", "BNB"]:
        if asset not in data:
            print("\n  SKIP %s — pas de donnees" % asset)
            continue

        results = run_full_sweep(data, bt, asset)
        print_top(asset, results, n=20)

        # Extract best config for portfolio evaluation
        if results and results[0]["stability"] >= 3:
            best = results[0]
            # Build config dict
            filt = SIGNAL_FILTERS.get(best.get("filter", "none"))
            cfg = {
                "strat_class": best["strat_class"],
                "params": best["params"],
                "ec": EC_PRESETS.get(best.get("ec", "moderate"), EC_PRESETS["moderate"]),
                "signal_filter": filt,
            }
            best_configs[asset] = cfg

    # Portfolio evaluation
    print("\n" + "=" * 130)
    print("  EVALUATION PORTFOLIO — Faut-il ajouter XRP / BNB ?")
    print("=" * 130)

    evaluations = {}
    for asset, cfg in best_configs.items():
        evaluations[asset] = evaluate_portfolio_addition(data, bt, asset, cfg)

    # Final summary
    elapsed = time.time() - t_start
    print("\n" + "=" * 130)
    print("  RESUME FINAL")
    print("=" * 130)
    print("\n  Temps total: %.0fs (%.1f min)" % (elapsed, elapsed / 60))

    for asset, ev in evaluations.items():
        print("\n  %s: %s" % (asset, ev["verdict"]))
        print("    PnL $%+.0f  Trades %d  DD %.1f%%  Stability %d/6" %
              (ev["pnl"], ev["trades"], ev["dd"] * 100, ev["stability"]))
        print("    Best config: %s %s" % (ev["config"]["strat_class"], ev["config"]["params"]))

    print("\n" + "=" * 130)


if __name__ == "__main__":
    main()
