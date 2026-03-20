#!/usr/bin/env python3
"""
optimize_equity.py — Optimise les equity_pct du portfolio 5 assets.

Teste toutes les combinaisons d'equity_pct pour trouver le meilleur
Sharpe portfolio avec DD acceptable.

Contraintes:
  - equity_pct par asset: 5% a 35% (pas de step)
  - Total exposure (eq * lev) <= 600% (raisonnable pour 5 assets)
  - Chaque asset doit avoir au moins 5%
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
LEVERAGE = 5


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
# Asset configs (fixed strategy params, only equity_pct varies)
# ═══════════════════════════════════════════════════════════════

ASSET_CONFIGS = {
    "BTC": {
        "strat_class": "StratInsideBarBreakout",
        "params": {"vol_min": 0.8, "trend_filter": True, "atr_filter": True,
                   "sl_pct": 2.5, "tp_pct": 4.5},
        "cooldown_bars": 4, "max_hold_bars": 72,
        "signal_filter": filter_hours_8_20(),
    },
    "SOL": {
        "strat_class": "StratBreakoutRelaxed",
        "params": {"lookback": 14, "vol_breakout_min": 2.5, "use_compression": False,
                   "sl_pct": 0.9, "tp_pct": 4.0},
        "cooldown_bars": 4, "max_hold_bars": 48,
        "signal_filter": filter_anti_wick(0.4),
    },
    "ETH": {
        "strat_class": "StratBreakoutRelaxed",
        "params": {"lookback": 35, "vol_breakout_min": 4.5, "use_compression": False,
                   "sl_pct": 1.8, "tp_pct": 3.5},
        "cooldown_bars": 4, "max_hold_bars": 48,
        "signal_filter": filter_anti_wick(0.6),
    },
    "XRP": {
        "strat_class": "StratMeanReversionBB",
        "params": {"rsi_oversold": 20, "rsi_overbought": 70,
                   "bb_entry_low": 0.08, "bb_entry_high": 0.95,
                   "sl_pct": 0.7, "tp_pct": 8.0},
        "cooldown_bars": 4, "max_hold_bars": 48,
        "signal_filter": filter_anti_wick(0.5),
    },
    "BNB": {
        "strat_class": "StratBreakoutRelaxed",
        "params": {"lookback": 32, "vol_breakout_min": 0.8, "use_compression": False,
                   "sl_pct": 0.3, "tp_pct": 4.0},
        "cooldown_bars": 3, "max_hold_bars": 48,
        "signal_filter": None,
    },
}

# Equity_pct grid per asset (step 5%)
EQ_OPTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

# Pre-filter: max total exposure
MAX_TOTAL_EXPOSURE = 6.0  # 600%


def precompute_signals(data, bt):
    """Pre-compute signals once per asset to avoid redundant work."""
    cache = {}
    for asset, cfg in ASSET_CONFIGS.items():
        cls = V2_STRATEGY_REGISTRY[cfg["strat_class"]]
        strat = cls(cfg["params"])
        df_w = slice_window(data[asset], "2023-01-01", "2026-01-01")
        signals = strat.generate_signals(df_w)
        if cfg.get("signal_filter"):
            signals = cfg["signal_filter"](signals, df_w)
        cache[asset] = {
            "df": df_w,
            "signals": signals,
            "sl_pct": strat.sl_pct,
            "tp_pct": strat.tp_pct,
            "max_hold": strat.max_hold,
            "cooldown_bars": cfg["cooldown_bars"],
            "max_hold_bars": cfg["max_hold_bars"],
        }
    return cache


def run_portfolio(bt, cache, eq_pcts):
    """Run portfolio with given equity allocations. Returns metrics."""
    total_pnl = 0
    total_trades = 0
    max_dd = 0
    all_pnls = []
    asset_details = {}

    for asset, eq in eq_pcts.items():
        c = cache[asset]
        ec = ExecConfig(
            equity_pct=eq, leverage=LEVERAGE,
            cooldown_bars=c["cooldown_bars"],
            max_hold_bars=c["max_hold_bars"],
        )
        m = bt.run(c["df"], c["signals"], c["sl_pct"], c["tp_pct"], c["max_hold"],
                   exec_config=ec, initial_equity=INITIAL_EQUITY)

        pnl = m.get("dollar_pnl", 0)
        total_pnl += pnl
        total_trades += m["nb_trades"]
        max_dd = max(max_dd, m["max_drawdown"])
        if "trades_detail" in m:
            all_pnls.extend([t["pnl_pct"] for t in m["trades_detail"]])
        asset_details[asset] = {"pnl": pnl, "dd": m["max_drawdown"],
                                "trades": m["nb_trades"], "wr": m["win_rate"]}

    # Portfolio Sharpe
    if len(all_pnls) > 1:
        pa = np.array(all_pnls)
        days = (cache["BTC"]["df"].index[-1] - cache["BTC"]["df"].index[0]).total_seconds() / 86400
        tpy = len(pa) / max(days / 365.25, 0.01)
        sharpe = (pa.mean() / pa.std(ddof=1)) * np.sqrt(tpy)
        sharpe = max(-10.0, min(10.0, sharpe))
    else:
        sharpe = 0.0

    return {
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_trades": total_trades,
        "asset_details": asset_details,
    }


def run_stability(bt, cache, eq_pcts, data):
    """Check stability per semester."""
    sub_windows = [
        ("2023-H1", "2023-01-01", "2023-07-01"),
        ("2023-H2", "2023-07-01", "2024-01-01"),
        ("2024-H1", "2024-01-01", "2024-07-01"),
        ("2024-H2", "2024-07-01", "2025-01-01"),
        ("2025-H1", "2025-01-01", "2025-07-01"),
        ("2025-H2", "2025-07-01", "2026-01-01"),
    ]
    semester_pnls = []
    for wname, start, end in sub_windows:
        sem_pnl = 0
        for asset, eq in eq_pcts.items():
            cfg = ASSET_CONFIGS[asset]
            cls = V2_STRATEGY_REGISTRY[cfg["strat_class"]]
            strat = cls(cfg["params"])
            df_sw = slice_window(data[asset], start, end)
            if len(df_sw) < 50:
                continue
            signals = strat.generate_signals(df_sw)
            if cfg.get("signal_filter"):
                signals = cfg["signal_filter"](signals, df_sw)
            ec = ExecConfig(equity_pct=eq, leverage=LEVERAGE,
                            cooldown_bars=cfg["cooldown_bars"],
                            max_hold_bars=cfg["max_hold_bars"])
            m = bt.run(df_sw, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                       exec_config=ec, initial_equity=INITIAL_EQUITY)
            sem_pnl += m.get("dollar_pnl", 0)
        semester_pnls.append(sem_pnl)
    return semester_pnls


def main():
    t_start = time.time()

    print("=" * 130)
    print("  OPTIMISATION EQUITY_PCT — Portfolio 5 assets")
    print("=" * 130)

    # Generate all valid combos
    assets = ["BTC", "SOL", "ETH", "XRP", "BNB"]
    all_combos = []
    for combo in product(EQ_OPTIONS, repeat=5):
        total_exposure = sum(eq * LEVERAGE for eq in combo)
        if total_exposure <= MAX_TOTAL_EXPOSURE:
            all_combos.append(dict(zip(assets, combo)))

    print("\n  %d equity_pct options par asset: %s" % (len(EQ_OPTIONS), EQ_OPTIONS))
    print("  Max exposure: %.0f%%" % (MAX_TOTAL_EXPOSURE * 100))
    print("  Combos valides: %d (sur %d total)" % (len(all_combos), len(EQ_OPTIONS) ** 5))

    # Load data
    print("\n-- Chargement des donnees --")
    data = {}
    for sym in assets:
        data[sym] = load_asset(sym)
        print("  %s: %d barres" % (sym, len(data[sym])))

    bt = SweepBacktester()

    # Pre-compute signals
    print("\n-- Pre-calcul des signaux --")
    cache = precompute_signals(data, bt)
    for asset in assets:
        print("  %s: %d signaux non-zero" % (
            asset, (cache[asset]["signals"] != 0).sum()))

    # Run all combos
    print("\n-- Sweep %d combinaisons --" % len(all_combos))
    results = []
    t0 = time.time()

    for idx, eq_pcts in enumerate(all_combos):
        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(all_combos) - idx - 1) / rate
            print("  [%d/%d] %.0f/s — ETA %.0fs" % (idx + 1, len(all_combos), rate, eta))

        m = run_portfolio(bt, cache, eq_pcts)
        m["eq_pcts"] = eq_pcts
        m["total_exposure"] = sum(eq * LEVERAGE for eq in eq_pcts.values())
        results.append(m)

    elapsed = time.time() - t0
    print("  %d combos en %.1fs (%.0f/s)" % (len(all_combos), elapsed, len(all_combos) / elapsed))

    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    # Current baseline
    current = {"BTC": 0.20, "SOL": 0.30, "ETH": 0.20, "XRP": 0.15, "BNB": 0.15}
    current_m = run_portfolio(bt, cache, current)
    current_m["eq_pcts"] = current

    print("\n" + "=" * 130)
    print("  BASELINE ACTUEL")
    print("=" * 130)
    print("  Allocation: BTC=%.0f%% SOL=%.0f%% ETH=%.0f%% XRP=%.0f%% BNB=%.0f%%" %
          tuple(current[a] * 100 for a in assets))
    print("  Exposure: %.0f%%" % (sum(current[a] * LEVERAGE for a in assets) * 100))
    print("  $PnL: $%+.0f  Sharpe: %.2f  DD: %.1f%%  Trades: %d" %
          (current_m["total_pnl"], current_m["sharpe"], current_m["max_dd"] * 100,
           current_m["total_trades"]))

    # Top results
    print("\n" + "=" * 130)
    print("  TOP 30 — Par Sharpe portfolio")
    print("=" * 130)
    print("  %3s  %4s %4s %4s %4s %4s  %5s  %7s %8s %6s %5s" %
          ("#", "BTC", "SOL", "ETH", "XRP", "BNB", "Exp", "Sharpe", "$PnL", "DD", "Tr"))
    print("  " + "-" * 90)

    for i, r in enumerate(results[:30]):
        eq = r["eq_pcts"]
        print("  %3d  %3.0f%% %3.0f%% %3.0f%% %3.0f%% %3.0f%%  %4.0f%%  %+7.2f %+8.0f %5.1f%% %5d" %
              (i + 1,
               eq["BTC"] * 100, eq["SOL"] * 100, eq["ETH"] * 100,
               eq["XRP"] * 100, eq["BNB"] * 100,
               r["total_exposure"] * 100,
               r["sharpe"], r["total_pnl"], r["max_dd"] * 100,
               r["total_trades"]))

    # Top by PnL
    results_pnl = sorted(results, key=lambda x: x["total_pnl"], reverse=True)
    print("\n  TOP 15 — Par $PnL")
    print("  %3s  %4s %4s %4s %4s %4s  %5s  %7s %8s %6s %5s" %
          ("#", "BTC", "SOL", "ETH", "XRP", "BNB", "Exp", "Sharpe", "$PnL", "DD", "Tr"))
    print("  " + "-" * 90)

    for i, r in enumerate(results_pnl[:15]):
        eq = r["eq_pcts"]
        print("  %3d  %3.0f%% %3.0f%% %3.0f%% %3.0f%% %3.0f%%  %4.0f%%  %+7.2f %+8.0f %5.1f%% %5d" %
              (i + 1,
               eq["BTC"] * 100, eq["SOL"] * 100, eq["ETH"] * 100,
               eq["XRP"] * 100, eq["BNB"] * 100,
               r["total_exposure"] * 100,
               r["sharpe"], r["total_pnl"], r["max_dd"] * 100,
               r["total_trades"]))

    # Top by Sharpe with DD < 15%
    safe = [r for r in results if r["max_dd"] < 0.15]
    safe.sort(key=lambda x: x["sharpe"], reverse=True)
    print("\n  TOP 15 — Sharpe avec DD < 15%%")
    print("  %3s  %4s %4s %4s %4s %4s  %5s  %7s %8s %6s %5s" %
          ("#", "BTC", "SOL", "ETH", "XRP", "BNB", "Exp", "Sharpe", "$PnL", "DD", "Tr"))
    print("  " + "-" * 90)

    for i, r in enumerate(safe[:15]):
        eq = r["eq_pcts"]
        print("  %3d  %3.0f%% %3.0f%% %3.0f%% %3.0f%% %3.0f%%  %4.0f%%  %+7.2f %+8.0f %5.1f%% %5d" %
              (i + 1,
               eq["BTC"] * 100, eq["SOL"] * 100, eq["ETH"] * 100,
               eq["XRP"] * 100, eq["BNB"] * 100,
               r["total_exposure"] * 100,
               r["sharpe"], r["total_pnl"], r["max_dd"] * 100,
               r["total_trades"]))

    # Best balanced (Sharpe * PnL / DD)
    for r in results:
        if r["max_dd"] > 0:
            r["score"] = r["sharpe"] * r["total_pnl"] / (r["max_dd"] * 100)
        else:
            r["score"] = 0
    results_balanced = sorted(results, key=lambda x: x["score"], reverse=True)

    print("\n  TOP 15 — Score equilibre (Sharpe × $PnL / DD)")
    print("  %3s  %4s %4s %4s %4s %4s  %5s  %7s %8s %6s %5s  %8s" %
          ("#", "BTC", "SOL", "ETH", "XRP", "BNB", "Exp", "Sharpe", "$PnL", "DD", "Tr", "Score"))
    print("  " + "-" * 100)

    for i, r in enumerate(results_balanced[:15]):
        eq = r["eq_pcts"]
        print("  %3d  %3.0f%% %3.0f%% %3.0f%% %3.0f%% %3.0f%%  %4.0f%%  %+7.2f %+8.0f %5.1f%% %5d  %8.0f" %
              (i + 1,
               eq["BTC"] * 100, eq["SOL"] * 100, eq["ETH"] * 100,
               eq["XRP"] * 100, eq["BNB"] * 100,
               r["total_exposure"] * 100,
               r["sharpe"], r["total_pnl"], r["max_dd"] * 100,
               r["total_trades"], r["score"]))

    # Stability check on top 3 balanced
    print("\n" + "=" * 130)
    print("  STABILITE — Top 3 allocations equilibrees")
    print("=" * 130)

    for i, r in enumerate(results_balanced[:3]):
        eq = r["eq_pcts"]
        sem_pnls = run_stability(bt, cache, eq, data)
        n_pos = sum(1 for p in sem_pnls if p > 0)
        pnl_str = "  ".join("%+.0f" % p for p in sem_pnls)
        print("\n  #%d  BTC=%2.0f%% SOL=%2.0f%% ETH=%2.0f%% XRP=%2.0f%% BNB=%2.0f%%" %
              (i + 1, eq["BTC"] * 100, eq["SOL"] * 100, eq["ETH"] * 100,
               eq["XRP"] * 100, eq["BNB"] * 100))
        print("     $PnL par semestre: [%s]  Positifs: %d/6" % (pnl_str, n_pos))
        print("     Total: $%+.0f  Sharpe: %.2f  DD: %.1f%%" %
              (r["total_pnl"], r["sharpe"], r["max_dd"] * 100))

    # Also check current baseline stability
    sem_pnls_cur = run_stability(bt, cache, current, data)
    n_pos_cur = sum(1 for p in sem_pnls_cur if p > 0)
    pnl_str_cur = "  ".join("%+.0f" % p for p in sem_pnls_cur)
    print("\n  Baseline actuel:")
    print("     BTC=20%% SOL=30%% ETH=20%% XRP=15%% BNB=15%%")
    print("     $PnL par semestre: [%s]  Positifs: %d/6" % (pnl_str_cur, n_pos_cur))
    print("     Total: $%+.0f  Sharpe: %.2f  DD: %.1f%%" %
          (current_m["total_pnl"], current_m["sharpe"], current_m["max_dd"] * 100))

    # Delta
    best = results_balanced[0]
    print("\n  DELTA best vs baseline:")
    print("    Sharpe: %.2f -> %.2f (%+.2f)" %
          (current_m["sharpe"], best["sharpe"], best["sharpe"] - current_m["sharpe"]))
    print("    $PnL:   $%+.0f -> $%+.0f ($%+.0f)" %
          (current_m["total_pnl"], best["total_pnl"], best["total_pnl"] - current_m["total_pnl"]))
    print("    DD:     %.1f%% -> %.1f%% (%+.1f%%)" %
          (current_m["max_dd"] * 100, best["max_dd"] * 100,
           (best["max_dd"] - current_m["max_dd"]) * 100))

    elapsed = time.time() - t_start
    print("\n  Temps total: %.0fs (%.1f min)" % (elapsed, elapsed / 60))
    print("\n" + "=" * 130)


if __name__ == "__main__":
    main()
