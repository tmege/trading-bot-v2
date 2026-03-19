#!/usr/bin/env python3
"""
Mega Sweep — 10,000+ variantes d'optimisation du portfolio V2.

4 phases :
  Phase 1 : Param sweep exhaustif par stratégie principale (BTC/SOL/ETH)
  Phase 2 : Stratégies alternatives sur chaque asset
  Phase 3 : ExecConfig sweep sur les meilleurs configs
  Phase 4 : Filtres de signaux sur les meilleurs configs

Résultat : top configs par asset + meilleur portfolio combiné.
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

# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Signal filters
# ═══════════════════════════════════════════════════════════════

def filter_anti_wick(max_wick_ratio=0.5):
    def _filter(signals, df):
        body = (df["close"] - df["open"]).abs()
        total_range = df["high"] - df["low"]
        wick_ratio = 1 - body / total_range.replace(0, 1)
        return signals.where(wick_ratio < max_wick_ratio, 0)
    return _filter

def filter_hours(blocked_hours):
    def _filter(signals, df):
        hours = df.index.hour
        mask = pd.Series(True, index=df.index)
        for h in blocked_hours:
            mask = mask & (hours != h)
        return signals.where(mask, 0)
    return _filter

def filter_weekdays(blocked_days):
    def _filter(signals, df):
        days = df.index.dayofweek
        mask = pd.Series(True, index=df.index)
        for d in blocked_days:
            mask = mask & (days != d)
        return signals.where(mask, 0)
    return _filter

def filter_rsi_confirm():
    def _filter(signals, df):
        rsi = df.get("RSI_14")
        if rsi is None:
            return signals
        out = signals.copy()
        out.loc[(signals == 1) & (rsi > 70)] = 0
        out.loc[(signals == -1) & (rsi < 30)] = 0
        return out
    return _filter

def filter_adx_trending(min_adx=20):
    def _filter(signals, df):
        adx = df.get("ADX_14")
        if adx is None:
            return signals
        return signals.where(adx > min_adx, 0)
    return _filter

def filter_regime_aligned():
    def _filter(signals, df):
        regime = df.get("regime")
        if regime is None:
            return signals
        out = signals.copy()
        out.loc[(signals == 1) & (regime == "bear")] = 0
        out.loc[(signals == -1) & (regime == "bull")] = 0
        return out
    return _filter

def filter_bb_position():
    def _filter(signals, df):
        pct_b = df.get("pct_B")
        if pct_b is None:
            return signals
        out = signals.copy()
        out.loc[(signals == 1) & (pct_b > 0.7)] = 0
        out.loc[(signals == -1) & (pct_b < 0.3)] = 0
        return out
    return _filter

def filter_volume_spike(min_ratio=2.0):
    def _filter(signals, df):
        vr = df.get("volume_ratio")
        if vr is None:
            return signals
        return signals.where(vr >= min_ratio, 0)
    return _filter

def chain_filters(*filters):
    def _filter(signals, df):
        s = signals
        for f in filters:
            s = f(s, df)
        return s
    return _filter


# ═══════════════════════════════════════════════════════════════
# Core runner
# ═══════════════════════════════════════════════════════════════

def run_eval(data, bt, asset, strat_class, params, ec, signal_filter=None):
    """Run une config sur toutes les fenêtres, retourne métriques de stabilité."""
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
        "stability": n_pos,
        "sharpes": sharpes,
    }


# ═══════════════════════════════════════════════════════════════
# Phase 1 : Param sweep exhaustif
# ═══════════════════════════════════════════════════════════════

BTC_GRID = {
    "strat_class": "StratInsideBarBreakout",
    "params_grid": {
        "vol_min": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "trend_filter": [True, False],
        "atr_filter": [True, False],
        "sl_pct": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "tp_pct": [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    },
    "ec": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=72),
}

SOL_GRID = {
    "strat_class": "StratBreakoutRelaxed",
    "params_grid": {
        "lookback": [5, 8, 10, 12, 15, 20, 25, 30],
        "vol_breakout_min": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        "use_compression": [True, False],
        "sl_pct": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "tp_pct": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    },
    "ec": ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48),
}

ETH_GRID = {
    "strat_class": "StratBreakoutRelaxed",
    "params_grid": {
        "lookback": [5, 8, 10, 12, 15, 20, 25, 30],
        "vol_breakout_min": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        "use_compression": [True, False],
        "sl_pct": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "tp_pct": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    },
    "ec": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48),
}


def expand_grid(params_grid):
    """Expand param grid dict into list of param dicts."""
    keys = list(params_grid.keys())
    values = list(params_grid.values())
    combos = []
    for combo in product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def run_phase1(data, bt):
    """Phase 1: Sweep exhaustif des params de stratégie principale."""
    print("\n" + "=" * 120)
    print("  PHASE 1 : PARAM SWEEP EXHAUSTIF — Stratégies principales")
    print("=" * 120)

    phase1_results = {}

    for asset, grid_cfg in [("BTC", BTC_GRID), ("SOL", SOL_GRID), ("ETH", ETH_GRID)]:
        combos = expand_grid(grid_cfg["params_grid"])
        n = len(combos)
        print(f"\n  {asset} — {grid_cfg['strat_class']} — {n} combinaisons")

        results = []
        t0 = time.time()

        for idx, params in enumerate(combos):
            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (n - idx - 1) / rate
                print(f"    [{idx+1}/{n}] {rate:.0f} combos/s — ETA {eta:.0f}s")

            r = run_eval(data, bt, asset, grid_cfg["strat_class"], params, grid_cfg["ec"])
            r["params"] = params
            r["strat_class"] = grid_cfg["strat_class"]
            results.append(r)

        elapsed = time.time() - t0
        print(f"    Terminé en {elapsed:.1f}s ({n/elapsed:.0f} combos/s)")

        # Trier par stabilité puis sharpe 3Y
        results.sort(key=lambda x: (x["stability"], x["sharpe_3y"]), reverse=True)
        phase1_results[asset] = results

        # Top 10
        print(f"\n  Top 10 {asset}:")
        print(f"  {'#':>3} {'Stab':>4} {'SR 3Y':>7} {'AvgSR':>7} {'$PnL':>8} {'DD':>6} {'Tr':>4} {'WR':>4}  Params")
        for i, r in enumerate(results[:10]):
            p = r["params"]
            p_str = " ".join(f"{k}={v}" for k, v in p.items() if k not in ("sl_pct", "tp_pct"))
            print(f"  {i+1:3d} {r['stability']:>4}/5 {r['sharpe_3y']:+7.2f} {r['avg_sharpe']:+7.2f} "
                  f"${r['dollar_pnl']:+7.0f} {r['max_drawdown']*100:5.1f}% {r['nb_trades']:4d} "
                  f"{r['win_rate']*100:3.0f}%  SL={p['sl_pct']} TP={p['tp_pct']} {p_str}")

        # Stats de stabilité
        n_stable_5 = sum(1 for r in results if r["stability"] == 5)
        n_stable_4 = sum(1 for r in results if r["stability"] >= 4)
        print(f"\n  {asset}: {n_stable_5} configs 5/5, {n_stable_4} configs 4+/5 sur {n}")

    return phase1_results


# ═══════════════════════════════════════════════════════════════
# Phase 2 : Stratégies alternatives
# ═══════════════════════════════════════════════════════════════

ALT_STRATEGIES = {
    "StratMomentumScore": [
        {"threshold_low": tl, "threshold_high": th, "sl_pct": sl, "tp_pct": tp}
        for tl in [1, 2]
        for th in [3, 4]
        for sl in [1.0, 1.5, 2.0, 2.5]
        for tp in [3.0, 4.0, 5.0, 6.0, 8.0]
    ],
    "StratEmaCrossover": [
        {"ema_fast": ef, "ema_slow": es, "use_regime_filter": rf, "sl_buffer_pct": sb, "tp_pct": tp}
        for ef, es in [(9, 21), (9, 50), (12, 26), (21, 50)]
        for rf in [True, False]
        for sb in [0.5, 1.0, 1.5]
        for tp in [3.0, 5.0, 8.0, 10.0]
    ],
    "StratMeanReversionBB": [
        {"rsi_oversold": ro, "rsi_overbought": rb, "bb_entry_low": bl, "bb_entry_high": bh, "sl_pct": sl, "tp_pct": tp}
        for ro in [20, 25, 30, 35]
        for rb in [65, 70, 75, 80]
        for bl in [0.05, 0.10, 0.15]
        for bh in [0.85, 0.90, 0.95]
        for sl in [1.5, 2.0, 2.5]
        for tp in [3.0, 4.0, 5.0, 6.0]
    ],
    "StratStochReversal": [
        {"oversold": os_val, "overbought": ob, "vol_min": vm, "sl_pct": sl, "tp_pct": tp}
        for os_val in [15, 20, 25, 30]
        for ob in [70, 75, 80, 85]
        for vm in [0.5, 1.0, 1.5, 2.0]
        for sl in [1.0, 1.5, 2.0, 2.5]
        for tp in [3.0, 4.0, 5.0, 6.0, 8.0]
    ],
    "StratRegimeAdaptive": [
        {"rsi_bull": rb, "rsi_range_low": rl, "rsi_range_high": rh,
         "use_ranging_only": uro, "sl_pct": sl, "tp_pct": tp}
        for rb in [50, 55, 60]
        for rl in [25, 30, 35]
        for rh in [65, 70, 75]
        for uro in [True, False]
        for sl in [1.5, 2.0, 2.5]
        for tp in [3.0, 4.0, 5.0, 6.0]
    ],
}

# ExecConfigs par asset pour les alternatives
ALT_EC = {
    "BTC": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=72),
    "SOL": ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48),
    "ETH": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48),
}


def run_phase2(data, bt):
    """Phase 2: Tester les stratégies alternatives sur chaque asset."""
    print("\n" + "=" * 120)
    print("  PHASE 2 : STRATÉGIES ALTERNATIVES — Chaque asset × 5 strats")
    print("=" * 120)

    # Exclure la strat principale de chaque asset
    primary = {"BTC": "StratInsideBarBreakout", "SOL": "StratBreakoutRelaxed", "ETH": "StratBreakoutRelaxed"}

    phase2_results = {}

    for asset in ["BTC", "SOL", "ETH"]:
        ec = ALT_EC[asset]
        asset_results = []
        total_combos = 0

        for strat_name, param_list in ALT_STRATEGIES.items():
            if strat_name == primary[asset]:
                continue
            # Also skip InsideBarBreakout for SOL/ETH and BreakoutRelaxed for BTC
            # since they're handled in Phase 1

            n = len(param_list)
            total_combos += n

            for params in param_list:
                r = run_eval(data, bt, asset, strat_name, params, ec)
                r["params"] = params
                r["strat_class"] = strat_name
                asset_results.append(r)

        asset_results.sort(key=lambda x: (x["stability"], x["sharpe_3y"]), reverse=True)
        phase2_results[asset] = asset_results

        print(f"\n  {asset} — {total_combos} combos alternatives")
        # Top 5
        if asset_results:
            print(f"  {'#':>3} {'Strat':>22} {'Stab':>4} {'SR 3Y':>7} {'AvgSR':>7} {'$PnL':>8} {'DD':>6} {'Tr':>4} {'WR':>4}")
            for i, r in enumerate(asset_results[:5]):
                print(f"  {i+1:3d} {r['strat_class']:>22} {r['stability']:>4}/5 {r['sharpe_3y']:+7.2f} "
                      f"{r['avg_sharpe']:+7.2f} ${r['dollar_pnl']:+7.0f} "
                      f"{r['max_drawdown']*100:5.1f}% {r['nb_trades']:4d} {r['win_rate']*100:3.0f}%")

    return phase2_results


# ═══════════════════════════════════════════════════════════════
# Phase 3 : ExecConfig sweep sur top configs
# ═══════════════════════════════════════════════════════════════

EC_GRID = {
    "equity_pct": [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    "cooldown_bars": [2, 4, 6, 8],
    "max_hold_bars": [24, 36, 48, 72, 96],
}


def run_phase3(data, bt, phase1_results):
    """Phase 3: ExecConfig sweep sur les top-5 configs de chaque asset."""
    print("\n" + "=" * 120)
    print("  PHASE 3 : EXEC CONFIG SWEEP — Top 5 configs × sizing/timing")
    print("=" * 120)

    ec_combos = list(product(
        EC_GRID["equity_pct"],
        EC_GRID["cooldown_bars"],
        EC_GRID["max_hold_bars"],
    ))
    print(f"  {len(ec_combos)} ExecConfig variants × 5 top configs × 3 assets = {len(ec_combos) * 5 * 3} runs")

    phase3_results = {}

    for asset in ["BTC", "SOL", "ETH"]:
        top5 = phase1_results[asset][:5]
        asset_results = []
        t0 = time.time()

        for cfg_rank, top_cfg in enumerate(top5):
            for eq_pct, cd_bars, mh_bars in ec_combos:
                ec = ExecConfig(
                    equity_pct=eq_pct,
                    leverage=5,
                    cooldown_bars=cd_bars,
                    max_hold_bars=mh_bars,
                )
                r = run_eval(data, bt, asset, top_cfg["strat_class"], top_cfg["params"], ec)
                r["params"] = top_cfg["params"]
                r["strat_class"] = top_cfg["strat_class"]
                r["ec_equity_pct"] = eq_pct
                r["ec_cooldown"] = cd_bars
                r["ec_max_hold"] = mh_bars
                r["base_rank"] = cfg_rank + 1
                asset_results.append(r)

        elapsed = time.time() - t0
        asset_results.sort(key=lambda x: (x["stability"], x["sharpe_3y"]), reverse=True)
        phase3_results[asset] = asset_results

        print(f"\n  {asset} — {len(asset_results)} combos en {elapsed:.1f}s")
        print(f"  {'#':>3} {'Stab':>4} {'SR 3Y':>7} {'$PnL':>8} {'DD':>6} {'Eq%':>4} {'CD':>3} {'MH':>3} {'Tr':>4} {'WR':>4} {'BaseR':>5}")
        for i, r in enumerate(asset_results[:8]):
            print(f"  {i+1:3d} {r['stability']:>4}/5 {r['sharpe_3y']:+7.2f} ${r['dollar_pnl']:+7.0f} "
                  f"{r['max_drawdown']*100:5.1f}% {r['ec_equity_pct']*100:3.0f}% "
                  f"{r['ec_cooldown']:3d} {r['ec_max_hold']:3d} {r['nb_trades']:4d} "
                  f"{r['win_rate']*100:3.0f}% #{r['base_rank']}")

    return phase3_results


# ═══════════════════════════════════════════════════════════════
# Phase 4 : Filtres de signaux
# ═══════════════════════════════════════════════════════════════

SIGNAL_FILTERS = {
    "anti_wick_40": filter_anti_wick(0.4),
    "anti_wick_50": filter_anti_wick(0.5),
    "anti_wick_60": filter_anti_wick(0.6),
    "rsi_confirm": filter_rsi_confirm(),
    "adx_20": filter_adx_trending(20),
    "adx_25": filter_adx_trending(25),
    "regime_aligned": filter_regime_aligned(),
    "bb_position": filter_bb_position(),
    "vol_spike_2": filter_volume_spike(2.0),
    "vol_spike_3": filter_volume_spike(3.0),
    "hours_8_20": filter_hours(list(range(0, 8)) + list(range(21, 24))),
    "no_weekend": filter_weekdays([5, 6]),
    "wick50+rsi": chain_filters(filter_anti_wick(0.5), filter_rsi_confirm()),
    "wick50+adx20": chain_filters(filter_anti_wick(0.5), filter_adx_trending(20)),
    "regime+wick50": chain_filters(filter_regime_aligned(), filter_anti_wick(0.5)),
    "8-20+wick50": chain_filters(filter_hours(list(range(0,8))+list(range(21,24))), filter_anti_wick(0.5)),
    "regime+rsi+wick50": chain_filters(filter_regime_aligned(), filter_rsi_confirm(), filter_anti_wick(0.5)),
    "noweek+rsi": chain_filters(filter_weekdays([5,6]), filter_rsi_confirm()),
}


def run_phase4(data, bt, phase1_results):
    """Phase 4: Filtres de signaux sur top-10 configs de chaque asset."""
    print("\n" + "=" * 120)
    print("  PHASE 4 : FILTRES DE SIGNAUX — Top 10 configs × %d filtres" % len(SIGNAL_FILTERS))
    print("=" * 120)

    # Use baseline ExecConfigs
    baseline_ec = {
        "BTC": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=72),
        "SOL": ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48),
        "ETH": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48),
    }

    phase4_results = {}

    for asset in ["BTC", "SOL", "ETH"]:
        top10 = phase1_results[asset][:10]
        ec = baseline_ec[asset]
        asset_results = []
        t0 = time.time()

        for cfg_rank, top_cfg in enumerate(top10):
            # Baseline sans filtre
            r_base = run_eval(data, bt, asset, top_cfg["strat_class"], top_cfg["params"], ec)

            for fname, filt in SIGNAL_FILTERS.items():
                r = run_eval(data, bt, asset, top_cfg["strat_class"], top_cfg["params"], ec,
                            signal_filter=filt)
                # Comparer vs baseline
                improvement = r["dollar_pnl"] - r_base["dollar_pnl"]
                r["params"] = top_cfg["params"]
                r["strat_class"] = top_cfg["strat_class"]
                r["filter_name"] = fname
                r["base_rank"] = cfg_rank + 1
                r["improvement_$"] = improvement
                asset_results.append(r)

        elapsed = time.time() - t0
        asset_results.sort(key=lambda x: (x["stability"], x["improvement_$"]), reverse=True)
        phase4_results[asset] = asset_results

        print(f"\n  {asset} — {len(asset_results)} combos en {elapsed:.1f}s")
        print(f"  {'#':>3} {'Filter':>20} {'Stab':>4} {'SR 3Y':>7} {'$PnL':>8} {'Δ$':>8} {'DD':>6} {'Tr':>4} {'WR':>4} {'BaseR':>5}")
        # Show top configs where filter improves
        improved = [r for r in asset_results if r["improvement_$"] > 0 and r["stability"] >= 4]
        improved.sort(key=lambda x: x["improvement_$"], reverse=True)
        for i, r in enumerate(improved[:10]):
            print(f"  {i+1:3d} {r['filter_name']:>20} {r['stability']:>4}/5 {r['sharpe_3y']:+7.2f} "
                  f"${r['dollar_pnl']:+7.0f} {r['improvement_$']:+7.0f} "
                  f"{r['max_drawdown']*100:5.1f}% {r['nb_trades']:4d} {r['win_rate']*100:3.0f}% #{r['base_rank']}")

        if not improved:
            print(f"    Aucun filtre n'améliore les top-10 configs (stability >= 4)")

    return phase4_results


# ═══════════════════════════════════════════════════════════════
# Portfolio combiné
# ═══════════════════════════════════════════════════════════════

def run_portfolio_eval(data, bt, btc_cfg, sol_cfg, eth_cfg):
    """Évalue le portfolio combiné 3 assets sur toutes les fenêtres."""
    results_per_window = {}

    for wname, start, end in WINDOWS_FULL:
        total_pnl = 0
        total_trades = 0
        max_dd = 0
        all_pnls = []

        for asset, cfg in [("BTC", btc_cfg), ("SOL", sol_cfg), ("ETH", eth_cfg)]:
            cls = V2_STRATEGY_REGISTRY[cfg["strat_class"]]
            strat = cls(cfg["params"])
            df_w = slice_window(data[asset], start, end)
            if len(df_w) < 100:
                continue
            signals = strat.generate_signals(df_w)
            if cfg.get("signal_filter"):
                signals = cfg["signal_filter"](signals, df_w)

            m = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                       exec_config=cfg["ec"], initial_equity=INITIAL_EQUITY)
            total_pnl += m.get("dollar_pnl", 0)
            total_trades += m["nb_trades"]
            max_dd = max(max_dd, m["max_drawdown"])
            if "trades_detail" in m:
                all_pnls.extend([t["pnl_pct"] for t in m["trades_detail"]])

        if len(all_pnls) > 1:
            pa = np.array(all_pnls)
            df_ref = slice_window(data["BTC"], start, end)
            if len(df_ref) > 1:
                days = (df_ref.index[-1] - df_ref.index[0]).total_seconds() / 86400
                tpy = len(pa) / max(days / 365.25, 0.01)
                sharpe = (pa.mean() / pa.std(ddof=1)) * np.sqrt(tpy)
                sharpe = max(-10.0, min(10.0, sharpe))
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        results_per_window[wname] = {
            "sharpe": sharpe,
            "dollar_pnl": total_pnl,
            "nb_trades": total_trades,
            "max_drawdown": max_dd,
        }

    r3y = results_per_window.get("Full 3Y", {})
    sharpes = [results_per_window[w[0]]["sharpe"] for w in WINDOWS if w[0] in results_per_window]
    n_pos = sum(1 for s in sharpes if s > 0)

    return {
        "sharpe_3y": r3y.get("sharpe", 0),
        "avg_sharpe": np.mean(sharpes) if sharpes else 0,
        "dollar_pnl": r3y.get("dollar_pnl", 0),
        "max_drawdown": r3y.get("max_drawdown", 0),
        "nb_trades": r3y.get("nb_trades", 0),
        "stability": n_pos,
    }


def run_portfolio_combos(data, bt, phase1_results, phase3_results):
    """Test les top configs combinées en portfolio."""
    print("\n" + "=" * 120)
    print("  PORTFOLIO COMBINÉ — Top 5 × Top 5 × Top 5")
    print("=" * 120)

    # Prendre les top 5 par asset (Phase 1 + Phase 3 mélangés)
    def get_top_configs(asset, n=5):
        """Retourne les n meilleures configs uniques par asset."""
        p1 = phase1_results.get(asset, [])
        p3 = phase3_results.get(asset, [])
        # Phase 3 a déjà les EC variants, utiliser ceux-là
        all_cfgs = []
        for r in p1[:n]:
            cfg = {
                "strat_class": r["strat_class"],
                "params": r["params"],
                "ec": {"BTC": BTC_GRID, "SOL": SOL_GRID, "ETH": ETH_GRID}[asset]["ec"],
            }
            all_cfgs.append(cfg)
        for r in p3:
            if len(all_cfgs) >= n:
                break
            if r["stability"] >= 4:
                cfg = {
                    "strat_class": r["strat_class"],
                    "params": r["params"],
                    "ec": ExecConfig(
                        equity_pct=r["ec_equity_pct"],
                        leverage=5,
                        cooldown_bars=r["ec_cooldown"],
                        max_hold_bars=r["ec_max_hold"],
                    ),
                }
                # Avoid duplicates
                is_dup = any(c["params"] == cfg["params"] and
                            c["ec"].equity_pct == cfg["ec"].equity_pct for c in all_cfgs)
                if not is_dup:
                    all_cfgs.append(cfg)
        return all_cfgs[:n]

    btc_top = get_top_configs("BTC", 5)
    sol_top = get_top_configs("SOL", 5)
    eth_top = get_top_configs("ETH", 5)

    n_combos = len(btc_top) * len(sol_top) * len(eth_top)
    print(f"  {len(btc_top)} BTC × {len(sol_top)} SOL × {len(eth_top)} ETH = {n_combos} combos")

    portfolio_results = []
    t0 = time.time()

    for bi, bc in enumerate(btc_top):
        for si, sc in enumerate(sol_top):
            for ei, ec_cfg in enumerate(eth_top):
                r = run_portfolio_eval(data, bt, bc, sc, ec_cfg)
                r["btc_idx"] = bi
                r["sol_idx"] = si
                r["eth_idx"] = ei
                portfolio_results.append(r)

    elapsed = time.time() - t0
    portfolio_results.sort(key=lambda x: (x["stability"], x["sharpe_3y"]), reverse=True)

    print(f"\n  {n_combos} portfolios évalués en {elapsed:.1f}s")
    print(f"\n  {'#':>3} {'Stab':>4} {'SR 3Y':>7} {'AvgSR':>7} {'$PnL':>8} {'DD':>6} {'Tr':>5}  Config")
    for i, r in enumerate(portfolio_results[:10]):
        print(f"  {i+1:3d} {r['stability']:>4}/5 {r['sharpe_3y']:+7.2f} {r['avg_sharpe']:+7.2f} "
              f"${r['dollar_pnl']:+7.0f} {r['max_drawdown']*100:5.1f}% {r['nb_trades']:5d}  "
              f"BTC#{r['btc_idx']+1} SOL#{r['sol_idx']+1} ETH#{r['eth_idx']+1}")

    return portfolio_results


# ═══════════════════════════════════════════════════════════════
# Baseline comparison
# ═══════════════════════════════════════════════════════════════

BASELINE = {
    "BTC": {
        "strat_class": "StratInsideBarBreakout",
        "params": {"vol_min": 1.5, "trend_filter": True, "atr_filter": True, "sl_pct": 1.5, "tp_pct": 3.0},
        "ec": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=72),
    },
    "SOL": {
        "strat_class": "StratBreakoutRelaxed",
        "params": {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
        "ec": ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48),
    },
    "ETH": {
        "strat_class": "StratBreakoutRelaxed",
        "params": {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 5.0},
        "ec": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48),
    },
}


def print_baseline(data, bt):
    """Affiche les résultats baseline pour comparaison."""
    print("\n" + "=" * 120)
    print("  BASELINE ACTUEL (référence)")
    print("=" * 120)

    for asset, cfg in BASELINE.items():
        r = run_eval(data, bt, asset, cfg["strat_class"], cfg["params"], cfg["ec"])
        print(f"  {asset:>3}: Stab {r['stability']}/5  SR {r['sharpe_3y']:+.2f}  "
              f"AvgSR {r['avg_sharpe']:+.2f}  ${r['dollar_pnl']:+.0f}  "
              f"DD {r['max_drawdown']*100:.1f}%  Tr {r['nb_trades']}  WR {r['win_rate']*100:.0f}%")

    # Portfolio baseline
    r_port = run_portfolio_eval(data, bt, BASELINE["BTC"], BASELINE["SOL"], BASELINE["ETH"])
    print(f"\n  PORTFOLIO: Stab {r_port['stability']}/5  SR {r_port['sharpe_3y']:+.2f}  "
          f"AvgSR {r_port['avg_sharpe']:+.2f}  ${r_port['dollar_pnl']:+.0f}  "
          f"DD {r_port['max_drawdown']*100:.1f}%  Tr {r_port['nb_trades']}")

    return r_port


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 120)
    print("  MEGA SWEEP — 10,000+ variantes d'optimisation")
    print("=" * 120)

    # Count total variants
    n_p1 = sum(
        len(expand_grid(g["params_grid"]))
        for g in [BTC_GRID, SOL_GRID, ETH_GRID]
    )
    n_p2 = sum(len(v) for v in ALT_STRATEGIES.values()) * 3  # approx (minus primary)
    n_p3 = len(list(product(*EC_GRID.values()))) * 5 * 3
    n_p4 = 10 * len(SIGNAL_FILTERS) * 3
    n_portfolio = 5 * 5 * 5
    total = n_p1 + n_p2 + n_p3 + n_p4 + n_portfolio

    print(f"\n  Phase 1 : {n_p1:,} param sweep configs")
    print(f"  Phase 2 : ~{n_p2:,} stratégies alternatives")
    print(f"  Phase 3 : {n_p3:,} ExecConfig variants")
    print(f"  Phase 4 : {n_p4:,} filtres de signaux")
    print(f"  Portfolio: {n_portfolio} combinaisons")
    print(f"  TOTAL   : ~{total:,} backtests")

    # Load data
    print("\n-- Chargement des données --")
    data = {}
    for sym in ["BTC", "SOL", "ETH"]:
        data[sym] = load_asset(sym)
        print(f"  {sym} OK ({len(data[sym])} barres)")

    bt = SweepBacktester()

    # Baseline
    r_baseline = print_baseline(data, bt)

    # Phase 1
    p1 = run_phase1(data, bt)

    # Phase 2
    p2 = run_phase2(data, bt)

    # Phase 3
    p3 = run_phase3(data, bt, p1)

    # Phase 4
    p4 = run_phase4(data, bt, p1)

    # Portfolio combos
    port = run_portfolio_combos(data, bt, p1, p3)

    # Final summary
    elapsed = time.time() - t_start
    print("\n" + "=" * 120)
    print("  RÉSUMÉ FINAL")
    print("=" * 120)

    print(f"\n  Temps total : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\n  BASELINE PORTFOLIO : SR {r_baseline['sharpe_3y']:+.2f}  ${r_baseline['dollar_pnl']:+.0f}")

    if port:
        best = port[0]
        print(f"  MEILLEUR PORTFOLIO : SR {best['sharpe_3y']:+.2f}  ${best['dollar_pnl']:+.0f}  "
              f"Stab {best['stability']}/5  DD {best['max_drawdown']*100:.1f}%")
        delta = best["dollar_pnl"] - r_baseline["dollar_pnl"]
        print(f"  AMÉLIORATION       : ${delta:+.0f}")

    # Per-asset best vs baseline
    print(f"\n  Par asset (meilleur Phase 1 vs baseline):")
    for asset in ["BTC", "SOL", "ETH"]:
        base_r = run_eval(data, bt, asset, BASELINE[asset]["strat_class"],
                         BASELINE[asset]["params"], BASELINE[asset]["ec"])
        if p1[asset]:
            best_r = p1[asset][0]
            delta = best_r["dollar_pnl"] - base_r["dollar_pnl"]
            print(f"    {asset}: Baseline SR {base_r['sharpe_3y']:+.2f} ${base_r['dollar_pnl']:+.0f} "
                  f"→ Best SR {best_r['sharpe_3y']:+.2f} ${best_r['dollar_pnl']:+.0f} (Δ${delta:+.0f})")

    print("\n" + "=" * 120)


if __name__ == "__main__":
    main()
