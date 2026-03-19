"""
Test des stratégies V2 sur fenêtres de 6 mois et 1 an.
Données réelles BTC, ETH, SOL (2021-2026).
"""
import sys
sys.path.insert(0, ".")

import logging
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from modules.data_loader import DataLoader
from modules.feature_engine import FeatureEngine
from modules.strategies import V2_STRATEGY_REGISTRY, PARAM_GRID
from param_sweep import FULL_GRID, expand_grid
from exec_config import ExecConfig
from sweep_runner import SweepBacktester

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
)

# ── Configuration des fenêtres ────────────────────────────────

WINDOWS_6M = [
    ("2023-H1", "2023-01-01", "2023-07-01"),
    ("2023-H2", "2023-07-01", "2024-01-01"),
    ("2024-H1", "2024-01-01", "2024-07-01"),
    ("2024-H2", "2024-07-01", "2025-01-01"),
    ("2025-H1", "2025-01-01", "2025-07-01"),
    ("2025-H2", "2025-07-01", "2026-01-01"),
]

WINDOWS_1Y = [
    ("2022",    "2022-01-01", "2023-01-01"),
    ("2023",    "2023-01-01", "2024-01-01"),
    ("2024",    "2024-01-01", "2025-01-01"),
    ("2025",    "2025-01-01", "2026-01-01"),
]

ASSETS = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]


# ── Chargement et préparation ─────────────────────────────────

def load_and_prepare():
    """Charge les données 5m, resample en 1h, calcule les features."""
    fe = FeatureEngine()
    datasets = {}

    for asset in ASSETS:
        path = f"data/{asset}_5m_ohlcv.parquet"
        df_5m = pd.read_parquet(path)
        df_5m = df_5m[~df_5m.index.duplicated(keep="first")]

        # Resample 5m → 1h
        df_1h = df_5m.resample("1h").agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum",
        }).dropna(subset=["open"])

        # Features
        df_1h = fe.compute_all(df_1h)
        symbol = asset.replace("_", "/")
        datasets[symbol] = df_1h
        print(f"  {symbol}: {len(df_1h):,} bougies 1h "
              f"[{df_1h.index[0].date()} → {df_1h.index[-1].date()}]")

    return datasets


def slice_window(df, start, end):
    """Découpe un DataFrame sur une fenêtre temporelle."""
    mask = (df.index >= pd.Timestamp(start, tz="UTC")) & \
           (df.index < pd.Timestamp(end, tz="UTC"))
    return df.loc[mask]


# ── Exécution ─────────────────────────────────────────────────

def run_all_variants(df, bt, grid=None, exec_config=None, initial_equity=None):
    """Exécute toutes les variantes sur un DataFrame.

    Args:
        df: DataFrame avec OHLCV + features
        bt: SweepBacktester instance
        grid: grille de paramètres
        exec_config: ExecConfig optionnel pour mode réaliste
        initial_equity: capital initial en $ (requis si exec_config)

    Retourne une liste de dicts de résultats.
    """
    if grid is None:
        grid = PARAM_GRID

    # Expandre FULL_GRID (dict de listes) en listes de dicts si nécessaire
    expanded = {}
    for strat_name, param_def in grid.items():
        if isinstance(param_def, dict) and all(isinstance(v, list) for v in param_def.values()):
            expanded[strat_name] = expand_grid(param_def)
        else:
            expanded[strat_name] = param_def

    results = []
    for strat_name, variants in expanded.items():
        cls = V2_STRATEGY_REGISTRY.get(strat_name)
        if cls is None:
            continue
        for params in variants:
            strat = cls(params)
            freq = strat.signal_frequency(df)

            if freq["total_signaux"] < 3:
                continue

            signals = strat.generate_signals(df)
            metrics = bt.run(
                df, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                exec_config=exec_config,
                initial_equity=initial_equity,
            )

            # Exclure trades_detail (trop volumineux pour l'affichage)
            metrics.pop("trades_detail", None)

            results.append({
                "strat_name":       strat_name,
                "params":           params,
                "signaux":          freq["total_signaux"],
                "signaux_par_mois": freq["signaux_par_mois"],
                **metrics,
            })
    return results


def compute_buy_hold(df):
    """Buy & hold return sur la fenêtre."""
    if len(df) < 2:
        return 0.0
    return (df["close"].iloc[-1] / df["close"].iloc[0] - 1)


# ── Rapport ───────────────────────────────────────────────────

def print_header(title):
    w = 100
    print(f"\n{'━' * w}")
    print(f"  {title}")
    print(f"{'━' * w}")


def print_window_results(window_name, asset_results, buy_holds):
    """Affiche les résultats agrégés par stratégie pour une fenêtre."""
    # Agréger par strat_name : meilleure variante par Sharpe
    best_by_strat = {}
    for asset, results in asset_results.items():
        for r in results:
            key = r["strat_name"]
            if key not in best_by_strat or r["sharpe_ratio"] > best_by_strat[key]["sharpe_ratio"]:
                best_by_strat[key] = {**r, "asset": asset}

    if not best_by_strat:
        print("  (aucun résultat)")
        return

    # Trier par Sharpe
    sorted_strats = sorted(best_by_strat.values(),
                           key=lambda x: x["sharpe_ratio"], reverse=True)

    # Header
    print(f"  {'Stratégie':<28} {'Asset':<10} {'Trades':>6} {'WR':>5} "
          f"{'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'PF':>5} "
          f"{'Sig/m':>6} {'Params'}")
    print(f"  {'-'*28} {'-'*10} {'-'*6} {'-'*5} "
          f"{'-'*7} {'-'*8} {'-'*7} {'-'*5} "
          f"{'-'*6} {'-'*30}")

    for r in sorted_strats:
        wr = r["win_rate"] * 100
        ret = r["total_return"] * 100
        dd = r["max_drawdown"] * 100
        pf = r["profit_factor"]
        pf_str = f"{pf:>5.2f}" if pf < 100 else "  inf"
        # Params résumés
        p = r["params"]
        p_short = ", ".join(f"{k}={v}" for k, v in list(p.items())[:3])
        if len(p) > 3:
            p_short += "..."

        print(f"  {r['strat_name']:<28} {r['asset']:<10} "
              f"{r['nb_trades']:>6} {wr:>4.0f}% "
              f"{r['sharpe_ratio']:>+7.2f} {ret:>+7.1f}% {dd:>6.1f}% "
              f"{pf_str} {r['signaux_par_mois']:>5.1f} "
              f"{p_short}")

    # Buy & hold
    print(f"\n  Buy & Hold :")
    for asset, bh in buy_holds.items():
        print(f"    {asset:<12} {bh:>+7.1%}")


def print_detailed_table(window_name, all_results_flat):
    """Tableau détaillé : chaque variante × asset."""
    if not all_results_flat:
        return

    df = pd.DataFrame(all_results_flat)
    df["return_pct"] = df["total_return"] * 100
    df["dd_pct"] = df["max_drawdown"] * 100
    df["wr_pct"] = df["win_rate"] * 100

    # Top 15 par Sharpe
    top = df.nlargest(15, "sharpe_ratio")

    print(f"\n  Top 15 variantes ({window_name}) :")
    print(f"  {'Strat':<25} {'Asset':<10} {'Trades':>6} {'WR%':>5} "
          f"{'Sharpe':>7} {'Return%':>8} {'DD%':>6} {'PF':>5} {'Params'}")
    print(f"  {'-'*90}")

    for _, r in top.iterrows():
        pf = r["profit_factor"]
        pf_s = f"{pf:>5.2f}" if pf < 100 else "  inf"
        p = r["params"]
        p_short = str({k: v for k, v in list(p.items())[:3]})
        print(f"  {r['strat_name']:<25} {r['asset']:<10} "
              f"{r['nb_trades']:>6} {r['wr_pct']:>4.0f}% "
              f"{r['sharpe_ratio']:>+7.2f} {r['return_pct']:>+7.1f}% "
              f"{r['dd_pct']:>5.1f}% {pf_s} {p_short}")

    # Résumé par stratégie
    print(f"\n  Résumé par stratégie ({window_name}) :")
    summary = df.groupby("strat_name").agg(
        n_variants=("sharpe_ratio", "count"),
        avg_sharpe=("sharpe_ratio", "mean"),
        best_sharpe=("sharpe_ratio", "max"),
        avg_return=("return_pct", "mean"),
        avg_trades=("nb_trades", "mean"),
        avg_wr=("wr_pct", "mean"),
    ).sort_values("best_sharpe", ascending=False)

    print(f"  {'Stratégie':<28} {'Variants':>8} {'AvgSharpe':>10} "
          f"{'BestSharpe':>11} {'AvgRet%':>8} {'AvgTrades':>10} {'AvgWR%':>7}")
    print(f"  {'-'*92}")
    for name, row in summary.iterrows():
        print(f"  {name:<28} {row['n_variants']:>8.0f} "
              f"{row['avg_sharpe']:>+10.2f} {row['best_sharpe']:>+11.2f} "
              f"{row['avg_return']:>+7.1f}% {row['avg_trades']:>10.1f} "
              f"{row['avg_wr']:>6.1f}%")


def print_stability_analysis(results_by_window):
    """Analyse la stabilité des stratégies entre fenêtres."""
    print_header("ANALYSE DE STABILITÉ (cohérence entre fenêtres)")

    # Pour chaque (strat_name, params_key), collecter les Sharpe par fenêtre
    strat_perf = defaultdict(lambda: defaultdict(list))

    for window_name, flat_results in results_by_window.items():
        for r in flat_results:
            key = f"{r['strat_name']}|{str(sorted(r['params'].items()))}"
            strat_perf[key][window_name].append(r["sharpe_ratio"])

    # Filtrer les strats présentes dans au moins 3 fenêtres
    stable = {}
    for key, windows in strat_perf.items():
        sharpes = []
        for w, s_list in windows.items():
            sharpes.append(np.mean(s_list))
        if len(sharpes) >= 3:
            stable[key] = {
                "avg_sharpe": np.mean(sharpes),
                "std_sharpe": np.std(sharpes),
                "min_sharpe": np.min(sharpes),
                "max_sharpe": np.max(sharpes),
                "n_windows":  len(sharpes),
                "all_positive": all(s > 0 for s in sharpes),
            }

    if not stable:
        print("  Pas assez de données pour l'analyse de stabilité.")
        return

    # Trier par avg_sharpe
    sorted_stable = sorted(stable.items(),
                           key=lambda x: x[1]["avg_sharpe"], reverse=True)

    print(f"\n  Stratégies présentes dans >= 3 fenêtres :")
    print(f"  {'Strat|Params':<55} {'AvgSR':>6} {'StdSR':>6} "
          f"{'MinSR':>7} {'MaxSR':>7} {'Win':>4} {'Stable':>7}")
    print(f"  {'-'*100}")

    shown = 0
    for key, m in sorted_stable[:20]:
        parts = key.split("|", 1)
        name = parts[0]
        params_str = parts[1][:35] if len(parts) > 1 else ""
        label = f"{name} {params_str}"

        stable_flag = "OUI" if m["all_positive"] and m["std_sharpe"] < 1.0 else "non"
        print(f"  {label:<55} {m['avg_sharpe']:>+5.2f} {m['std_sharpe']:>6.2f} "
              f"{m['min_sharpe']:>+6.2f} {m['max_sharpe']:>+6.2f} "
              f"{m['n_windows']:>4} {stable_flag:>7}")
        shown += 1

    n_all_pos = sum(1 for m in stable.values() if m["all_positive"])
    print(f"\n  {n_all_pos}/{len(stable)} variantes avec Sharpe > 0 "
          f"sur TOUTES les fenêtres")


# ── Main ──────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("=" * 100)
    print("  TEST DES STRATÉGIES V2 — FENÊTRES DE 6 MOIS ET 1 AN")
    print("  Données réelles : BTC, ETH, SOL (2021-2026)")
    print("=" * 100)

    # Charger les données
    print("\n── Chargement des données ──")
    datasets = load_and_prepare()
    bt = SweepBacktester()

    results_by_window = {}  # Pour l'analyse de stabilité

    # ── Fenêtres de 6 mois ────────────────────────────────────
    print_header("FENÊTRES DE 6 MOIS")

    for window_name, start, end in WINDOWS_6M:
        print_header(f"6M — {window_name} ({start} → {end})")

        asset_results = {}
        buy_holds = {}
        all_flat = []

        for symbol, df_full in datasets.items():
            df_w = slice_window(df_full, start, end)
            if len(df_w) < 100:
                print(f"  {symbol}: pas assez de données ({len(df_w)} bougies)")
                continue

            bh = compute_buy_hold(df_w)
            buy_holds[symbol] = bh

            results = run_all_variants(df_w, bt, grid=FULL_GRID)
            asset_results[symbol] = results

            for r in results:
                all_flat.append({**r, "asset": symbol})

        print_window_results(window_name, asset_results, buy_holds)
        print_detailed_table(window_name, all_flat)
        results_by_window[window_name] = all_flat

    # ── Fenêtres de 1 an ──────────────────────────────────────
    print_header("FENÊTRES DE 1 AN")

    for window_name, start, end in WINDOWS_1Y:
        print_header(f"1Y — {window_name} ({start} → {end})")

        asset_results = {}
        buy_holds = {}
        all_flat = []

        for symbol, df_full in datasets.items():
            df_w = slice_window(df_full, start, end)
            if len(df_w) < 100:
                print(f"  {symbol}: pas assez de données ({len(df_w)} bougies)")
                continue

            bh = compute_buy_hold(df_w)
            buy_holds[symbol] = bh

            results = run_all_variants(df_w, bt, grid=FULL_GRID)
            asset_results[symbol] = results

            for r in results:
                all_flat.append({**r, "asset": symbol})

        print_window_results(window_name, asset_results, buy_holds)
        print_detailed_table(window_name, all_flat)
        results_by_window[window_name] = all_flat

    # ── Analyse de stabilité ──────────────────────────────────
    print_stability_analysis(results_by_window)

    # ── Résumé global ─────────────────────────────────────────
    print_header("RÉSUMÉ GLOBAL")

    total_tests = sum(len(v) for v in results_by_window.values())
    all_sharpes = [r["sharpe_ratio"] for v in results_by_window.values() for r in v]
    positive_sharpe = sum(1 for s in all_sharpes if s > 0)
    sharpe_above_1 = sum(1 for s in all_sharpes if s > 1.0)

    print(f"  Total tests exécutés         : {total_tests:,}")
    print(f"  Sharpe > 0                   : {positive_sharpe:,} ({positive_sharpe/max(total_tests,1)*100:.1f}%)")
    print(f"  Sharpe > 1.0                 : {sharpe_above_1:,} ({sharpe_above_1/max(total_tests,1)*100:.1f}%)")
    if all_sharpes:
        print(f"  Sharpe moyen                 : {np.mean(all_sharpes):+.3f}")
        print(f"  Sharpe médian                : {np.median(all_sharpes):+.3f}")

    elapsed = time.time() - t0
    print(f"\n  Temps total : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
