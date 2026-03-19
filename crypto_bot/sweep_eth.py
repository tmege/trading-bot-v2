#!/usr/bin/env python3
"""
Sweep massif de stratégies sur ETH — mode simplifié + réaliste.
Teste ~1880 combinaisons de FULL_GRID (7 stratégies) sur ETH 1h, fenêtres multiples.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from exec_config import ExecConfig
from modules.feature_engine import FeatureEngine
from modules.strategies import V2_STRATEGY_REGISTRY
from param_sweep import FULL_GRID, expand_grid
from sweep_runner import SweepBacktester


# ── Config ──────────────────────────────────────────────────────

WINDOWS = [
    ("2023-H1", "2023-01-01", "2023-07-01"),
    ("2023-H2", "2023-07-01", "2024-01-01"),
    ("2024-H1", "2024-01-01", "2024-07-01"),
    ("2024-H2", "2024-07-01", "2025-01-01"),
    ("2025-H1", "2025-01-01", "2025-07-01"),
    ("Full 2Y",  "2023-01-01", "2025-01-01"),
    ("Full 3Y",  "2023-01-01", "2026-01-01"),
]

REALISTIC_EC = ExecConfig(
    equity_pct=0.30, leverage=5, cooldown_bars=4,
    max_hold_bars=48,
)
INITIAL_EQUITY = 1000.0


# ── Chargement données ─────────────────────────────────────────

def load_eth():
    fe = FeatureEngine()
    path = "data/ETH_USDT_5m_ohlcv.parquet"
    df_5m = pd.read_parquet(path)
    df_5m = df_5m[~df_5m.index.duplicated(keep="first")]

    df_1h = df_5m.resample("1h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["open"])

    df_1h = fe.compute_all(df_1h)
    print(f"  ETH/USDT: {len(df_1h):,} bougies 1h "
          f"[{df_1h.index[0].date()} -> {df_1h.index[-1].date()}]")
    return df_1h


# ── Sweep sur une fenêtre ──────────────────────────────────────

def run_sweep_window(df, bt, exec_config=None, initial_equity=None):
    """Teste toutes les combinaisons FULL_GRID sur un DataFrame."""
    results = []
    for strat_name, param_grid in FULL_GRID.items():
        cls = V2_STRATEGY_REGISTRY.get(strat_name)
        if cls is None:
            continue

        if isinstance(param_grid, dict) and all(isinstance(v, list) for v in param_grid.values()):
            combos = expand_grid(param_grid)
        else:
            combos = param_grid

        for params in combos:
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
            metrics.pop("trades_detail", None)

            results.append({
                "strat_name":       strat_name,
                "params":           params,
                "signaux":          freq["total_signaux"],
                "signaux_par_mois": freq["signaux_par_mois"],
                **metrics,
            })
    return results


# ── Analyse stabilité ─────────────────────────────────────────

def analyze_stability(all_window_results):
    """Trouve les variantes stables (Sharpe > 0 dans toutes les fenêtres 6M)."""
    from collections import defaultdict

    perf = defaultdict(dict)  # key -> {window: sharpe}

    for window_name, results in all_window_results.items():
        if "Full" in window_name:
            continue
        for r in results:
            key = f"{r['strat_name']}|{str(sorted(r['params'].items()))}"
            perf[key][window_name] = r["sharpe_ratio"]

    stable = []
    for key, windows in perf.items():
        sharpes = list(windows.values())
        if len(sharpes) < 3:
            continue
        all_positive = all(s > 0 for s in sharpes)
        stable.append({
            "key": key,
            "avg_sharpe": np.mean(sharpes),
            "std_sharpe": np.std(sharpes),
            "min_sharpe": np.min(sharpes),
            "n_windows": len(sharpes),
            "all_positive": all_positive,
        })

    return sorted(stable, key=lambda x: x["avg_sharpe"], reverse=True)


# ── Main ──────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("=" * 100)
    print("  SWEEP MASSIF ETH/USDT — ~1880 COMBINAISONS × FENÊTRES MULTIPLES")
    print("  Mode simplifié + réaliste (frais Hyperliquid)")
    print("=" * 100)

    # Charger ETH
    print("\n-- Chargement --")
    df_full = load_eth()
    bt = SweepBacktester()

    # ══════════════════════════════════════════════════════════════
    # PARTIE 1 : MODE SIMPLIFIÉ
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  PARTIE 1 : MODE SIMPLIFIÉ (frais flat)")
    print("=" * 100)

    all_simple = {}
    all_simple_flat = []

    for window_name, start, end in WINDOWS:
        mask = (df_full.index >= pd.Timestamp(start, tz="UTC")) & \
               (df_full.index < pd.Timestamp(end, tz="UTC"))
        df_w = df_full.loc[mask]

        if len(df_w) < 200:
            print(f"  {window_name}: pas assez de données ({len(df_w)})")
            continue

        bh = (df_w["close"].iloc[-1] / df_w["close"].iloc[0] - 1) * 100

        results = run_sweep_window(df_w, bt)
        all_simple[window_name] = results

        if not results:
            print(f"  {window_name}: aucun résultat")
            continue

        for r in results:
            all_simple_flat.append({**r, "window": window_name})

        # Top 5 par Sharpe
        sorted_r = sorted(results, key=lambda x: x["sharpe_ratio"], reverse=True)
        print(f"\n  ── {window_name} ({len(df_w)} bougies, B&H: {bh:+.1f}%) "
              f"— {len(results)} variantes valides ──")
        print(f"  {'Stratégie':<25} {'Trades':>6} {'WR':>5} {'Sharpe':>7} "
              f"{'Return':>8} {'MaxDD':>7} {'PF':>6} {'Params'}")
        print(f"  {'-'*90}")

        for r in sorted_r[:5]:
            wr = r["win_rate"] * 100
            ret = r["total_return"] * 100
            dd = r["max_drawdown"] * 100
            pf = r["profit_factor"]
            pf_s = f"{pf:>6.2f}" if pf < 100 else "   inf"
            p = ", ".join(f"{k}={v}" for k, v in list(r["params"].items())[:3])
            print(f"  {r['strat_name']:<25} {r['nb_trades']:>6} {wr:>4.0f}% "
                  f"{r['sharpe_ratio']:>+7.2f} {ret:>+7.1f}% {dd:>6.1f}% "
                  f"{pf_s} {p}")

    # ══════════════════════════════════════════════════════════════
    # PARTIE 2 : MODE RÉALISTE
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  PARTIE 2 : MODE RÉALISTE (frais Hyperliquid, sizing composé, cooldown, DD mult)")
    print(f"  Config: equity={REALISTIC_EC.equity_pct*100:.0f}% | "
          f"lev={REALISTIC_EC.leverage}x | cooldown={REALISTIC_EC.cooldown_bars}h | "
          f"capital=${INITIAL_EQUITY:.0f}")
    print("=" * 100)

    all_realistic = {}
    all_realistic_flat = []

    for window_name, start, end in WINDOWS:
        mask = (df_full.index >= pd.Timestamp(start, tz="UTC")) & \
               (df_full.index < pd.Timestamp(end, tz="UTC"))
        df_w = df_full.loc[mask]

        if len(df_w) < 200:
            continue

        bh = (df_w["close"].iloc[-1] / df_w["close"].iloc[0] - 1) * 100

        results = run_sweep_window(df_w, bt, exec_config=REALISTIC_EC,
                                   initial_equity=INITIAL_EQUITY)
        all_realistic[window_name] = results

        if not results:
            print(f"  {window_name}: aucun résultat")
            continue

        for r in results:
            all_realistic_flat.append({**r, "window": window_name})

        sorted_r = sorted(results, key=lambda x: x["sharpe_ratio"], reverse=True)
        print(f"\n  ── {window_name} (B&H: {bh:+.1f}%) — {len(results)} variantes ──")
        print(f"  {'Stratégie':<25} {'Trades':>6} {'WR':>5} {'Sharpe':>7} "
              f"{'$PnL':>9} {'Final$':>9} {'MaxDD':>7} {'Fees$':>8} {'Params'}")
        print(f"  {'-'*95}")

        for r in sorted_r[:5]:
            wr = r["win_rate"] * 100
            dd = r["max_drawdown"] * 100
            pnl = r.get("dollar_pnl", 0)
            final = r.get("final_equity", INITIAL_EQUITY)
            fees = r.get("total_fees", 0) + r.get("total_funding", 0)
            p = ", ".join(f"{k}={v}" for k, v in list(r["params"].items())[:3])
            print(f"  {r['strat_name']:<25} {r['nb_trades']:>6} {wr:>4.0f}% "
                  f"{r['sharpe_ratio']:>+7.2f} {pnl:>+9.2f} {final:>9.2f} "
                  f"{dd:>6.1f}% {fees:>8.2f} {p}")

    # ══════════════════════════════════════════════════════════════
    # PARTIE 3 : ANALYSE DE STABILITÉ
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  PARTIE 3 : ANALYSE DE STABILITÉ (simplifié)")
    print("=" * 100)

    stable = analyze_stability(all_simple)
    n_all_pos = sum(1 for s in stable if s["all_positive"])
    print(f"\n  {n_all_pos}/{len(stable)} variantes avec Sharpe > 0 "
          f"sur TOUTES les fenêtres 6M\n")

    if stable:
        print(f"  {'Stratégie + Params':<60} {'AvgSR':>6} {'StdSR':>6} "
              f"{'MinSR':>7} {'Win':>4} {'Stable':>7}")
        print(f"  {'-'*96}")

        for s in stable[:20]:
            parts = s["key"].split("|", 1)
            name = parts[0]
            params_str = parts[1][:40] if len(parts) > 1 else ""
            label = f"{name} {params_str}"
            flag = "OUI" if s["all_positive"] and s["std_sharpe"] < 1.0 else "non"
            print(f"  {label:<60} {s['avg_sharpe']:>+5.2f} {s['std_sharpe']:>6.2f} "
                  f"{s['min_sharpe']:>+6.2f} {s['n_windows']:>4} {flag:>7}")

    # ══════════════════════════════════════════════════════════════
    # PARTIE 4 : RÉSUMÉ GLOBAL
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  RÉSUMÉ GLOBAL ETH/USDT")
    print("=" * 100)

    # Par stratégie (mode simplifié, toutes fenêtres confondues)
    if all_simple_flat:
        df_s = pd.DataFrame(all_simple_flat)
        print(f"\n  MODE SIMPLIFIÉ — {len(df_s)} tests au total")
        summary = df_s.groupby("strat_name").agg(
            n_tests=("sharpe_ratio", "count"),
            avg_sharpe=("sharpe_ratio", "mean"),
            best_sharpe=("sharpe_ratio", "max"),
            avg_return_pct=("total_return", lambda x: x.mean() * 100),
            avg_wr=("win_rate", lambda x: x.mean() * 100),
            avg_trades=("nb_trades", "mean"),
            pct_profitable=("total_return", lambda x: (x > 0).mean() * 100),
        ).sort_values("avg_sharpe", ascending=False)

        print(f"\n  {'Stratégie':<25} {'Tests':>6} {'AvgSR':>7} {'BestSR':>8} "
              f"{'AvgRet%':>8} {'AvgWR%':>7} {'AvgTr':>6} {'%Prof':>6}")
        print(f"  {'-'*82}")
        for name, row in summary.iterrows():
            print(f"  {name:<25} {row['n_tests']:>6.0f} {row['avg_sharpe']:>+7.2f} "
                  f"{row['best_sharpe']:>+8.2f} {row['avg_return_pct']:>+7.1f}% "
                  f"{row['avg_wr']:>6.1f}% {row['avg_trades']:>6.1f} "
                  f"{row['pct_profitable']:>5.1f}%")

    # Mode réaliste
    if all_realistic_flat:
        df_r = pd.DataFrame(all_realistic_flat)
        print(f"\n  MODE RÉALISTE — {len(df_r)} tests au total")
        summary_r = df_r.groupby("strat_name").agg(
            n_tests=("sharpe_ratio", "count"),
            avg_sharpe=("sharpe_ratio", "mean"),
            best_sharpe=("sharpe_ratio", "max"),
            avg_dollar_pnl=("dollar_pnl", "mean"),
            avg_wr=("win_rate", lambda x: x.mean() * 100),
            pct_profitable=("dollar_pnl", lambda x: (x > 0).mean() * 100),
        ).sort_values("avg_sharpe", ascending=False)

        print(f"\n  {'Stratégie':<25} {'Tests':>6} {'AvgSR':>7} {'BestSR':>8} "
              f"{'Avg$PnL':>9} {'AvgWR%':>7} {'%Prof':>6}")
        print(f"  {'-'*72}")
        for name, row in summary_r.iterrows():
            print(f"  {name:<25} {row['n_tests']:>6.0f} {row['avg_sharpe']:>+7.2f} "
                  f"{row['best_sharpe']:>+8.2f} {row['avg_dollar_pnl']:>+9.2f} "
                  f"{row['avg_wr']:>6.1f}% {row['pct_profitable']:>5.1f}%")

    # ── Top 10 candidats réalistes (Full 3Y ou Full 2Y) ─────────
    print(f"\n  {'─'*100}")
    print(f"  TOP 10 CANDIDATS ETH RÉALISTES (période longue)")
    print(f"  {'─'*100}")

    long_window = "Full 3Y" if "Full 3Y" in all_realistic else "Full 2Y"
    if long_window in all_realistic:
        top_real = sorted(all_realistic[long_window],
                          key=lambda x: x["sharpe_ratio"], reverse=True)

        print(f"\n  Fenêtre: {long_window}")
        print(f"  {'Stratégie':<25} {'Trades':>6} {'WR':>5} {'Sharpe':>7} "
              f"{'$PnL':>9} {'Final$':>9} {'MaxDD':>7} {'PF':>6} {'Params'}")
        print(f"  {'-'*100}")

        for r in top_real[:10]:
            wr = r["win_rate"] * 100
            dd = r["max_drawdown"] * 100
            pnl = r.get("dollar_pnl", 0)
            final = r.get("final_equity", INITIAL_EQUITY)
            pf = r["profit_factor"]
            pf_s = f"{pf:>6.2f}" if pf < 100 else "   inf"
            p = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            print(f"  {r['strat_name']:<25} {r['nb_trades']:>6} {wr:>4.0f}% "
                  f"{r['sharpe_ratio']:>+7.2f} {pnl:>+9.2f} {final:>9.2f} "
                  f"{dd:>6.1f}% {pf_s} {p}")

    elapsed = time.time() - t0
    print(f"\n  Temps total : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
