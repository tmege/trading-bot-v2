"""
Point d'entrée pour le sweep de stratégies V2.

Usage :
    python main_sweep.py                    # Sweep complet (FULL_GRID)
    python main_sweep.py --quick            # Sweep réduit (PARAM_GRID)
    python main_sweep.py --realistic        # Sweep réaliste (frais Hyperliquid)
    python main_sweep.py --realistic --equity 5000
    python main_sweep.py --verify-only      # Vérification moteur uniquement
    python main_sweep.py --analyze-only     # Analyse de résultats existants
"""
from __future__ import annotations

import argparse
import logging
import sys

from exec_config import ExecConfig
from modules.strategies import PARAM_GRID
from param_sweep import FULL_GRID, count_combinations
from sweep_runner import run_sweep, verify_engine
from sweep_analysis import load_results, filter_and_rank, report, export_top_candidates


def main():
    parser = argparse.ArgumentParser(description="Sweep de stratégies V2")
    parser.add_argument(
        "--quick", action="store_true",
        help="Utiliser PARAM_GRID (réduit) au lieu de FULL_GRID"
    )
    parser.add_argument(
        "--realistic", action="store_true",
        help="Mode réaliste (frais Hyperliquid, sizing composé, cooldown, DD mult)"
    )
    parser.add_argument(
        "--equity", type=float, default=1000.0,
        help="Capital initial en $ pour le mode réaliste (défaut: 1000)"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Vérifier le moteur de backtest uniquement"
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Analyser des résultats existants (sweep_results.pkl)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Nombre de workers parallèles"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Chemin vers config.yaml"
    )
    parser.add_argument(
        "--results-file", default="sweep_results.pkl",
        help="Fichier de résultats"
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Nombre de top stratégies à afficher"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
    )

    # Mode vérification uniquement
    if args.verify_only:
        print("═" * 60)
        print("VÉRIFICATION DU MOTEUR DE BACKTEST")
        print("═" * 60)
        ok = verify_engine(args.config)
        sys.exit(0 if ok else 1)

    # Mode analyse uniquement
    if args.analyze_only:
        print("═" * 60)
        print("ANALYSE DES RÉSULTATS EXISTANTS")
        print("═" * 60)
        try:
            df = load_results(args.results_file)
        except FileNotFoundError:
            print(f"Fichier {args.results_file} introuvable.")
            print("Lancez d'abord un sweep : python main_sweep.py")
            sys.exit(1)

        print(f"Total résultats chargés : {len(df):,}")
        df_ranked = filter_and_rank(df)
        report(df_ranked, top_n=args.top)
        export_top_candidates(df_ranked)
        sys.exit(0)

    # Préparer exec_config si mode réaliste
    exec_config = None
    initial_equity = None
    if args.realistic:
        exec_config = ExecConfig()  # défauts Hyperliquid
        initial_equity = args.equity

    # Sweep complet
    grid = PARAM_GRID if args.quick else FULL_GRID
    mode = "PARAM_GRID (réduit)" if args.quick else "FULL_GRID (exhaustif)"
    if args.realistic:
        mode += " + RÉALISTE"

    print("═" * 60)
    print(f"SWEEP V2 — {mode}")
    print("═" * 60)
    count_combinations(grid)

    results = run_sweep(
        config_path=args.config,
        grid=grid,
        n_workers=args.workers,
        results_file=args.results_file,
        exec_config=exec_config,
        initial_equity=initial_equity,
    )

    if not results:
        print("\nAucun résultat. Vérifiez les données et le moteur.")
        sys.exit(1)

    # Analyse
    print("\n" + "═" * 60)
    print("ANALYSE DES RÉSULTATS")
    print("═" * 60)

    import pandas as pd
    df = pd.DataFrame(results)
    print(f"Total résultats valides : {len(df):,}")

    df_ranked = filter_and_rank(df)
    report(df_ranked, top_n=args.top)
    export_top_candidates(df_ranked)


if __name__ == "__main__":
    main()
