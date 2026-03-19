"""
Analyse des résultats du sweep + correction statistique.

Le problème : sur 10 000 tests à p<0.05, ~500 sembleront significatifs
par pure chance. La correction Benjamini-Hochberg (FDR) contrôle
le taux de faux positifs.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def load_results(path: str = "sweep_results.pkl") -> pd.DataFrame:
    """Charge les résultats du sweep depuis un fichier pickle."""
    with open(path, "rb") as f:
        results = pickle.load(f)
    return pd.DataFrame(results)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Correction statistique
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_multiple_testing_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Correction Benjamini-Hochberg (FDR 5%).

    p-value brute basée sur le Sharpe ratio :
        t_stat = sharpe / sqrt(1/N)  ~  t-distribution(N-1)

    La correction FDR contrôle le taux de faux positifs parmi
    les résultats déclarés significatifs. Moins conservateur
    que Bonferroni, plus adapté à l'exploration.
    """
    df = df.copy()

    # Nombre de trades minimum pour avoir un Sharpe fiable
    df["n_trades_safe"] = df["nb_trades"].clip(lower=2)

    # Erreur standard approx du Sharpe
    df["sharpe_se"] = 1 / np.sqrt(df["n_trades_safe"])

    # Statistique t
    df["t_stat"] = df["sharpe_ratio"] / df["sharpe_se"]

    # p-value unilatérale (H0 : Sharpe <= 0)
    df["p_value_raw"] = stats.t.sf(
        df["t_stat"].values,
        df=df["n_trades_safe"].values - 1,
    )

    # Remplacer NaN par 1 (non significatif)
    p_raw = df["p_value_raw"].fillna(1.0).values

    # Correction Benjamini-Hochberg
    reject, p_corrected, _, _ = multipletests(
        p_raw, alpha=0.05, method="fdr_bh"
    )

    df["p_value_corrected"] = p_corrected
    df["significant"]       = reject

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Filtrage et ranking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def filter_and_rank(
    df: pd.DataFrame,
    min_signals_per_month: float = 5,
    min_trades: int = 30,
    max_drawdown: float = 0.50,
    min_monthly_return: float = 0.04,
) -> pd.DataFrame:
    """Filtres cumulatifs + correction FDR + ranking.

    Chaque filtre réduit le nombre de tests, améliorant
    la puissance statistique de la correction FDR.

    Args:
        df: DataFrame brut du sweep
        min_signals_per_month: seuil de fréquence
        min_trades: minimum de trades pour un Sharpe fiable
        max_drawdown: drawdown max acceptable (ratio, 0.5 = 50%)
        min_monthly_return: rendement mensuel minimum (ratio, 0.04 = 4%)

    Returns:
        DataFrame trié par score composite
    """
    n0 = len(df)
    print(f"\nRésultats bruts : {n0:,}")

    # 1. Fréquence minimale
    df = df[df["signaux_par_mois"] >= min_signals_per_month].copy()
    print(f"Après filtre fréquence ({min_signals_per_month}/mois) : "
          f"{len(df):>6} / {n0:,}")

    # 2. Nombre de trades suffisant
    df = df[df["nb_trades"] >= min_trades].copy()
    print(f"Après filtre nb_trades (>= {min_trades})       : {len(df):>6}")

    if df.empty:
        print("Aucune stratégie ne passe les filtres de base.")
        return df

    # 3. Drawdown acceptable
    df = df[df["max_drawdown"] <= max_drawdown].copy()
    print(f"Après filtre drawdown (<= {max_drawdown:.0%})       : {len(df):>6}")

    if df.empty:
        print("Aucune stratégie ne passe le filtre de drawdown.")
        return df

    # 4. Correction statistique (élimine les faux positifs)
    df = apply_multiple_testing_correction(df)
    n_before_fdr = len(df)
    df = df[df["significant"]].copy()
    print(f"Après correction FDR 5%                : {len(df):>6}"
          f"  ← vrais edges (sur {n_before_fdr})")

    if df.empty:
        print("Aucune stratégie significative après correction FDR.")
        print("Cela signifie que les résultats pourraient être du bruit.")
        return df

    # 5. Objectif rendement
    df = df[df["avg_monthly_return"] >= min_monthly_return].copy()
    print(f"Après filtre rendement (>= {min_monthly_return:.0%}/mois) : {len(df):>6}")

    if df.empty:
        print("Aucune stratégie ne passe le filtre de rendement.")
        return df

    # Score composite pour ranking
    df["score"] = (
        df["sharpe_ratio"]          * 0.35
        + df["avg_monthly_return"]  * 100 * 0.25
        + df["pct_months_above_5pct"] * 0.20
        + (1 - df["max_drawdown"])  * 0.20
    )

    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rapport
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def report(df_ranked: pd.DataFrame, top_n: int = 20) -> None:
    """Affiche le rapport du sweep."""
    if df_ranked.empty:
        print("\nAucune stratégie à afficher.")
        return

    cols = [
        "strat_name", "asset", "timeframe",
        "avg_monthly_return", "sharpe_ratio",
        "max_drawdown", "nb_trades",
        "signaux_par_mois", "p_value_corrected", "score",
    ]
    # Filtrer les colonnes existantes
    cols = [c for c in cols if c in df_ranked.columns]

    print(f"\n{'=' * 80}")
    print(f"Top {min(top_n, len(df_ranked))} stratégies (corrigées FDR)")
    print("=" * 80)

    display = df_ranked[cols].head(top_n).copy()

    # Formatage
    for col in ["avg_monthly_return", "max_drawdown"]:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{x:.2%}")
    for col in ["sharpe_ratio", "score"]:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{x:.3f}")
    if "p_value_corrected" in display.columns:
        display["p_value_corrected"] = display["p_value_corrected"].apply(
            lambda x: f"{x:.4f}"
        )

    print(display.to_string(index=False))

    # Params du top 1
    best = df_ranked.iloc[0]
    print(f"\nMeilleure config :")
    print(f"  Stratégie : {best['strat_name']}")
    print(f"  Asset     : {best['asset']}")
    print(f"  Timeframe : {best['timeframe']}")
    print(f"  Params    : {best['params']}")
    print(f"  Sharpe    : {best['sharpe_ratio']:.3f}")
    print(f"  Return/m  : {best['avg_monthly_return']:.2%}")
    print(f"  Drawdown  : {best['max_drawdown']:.2%}")
    print(f"  p-value   : {best['p_value_corrected']:.4f}")


def export_top_candidates(
    df_ranked: pd.DataFrame,
    top_n: int = 50,
    output: str = "top_candidates.csv",
) -> None:
    """Exporte les top candidates en CSV pour analyse manuelle."""
    if df_ranked.empty:
        return
    df_ranked.head(top_n).to_csv(output, index=False)
    print(f"\nTop {top_n} exportés dans {output}")


# ── Standalone ────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_results()
    print(f"Total résultats chargés : {len(df):,}")
    df_ranked = filter_and_rank(df)
    report(df_ranked)
    export_top_candidates(df_ranked)
