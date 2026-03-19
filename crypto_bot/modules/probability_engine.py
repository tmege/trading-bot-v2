"""
Module 3 — ProbabilityEngine
Coeur analytique : probabilités conditionnelles d'événements,
stabilité glissante, corrélations laggées multi-asset.
"""
from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import yaml

logger = logging.getLogger(__name__)


class ProbabilityEngine:
    """Calcule les probabilités conditionnelles de succès
    pour des combinaisons de conditions techniques."""

    # Seuils par défaut pour les horizons (en %)
    DEFAULT_THRESHOLDS = {
        "up_3j": 5.0,    # P(max_return_3j > +5%)
        "up_7j": 10.0,   # P(max_return_7j > +10%)
        "down_3j": 5.0,  # P(min_return_3j < -5%)
    }

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.train_ratio: float = self.cfg["train_ratio"]
        logger.info("ProbabilityEngine initialisé")

    # ── Forward returns ───────────────────────────────────────

    @staticmethod
    def _add_forward_returns(df: pd.DataFrame,
                             horizons_bars: dict[str, int]) -> pd.DataFrame:
        """Ajoute les max/min forward returns pour chaque horizon.

        Args:
            horizons_bars: {"3j": 72, "7j": 168} pour du 1h
                           {"3j": 18, "7j": 42}  pour du 4h
        """
        for label, n_bars in horizons_bars.items():
            # Max return sur l'horizon (best case)
            df[f"fwd_max_{label}"] = (
                df["close"]
                .shift(-1)
                .rolling(window=n_bars)
                .max()
                .shift(-n_bars + 1)
                / df["close"] - 1
            ) * 100  # en %

            # Min return sur l'horizon (worst case)
            df[f"fwd_min_{label}"] = (
                df["close"]
                .shift(-1)
                .rolling(window=n_bars)
                .min()
                .shift(-n_bars + 1)
                / df["close"] - 1
            ) * 100  # en %

        return df

    @staticmethod
    def _infer_horizons(df: pd.DataFrame) -> dict[str, int]:
        """Infère le nombre de bougies pour 3j et 7j selon le timeframe."""
        if len(df) < 2:
            return {"3j": 72, "7j": 168}

        diffs = df.index.to_series().diff().dropna()
        median_minutes = diffs.dt.total_seconds().median() / 60

        if median_minutes <= 10:       # 5min
            bars_per_day = 288
        elif median_minutes <= 65:     # 1h
            bars_per_day = 24
        elif median_minutes <= 250:    # 4h
            bars_per_day = 6
        else:                          # 1d
            bars_per_day = 1

        return {
            "3j": int(3 * bars_per_day),
            "7j": int(7 * bars_per_day),
        }

    # ── Évaluation d'un événement ─────────────────────────────

    @staticmethod
    def _apply_condition(series: pd.Series, condition) -> pd.Series:
        """Applique une condition sur une série.

        Formats supportés:
            True / False          → série == valeur
            (">", 50)             → série > 50
            ("<", 30)             → série < 30
            (">=", 50)            → série >= 50
            ("<=", 30)            → série <= 30
            ("between", 30, 70)   → 30 <= série <= 70
            "bull" / "bear" etc.  → série == valeur (catégoriel)
        """
        if isinstance(condition, bool):
            return series == condition

        if isinstance(condition, str):
            return series == condition

        if isinstance(condition, tuple):
            op = condition[0]
            if op == ">":
                return series > condition[1]
            elif op == "<":
                return series < condition[1]
            elif op == ">=":
                return series >= condition[1]
            elif op == "<=":
                return series <= condition[1]
            elif op == "between":
                return (series >= condition[1]) & (series <= condition[2])
            else:
                raise ValueError(f"Opérateur inconnu: {op}")

        raise ValueError(f"Format de condition non supporté: {condition}")

    def _build_event_mask(self, df: pd.DataFrame,
                          event: dict) -> pd.Series:
        """Construit le masque booléen d'un événement
        (toutes les conditions simultanées)."""

        mask = pd.Series(True, index=df.index)

        for col, condition in event.items():
            if col not in df.columns:
                logger.warning("Colonne '%s' absente — condition ignorée", col)
                continue
            mask = mask & self._apply_condition(df[col], condition)

        return mask

    def compute_baseline(self, df: pd.DataFrame,
                         horizons_bars: dict[str, int] | None = None,
                         thresholds: dict[str, float] | None = None
                         ) -> dict[str, float]:
        """Calcule P(max_return > seuil) sans condition (baseline).

        Retourne: {"p_up_3j": float, "p_up_7j": float, "p_down_3j": float}
        """
        if horizons_bars is None:
            horizons_bars = self._infer_horizons(df)
        if thresholds is None:
            thresholds = self.DEFAULT_THRESHOLDS

        df = self._add_forward_returns(df.copy(), horizons_bars)

        baseline = {}
        total = df[f"fwd_max_3j"].notna().sum()

        if total == 0:
            return {"p_up_3j": 0, "p_up_7j": 0, "p_down_3j": 0}

        baseline["p_up_3j"] = (
            (df[f"fwd_max_3j"] > thresholds["up_3j"]).sum() / total
        )
        baseline["p_up_7j"] = (
            (df[f"fwd_max_7j"] > thresholds["up_7j"]).sum() / total
        )
        baseline["p_down_3j"] = (
            (df[f"fwd_min_3j"] < -thresholds["down_3j"]).sum() / total
        )

        logger.info("Baseline — p_up_3j=%.3f  p_up_7j=%.3f  p_down_3j=%.3f  (N=%d)",
                     baseline["p_up_3j"], baseline["p_up_7j"],
                     baseline["p_down_3j"], total)
        return baseline

    def evaluate_event(self, df: pd.DataFrame, event: dict,
                       horizons_bars: dict[str, int] | None = None,
                       thresholds: dict[str, float] | None = None,
                       baseline: dict[str, float] | None = None
                       ) -> dict:
        """Évalue un événement : probabilités conditionnelles,
        ratio risk/reward, p-value, validité statistique.

        Retourne:
            freq, N, p_up_3j, p_up_7j, p_down_3j, rr_ratio,
            p_value, valide, event_desc
        """
        if horizons_bars is None:
            horizons_bars = self._infer_horizons(df)
        if thresholds is None:
            thresholds = self.DEFAULT_THRESHOLDS

        df = self._add_forward_returns(df.copy(), horizons_bars)

        mask = self._build_event_mask(df, event)
        total = len(df)
        n_event = mask.sum()

        result = {
            "event": event,
            "event_desc": self._describe_event(event),
            "N": int(n_event),
            "freq": n_event / total if total > 0 else 0,
        }

        if n_event == 0:
            result.update({
                "p_up_3j": 0, "p_up_7j": 0, "p_down_3j": 0,
                "rr_ratio": 0, "p_value": 1.0, "valide": False,
            })
            return result

        event_df = df.loc[mask]

        # Probabilités conditionnelles
        valid_3j = event_df["fwd_max_3j"].notna()
        valid_7j = event_df["fwd_max_7j"].notna()
        n_valid_3j = valid_3j.sum()
        n_valid_7j = valid_7j.sum()

        p_up_3j = (
            (event_df.loc[valid_3j, "fwd_max_3j"] > thresholds["up_3j"]).sum()
            / n_valid_3j
        ) if n_valid_3j > 0 else 0

        p_up_7j = (
            (event_df.loc[valid_7j, "fwd_max_7j"] > thresholds["up_7j"]).sum()
            / n_valid_7j
        ) if n_valid_7j > 0 else 0

        p_down_3j = (
            (event_df.loc[valid_3j, "fwd_min_3j"] < -thresholds["down_3j"]).sum()
            / n_valid_3j
        ) if n_valid_3j > 0 else 0

        # Risk/reward ratio
        rr_ratio = p_up_3j / p_down_3j if p_down_3j > 0 else float("inf")

        # Test binomial : p_up_3j est-il significativement > baseline ?
        if baseline is None:
            baseline = self.compute_baseline(df, horizons_bars, thresholds)

        baseline_p = baseline.get("p_up_3j", 0.5)
        successes = int((event_df.loc[valid_3j, "fwd_max_3j"] > thresholds["up_3j"]).sum())

        if n_valid_3j > 0 and baseline_p > 0:
            p_value = stats.binomtest(
                successes, n_valid_3j, baseline_p,
                alternative="greater"
            ).pvalue
        else:
            p_value = 1.0

        # Validité statistique
        valide = (n_event >= 30) and (p_value < 0.05)

        result.update({
            "p_up_3j": round(p_up_3j, 4),
            "p_up_7j": round(p_up_7j, 4),
            "p_down_3j": round(p_down_3j, 4),
            "rr_ratio": round(rr_ratio, 3),
            "p_value": round(p_value, 6),
            "valide": valide,
        })

        return result

    # ── Stabilité glissante ───────────────────────────────────

    def rolling_stability(self, df: pd.DataFrame, event: dict,
                          window_months: int = 6,
                          step_months: int = 1,
                          horizons_bars: dict[str, int] | None = None,
                          thresholds: dict[str, float] | None = None
                          ) -> pd.DataFrame:
        """Recalcule p_up_3j sur des fenêtres glissantes de 6 mois.
        Détecte la dégradation de l'edge dans le temps.

        Retourne DataFrame: [window_start, window_end, N, p_up_3j, p_down_3j, rr_ratio]
        """
        if horizons_bars is None:
            horizons_bars = self._infer_horizons(df)
        if thresholds is None:
            thresholds = self.DEFAULT_THRESHOLDS

        df = self._add_forward_returns(df.copy(), horizons_bars)

        results = []
        start = df.index[0]
        end = df.index[-1]
        window_start = start

        while True:
            window_end = window_start + pd.DateOffset(months=window_months)
            if window_end > end:
                break

            window_df = df.loc[window_start:window_end]
            mask = self._build_event_mask(window_df, event)
            n_event = mask.sum()

            if n_event >= 5:  # minimum pour calculer
                event_df = window_df.loc[mask]
                valid = event_df["fwd_max_3j"].notna()
                n_valid = valid.sum()

                if n_valid > 0:
                    p_up = (event_df.loc[valid, "fwd_max_3j"] > thresholds["up_3j"]).sum() / n_valid
                    p_down = (event_df.loc[valid, "fwd_min_3j"] < -thresholds["down_3j"]).sum() / n_valid
                    rr = p_up / p_down if p_down > 0 else float("inf")
                else:
                    p_up = p_down = 0
                    rr = 0

                results.append({
                    "window_start": window_start,
                    "window_end": window_end,
                    "N": int(n_event),
                    "p_up_3j": round(p_up, 4),
                    "p_down_3j": round(p_down, 4),
                    "rr_ratio": round(rr, 3),
                })

            window_start += pd.DateOffset(months=step_months)

        stability_df = pd.DataFrame(results)
        if not stability_df.empty:
            # Score de stabilité : coefficient de variation de p_up_3j
            cv = stability_df["p_up_3j"].std() / stability_df["p_up_3j"].mean() \
                if stability_df["p_up_3j"].mean() > 0 else float("inf")
            logger.info("Stabilité — %d fenêtres, CV(p_up_3j)=%.3f", len(stability_df), cv)

        return stability_df

    # ── Corrélation laggée multi-asset ────────────────────────

    def lagged_correlation_matrix(
        self,
        datasets: dict[str, pd.DataFrame],
        max_lag: int = 12
    ) -> pd.DataFrame:
        """Corrélation croisée entre return_A[t] et return_B[t+lag].
        Identifie les paires leader/follower les plus stables.

        Args:
            datasets: {symbol: DataFrame} — chaque df doit avoir 'log_return'
            max_lag: lag max en nombre de bougies

        Retourne DataFrame:
            asset_A, asset_B, optimal_lag, correlation, direction
        """
        symbols = list(datasets.keys())
        results = []

        for sym_a, sym_b in combinations(symbols, 2):
            df_a = datasets[sym_a]
            df_b = datasets[sym_b]

            # Aligner les index
            common_idx = df_a.index.intersection(df_b.index)
            if len(common_idx) < 100:
                logger.warning("Pas assez de données communes %s/%s", sym_a, sym_b)
                continue

            ret_a = df_a.loc[common_idx, "log_return"].dropna()
            ret_b = df_b.loc[common_idx, "log_return"].dropna()

            # Réaligner après dropna
            common = ret_a.index.intersection(ret_b.index)
            ret_a = ret_a.loc[common]
            ret_b = ret_b.loc[common]

            best_corr = 0
            best_lag = 0

            for lag in range(1, max_lag + 1):
                # A mène B : corr(ret_A[t], ret_B[t+lag])
                corr_ab = ret_a.iloc[:-lag].corr(ret_b.iloc[lag:])
                # B mène A : corr(ret_B[t], ret_A[t+lag])
                corr_ba = ret_b.iloc[:-lag].corr(ret_a.iloc[lag:])

                if abs(corr_ab) > abs(best_corr):
                    best_corr = corr_ab
                    best_lag = lag
                    direction = f"{sym_a} → {sym_b}"

                if abs(corr_ba) > abs(best_corr):
                    best_corr = corr_ba
                    best_lag = lag
                    direction = f"{sym_b} → {sym_a}"

            results.append({
                "asset_A": sym_a,
                "asset_B": sym_b,
                "optimal_lag": best_lag,
                "correlation": round(best_corr, 4),
                "direction": direction,
            })

            logger.info("Corrélation %s/%s — lag=%d, corr=%.4f (%s)",
                        sym_a, sym_b, best_lag, best_corr, direction)

        return pd.DataFrame(results).sort_values(
            "correlation", key=abs, ascending=False
        ).reset_index(drop=True)

    # ── Scan d'événements ─────────────────────────────────────

    def scan_events(self, df: pd.DataFrame,
                    events: list[dict],
                    horizons_bars: dict[str, int] | None = None,
                    thresholds: dict[str, float] | None = None
                    ) -> pd.DataFrame:
        """Évalue une liste d'événements et retourne le ranking
        trié par rr_ratio décroissant, filtré sur valide=True."""

        if horizons_bars is None:
            horizons_bars = self._infer_horizons(df)
        if thresholds is None:
            thresholds = self.DEFAULT_THRESHOLDS

        baseline = self.compute_baseline(df, horizons_bars, thresholds)

        results = []
        for i, event in enumerate(events):
            logger.info("Événement %d/%d : %s", i + 1, len(events),
                        self._describe_event(event))
            result = self.evaluate_event(
                df, event, horizons_bars, thresholds, baseline
            )
            results.append(result)

        results_df = pd.DataFrame(results)

        # Trier par rr_ratio décroissant
        if not results_df.empty:
            results_df = results_df.sort_values(
                "rr_ratio", ascending=False
            ).reset_index(drop=True)

            n_valid = results_df["valide"].sum()
            logger.info("Scan terminé — %d/%d événements valides (N≥30, p<0.05)",
                        n_valid, len(events))

        return results_df

    # ── Bibliothèque d'événements prédéfinis ──────────────────

    @staticmethod
    def default_events() -> list[dict]:
        """Retourne une bibliothèque d'événements à scanner."""
        return [
            # --- Oversold bounces ---
            {
                "RSI_14": ("<", 25),
                "pct_B": ("<", 0),
                "volume_ratio": (">", 1.5),
            },
            {
                "RSI_14": ("<", 20),
                "regime": "bull",
            },
            {
                "RSI_14": ("<", 30),
                "pct_B": ("<", 0.1),
                "body_ratio": ("<", 0.3),
            },
            {
                "RSI_14": ("<", 25),
                "STOCH_K": ("<", 15),
            },
            # --- Overbought reversals ---
            {
                "RSI_14": (">", 75),
                "pct_B": (">", 1),
                "volume_ratio": (">", 2.0),
            },
            {
                "RSI_14": (">", 80),
                "regime": "bear",
            },
            # --- Momentum entries ---
            {
                "golden_cross": True,
                "regime": "bull",
                "RSI_14": ("between", 50, 65),
            },
            {
                "golden_cross": True,
                "volume_ratio": (">", 1.5),
            },
            {
                "death_cross": True,
                "regime": "bear",
                "RSI_14": ("between", 35, 50),
            },
            # --- Breakout patterns ---
            {
                "compression": True,
                "higher_high": True,
                "body_ratio": (">", 0.7),
                "volume_ratio": (">", 2.5),
            },
            {
                "compression": True,
                "volume_ratio": (">", 3.0),
                "body_ratio": (">", 0.6),
            },
            # --- Mean reversion ---
            {
                "RSI_14": ("<", 25),
                "pct_B": ("<", 0),
                "body_ratio": ("<", 0.3),
                "volume_ratio": ("<", 0.8),
            },
            {
                "pct_B": ("<", -0.1),
                "inside_bar": True,
            },
            # --- Trend continuation ---
            {
                "MACD_hist": (">", 0),
                "RSI_14": ("between", 50, 70),
                "regime": "bull",
                "volume_ratio": (">", 1.2),
            },
            {
                "MACD_hist": ("<", 0),
                "RSI_14": ("between", 30, 50),
                "regime": "bear",
                "volume_ratio": (">", 1.2),
            },
            # --- Volume anomalies ---
            {
                "volume_ratio": (">", 3.0),
                "body_ratio": (">", 0.7),
                "regime": "bull",
            },
            {
                "volume_ratio": (">", 3.0),
                "body_ratio": (">", 0.7),
                "regime": "bear",
            },
            # --- Multi-indicator convergence ---
            {
                "RSI_14": ("<", 30),
                "STOCH_K": ("<", 20),
                "WILLR_14": ("<", -80),
                "pct_B": ("<", 0.1),
            },
            {
                "RSI_14": (">", 70),
                "STOCH_K": (">", 80),
                "WILLR_14": (">", -20),
                "pct_B": (">", 0.9),
            },
            # --- Bollinger squeeze + directional ---
            {
                "compression": True,
                "RSI_14": (">", 55),
                "regime": "bull",
            },
            {
                "compression": True,
                "RSI_14": ("<", 45),
                "regime": "bear",
            },
        ]

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _describe_event(event: dict) -> str:
        """Génère une description lisible d'un événement."""
        parts = []
        for col, cond in event.items():
            if isinstance(cond, bool):
                parts.append(f"{col}={cond}")
            elif isinstance(cond, str):
                parts.append(f"{col}='{cond}'")
            elif isinstance(cond, tuple):
                if cond[0] == "between":
                    parts.append(f"{cond[1]}≤{col}≤{cond[2]}")
                else:
                    parts.append(f"{col}{cond[0]}{cond[1]}")
            else:
                parts.append(f"{col}={cond}")
        return " & ".join(parts)


# ── Standalone test ───────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s"
    )

    # Données synthétiques avec indicateurs
    np.random.seed(42)
    n = 2000
    dates = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    trend = np.cumsum(np.random.randn(n) * 0.3)
    price = 40000 * np.exp(trend / 100)

    df = pd.DataFrame({
        "open": price,
        "high": price * (1 + np.abs(np.random.randn(n)) * 0.005),
        "low": price * (1 - np.abs(np.random.randn(n)) * 0.005),
        "close": price * (1 + np.random.randn(n) * 0.003),
        "volume": np.random.exponential(1000, n),
        "log_return": np.random.randn(n) * 0.01,
        "RSI_14": np.random.uniform(10, 90, n),
        "pct_B": np.random.uniform(-0.2, 1.2, n),
        "volume_ratio": np.random.exponential(1.0, n),
        "body_ratio": np.random.uniform(0, 1, n),
        "golden_cross": np.random.choice([True, False], n, p=[0.02, 0.98]),
        "death_cross": np.random.choice([True, False], n, p=[0.02, 0.98]),
        "regime": np.random.choice(["bull", "bear", "ranging"], n, p=[0.5, 0.3, 0.2]),
        "compression": np.random.choice([True, False], n, p=[0.1, 0.9]),
        "higher_high": np.random.choice([True, False], n, p=[0.15, 0.85]),
        "inside_bar": np.random.choice([True, False], n, p=[0.05, 0.95]),
        "MACD_hist": np.random.randn(n) * 100,
        "STOCH_K": np.random.uniform(0, 100, n),
        "WILLR_14": np.random.uniform(-100, 0, n),
    }, index=dates)

    engine = ProbabilityEngine()

    # Test baseline
    baseline = engine.compute_baseline(df)
    print(f"\nBaseline: {baseline}")

    # Test événement
    event = {"RSI_14": ("<", 25), "pct_B": ("<", 0), "volume_ratio": (">", 1.5)}
    result = engine.evaluate_event(df, event, baseline=baseline)
    print(f"\nÉvénement: {result['event_desc']}")
    print(f"  N={result['N']}, p_up_3j={result['p_up_3j']}, "
          f"rr={result['rr_ratio']}, p_value={result['p_value']}, "
          f"valide={result['valide']}")

    # Test scan
    events = ProbabilityEngine.default_events()[:5]
    scan_df = engine.scan_events(df, events)
    print(f"\nScan — {len(scan_df)} événements évalués:")
    print(scan_df[["event_desc", "N", "p_up_3j", "rr_ratio", "p_value", "valide"]].to_string())
