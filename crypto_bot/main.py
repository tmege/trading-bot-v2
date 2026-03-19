"""
Crypto Strategy Research — Orchestrateur principal
Exécute les 7 modules dans l'ordre :
  1. DataLoader      → Téléchargement & cache OHLCV 5min + funding
  2. FeatureEngine   → Indicateurs techniques sur 1h/4h
  3. ProbabilityEngine → Probabilités conditionnelles + corrélations
  4. LiquidationEngine (utilisé par le Backtester)
  5. Strategies      → Génération de signaux
  6. Backtester      → Train + Test + Kelly
  7. Reporter        → Rapport HTML + Markdown
"""

import logging
import sys
import time
from pathlib import Path

# S'assurer que le répertoire courant est celui du script
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from modules.data_loader import DataLoader
from modules.feature_engine import FeatureEngine
from modules.probability_engine import ProbabilityEngine
from modules.backtester import Backtester
from modules.strategies import get_all_variants, STRATEGY_REGISTRY
from modules.reporter import Reporter

CONFIG = str(ROOT / "config.yaml")

# ── Logging ───────────────────────────────────────────────────

def setup_logging():
    import yaml
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    log_cfg = cfg.get("logging", {})

    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format",
            "%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s"),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "research.log", encoding="utf-8"),
        ]
    )


def main():
    setup_logging()
    logger = logging.getLogger("main")
    t_start = time.time()

    logger.info("=" * 70)
    logger.info("  CRYPTO STRATEGY RESEARCH — START")
    logger.info("=" * 70)

    # ══════════════════════════════════════════════════════════
    # ÉTAPE 1 — Chargement des données
    # ══════════════════════════════════════════════════════════
    logger.info("\n[1/7] DataLoader — Téléchargement & cache")
    loader = DataLoader(CONFIG)
    datasets = loader.load_all()
    funding_data = loader.load_funding()

    n_assets = len(datasets)
    n_candles = sum(
        len(df) for tfs in datasets.values() for df in tfs.values()
    )
    logger.info("  → %d assets, %d bougies totales chargées", n_assets, n_candles)

    # ══════════════════════════════════════════════════════════
    # ÉTAPE 2 — Indicateurs techniques
    # ══════════════════════════════════════════════════════════
    logger.info("\n[2/7] FeatureEngine — Calcul des indicateurs")
    feature_engine = FeatureEngine(CONFIG)
    datasets = feature_engine.compute_all_datasets(datasets, skip_base_tf=True)

    # Vérification rapide
    for symbol, tfs in datasets.items():
        for tf, df in tfs.items():
            if tf == "5m":
                continue
            n_indicators = len([c for c in df.columns
                                if c not in ("open", "high", "low", "close",
                                             "volume", "log_return",
                                             "volatility_7d", "volatility_30d",
                                             "volume_zscore")])
            logger.info("  %s %s — %d bougies, %d indicateurs",
                        symbol, tf, len(df), n_indicators)

    # ══════════════════════════════════════════════════════════
    # ÉTAPE 3 — Analyse probabiliste
    # ══════════════════════════════════════════════════════════
    logger.info("\n[3/7] ProbabilityEngine — Probabilités conditionnelles")
    prob_engine = ProbabilityEngine(CONFIG)

    # Scanner les événements sur chaque asset × timeframe (1h)
    all_events_results = []
    events_library = ProbabilityEngine.default_events()

    for symbol, tfs in datasets.items():
        df_1h = tfs.get("1h")
        if df_1h is None or df_1h.empty:
            continue

        # Split train pour l'analyse
        n = len(df_1h)
        split_idx = int(n * prob_engine.train_ratio)
        df_train = df_1h.iloc[:split_idx]

        logger.info("  Scan %s 1h — %d événements sur %d bougies train",
                     symbol, len(events_library), len(df_train))

        scan_df = prob_engine.scan_events(df_train, events_library)
        scan_df["asset"] = symbol
        scan_df["timeframe"] = "1h"
        all_events_results.append(scan_df)

    import pandas as pd
    events_df = pd.concat(all_events_results, ignore_index=True) if all_events_results else pd.DataFrame()

    if not events_df.empty:
        n_valid = events_df["valide"].sum()
        logger.info("  → %d événements évalués, %d valides (N≥30, p<0.05)",
                     len(events_df), n_valid)

    # Corrélations laggées multi-asset
    logger.info("  Calcul des corrélations laggées...")
    corr_datasets = {sym: tfs["1h"] for sym, tfs in datasets.items() if "1h" in tfs}
    correlation_df = prob_engine.lagged_correlation_matrix(corr_datasets, max_lag=12)

    # Stabilité glissante pour les top événements
    if not events_df.empty:
        top_events = events_df[events_df["valide"] == True].nlargest(5, "rr_ratio")
        for _, row in top_events.iterrows():
            event = row.get("event")
            asset = row.get("asset")
            if event and asset and asset in datasets:
                df_1h = datasets[asset]["1h"]
                stability = prob_engine.rolling_stability(df_1h, event)
                if not stability.empty:
                    cv = stability["p_up_3j"].std() / stability["p_up_3j"].mean() \
                        if stability["p_up_3j"].mean() > 0 else float("inf")
                    logger.info("  Stabilité %s %s — CV=%.3f (%s)",
                                asset, row.get("event_desc", "")[:40],
                                cv, "stable" if cv < 0.5 else "instable")

    # ══════════════════════════════════════════════════════════
    # ÉTAPE 4+5+6 — Backtest (Train)
    # ══════════════════════════════════════════════════════════
    logger.info("\n[4-6/7] Backtester — Phase TRAIN")
    backtester = Backtester(CONFIG)
    strategies = get_all_variants(CONFIG)

    logger.info("  %d variantes × %d assets = %d backtests",
                 len(strategies), n_assets, len(strategies) * n_assets)

    train_results = backtester.run_all_variants(
        datasets, strategies, funding_data, phase="train"
    )

    if not train_results.empty:
        logger.info("  Train — Top 5 par Sharpe:")
        top5 = train_results.nlargest(5, "sharpe_ratio")
        for _, r in top5.iterrows():
            logger.info("    %s %s %s %.0fx — Sharpe=%.3f, Return=%.1f%%, DD=%.1f%%",
                        r["strategy"], r["asset"], r["timeframe"],
                        r["leverage"], r["sharpe_ratio"],
                        r["total_return"], r["max_drawdown"])

    # ══════════════════════════════════════════════════════════
    # ÉTAPE 6b — Backtest (Test)
    # ══════════════════════════════════════════════════════════
    logger.info("\n[6b/7] Backtester — Phase TEST")
    test_results = backtester.run_all_variants(
        datasets, strategies, funding_data, phase="test"
    )

    # Merger train + test pour le rapport
    if not train_results.empty:
        train_results["phase"] = "train"
    if not test_results.empty:
        test_results["phase"] = "test"

    # ══════════════════════════════════════════════════════════
    # ÉTAPE 6c — Kelly sur le test set
    # ══════════════════════════════════════════════════════════
    logger.info("\n[6c/7] Kelly Fractional")
    kelly_results = backtester.run_with_kelly(
        datasets, train_results, funding_data,
        min_sharpe=0.5, min_trades=10,
    )

    if not kelly_results.empty:
        logger.info("  Kelly — %d configurations testées", len(kelly_results))

    # ══════════════════════════════════════════════════════════
    # ÉTAPE 7 — Rapport
    # ══════════════════════════════════════════════════════════
    logger.info("\n[7/7] Reporter — Génération du rapport")
    reporter = Reporter(CONFIG)

    # Alimenter le reporter
    reporter.set_events(events_df)
    reporter.set_correlations(correlation_df)
    reporter.set_results(test_results)
    reporter.set_kelly(kelly_results)

    # Equity curves des top 5 stratégies (test set)
    if not test_results.empty:
        top5_test = test_results.nlargest(5, "sharpe_ratio")
        for _, r in top5_test.iterrows():
            strat_name = r["strategy"]
            symbol = r["asset"]
            tf = r["timeframe"]

            # Recréer et relancer pour obtenir l'equity curve
            cls = STRATEGY_REGISTRY.get(strat_name)
            if cls is None:
                continue

            params = {
                "leverage": r["leverage"],
                "size_pct": r["size_pct"],
                "sl_pct": r["sl_pct"],
                "tp_pct": r["tp_pct"],
            }
            for key in ("grid_spacing_pct", "rsi_entry", "tp1_pct",
                         "trailing_pct", "atr_percentile_max",
                         "volume_ratio_min"):
                if key in r.index and pd.notna(r[key]):
                    params[key] = r[key]

            strategy = cls(config_path=CONFIG, **params)

            tfs = datasets.get(symbol, {})
            df_5m = tfs.get("5m")
            df_tf = tfs.get(tf)
            if df_5m is None or df_tf is None:
                continue

            _, test_5m = backtester.split_data(df_5m)
            _, test_tf = backtester.split_data(df_tf)

            df_aux = {}
            for aux_tf in ("1h", "4h"):
                if aux_tf != tf and aux_tf in tfs:
                    _, aux_test = backtester.split_data(tfs[aux_tf])
                    df_aux[aux_tf] = aux_test

            result = backtester.run_strategy(
                df_strategy=test_tf,
                df_5m=test_5m,
                strategy=strategy,
                funding_rates=funding_data.get(symbol),
                df_aux=df_aux if df_aux else None,
            )

            label = f"{strat_name} {symbol} {r['leverage']:.0f}x"
            reporter.set_equity_curve(label, result.equity_curve,
                                      result.liquidation_indices)

    # BTC buy-and-hold benchmark
    btc_data = datasets.get("BTC/USDT", {}).get("1h")
    if btc_data is not None and not btc_data.empty:
        n = len(btc_data)
        split_idx = int(n * backtester.train_ratio)
        btc_test = btc_data.iloc[split_idx:]
        btc_bh = Backtester.buy_hold_benchmark(btc_test, initial=100.0)
        reporter.set_btc_benchmark(btc_bh)

    # Générer
    report_path = reporter.generate_report(str(ROOT / "reports"))

    # ══════════════════════════════════════════════════════════
    # RÉSUMÉ FINAL
    # ══════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 70)
    logger.info("  RECHERCHE TERMINÉE en %.1f secondes", elapsed)
    logger.info("=" * 70)
    logger.info("  Données     : %d assets, %d bougies", n_assets, n_candles)
    logger.info("  Événements  : %d évalués, %d valides",
                 len(events_df) if not events_df.empty else 0,
                 events_df["valide"].sum() if not events_df.empty else 0)
    logger.info("  Backtests   : %d train + %d test",
                 len(train_results) if not train_results.empty else 0,
                 len(test_results) if not test_results.empty else 0)
    logger.info("  Kelly       : %d configurations",
                 len(kelly_results) if not kelly_results.empty else 0)
    logger.info("  Rapport     : %s", report_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
