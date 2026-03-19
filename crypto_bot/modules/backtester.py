"""
Module 6 — Backtester + Kelly
Exécute les stratégies sur données historiques avec simulation
de liquidation sur bougies 5min. Split train/test temporel.
Kelly fractional en post-processing.
Tout en % du portfolio.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from .liquidation_engine import LiquidationEngine, ExitReason
from .strategies import BaseStrategy, TradeSignal, STRATEGY_REGISTRY, get_all_variants

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Résultat complet d'un backtest."""
    strategy_name: str
    asset: str
    timeframe: str
    leverage: float
    params: dict
    trades: list[dict]
    equity_curve: pd.Series       # en % du portfolio (part de 100)
    metrics: dict
    liquidation_indices: list[int]  # indices dans l'equity curve


class Backtester:
    """Backtester avec simulation de liquidation 5min,
    funding rates, frais maker/taker, et Kelly sizing."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.config_path = config_path
        self.train_ratio: float = self.cfg["train_ratio"]
        self.capital_initial: float = self.cfg["capital_initial"]  # 100.0%
        self.stop_global: float = self.cfg["stop_global_portfolio"]

        self.liq_engine = LiquidationEngine(config_path)

        logger.info("Backtester initialisé — train_ratio=%.2f, stop=%.1f%%",
                     self.train_ratio, self.stop_global)

    # ── Split temporel ────────────────────────────────────────

    def split_data(self, df: pd.DataFrame
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split train/test temporel strict.
        Train: premiers 75%, Test: derniers 25%.
        Ne mélange jamais."""
        n = len(df)
        split_idx = int(n * self.train_ratio)
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        logger.info("Split — train: %d bougies [%s → %s] | test: %d [%s → %s]",
                     len(train), train.index[0].date(), train.index[-1].date(),
                     len(test), test.index[0].date(), test.index[-1].date())
        return train, test

    # ── Mapping timeframe → 5min ──────────────────────────────

    @staticmethod
    def _find_5m_range(ts_start: pd.Timestamp, ts_end: pd.Timestamp,
                       df_5m: pd.DataFrame) -> tuple[int, int]:
        """Trouve les indices 5min correspondant à une période."""
        mask = (df_5m.index >= ts_start) & (df_5m.index < ts_end)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return -1, -1
        return int(indices[0]), int(indices[-1])

    @staticmethod
    def _find_5m_index(ts: pd.Timestamp, df_5m: pd.DataFrame) -> int:
        """Trouve l'index 5min le plus proche d'un timestamp."""
        idx = df_5m.index.searchsorted(ts)
        return min(idx, len(df_5m) - 1)

    # ── Exit condition builder ────────────────────────────────

    @staticmethod
    def _build_exit_condition(signal: TradeSignal,
                              df_strategy: pd.DataFrame
                              ) -> callable | None:
        """Construit la fonction de sortie technique pour une position.
        Fonctionne sur les bougies 5min en interpolant l'état des indicateurs
        de la bougie strategy-timeframe la plus récente."""

        if signal.exit_condition is None:
            return None

        if signal.exit_condition == "rsi_above_50_or_above_ema21":
            # Pour mean reversion : on vérifie sur les bougies strategy
            # Comme on simule en 5min, on prend l'état du RSI/EMA21
            # de la dernière bougie 1h connue
            def condition(row_5m):
                # Le backtester injectera les indicateurs interpolés
                rsi = row_5m.get("RSI_14_interp")
                ema21 = row_5m.get("EMA21_interp")
                close = row_5m["close"]
                if rsi is not None and rsi > 50:
                    return True
                if ema21 is not None and close > ema21:
                    return True
                return False
            return condition

        return None

    # ── Interpolation indicateurs sur 5min ────────────────────

    @staticmethod
    def _interpolate_indicators(df_5m: pd.DataFrame,
                                df_strategy: pd.DataFrame,
                                columns: list[str]) -> pd.DataFrame:
        """Forward-fill les indicateurs du timeframe stratégie sur le 5min.
        Chaque bougie 5min hérite de la dernière valeur connue."""
        df_5m = df_5m.copy()
        for col in columns:
            if col not in df_strategy.columns:
                continue
            # Créer une série au timeframe stratégie
            indicator = df_strategy[col].rename(f"{col}_interp")
            # Reindex sur le 5min et forward fill
            df_5m[f"{col}_interp"] = indicator.reindex(
                df_5m.index, method="ffill"
            )
        return df_5m

    # ── Run une stratégie ─────────────────────────────────────

    def run_strategy(self, df_strategy: pd.DataFrame,
                     df_5m: pd.DataFrame,
                     strategy: BaseStrategy,
                     funding_rates: pd.Series | None = None,
                     df_aux: dict[str, pd.DataFrame] | None = None,
                     size_override: float | None = None,
                     ) -> BacktestResult:
        """Exécute une stratégie sur une période.

        Args:
            df_strategy: DataFrame au timeframe de la stratégie (avec indicateurs)
            df_5m: DataFrame 5min (pour simulation de liquidation précise)
            strategy: instance de la stratégie
            funding_rates: Series optionnelle
            df_aux: DataFrames auxiliaires (ex: {"4h": df_4h})
            size_override: si défini, remplace le size_pct de chaque signal (Kelly)
        """
        # 1. Générer les signaux
        signals = strategy.generate_signals(df_strategy, df_5m, df_aux)

        if not signals:
            logger.info("  %s — aucun signal", strategy)
            return BacktestResult(
                strategy_name=strategy.name,
                asset="", timeframe=strategy.preferred_timeframe,
                leverage=strategy.leverage, params={},
                trades=[], equity_curve=pd.Series(dtype=float),
                metrics=self._empty_metrics(), liquidation_indices=[]
            )

        # 2. Interpoler les indicateurs nécessaires sur le 5min
        interp_cols = ["RSI_14", "EMA21", "EMA50", "EMA200"]
        df_5m_enriched = self._interpolate_indicators(df_5m, df_strategy, interp_cols)

        # 3. Simuler chaque trade
        capital = self.capital_initial  # 100.0%
        trades: list[dict] = []
        equity_points: list[tuple[pd.Timestamp, float]] = []
        liquidation_indices: list[int] = []
        open_positions: int = 0
        last_exit_idx: int = 0
        global_stopped = False

        # Point initial de l'equity curve
        equity_points.append((df_5m.index[0], capital))

        for signal in signals:
            if global_stopped:
                break

            # Vérifier stop global
            if capital < self.stop_global:
                logger.warning("  STOP GLOBAL à %.2f%% — arrêt du trading", capital)
                global_stopped = True
                break

            # Trouver le timestamp de la bougie signal dans le timeframe stratégie
            if signal.idx >= len(df_strategy):
                continue
            signal_ts = df_strategy.index[signal.idx]

            # Mapper vers l'index 5min
            entry_5m_idx = self._find_5m_index(signal_ts, df_5m_enriched)

            # Ne pas ouvrir si on a encore une position active (selon max_positions)
            if entry_5m_idx <= last_exit_idx:
                continue

            # Override de taille si Kelly
            size = size_override if size_override is not None else signal.size_pct

            # Vérifier que le capital permet d'ouvrir la position
            if size > capital:
                logger.debug("  Capital insuffisant: %.2f%% < %.2f%%", capital, size)
                continue

            # Exit condition
            exit_cond = self._build_exit_condition(signal, df_strategy)

            # 4. Simuler la position sur les bougies 5min
            result = self.liq_engine.simulate_position(
                df_5m=df_5m_enriched,
                entry_idx=entry_5m_idx,
                side=signal.side,
                leverage=signal.leverage if size_override is None else strategy.leverage,
                size_pct=size,
                sl_pct=signal.sl_pct,
                tp_pct=signal.tp_pct,
                funding_rates=funding_rates,
                tp_partial=signal.tp_partial,
                trailing_stop_pct=signal.trailing_stop_pct,
                exit_condition=exit_cond,
                entry_order_type=signal.entry_order_type,
                exit_tp_order_type=signal.exit_tp_order_type,
                exit_sl_order_type=signal.exit_sl_order_type,
            )

            # Mettre à jour le capital
            capital += result["pnl_pct"]
            result["capital_after"] = round(capital, 6)

            # Tracker
            trades.append(result)
            if result["exit_time"] is not None:
                equity_points.append((result["exit_time"], capital))

            if result["exit_reason"] == "liquidation":
                liquidation_indices.append(len(equity_points) - 1)

            last_exit_idx = result["exit_idx"]

        # Point final
        if df_5m.index[-1] != equity_points[-1][0]:
            equity_points.append((df_5m.index[-1], capital))

        # Construire l'equity curve
        eq_idx, eq_vals = zip(*equity_points) if equity_points else ([], [])
        equity_curve = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_idx))
        equity_curve = equity_curve[~equity_curve.index.duplicated(keep="last")]

        # 5. Calculer les métriques
        metrics = self.compute_metrics(trades, equity_curve)

        logger.info("  %s — %d trades, PnL=%.2f%%, Sharpe=%.3f, MaxDD=%.2f%%",
                     strategy, len(trades), metrics["total_return"],
                     metrics["sharpe_ratio"], metrics["max_drawdown"])

        return BacktestResult(
            strategy_name=strategy.name,
            asset="",  # sera rempli par run_all_variants
            timeframe=strategy.preferred_timeframe,
            leverage=strategy.leverage,
            params={k: v for k, v in strategy.__dict__.items()
                    if k not in ("cfg", "config_path")},
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
            liquidation_indices=liquidation_indices,
        )

    # ── Métriques ─────────────────────────────────────────────

    def compute_metrics(self, trades: list[dict],
                        equity_curve: pd.Series) -> dict:
        """Calcule toutes les métriques de performance.
        Tout en % du portfolio."""

        if not trades:
            return self._empty_metrics()

        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        durations = [t["duration_hours"] for t in trades]

        # Total return
        total_return = equity_curve.iloc[-1] - self.capital_initial

        # CAGR
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = max(days / 365.25, 0.01)
        final_ratio = equity_curve.iloc[-1] / self.capital_initial
        cagr = (final_ratio ** (1 / years) - 1) * 100 if final_ratio > 0 else -100

        # Sharpe ratio (annualisé)
        if len(pnls) > 1:
            avg_trade_hours = np.mean(durations) if durations else 24
            trades_per_year = (365.25 * 24) / max(avg_trade_hours, 1)
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls, ddof=1)
            sharpe = (mean_pnl / std_pnl * np.sqrt(trades_per_year)
                      if std_pnl > 0 else 0)
        else:
            sharpe = 0

        # Sortino ratio (annualisé, downside deviation)
        downside = [p for p in pnls if p < 0]
        if len(downside) > 1:
            avg_trade_hours = np.mean(durations) if durations else 24
            trades_per_year = (365.25 * 24) / max(avg_trade_hours, 1)
            downside_std = np.std(downside, ddof=1)
            sortino = (np.mean(pnls) / downside_std * np.sqrt(trades_per_year)
                       if downside_std > 0 else 0)
        else:
            sortino = 0

        # Max drawdown (sur l'equity curve)
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Drawdown duration (en jours)
        in_dd = drawdown < 0
        if in_dd.any():
            dd_groups = (~in_dd).cumsum()
            dd_durations = []
            for _, group in in_dd.groupby(dd_groups):
                if group.any():
                    dur = (group.index[-1] - group.index[0]).days
                    dd_durations.append(dur)
            max_dd_duration = max(dd_durations) if dd_durations else 0
        else:
            max_dd_duration = 0

        # Win rate
        win_rate = len(wins) / len(trades) if trades else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Avg win / avg loss
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0

        # Nb trades & liquidations
        nb_trades = len(trades)
        nb_liquidations = sum(1 for t in trades if t["exit_reason"] == "liquidation")

        # Funding cost total
        funding_total = sum(t["funding_pct"] for t in trades)

        # Fees total
        fees_total = sum(t["fees_pct"] for t in trades)

        return {
            "total_return": round(total_return, 4),
            "cagr": round(cagr, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_drawdown, 4),
            "max_dd_duration_days": max_dd_duration,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "nb_trades": nb_trades,
            "nb_liquidations": nb_liquidations,
            "funding_cost_total": round(funding_total, 4),
            "fees_total": round(fees_total, 4),
            "avg_duration_hours": round(np.mean(durations), 2) if durations else 0,
            "final_capital": round(equity_curve.iloc[-1], 4),
        }

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "total_return": 0, "cagr": 0, "sharpe_ratio": 0,
            "sortino_ratio": 0, "max_drawdown": 0,
            "max_dd_duration_days": 0, "win_rate": 0,
            "profit_factor": 0, "avg_win": 0, "avg_loss": 0,
            "nb_trades": 0, "nb_liquidations": 0,
            "funding_cost_total": 0, "fees_total": 0,
            "avg_duration_hours": 0, "final_capital": 100.0,
        }

    # ── Buy & Hold BTC benchmark ──────────────────────────────

    @staticmethod
    def buy_hold_benchmark(df: pd.DataFrame,
                           initial: float = 100.0) -> pd.Series:
        """Equity curve buy-and-hold normalisée à initial%."""
        return (df["close"] / df["close"].iloc[0]) * initial

    # ── Kelly Criterion ───────────────────────────────────────

    @staticmethod
    def kelly_fraction(trades: list[dict]) -> float:
        """Kelly fractional (quart) pour le sizing conservateur.

        f* = W/a - (1-W)/b  (Kelly complet)
        f_kelly = f* / 4     (Kelly quart)

        W = win rate
        b = avg win (en %)
        a = avg loss (en %, valeur absolue)

        Retourne le sizing en % du portfolio.
        Clampé entre 0 et 25% pour éviter les extrêmes.
        """
        if not trades:
            return 0

        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p <= 0]

        if not wins or not losses:
            return 0

        W = len(wins) / len(trades)
        b = np.mean(wins)
        a = np.mean(losses)

        if a == 0 or b == 0:
            return 0

        f_star = W / a - (1 - W) / b
        f_kelly = f_star / 4  # Kelly quart

        # Clamper entre 0 et 25%
        f_kelly = max(0, min(25.0, f_kelly))

        return round(f_kelly, 4)

    # ── Run all variants ──────────────────────────────────────

    def run_all_variants(
        self,
        datasets: dict[str, dict[str, pd.DataFrame]],
        strategies: list[BaseStrategy] | None = None,
        funding_data: dict[str, pd.Series] | None = None,
        phase: str = "train",
    ) -> pd.DataFrame:
        """Boucle sur toutes les variantes (strat × asset × levier × tf).

        Args:
            datasets: {symbol: {"5m": df, "1h": df, "4h": df}}
            strategies: liste de stratégies (défaut: toutes les variantes)
            funding_data: {symbol: Series}
            phase: "train" ou "test" — détermine quel split utiliser

        Retourne DataFrame de résultats classés par Sharpe.
        """
        if strategies is None:
            strategies = get_all_variants(self.config_path)

        if funding_data is None:
            funding_data = {}

        all_results: list[dict] = []
        total_runs = len(strategies) * len(datasets)
        run_idx = 0

        for symbol, tfs in datasets.items():
            df_5m = tfs.get("5m")
            if df_5m is None or df_5m.empty:
                logger.warning("Pas de données 5min pour %s — skip", symbol)
                continue

            # Split le 5min
            train_5m, test_5m = self.split_data(df_5m)
            df_5m_phase = train_5m if phase == "train" else test_5m

            # Funding
            funding = funding_data.get(symbol)

            for strategy in strategies:
                run_idx += 1
                tf = strategy.preferred_timeframe
                df_tf = tfs.get(tf)

                if df_tf is None or df_tf.empty:
                    logger.warning("Pas de données %s pour %s — skip", tf, symbol)
                    continue

                # Split le timeframe stratégie
                train_tf, test_tf = self.split_data(df_tf)
                df_tf_phase = train_tf if phase == "train" else test_tf

                # DataFrames auxiliaires (pour mean_reversion qui utilise 4h)
                df_aux = {}
                for aux_tf in ("1h", "4h"):
                    if aux_tf != tf and aux_tf in tfs:
                        aux_train, aux_test = self.split_data(tfs[aux_tf])
                        df_aux[aux_tf] = aux_train if phase == "train" else aux_test

                logger.info("[%d/%d] %s × %s × %s (phase=%s)",
                            run_idx, total_runs, strategy.name,
                            symbol, tf, phase)

                # Run
                result = self.run_strategy(
                    df_strategy=df_tf_phase,
                    df_5m=df_5m_phase,
                    strategy=strategy,
                    funding_rates=funding,
                    df_aux=df_aux if df_aux else None,
                )
                result.asset = symbol

                # Kelly sur les trades du train
                kelly = self.kelly_fraction(result.trades)

                # Stocker le résultat
                row = {
                    "strategy": strategy.name,
                    "asset": symbol,
                    "timeframe": tf,
                    "leverage": strategy.leverage,
                    "size_pct": strategy.size_pct,
                    "sl_pct": strategy.sl_pct,
                    "tp_pct": strategy.tp_pct,
                    "phase": phase,
                    "kelly_fraction": kelly,
                    **result.metrics,
                }

                # Ajouter les params spécifiques de la stratégie
                for key in ("grid_spacing_pct", "rsi_entry", "tp1_pct",
                            "trailing_pct", "atr_percentile_max",
                            "volume_ratio_min"):
                    if hasattr(strategy, key):
                        row[key] = getattr(strategy, key)

                all_results.append(row)

        # DataFrame final
        results_df = pd.DataFrame(all_results)
        if not results_df.empty:
            results_df = results_df.sort_values(
                "sharpe_ratio", ascending=False
            ).reset_index(drop=True)

            logger.info("━" * 60)
            logger.info("%s — %d backtests terminés", phase.upper(), len(results_df))
            if len(results_df) > 0:
                best = results_df.iloc[0]
                logger.info("  Meilleur: %s %s %s — Sharpe=%.3f, Return=%.2f%%",
                            best["strategy"], best["asset"], best["timeframe"],
                            best["sharpe_ratio"], best["total_return"])

        return results_df

    # ── Run with Kelly ────────────────────────────────────────

    def run_with_kelly(
        self,
        datasets: dict[str, dict[str, pd.DataFrame]],
        train_results: pd.DataFrame,
        funding_data: dict[str, pd.Series] | None = None,
        min_sharpe: float = 0.5,
        min_trades: int = 10,
    ) -> pd.DataFrame:
        """Relance les backtests avec Kelly sizing sur le test set.

        Filtre les stratégies qui ont passé le train :
        - Sharpe > min_sharpe
        - nb_trades >= min_trades
        - kelly_fraction > 0

        Retourne DataFrame avec colonnes supplémentaires:
        sharpe_kelly, total_return_kelly
        """
        # Filtrer les candidats
        candidates = train_results[
            (train_results["sharpe_ratio"] > min_sharpe) &
            (train_results["nb_trades"] >= min_trades) &
            (train_results["kelly_fraction"] > 0)
        ].copy()

        if candidates.empty:
            logger.warning("Aucun candidat pour Kelly (sharpe>%.1f, trades>=%d)",
                           min_sharpe, min_trades)
            return pd.DataFrame()

        logger.info("Kelly — %d candidats sur %d",
                     len(candidates), len(train_results))

        kelly_results: list[dict] = []

        for _, row in candidates.iterrows():
            strat_name = row["strategy"]
            symbol = row["asset"]
            tf = row["timeframe"]

            # Recréer la stratégie avec les mêmes paramètres
            cls = STRATEGY_REGISTRY.get(strat_name)
            if cls is None:
                continue

            params = {
                "leverage": row["leverage"],
                "size_pct": row["size_pct"],
                "sl_pct": row["sl_pct"],
                "tp_pct": row["tp_pct"],
            }
            for key in ("grid_spacing_pct", "rsi_entry", "tp1_pct",
                         "trailing_pct", "atr_percentile_max",
                         "volume_ratio_min"):
                if key in row.index and pd.notna(row[key]):
                    params[key] = row[key]

            strategy = cls(config_path=self.config_path, **params)

            # Données
            tfs = datasets.get(symbol, {})
            df_5m = tfs.get("5m")
            df_tf = tfs.get(tf)
            if df_5m is None or df_tf is None:
                continue

            _, test_5m = self.split_data(df_5m)
            _, test_tf = self.split_data(df_tf)

            df_aux = {}
            for aux_tf in ("1h", "4h"):
                if aux_tf != tf and aux_tf in tfs:
                    _, aux_test = self.split_data(tfs[aux_tf])
                    df_aux[aux_tf] = aux_test

            funding = (funding_data or {}).get(symbol)

            # Run avec Kelly sizing
            kelly_size = row["kelly_fraction"]
            logger.info("  Kelly %s %s %s — size=%.2f%% (was %.2f%%)",
                        strat_name, symbol, tf, kelly_size, row["size_pct"])

            result = self.run_strategy(
                df_strategy=test_tf,
                df_5m=test_5m,
                strategy=strategy,
                funding_rates=funding,
                df_aux=df_aux if df_aux else None,
                size_override=kelly_size,
            )

            kelly_results.append({
                "strategy": strat_name,
                "asset": symbol,
                "timeframe": tf,
                "leverage": row["leverage"],
                "kelly_fraction": kelly_size,
                "sharpe_kelly": result.metrics["sharpe_ratio"],
                "total_return_kelly": result.metrics["total_return"],
                "max_drawdown_kelly": result.metrics["max_drawdown"],
                "nb_trades_kelly": result.metrics["nb_trades"],
                "nb_liquidations_kelly": result.metrics["nb_liquidations"],
                # Résultats originaux du train pour comparaison
                "sharpe_train": row["sharpe_ratio"],
                "total_return_train": row["total_return"],
            })

        kelly_df = pd.DataFrame(kelly_results)
        if not kelly_df.empty:
            kelly_df = kelly_df.sort_values(
                "sharpe_kelly", ascending=False
            ).reset_index(drop=True)

        return kelly_df


# ── Standalone test ───────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s"
    )

    # Test Kelly
    print("=== Test Kelly Fraction ===")
    fake_trades = [
        {"pnl_pct": 3.0}, {"pnl_pct": -1.5}, {"pnl_pct": 2.0},
        {"pnl_pct": -1.0}, {"pnl_pct": 4.0}, {"pnl_pct": -2.0},
        {"pnl_pct": 1.5}, {"pnl_pct": 3.0}, {"pnl_pct": -1.0},
        {"pnl_pct": 2.5},
    ]
    kelly = Backtester.kelly_fraction(fake_trades)
    W = sum(1 for t in fake_trades if t["pnl_pct"] > 0) / len(fake_trades)
    print(f"  Win rate: {W:.0%}")
    print(f"  Kelly quart: {kelly:.4f}% du portfolio")

    # Test métriques
    print("\n=== Test Métriques ===")
    bt = Backtester()
    dates = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    eq = pd.Series(
        [100, 103, 101, 106, 104, 110, 107, 112, 109, 115],
        index=dates
    )
    trades = [
        {"pnl_pct": 3, "duration_hours": 24, "exit_reason": "tp",
         "fees_pct": 0.1, "funding_pct": 0.01},
        {"pnl_pct": -2, "duration_hours": 12, "exit_reason": "sl",
         "fees_pct": 0.1, "funding_pct": 0.01},
        {"pnl_pct": 5, "duration_hours": 48, "exit_reason": "tp",
         "fees_pct": 0.1, "funding_pct": 0.02},
        {"pnl_pct": -2, "duration_hours": 6, "exit_reason": "liquidation",
         "fees_pct": 0, "funding_pct": 0},
        {"pnl_pct": 6, "duration_hours": 36, "exit_reason": "trailing",
         "fees_pct": 0.1, "funding_pct": 0.01},
        {"pnl_pct": -3, "duration_hours": 8, "exit_reason": "sl",
         "fees_pct": 0.1, "funding_pct": 0.01},
        {"pnl_pct": 5, "duration_hours": 24, "exit_reason": "tp",
         "fees_pct": 0.1, "funding_pct": 0.01},
        {"pnl_pct": -2, "duration_hours": 12, "exit_reason": "sl",
         "fees_pct": 0.1, "funding_pct": 0.01},
        {"pnl_pct": 3, "duration_hours": 24, "exit_reason": "tp",
         "fees_pct": 0.1, "funding_pct": 0.01},
        {"pnl_pct": 2, "duration_hours": 18, "exit_reason": "tp",
         "fees_pct": 0.1, "funding_pct": 0.01},
    ]
    metrics = bt.compute_metrics(trades, eq)
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")
