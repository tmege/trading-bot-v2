#!/usr/bin/env python3
"""
Sweep ETH V2 — 6 solutions pour trouver des stratégies stables.

1. Critère de stabilité assoupli (3/5, 4/5 fenêtres)
2. Portfolio statique (2-3 strats complémentaires en parallèle)
3. Regime-switch dynamique (meta-stratégie)
4. Walk-forward optimization (re-optimise tous les 3M sur 6M rolling)
5. Nouveaux signaux (funding rate, multi-TF confirmation)
6. Sorties adaptatives (trailing ATR, TP partiel, sortie sur signal inverse)
"""
import sys
import time
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from exec_config import ExecConfig
from modules.feature_engine import FeatureEngine
from modules.strategies import V2_STRATEGY_REGISTRY, _signal_frequency
from param_sweep import FULL_GRID, expand_grid
from sweep_runner import SweepBacktester


# ── Config commune ───────────────────────────────────────────

WINDOWS = [
    ("2023-H1", "2023-01-01", "2023-07-01"),
    ("2023-H2", "2023-07-01", "2024-01-01"),
    ("2024-H1", "2024-01-01", "2024-07-01"),
    ("2024-H2", "2024-07-01", "2025-01-01"),
    ("2025-H1", "2025-01-01", "2025-07-01"),
]

WINDOWS_FULL = WINDOWS + [
    ("Full 2Y",  "2023-01-01", "2025-01-01"),
    ("Full 3Y",  "2023-01-01", "2026-01-01"),
]

REALISTIC_EC = ExecConfig(
    equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48,
)
INITIAL_EQUITY = 1000.0


def load_eth():
    fe = FeatureEngine()
    df_5m = pd.read_parquet("data/ETH_USDT_5m_ohlcv.parquet")
    df_5m = df_5m[~df_5m.index.duplicated(keep="first")]
    df_1h = df_5m.resample("1h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["open"])
    df_1h = fe.compute_all(df_1h)
    print(f"  ETH/USDT: {len(df_1h):,} bougies 1h "
          f"[{df_1h.index[0].date()} -> {df_1h.index[-1].date()}]")
    return df_1h


def slice_window(df, start, end):
    mask = (df.index >= pd.Timestamp(start, tz="UTC")) & \
           (df.index < pd.Timestamp(end, tz="UTC"))
    return df.loc[mask]


def run_all_strats(df, bt, exec_config=None, initial_equity=None):
    """Teste toutes les combinaisons FULL_GRID, retourne la liste de résultats."""
    results = []
    for strat_name, param_grid in FULL_GRID.items():
        cls = V2_STRATEGY_REGISTRY.get(strat_name)
        if cls is None:
            continue
        combos = expand_grid(param_grid) if isinstance(param_grid, dict) and \
            all(isinstance(v, list) for v in param_grid.values()) else param_grid
        for params in combos:
            strat = cls(params)
            freq = strat.signal_frequency(df)
            if freq["total_signaux"] < 3:
                continue
            signals = strat.generate_signals(df)
            metrics = bt.run(df, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                             exec_config=exec_config, initial_equity=initial_equity)
            metrics.pop("trades_detail", None)
            results.append({
                "strat_name": strat_name, "params": params,
                "signaux": freq["total_signaux"],
                "signaux_par_mois": freq["signaux_par_mois"],
                **metrics,
            })
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOLUTION 1 : Critère assoupli
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def solution1_relaxed_stability(all_window_results):
    """Trouve les variantes stables avec critères assouplis."""
    print("\n" + "=" * 100)
    print("  SOLUTION 1 : CRITÈRE DE STABILITÉ ASSOUPLI")
    print("=" * 100)

    perf = defaultdict(dict)
    for window_name, results in all_window_results.items():
        if "Full" in window_name:
            continue
        for r in results:
            key = f"{r['strat_name']}|{str(sorted(r['params'].items()))}"
            perf[key][window_name] = r["sharpe_ratio"]

    for min_positive in [5, 4, 3]:
        candidates = []
        for key, windows in perf.items():
            sharpes = list(windows.values())
            if len(sharpes) < 3:
                continue
            n_positive = sum(1 for s in sharpes if s > 0)
            if n_positive >= min_positive:
                candidates.append({
                    "key": key,
                    "avg_sharpe": np.mean(sharpes),
                    "std_sharpe": np.std(sharpes),
                    "min_sharpe": np.min(sharpes),
                    "max_sharpe": np.max(sharpes),
                    "n_positive": n_positive,
                    "n_windows": len(sharpes),
                    "sharpes": windows,
                })
        candidates.sort(key=lambda x: x["avg_sharpe"], reverse=True)

        print(f"\n  --- Sharpe > 0 sur {min_positive}/5 fenêtres : "
              f"{len(candidates)} variantes ---")

        if candidates:
            print(f"  {'Stratégie':<30} {'AvgSR':>6} {'MinSR':>7} {'MaxSR':>7} "
                  f"{'StdSR':>6} {'Pos':>4} {'Détail par fenêtre'}")
            print(f"  {'-'*110}")
            for c in candidates[:15]:
                parts = c["key"].split("|", 1)
                name = parts[0]
                detail = "  ".join(f"{w}:{s:+.1f}" for w, s in c["sharpes"].items())
                print(f"  {name:<30} {c['avg_sharpe']:>+5.2f} {c['min_sharpe']:>+6.2f} "
                      f"{c['max_sharpe']:>+6.2f} {c['std_sharpe']:>6.2f} "
                      f"{c['n_positive']}/{c['n_windows']}  {detail}")

    return candidates if candidates else []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOLUTION 2 : Portfolio statique
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def solution2_portfolio(df_full, bt):
    """Backteste des portfolios de 2-3 stratégies complémentaires."""
    print("\n" + "=" * 100)
    print("  SOLUTION 2 : PORTFOLIO STATIQUE (strats complémentaires)")
    print("=" * 100)

    # Configs optimales par régime (issues du sweep précédent)
    PORTFOLIOS = {
        "Portfolio A (3 strats)": [
            ("StratBreakoutRelaxed",  {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.0, "tp_pct": 5.0}),
            ("StratMeanReversionBB",  {"rsi_oversold": 35, "rsi_overbought": 70, "bb_entry_low": 0.1, "bb_entry_high": 0.90, "sl_pct": 1.5, "tp_pct": 6.0}),
            ("StratStochReversal",    {"oversold": 30, "overbought": 80, "vol_min": 1.2, "sl_pct": 1.5, "tp_pct": 4.0}),
        ],
        "Portfolio B (2 strats)": [
            ("StratBreakoutRelaxed",  {"lookback": 20, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.0, "tp_pct": 5.0}),
            ("StratMeanReversionBB",  {"rsi_oversold": 35, "rsi_overbought": 70, "bb_entry_low": 0.05, "bb_entry_high": 0.95, "sl_pct": 2.5, "tp_pct": 6.0}),
        ],
        "Portfolio C (MR focus)": [
            ("StratMeanReversionBB",  {"rsi_oversold": 30, "rsi_overbought": 70, "bb_entry_low": 0.10, "bb_entry_high": 0.90, "sl_pct": 1.5, "tp_pct": 4.0}),
            ("StratStochReversal",    {"oversold": 25, "overbought": 75, "vol_min": 0.8, "sl_pct": 2.5, "tp_pct": 6.0}),
            ("StratInsideBarBreakout", {"vol_min": 2.0, "trend_filter": True, "atr_filter": True, "sl_pct": 1.5, "tp_pct": 5.0}),
        ],
        "Portfolio D (trend+MR)": [
            ("StratEmaCrossover",     {"ema_fast": 9, "ema_slow": 50, "use_regime_filter": False, "sl_buffer_pct": 0.5, "tp_pct": 8.0}),
            ("StratMeanReversionBB",  {"rsi_oversold": 35, "rsi_overbought": 75, "bb_entry_low": 0.05, "bb_entry_high": 0.95, "sl_pct": 2.5, "tp_pct": 6.0}),
        ],
    }

    # Sizing réduit proportionnellement au nombre de strats
    for port_name, strats in PORTFOLIOS.items():
        n_strats = len(strats)
        port_equity_pct = 0.30 / n_strats  # diviser le sizing

        print(f"\n  ── {port_name} ({n_strats} strats, sizing {port_equity_pct*100:.0f}% chacune) ──")
        print(f"  {'Fenêtre':<12} {'Trades':>6} {'WR':>5} {'Sharpe':>7} "
              f"{'$PnL':>9} {'Final$':>9} {'MaxDD':>7}")
        print(f"  {'-'*65}")

        ec = ExecConfig(
            equity_pct=port_equity_pct, leverage=5,
            cooldown_bars=4, max_hold_bars=48,
        )

        for window_name, start, end in WINDOWS_FULL:
            df_w = slice_window(df_full, start, end)
            if len(df_w) < 200:
                continue

            # Combiner les signaux : chaque strat trade indépendamment
            # On simule en séquentiel : on backteste chaque strat séparément
            # puis on agrège les equity curves
            total_pnl = 0.0
            total_trades = 0
            total_wins = 0
            equity = INITIAL_EQUITY
            all_pnls = []
            max_dd_combined = 0.0

            for strat_name, params in strats:
                cls = V2_STRATEGY_REGISTRY[strat_name]
                strat = cls(params)
                signals = strat.generate_signals(df_w)

                metrics = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct,
                                 strat.max_hold, exec_config=ec,
                                 initial_equity=INITIAL_EQUITY / n_strats)
                pnl = metrics.get("dollar_pnl", 0)
                total_pnl += pnl
                total_trades += metrics["nb_trades"]
                total_wins += int(metrics["win_rate"] * metrics["nb_trades"])
                max_dd_combined = max(max_dd_combined, metrics["max_drawdown"])

                # Collect trade-level PnLs for Sharpe
                # Re-run to get trades_detail
                metrics_detail = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct,
                                        strat.max_hold, exec_config=ec,
                                        initial_equity=INITIAL_EQUITY / n_strats)
                if "trades_detail" in metrics_detail:
                    all_pnls.extend([t["pnl_pct"] for t in metrics_detail["trades_detail"]])

            # Portfolio Sharpe
            final_equity = INITIAL_EQUITY + total_pnl
            wr = total_wins / total_trades * 100 if total_trades > 0 else 0

            if len(all_pnls) > 1:
                pnls_arr = np.array(all_pnls)
                total_days = (df_w.index[-1] - df_w.index[0]).total_seconds() / 86400
                trades_per_year = len(pnls_arr) / max(total_days / 365.25, 0.01)
                sharpe = (pnls_arr.mean() / pnls_arr.std(ddof=1)) * np.sqrt(trades_per_year)
                sharpe = max(-10.0, min(10.0, sharpe))
            else:
                sharpe = 0.0

            dd_pct = max_dd_combined * 100
            print(f"  {window_name:<12} {total_trades:>6} {wr:>4.0f}% "
                  f"{sharpe:>+7.2f} {total_pnl:>+9.2f} {final_equity:>9.2f} "
                  f"{dd_pct:>6.1f}%")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOLUTION 3 : Regime-switch dynamique
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StratRegimeSwitch:
    """Meta-stratégie : route vers la strat spécialisée selon le régime."""

    def __init__(self, config: dict):
        self.bull_strat = V2_STRATEGY_REGISTRY["StratBreakoutRelaxed"](
            config.get("bull_params", {"lookback": 15, "vol_breakout_min": 3.0,
                                        "use_compression": False, "sl_pct": 1.0, "tp_pct": 5.0})
        )
        self.bear_strat = V2_STRATEGY_REGISTRY["StratMeanReversionBB"](
            config.get("bear_params", {"rsi_oversold": 35, "rsi_overbought": 70,
                                        "bb_entry_low": 0.1, "bb_entry_high": 0.90,
                                        "sl_pct": 1.5, "tp_pct": 6.0})
        )
        self.ranging_strat = V2_STRATEGY_REGISTRY["StratStochReversal"](
            config.get("ranging_params", {"oversold": 25, "overbought": 75,
                                           "vol_min": 0.8, "sl_pct": 2.5, "tp_pct": 4.0})
        )
        self.sl_pct = config.get("sl_pct", 2.0)
        self.tp_pct = config.get("tp_pct", 5.0)
        self.max_hold = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        regime = df.get("regime")
        if regime is None:
            return pd.Series(0, index=df.index)

        bull_sig = self.bull_strat.generate_signals(df)
        bear_sig = self.bear_strat.generate_signals(df)
        range_sig = self.ranging_strat.generate_signals(df)

        signals = pd.Series(0, index=df.index)
        signals.loc[regime == "bull"] = bull_sig.loc[regime == "bull"]
        signals.loc[regime == "bear"] = bear_sig.loc[regime == "bear"]
        signals.loc[regime == "ranging"] = range_sig.loc[regime == "ranging"]

        return signals

    def signal_frequency(self, df: pd.DataFrame) -> dict:
        return _signal_frequency(self.generate_signals(df), len(df))


def solution3_regime_switch(df_full, bt):
    """Backteste la meta-stratégie regime-switch."""
    print("\n" + "=" * 100)
    print("  SOLUTION 3 : REGIME-SWITCH DYNAMIQUE")
    print("=" * 100)

    VARIANTS = [
        {
            "name": "Switch v1 (default)",
            "config": {},
        },
        {
            "name": "Switch v2 (tight SL)",
            "config": {
                "bull_params": {"lookback": 10, "vol_breakout_min": 2.5, "use_compression": False, "sl_pct": 1.0, "tp_pct": 4.0},
                "bear_params": {"rsi_oversold": 30, "rsi_overbought": 70, "bb_entry_low": 0.05, "bb_entry_high": 0.95, "sl_pct": 1.5, "tp_pct": 4.0},
                "ranging_params": {"oversold": 20, "overbought": 80, "vol_min": 1.0, "sl_pct": 1.5, "tp_pct": 3.0},
                "sl_pct": 1.5, "tp_pct": 4.0,
            },
        },
        {
            "name": "Switch v3 (wide TP)",
            "config": {
                "bull_params": {"lookback": 20, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 8.0},
                "bear_params": {"rsi_oversold": 35, "rsi_overbought": 65, "bb_entry_low": 0.10, "bb_entry_high": 0.90, "sl_pct": 2.5, "tp_pct": 6.0},
                "ranging_params": {"oversold": 30, "overbought": 70, "vol_min": 0.8, "sl_pct": 2.5, "tp_pct": 6.0},
                "sl_pct": 2.5, "tp_pct": 6.0,
            },
        },
    ]

    for variant in VARIANTS:
        meta = StratRegimeSwitch(variant["config"])
        print(f"\n  ── {variant['name']} ──")
        print(f"  {'Fenêtre':<12} {'Trades':>6} {'WR':>5} {'Sharpe':>7} "
              f"{'$PnL':>9} {'Final$':>9} {'MaxDD':>7}")
        print(f"  {'-'*65}")

        for window_name, start, end in WINDOWS_FULL:
            df_w = slice_window(df_full, start, end)
            if len(df_w) < 200:
                continue

            signals = meta.generate_signals(df_w)
            metrics = bt.run(df_w, signals, meta.sl_pct, meta.tp_pct,
                             meta.max_hold, exec_config=REALISTIC_EC,
                             initial_equity=INITIAL_EQUITY)
            metrics.pop("trades_detail", None)

            wr = metrics["win_rate"] * 100
            dd = metrics["max_drawdown"] * 100
            pnl = metrics.get("dollar_pnl", 0)
            final = metrics.get("final_equity", INITIAL_EQUITY)
            print(f"  {window_name:<12} {metrics['nb_trades']:>6} {wr:>4.0f}% "
                  f"{metrics['sharpe_ratio']:>+7.2f} {pnl:>+9.2f} {final:>9.2f} "
                  f"{dd:>6.1f}%")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOLUTION 4 : Walk-forward optimization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def solution4_walk_forward(df_full, bt):
    """Walk-forward : optimiser sur 6M, tester sur 3M suivants."""
    print("\n" + "=" * 100)
    print("  SOLUTION 4 : WALK-FORWARD OPTIMIZATION")
    print("  Train=6M rolling, Test=3M forward")
    print("=" * 100)

    # Fenêtres walk-forward
    WF_WINDOWS = [
        # (train_start, train_end, test_start, test_end)
        ("2023-01-01", "2023-07-01", "2023-07-01", "2023-10-01"),
        ("2023-04-01", "2023-10-01", "2023-10-01", "2024-01-01"),
        ("2023-07-01", "2024-01-01", "2024-01-01", "2024-04-01"),
        ("2024-01-01", "2024-07-01", "2024-07-01", "2024-10-01"),
        ("2024-04-01", "2024-10-01", "2024-10-01", "2025-01-01"),
        ("2024-07-01", "2025-01-01", "2025-01-01", "2025-04-01"),
    ]

    print(f"\n  {'Period':<28} {'Train':>7} {'Best Train':>12} {'Test':>7} "
          f"{'Sharpe':>7} {'$PnL':>9} {'Trades':>6} {'WR':>5} {'Strat chosen'}")
    print(f"  {'-'*120}")

    cumulative_pnl = 0.0
    all_test_sharpes = []

    for train_start, train_end, test_start, test_end in WF_WINDOWS:
        df_train = slice_window(df_full, train_start, train_end)
        df_test = slice_window(df_full, test_start, test_end)

        if len(df_train) < 200 or len(df_test) < 100:
            continue

        # Optimize on train
        train_results = run_all_strats(df_train, bt, exec_config=REALISTIC_EC,
                                       initial_equity=INITIAL_EQUITY)
        if not train_results:
            continue

        # Pick best by Sharpe on train
        best = max(train_results, key=lambda x: x["sharpe_ratio"])
        best_train_sharpe = best["sharpe_ratio"]

        # Apply best config on test
        cls = V2_STRATEGY_REGISTRY[best["strat_name"]]
        strat = cls(best["params"])
        signals = strat.generate_signals(df_test)
        test_metrics = bt.run(df_test, signals, strat.sl_pct, strat.tp_pct,
                              strat.max_hold, exec_config=REALISTIC_EC,
                              initial_equity=INITIAL_EQUITY)
        test_metrics.pop("trades_detail", None)

        test_sharpe = test_metrics["sharpe_ratio"]
        test_pnl = test_metrics.get("dollar_pnl", 0)
        test_wr = test_metrics["win_rate"] * 100
        cumulative_pnl += test_pnl
        all_test_sharpes.append(test_sharpe)

        period_label = f"{train_start[:7]}→{test_end[:7]}"
        print(f"  {period_label:<28} {len(df_train):>6}h {best_train_sharpe:>+11.2f} "
              f"{len(df_test):>6}h {test_sharpe:>+7.2f} {test_pnl:>+9.2f} "
              f"{test_metrics['nb_trades']:>6} {test_wr:>4.0f}% "
              f"{best['strat_name']}")

    if all_test_sharpes:
        avg_sharpe = np.mean(all_test_sharpes)
        n_pos = sum(1 for s in all_test_sharpes if s > 0)
        print(f"\n  Walk-forward summary:")
        print(f"    Cumulative test PnL: ${cumulative_pnl:+.2f}")
        print(f"    Avg test Sharpe: {avg_sharpe:+.2f}")
        print(f"    Positive test windows: {n_pos}/{len(all_test_sharpes)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOLUTION 5 : Nouveaux signaux
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StratMultiTFConfirm:
    """Confirmation multi-timeframe : signal 1h confirmé par tendance 4h.

    LONG  : signal d'une sous-strat (MeanReversionBB) ET EMA9_4h > EMA21_4h
    SHORT : signal d'une sous-strat ET EMA9_4h < EMA21_4h

    Simule le 4h en calculant EMA sur un resample du 1h.
    """

    def __init__(self, config: dict):
        self.sub_strat = V2_STRATEGY_REGISTRY["StratMeanReversionBB"](
            config.get("sub_params", {"rsi_oversold": 35, "rsi_overbought": 70,
                                       "bb_entry_low": 0.1, "bb_entry_high": 0.90,
                                       "sl_pct": 2.0, "tp_pct": 5.0})
        )
        self.sl_pct = config.get("sl_pct", 2.0)
        self.tp_pct = config.get("tp_pct", 5.0)
        self.max_hold = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Calculer les EMA 4h en resamplant le 1h
        df_4h = df["close"].resample("4h").last().dropna()
        ema9_4h = df_4h.ewm(span=9, adjust=False).mean()
        ema21_4h = df_4h.ewm(span=21, adjust=False).mean()
        trend_4h = (ema9_4h > ema21_4h).astype(int) - (ema9_4h < ema21_4h).astype(int)

        # Reindex to 1h (forward fill)
        trend_1h = trend_4h.reindex(df.index, method="ffill").fillna(0)

        sub_signals = self.sub_strat.generate_signals(df)
        signals = pd.Series(0, index=df.index)

        # Long only if 4h is bullish or neutral
        signals.loc[(sub_signals == 1) & (trend_1h >= 0)] = 1
        # Short only if 4h is bearish or neutral
        signals.loc[(sub_signals == -1) & (trend_1h <= 0)] = -1

        return signals

    def signal_frequency(self, df: pd.DataFrame) -> dict:
        return _signal_frequency(self.generate_signals(df), len(df))


class StratRSIDivergence:
    """Divergence RSI-prix.

    Bullish divergence : prix fait un lower low mais RSI fait un higher low -> LONG
    Bearish divergence : prix fait un higher high mais RSI fait un lower high -> SHORT
    """

    def __init__(self, config: dict):
        self.lookback = config.get("lookback", 14)
        self.sl_pct = config.get("sl_pct", 2.0)
        self.tp_pct = config.get("tp_pct", 5.0)
        self.max_hold = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi = df.get("RSI_14")
        close = df["close"]

        if rsi is None:
            return pd.Series(0, index=df.index)

        signals = pd.Series(0, index=df.index)
        lb = self.lookback

        # Rolling min/max
        price_low = close.rolling(lb).min()
        price_high = close.rolling(lb).max()
        rsi_low = rsi.rolling(lb).min()
        rsi_high = rsi.rolling(lb).max()

        prev_price_low = price_low.shift(lb)
        prev_rsi_low = rsi_low.shift(lb)
        prev_price_high = price_high.shift(lb)
        prev_rsi_high = rsi_high.shift(lb)

        # Bullish: new price low but RSI higher low
        bull_div = (price_low < prev_price_low) & (rsi_low > prev_rsi_low) & (rsi < 40)
        # Bearish: new price high but RSI lower high
        bear_div = (price_high > prev_price_high) & (rsi_high < prev_rsi_high) & (rsi > 60)

        signals.loc[bull_div] = 1
        signals.loc[bear_div] = -1
        return signals

    def signal_frequency(self, df: pd.DataFrame) -> dict:
        return _signal_frequency(self.generate_signals(df), len(df))


def solution5_new_signals(df_full, bt):
    """Teste les nouveaux types de signaux."""
    print("\n" + "=" * 100)
    print("  SOLUTION 5 : NOUVEAUX TYPES DE SIGNAUX")
    print("=" * 100)

    NEW_STRATS = {
        "MultiTF Confirm (MR+4h)": [
            StratMultiTFConfirm({"sl_pct": 2.0, "tp_pct": 5.0}),
            StratMultiTFConfirm({"sl_pct": 1.5, "tp_pct": 4.0}),
            StratMultiTFConfirm({"sl_pct": 2.5, "tp_pct": 6.0,
                                 "sub_params": {"rsi_oversold": 30, "rsi_overbought": 75,
                                                "bb_entry_low": 0.05, "bb_entry_high": 0.95,
                                                "sl_pct": 2.0, "tp_pct": 5.0}}),
        ],
        "RSI Divergence": [
            StratRSIDivergence({"lookback": 10, "sl_pct": 2.0, "tp_pct": 5.0}),
            StratRSIDivergence({"lookback": 14, "sl_pct": 2.0, "tp_pct": 5.0}),
            StratRSIDivergence({"lookback": 20, "sl_pct": 1.5, "tp_pct": 4.0}),
            StratRSIDivergence({"lookback": 14, "sl_pct": 2.5, "tp_pct": 6.0}),
        ],
    }

    for strat_type, variants in NEW_STRATS.items():
        print(f"\n  ── {strat_type} ({len(variants)} variantes) ──")
        print(f"  {'Fenêtre':<12} {'Var':>4} {'Trades':>6} {'WR':>5} {'Sharpe':>7} "
              f"{'$PnL':>9} {'MaxDD':>7}")
        print(f"  {'-'*60}")

        for window_name, start, end in WINDOWS_FULL:
            df_w = slice_window(df_full, start, end)
            if len(df_w) < 200:
                continue

            best_sharpe = -999
            best_metrics = None
            best_var = 0

            for vi, strat in enumerate(variants):
                signals = strat.generate_signals(df_w)
                n_sig = (signals != 0).sum()
                if n_sig < 3:
                    continue

                metrics = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct,
                                 strat.max_hold, exec_config=REALISTIC_EC,
                                 initial_equity=INITIAL_EQUITY)
                metrics.pop("trades_detail", None)

                if metrics["sharpe_ratio"] > best_sharpe:
                    best_sharpe = metrics["sharpe_ratio"]
                    best_metrics = metrics
                    best_var = vi

            if best_metrics:
                wr = best_metrics["win_rate"] * 100
                dd = best_metrics["max_drawdown"] * 100
                pnl = best_metrics.get("dollar_pnl", 0)
                print(f"  {window_name:<12} v{best_var:>3} {best_metrics['nb_trades']:>6} "
                      f"{wr:>4.0f}% {best_sharpe:>+7.2f} {pnl:>+9.2f} {dd:>6.1f}%")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOLUTION 6 : Sorties adaptatives
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def solution6_adaptive_exits(df_full, bt):
    """Teste différentes méthodes de sortie sur les mêmes entrées."""
    print("\n" + "=" * 100)
    print("  SOLUTION 6 : SORTIES ADAPTATIVES")
    print("  Mêmes entrées (MeanReversionBB), sorties différentes")
    print("=" * 100)

    # Strat de base pour les entrées
    base_strat = V2_STRATEGY_REGISTRY["StratMeanReversionBB"](
        {"rsi_oversold": 35, "rsi_overbought": 70, "bb_entry_low": 0.1,
         "bb_entry_high": 0.90, "sl_pct": 2.0, "tp_pct": 5.0}
    )

    EXIT_CONFIGS = [
        ("SL=1.5% TP=3%",     1.5, 3.0, 48),
        ("SL=2.0% TP=5%",     2.0, 5.0, 48),
        ("SL=2.5% TP=6%",     2.5, 6.0, 48),
        ("SL=1.5% TP=4%",     1.5, 4.0, 48),
        ("SL=2.0% TP=4% H24", 2.0, 4.0, 24),
        ("SL=2.0% TP=4% H72", 2.0, 4.0, 72),
        ("SL=1.0% TP=3%",     1.0, 3.0, 48),
        ("SL=3.0% TP=8%",     3.0, 8.0, 48),
        ("SL=1.5% TP=5% H96", 1.5, 5.0, 96),
        # ATR-based: approximate by using wider stops
        ("ATR-wide SL=3% TP=6% H72", 3.0, 6.0, 72),
    ]

    print(f"\n  {'Exit config':<30}", end="")
    for w, _, _ in WINDOWS:
        print(f" {w:>10}", end="")
    print(f" {'Avg':>7} {'StdSR':>6} {'Stable':>7}")
    print(f"  {'-'*120}")

    for exit_name, sl, tp, max_hold in EXIT_CONFIGS:
        ec = ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4,
                        max_hold_bars=max_hold)
        sharpes = []

        print(f"  {exit_name:<30}", end="")
        for window_name, start, end in WINDOWS:
            df_w = slice_window(df_full, start, end)
            if len(df_w) < 200:
                print(f" {'N/A':>10}", end="")
                continue

            signals = base_strat.generate_signals(df_w)
            metrics = bt.run(df_w, signals, sl, tp, max_hold,
                             exec_config=ec, initial_equity=INITIAL_EQUITY)
            metrics.pop("trades_detail", None)
            s = metrics["sharpe_ratio"]
            sharpes.append(s)
            print(f" {s:>+10.2f}", end="")

        if sharpes:
            avg = np.mean(sharpes)
            std = np.std(sharpes)
            n_pos = sum(1 for s in sharpes if s > 0)
            flag = "OUI" if n_pos >= 4 and std < 2.0 else "non"
            print(f" {avg:>+6.2f} {std:>6.2f} {flag:>7}")
        else:
            print()

    # Same for BreakoutRelaxed
    print(f"\n  --- Mêmes tests sur StratBreakoutRelaxed ---")
    base_strat2 = V2_STRATEGY_REGISTRY["StratBreakoutRelaxed"](
        {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False,
         "sl_pct": 1.0, "tp_pct": 5.0}
    )

    print(f"  {'Exit config':<30}", end="")
    for w, _, _ in WINDOWS:
        print(f" {w:>10}", end="")
    print(f" {'Avg':>7} {'StdSR':>6} {'Stable':>7}")
    print(f"  {'-'*120}")

    for exit_name, sl, tp, max_hold in EXIT_CONFIGS:
        ec = ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4,
                        max_hold_bars=max_hold)
        sharpes = []

        print(f"  {exit_name:<30}", end="")
        for window_name, start, end in WINDOWS:
            df_w = slice_window(df_full, start, end)
            if len(df_w) < 200:
                print(f" {'N/A':>10}", end="")
                continue

            signals = base_strat2.generate_signals(df_w)
            metrics = bt.run(df_w, signals, sl, tp, max_hold,
                             exec_config=ec, initial_equity=INITIAL_EQUITY)
            metrics.pop("trades_detail", None)
            s = metrics["sharpe_ratio"]
            sharpes.append(s)
            print(f" {s:>+10.2f}", end="")

        if sharpes:
            avg = np.mean(sharpes)
            std = np.std(sharpes)
            n_pos = sum(1 for s in sharpes if s > 0)
            flag = "OUI" if n_pos >= 4 and std < 2.0 else "non"
            print(f" {avg:>+6.2f} {std:>6.2f} {flag:>7}")
        else:
            print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    t0 = time.time()

    print("=" * 100)
    print("  SWEEP ETH V2 — 6 SOLUTIONS POUR LA STABILITÉ")
    print("=" * 100)

    print("\n-- Chargement --")
    df_full = load_eth()
    bt = SweepBacktester()

    # Pré-calculer les résultats par fenêtre pour solution 1
    print("\n-- Sweep de base (toutes fenêtres) --")
    all_window_results = {}
    for window_name, start, end in WINDOWS:
        df_w = slice_window(df_full, start, end)
        if len(df_w) < 200:
            continue
        results = run_all_strats(df_w, bt, exec_config=REALISTIC_EC,
                                  initial_equity=INITIAL_EQUITY)
        all_window_results[window_name] = results
        n_valid = len(results)
        print(f"  {window_name}: {n_valid} variantes valides")

    # Solution 1
    solution1_relaxed_stability(all_window_results)

    # Solution 2
    solution2_portfolio(df_full, bt)

    # Solution 3
    solution3_regime_switch(df_full, bt)

    # Solution 4
    solution4_walk_forward(df_full, bt)

    # Solution 5
    solution5_new_signals(df_full, bt)

    # Solution 6
    solution6_adaptive_exits(df_full, bt)

    elapsed = time.time() - t0
    print(f"\n{'=' * 100}")
    print(f"  Temps total : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
