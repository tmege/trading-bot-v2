#!/usr/bin/env python3
"""
Sweep BTC V2 — Trouver un remplaçant pour btc_momentum_score_1h.

Teste les 7 stratégies V2 × grille complète sur BTC/USDT 1h.
6 solutions identiques au sweep ETH, adaptées pour BTC.
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

# BTC : funding rate plus bas, hold plus long (trends longer)
REALISTIC_EC = ExecConfig(
    equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=72,
)
INITIAL_EQUITY = 1000.0


def load_btc():
    fe = FeatureEngine()
    df_5m = pd.read_parquet("data/BTC_USDT_5m_ohlcv.parquet")
    df_5m = df_5m[~df_5m.index.duplicated(keep="first")]
    df_1h = df_5m.resample("1h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["open"])
    df_1h = fe.compute_all(df_1h)
    print(f"  BTC/USDT: {len(df_1h):,} bougies 1h "
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
    print("  SOLUTION 1 : CRITÈRE DE STABILITÉ ASSOUPLI (BTC)")
    print("=" * 100)

    perf = defaultdict(dict)
    for window_name, results in all_window_results.items():
        if "Full" in window_name:
            continue
        for r in results:
            key = "%s|%s" % (r["strat_name"], str(sorted(r["params"].items())))
            perf[key][window_name] = r["sharpe_ratio"]

    all_candidates = []
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

        print("\n  --- Sharpe > 0 sur %d/5 fenêtres : %d variantes ---" %
              (min_positive, len(candidates)))

        if candidates:
            print("  %-35s %6s %7s %7s %6s %4s %s" %
                  ("Stratégie", "AvgSR", "MinSR", "MaxSR", "StdSR", "Pos", "Détail par fenêtre"))
            print("  " + "-" * 110)
            for c in candidates[:20]:
                parts = c["key"].split("|", 1)
                name = parts[0]
                detail = "  ".join("%s:%+.1f" % (w, s) for w, s in c["sharpes"].items())
                print("  %-35s %+5.2f %+6.2f %+6.2f %6.2f %d/%d  %s" %
                      (name, c["avg_sharpe"], c["min_sharpe"],
                       c["max_sharpe"], c["std_sharpe"],
                       c["n_positive"], c["n_windows"], detail))
            all_candidates.extend(candidates)

    return all_candidates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOLUTION 2 : Portfolio statique
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def solution2_portfolio(df_full, bt):
    """Backteste des portfolios de 2-3 stratégies complémentaires."""
    print("\n" + "=" * 100)
    print("  SOLUTION 2 : PORTFOLIO STATIQUE BTC")
    print("=" * 100)

    # BTC-oriented portfolio configs
    PORTFOLIOS = {
        "Port A: Breakout + EMA": [
            ("StratBreakoutRelaxed",  {"lookback": 15, "vol_breakout_min": 2.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 5.0}),
            ("StratEmaCrossover",     {"ema_fast": 9, "ema_slow": 50, "use_regime_filter": True, "sl_buffer_pct": 0.5, "tp_pct": 5.0}),
        ],
        "Port B: EMA + MR": [
            ("StratEmaCrossover",     {"ema_fast": 12, "ema_slow": 50, "use_regime_filter": True, "sl_buffer_pct": 0.5, "tp_pct": 8.0}),
            ("StratMeanReversionBB",  {"rsi_oversold": 30, "rsi_overbought": 70, "bb_entry_low": 0.05, "bb_entry_high": 0.95, "sl_pct": 2.0, "tp_pct": 5.0}),
        ],
        "Port C: Breakout + Stoch": [
            ("StratBreakoutRelaxed",  {"lookback": 20, "vol_breakout_min": 2.5, "use_compression": False, "sl_pct": 1.5, "tp_pct": 6.0}),
            ("StratStochReversal",    {"oversold": 20, "overbought": 80, "vol_min": 1.0, "sl_pct": 2.0, "tp_pct": 5.0}),
        ],
        "Port D: Triple": [
            ("StratBreakoutRelaxed",  {"lookback": 15, "vol_breakout_min": 2.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 5.0}),
            ("StratEmaCrossover",     {"ema_fast": 9, "ema_slow": 50, "use_regime_filter": True, "sl_buffer_pct": 0.5, "tp_pct": 5.0}),
            ("StratMeanReversionBB",  {"rsi_oversold": 30, "rsi_overbought": 70, "bb_entry_low": 0.05, "bb_entry_high": 0.95, "sl_pct": 2.0, "tp_pct": 5.0}),
        ],
    }

    for port_name, strats in PORTFOLIOS.items():
        n_strats = len(strats)
        port_equity_pct = 0.30 / n_strats

        print("\n  -- %s (%d strats, sizing %.0f%% chacune) --" %
              (port_name, n_strats, port_equity_pct * 100))
        print("  %-12s %6s %5s %7s %9s %9s %7s" %
              ("Fenêtre", "Trades", "WR", "Sharpe", "$PnL", "Final$", "MaxDD"))
        print("  " + "-" * 65)

        ec = ExecConfig(
            equity_pct=port_equity_pct, leverage=5,
            cooldown_bars=4, max_hold_bars=72,
        )

        for window_name, start, end in WINDOWS_FULL:
            df_w = slice_window(df_full, start, end)
            if len(df_w) < 200:
                continue

            total_pnl = 0.0
            total_trades = 0
            total_wins = 0
            max_dd_combined = 0.0
            all_pnls = []

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

                if "trades_detail" in metrics:
                    all_pnls.extend([t["pnl_pct"] for t in metrics["trades_detail"]])

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
            print("  %-12s %6d %4.0f%% %+7.2f %+9.2f %9.2f %6.1f%%" %
                  (window_name, total_trades, wr, sharpe, total_pnl, final_equity, dd_pct))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOLUTION 3 : Regime-switch dynamique
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StratRegimeSwitch:
    """Meta-stratégie : route vers la strat spécialisée selon le régime."""

    def __init__(self, config: dict):
        self.bull_strat = V2_STRATEGY_REGISTRY["StratBreakoutRelaxed"](
            config.get("bull_params", {"lookback": 15, "vol_breakout_min": 2.0,
                                        "use_compression": False, "sl_pct": 1.5, "tp_pct": 5.0})
        )
        self.bear_strat = V2_STRATEGY_REGISTRY["StratMeanReversionBB"](
            config.get("bear_params", {"rsi_oversold": 30, "rsi_overbought": 70,
                                        "bb_entry_low": 0.05, "bb_entry_high": 0.95,
                                        "sl_pct": 2.0, "tp_pct": 5.0})
        )
        self.ranging_strat = V2_STRATEGY_REGISTRY["StratStochReversal"](
            config.get("ranging_params", {"oversold": 20, "overbought": 80,
                                           "vol_min": 1.0, "sl_pct": 2.0, "tp_pct": 4.0})
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
    print("  SOLUTION 3 : REGIME-SWITCH DYNAMIQUE (BTC)")
    print("=" * 100)

    VARIANTS = [
        {
            "name": "Switch v1 (default)",
            "config": {},
        },
        {
            "name": "Switch v2 (tight SL)",
            "config": {
                "bull_params": {"lookback": 10, "vol_breakout_min": 1.5, "use_compression": False, "sl_pct": 1.0, "tp_pct": 4.0},
                "bear_params": {"rsi_oversold": 25, "rsi_overbought": 75, "bb_entry_low": 0.05, "bb_entry_high": 0.95, "sl_pct": 1.5, "tp_pct": 4.0},
                "ranging_params": {"oversold": 20, "overbought": 80, "vol_min": 0.8, "sl_pct": 1.5, "tp_pct": 3.0},
                "sl_pct": 1.5, "tp_pct": 4.0,
            },
        },
        {
            "name": "Switch v3 (wide TP)",
            "config": {
                "bull_params": {"lookback": 20, "vol_breakout_min": 2.5, "use_compression": False, "sl_pct": 2.0, "tp_pct": 8.0},
                "bear_params": {"rsi_oversold": 30, "rsi_overbought": 65, "bb_entry_low": 0.10, "bb_entry_high": 0.90, "sl_pct": 2.5, "tp_pct": 6.0},
                "ranging_params": {"oversold": 25, "overbought": 75, "vol_min": 1.0, "sl_pct": 2.5, "tp_pct": 6.0},
                "sl_pct": 2.5, "tp_pct": 6.0,
            },
        },
    ]

    for variant in VARIANTS:
        meta = StratRegimeSwitch(variant["config"])
        print("\n  -- %s --" % variant["name"])
        print("  %-12s %6s %5s %7s %9s %9s %7s" %
              ("Fenêtre", "Trades", "WR", "Sharpe", "$PnL", "Final$", "MaxDD"))
        print("  " + "-" * 65)

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
            print("  %-12s %6d %4.0f%% %+7.2f %+9.2f %9.2f %6.1f%%" %
                  (window_name, metrics["nb_trades"], wr,
                   metrics["sharpe_ratio"], pnl, final, dd))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOLUTION 4 : Walk-forward optimization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def solution4_walk_forward(df_full, bt):
    """Walk-forward : optimiser sur 6M, tester sur 3M suivants."""
    print("\n" + "=" * 100)
    print("  SOLUTION 4 : WALK-FORWARD OPTIMIZATION (BTC)")
    print("  Train=6M rolling, Test=3M forward")
    print("=" * 100)

    WF_WINDOWS = [
        ("2023-01-01", "2023-07-01", "2023-07-01", "2023-10-01"),
        ("2023-04-01", "2023-10-01", "2023-10-01", "2024-01-01"),
        ("2023-07-01", "2024-01-01", "2024-01-01", "2024-04-01"),
        ("2024-01-01", "2024-07-01", "2024-07-01", "2024-10-01"),
        ("2024-04-01", "2024-10-01", "2024-10-01", "2025-01-01"),
        ("2024-07-01", "2025-01-01", "2025-01-01", "2025-04-01"),
    ]

    print("\n  %-28s %7s %12s %7s %7s %9s %6s %5s %s" %
          ("Period", "Train", "Best Train", "Test", "Sharpe", "$PnL", "Trades", "WR", "Strat chosen"))
    print("  " + "-" * 120)

    cumulative_pnl = 0.0
    all_test_sharpes = []

    for train_start, train_end, test_start, test_end in WF_WINDOWS:
        df_train = slice_window(df_full, train_start, train_end)
        df_test = slice_window(df_full, test_start, test_end)

        if len(df_train) < 200 or len(df_test) < 100:
            continue

        train_results = run_all_strats(df_train, bt, exec_config=REALISTIC_EC,
                                       initial_equity=INITIAL_EQUITY)
        if not train_results:
            continue

        best = max(train_results, key=lambda x: x["sharpe_ratio"])
        best_train_sharpe = best["sharpe_ratio"]

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

        period_label = "%s->%s" % (train_start[:7], test_end[:7])
        print("  %-28s %6dh %+11.2f %6dh %+7.2f %+9.2f %6d %4.0f%% %s" %
              (period_label, len(df_train), best_train_sharpe,
               len(df_test), test_sharpe, test_pnl,
               test_metrics["nb_trades"], test_wr,
               best["strat_name"]))

    if all_test_sharpes:
        avg_sharpe = np.mean(all_test_sharpes)
        n_pos = sum(1 for s in all_test_sharpes if s > 0)
        print("\n  Walk-forward summary:")
        print("    Cumulative test PnL: $%+.2f" % cumulative_pnl)
        print("    Avg test Sharpe: %+.2f" % avg_sharpe)
        print("    Positive test windows: %d/%d" % (n_pos, len(all_test_sharpes)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOLUTION 5 : Nouveaux signaux
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StratMultiTFConfirm:
    """Confirmation multi-timeframe : signal 1h confirmé par tendance 4h."""

    def __init__(self, config: dict):
        self.sub_strat = V2_STRATEGY_REGISTRY["StratMeanReversionBB"](
            config.get("sub_params", {"rsi_oversold": 30, "rsi_overbought": 70,
                                       "bb_entry_low": 0.05, "bb_entry_high": 0.95,
                                       "sl_pct": 2.0, "tp_pct": 5.0})
        )
        self.sl_pct = config.get("sl_pct", 2.0)
        self.tp_pct = config.get("tp_pct", 5.0)
        self.max_hold = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df_4h = df["close"].resample("4h").last().dropna()
        ema9_4h = df_4h.ewm(span=9, adjust=False).mean()
        ema21_4h = df_4h.ewm(span=21, adjust=False).mean()
        trend_4h = (ema9_4h > ema21_4h).astype(int) - (ema9_4h < ema21_4h).astype(int)
        trend_1h = trend_4h.reindex(df.index, method="ffill").fillna(0)

        sub_signals = self.sub_strat.generate_signals(df)
        signals = pd.Series(0, index=df.index)
        signals.loc[(sub_signals == 1) & (trend_1h >= 0)] = 1
        signals.loc[(sub_signals == -1) & (trend_1h <= 0)] = -1
        return signals

    def signal_frequency(self, df: pd.DataFrame) -> dict:
        return _signal_frequency(self.generate_signals(df), len(df))


class StratRSIDivergence:
    """Divergence RSI-prix."""

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

        price_low = close.rolling(lb).min()
        price_high = close.rolling(lb).max()
        rsi_low = rsi.rolling(lb).min()
        rsi_high = rsi.rolling(lb).max()

        prev_price_low = price_low.shift(lb)
        prev_rsi_low = rsi_low.shift(lb)
        prev_price_high = price_high.shift(lb)
        prev_rsi_high = rsi_high.shift(lb)

        bull_div = (price_low < prev_price_low) & (rsi_low > prev_rsi_low) & (rsi < 40)
        bear_div = (price_high > prev_price_high) & (rsi_high < prev_rsi_high) & (rsi > 60)

        signals.loc[bull_div] = 1
        signals.loc[bear_div] = -1
        return signals

    def signal_frequency(self, df: pd.DataFrame) -> dict:
        return _signal_frequency(self.generate_signals(df), len(df))


class StratEMACrossoverBTC:
    """EMA crossover BTC-optimisé : slow EMA + wide stops pour BTC trends."""

    def __init__(self, config: dict):
        self.ema_fast = config.get("ema_fast", 12)
        self.ema_slow = config.get("ema_slow", 50)
        self.sl_pct = config.get("sl_pct", 2.0)
        self.tp_pct = config.get("tp_pct", 8.0)
        self.max_hold = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = df["close"].ewm(span=self.ema_fast, adjust=False).mean()
        slow = df["close"].ewm(span=self.ema_slow, adjust=False).mean()

        signals = pd.Series(0, index=df.index)
        cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        signals.loc[cross_up] = 1
        signals.loc[cross_down] = -1
        return signals

    def signal_frequency(self, df: pd.DataFrame) -> dict:
        return _signal_frequency(self.generate_signals(df), len(df))


class StratMACDMomentum:
    """MACD Momentum : trade MACD histogram changes with volume confirmation."""

    def __init__(self, config: dict):
        self.fast = config.get("fast", 12)
        self.slow = config.get("slow", 26)
        self.signal = config.get("signal", 9)
        self.vol_min = config.get("vol_min", 1.0)
        self.sl_pct = config.get("sl_pct", 2.0)
        self.tp_pct = config.get("tp_pct", 5.0)
        self.max_hold = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal, adjust=False).mean()
        hist = macd - signal_line

        # Volume filter
        vol_ratio = df.get("vol_ratio")
        if vol_ratio is None:
            vol_sma = df["volume"].rolling(20).mean()
            vol_ratio = df["volume"] / vol_sma.replace(0, 1)

        signals = pd.Series(0, index=df.index)

        # Bullish: histogram crosses above 0 with volume
        bull = (hist > 0) & (hist.shift(1) <= 0) & (vol_ratio >= self.vol_min)
        # Bearish: histogram crosses below 0 with volume
        bear = (hist < 0) & (hist.shift(1) >= 0) & (vol_ratio >= self.vol_min)

        signals.loc[bull] = 1
        signals.loc[bear] = -1
        return signals

    def signal_frequency(self, df: pd.DataFrame) -> dict:
        return _signal_frequency(self.generate_signals(df), len(df))


def solution5_new_signals(df_full, bt):
    """Teste les nouveaux types de signaux pour BTC."""
    print("\n" + "=" * 100)
    print("  SOLUTION 5 : NOUVEAUX TYPES DE SIGNAUX (BTC)")
    print("=" * 100)

    NEW_STRATS = {
        "MultiTF Confirm (MR+4h)": [
            StratMultiTFConfirm({"sl_pct": 2.0, "tp_pct": 5.0}),
            StratMultiTFConfirm({"sl_pct": 1.5, "tp_pct": 4.0}),
            StratMultiTFConfirm({"sl_pct": 2.5, "tp_pct": 6.0}),
        ],
        "RSI Divergence": [
            StratRSIDivergence({"lookback": 10, "sl_pct": 2.0, "tp_pct": 5.0}),
            StratRSIDivergence({"lookback": 14, "sl_pct": 2.0, "tp_pct": 5.0}),
            StratRSIDivergence({"lookback": 20, "sl_pct": 1.5, "tp_pct": 4.0}),
            StratRSIDivergence({"lookback": 14, "sl_pct": 2.5, "tp_pct": 6.0}),
        ],
        "EMA Crossover BTC": [
            StratEMACrossoverBTC({"ema_fast": 9, "ema_slow": 50, "sl_pct": 2.0, "tp_pct": 6.0}),
            StratEMACrossoverBTC({"ema_fast": 12, "ema_slow": 50, "sl_pct": 2.0, "tp_pct": 8.0}),
            StratEMACrossoverBTC({"ema_fast": 9, "ema_slow": 100, "sl_pct": 2.5, "tp_pct": 8.0}),
            StratEMACrossoverBTC({"ema_fast": 21, "ema_slow": 100, "sl_pct": 3.0, "tp_pct": 10.0}),
        ],
        "MACD Momentum": [
            StratMACDMomentum({"fast": 12, "slow": 26, "signal": 9, "vol_min": 1.0, "sl_pct": 2.0, "tp_pct": 5.0}),
            StratMACDMomentum({"fast": 12, "slow": 26, "signal": 9, "vol_min": 1.5, "sl_pct": 2.0, "tp_pct": 6.0}),
            StratMACDMomentum({"fast": 8, "slow": 21, "signal": 5, "vol_min": 1.0, "sl_pct": 1.5, "tp_pct": 5.0}),
            StratMACDMomentum({"fast": 12, "slow": 26, "signal": 9, "vol_min": 0.8, "sl_pct": 2.5, "tp_pct": 8.0}),
        ],
    }

    for strat_type, variants in NEW_STRATS.items():
        print("\n  -- %s (%d variantes) --" % (strat_type, len(variants)))

        # Header with window names
        header = "  %-12s %4s" % ("Fenêtre", "Var")
        for w, _, _ in WINDOWS:
            header += " %10s" % w
        header += " %7s %6s %7s" % ("Avg", "Std", "Stable")
        print(header)
        print("  " + "-" * 100)

        # Test each variant across all windows
        for vi, strat in enumerate(variants):
            sharpes = []
            row = "  %-12s v%-3d" % (strat_type[:12], vi)

            for window_name, start, end in WINDOWS:
                df_w = slice_window(df_full, start, end)
                if len(df_w) < 200:
                    row += " %10s" % "N/A"
                    continue

                signals = strat.generate_signals(df_w)
                n_sig = (signals != 0).sum()
                if n_sig < 3:
                    row += " %10s" % "no_sig"
                    continue

                metrics = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct,
                                 strat.max_hold, exec_config=REALISTIC_EC,
                                 initial_equity=INITIAL_EQUITY)
                metrics.pop("trades_detail", None)
                s = metrics["sharpe_ratio"]
                sharpes.append(s)
                row += " %+10.2f" % s

            if sharpes:
                avg = np.mean(sharpes)
                std = np.std(sharpes)
                n_pos = sum(1 for s in sharpes if s > 0)
                flag = "OUI" if n_pos >= 4 and std < 2.0 else ("3/5" if n_pos >= 3 else "non")
                row += " %+6.2f %6.2f %7s" % (avg, std, flag)
            print(row)

        # Full 3Y performance for each variant
        print("\n  Full 3Y performance:")
        print("  %-12s %4s %6s %5s %7s %9s %9s %7s" %
              ("Type", "Var", "Trades", "WR", "Sharpe", "$PnL", "Final$", "MaxDD"))
        print("  " + "-" * 70)

        for vi, strat in enumerate(variants):
            for window_name, start, end in [("Full 3Y", "2023-01-01", "2026-01-01")]:
                df_w = slice_window(df_full, start, end)
                if len(df_w) < 200:
                    continue
                signals = strat.generate_signals(df_w)
                n_sig = (signals != 0).sum()
                if n_sig < 3:
                    print("  %-12s v%-3d   -- pas assez de signaux --" % (strat_type[:12], vi))
                    continue
                metrics = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct,
                                 strat.max_hold, exec_config=REALISTIC_EC,
                                 initial_equity=INITIAL_EQUITY)
                metrics.pop("trades_detail", None)
                wr = metrics["win_rate"] * 100
                dd = metrics["max_drawdown"] * 100
                pnl = metrics.get("dollar_pnl", 0)
                final = metrics.get("final_equity", INITIAL_EQUITY)
                print("  %-12s v%-3d %6d %4.0f%% %+7.2f %+9.2f %9.2f %6.1f%%" %
                      (strat_type[:12], vi, metrics["nb_trades"], wr,
                       metrics["sharpe_ratio"], pnl, final, dd))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOLUTION 6 : Sorties adaptatives
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def solution6_adaptive_exits(df_full, bt):
    """Teste différentes méthodes de sortie sur les mêmes entrées."""
    print("\n" + "=" * 100)
    print("  SOLUTION 6 : SORTIES ADAPTATIVES (BTC)")
    print("  Mêmes entrées, sorties différentes")
    print("=" * 100)

    # Test with multiple entry strategies
    ENTRY_STRATS = [
        ("StratBreakoutRelaxed", {"lookback": 15, "vol_breakout_min": 2.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 5.0}),
        ("StratEmaCrossover", {"ema_fast": 9, "ema_slow": 50, "use_regime_filter": True, "sl_buffer_pct": 0.5, "tp_pct": 5.0}),
        ("StratMeanReversionBB", {"rsi_oversold": 30, "rsi_overbought": 70, "bb_entry_low": 0.05, "bb_entry_high": 0.95, "sl_pct": 2.0, "tp_pct": 5.0}),
    ]

    EXIT_CONFIGS = [
        ("SL=1.5% TP=4%",      1.5, 4.0, 48),
        ("SL=2.0% TP=5%",      2.0, 5.0, 48),
        ("SL=2.0% TP=6%",      2.0, 6.0, 72),
        ("SL=2.5% TP=8%",      2.5, 8.0, 72),
        ("SL=1.5% TP=5% H96",  1.5, 5.0, 96),
        ("SL=2.0% TP=8% H96",  2.0, 8.0, 96),
        ("SL=3.0% TP=10%",     3.0, 10.0, 72),
        ("SL=1.0% TP=3% H24",  1.0, 3.0, 24),
        ("SL=2.0% TP=5% H24",  2.0, 5.0, 24),
        ("SL=2.5% TP=6% H48",  2.5, 6.0, 48),
    ]

    for strat_name, params in ENTRY_STRATS:
        cls = V2_STRATEGY_REGISTRY[strat_name]
        base_strat = cls(params)

        print("\n  -- Entrées : %s --" % strat_name)
        header = "  %-25s" % "Exit config"
        for w, _, _ in WINDOWS:
            header += " %10s" % w
        header += " %7s %6s %7s" % ("Avg", "Std", "Stable")
        print(header)
        print("  " + "-" * 110)

        for exit_name, sl, tp, max_hold in EXIT_CONFIGS:
            ec = ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4,
                            max_hold_bars=max_hold)
            sharpes = []

            row = "  %-25s" % exit_name
            for window_name, start, end in WINDOWS:
                df_w = slice_window(df_full, start, end)
                if len(df_w) < 200:
                    row += " %10s" % "N/A"
                    continue

                signals = base_strat.generate_signals(df_w)
                metrics = bt.run(df_w, signals, sl, tp, max_hold,
                                 exec_config=ec, initial_equity=INITIAL_EQUITY)
                metrics.pop("trades_detail", None)
                s = metrics["sharpe_ratio"]
                sharpes.append(s)
                row += " %+10.2f" % s

            if sharpes:
                avg = np.mean(sharpes)
                std = np.std(sharpes)
                n_pos = sum(1 for s in sharpes if s > 0)
                flag = "OUI" if n_pos >= 4 and std < 2.0 else ("3/5" if n_pos >= 3 else "non")
                row += " %+6.2f %6.2f %7s" % (avg, std, flag)
            print(row)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    t0 = time.time()

    print("=" * 100)
    print("  SWEEP BTC V2 — TROUVER UN REMPLAÇANT POUR btc_momentum_score_1h")
    print("  7 stratégies × 1,880 combinaisons × 5 fenêtres 6M + full 3Y")
    print("=" * 100)

    print("\n-- Chargement --")
    df_full = load_btc()
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
        print("  %s: %d variantes valides" % (window_name, n_valid))

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
    print("\n" + "=" * 100)
    print("  Temps total : %.1fs (%.1f min)" % (elapsed, elapsed / 60))
    print("=" * 100)


if __name__ == "__main__":
    main()
