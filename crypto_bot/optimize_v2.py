#!/usr/bin/env python3
"""
Optimisations réelles du portfolio V2.
Axes qui ne sont PAS du simple tuning de params :

1. Filtre horaire : éviter certaines heures (manipulation asiatique/US open)
2. Filtre jour de semaine : weekend crypto différent
3. Confirmation multi-signal : n'entrer que si 2+ indicateurs concordent
4. Anti-wick : ignorer les signaux sur des bougies à gros wick
5. Trailing stop : remplacer TP fixe par trailing
6. SOL regime filter : ne trader SOL qu'en tendance
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from exec_config import ExecConfig
from modules.feature_engine import FeatureEngine
from modules.strategies import V2_STRATEGY_REGISTRY, _signal_frequency
from sweep_runner import SweepBacktester

INITIAL_EQUITY = 1000.0

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

BASELINE = [
    {"name": "BTC", "asset": "BTC", "strat_class": "StratInsideBarBreakout",
     "params": {"vol_min": 1.5, "trend_filter": True, "atr_filter": True, "sl_pct": 1.5, "tp_pct": 3.0},
     "exec_config": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=72)},
    {"name": "SOL", "asset": "SOL", "strat_class": "StratBreakoutRelaxed",
     "params": {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 4.0},
     "exec_config": ExecConfig(equity_pct=0.30, leverage=5, cooldown_bars=4, max_hold_bars=48)},
    {"name": "ETH", "asset": "ETH", "strat_class": "StratBreakoutRelaxed",
     "params": {"lookback": 15, "vol_breakout_min": 3.0, "use_compression": False, "sl_pct": 1.5, "tp_pct": 5.0},
     "exec_config": ExecConfig(equity_pct=0.20, leverage=5, cooldown_bars=4, max_hold_bars=48)},
]


def run_single(data, bt, asset, strat_class, params, ec, signal_filter=None):
    """Run une stratégie sur toutes les fenêtres, retourne résumé."""
    cls = V2_STRATEGY_REGISTRY[strat_class]
    strat = cls(params)
    df_asset = data[asset]

    results = {}
    for wname, start, end in WINDOWS_FULL:
        df_w = slice_window(df_asset, start, end)
        if len(df_w) < 100:
            continue
        signals = strat.generate_signals(df_w)

        # Apply optional filter
        if signal_filter is not None:
            signals = signal_filter(signals, df_w)

        m = bt.run(df_w, signals, strat.sl_pct, strat.tp_pct, strat.max_hold,
                   exec_config=ec, initial_equity=INITIAL_EQUITY)
        m.pop("trades_detail", None)
        results[wname] = m
    return results


def summarize(results, label):
    """Print one-line summary."""
    r3y = results.get("Full 3Y", {})
    sharpes = [results[w[0]].get("sharpe_ratio", 0) for w in WINDOWS if w[0] in results]
    n_pos = sum(1 for s in sharpes if s > 0)
    avg_sr = np.mean(sharpes) if sharpes else 0

    print("  %-40s  SR %+.2f  AvgSR %+.2f  $%+7.0f  DD %5.1f%%  Tr %4d  WR %4.0f%%  Stab %d/5" %
          (label,
           r3y.get("sharpe_ratio", 0), avg_sr,
           r3y.get("dollar_pnl", 0),
           r3y.get("max_drawdown", 0) * 100,
           r3y.get("nb_trades", 0),
           r3y.get("win_rate", 0) * 100,
           n_pos))


# ═══════════════════════════════════════════════════════════════
# Filtres de signaux
# ═══════════════════════════════════════════════════════════════

def filter_hours(blocked_hours):
    """Bloque les signaux à certaines heures UTC."""
    def _filter(signals, df):
        hours = df.index.hour
        mask = pd.Series(True, index=df.index)
        for h in blocked_hours:
            mask = mask & (hours != h)
        return signals.where(mask, 0)
    return _filter


def filter_weekdays(blocked_days):
    """Bloque certains jours (0=lundi, 6=dimanche)."""
    def _filter(signals, df):
        days = df.index.dayofweek
        mask = pd.Series(True, index=df.index)
        for d in blocked_days:
            mask = mask & (days != d)
        return signals.where(mask, 0)
    return _filter


def filter_anti_wick(max_wick_ratio=0.6):
    """Ignore les signaux sur des bougies à gros wick (manipulation)."""
    def _filter(signals, df):
        body = (df["close"] - df["open"]).abs()
        total_range = df["high"] - df["low"]
        wick_ratio = 1 - body / total_range.replace(0, 1)
        return signals.where(wick_ratio < max_wick_ratio, 0)
    return _filter


def filter_rsi_confirm():
    """N'entre long que si RSI < 70 (pas surachat), short que si RSI > 30."""
    def _filter(signals, df):
        rsi = df.get("RSI_14")
        if rsi is None:
            return signals
        out = signals.copy()
        out.loc[(signals == 1) & (rsi > 70)] = 0   # pas de long en surachat
        out.loc[(signals == -1) & (rsi < 30)] = 0   # pas de short en survente
        return out
    return _filter


def filter_adx_trending(min_adx=20):
    """Ne trade que si ADX > seuil (marché en tendance)."""
    def _filter(signals, df):
        adx = df.get("ADX_14")
        if adx is None:
            return signals
        return signals.where(adx > min_adx, 0)
    return _filter


def filter_regime_bull_only():
    """Ne prend que les longs en bull, shorts en bear."""
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
    """Long uniquement si prix dans moitié basse des BB, short si moitié haute."""
    def _filter(signals, df):
        pct_b = df.get("pct_B")
        if pct_b is None:
            return signals
        out = signals.copy()
        out.loc[(signals == 1) & (pct_b > 0.7)] = 0   # pas de long en haut des BB
        out.loc[(signals == -1) & (pct_b < 0.3)] = 0   # pas de short en bas des BB
        return out
    return _filter


def filter_volume_spike(min_ratio=2.0):
    """Ne trade que sur spike de volume (confirmation)."""
    def _filter(signals, df):
        vr = df.get("volume_ratio")
        if vr is None:
            return signals
        return signals.where(vr >= min_ratio, 0)
    return _filter


def main():
    t0 = time.time()

    print("=" * 120)
    print("  OPTIMISATIONS V2 — Filtres intelligents (pas du tuning)")
    print("=" * 120)

    print("\n-- Chargement --")
    data = {}
    for sym in ["BTC", "SOL", "ETH"]:
        data[sym] = load_asset(sym)
        print("  %s OK" % sym)

    bt = SweepBacktester()

    # ══════════════════════════════════════════════════════
    # BASELINE par stratégie
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("  BASELINES INDIVIDUELLES")
    print("=" * 120)

    for sc in BASELINE:
        r = run_single(data, bt, sc["asset"], sc["strat_class"], sc["params"], sc["exec_config"])
        summarize(r, "BASELINE %s" % sc["name"])

    # ══════════════════════════════════════════════════════
    # OPT 1 : Filtre horaire
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("  OPT 1 : FILTRE HORAIRE (heures UTC à éviter)")
    print("=" * 120)

    hour_sets = {
        "Éviter 0-3 UTC (Asie basse liq)": [0, 1, 2, 3],
        "Éviter 13-15 UTC (US open)": [13, 14, 15],
        "Éviter 0-3 + 13-15": [0, 1, 2, 3, 13, 14, 15],
        "Ne trader que 6-12 UTC (Europe)": list(range(0, 6)) + list(range(13, 24)),
        "Ne trader que 8-20 UTC": list(range(0, 8)) + list(range(21, 24)),
    }

    for sc in BASELINE:
        print("\n  -- %s --" % sc["name"])
        r_base = run_single(data, bt, sc["asset"], sc["strat_class"], sc["params"], sc["exec_config"])
        summarize(r_base, "BASELINE")

        for hname, hours in hour_sets.items():
            r = run_single(data, bt, sc["asset"], sc["strat_class"], sc["params"], sc["exec_config"],
                          signal_filter=filter_hours(hours))
            summarize(r, hname)

    # ══════════════════════════════════════════════════════
    # OPT 2 : Filtre jour de semaine
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("  OPT 2 : FILTRE JOUR DE SEMAINE")
    print("=" * 120)

    day_sets = {
        "Sans weekend (sam+dim)": [5, 6],
        "Sans lundi": [0],
        "Sans vendredi": [4],
        "Seulement mar-jeu": [0, 4, 5, 6],
    }

    for sc in BASELINE:
        print("\n  -- %s --" % sc["name"])
        r_base = run_single(data, bt, sc["asset"], sc["strat_class"], sc["params"], sc["exec_config"])
        summarize(r_base, "BASELINE")

        for dname, days in day_sets.items():
            r = run_single(data, bt, sc["asset"], sc["strat_class"], sc["params"], sc["exec_config"],
                          signal_filter=filter_weekdays(days))
            summarize(r, dname)

    # ══════════════════════════════════════════════════════
    # OPT 3 : Filtres qualitatifs
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("  OPT 3 : FILTRES QUALITATIFS (RSI, ADX, wick, volume, BB, regime)")
    print("=" * 120)

    quality_filters = {
        "Anti-wick 60%": filter_anti_wick(0.6),
        "Anti-wick 50%": filter_anti_wick(0.5),
        "RSI confirm": filter_rsi_confirm(),
        "ADX > 20": filter_adx_trending(20),
        "ADX > 25": filter_adx_trending(25),
        "Regime aligned": filter_regime_bull_only(),
        "BB position": filter_bb_position(),
        "Vol spike > 2.0": filter_volume_spike(2.0),
        "Vol spike > 3.0": filter_volume_spike(3.0),
    }

    for sc in BASELINE:
        print("\n  -- %s --" % sc["name"])
        r_base = run_single(data, bt, sc["asset"], sc["strat_class"], sc["params"], sc["exec_config"])
        summarize(r_base, "BASELINE")

        for fname, filt in quality_filters.items():
            r = run_single(data, bt, sc["asset"], sc["strat_class"], sc["params"], sc["exec_config"],
                          signal_filter=filt)
            summarize(r, fname)

    # ══════════════════════════════════════════════════════
    # OPT 4 : Combos de filtres
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("  OPT 4 : COMBOS DE FILTRES (meilleurs de chaque axe)")
    print("=" * 120)

    def chain_filters(*filters):
        def _filter(signals, df):
            s = signals
            for f in filters:
                s = f(s, df)
            return s
        return _filter

    combos = {
        "Anti-wick + RSI": chain_filters(filter_anti_wick(0.6), filter_rsi_confirm()),
        "Anti-wick + ADX>20": chain_filters(filter_anti_wick(0.6), filter_adx_trending(20)),
        "Regime + Anti-wick": chain_filters(filter_regime_bull_only(), filter_anti_wick(0.6)),
        "RSI + Vol>2": chain_filters(filter_rsi_confirm(), filter_volume_spike(2.0)),
        "Regime + RSI + Anti-wick": chain_filters(filter_regime_bull_only(), filter_rsi_confirm(), filter_anti_wick(0.6)),
        "8-20 UTC + Anti-wick": chain_filters(filter_hours(list(range(0,8))+list(range(21,24))), filter_anti_wick(0.6)),
        "Sans weekend + RSI": chain_filters(filter_weekdays([5,6]), filter_rsi_confirm()),
    }

    for sc in BASELINE:
        print("\n  -- %s --" % sc["name"])
        r_base = run_single(data, bt, sc["asset"], sc["strat_class"], sc["params"], sc["exec_config"])
        summarize(r_base, "BASELINE")

        for cname, filt in combos.items():
            r = run_single(data, bt, sc["asset"], sc["strat_class"], sc["params"], sc["exec_config"],
                          signal_filter=filt)
            summarize(r, cname)

    elapsed = time.time() - t0
    print("\n" + "=" * 120)
    print("  Temps total : %.1fs" % elapsed)
    print("=" * 120)


if __name__ == "__main__":
    main()
