"""
Générateur de grille exhaustif pour sweep de stratégies V2.
Produit toutes les combinaisons (stratégie × params × asset × timeframe).
"""
from __future__ import annotations

import itertools

from modules.strategies import V2_STRATEGY_REGISTRY, PARAM_GRID

# ── Grille exhaustive (~12 000 combinaisons) ──────────────────

FULL_GRID = {
    "StratMomentumScore": {
        "threshold_low":  [1, 2],
        "threshold_high": [3, 4],
        "sl_pct":         [1.0, 1.5, 2.0, 2.5],
        "tp_pct":         [2.0, 3.0, 4.0, 5.0, 6.0],
    },
    "StratEmaCrossover": {
        "ema_fast":          [5, 9, 12, 21],
        "ema_slow":          [21, 50, 100],
        "use_regime_filter": [True, False],
        "sl_buffer_pct":     [0.3, 0.5, 1.0],
        "tp_pct":            [3.0, 5.0, 8.0, 10.0],
    },
    "StratBreakoutRelaxed": {
        "lookback":         [5, 10, 15, 20],
        "vol_breakout_min": [1.2, 1.5, 2.0, 2.5, 3.0],
        "use_compression":  [True, False],
        "sl_pct":           [1.0, 1.5, 2.0],
        "tp_pct":           [3.0, 4.0, 5.0, 6.0, 8.0],
    },
    # ── 4 nouvelles stratégies V2 (ETH mean-reversion & adaptive) ──
    "StratMeanReversionBB": {
        "rsi_oversold":   [25, 30, 35],
        "rsi_overbought": [65, 70, 75],
        "bb_entry_low":   [0.05, 0.10],
        "bb_entry_high":  [0.90, 0.95],
        "sl_pct":         [1.5, 2.5],
        "tp_pct":         [3.0, 4.0, 6.0],
    },
    "StratStochReversal": {
        "oversold":   [15, 20, 25, 30],
        "overbought": [70, 75, 80, 85],
        "vol_min":    [0.8, 1.0, 1.2],
        "sl_pct":     [1.5, 2.5],
        "tp_pct":     [3.0, 4.0, 6.0],
    },
    "StratInsideBarBreakout": {
        "vol_min":       [0.8, 1.0, 1.5, 2.0],
        "trend_filter":  [True, False],
        "atr_filter":    [True, False],
        "sl_pct":        [1.5, 2.0, 2.5],
        "tp_pct":        [3.0, 5.0, 6.0, 8.0],
    },
    "StratRegimeAdaptive": {
        "rsi_bull":         [50, 55],
        "rsi_range_low":    [25, 30, 35],
        "rsi_range_high":   [65, 70, 75],
        "use_ranging_only": [True, False],
        "sl_pct":           [1.5, 2.5],
        "tp_pct":           [3.0, 4.0, 6.0],
    },
}


def expand_grid(param_dict: dict) -> list[dict]:
    """Toutes les combinaisons d'un dict de listes."""
    keys = list(param_dict.keys())
    values = list(param_dict.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def build_all_combinations(
    assets: list[str],
    timeframes: list[str],
    grid: dict | None = None,
) -> list[dict]:
    """Liste complète des jobs : strat × params × asset × tf."""
    if grid is None:
        grid = FULL_GRID

    jobs = []
    for strat_name, param_dict in grid.items():
        combos = expand_grid(param_dict) if isinstance(param_dict, dict) else param_dict
        for params in combos:
            for asset in assets:
                for tf in timeframes:
                    jobs.append({
                        "strat_name": strat_name,
                        "params":     params,
                        "asset":      asset,
                        "timeframe":  tf,
                    })
    return jobs


def count_combinations(grid: dict | None = None) -> int:
    """Affiche le décompte avant lancement."""
    if grid is None:
        grid = FULL_GRID

    total = 0
    for strat_name, param_dict in grid.items():
        if isinstance(param_dict, dict):
            n = 1
            for v in param_dict.values():
                n *= len(v)
        else:
            n = len(param_dict)
        print(f"  {strat_name:<30} {n:>6} variantes")
        total += n
    print(f"  {'TOTAL':<30} {total:>6} variantes de params")
    return total
