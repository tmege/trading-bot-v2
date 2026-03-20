"""
Monte Carlo simulation module for backtest validation.

Provides:
  - Xoshiro256ss: Fast deterministic PRNG (Blackman & Vigna 2018)
  - MCResult: Immutable result dataclass
  - bootstrap_mc: Bootstrap resample of trade PnLs (same as SweepBacktester.monte_carlo)
  - sequence_mc: Permutation test on trade ordering
  - removal_mc: Random trade removal for fragility scoring
  - param_sensitivity_mc: SL/TP perturbation for overfitting detection
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Xoshiro256** PRNG (Blackman & Vigna 2018)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MASK64 = (1 << 64) - 1


def _rotl64(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & _MASK64


def _splitmix64(state: int) -> tuple[int, int]:
    """SplitMix64 — used to seed Xoshiro256** from a single integer."""
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z = (z ^ (z >> 31)) & _MASK64
    return state, z


class Xoshiro256ss:
    """Fast deterministic PRNG — Xoshiro256** (Blackman & Vigna 2018).

    Seeded via SplitMix64. State: 4 x uint64.
    Period: 2^256 - 1.
    """

    __slots__ = ("_s",)

    def __init__(self, seed: int = 42):
        sm = seed & _MASK64
        s = [0, 0, 0, 0]
        for i in range(4):
            sm, s[i] = _splitmix64(sm)
        self._s = s

    def _next(self) -> int:
        s = self._s
        result = (_rotl64((s[1] * 5) & _MASK64, 7) * 9) & _MASK64

        t = (s[1] << 17) & _MASK64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl64(s[3], 45)

        return result

    def rand_int(self, n: int) -> int:
        """Return uniform random int in [0, n)."""
        if n <= 0:
            return 0
        return self._next() % n

    def rand_float(self) -> float:
        """Return uniform random float in [0, 1)."""
        return (self._next() >> 11) / (1 << 53)

    def shuffle(self, arr: list) -> None:
        """Fisher-Yates in-place shuffle."""
        for i in range(len(arr) - 1, 0, -1):
            j = self.rand_int(i + 1)
            arr[i], arr[j] = arr[j], arr[i]

    def choice(self, arr: list | np.ndarray, size: int) -> list:
        """Sample *size* elements with replacement."""
        n = len(arr)
        if n == 0:
            return []
        return [arr[self.rand_int(n)] for _ in range(size)]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MCResult dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class MCResult:
    """Immutable result of a bootstrap Monte Carlo simulation."""

    n_simulations: int
    n_trades: int
    median_final_usd: float
    p5_final_usd: float
    p1_final_usd: float
    p95_final_usd: float
    median_return_pct: float
    p5_return_pct: float
    p1_return_pct: float
    median_maxdd_pct: float
    p95_maxdd_pct: float
    p99_maxdd_pct: float
    p_ruin_50pct: float
    p_ruin_count: int

    def to_dict(self) -> dict:
        """Convert to dict with same keys as SweepBacktester.monte_carlo()."""
        return {
            "n_simulations": self.n_simulations,
            "n_trades": self.n_trades,
            "median_final_$": self.median_final_usd,
            "p5_final_$": self.p5_final_usd,
            "p1_final_$": self.p1_final_usd,
            "p95_final_$": self.p95_final_usd,
            "median_return_%": self.median_return_pct,
            "p5_return_%": self.p5_return_pct,
            "p1_return_%": self.p1_return_pct,
            "median_maxdd_%": self.median_maxdd_pct,
            "p95_maxdd_%": self.p95_maxdd_pct,
            "p99_maxdd_%": self.p99_maxdd_pct,
            "p_ruin_50%": self.p_ruin_50pct,
            "p_ruin_count": self.p_ruin_count,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bootstrap Monte Carlo (extracted from SweepBacktester.monte_carlo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def bootstrap_mc(
    trade_pnls: list[float],
    initial_equity: float = 1000.0,
    n_sims: int = 10000,
    seed: int = 42,
) -> MCResult | None:
    """Bootstrap Monte Carlo on trade PnLs (resample WITH replacement).

    Same logic as SweepBacktester.monte_carlo() but uses Xoshiro256ss PRNG.

    Args:
        trade_pnls: Net PnL in $ per trade.
        initial_equity: Starting capital.
        n_sims: Number of simulations.
        seed: PRNG seed for reproducibility.

    Returns:
        MCResult or None if fewer than 5 trades.
    """
    if len(trade_pnls) < 5:
        return None

    rng = Xoshiro256ss(seed)
    n_trades = len(trade_pnls)
    pnls = np.array(trade_pnls, dtype=np.float64)

    final_equities = np.zeros(n_sims)
    max_drawdowns = np.zeros(n_sims)
    ruin_count = 0
    ruin_threshold = initial_equity * 0.5

    for s in range(n_sims):
        # Resample with replacement using Xoshiro256ss
        indices = [rng.rand_int(n_trades) for _ in range(n_trades)]
        shuffled = pnls[indices]
        equity_curve = initial_equity + np.cumsum(shuffled)
        equity_curve = np.insert(equity_curve, 0, initial_equity)

        peak = np.maximum.accumulate(equity_curve)
        dd = (peak - equity_curve) / np.where(peak > 0, peak, 1)

        final_equities[s] = equity_curve[-1]
        max_drawdowns[s] = dd.max()

        if equity_curve[-1] <= ruin_threshold:
            ruin_count += 1

    return MCResult(
        n_simulations=n_sims,
        n_trades=n_trades,
        median_final_usd=float(np.median(final_equities)),
        p5_final_usd=float(np.percentile(final_equities, 5)),
        p1_final_usd=float(np.percentile(final_equities, 1)),
        p95_final_usd=float(np.percentile(final_equities, 95)),
        median_return_pct=float((np.median(final_equities) - initial_equity) / initial_equity * 100),
        p5_return_pct=float((np.percentile(final_equities, 5) - initial_equity) / initial_equity * 100),
        p1_return_pct=float((np.percentile(final_equities, 1) - initial_equity) / initial_equity * 100),
        median_maxdd_pct=float(np.median(max_drawdowns) * 100),
        p95_maxdd_pct=float(np.percentile(max_drawdowns, 95) * 100),
        p99_maxdd_pct=float(np.percentile(max_drawdowns, 99) * 100),
        p_ruin_50pct=float(ruin_count / n_sims),
        p_ruin_count=ruin_count,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sequence Monte Carlo (permutation test on trade ordering)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sequence_mc(
    trade_pnls: list[float],
    initial_equity: float = 1000.0,
    n_sims: int = 5000,
    seed: int = 42,
) -> dict | None:
    """Permutation test: shuffle the EXACT order of trades (without replacement).

    Measures whether the real trade sequence produces significantly different
    results than random orderings of the same trades.

    Args:
        trade_pnls: Net PnL in $ per trade (in chronological order).
        initial_equity: Starting capital.
        n_sims: Number of shuffled simulations.
        seed: PRNG seed.

    Returns:
        dict with real_final, median_shuffled, p5/p95, order_matters, percentile_rank.
        None if fewer than 5 trades.
    """
    if len(trade_pnls) < 5:
        return None

    rng = Xoshiro256ss(seed)
    pnls = list(trade_pnls)
    n_trades = len(pnls)

    # Real equity curve
    real_equity = initial_equity
    for p in pnls:
        real_equity += p
    real_final = real_equity

    # Shuffled simulations (same trades, different order)
    shuffled_finals = np.zeros(n_sims)
    for s in range(n_sims):
        perm = list(pnls)
        rng.shuffle(perm)
        eq = initial_equity
        for p in perm:
            eq += p
        shuffled_finals[s] = eq

    # Since all permutations have the same sum, final equity is always the same
    # for additive PnL. The real insight is in the path (max drawdown, etc.)
    # So we measure max drawdown instead.
    real_dd = _max_drawdown_from_pnls(pnls, initial_equity)

    shuffled_dds = np.zeros(n_sims)
    rng2 = Xoshiro256ss(seed)  # Reset for reproducibility
    for s in range(n_sims):
        perm = list(pnls)
        rng2.shuffle(perm)
        shuffled_dds[s] = _max_drawdown_from_pnls(perm, initial_equity)

    # Percentile rank of real drawdown vs shuffled
    rank = float(np.sum(shuffled_dds <= real_dd) / n_sims * 100)

    return {
        "real_final": round(real_final, 2),
        "median_shuffled": round(float(np.median(shuffled_finals)), 2),
        "p5_final": round(float(np.percentile(shuffled_finals, 5)), 2),
        "p95_final": round(float(np.percentile(shuffled_finals, 95)), 2),
        "real_max_dd_pct": round(real_dd * 100, 2),
        "median_shuffled_dd_pct": round(float(np.median(shuffled_dds)) * 100, 2),
        "p5_dd_pct": round(float(np.percentile(shuffled_dds, 5)) * 100, 2),
        "p95_dd_pct": round(float(np.percentile(shuffled_dds, 95)) * 100, 2),
        "order_matters": bool(rank < 10 or rank > 90),
        "percentile_rank": round(rank, 1),
    }


def _max_drawdown_from_pnls(pnls: list[float], initial_equity: float) -> float:
    """Compute max drawdown ratio from a sequence of PnLs."""
    equity = initial_equity
    peak = equity
    max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Removal Monte Carlo (fragility scoring)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def removal_mc(
    trade_pnls: list[float],
    initial_equity: float = 1000.0,
    n_sims: int = 5000,
    removal_pcts: tuple[float, ...] = (0.10, 0.20, 0.30),
    seed: int = 42,
) -> dict | None:
    """Remove 10-30% of trades randomly and measure robustness.

    If the strategy's profitability depends on a few lucky trades, removing
    a small fraction will dramatically change the results → fragile.

    Args:
        trade_pnls: Net PnL in $ per trade.
        initial_equity: Starting capital.
        n_sims: Simulations per removal percentage.
        removal_pcts: Fractions of trades to remove.
        seed: PRNG seed.

    Returns:
        dict with per-pct results and overall fragility_score (0=robust, 1=fragile).
        None if fewer than 10 trades.
    """
    if len(trade_pnls) < 10:
        return None

    rng = Xoshiro256ss(seed)
    pnls = np.array(trade_pnls, dtype=np.float64)
    n_trades = len(pnls)

    # Real final equity
    real_final = initial_equity + float(pnls.sum())
    real_return = (real_final - initial_equity) / initial_equity * 100

    results_by_pct = {}
    fragility_scores = []

    for pct in removal_pcts:
        n_remove = max(1, int(n_trades * pct))
        n_keep = n_trades - n_remove
        finals = np.zeros(n_sims)

        for s in range(n_sims):
            # Select indices to keep (shuffle and take first n_keep)
            indices = list(range(n_trades))
            rng.shuffle(indices)
            kept = pnls[indices[:n_keep]]
            finals[s] = initial_equity + float(kept.sum())

        median_ret = (float(np.median(finals)) - initial_equity) / initial_equity * 100
        p5_ret = (float(np.percentile(finals, 5)) - initial_equity) / initial_equity * 100
        p95_ret = (float(np.percentile(finals, 95)) - initial_equity) / initial_equity * 100
        pct_profitable = float(np.sum(finals > initial_equity) / n_sims * 100)

        # Fragility: how much does removing trades change the result?
        if abs(real_return) > 0.01:
            frag = abs(real_return - median_ret) / abs(real_return)
        else:
            frag = 0.0
        frag = min(frag, 1.0)
        fragility_scores.append(frag)

        results_by_pct[f"remove_{int(pct*100)}pct"] = {
            "n_removed": n_remove,
            "median_return_pct": round(median_ret, 2),
            "p5_return_pct": round(p5_ret, 2),
            "p95_return_pct": round(p95_ret, 2),
            "pct_profitable": round(pct_profitable, 1),
            "fragility": round(frag, 3),
        }

    overall_fragility = float(np.mean(fragility_scores))

    return {
        "real_return_pct": round(real_return, 2),
        "results": results_by_pct,
        "fragility_score": round(overall_fragility, 3),
        "verdict": (
            "ROBUST" if overall_fragility < 0.3
            else "MODERATE" if overall_fragility < 0.6
            else "FRAGILE"
        ),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Parameter Sensitivity Monte Carlo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def param_sensitivity_mc(
    run_backtest_fn,
    base_sl_pct: float,
    base_tp_pct: float,
    perturbation: float = 0.10,
    n_samples: int = 20,
    seed: int = 42,
) -> dict | None:
    """Vary SL/TP by +/-perturbation, re-run backtest for each sample.

    Measures how sensitive the strategy is to small parameter changes.
    High sensitivity = likely overfitted.

    Args:
        run_backtest_fn: Callable(sl_pct, tp_pct) -> dict with 'total_return' and 'sharpe_ratio'.
        base_sl_pct: Base stop-loss percentage.
        base_tp_pct: Base take-profit percentage.
        perturbation: Fraction to vary (+/- this amount).
        n_samples: Number of parameter samples.
        seed: PRNG seed.

    Returns:
        dict with return_std, sharpe_std, sensitivity_score, or None on error.
    """
    rng = Xoshiro256ss(seed)

    returns = []
    sharpes = []
    params_tested = []

    for _ in range(n_samples):
        # Uniform perturbation in [-perturbation, +perturbation]
        sl_mult = 1.0 + (rng.rand_float() * 2 - 1) * perturbation
        tp_mult = 1.0 + (rng.rand_float() * 2 - 1) * perturbation
        sl = base_sl_pct * sl_mult
        tp = base_tp_pct * tp_mult

        try:
            metrics = run_backtest_fn(sl, tp)
            ret = metrics.get("total_return", 0) * 100
            sharpe = metrics.get("sharpe_ratio", 0)
            returns.append(ret)
            sharpes.append(sharpe)
            params_tested.append({"sl_pct": round(sl, 3), "tp_pct": round(tp, 3),
                                  "return_pct": round(ret, 2), "sharpe": round(sharpe, 3)})
        except Exception:
            continue

    if len(returns) < 3:
        return None

    returns_arr = np.array(returns)
    sharpes_arr = np.array(sharpes)

    return_std = float(np.std(returns_arr))
    sharpe_std = float(np.std(sharpes_arr))
    return_mean = float(np.mean(returns_arr))
    sharpe_mean = float(np.mean(sharpes_arr))

    # Sensitivity: coefficient of variation (higher = more sensitive)
    return_cv = return_std / abs(return_mean) if abs(return_mean) > 0.01 else return_std
    sharpe_cv = sharpe_std / abs(sharpe_mean) if abs(sharpe_mean) > 0.01 else sharpe_std
    sensitivity = (return_cv + sharpe_cv) / 2
    sensitivity = min(sensitivity, 1.0)

    return {
        "base_sl_pct": base_sl_pct,
        "base_tp_pct": base_tp_pct,
        "perturbation": perturbation,
        "n_samples": len(returns),
        "return_mean_pct": round(return_mean, 2),
        "return_std_pct": round(return_std, 2),
        "sharpe_mean": round(sharpe_mean, 3),
        "sharpe_std": round(sharpe_std, 3),
        "sensitivity_score": round(sensitivity, 3),
        "verdict": (
            "STABLE" if sensitivity < 0.2
            else "MODERATE" if sensitivity < 0.5
            else "SENSITIVE"
        ),
        "samples": params_tested,
    }
