from dataclasses import dataclass


@dataclass
class MonteCarloResult:
    n_simulations: int = 10000
    p_ruin_50pct: float = 0.0
    p95_drawdown: float = 0.0
    p99_drawdown: float = 0.0
    median_return: float = 0.0
    p5_return: float = 0.0
    p95_return: float = 0.0
    final_equity_median: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_simulations": self.n_simulations,
            "p_ruin_50pct": round(self.p_ruin_50pct, 4),
            "p95_drawdown": round(self.p95_drawdown, 4),
            "p99_drawdown": round(self.p99_drawdown, 4),
            "median_return": round(self.median_return, 4),
            "p5_return": round(self.p5_return, 4),
            "p95_return": round(self.p95_return, 4),
            "final_equity_median": round(self.final_equity_median, 4),
        }


class Xoshiro256ss:
    def __init__(self, seed: int = 42):
        self.s = [0, 0, 0, 0]
        self.s[0] = seed
        self.s[1] = seed ^ 0xDEADBEEF
        self.s[2] = seed ^ 0xCAFEBABE
        self.s[3] = seed ^ 0x12345678
        for _ in range(20):
            self.next()

    def _rotl(self, x: int, k: int) -> int:
        x &= 0xFFFFFFFFFFFFFFFF
        return ((x << k) | (x >> (64 - k))) & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        s = self.s
        result = self._rotl((s[1] * 5) & 0xFFFFFFFFFFFFFFFF, 7)
        result = (result * 9) & 0xFFFFFFFFFFFFFFFF

        t = (s[1] << 17) & 0xFFFFFFFFFFFFFFFF
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = self._rotl(s[3], 45)

        return result

    def rand_int(self, n: int) -> int:
        return self.next() % n

    def rand_float(self) -> float:
        return self.next() / 0xFFFFFFFFFFFFFFFF


def run_monte_carlo(
    trade_pnls: list[float],
    initial_balance: float = 100.0,
    n_simulations: int = 10000,
    seed: int = 42,
) -> MonteCarloResult:
    if len(trade_pnls) < 5:
        return MonteCarloResult()

    rng = Xoshiro256ss(seed)
    n_trades = len(trade_pnls)

    final_equities = []
    max_drawdowns = []
    ruin_count = 0
    ruin_threshold = initial_balance * 0.5

    for _ in range(n_simulations):
        equity = initial_balance
        peak = equity
        max_dd = 0.0

        for _ in range(n_trades):
            idx = rng.rand_int(n_trades)
            equity += trade_pnls[idx]

            if equity > peak:
                peak = equity

            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        final_equities.append(equity)
        max_drawdowns.append(max_dd)

        if equity <= ruin_threshold:
            ruin_count += 1

    final_equities.sort()
    max_drawdowns.sort()

    result = MonteCarloResult(n_simulations=n_simulations)
    result.p_ruin_50pct = ruin_count / n_simulations

    p95_idx = int(n_simulations * 0.95)
    p99_idx = int(n_simulations * 0.99)
    p5_idx = int(n_simulations * 0.05)
    median_idx = n_simulations // 2

    result.p95_drawdown = max_drawdowns[p95_idx] * 100
    result.p99_drawdown = max_drawdowns[p99_idx] * 100

    result.median_return = (final_equities[median_idx] - initial_balance) / initial_balance * 100
    result.p5_return = (final_equities[p5_idx] - initial_balance) / initial_balance * 100
    result.p95_return = (final_equities[p95_idx] - initial_balance) / initial_balance * 100
    result.final_equity_median = final_equities[median_idx]

    return result
