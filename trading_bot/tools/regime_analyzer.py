import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from trading_bot.types import Candle
from trading_bot.strategy.indicators import compute_indicators
from trading_bot.backtest.monte_carlo import Xoshiro256ss
from trading_bot.db import Database

log = logging.getLogger(__name__)

HYSTERESIS = 3


@dataclass
class RegimeStats:
    regime: str
    n_bars: int = 0
    avg_return: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    n_trades: int = 0


def classify_regime(candles: list[Candle], idx: int) -> str:
    end = idx + 1
    start = max(0, end - 300)
    ind = compute_indicators(candles[start:end])

    if not ind.valid:
        return "NEUTRAL"

    if ind.sma_50 > ind.sma_200 and ind.ema_12 > ind.ema_26 and (ind.adx_14 > 20 or ind.di_bull):
        return "BULL"
    if ind.sma_50 < ind.sma_200 and ind.ema_12 < ind.ema_26 and (ind.adx_14 > 20 or ind.di_bear):
        return "BEAR"
    return "NEUTRAL"


def classify_all(candles: list[Candle]) -> list[str]:
    raw_regimes = [classify_regime(candles, i) for i in range(len(candles))]

    # Apply hysteresis
    result = [raw_regimes[0]] if raw_regimes else []
    pending = ""
    pending_count = 0

    for i in range(1, len(raw_regimes)):
        if raw_regimes[i] != result[-1]:
            if raw_regimes[i] == pending:
                pending_count += 1
            else:
                pending = raw_regimes[i]
                pending_count = 1

            if pending_count >= HYSTERESIS:
                result.append(pending)
                pending = ""
                pending_count = 0
            else:
                result.append(result[-1])
        else:
            result.append(raw_regimes[i])
            pending = ""
            pending_count = 0

    return result


def transition_matrix(regimes: list[str]) -> dict[str, dict[str, int]]:
    labels = ["BULL", "BEAR", "NEUTRAL"]
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for i in range(1, len(regimes)):
        matrix[regimes[i - 1]][regimes[i]] += 1
    return matrix


def walk_forward_analysis(
    candles: list[Candle], regimes: list[str],
    n_splits: int = 5, overlap: float = 0.6,
) -> list[dict]:
    n = len(candles)
    window_size = int(n * 0.4)
    step = int(window_size * (1 - overlap)) if overlap < 1 else window_size // n_splits

    results = []

    for i in range(n_splits):
        start = i * step
        end = min(start + window_size, n)
        if end - start < 100:
            break

        split_point = start + int((end - start) * 0.7)

        is_regimes = regimes[start:split_point]
        oos_regimes = regimes[split_point:end]
        is_candles = candles[start:split_point]
        oos_candles = candles[split_point:end]

        is_stats = _compute_regime_stats(is_candles, is_regimes)
        oos_stats = _compute_regime_stats(oos_candles, oos_regimes)

        results.append({
            "split": i + 1,
            "is_range": f"{start}-{split_point}",
            "oos_range": f"{split_point}-{end}",
            "is": is_stats,
            "oos": oos_stats,
        })

    return results


def _compute_regime_stats(candles: list[Candle], regimes: list[str]) -> dict[str, RegimeStats]:
    stats = {}
    for regime in ("BULL", "BEAR", "NEUTRAL"):
        indices = [i for i, r in enumerate(regimes) if r == regime]
        if not indices:
            stats[regime] = RegimeStats(regime=regime)
            continue

        returns = []
        for i in indices:
            if i > 0 and i < len(candles) and candles[i - 1].close > 0:
                ret = (candles[i].close - candles[i - 1].close) / candles[i - 1].close
                returns.append(ret)

        n = len(returns)
        avg = sum(returns) / n if n > 0 else 0
        wr = sum(1 for r in returns if r > 0) / n if n > 0 else 0

        var = sum((r - avg) ** 2 for r in returns) / n if n > 1 else 0
        std = var ** 0.5
        sharpe = (252 ** 0.5) * avg / std if std > 0 else 0

        stats[regime] = RegimeStats(
            regime=regime, n_bars=len(indices),
            avg_return=round(avg * 100, 4), win_rate=round(wr, 4),
            sharpe=round(sharpe, 2), n_trades=n,
        )

    return stats


def regime_monte_carlo(
    candles: list[Candle], regimes: list[str],
    n_simulations: int = 10000, seed: int = 42,
) -> dict:
    rng = Xoshiro256ss(seed)

    # Collect per-regime returns
    regime_returns: dict[str, list[float]] = {"BULL": [], "BEAR": [], "NEUTRAL": []}
    for i in range(1, len(candles)):
        if i < len(regimes) and candles[i - 1].close > 0:
            ret = (candles[i].close - candles[i - 1].close) / candles[i - 1].close
            regime_returns[regimes[i]].append(ret)

    # Transition probabilities
    trans = transition_matrix(regimes)
    labels = ["BULL", "BEAR", "NEUTRAL"]

    final_returns = []
    max_drawdowns = []

    n_steps = len(candles)

    for _ in range(n_simulations):
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        current_regime = regimes[-1] if regimes else "NEUTRAL"

        for _ in range(n_steps):
            # Transition
            row = trans.get(current_regime, {})
            total = sum(row.values())
            if total > 0:
                r = rng.rand_float() * total
                cumulative = 0
                for label in labels:
                    cumulative += row.get(label, 0)
                    if r <= cumulative:
                        current_regime = label
                        break

            # Sample return
            pool = regime_returns.get(current_regime, [])
            if pool:
                idx = rng.rand_int(len(pool))
                equity *= (1 + pool[idx])

            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        final_returns.append((equity - 1.0) * 100)
        max_drawdowns.append(max_dd * 100)

    final_returns.sort()
    max_drawdowns.sort()

    n = n_simulations
    ruin_count = sum(1 for r in final_returns if r < -50)

    return {
        "p_ruin_50pct": round(ruin_count / n, 4),
        "p95_drawdown": round(max_drawdowns[int(n * 0.95)], 2),
        "p99_drawdown": round(max_drawdowns[int(n * 0.99)], 2),
        "median_return": round(final_returns[n // 2], 2),
        "p5_return": round(final_returns[int(n * 0.05)], 2),
        "p95_return": round(final_returns[int(n * 0.95)], 2),
    }


def generate_report(
    coin: str, candles: list[Candle], regimes: list[str],
    wf_results: list[dict], mc_results: dict | None = None,
) -> str:
    lines = [f"# Regime Analysis: {coin}", ""]

    # Distribution
    n = len(regimes)
    bull = sum(1 for r in regimes if r == "BULL")
    bear = sum(1 for r in regimes if r == "BEAR")
    neutral = n - bull - bear

    lines.append(f"## Distribution ({n} bars)")
    lines.append(f"- BULL: {bull} ({bull/n*100:.1f}%)")
    lines.append(f"- BEAR: {bear} ({bear/n*100:.1f}%)")
    lines.append(f"- NEUTRAL: {neutral} ({neutral/n*100:.1f}%)")
    lines.append("")

    # Transition matrix
    trans = transition_matrix(regimes)
    lines.append("## Transition Matrix")
    lines.append("| From \\ To | BULL | BEAR | NEUTRAL |")
    lines.append("|-----------|------|------|---------|")
    for r in ("BULL", "BEAR", "NEUTRAL"):
        row = trans[r]
        total = sum(row.values()) or 1
        lines.append(f"| {r:9s} | {row['BULL']/total*100:4.1f}% | {row['BEAR']/total*100:4.1f}% | {row['NEUTRAL']/total*100:5.1f}% |")
    lines.append("")

    # Walk-forward
    lines.append("## Walk-Forward Results")
    for wf in wf_results:
        lines.append(f"\n### Split {wf['split']} (IS: {wf['is_range']}, OOS: {wf['oos_range']})")
        for period in ("is", "oos"):
            lines.append(f"\n**{period.upper()}:**")
            for regime in ("BULL", "BEAR", "NEUTRAL"):
                s = wf[period].get(regime, RegimeStats(regime=regime))
                lines.append(f"  - {regime}: bars={s.n_bars} WR={s.win_rate:.2%} Sharpe={s.sharpe:.2f}")

    # Monte Carlo
    if mc_results:
        lines.append("\n## Monte Carlo (10k simulations)")
        lines.append(f"- P(ruin 50%): {mc_results['p_ruin_50pct']:.2%}")
        lines.append(f"- P95 Max DD: {mc_results['p95_drawdown']:.1f}%")
        lines.append(f"- P99 Max DD: {mc_results['p99_drawdown']:.1f}%")
        lines.append(f"- Median Return: {mc_results['median_return']:.1f}%")
        lines.append(f"- P5/P95 Return: {mc_results['p5_return']:.1f}% / {mc_results['p95_return']:.1f}%")

    return "\n".join(lines)


def analyze(
    coin: str,
    n_candles: int = 2000,
    interval: str = "1h",
    db_path: str = "./data/trading_bot.db",
    validate: bool = False,
    montecarlo: bool = False,
) -> str:
    db = Database(db_path)
    db.open()

    try:
        rows = db.fetchall(
            "SELECT * FROM candles WHERE coin=? AND interval='5m' ORDER BY time_open",
            (coin,)
        )
        candles_5m = [
            Candle(
                time_open=r["time_open"], time_close=r["time_open"] + 300000,
                open=r["open"], high=r["high"], low=r["low"],
                close=r["close"], volume=r["volume"], n_trades=r["n_trades"] or 0,
            )
            for r in rows
        ]

        # Aggregate
        interval_ms = {"1h": 3600000, "4h": 14400000, "1d": 86400000}.get(interval, 3600000)
        bars_per_tf = interval_ms // 300000
        candles = []
        for i in range(0, len(candles_5m), bars_per_tf):
            chunk = candles_5m[i:i + bars_per_tf]
            if len(chunk) < bars_per_tf:
                break
            candles.append(Candle(
                time_open=chunk[0].time_open, time_close=chunk[-1].time_close,
                open=chunk[0].open, high=max(c.high for c in chunk),
                low=min(c.low for c in chunk), close=chunk[-1].close,
                volume=sum(c.volume for c in chunk),
            ))

        candles = candles[-n_candles:]
        log.info(f"Analyzing {len(candles)} candles for {coin}/{interval}")

        regimes = classify_all(candles)
        wf = walk_forward_analysis(candles, regimes)

        mc = None
        if montecarlo:
            mc = regime_monte_carlo(candles, regimes)

        report = generate_report(coin, candles, regimes, wf, mc)

        out_dir = Path("data/analysis")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{coin}_regime_report.md"
        with open(out_path, "w") as f:
            f.write(report)

        log.info(f"Report saved to {out_path}")
        return report

    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Regime analyzer")
    parser.add_argument("coin", help="Coin symbol")
    parser.add_argument("n_candles", type=int, nargs="?", default=2000)
    parser.add_argument("interval", nargs="?", default="1h")
    parser.add_argument("--db", default="./data/trading_bot.db")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--montecarlo", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    report = analyze(args.coin, args.n_candles, args.interval, args.db, args.validate, args.montecarlo)
    print(report)


if __name__ == "__main__":
    main()
