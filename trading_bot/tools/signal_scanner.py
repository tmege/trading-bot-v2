import argparse
import itertools
import logging
import sys
from dataclasses import dataclass

from trading_bot.types import Candle
from trading_bot.strategy.indicators import Indicators, compute_indicators
from trading_bot.db import Database

log = logging.getLogger(__name__)

TP_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
SL_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
MAKER_FEE = 0.00015
TAKER_FEE = 0.00045
MIN_OCCURRENCES = 20
TOP_N = 500


def _signal_functions() -> dict[str, callable]:
    return {
        "rsi_gt65": lambda i: i.rsi_14 > 65,
        "rsi_gt70": lambda i: i.rsi_14 > 70,
        "rsi_lt35": lambda i: i.rsi_14 < 35,
        "rsi_lt30": lambda i: i.rsi_14 < 30,
        "rsi_40_60": lambda i: 40 < i.rsi_14 < 60,
        "low_vol": lambda i: i.atr_pct < 0.005 if i.atr_pct > 0 else False,
        "very_low_vol": lambda i: i.atr_pct < 0.003 if i.atr_pct > 0 else False,
        "high_vol": lambda i: i.atr_pct > 0.01 if i.atr_pct > 0 else False,
        "atr_p80": lambda i: i.atr_pct_rank > 0.8,
        "atr_p20": lambda i: i.atr_pct_rank < 0.2,
        "adx_gt25": lambda i: i.adx_14 > 25,
        "adx_lt20": lambda i: i.adx_14 < 20,
        "adx_gt30": lambda i: i.adx_14 > 30,
        "di_bull": lambda i: i.di_bull,
        "di_bear": lambda i: i.di_bear,
        "macd_gt0": lambda i: i.macd_line > 0,
        "macd_lt0": lambda i: i.macd_line < 0,
        "macd_accel": lambda i: i.macd_hist_incr,
        "macd_decel": lambda i: i.macd_hist_decr,
        "macd_bull_cross": lambda i: i.macd_bullish_cross,
        "bb_squeeze": lambda i: i.bb_squeeze,
        "above_bb_upper": lambda i: i.n_candles > 0 and i.bb_upper > 0 and i.sma_20 > i.bb_upper * 0.999,
        "below_bb_lower": lambda i: i.n_candles > 0 and i.bb_lower > 0 and i.sma_20 < i.bb_lower * 1.001,
        "below_bb_mid": lambda i: i.sma_20 < i.bb_middle if i.bb_middle > 0 else False,
        "above_sma200": lambda i: i.above_sma_200,
        "below_sma200": lambda i: not i.above_sma_200 and i.sma_200 > 0,
        "golden_cross": lambda i: i.golden_cross,
        "death_cross": lambda i: i.sma_200 > 0 and i.sma_50 < i.sma_200,
        "sma20_far_above": lambda i: i.sma20_dist_pct > 2.0,
        "sma20_far_below": lambda i: i.sma20_dist_pct < -2.0,
        "ema12_far_above": lambda i: i.ema12_dist_pct > 2.0,
        "ema12_far_below": lambda i: i.ema12_dist_pct < -2.0,
        "ema_up": lambda i: i.ema_12 > i.ema_26 if i.ema_26 > 0 else False,
        "ema_down": lambda i: i.ema_12 < i.ema_26 if i.ema_26 > 0 else False,
        "vol_spike_2x": lambda i: i.vol_ratio > 2.0,
        "vol_spike_3x": lambda i: i.vol_ratio > 3.0,
        "low_volume": lambda i: i.vol_ratio < 0.5 if i.vol_ratio > 0 else False,
        "obv_above_sma": lambda i: i.obv > i.obv_sma if i.obv_sma != 0 else False,
        "obv_below_sma": lambda i: i.obv < i.obv_sma if i.obv_sma != 0 else False,
        "mfi_gt70": lambda i: i.mfi_14 > 70,
        "mfi_lt20": lambda i: i.mfi_14 < 20,
        "cmf_gt05": lambda i: i.cmf_20 > 0.05,
        "cmf_lt_neg05": lambda i: i.cmf_20 < -0.05,
        "cci_gt100": lambda i: i.cci_20 > 100,
        "cci_lt_neg100": lambda i: i.cci_20 < -100,
        "bullish_engulf": lambda i: i.bullish_engulf,
        "bearish_engulf": lambda i: i.bearish_engulf,
        "hammer": lambda i: i.hammer,
        "shooting_star": lambda i: i.shooting_star,
        "doji": lambda i: i.doji,
        "ichi_bullish": lambda i: i.ichi_bullish,
        "ichi_bearish": lambda i: not i.ichi_bullish and i.ichi_tenkan > 0,
        "rsi_bull_div": lambda i: i.rsi_bull_div,
        "rsi_bear_div": lambda i: i.rsi_bear_div,
        "squeeze_on": lambda i: i.squeeze_on,
        "squeeze_off": lambda i: not i.squeeze_on and i.bb_width > 0,
        # Funding signals (3 additional → 68 total)
        "funding_extreme_long": lambda i: i.funding_extreme_long,
        "funding_extreme_short": lambda i: i.funding_extreme_short,
        "funding_neutral": lambda i: abs(i.funding_rate) < 0.00005,
    }


@dataclass
class ScanResult:
    signals: str
    side: str
    tp: float
    sl: float
    is_wr: float
    is_pf: float
    is_sharpe: float
    is_ev: float
    oos_wr: float
    oos_pf: float
    oos_sharpe: float
    oos_ev: float
    n_trades_is: int
    n_trades_oos: int


def scan(
    coin: str,
    db_path: str = "./data/trading_bot.db",
    max_combo: int = 2,
    interval: str = "1h",
    output_path: str | None = None,
) -> list[ScanResult]:
    db = Database(db_path)
    db.open()

    try:
        candles = _load_candles(db, coin, interval)
        if len(candles) < 300:
            log.error(f"Need 300+ candles, got {len(candles)}")
            return []

        log.info(f"Loaded {len(candles)} candles for {coin}/{interval}")

        split = int(len(candles) * 0.7)
        is_candles = candles[:split]
        oos_candles = candles[split:]

        log.info(f"IS: {len(is_candles)}, OOS: {len(oos_candles)}")

        # Pre-compute indicators for all candles
        signals = _signal_functions()
        signal_names = sorted(signals.keys())

        is_indicators = _compute_all_indicators(is_candles)
        oos_indicators = _compute_all_indicators(oos_candles)

        results: list[ScanResult] = []

        for combo_size in range(1, max_combo + 1):
            for combo in itertools.combinations(signal_names, combo_size):
                for side in ("buy", "sell"):
                    is_entries = _find_entries(is_candles, is_indicators, signals, combo, side)
                    if len(is_entries) < MIN_OCCURRENCES:
                        continue

                    oos_entries = _find_entries(oos_candles, oos_indicators, signals, combo, side)

                    for tp in TP_GRID:
                        for sl in SL_GRID:
                            is_stats = _simulate(is_candles, is_entries, side, tp / 100, sl / 100)
                            oos_stats = _simulate(oos_candles, oos_entries, side, tp / 100, sl / 100)

                            if oos_stats["sharpe"] > 0:
                                results.append(ScanResult(
                                    signals="+".join(combo),
                                    side=side,
                                    tp=tp,
                                    sl=sl,
                                    is_wr=is_stats["wr"],
                                    is_pf=is_stats["pf"],
                                    is_sharpe=is_stats["sharpe"],
                                    is_ev=is_stats["ev"],
                                    oos_wr=oos_stats["wr"],
                                    oos_pf=oos_stats["pf"],
                                    oos_sharpe=oos_stats["sharpe"],
                                    oos_ev=oos_stats["ev"],
                                    n_trades_is=is_stats["n"],
                                    n_trades_oos=oos_stats["n"],
                                ))

        results.sort(key=lambda r: r.oos_sharpe, reverse=True)
        results = results[:TOP_N]

        if output_path:
            _write_tsv(results, output_path)

        return results

    finally:
        db.close()


def _load_candles(db: Database, coin: str, interval: str) -> list[Candle]:
    # Load 5m and aggregate
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

    interval_ms = {"5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000}
    bars_per_tf = interval_ms.get(interval, 3600000) // 300000

    if bars_per_tf <= 1:
        return candles_5m

    aggregated = []
    for i in range(0, len(candles_5m), bars_per_tf):
        chunk = candles_5m[i:i + bars_per_tf]
        if len(chunk) < bars_per_tf:
            break
        aggregated.append(Candle(
            time_open=chunk[0].time_open,
            time_close=chunk[-1].time_close,
            open=chunk[0].open,
            high=max(c.high for c in chunk),
            low=min(c.low for c in chunk),
            close=chunk[-1].close,
            volume=sum(c.volume for c in chunk),
            n_trades=sum(c.n_trades for c in chunk),
        ))

    return aggregated


def _compute_all_indicators(candles: list[Candle]) -> list[Indicators]:
    result = []
    for i in range(len(candles)):
        end = i + 1
        start = max(0, end - 300)
        ind = compute_indicators(candles[start:end])
        result.append(ind)
    return result


def _find_entries(
    candles: list[Candle], indicators: list[Indicators],
    signals: dict, combo: tuple, side: str,
) -> list[int]:
    entries = []
    for i in range(200, len(candles)):
        ind = indicators[i]
        if not ind.valid:
            continue
        if all(signals[s](ind) for s in combo):
            entries.append(i)
    return entries


def _simulate(
    candles: list[Candle], entries: list[int],
    side: str, tp_pct: float, sl_pct: float,
) -> dict:
    pnls = []
    in_trade = False
    entry_price = 0.0
    entry_idx = 0

    for idx in entries:
        if in_trade:
            continue
        if idx >= len(candles):
            break

        entry_price = candles[idx].close
        in_trade = True
        entry_idx = idx

        for j in range(idx + 1, min(idx + 200, len(candles))):
            c = candles[j]

            if side == "buy":
                if c.high >= entry_price * (1 + tp_pct):
                    pnl = tp_pct - MAKER_FEE - TAKER_FEE
                    pnls.append(pnl)
                    in_trade = False
                    break
                if c.low <= entry_price * (1 - sl_pct):
                    pnl = -sl_pct - MAKER_FEE - TAKER_FEE
                    pnls.append(pnl)
                    in_trade = False
                    break
            else:
                if c.low <= entry_price * (1 - tp_pct):
                    pnl = tp_pct - MAKER_FEE - TAKER_FEE
                    pnls.append(pnl)
                    in_trade = False
                    break
                if c.high >= entry_price * (1 + sl_pct):
                    pnl = -sl_pct - MAKER_FEE - TAKER_FEE
                    pnls.append(pnl)
                    in_trade = False
                    break

        if in_trade:
            last = candles[min(entry_idx + 199, len(candles) - 1)]
            if side == "buy":
                pnl = (last.close - entry_price) / entry_price - MAKER_FEE - TAKER_FEE
            else:
                pnl = (entry_price - last.close) / entry_price - MAKER_FEE - TAKER_FEE
            pnls.append(pnl)
            in_trade = False

    if not pnls:
        return {"n": 0, "wr": 0, "pf": 0, "sharpe": 0, "ev": 0}

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    n = len(pnls)
    wr = len(wins) / n
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf') if gross_win > 0 else 0
    ev = sum(pnls) / n

    mean = ev
    var = sum((p - mean) ** 2 for p in pnls) / n if n > 1 else 0
    std = var ** 0.5
    sharpe = (252 ** 0.5) * mean / std if std > 0 else 0

    return {"n": n, "wr": round(wr, 4), "pf": round(pf, 4), "sharpe": round(sharpe, 4), "ev": round(ev, 6)}


def _write_tsv(results: list[ScanResult], path: str):
    from pathlib import Path
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    with open(resolved, "w") as f:
        header = "signals\tside\ttp\tsl\tis_wr\tis_pf\tis_sharpe\tis_ev\toos_wr\toos_pf\toos_sharpe\toos_ev\tn_is\tn_oos"
        f.write(header + "\n")
        for r in results:
            line = f"{r.signals}\t{r.side}\t{r.tp}\t{r.sl}\t{r.is_wr}\t{r.is_pf}\t{r.is_sharpe}\t{r.is_ev}\t{r.oos_wr}\t{r.oos_pf}\t{r.oos_sharpe}\t{r.oos_ev}\t{r.n_trades_is}\t{r.n_trades_oos}"
            f.write(line + "\n")

    log.info(f"Results written to {resolved}")


def main():
    parser = argparse.ArgumentParser(description="Signal scanner")
    parser.add_argument("coin", help="Coin symbol")
    parser.add_argument("--db", default="./data/trading_bot.db")
    parser.add_argument("--max-combo", type=int, default=2)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    results = scan(args.coin, args.db, args.max_combo, args.interval, args.output)
    log.info(f"Found {len(results)} combos with OOS Sharpe > 0")

    if results:
        print(f"\nTop 10 by OOS Sharpe:")
        for r in results[:10]:
            print(f"  {r.signals:40s} {r.side:4s} TP={r.tp}% SL={r.sl}% "
                  f"OOS: WR={r.oos_wr:.2%} PF={r.oos_pf:.2f} Sharpe={r.oos_sharpe:.2f}")


if __name__ == "__main__":
    main()
