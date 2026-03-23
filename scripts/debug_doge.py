"""Debug DOGE anomaly: +10141% vs expected +2217%.

Checks:
1. Timeframe: verify 1h resample is correct
2. Signal count: compare to other coins
3. Vol filter: how many signals filtered vs passed
4. Fee verification
5. Trade distribution over time
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "crypto_bot"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exec_config import ExecConfig
from modules.feature_engine import FeatureEngine
from modules.strategies import V2_STRATEGY_REGISTRY
from sweep_runner import SweepBacktester

from trading_bot.db import Database

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "crypto_bot", "config.yaml")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trading_bot.db")

V2_PARAMS = {
    "lookback": 32,
    "vol_breakout_min": 0.8,
    "sl_pct": 0.3,
    "tp_pct": 4.0,
}
EC = ExecConfig(
    equity_pct=0.35, leverage=5,
    cooldown_bars=3, max_hold_bars=48,
)


def load_candles_df(db, coin):
    rows = db.fetchall(
        "SELECT time_open, open, high, low, close, volume FROM candles "
        "WHERE coin=? AND interval='5m' ORDER BY time_open",
        (coin,),
    )
    if not rows:
        return None
    data = [
        {"time_open": r["time_open"], "open": r["open"], "high": r["high"],
         "low": r["low"], "close": r["close"], "volume": r["volume"]}
        for r in rows
    ]
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["time_open"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["time_open"])
    df = df[~df.index.duplicated(keep="first")]
    return df


def analyze_coin(db, coin, fe):
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC: {coin}")
    print(f"{'='*60}")

    # Load and resample
    df_5m = load_candles_df(db, coin)
    print(f"\n  [1] RAW DATA")
    print(f"      5m candles: {len(df_5m)}")
    print(f"      Range: {df_5m.index[0]} → {df_5m.index[-1]}")

    # Check 5m interval consistency
    time_diffs = df_5m.index.to_series().diff().dropna()
    median_diff = time_diffs.median()
    print(f"      Median interval: {median_diff}")
    print(f"      Min interval: {time_diffs.min()}")
    print(f"      Max interval: {time_diffs.max()}")

    df_1h = df_5m.resample("1h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["open"])

    print(f"\n  [2] 1H RESAMPLE")
    print(f"      1h candles: {len(df_1h)}")
    print(f"      Range: {df_1h.index[0]} → {df_1h.index[-1]}")

    # Check 1h interval consistency
    time_diffs_1h = df_1h.index.to_series().diff().dropna()
    median_1h = time_diffs_1h.median()
    print(f"      Median interval: {median_1h}")
    non_1h = (time_diffs_1h != pd.Timedelta(hours=1)).sum()
    print(f"      Non-1h gaps: {non_1h}")

    # Compute features
    df_1h = fe.compute_all(df_1h)

    # Generate signals WITHOUT vol filter first
    v2_cls = V2_STRATEGY_REGISTRY["StratBreakoutRelaxed"]

    # With vol filter
    strat = v2_cls(V2_PARAMS)
    signals = strat.generate_signals(df_1h)
    n_long = (signals == 1).sum()
    n_short = (signals == -1).sum()
    n_total = n_long + n_short

    # Without vol filter
    no_vol_params = dict(V2_PARAMS)
    no_vol_params["vol_breakout_min"] = 0.0
    strat_no_vol = v2_cls(no_vol_params)
    signals_no_vol = strat_no_vol.generate_signals(df_1h)
    n_total_no_vol = (signals_no_vol != 0).sum()

    print(f"\n  [3] SIGNALS")
    print(f"      With vol filter (>= 0.8): {n_total} ({n_long} long, {n_short} short)")
    print(f"      Without vol filter:       {n_total_no_vol}")
    print(f"      Vol filter rejection:     {n_total_no_vol - n_total} ({(1 - n_total/max(n_total_no_vol,1))*100:.1f}%)")

    # Signal frequency
    years = (df_1h.index[-1] - df_1h.index[0]).total_seconds() / (365.25 * 86400)
    print(f"      Signals per year:         {n_total / years:.1f}")
    print(f"      Span:                     {years:.2f} years")

    # Volume ratio analysis
    vol_ratio = df_1h.get("volume_ratio")
    if vol_ratio is not None:
        print(f"\n  [4] VOLUME RATIO DISTRIBUTION")
        print(f"      Mean:   {vol_ratio.mean():.3f}")
        print(f"      Median: {vol_ratio.median():.3f}")
        print(f"      Std:    {vol_ratio.std():.3f}")
        print(f"      % >= 0.8: {(vol_ratio >= 0.8).mean()*100:.1f}%")
        print(f"      % >= 1.0: {(vol_ratio >= 1.0).mean()*100:.1f}%")
        print(f"      % >= 2.0: {(vol_ratio >= 2.0).mean()*100:.1f}%")

    # Run backtest and check fee/trade details
    bt = SweepBacktester(CONFIG_PATH)
    metrics = bt.run(
        df_1h, signals,
        sl_pct=strat.sl_pct, tp_pct=strat.tp_pct,
        exec_config=EC, initial_equity=1000.0,
    )

    trades = metrics.get("trades_detail", [])

    print(f"\n  [5] BACKTEST RESULTS")
    print(f"      Return:       {metrics['total_return']*100:+.1f}%")
    print(f"      Trades:       {metrics['nb_trades']}")
    print(f"      Win rate:     {metrics['win_rate']*100:.1f}%")
    print(f"      PF:           {metrics['profit_factor']:.2f}")
    print(f"      Sharpe:       {metrics['sharpe_ratio']:.3f}")
    print(f"      MaxDD:        {metrics['max_drawdown']*100:.2f}%")
    print(f"      Total fees:   ${metrics.get('total_fees', 0):.4f}")
    print(f"      Total funding:${metrics.get('total_funding', 0):.4f}")

    if trades:
        # Exit reason breakdown
        exits = {}
        for t in trades:
            r = t.get("exit_reason", "?")
            exits[r] = exits.get(r, 0) + 1
        print(f"      Exit reasons: {exits}")

        # Fee per trade
        fees = [t.get("entry_fee", 0) + t.get("exit_fee", 0) for t in trades]
        print(f"      Avg fee/trade: ${np.mean(fees):.6f}")

        # PnL distribution
        pnls = [t["net_pnl"] for t in trades]
        print(f"      Avg PnL/trade: ${np.mean(pnls):.4f}")
        print(f"      Median PnL:    ${np.median(pnls):.4f}")
        print(f"      Max win:       ${max(pnls):.4f}")
        print(f"      Max loss:      ${min(pnls):.4f}")

        # Trade frequency by year
        print(f"\n  [6] TRADE DISTRIBUTION BY YEAR")
        for t in trades:
            t["_year"] = t.get("exit_time").year if hasattr(t.get("exit_time"), "year") else "?"
        years_seen = sorted(set(t["_year"] for t in trades if t["_year"] != "?"))
        for y in years_seen:
            yr_trades = [t for t in trades if t.get("_year") == y]
            yr_wins = [t for t in yr_trades if t["net_pnl"] > 0]
            yr_ret = sum(t["net_pnl"] for t in yr_trades)
            print(f"      {y}: {len(yr_trades)} trades, {len(yr_wins)} wins, PnL ${yr_ret:.2f}")

    return metrics


def main():
    db = Database(DB_PATH)
    db.open()

    try:
        fe = FeatureEngine(CONFIG_PATH)

        # Run DOGE and a control coin (BTC) for comparison
        analyze_coin(db, "DOGE", fe)
        analyze_coin(db, "BTC", fe)

    finally:
        db.close()


if __name__ == "__main__":
    main()
