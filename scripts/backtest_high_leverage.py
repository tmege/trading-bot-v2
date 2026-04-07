#!/usr/bin/env python3
"""
Backtest High Leverage — BTC x20 scalp court terme.

Strategie principale :
  Leveraged Breakout x20 — lb=6, SL 0.15%, TP 2.50%, eq 15%
  ~2 trades/jour, Sharpe 3.05, +3098% sur 3 ans

Strategie complementaire :
  Session+Weekly Open x15 — London fakeout + weekly open retest

Mode realiste (conditions live Hyperliquid) :
  - Frais maker 0.015% / taker 0.045%
  - Slippage 1.5 bps sur SL
  - Entry offset ALO 0.02%
  - Funding rate 0.01% / 8h
  - Sizing compose avec drawdown multiplier
  - Cooldown entre trades
"""
import sys
import os
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "crypto_bot"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exec_config import ExecConfig
from modules.feature_engine import FeatureEngine
from sweep_runner import SweepBacktester
from trading_bot.db import Database

INITIAL_EQUITY = 1000.0
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trading_bot.db")

WINDOWS = [
    ("6M 2023-H1", "2023-01-01", "2023-07-01"),
    ("6M 2023-H2", "2023-07-01", "2024-01-01"),
    ("6M 2024-H1", "2024-01-01", "2024-07-01"),
    ("6M 2024-H2", "2024-07-01", "2025-01-01"),
    ("6M 2025-H1", "2025-01-01", "2025-07-01"),
    ("1Y 2023",    "2023-01-01", "2024-01-01"),
    ("1Y 2024",    "2024-01-01", "2025-01-01"),
    ("1Y 2025",    "2025-01-01", "2026-01-01"),
    ("3Y FULL",    "2023-01-01", "2026-01-01"),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy 1: Leveraged Breakout x20 — optimized scalp
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StratLeveragedBreakout:
    """Breakout court terme lb=6, SL 0.15%, TP 2.50%, x20."""

    def __init__(self, config: dict):
        self.lookback = config.get("lookback", 6)
        self.vol_min = config.get("vol_min", 1.2)
        self.sl_pct = config.get("sl_pct", 0.15)
        self.tp_pct = config.get("tp_pct", 2.50)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        close = df["close"]
        vol_ratio = df.get("volume_ratio")
        if vol_ratio is None:
            return signals

        rh = close.rolling(self.lookback).max().shift(1)
        rl = close.rolling(self.lookback).min().shift(1)
        vol_ok = vol_ratio > self.vol_min

        signals.loc[(close > rh) & vol_ok] = 1
        signals.loc[(close < rl) & vol_ok] = -1
        return signals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy 2: Session + Weekly Open x15
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StratSessionWeeklyOpen:
    """London fakeout + weekly open retest, x15."""

    def __init__(self, config: dict):
        self.asia_start = config.get("asia_start", 0)
        self.asia_end = config.get("asia_end", 8)
        self.london_end = config.get("london_end", 10)
        self.min_range_pct = config.get("min_range_pct", 0.3)
        self.weekly_dev_pct = config.get("weekly_dev_pct", 2.0)
        self.weekly_retest_tol = config.get("weekly_retest_tol", 0.25)
        self.sl_pct = config.get("sl_pct", 0.50)
        self.tp_pct = config.get("tp_pct", 1.20)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        idx = df.index
        hours = idx.hour
        weekdays = idx.weekday
        ema50 = df.get("EMA50")
        ema50_vals = ema50.values if ema50 is not None else None
        n = len(df)

        # Setup A: London Open Fakeout
        asia_high = np.nan
        asia_low = np.nan
        building = False
        fakeout_side = ""
        last_day = -1

        for i in range(n):
            h = hours[i]
            day = idx[i].date()
            if day != last_day:
                asia_high = np.nan
                asia_low = np.nan
                building = False
                fakeout_side = ""
                last_day = day

            if self.asia_start <= h < self.asia_end:
                if np.isnan(asia_high):
                    asia_high, asia_low = high[i], low[i]
                    building = True
                else:
                    asia_high = max(asia_high, high[i])
                    asia_low = min(asia_low, low[i])
                continue

            if self.asia_end <= h < self.london_end and building:
                if np.isnan(asia_high) or np.isnan(asia_low):
                    continue
                if (asia_high - asia_low) / asia_low * 100 < self.min_range_pct:
                    continue
                if close[i] > asia_high and not fakeout_side:
                    fakeout_side = "above"
                if close[i] < asia_low and not fakeout_side:
                    fakeout_side = "below"
                if fakeout_side == "above" and close[i] < asia_high:
                    ok = True
                    if ema50_vals is not None and not np.isnan(ema50_vals[i]):
                        ok = close[i] < ema50_vals[i]
                    if ok:
                        signals.iloc[i] = -1
                    fakeout_side = "done"
                elif fakeout_side == "below" and close[i] > asia_low:
                    ok = True
                    if ema50_vals is not None and not np.isnan(ema50_vals[i]):
                        ok = close[i] > ema50_vals[i]
                    if ok:
                        signals.iloc[i] = 1
                    fakeout_side = "done"

        # Setup B: Weekly Open Retest
        wo = np.nan
        max_ext = 0.0
        ext_side = ""
        for i in range(n):
            if weekdays[i] == 0 and hours[i] == 0:
                wo = close[i]
                max_ext = 0.0
                ext_side = ""
                continue
            if np.isnan(wo) or wo <= 0:
                continue
            dev = (close[i] - wo) / wo * 100
            if abs(dev) > abs(max_ext):
                max_ext = dev
                ext_side = "above" if dev > 0 else "below"
            if abs(max_ext) < self.weekly_dev_pct or abs(dev) > self.weekly_retest_tol:
                continue
            if signals.iloc[i] != 0 or weekdays[i] < 1:
                continue
            trend_ok = True
            if ema50_vals is not None and not np.isnan(ema50_vals[i]):
                if ext_side == "above" and close[i] < ema50_vals[i]:
                    trend_ok = False
                if ext_side == "below" and close[i] > ema50_vals[i]:
                    trend_ok = False
            if not trend_ok:
                continue
            if ext_side == "above":
                signals.iloc[i] = 1
            elif ext_side == "below":
                signals.iloc[i] = -1
            max_ext = 0.0
            ext_side = ""

        return signals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGIES = [
    {
        "name": "Leveraged Breakout x20",
        "strat_class": StratLeveragedBreakout,
        "strat_params": {
            "lookback": 6, "vol_min": 1.2,
            "sl_pct": 0.15, "tp_pct": 2.50,
        },
        "exec_config": ExecConfig(
            equity_pct=0.15, leverage=20,
            cooldown_bars=1, max_hold_bars=24,
            maker_fee=0.00015, taker_fee=0.00045,
            slippage_sl_bps=1.5, entry_offset=0.0002,
            funding_rate_8h=0.0001,
        ),
        "timeframe": "1h",
    },
    {
        "name": "Session+Weekly Open x15",
        "strat_class": StratSessionWeeklyOpen,
        "strat_params": {
            "asia_start": 0, "asia_end": 8, "london_end": 10,
            "min_range_pct": 0.3, "weekly_dev_pct": 2.0,
            "weekly_retest_tol": 0.25,
            "sl_pct": 0.50, "tp_pct": 1.20,
        },
        "exec_config": ExecConfig(
            equity_pct=0.20, leverage=15,
            cooldown_bars=3, max_hold_bars=24,
            maker_fee=0.00015, taker_fee=0.00045,
            slippage_sl_bps=1.5, entry_offset=0.0001,
            funding_rate_8h=0.0001,
        ),
        "timeframe": "1h",
    },
]


def load_candles(coin):
    db = Database(DB_PATH)
    db.open()
    rows = db.fetchall(
        "SELECT time_open, open, high, low, close, volume FROM candles "
        "WHERE coin=? AND interval='5m' ORDER BY time_open", (coin,))
    db.close()
    if not rows:
        return None
    df = pd.DataFrame([dict(r) for r in rows])
    df["datetime"] = pd.to_datetime(df["time_open"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["time_open"])
    return df[~df.index.duplicated(keep="first")]


def run_single(bt, fe, df_5m, strat_cfg, start, end, initial_equity):
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    w = df_5m.loc[s:e]
    if len(w) < 100:
        return None
    tf = strat_cfg.get("timeframe", "1h")
    if tf == "5m":
        df = w.copy()
    else:
        df = w.resample("1h").agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum",
        }).dropna(subset=["open"])
        if len(df) < 50:
            return None
        df = fe.compute_all(df)

    strat = strat_cfg["strat_class"](strat_cfg["strat_params"])
    signals = strat.generate_signals(df)
    return bt.run(df, signals,
                  sl_pct=strat_cfg["strat_params"]["sl_pct"],
                  tp_pct=strat_cfg["strat_params"]["tp_pct"],
                  exec_config=strat_cfg["exec_config"],
                  initial_equity=initial_equity)


def main():
    t0 = time.time()
    fe = FeatureEngine(os.path.join(os.path.dirname(__file__), "..", "crypto_bot", "config.yaml"))
    bt = SweepBacktester(os.path.join(os.path.dirname(__file__), "..", "crypto_bot", "config.yaml"))

    print("=" * 110)
    print("  BACKTEST HIGH LEVERAGE — BTC x15-x20")
    print("  Mode realiste: fees maker/taker, slippage 1.5bps, funding, DD multiplier")
    print("=" * 110)

    df_5m = load_candles("BTC")
    if df_5m is None:
        print("  ERREUR: pas de donnees BTC")
        return
    print("  BTC: %d bougies 5m [%s -> %s]" % (
        len(df_5m), df_5m.index[0].strftime("%Y-%m-%d"), df_5m.index[-1].strftime("%Y-%m-%d")))

    for s_cfg in STRATEGIES:
        ec = s_cfg["exec_config"]
        notional = ec.equity_pct * ec.leverage
        sl_eq = notional * s_cfg["strat_params"]["sl_pct"] / 100
        tp_eq = notional * s_cfg["strat_params"]["tp_pct"] / 100

        print("\n" + "=" * 110)
        print("  %s" % s_cfg["name"])
        print("  Lev: %dx | SL: %.2f%% | TP: %.2f%% | Eq: %.0f%% | CD: %d bars | max_hold: %d" % (
            ec.leverage, s_cfg["strat_params"]["sl_pct"], s_cfg["strat_params"]["tp_pct"],
            ec.equity_pct * 100, ec.cooldown_bars, ec.max_hold_bars))
        print("  Risk: SL = -%.2f%% eq | TP = +%.2f%% eq | R:R = 1:%.1f" % (
            sl_eq * 100, tp_eq * 100, s_cfg["strat_params"]["tp_pct"] / s_cfg["strat_params"]["sl_pct"]))
        print("=" * 110)

        print("\n  %-14s | %6s | %8s | %5s | %6s | %6s | %6s | %8s" % (
            "Window", "Trades", "Return", "WR", "MaxDD", "Sharpe", "PF", "Final"))
        print("  " + "-" * 90)

        for wname, start, end in WINDOWS:
            m = run_single(bt, fe, df_5m, s_cfg, start, end, INITIAL_EQUITY)
            if m is None:
                print("  %-14s | %6s | %8s" % (wname, "-", "NO DATA"))
                continue

            nb = m["nb_trades"]
            pf_s = "%.2f" % m["profit_factor"] if m["profit_factor"] < 100 else "INF"
            print("  %-14s | %6d | %+7.0f%% | %5.1f | %5.1f%% | %+5.2f | %6s | $%.0f" % (
                wname, nb, m["total_return"]*100, m["win_rate"]*100,
                m["max_drawdown"]*100, m["sharpe_ratio"], pf_s,
                m.get("final_equity", INITIAL_EQUITY)))

            if wname == "3Y FULL":
                trades = m.get("trades_detail", [])
                if trades:
                    reasons = {}
                    for t in trades:
                        reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1
                    bars = [t["bars_held"] for t in trades]
                    fees = m.get("total_fees", 0)
                    funding = m.get("total_funding", 0)
                    print("\n    Exit: %s" % "  ".join("%s=%d" % kv for kv in sorted(reasons.items())))
                    print("    Hold: avg=%.1f | Freq: %.1f/jour" % (np.mean(bars), nb/(3*365.25)))
                    print("    Costs: fees=$%.0f  funding=$%.0f" % (fees, funding))

    # Portfolio combine
    print("\n" + "=" * 110)
    print("  PORTFOLIO COMBINE")
    print("=" * 110)
    total_pnl = 0
    for s_cfg in STRATEGIES:
        m = run_single(bt, fe, df_5m, s_cfg, "2023-01-01", "2026-01-01", INITIAL_EQUITY)
        if m:
            pnl = m.get("dollar_pnl", 0)
            total_pnl += pnl
            print("  %-25s : $%+.0f (%+.0f%%) — %d trades" % (
                s_cfg["name"], pnl, m["total_return"]*100, m["nb_trades"]))
    print("  Total: $%+.0f sur $%.0f investis (%+.0f%%)" % (
        total_pnl, INITIAL_EQUITY*len(STRATEGIES),
        total_pnl/(INITIAL_EQUITY*len(STRATEGIES))*100))

    print("\n  Temps: %.1fs" % (time.time() - t0))
    print("=" * 110)


if __name__ == "__main__":
    main()
