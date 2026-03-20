import logging

from trading_bot.strategies.template import TemplateStrategy


class EthBreakoutRelaxed1h(TemplateStrategy):
    """ETH Breakout Relaxed — Lookback 35, SL 1.8%, TP 3.5%.

    Backtest: Sharpe +2.45 (3Y), stable 5/5 fenêtres.
    Fine-tune: lb=35, vol=4.5, SL=1.8%, TP=3.5%, max_hold=36h.
    $+1,782, DD 8.1%, WR 58%, 113 trades.
    Anti-wick 60% filter.

    Logique :
      - Breakout haut : mid_price > HIGH(35 dernières bougies)
      - Breakout bas  : mid_price < LOW(35 dernières bougies)
      - Filtre volume : vol_ratio >= 4.5
      - Filtre anti-wick : ignore signaux si wick > 60% de la bougie
      - Direction     : long si prix > SMA50, short sinon

    Sizing réduit (20%) pour cohabiter avec BTC + SOL sans
    dépasser le daily loss limit.
    """

    def __init__(self):
        super().__init__()
        self.name = "eth_breakout_relaxed_1h"
        self.tp_pct = 0.035
        self.sl_pct = 0.018
        self.equity_pct = 0.20
        self.leverage = 5
        self.cooldown_sec = 14400.0
        self.max_hold_sec = 129600.0
        self.entry_offset_pct = 0.0002
        self.entry_timeout_sec = 90.0

        self.lookback = 35
        self.vol_min = 4.5
        self.max_wick_ratio = 0.60

    def _scan_signals(self, ind, mid_price):
        if mid_price <= 0:
            return None

        if ind.vol_ratio < self.vol_min:
            return None

        if not self._sentiment_ok():
            return None

        candles = self.api.get_candles(self.coin, "1h", 200)
        if not candles or len(candles) < self.lookback + 2:
            return None

        # Anti-wick filter: skip signals on high-wick candles (manipulation)
        last_candle = candles[-2]
        body = abs(last_candle.close - last_candle.open)
        total_range = last_candle.high - last_candle.low
        if total_range > 0:
            wick_ratio = 1 - body / total_range
            if wick_ratio >= self.max_wick_ratio:
                return None

        recent = candles[-(self.lookback + 1):-1]
        rolling_high = max(c.high for c in recent)
        rolling_low = min(c.low for c in recent)

        trend_bull = mid_price > ind.sma_50 if ind.sma_50 > 0 else True

        if mid_price > rolling_high and trend_bull:
            self.api.log(logging.INFO,
                         f"BREAKOUT UP: {mid_price:.2f} > {rolling_high:.2f} "
                         f"vol={ind.vol_ratio:.1f}")
            return {"side": "buy", "signal": "BREAKOUT_UP"}

        if mid_price < rolling_low and not trend_bull:
            self.api.log(logging.INFO,
                         f"BREAKOUT DOWN: {mid_price:.2f} < {rolling_low:.2f} "
                         f"vol={ind.vol_ratio:.1f}")
            return {"side": "sell", "signal": "BREAKOUT_DOWN"}

        return None

    def _sentiment_ok(self):
        sentiment = self.api.get_sentiment(self.coin)
        if sentiment < self.api.config.sentiment.hard_block_threshold:
            self.api.log(logging.WARNING, f"Trade blocked — sentiment {sentiment:.2f}")
            return False
        return True
