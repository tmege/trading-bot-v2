import logging

from trading_bot.strategies.template import TemplateStrategy


class BtcInsideBarBreakout1h(TemplateStrategy):
    """BTC Inside Bar Breakout — SL 1.5%, TP 3%, trend+ATR filters.

    Backtest: Sharpe +1.36 (3Y), stable 5/5 fenêtres 6M.
    $1000 → $1,429 (3Y compound, 20% equity).
    54% win rate, max drawdown 5.4%.

    Logique :
      - Inside bar : high[i-1] < high[i-2] AND low[i-1] > low[i-2]
      - Breakout   : mid_price > high[i-1] → LONG, < low[i-1] → SHORT
      - Filtre vol : vol_ratio >= 1.5
      - Filtre trend : close > EMA21 pour long, < EMA21 pour short
      - Filtre ATR : compression (ATR percentile < 20%)

    Sizing 20% pour cohabiter avec SOL + ETH.
    """

    def __init__(self):
        super().__init__()
        self.name = "btc_inside_bar_breakout_1h"
        self.tp_pct = 0.03
        self.sl_pct = 0.015
        self.equity_pct = 0.20
        self.leverage = 5
        self.cooldown_sec = 14400.0
        self.max_hold_sec = 259200.0
        self.entry_offset_pct = 0.0002
        self.entry_timeout_sec = 90.0

        self.vol_min = 1.5

    def _scan_signals(self, ind, mid_price):
        if mid_price <= 0:
            return None

        if ind.vol_ratio < self.vol_min:
            return None

        if not self._sentiment_ok():
            return None

        candles = self.api.get_candles(self.coin, "1h", 200)
        if not candles or len(candles) < 5:
            return None

        # Inside bar: bar[-2] contained within bar[-3]
        mother = candles[-3]
        inside = candles[-2]

        is_inside = inside.high < mother.high and inside.low > mother.low
        if not is_inside:
            return None

        # ATR compression filter
        if ind.atr_pct_rank >= 0.20:
            return None

        # EMA21 for trend filter
        closes = [c.close for c in candles]
        ema21 = self._compute_ema(closes, 21)

        # Breakout up
        if mid_price > inside.high:
            if ema21 > 0 and mid_price <= ema21:
                return None
            self.api.log(logging.INFO,
                         "IB BREAKOUT UP: %.2f > %.2f "
                         "vol=%.1f atr_rank=%.2f" %
                         (mid_price, inside.high,
                          ind.vol_ratio, ind.atr_pct_rank))
            return {"side": "buy", "signal": "IB_BREAKOUT_UP"}

        # Breakout down
        if mid_price < inside.low:
            if ema21 > 0 and mid_price >= ema21:
                return None
            self.api.log(logging.INFO,
                         "IB BREAKOUT DOWN: %.2f < %.2f "
                         "vol=%.1f atr_rank=%.2f" %
                         (mid_price, inside.low,
                          ind.vol_ratio, ind.atr_pct_rank))
            return {"side": "sell", "signal": "IB_BREAKOUT_DOWN"}

        return None

    @staticmethod
    def _compute_ema(values, period):
        if len(values) < period:
            return 0.0
        k = 2.0 / (period + 1)
        ema = sum(values[:period]) / period
        for i in range(period, len(values)):
            ema = values[i] * k + ema * (1 - k)
        return ema

    def _sentiment_ok(self):
        sentiment = self.api.get_sentiment(self.coin)
        if sentiment < self.api.config.sentiment.hard_block_threshold:
            self.api.log(logging.WARNING,
                         "Trade blocked — sentiment %.2f" % sentiment)
            return False
        return True
