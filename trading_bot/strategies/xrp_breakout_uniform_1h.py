import logging

from trading_bot.strategies.template import TemplateStrategy


class XrpBreakoutUniform1h(TemplateStrategy):
    """XRP Breakout Uniform — Lookback 32, SL 0.3%, TP 4%.

    6-Coin Uniform profile: identical parameters across all coins.
    Backtest: +417% (3Y), MaxDD 13.8%, Sharpe ~2.1.

    Logique :
      - Breakout haut : mid_price > HIGH(32 dernieres bougies)
      - Breakout bas  : mid_price < LOW(32 dernieres bougies)
      - Filtre volume : vol_ratio >= 0.8
      - Direction     : long si prix > SMA50, short sinon

    SL tres serre (0.3%) — scalping de breakout. Beaucoup de trades
    mais ratio TP/SL de 13:1 compense le faible WR.
    Sizing 35% — excellent ratio rendement/risque.
    """

    def __init__(self):
        super().__init__()
        self.name = "xrp_breakout_uniform_1h"
        self.tp_pct = 0.04
        self.sl_pct = 0.003
        self.equity_pct = 0.35
        self.leverage = 5
        self.cooldown_sec = 10800.0  # 3h (cooldown_bars=3)
        self.max_hold_sec = 172800.0
        self.entry_offset_pct = 0.0002
        self.entry_timeout_sec = 90.0

        self.lookback = 32
        self.vol_min = 0.8

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

        recent = candles[-(self.lookback + 1):-1]
        rolling_high = max(c.high for c in recent)
        rolling_low = min(c.low for c in recent)

        trend_bull = mid_price > ind.sma_50 if ind.sma_50 > 0 else True

        if mid_price > rolling_high and trend_bull:
            self.api.log(logging.INFO,
                         "BREAKOUT UP: %.2f > %.2f vol=%.1f" %
                         (mid_price, rolling_high, ind.vol_ratio))
            return {"side": "buy", "signal": "BREAKOUT_UP"}

        if mid_price < rolling_low and not trend_bull:
            self.api.log(logging.INFO,
                         "BREAKOUT DOWN: %.2f < %.2f vol=%.1f" %
                         (mid_price, rolling_low, ind.vol_ratio))
            return {"side": "sell", "signal": "BREAKOUT_DOWN"}

        return None

    def _sentiment_ok(self):
        sentiment = self.api.get_sentiment(self.coin)
        if sentiment < self.api.config.sentiment.hard_block_threshold:
            self.api.log(logging.WARNING,
                         "Trade blocked — sentiment %.2f" % sentiment)
            return False
        return True
