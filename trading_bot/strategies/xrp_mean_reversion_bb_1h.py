import logging

from trading_bot.strategies.template import TemplateStrategy


class XrpMeanReversionBb1h(TemplateStrategy):
    """XRP Mean Reversion Bollinger Bands — RSI 20/70, BB 0.08/0.95, SL 0.7%, TP 8%.

    Backtest: Sharpe +1.86 (3Y), stable 5/5 fenetres 6M.
    Fine-tune: RSI 20/70, BB %B 0.08/0.95, SL=0.7%, TP=8.0%, anti-wick 50%.
    $+761, DD 3.3%, WR 29%, 143 trades, PF 2.55.

    Logique :
      - LONG : RSI < 20 AND %B < 0.08 (prix sous BB lower)
      - SHORT : RSI > 70 AND %B > 0.95 (prix au-dessus BB upper)
      - Filtre anti-wick : ignore si wick > 50% de la bougie
      - Filtre sentiment : bloque si sentiment < seuil

    Sizing 35% — le meilleur ratio rendement/risque du portfolio.
    """

    def __init__(self):
        super().__init__()
        self.name = "xrp_mean_reversion_bb_1h"
        self.tp_pct = 0.08
        self.sl_pct = 0.007
        self.equity_pct = 0.35
        self.leverage = 5
        self.cooldown_sec = 14400.0
        self.max_hold_sec = 172800.0
        self.entry_offset_pct = 0.0002
        self.entry_timeout_sec = 90.0

        self.rsi_oversold = 20
        self.rsi_overbought = 70
        self.bb_entry_low = 0.08
        self.bb_entry_high = 0.95
        self.max_wick_ratio = 0.50

    def _scan_signals(self, ind, mid_price):
        if mid_price <= 0:
            return None

        if not self._sentiment_ok():
            return None

        candles = self.api.get_candles(self.coin, "1h", 200)
        if not candles or len(candles) < 5:
            return None

        # Anti-wick filter
        last_candle = candles[-2]
        body = abs(last_candle.close - last_candle.open)
        total_range = last_candle.high - last_candle.low
        if total_range > 0:
            wick_ratio = 1 - body / total_range
            if wick_ratio >= self.max_wick_ratio:
                return None

        # Bollinger Band %B
        bb_range = ind.bb_upper - ind.bb_lower
        if bb_range <= 0:
            return None
        pct_b = (mid_price - ind.bb_lower) / bb_range

        # Mean reversion LONG: oversold + under BB lower zone
        if ind.rsi_14 < self.rsi_oversold and pct_b < self.bb_entry_low:
            self.api.log(logging.INFO,
                         "MR LONG: RSI %.1f < %d, %%B %.2f < %.2f" %
                         (ind.rsi_14, self.rsi_oversold, pct_b, self.bb_entry_low))
            return {"side": "buy", "signal": "MR_LONG"}

        # Mean reversion SHORT: overbought + above BB upper zone
        if ind.rsi_14 > self.rsi_overbought and pct_b > self.bb_entry_high:
            self.api.log(logging.INFO,
                         "MR SHORT: RSI %.1f > %d, %%B %.2f > %.2f" %
                         (ind.rsi_14, self.rsi_overbought, pct_b, self.bb_entry_high))
            return {"side": "sell", "signal": "MR_SHORT"}

        return None

    def _sentiment_ok(self):
        sentiment = self.api.get_sentiment(self.coin)
        if sentiment < self.api.config.sentiment.hard_block_threshold:
            self.api.log(logging.WARNING,
                         "Trade blocked — sentiment %.2f" % sentiment)
            return False
        return True
