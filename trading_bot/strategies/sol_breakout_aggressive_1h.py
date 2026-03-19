import logging

from trading_bot.strategies.template import TemplateStrategy


class SolBreakoutAggressive1h(TemplateStrategy):
    """SOL Breakout Aggressive — Lookback 10, SL 1%, TP 8%.

    Backtest: Sharpe +1.54 moyen, positif sur 10/10 fenêtres.
    $1000 → $5,668 (6M compound), $10,113 (1Y compound).

    Logique :
      - Breakout haut : mid_price > HIGH(10 dernières bougies)
      - Breakout bas  : mid_price < LOW(10 dernières bougies)
      - Filtre volume : vol_ratio >= 2.5
      - Direction     : long si prix > SMA50, short sinon
    """

    def __init__(self):
        super().__init__()
        self.name = "sol_breakout_aggressive_1h"
        self.tp_pct = 0.08
        self.sl_pct = 0.01
        self.equity_pct = 0.50
        self.leverage = 7
        self.cooldown_sec = 7200.0
        self.max_hold_sec = 172800.0
        self.entry_offset_pct = 0.0002
        self.entry_timeout_sec = 90.0

        self.lookback = 10
        self.vol_min = 2.5

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
                         f"BREAKOUT UP: {mid_price:.2f} > {rolling_high:.2f} vol={ind.vol_ratio:.1f}")
            return {"side": "buy", "signal": "BREAKOUT_UP"}

        if mid_price < rolling_low and not trend_bull:
            self.api.log(logging.INFO,
                         f"BREAKOUT DOWN: {mid_price:.2f} < {rolling_low:.2f} vol={ind.vol_ratio:.1f}")
            return {"side": "sell", "signal": "BREAKOUT_DOWN"}

        return None

    def _sentiment_ok(self):
        sentiment = self.api.get_sentiment(self.coin)
        if sentiment < self.api.config.sentiment.hard_block_threshold:
            self.api.log(logging.WARNING, f"Trade blocked — sentiment {sentiment:.2f}")
            return False
        return True
