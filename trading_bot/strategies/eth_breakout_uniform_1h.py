import logging

from trading_bot.strategies.template import TemplateStrategy


class EthBreakoutUniform1h(TemplateStrategy):
    """ETH Breakout Uniform 1H
    ========================
    Profil : SL 0.3% / TP 4% / lookback 32 / equity 35% / lev 5x / no anti-wick

    Backtest realiste (2019-11 -> 2026-03) — compounding, frais maker/taker, funding :
      Return   : +3 255%    Sharpe : 2.26   MaxDD : 12.8%   Trades : 1 829
      PF       : 1.76       Fees   : $9 541

    Contexte :
      66% du PnL genere en 2024-2026 (effet compounding).
      Meilleur return absolu apres DOGE. Nombre de trades le plus eleve
      du groupe (1 829) — bonne liquidite et volatilite suffisante.

    Groupe : 6-coin-uniform
    """

    def __init__(self):
        super().__init__()
        self.name = "eth_breakout_uniform_1h"
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
