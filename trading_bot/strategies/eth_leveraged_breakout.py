import logging

from trading_bot.strategies.template import TemplateStrategy


class EthLeveragedBreakout(TemplateStrategy):
    """ETH Leveraged Breakout x20 — High Leverage Scalp
    ====================================================
    Meme logique que BTC Leveraged Breakout, appliquee a ETH.
    Breakout court terme (lb=6) avec x20 et SL ultra-serre.

    Backtest realiste (2023-01 -> 2026-01) sans compounding :
      Return fixe : +190%/an  |  WR : 13.5%  |  PF : 1.67
      Avg win: +$73  |  Avg loss: -$7  |  Ratio: 10.6x

    Groupe : high-leverage
    """

    def __init__(self):
        super().__init__()
        self.name = "eth_leveraged_breakout"
        self.tp_pct = 0.025
        self.sl_pct = 0.0015
        self.equity_pct = 0.15
        self.leverage = 20
        self.cooldown_sec = 3600.0
        self.max_hold_sec = 86400.0
        self.entry_offset_pct = 0.0002
        self.entry_timeout_sec = 60.0
        self.check_sec = 3.0

        self.lookback = 6
        self.vol_min = 1.2

    def _scan_signals(self, ind, mid_price):
        if mid_price <= 0:
            return None

        if ind.vol_ratio < self.vol_min:
            return None

        candles = self.api.get_candles(self.coin, "1h", 200)
        if not candles or len(candles) < self.lookback + 2:
            return None

        recent = candles[-(self.lookback + 1):-1]
        rolling_high = max(c.high for c in recent)
        rolling_low = min(c.low for c in recent)

        if mid_price > rolling_high:
            self.api.log(logging.INFO,
                         "BREAKOUT UP x20: %.2f > %.2f (lb=%d) vol=%.1f" %
                         (mid_price, rolling_high, self.lookback, ind.vol_ratio))
            return {"side": "buy", "signal": "BREAKOUT_UP"}

        if mid_price < rolling_low:
            self.api.log(logging.INFO,
                         "BREAKOUT DOWN x20: %.2f < %.2f (lb=%d) vol=%.1f" %
                         (mid_price, rolling_low, self.lookback, ind.vol_ratio))
            return {"side": "sell", "signal": "BREAKOUT_DOWN"}

        return None
