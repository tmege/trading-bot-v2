import logging

from trading_bot.strategies.template import TemplateStrategy


class BtcLeveragedBreakout(TemplateStrategy):
    """BTC Leveraged Breakout x20 — High Leverage Scalp
    ====================================================
    Breakout court terme (lb=6) avec x20 et SL ultra-serre.
    Logique : breakout sur 6h de range avec spike de volume.
    Le R:R extreme (1:16.7) compense le WR bas (~16%).

    Backtest realiste (2023-01 -> 2026-01) — compounding, frais, funding :
      Return   : +3 098%    Sharpe : 3.05   MaxDD : 17.6%   Trades : 2 210
      PF       : 1.64       WR     : 13.6%
      ~2 trades/jour

    Risk par trade :
      SL 0.15% x 20 x 15% = -0.45% equity
      TP 2.50% x 20 x 15% = +7.50% equity
      R:R = 1:16.7

    Groupe : high-leverage
    """

    def __init__(self):
        super().__init__()
        self.name = "btc_leveraged_breakout"
        self.tp_pct = 0.025            # 2.50%
        self.sl_pct = 0.0015           # 0.15%
        self.equity_pct = 0.15
        self.leverage = 20
        self.cooldown_sec = 3600.0     # 1h (cooldown_bars=1)
        self.max_hold_sec = 86400.0    # 24h
        self.entry_offset_pct = 0.0002
        self.entry_timeout_sec = 60.0
        self.check_sec = 3.0           # check toutes les 3s pour reactivite

        # --- Breakout params ---
        self.lookback = 6              # 6h rolling window
        self.vol_min = 1.2             # volume > 1.2x average

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
