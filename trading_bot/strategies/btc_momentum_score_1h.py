import logging

from trading_bot.strategies.template import TemplateStrategy


class BtcMomentumScore1h(TemplateStrategy):
    """BTC Momentum Score — Threshold 1/3, SL 2.5%, TP 6%.

    Backtest: Sharpe +0.65 moyen, positif sur 10/10 fenêtres.
    $1000 → $1,590 (6M compound), $1,634 (1Y compound).

    Score composite (4 conditions booléennes) :
      1. close > SMA20  (proxy EMA21)
      2. RSI(14) > 50
      3. MACD_histogram > 0
      4. vol_ratio > 1.2

    LONG  : score passe de < 1 à >= 3
    SHORT : score passe de > 3 à <= 1
    """

    def __init__(self):
        super().__init__()
        self.name = "btc_momentum_score_1h"
        self.tp_pct = 0.06
        self.sl_pct = 0.025
        self.equity_pct = 0.35
        self.leverage = 5
        self.cooldown_sec = 14400.0
        self.max_hold_sec = 259200.0
        self.entry_offset_pct = 0.0002
        self.entry_timeout_sec = 90.0

        self.threshold_low = 1
        self.threshold_high = 3
        self.prev_score = -1

    def _scan_signals(self, ind, mid_price):
        if mid_price <= 0:
            return None

        if not self._sentiment_ok():
            return None

        score = (
            (1 if mid_price > ind.sma_20 and ind.sma_20 > 0 else 0)
            + (1 if ind.rsi_14 > 50 else 0)
            + (1 if ind.macd_histogram > 0 else 0)
            + (1 if ind.vol_ratio > 1.2 else 0)
        )

        signal = None

        if self.prev_score >= 0:
            if self.prev_score < self.threshold_low and score >= self.threshold_high:
                self.api.log(logging.INFO,
                             f"MOMENTUM LONG: score {self.prev_score}→{score} "
                             f"RSI={ind.rsi_14:.0f} MACD={ind.macd_histogram:.2f}")
                signal = {"side": "buy", "signal": "MOM_LONG"}

            elif self.prev_score > self.threshold_high and score <= self.threshold_low:
                self.api.log(logging.INFO,
                             f"MOMENTUM SHORT: score {self.prev_score}→{score} "
                             f"RSI={ind.rsi_14:.0f} MACD={ind.macd_histogram:.2f}")
                signal = {"side": "sell", "signal": "MOM_SHORT"}

        self.prev_score = score
        return signal

    def _sentiment_ok(self):
        sentiment = self.api.get_sentiment(self.coin)
        if sentiment < self.api.config.sentiment.hard_block_threshold:
            self.api.log(logging.WARNING, f"Trade blocked — sentiment {sentiment:.2f}")
            return False
        return True

    def _save_state(self):
        import json
        state = {
            "in_position": self.in_position,
            "position_side": self.position_side,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "consec_losses": self.consec_losses,
            "peak_equity": self.peak_equity,
            "last_trade": self.last_trade,
            "sl_oid": self.sl_oid,
            "tp_oid": self.tp_oid,
            "prev_score": self.prev_score,
        }
        self.api.save_state("state", json.dumps(state))

    def _restore_state(self):
        import json
        raw = self.api.load_state("state")
        if not raw:
            return
        try:
            state = json.loads(raw)
            self.in_position = state.get("in_position", False)
            self.position_side = state.get("position_side", "")
            self.entry_price = state.get("entry_price", 0.0)
            self.entry_time = state.get("entry_time", 0.0)
            self.trade_count = state.get("trade_count", 0)
            self.win_count = state.get("win_count", 0)
            self.consec_losses = state.get("consec_losses", 0)
            self.peak_equity = state.get("peak_equity", 0.0)
            self.last_trade = state.get("last_trade", 0.0)
            self.sl_oid = state.get("sl_oid", 0)
            self.tp_oid = state.get("tp_oid", 0)
            self.prev_score = state.get("prev_score", -1)
        except (json.JSONDecodeError, KeyError):
            pass
