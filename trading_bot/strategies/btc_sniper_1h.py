import logging
import json

from trading_bot.strategies.template import TemplateStrategy

TRAIL_ACTIVATE_PCT = 0.020
TRAIL_OFFSET_PCT = 0.010
TRAIL_STEP_PCT = 0.005


class BtcSniper1h(TemplateStrategy):
    def __init__(self):
        super().__init__()
        self.name = "btc_sniper_1h"
        self.equity_pct = 0.40
        self.leverage = 5
        self.cooldown_sec = 7200.0
        self.max_hold_sec = 172800.0
        self.tp_pct = 0.03
        self.sl_pct = 0.015
        self.entry_offset_pct = 0.0002
        self.entry_timeout_sec = 90.0

        self.trailing_enabled = True
        self.peak_price = 0.0
        self.current_sl_price = 0.0

    def _scan_signals(self, ind, mid_price):
        atr_ratio = ind.atr_14 / mid_price if mid_price > 0 else 1.0

        # L1 — Pullback Long: trend-following pullback in uptrend
        if (40 <= ind.rsi_14 <= 60 and
                mid_price > ind.sma_50 > 0 and
                ind.sma_20 > ind.sma_50 and
                ind.macd_hist_incr and
                ind.adx_14 > 20 and
                atr_ratio < 0.01):
            self.api.log(logging.INFO,
                         f"L1 signal: RSI={ind.rsi_14:.1f} ADX={ind.adx_14:.1f} ATR%={atr_ratio:.5f}")
            return {"side": "buy", "signal": "L1"}

        # S1 — Pullback Short: trend-following pullback in downtrend
        if (40 <= ind.rsi_14 <= 60 and
                mid_price < ind.sma_50 and ind.sma_50 > 0 and
                ind.sma_20 < ind.sma_50 and
                ind.macd_hist_decr and
                ind.adx_14 > 20 and
                atr_ratio < 0.01):
            self.api.log(logging.INFO,
                         f"S1 signal: RSI={ind.rsi_14:.1f} ADX={ind.adx_14:.1f} ATR%={atr_ratio:.5f}")
            return {"side": "sell", "signal": "S1"}

        # L2 — Supertrend Long: higher conviction trend trade
        if (ind.supertrend_up and
                mid_price > ind.sma_50 > 0 and
                ind.rsi_14 > 45 and
                ind.macd_histogram > 0):
            self.api.log(logging.INFO,
                         f"L2 signal: RSI={ind.rsi_14:.1f} MACD_H={ind.macd_histogram:.6f}")
            return {"side": "buy", "signal": "L2"}

        # S2 — Supertrend Short: higher conviction trend trade
        if (not ind.supertrend_up and
                mid_price < ind.sma_50 and ind.sma_50 > 0 and
                ind.rsi_14 < 55 and
                ind.macd_histogram < 0):
            self.api.log(logging.INFO,
                         f"S2 signal: RSI={ind.rsi_14:.1f} MACD_H={ind.macd_histogram:.6f}")
            return {"side": "sell", "signal": "S2"}

        return None

    def on_tick(self, coin, mid_price):
        if coin != self.coin:
            return
        now = self.api.time()
        if now - self.last_check < self.check_sec:
            return
        self.last_check = now

        self._update_macd_history(now)

        ind = self.api.get_indicators(self.coin, "1h", 200, mid_price)
        if ind:
            self.last_macd_val = ind.macd_histogram

        if self.in_position:
            self._update_trailing_stop(mid_price)
            self._monitor_position(mid_price, now)
            return

        if now - self.last_trade < self.cooldown_sec:
            return

        if not ind or not ind.valid:
            return

        signal = self._scan_signals(ind, mid_price)
        if signal:
            self._place_entry(signal, mid_price, now)

    def _handle_entry_fill(self, fill):
        self.in_position = True
        self.entry_price = fill.px.to_float()
        self.entry_time = fill.time_ms / 1000.0
        self.entry_oid = 0
        self.peak_price = self.entry_price
        self.current_sl_price = 0.0
        size = fill.sz.to_float()

        if self.position_side == "buy":
            tp_px = self.entry_price * (1 + self.tp_pct)
            sl_px = self.entry_price * (1 - self.sl_pct)
            self.tp_oid = self.api.place_trigger(
                self.coin, "sell", tp_px, size, tp_px, tpsl="tp"
            )
            self.sl_oid = self.api.place_trigger(
                self.coin, "sell", sl_px, size, sl_px, tpsl="sl"
            )
            self.current_sl_price = sl_px
        else:
            tp_px = self.entry_price * (1 - self.tp_pct)
            sl_px = self.entry_price * (1 + self.sl_pct)
            self.tp_oid = self.api.place_trigger(
                self.coin, "buy", tp_px, size, tp_px, tpsl="tp"
            )
            self.sl_oid = self.api.place_trigger(
                self.coin, "buy", sl_px, size, sl_px, tpsl="sl"
            )
            self.current_sl_price = sl_px

        self.api.log(logging.INFO,
                     f"Position opened: {self.position_side} @ {self.entry_price:.2f}")

    def _handle_exit_fill(self, fill):
        pnl = fill.closed_pnl.to_float()
        self.trade_count += 1
        if pnl > 0:
            self.win_count += 1
            self.consec_losses = 0
        else:
            self.consec_losses += 1

        self.api.cancel_all(self.coin)
        self.in_position = False
        self.entry_oid = 0
        self.sl_oid = 0
        self.tp_oid = 0
        self.last_trade = self.api.time()
        self.peak_price = 0.0
        self.current_sl_price = 0.0

        account = self.api.get_account_value()
        if account > self.peak_equity:
            self.peak_equity = account

        self.api.log(logging.INFO,
                     f"Position closed: PnL={pnl:.4f} trades={self.trade_count} wins={self.win_count}")
        self._save_state()

    def _update_trailing_stop(self, mid_price):
        if not self.trailing_enabled:
            return
        if not self.in_position or self.entry_price <= 0:
            return

        if self.position_side == "buy":
            unrealized_pct = (mid_price - self.entry_price) / self.entry_price
            if unrealized_pct < TRAIL_ACTIVATE_PCT:
                return
            self.peak_price = max(self.peak_price, mid_price)
            new_sl = self.peak_price * (1 - TRAIL_OFFSET_PCT)
            if new_sl > self.current_sl_price + self.entry_price * TRAIL_STEP_PCT:
                pos = self.api.get_position(self.coin)
                if pos:
                    size = abs(pos.size.to_float())
                    self.api.cancel(self.coin, self.sl_oid)
                    self.sl_oid = self.api.place_trigger(
                        self.coin, "sell", new_sl, size, new_sl, tpsl="sl"
                    )
                    self.current_sl_price = new_sl
        else:
            unrealized_pct = (self.entry_price - mid_price) / self.entry_price
            if unrealized_pct < TRAIL_ACTIVATE_PCT:
                return
            self.peak_price = (
                min(self.peak_price, mid_price) if self.peak_price > 0
                else mid_price
            )
            new_sl = self.peak_price * (1 + TRAIL_OFFSET_PCT)
            if new_sl < self.current_sl_price - self.entry_price * TRAIL_STEP_PCT:
                pos = self.api.get_position(self.coin)
                if pos:
                    size = abs(pos.size.to_float())
                    self.api.cancel(self.coin, self.sl_oid)
                    self.sl_oid = self.api.place_trigger(
                        self.coin, "buy", new_sl, size, new_sl, tpsl="sl"
                    )
                    self.current_sl_price = new_sl

    def _save_state(self):
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
        }
        self.api.save_state("state", json.dumps(state))

    def _restore_state(self):
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
        except (json.JSONDecodeError, KeyError):
            pass
