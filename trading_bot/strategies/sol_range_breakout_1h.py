import logging
import json

from trading_bot.strategies.template import TemplateStrategy

RB_LOOKBACK = 24
RB_MIN_RANGE_PCT = 0.008
RB_MAX_RANGE_PCT = 0.06
RB_VOL_MULT = 1.0

ATR_BASELINE = 0.008
B1_ATR_THRESHOLD = 0.015

TRAIL_ACTIVATE_PCT = 0.025
TRAIL_OFFSET_PCT = 0.010
TRAIL_STEP_PCT = 0.005


class SolRangeBreakout1h(TemplateStrategy):
    def __init__(self):
        super().__init__()
        self.name = "sol_range_breakout_1h"
        self.equity_pct = 0.40
        self.leverage = 5
        self.cooldown_sec = 7200.0
        self.max_hold_sec = 172800.0
        self.tp_pct = 0.04
        self.sl_pct = 0.015
        self.entry_offset_pct = 0.0002
        self.entry_timeout_sec = 90.0

        self.trailing_enabled = True
        self.peak_price = 0.0
        self.current_sl_price = 0.0

        self.grid_tp = 0.0
        self.grid_sl = 0.0

    @property
    def effective_tp(self) -> float:
        return self.grid_tp if self.grid_tp > 0 else self.tp_pct

    @property
    def effective_sl(self) -> float:
        return self.grid_sl if self.grid_sl > 0 else self.sl_pct

    def _scan_signals(self, ind, mid_price):
        # B1 (bear mode) has priority
        b1 = self._check_bear_b1(ind, mid_price)
        if b1:
            return b1

        # TF (trend follow) — alternative when not in range
        tf = self._check_trend_follow(ind, mid_price)
        if tf:
            return tf

        # RB (range breakout)
        rb = self._check_range_breakout(ind, mid_price)
        if rb:
            return rb

        return None

    def _check_bear_b1(self, ind, mid_price):
        if mid_price <= 0 or ind.sma_200 <= 0:
            return None

        # Simplified: removed death cross requirement (too slow to form)
        if not (mid_price < ind.sma_200):
            return None

        atr_ratio = ind.atr_14 / mid_price
        if (ind.rsi_14 < 40 and ind.macd_line < 0 and
                atr_ratio < B1_ATR_THRESHOLD):
            # Use local TP/SL for B1 — passed via signal dict
            self.api.log(logging.INFO,
                         f"B1 signal: RSI={ind.rsi_14:.1f} MACD={ind.macd_line:.6f}")
            return {"side": "sell", "signal": "B1", "tp": 0.06, "sl": 0.02}

        return None

    def _check_trend_follow(self, ind, mid_price):
        if mid_price <= 0 or ind.sma_50 <= 0:
            return None

        # TF Long: uptrend with momentum
        if (mid_price > ind.sma_50 and
                ind.supertrend_up and
                45 <= ind.rsi_14 <= 65 and
                ind.macd_histogram > 0 and
                ind.adx_14 > 20):
            self.api.log(logging.INFO,
                         f"TF_LONG signal: RSI={ind.rsi_14:.1f} ADX={ind.adx_14:.1f}")
            return {"side": "buy", "signal": "TF_LONG"}

        # TF Short: downtrend with momentum
        if (mid_price < ind.sma_50 and
                not ind.supertrend_up and
                35 <= ind.rsi_14 <= 55 and
                ind.macd_histogram < 0 and
                ind.adx_14 > 20):
            self.api.log(logging.INFO,
                         f"TF_SHORT signal: RSI={ind.rsi_14:.1f} ADX={ind.adx_14:.1f}")
            return {"side": "sell", "signal": "TF_SHORT"}

        return None

    def _check_range_breakout(self, ind, mid_price):
        candles = self.api.get_candles(self.coin, "1h", RB_LOOKBACK + 5, mid_price)
        if len(candles) < RB_LOOKBACK + 1:
            return None

        # Use completed candles only (exclude the last incomplete one)
        completed = candles[:-1]
        lookback_candles = completed[-RB_LOOKBACK:]

        if len(lookback_candles) < RB_LOOKBACK:
            return None

        range_hi = max(c.high for c in lookback_candles)
        range_lo = min(c.low for c in lookback_candles)

        if range_hi <= 0:
            return None

        range_pct = (range_hi - range_lo) / range_hi
        if range_pct < RB_MIN_RANGE_PCT or range_pct > RB_MAX_RANGE_PCT:
            return None

        # Compare last completed candle volume to average of lookback (all completed)
        vol_avg = sum(c.volume for c in lookback_candles) / RB_LOOKBACK
        last_completed_vol = completed[-1].volume
        if vol_avg <= 0 or last_completed_vol < vol_avg * RB_VOL_MULT:
            return None

        if mid_price > range_hi:
            self.api.log(logging.INFO,
                         f"RB LONG: price {mid_price:.2f} > range_hi {range_hi:.2f}")
            return {"side": "buy", "signal": "RB_LONG"}

        if mid_price < range_lo:
            self.api.log(logging.INFO,
                         f"RB SHORT: price {mid_price:.2f} < range_lo {range_lo:.2f}")
            return {"side": "sell", "signal": "RB_SHORT"}

        return None

    def _compute_size(self, mid_price):
        account = self.api.get_account_value()
        if account <= 0 or mid_price <= 0:
            return 0.0

        atr_ratio = self._get_atr_ratio(mid_price)
        atr_mult = ATR_BASELINE / atr_ratio if atr_ratio > 0 else 1.0
        atr_mult = max(0.5, min(1.5, atr_mult))

        dd_mult = self._drawdown_multiplier()
        notional = account * self.equity_pct * self.leverage * atr_mult * dd_mult
        return notional / mid_price

    def _drawdown_multiplier(self):
        account = self.api.get_account_value()
        if account <= 0 or self.peak_equity <= 0:
            return 1.0

        dd = (self.peak_equity - account) / self.peak_equity

        if dd > 0.34:
            self.api.log(logging.WARNING, f"Total DD {dd*100:.1f}% > 34% — STOPPED")
            return 0.0

        daily_pnl = self.api.get_daily_pnl()
        if account > 0 and daily_pnl < 0 and abs(daily_pnl) / account > 0.08:
            return 0.50

        if dd > 0.18:
            return 0.25
        if dd > 0.10:
            return 0.50

        return 1.0

    def _get_atr_ratio(self, mid_price):
        ind = self.api.get_indicators(self.coin, "1h", 50, mid_price)
        if ind and mid_price > 0:
            return ind.atr_14 / mid_price
        return ATR_BASELINE

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

    def _place_entry(self, signal, mid_price, now):
        side = signal["side"]
        size = self._compute_size(mid_price)
        if size <= 0:
            return

        # Store signal-specific TP/SL if provided (e.g. B1)
        self._signal_tp = signal.get("tp", 0)
        self._signal_sl = signal.get("sl", 0)

        if side == "buy":
            price = mid_price * (1 - self.entry_offset_pct)
        else:
            price = mid_price * (1 + self.entry_offset_pct)

        oid = self.api.place_limit(self.coin, side, price, size, tif="alo")
        if oid:
            self.entry_oid = oid
            self.entry_placed_at = now
            self.position_side = side
            self.api.log(logging.INFO, f"Entry placed: {side} {size:.6f} @ {price:.2f}")

    def _handle_entry_fill(self, fill):
        self.in_position = True
        self.entry_price = fill.px.to_float()
        self.entry_time = fill.time_ms / 1000.0
        self.entry_oid = 0
        self.peak_price = self.entry_price
        self.current_sl_price = 0.0
        size = fill.sz.to_float()

        # Use signal-specific TP/SL if set, otherwise defaults
        tp = self._signal_tp if getattr(self, "_signal_tp", 0) > 0 else self.effective_tp
        sl = self._signal_sl if getattr(self, "_signal_sl", 0) > 0 else self.effective_sl
        self._signal_tp = 0
        self._signal_sl = 0

        if self.position_side == "buy":
            tp_px = self.entry_price * (1 + tp)
            sl_px = self.entry_price * (1 - sl)
            self.tp_oid = self.api.place_trigger(self.coin, "sell", tp_px, size, tp_px, tpsl="tp")
            self.sl_oid = self.api.place_trigger(self.coin, "sell", sl_px, size, sl_px, tpsl="sl")
            self.current_sl_price = sl_px
        else:
            tp_px = self.entry_price * (1 - tp)
            sl_px = self.entry_price * (1 + sl)
            self.tp_oid = self.api.place_trigger(self.coin, "buy", tp_px, size, tp_px, tpsl="tp")
            self.sl_oid = self.api.place_trigger(self.coin, "buy", sl_px, size, sl_px, tpsl="sl")
            self.current_sl_price = sl_px

        self.api.log(logging.INFO, f"Position opened: {self.position_side} @ {self.entry_price:.4f}")

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
