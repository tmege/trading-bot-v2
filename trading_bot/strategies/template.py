import logging


class TemplateStrategy:
    def __init__(self):
        self.name = "template"
        self.coin = ""
        self.api = None
        self.errored = False

        self.in_position = False
        self.position_side = ""
        self.entry_price = 0.0
        self.entry_time = 0.0
        self.entry_oid = 0
        self.entry_placed_at = 0.0
        self.sl_oid = 0
        self.tp_oid = 0

        self.trade_count = 0
        self.win_count = 0
        self.consec_losses = 0
        self.peak_equity = 0.0

        self.last_check = 0.0
        self.last_trade = 0.0

        self.last_hour = 0
        self.last_macd_val = 0.0
        self.prev_macd = 0.0
        self.prev2_macd = 0.0

        # Config — override in subclass
        self.check_sec = 5.0
        self.cooldown_sec = 14400.0
        self.max_hold_sec = 172800.0
        self.tp_pct = 0.02
        self.sl_pct = 0.02
        self.equity_pct = 0.5
        self.leverage = 7
        self.entry_offset_pct = 0.0002
        self.entry_timeout_sec = 90.0
        self.max_atr_pct_rank = 0.90
        self._last_atr_log = 0.0

    def on_init(self, api):
        self.api = api
        self.coin = api.coin
        self._restore_state()
        self._check_existing_position()

    def on_tick(self, coin, mid_price):
        if coin != self.coin:
            return
        now = self.api.time()
        if now - self.last_check < self.check_sec:
            return
        self.last_check = now

        self._update_macd_history(now)

        if self.in_position:
            self._monitor_position(mid_price, now)
            return

        if now - self.last_trade < self.cooldown_sec:
            return

        ind = self.api.get_indicators(self.coin, "1h", 200, mid_price)
        if not ind or not ind.valid:
            return

        if ind.atr_pct_rank >= self.max_atr_pct_rank:
            if now - self._last_atr_log > 300:
                self.api.log(logging.DEBUG, f"ATR filter: atr_pct_rank={ind.atr_pct_rank:.2f} >= {self.max_atr_pct_rank}")
                self._last_atr_log = now
            return

        signal = self._scan_signals(ind, mid_price)
        if signal:
            self._place_entry(signal, mid_price, now)

    def on_fill(self, fill):
        if fill.closed_pnl.to_float() == 0:
            self._handle_entry_fill(fill)
        else:
            self._handle_exit_fill(fill)

    def on_timer(self):
        if self.in_position:
            now = self.api.time()
            if self.entry_oid and now - self.entry_placed_at > self.entry_timeout_sec:
                self.api.cancel(self.coin, self.entry_oid)
                self.entry_oid = 0

    def on_book(self, book):
        pass

    def on_shutdown(self):
        self._save_state()

    # --- Override these ---

    def _scan_signals(self, ind, mid_price):
        return None

    def _compute_size(self, mid_price):
        account = self.api.get_account_value()
        if account <= 0:
            return 0.0
        dd_mult = self._drawdown_multiplier()
        notional = account * self.equity_pct * self.leverage * dd_mult
        return notional / mid_price if mid_price > 0 else 0.0

    def _drawdown_multiplier(self):
        if self.consec_losses >= 3:
            return 0.25
        if self.consec_losses >= 2:
            return 0.50
        account = self.api.get_account_value()
        if account > 0 and self.peak_equity > 0:
            dd = (self.peak_equity - account) / self.peak_equity
            if dd > 0.20:
                return 0.0
            if dd > 0.15:
                return 0.25
            if dd > 0.10:
                return 0.50
        return 1.0

    # --- Internals ---

    def _place_entry(self, signal, mid_price, now):
        side = signal["side"]
        size = self._compute_size(mid_price)
        if size <= 0:
            return

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
        else:
            tp_px = self.entry_price * (1 - self.tp_pct)
            sl_px = self.entry_price * (1 + self.sl_pct)
            self.tp_oid = self.api.place_trigger(
                self.coin, "buy", tp_px, size, tp_px, tpsl="tp"
            )
            self.sl_oid = self.api.place_trigger(
                self.coin, "buy", sl_px, size, sl_px, tpsl="sl"
            )

        if not self.sl_oid:
            self.api.log(logging.ERROR, "SL trigger order REJECTED — closing position immediately")
            close_side = "sell" if self.position_side == "buy" else "buy"
            self.api.cancel_all(self.coin)
            self.api.place_limit(self.coin, close_side, fill.px.to_float(), size, tif="ioc")
            return

        self.api.log(logging.INFO, f"Position opened: {self.position_side} @ {self.entry_price:.2f}")

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

        account = self.api.get_account_value()
        if account > self.peak_equity:
            self.peak_equity = account

        self.api.log(logging.INFO, f"Position closed: PnL={pnl:.4f} trades={self.trade_count} wins={self.win_count}")
        self._save_state()

    def _monitor_position(self, mid_price, now):
        if self.entry_oid and now - self.entry_placed_at > self.entry_timeout_sec:
            self.api.cancel(self.coin, self.entry_oid)
            self.entry_oid = 0
            self.in_position = False
            return

        # Bot-side SL check (belt-and-suspenders with exchange trigger)
        if self.entry_price > 0 and mid_price > 0:
            if self.position_side == "buy":
                sl_hit = mid_price <= self.entry_price * (1 - self.sl_pct)
            else:
                sl_hit = mid_price >= self.entry_price * (1 + self.sl_pct)

            if sl_hit:
                self.api.log(logging.WARNING,
                             "BOT-SIDE SL HIT: mid=%.4f entry=%.4f sl_pct=%.3f%%" %
                             (mid_price, self.entry_price, self.sl_pct * 100))
                self.api.cancel_all(self.coin)
                pos = self.api.get_position(self.coin)
                if pos and abs(pos.size.to_float()) > 0:
                    size = abs(pos.size.to_float())
                    close_side = "sell" if self.position_side == "buy" else "buy"
                    slippage = 0.005
                    if close_side == "buy":
                        close_px = mid_price * (1 + slippage)
                    else:
                        close_px = mid_price * (1 - slippage)
                    self.api.place_limit(self.coin, close_side, close_px, size, tif="ioc")
                return

        if now - self.entry_time > self.max_hold_sec and self.entry_time > 0:
            self.api.cancel_all(self.coin)
            pos = self.api.get_position(self.coin)
            if pos:
                size = abs(pos.size.to_float())
                close_side = "sell" if self.position_side == "buy" else "buy"
                self.api.place_limit(self.coin, close_side, mid_price, size, tif="ioc")

    def _update_macd_history(self, now):
        hour = int(now // 3600)
        if hour != self.last_hour:
            self.prev2_macd = self.prev_macd
            self.prev_macd = self.last_macd_val
            self.last_hour = hour

    def _check_existing_position(self):
        pos = self.api.get_position(self.coin)
        if pos and abs(pos.size.to_float()) > 0:
            self.in_position = True
            self.entry_price = pos.entry_px.to_float()
            self.position_side = "buy" if pos.size.to_float() > 0 else "sell"
            self.api.log(logging.INFO, f"Existing position found: {self.position_side} @ {self.entry_price}")
            self._reconcile_tpsl(pos)
        else:
            self.in_position = False
            self.sl_oid = 0
            self.tp_oid = 0

    def _reconcile_tpsl(self, pos):
        """Verify TP/SL orders still exist on exchange. Re-place if missing."""
        open_orders = self.api.get_open_orders(self.coin)
        open_oids = {o.oid for o in open_orders}
        size = abs(pos.size.to_float())

        sl_exists = self.sl_oid in open_oids if self.sl_oid else False
        tp_exists = self.tp_oid in open_oids if self.tp_oid else False

        if sl_exists and tp_exists:
            return

        close_side = "sell" if self.position_side == "buy" else "buy"

        if not sl_exists:
            if self.position_side == "buy":
                sl_px = self.entry_price * (1 - self.sl_pct)
            else:
                sl_px = self.entry_price * (1 + self.sl_pct)
            self.sl_oid = self.api.place_trigger(
                self.coin, close_side, sl_px, size, sl_px, tpsl="sl"
            )
            self.api.log(logging.WARNING, f"SL trigger re-placed (oid={self.sl_oid}) @ {sl_px:.2f}")

        if not tp_exists:
            if self.position_side == "buy":
                tp_px = self.entry_price * (1 + self.tp_pct)
            else:
                tp_px = self.entry_price * (1 - self.tp_pct)
            self.tp_oid = self.api.place_trigger(
                self.coin, close_side, tp_px, size, tp_px, tpsl="tp"
            )
            self.api.log(logging.WARNING, f"TP trigger re-placed (oid={self.tp_oid}) @ {tp_px:.2f}")

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
        except (json.JSONDecodeError, KeyError):
            pass
