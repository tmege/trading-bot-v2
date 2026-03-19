import logging


class SolTest1USD:
    """One-shot test: market buy $1 of SOL, set TP +2% / SL -1%, then stop."""

    def __init__(self):
        self.name = "sol_test_1usd"
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
        self.last_trade = 0.0

        # --- Config ---
        self.notional_usd = 1.0     # $1 trade
        self.tp_pct = 0.02           # +2% take profit
        self.sl_pct = 0.01           # -1% stop loss
        self.leverage = 1            # no leverage
        self.done = False            # True after the trade completes
        self.entry_sent = False      # True once entry order placed

    def on_init(self, api):
        self.api = api
        self.coin = api.coin
        self._restore_state()

        # Check if we already have a position from a previous run
        pos = api.get_position(self.coin)
        if pos and abs(pos.size.to_float()) > 0:
            self.in_position = True
            self.entry_price = pos.entry_px.to_float()
            self.position_side = "buy" if pos.size.to_float() > 0 else "sell"
            api.log(logging.INFO, f"[TEST] Existing position found: {self.position_side} @ {self.entry_price}")

        if self.done:
            api.log(logging.INFO, "[TEST] Test already completed — strategy idle")

    def on_tick(self, coin, mid_price):
        if coin != self.coin or self.done or self.in_position or self.entry_sent:
            return

        # Place one market buy for $1
        size = self.notional_usd / mid_price
        if size <= 0:
            return

        # IOC = immediate-or-cancel → acts as market order
        try:
            oid = self.api.place_limit(
                self.coin, "buy", mid_price, size, tif="ioc"
            )
        except Exception as e:
            self.api.log(logging.WARNING, f"[TEST] Order failed: {e}")
            return

        if oid:
            self.entry_oid = oid
            self.entry_sent = True
            self.entry_placed_at = self.api.time()
            self.position_side = "buy"
            self.api.log(
                logging.INFO,
                f"[TEST] Market buy placed: {size:.6f} SOL @ ${mid_price:.2f} (~${self.notional_usd})",
            )
        else:
            self.api.log(logging.WARNING, "[TEST] Order returned oid=0, will retry next tick")

    def on_fill(self, fill):
        pnl = fill.closed_pnl.to_float()
        if pnl == 0:
            self._handle_entry_fill(fill)
        else:
            self._handle_exit_fill(fill)

    def on_timer(self):
        # If entry was sent but never filled after 30s, cancel and retry next tick
        if self.entry_sent and not self.in_position and self.entry_oid:
            now = self.api.time()
            if now - self.entry_placed_at > 30:
                self.api.cancel(self.coin, self.entry_oid)
                self.entry_oid = 0
                self.entry_sent = False
                self.api.log(logging.WARNING, "[TEST] Entry timed out — will retry")

    def on_book(self, book):
        pass

    def on_shutdown(self):
        self._save_state()

    # --- Internals ---

    def _handle_entry_fill(self, fill):
        self.in_position = True
        self.entry_price = fill.px.to_float()
        self.entry_time = fill.time_ms / 1000.0
        self.entry_oid = 0
        size = fill.sz.to_float()

        tp_px = self.entry_price * (1 + self.tp_pct)
        sl_px = self.entry_price * (1 - self.sl_pct)

        self.tp_oid = self.api.place_trigger(
            self.coin, "sell", tp_px, size, tp_px, tpsl="tp"
        )
        self.sl_oid = self.api.place_trigger(
            self.coin, "sell", sl_px, size, sl_px, tpsl="sl"
        )

        self.api.log(
            logging.INFO,
            f"[TEST] Position opened: BUY {size:.6f} SOL @ ${self.entry_price:.2f} "
            f"| TP: ${tp_px:.2f} (+{self.tp_pct*100:.0f}%) "
            f"| SL: ${sl_px:.2f} (-{self.sl_pct*100:.0f}%)",
        )

    def _handle_exit_fill(self, fill):
        pnl = fill.closed_pnl.to_float()
        self.trade_count += 1
        if pnl > 0:
            self.win_count += 1

        self.api.cancel_all(self.coin)
        self.in_position = False
        self.done = True
        self.entry_oid = 0
        self.sl_oid = 0
        self.tp_oid = 0
        self.last_trade = self.api.time()

        self.api.log(
            logging.INFO,
            f"[TEST] Position CLOSED — PnL: ${pnl:.6f} {'WIN' if pnl > 0 else 'LOSS'} "
            f"| Test complete. Strategy now idle.",
        )
        self._save_state()

    def _save_state(self):
        import json
        state = {
            "in_position": self.in_position,
            "position_side": self.position_side,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "done": self.done,
            "entry_sent": self.entry_sent,
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
            s = json.loads(raw)
            self.in_position = s.get("in_position", False)
            self.position_side = s.get("position_side", "")
            self.entry_price = s.get("entry_price", 0.0)
            self.entry_time = s.get("entry_time", 0.0)
            self.trade_count = s.get("trade_count", 0)
            self.win_count = s.get("win_count", 0)
            self.done = s.get("done", False)
            self.entry_sent = s.get("entry_sent", False)
            self.sl_oid = s.get("sl_oid", 0)
            self.tp_oid = s.get("tp_oid", 0)
        except (json.JSONDecodeError, KeyError):
            pass
