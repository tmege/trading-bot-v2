import logging
import json
from datetime import datetime, timezone

from trading_bot.strategies.template import TemplateStrategy


class BtcMondayRange1h(TemplateStrategy):
    def __init__(self):
        super().__init__()
        self.name = "btc_monday_range_1h"
        self.equity_pct = 0.40
        self.leverage = 5
        self.cooldown_sec = 43200.0
        self.max_hold_sec = 172800.0
        self.tp_pct = 0.03
        self.sl_pct = 0.015
        self.entry_offset_pct = 0.0002
        self.entry_timeout_sec = 90.0

        self.range_high = 0.0
        self.range_low = 0.0
        self.range_valid = False
        self.last_range_week = -1

    def _scan_signals(self, ind, mid_price):
        if not self.range_valid or mid_price <= 0:
            return None

        if not self._sentiment_gate():
            return None

        atr_ratio = ind.atr_14 / mid_price if mid_price > 0 else 1.0
        if atr_ratio >= 0.008:
            return None

        vol_sma = ind.vol_ratio
        if vol_sma < 1.3:
            return None

        if mid_price > self.range_high:
            self.api.log(logging.INFO, f"Monday LONG: {mid_price:.2f} > range_high {self.range_high:.2f}")
            return {"side": "buy", "signal": "MON_LONG"}

        if mid_price < self.range_low:
            self.api.log(logging.INFO, f"Monday SHORT: {mid_price:.2f} < range_low {self.range_low:.2f}")
            return {"side": "sell", "signal": "MON_SHORT"}

        return None

    def _sentiment_gate(self):
        sentiment = self.api.get_sentiment(self.coin)
        if sentiment < self.api.config.sentiment.hard_block_threshold:
            self.api.log(logging.WARNING, f"Trade blocked — sentiment {sentiment:.2f}")
            return False
        return True

    def on_tick(self, coin, mid_price):
        if coin != self.coin:
            return
        now = self.api.time()
        if now - self.last_check < self.check_sec:
            return
        self.last_check = now

        self._update_monday_range(now)

        if self.in_position:
            self._monitor_position(mid_price, now)
            return

        if now - self.last_trade < self.cooldown_sec:
            return

        ind = self.api.get_indicators(self.coin, "1h", 200, mid_price)
        if not ind or not ind.valid:
            return

        signal = self._scan_signals(ind, mid_price)
        if signal:
            self._place_entry(signal, mid_price, now)

    def _update_monday_range(self, now):
        dt = datetime.fromtimestamp(now, tz=timezone.utc)
        week = dt.isocalendar()[1]

        if week == self.last_range_week:
            return

        if dt.weekday() == 0:
            return

        candles = self.api.get_candles(self.coin, "1h", 200)
        if not candles:
            return

        result = self._build_monday_range(candles, now)
        if result:
            self.range_high, self.range_low = result
            self.range_valid = True
            self.last_range_week = week
            self.api.log(logging.INFO,
                         f"Monday range: [{self.range_low:.2f}, {self.range_high:.2f}]")
        else:
            self.range_valid = False

    def _build_monday_range(self, candles, now):
        dt_now = datetime.fromtimestamp(now, tz=timezone.utc)

        monday_candles = []
        for c in candles:
            c_dt = datetime.fromtimestamp(c.time_open / 1000, tz=timezone.utc)
            if c_dt.weekday() == 0:
                monday_candles.append((c_dt, c))

        if not monday_candles:
            return None

        last_monday_date = monday_candles[-1][0].date()

        if last_monday_date == dt_now.date():
            if len(monday_candles) < 2:
                return None
            target_monday = monday_candles[-2][0].date()
        else:
            target_monday = last_monday_date

        # Skip first monday of month
        if target_monday.day <= 7:
            return None

        target_candles = [c for dt, c in monday_candles if dt.date() == target_monday]
        if not target_candles:
            return None

        range_high = max(c.high for c in target_candles)
        range_low = min(c.low for c in target_candles)

        if range_high <= 0:
            return None

        range_pct = (range_high - range_low) / range_high
        if range_pct < 0.003 or range_pct > 0.05:
            return None

        return range_high, range_low

    def _drawdown_multiplier(self):
        account = self.api.get_account_value()
        if account <= 0 or self.peak_equity <= 0:
            return 1.0

        dd = (self.peak_equity - account) / self.peak_equity

        if dd > 0.25:
            self.api.log(logging.WARNING, f"Total DD {dd*100:.1f}% > 25% — STOPPED")
            return 0.0

        weekly_pnl = self.api.get_daily_pnl()
        if account > 0 and weekly_pnl < 0 and abs(weekly_pnl) / account > 0.12:
            return 0.25

        return 1.0
