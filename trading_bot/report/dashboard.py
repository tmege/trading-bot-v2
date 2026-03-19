import logging
import os
import threading
import time

log = logging.getLogger(__name__)

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_YELLOW = "\033[33m"
ANSI_CYAN = "\033[36m"
ANSI_DIM = "\033[2m"
ANSI_CLEAR = "\033[2J\033[H"


class Dashboard:
    def __init__(
        self,
        mid_prices: dict[str, float],
        strategies: list,
        order_manager=None,
        data_manager=None,
        refresh_sec: float = 5.0,
    ):
        self._mid_prices = mid_prices
        self._strategies = strategies
        self._order_manager = order_manager
        self._data_manager = data_manager
        self._refresh_sec = refresh_sec
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            try:
                self._render()
            except Exception:
                pass
            time.sleep(self._refresh_sec)

    def _render(self) -> None:
        lines = [ANSI_CLEAR]
        lines.append(f"{ANSI_BOLD}{ANSI_CYAN}=== Trading Bot v2 ==={ANSI_RESET}")
        lines.append(f"{ANSI_DIM}{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}{ANSI_RESET}")
        lines.append("")

        # Mid prices
        lines.append(f"{ANSI_BOLD}Mid Prices:{ANSI_RESET}")
        for coin, price in sorted(self._mid_prices.items()):
            lines.append(f"  {coin:8s} {price:>14,.4f}")
        lines.append("")

        # Strategies
        lines.append(f"{ANSI_BOLD}Strategies:{ANSI_RESET}")
        for info in self._strategies:
            if not info.instance:
                continue
            status = f"{ANSI_RED}ERRORED{ANSI_RESET}" if info.errored else f"{ANSI_GREEN}OK{ANSI_RESET}"
            inst = info.instance
            pos_str = ""
            if hasattr(inst, "in_position") and inst.in_position:
                side_color = ANSI_GREEN if getattr(inst, "position_side", "") == "buy" else ANSI_RED
                pos_str = f" [{side_color}{getattr(inst, 'position_side', '?').upper()}{ANSI_RESET} @ {getattr(inst, 'entry_price', 0):.4f}]"

            trades = getattr(inst, "trade_count", 0)
            wins = getattr(inst, "win_count", 0)
            wr = f"{wins/trades*100:.0f}%" if trades > 0 else "N/A"

            lines.append(f"  {info.name:30s} {status} T={trades} W={wr}{pos_str}")
        lines.append("")

        # Positions
        lines.append(f"{ANSI_BOLD}Open Positions:{ANSI_RESET}")
        has_pos = False
        for info in self._strategies:
            if not info.instance or not self._order_manager:
                continue
            for coin in info.coins:
                pos = self._order_manager.get_position(info.name, coin)
                if pos and abs(pos.size.to_float()) > 0:
                    has_pos = True
                    sz = pos.size.to_float()
                    side = "LONG" if sz > 0 else "SHORT"
                    color = ANSI_GREEN if sz > 0 else ANSI_RED
                    upnl = pos.unrealized_pnl.to_float()
                    upnl_color = ANSI_GREEN if upnl >= 0 else ANSI_RED
                    lines.append(
                        f"  {coin:8s} {color}{side:5s}{ANSI_RESET} "
                        f"sz={abs(sz):.6f} entry={pos.entry_px.to_float():.4f} "
                        f"uPnL={upnl_color}{upnl:+.4f}{ANSI_RESET}"
                    )

        if not has_pos:
            lines.append(f"  {ANSI_DIM}No open positions{ANSI_RESET}")
        lines.append("")

        # Open orders count
        lines.append(f"{ANSI_BOLD}Open Orders:{ANSI_RESET}")
        order_count = 0
        if self._order_manager:
            for info in self._strategies:
                for coin in info.coins:
                    orders = self._order_manager.get_open_orders(info.name, coin)
                    order_count += len(orders)
        lines.append(f"  Total: {order_count}")
        lines.append("")

        # Fear & Greed
        if self._data_manager:
            fg = self._data_manager.get_fear_greed()
            fg_label = "Extreme Fear" if fg < -0.5 else "Fear" if fg < 0 else "Greed" if fg < 0.5 else "Extreme Greed"
            fg_color = ANSI_RED if fg < 0 else ANSI_GREEN
            lines.append(f"{ANSI_BOLD}Fear & Greed:{ANSI_RESET} {fg_color}{fg:+.2f} ({fg_label}){ANSI_RESET}")

        print("\n".join(lines), flush=True)
