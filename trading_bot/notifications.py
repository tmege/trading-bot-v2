import logging
import os
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from trading_bot.types import Fill

log = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org"


def _get_telegram_credentials() -> tuple[str, str]:
    token = os.getenv("TB_TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TB_TELEGRAM_CHAT_ID", "")
    return token, chat_id


def send_telegram(message: str) -> bool:
    """Send a Telegram message. Returns True on success."""
    token, chat_id = _get_telegram_credentials()
    if not token or not chat_id:
        log.debug("Telegram not configured (TB_TELEGRAM_BOT_TOKEN / TB_TELEGRAM_CHAT_ID)")
        return False

    try:
        url = f"{_TELEGRAM_API}/bot{token}/sendMessage"
        resp = httpx.post(
            url,
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            log.info("Telegram notification sent")
            return True
        log.warning("Telegram API returned %d", resp.status_code)
        return False
    except Exception:
        log.warning("Failed to send Telegram notification")
        return False


async def send_telegram_async(message: str) -> bool:
    """Send a Telegram message asynchronously. Returns True on success."""
    token, chat_id = _get_telegram_credentials()
    if not token or not chat_id:
        log.debug("Telegram not configured (TB_TELEGRAM_BOT_TOKEN / TB_TELEGRAM_CHAT_ID)")
        return False

    try:
        url = f"{_TELEGRAM_API}/bot{token}/sendMessage"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            )
        if resp.status_code == 200:
            log.info("Telegram notification sent")
            return True
        log.warning("Telegram API returned %d", resp.status_code)
        return False
    except Exception:
        log.warning("Failed to send Telegram notification")
        return False


# --- Trade notification formatting ---


def _fmt_price(px: float) -> str:
    if px >= 10:
        return f"${px:,.2f}"
    return f"${px:,.4f}"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return "<1m"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def format_entry_notification(
    strategy_name: str,
    coin: str,
    side: str,
    px: float,
    sz: float,
    leverage: int,
    sl_pct: float,
    tp_pct: float,
) -> str:
    side_label = "LONG" if side == "buy" else "SHORT"
    notional = px * sz
    return (
        f"<b>{side_label} {coin} @ {_fmt_price(px)}</b>\n"
        f"Stratégie: {strategy_name}\n"
        f"Taille: {sz:g} {coin} (${notional:,.0f})\n"
        f"Levier: {leverage}x | SL: {sl_pct * 100:g}% | TP: {tp_pct * 100:g}%"
    )


def format_exit_notification(
    strategy_name: str,
    coin: str,
    exit_px: float,
    entry_px: float,
    entry_time: float,
    exit_time_ms: int,
    pnl: float,
    fee: float,
    trade_count: int,
    win_count: int,
) -> str:
    pnl_sign = "+" if pnl >= 0 else "-"
    duration = _fmt_duration(exit_time_ms / 1000.0 - entry_time) if entry_time > 0 else "?"
    return (
        f"<b>CLOSE {coin} @ {_fmt_price(exit_px)}</b>\n"
        f"Stratégie: {strategy_name}\n"
        f"Entrée: {_fmt_price(entry_px)} → Sortie: {_fmt_price(exit_px)}\n"
        f"PnL: <b>{pnl_sign}${abs(pnl):,.2f}</b> | Fee: ${fee:,.2f}\n"
        f"Durée: {duration}\n"
        f"Bilan: {trade_count} trades, {win_count} wins"
    )
