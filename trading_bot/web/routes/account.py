import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Query

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/account", tags=["account"])

_engine = None


def init(engine):
    global _engine
    _engine = engine


@router.get("")
async def get_account():
    if not _engine or not _engine.order_manager or not _engine._strategies:
        return {
            "account_value": 0.0,
            "cumulative_pnl": 0.0,
            "daily_pnl": 0.0,
            "daily_unrealized_pnl": 0.0,
            "daily_fees": 0.0,
            "daily_trades": 0,
            "open_positions": 0,
        }

    # Get account value: prefer live account, fallback to first strategy
    account_value = 0.0
    try:
        if _engine.rest and _engine.config and not _engine.config.mode.paper_trading:
            acct = _engine.rest.get_account(_engine.config.wallet_address)
            account_value = acct.account_value.to_float()
        else:
            account_value = _engine.order_manager.get_account_value(_engine._strategies[0].name)
    except Exception:
        try:
            account_value = _engine.order_manager.get_account_value(_engine._strategies[0].name)
        except Exception:
            account_value = 0.0

    cumulative_pnl = 0.0
    daily_pnl = 0.0
    daily_fees = 0.0
    daily_trades = 0
    daily_unrealized = 0.0
    open_positions = 0

    if _engine.db:
        now = datetime.now(timezone.utc)
        day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        day_start_ms = int(day_start.timestamp() * 1000)

        for info in _engine._strategies:
            daily_pnl += _engine.db.get_daily_pnl(info.name, day_start_ms)
            cumulative_pnl += _engine.db.get_total_pnl(info.name)

        row = _engine.db.fetchone(
            "SELECT COALESCE(SUM(fee), 0) as fees, COUNT(*) as cnt "
            "FROM trades WHERE time_ms >= ?",
            (day_start_ms,),
        )
        if row:
            daily_fees = float(row["fees"])
            daily_trades = int(row["cnt"])

    for info in _engine._strategies:
        for coin in info.coins:
            try:
                pos = _engine.order_manager.get_position(info.name, coin)
                if pos and abs(pos.size.to_float()) > 0:
                    open_positions += 1
                    daily_unrealized += pos.unrealized_pnl.to_float()
            except Exception:
                pass

    return {
        "account_value": round(account_value, 2),
        "cumulative_pnl": round(cumulative_pnl, 4),
        "daily_pnl": round(daily_pnl, 4),
        "daily_unrealized_pnl": round(daily_unrealized, 4),
        "daily_fees": round(daily_fees, 6),
        "daily_trades": daily_trades,
        "open_positions": open_positions,
    }


@router.get("/performance")
async def get_performance():
    from trading_bot.web.services.metrics import compute_performance
    return compute_performance(_engine.db if _engine else None, _engine)


@router.get("/equity-curve")
async def get_equity_curve(days: int = Query(default=30, le=365)):
    from trading_bot.web.services.metrics import compute_equity_curve
    return compute_equity_curve(_engine.db if _engine else None, _engine, days)
