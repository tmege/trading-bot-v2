import logging
import math
from datetime import datetime, timezone

log = logging.getLogger(__name__)


def compute_performance(db, engine) -> dict:
    if not db:
        return _defaults()

    try:
        now = datetime.now(timezone.utc)
        day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        day_start_ms = int(day_start.timestamp() * 1000)
        seven_days_ms = day_start_ms - 7 * 86400 * 1000
        thirty_days_ms = day_start_ms - 30 * 86400 * 1000

        account_value = _get_account_value(engine)

        sharpe = _sharpe_30d(db, thirty_days_ms)
        max_dd = _max_drawdown(db, engine)
        fee_drag = _fee_drag(db)
        wr_7d = _win_rate_7d(db, seven_days_ms)
        trades_per_day = _trades_per_day(db)
        pnl_7d = _pnl_7d(db, seven_days_ms)

        from trading_bot.web.services.alerts import compute_alerts
        alerts = compute_alerts(
            account_value=account_value,
            daily_pnl=_daily_pnl(db, day_start_ms),
            pnl_7d=pnl_7d,
            wr_7d=wr_7d,
            fee_drag=fee_drag,
            trades_per_day=trades_per_day,
            max_dd=max_dd,
            engine=engine,
        )

        return {
            "sharpe_30d": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd, 2),
            "fee_drag_pct": round(fee_drag, 2),
            "win_rate_7d": round(wr_7d, 2),
            "trades_per_day": round(trades_per_day, 2),
            "pnl_7d": round(pnl_7d, 4),
            "alerts": alerts,
        }
    except Exception:
        log.exception("Error computing performance metrics")
        return _defaults()


def compute_equity_curve(db, engine, days: int = 30) -> list[dict]:
    if not db:
        return []

    try:
        cutoff_ms = int(
            (datetime.now(timezone.utc).timestamp() - days * 86400) * 1000
        )
        rows = db.fetchall(
            "SELECT time_ms, closed_pnl, fee FROM trades "
            "WHERE time_ms >= ? ORDER BY time_ms",
            (cutoff_ms,),
        )
        if not rows:
            return []

        initial = _get_account_value(engine)
        base = initial - sum(float(r["closed_pnl"]) - float(r["fee"]) for r in rows)

        curve = []
        equity = base
        peak = base
        for r in rows:
            equity += float(r["closed_pnl"]) - float(r["fee"])
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            curve.append({
                "time_ms": r["time_ms"],
                "equity": round(equity, 4),
                "drawdown": round(dd, 4),
            })

        if len(curve) > 500:
            step = len(curve) // 500
            curve = curve[::step] + [curve[-1]]

        return curve
    except Exception:
        log.exception("Error computing equity curve")
        return []


def _get_account_value(engine) -> float:
    if not engine.order_manager or not engine._strategies:
        return 0.0
    try:
        return engine.order_manager.get_account_value(engine._strategies[0].name)
    except Exception:
        return 0.0


def _daily_pnl(db, day_start_ms: int) -> float:
    row = db.fetchone(
        "SELECT COALESCE(SUM(closed_pnl - fee), 0) as pnl FROM trades WHERE time_ms >= ?",
        (day_start_ms,),
    )
    return float(row["pnl"]) if row else 0.0


def _sharpe_30d(db, thirty_days_ms: int) -> float:
    rows = db.fetchall(
        "SELECT (time_ms / 86400000) as day, SUM(closed_pnl - fee) as daily_return "
        "FROM trades WHERE time_ms >= ? GROUP BY day ORDER BY day",
        (thirty_days_ms,),
    )
    if len(rows) < 2:
        return 0.0

    returns = [float(r["daily_return"]) for r in rows]
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    std = math.sqrt(var) if var > 0 else 0
    if std == 0:
        return 0.0
    return math.sqrt(365) * mean_r / std


def _max_drawdown(db, engine) -> float:
    rows = db.fetchall(
        "SELECT closed_pnl, fee FROM trades ORDER BY time_ms",
    )
    if not rows:
        return 0.0

    initial = _get_account_value(engine)
    base = initial - sum(float(r["closed_pnl"]) - float(r["fee"]) for r in rows)

    equity = base
    peak = base
    max_dd = 0.0
    for r in rows:
        equity += float(r["closed_pnl"]) - float(r["fee"])
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return max_dd * 100


def _fee_drag(db) -> float:
    row = db.fetchone(
        "SELECT COALESCE(SUM(fee), 0) as total_fee, "
        "COALESCE(SUM(CASE WHEN closed_pnl > 0 THEN closed_pnl ELSE 0 END), 0) as gross_profit "
        "FROM trades",
    )
    if not row:
        return 0.0
    total_fee = float(row["total_fee"])
    gross_profit = float(row["gross_profit"])
    if gross_profit <= 0:
        return 0.0
    return total_fee / gross_profit * 100


def _win_rate_7d(db, seven_days_ms: int) -> float:
    row = db.fetchone(
        "SELECT "
        "COUNT(CASE WHEN closed_pnl > 0 THEN 1 END) as wins, "
        "COUNT(CASE WHEN closed_pnl != 0 THEN 1 END) as exits "
        "FROM trades WHERE time_ms >= ?",
        (seven_days_ms,),
    )
    if not row or not row["exits"]:
        return 0.0
    return float(row["wins"]) / float(row["exits"]) * 100


def _trades_per_day(db) -> float:
    row = db.fetchone(
        "SELECT COUNT(CASE WHEN closed_pnl != 0 THEN 1 END) as exits, "
        "MIN(time_ms) as first_ms, MAX(time_ms) as last_ms "
        "FROM trades",
    )
    if not row or not row["exits"] or not row["first_ms"]:
        return 0.0
    span_days = max((float(row["last_ms"]) - float(row["first_ms"])) / 86400000, 1)
    return float(row["exits"]) / span_days


def _pnl_7d(db, seven_days_ms: int) -> float:
    row = db.fetchone(
        "SELECT COALESCE(SUM(closed_pnl - fee), 0) as pnl FROM trades WHERE time_ms >= ?",
        (seven_days_ms,),
    )
    return float(row["pnl"]) if row else 0.0


def _defaults() -> dict:
    return {
        "sharpe_30d": 0.0,
        "max_drawdown_pct": 0.0,
        "fee_drag_pct": 0.0,
        "win_rate_7d": 0.0,
        "trades_per_day": 0.0,
        "pnl_7d": 0.0,
        "alerts": [],
    }
