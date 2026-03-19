import logging

log = logging.getLogger(__name__)


def compute_alerts(
    account_value: float,
    daily_pnl: float,
    pnl_7d: float,
    wr_7d: float,
    fee_drag: float,
    trades_per_day: float,
    max_dd: float,
    engine=None,
) -> list[dict]:
    alerts = []

    if account_value > 0 and pnl_7d != 0:
        weekly_pct = pnl_7d / account_value * 100
        if weekly_pct < -8:
            alerts.append({
                "level": "red",
                "message": f"Weekly PnL: {weekly_pct:+.1f}% (< -8%)",
                "metric": "pnl_7d",
            })

    if account_value > 0 and daily_pnl != 0:
        daily_pct = daily_pnl / account_value * 100
        if daily_pct < -4:
            alerts.append({
                "level": "yellow",
                "message": f"Daily PnL: {daily_pct:+.1f}% (< -4%)",
                "metric": "daily_pnl",
            })

    if engine and engine._strategies:
        for info in engine._strategies:
            inst = info.instance
            if not inst:
                continue
            trades = getattr(inst, "trade_count", 0)
            wins = getattr(inst, "win_count", 0)
            if trades >= 5:
                wr = wins / trades * 100
                if wr < 30:
                    alerts.append({
                        "level": "yellow",
                        "message": f"{info.name}: WR {wr:.0f}% (< 30%)",
                        "metric": "strategy_wr",
                    })

    if fee_drag > 25:
        alerts.append({
            "level": "yellow",
            "message": f"Fee drag: {fee_drag:.1f}% (> 25%)",
            "metric": "fee_drag",
        })

    if trades_per_day > 15:
        alerts.append({
            "level": "yellow",
            "message": f"Trades/day: {trades_per_day:.1f} (> 15)",
            "metric": "trades_per_day",
        })

    if max_dd > 15:
        alerts.append({
            "level": "red",
            "message": f"Max DD: {max_dd:.1f}% (> 15%) — STOP RECOMMENDED",
            "metric": "max_dd",
        })

    return alerts
