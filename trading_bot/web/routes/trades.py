import csv
import io
import logging

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["trades"])

_engine = None


def init(engine):
    global _engine
    _engine = engine


@router.get("/trades")
async def get_trades(
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    coin: str = Query(default=""),
    strategy: str = Query(default=""),
):
    if not _engine or not _engine.db:
        return {"trades": [], "total": 0}

    where = []
    params = []

    if coin:
        where.append("coin = ?")
        params.append(coin)
    if strategy:
        where.append("strategy = ?")
        params.append(strategy)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    count_row = _engine.db.fetchone(
        f"SELECT COUNT(*) as cnt FROM trades {where_sql}", tuple(params)
    )
    total = int(count_row["cnt"]) if count_row else 0

    params_query = list(params)
    params_query.extend([limit, offset])
    rows = _engine.db.fetchall(
        f"SELECT id, oid, coin, side, price, size, fee, closed_pnl, strategy, time_ms "
        f"FROM trades {where_sql} ORDER BY time_ms DESC LIMIT ? OFFSET ?",
        tuple(params_query),
    )

    return {"trades": [dict(r) for r in rows], "total": total}


@router.get("/positions")
async def get_positions():
    if not _engine or not _engine.order_manager:
        return []

    # Pre-fetch live positions once to avoid repeated API calls
    live_positions = {}
    if _engine.rest and _engine.config:
        try:
            wallet = _engine.config.wallet_address
            if wallet:
                account = _engine.rest.get_account(wallet)
                for p in account.positions:
                    if abs(p.size.to_float()) > 0:
                        live_positions[p.coin] = p
        except Exception:
            log.debug("Failed to fetch live positions from Hyperliquid")

    positions = []
    seen_coins = set()

    # 1. Positions from strategies (paper or live)
    for info in _engine._strategies:
        for coin in info.coins:
            try:
                paper = _engine.order_manager._get_exchange(info.name)
                if paper:
                    pos = paper.get_position(coin)
                elif coin in live_positions:
                    pos = live_positions[coin]
                else:
                    pos = None
            except Exception:
                continue
            if pos and abs(pos.size.to_float()) > 0:
                sz = pos.size.to_float()
                entry = pos.entry_px.to_float()
                upnl = pos.unrealized_pnl.to_float()
                mid = _engine._mid_prices.get(coin, 0)

                notional = abs(entry * sz) if entry > 0 else 0
                roi = (upnl / notional * 100) if notional > 0 else 0

                positions.append({
                    "strategy": info.name,
                    "coin": coin,
                    "side": "LONG" if sz > 0 else "SHORT",
                    "size": abs(sz),
                    "entry_px": entry,
                    "unrealized_pnl": round(upnl, 4),
                    "roi_pct": round(roi, 2),
                    "leverage": pos.leverage,
                    "mid_price": mid,
                })
                seen_coins.add(coin)

    # 2. Live positions from Hyperliquid not managed by any strategy
    for coin, pos in live_positions.items():
        if coin in seen_coins:
            continue
        sz = pos.size.to_float()
        if abs(sz) == 0:
            continue
        entry = pos.entry_px.to_float()
        upnl = pos.unrealized_pnl.to_float()
        mid = _engine._mid_prices.get(coin, 0)

        notional = abs(entry * sz) if entry > 0 else 0
        roi = (upnl / notional * 100) if notional > 0 else 0

        positions.append({
            "strategy": "manual",
            "coin": coin,
            "side": "LONG" if sz > 0 else "SHORT",
            "size": abs(sz),
            "entry_px": entry,
            "unrealized_pnl": round(upnl, 4),
            "roi_pct": round(roi, 2),
            "leverage": pos.leverage,
            "mid_price": mid,
        })

    return positions


@router.get("/trades/export")
async def export_trades(
    coin: str = Query(default=""),
    strategy: str = Query(default=""),
):
    if not _engine or not _engine.db:
        return StreamingResponse(io.BytesIO(b""), media_type="text/csv")

    where = []
    params = []
    if coin:
        where.append("coin = ?")
        params.append(coin)
    if strategy:
        where.append("strategy = ?")
        params.append(strategy)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    rows = _engine.db.fetchall(
        f"SELECT id, coin, side, price, size, fee, closed_pnl, strategy, time_ms "
        f"FROM trades {where_sql} ORDER BY time_ms DESC",
        tuple(params),
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "coin", "side", "price", "size", "fee", "closed_pnl", "strategy", "time_ms"])
    for r in rows:
        writer.writerow([r["id"], r["coin"], r["side"], r["price"], r["size"], r["fee"], r["closed_pnl"], r["strategy"], r["time_ms"]])

    content = output.getvalue().encode("utf-8")
    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trades.csv"},
    )


@router.get("/mids")
async def get_mids():
    if not _engine:
        return {}
    return {k: v for k, v in _engine._mid_prices.items() if not k.startswith("@")}
