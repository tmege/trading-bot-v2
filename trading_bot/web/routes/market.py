import logging
import re

from fastapi import APIRouter, HTTPException, Query

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/market", tags=["market"])

_engine = None


def init(engine):
    global _engine
    _engine = engine


@router.get("/fear-greed")
async def get_fear_greed():
    if not _engine or not _engine.data_manager:
        return {"value": 50, "label": "N/A", "normalized": 0.0}
    normalized = _engine.data_manager.get_fear_greed()
    value = int(normalized * 50 + 50)
    if value < 25:
        label = "Extreme Fear"
    elif value < 45:
        label = "Fear"
    elif value < 55:
        label = "Neutral"
    elif value < 75:
        label = "Greed"
    else:
        label = "Extreme Greed"
    return {"value": value, "label": label, "normalized": normalized}


@router.get("/candles")
async def get_candles(
    coin: str = Query(...),
    interval: str = Query(default="1h"),
    limit: int = Query(default=100, le=500),
):
    if not re.match(r'^[A-Z]{1,10}$', coin):
        raise HTTPException(status_code=400, detail="Invalid coin format")
    if interval not in ("1m", "5m", "15m", "1h", "4h", "1d"):
        raise HTTPException(status_code=400, detail="Invalid interval")
    if not _engine or not _engine.db:
        return []
    rows = _engine.db.get_candles(coin, interval, limit)
    return [
        {
            "time_open": r["time_open"],
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["volume"],
        }
        for r in rows
    ]


@router.get("/global")
async def get_global():
    from trading_bot.web.services.market_data import get_global_market
    return get_global_market()


@router.get("/phase")
async def get_phase():
    from trading_bot.web.services.market_data import get_market_phase
    return get_market_phase(_engine.db if _engine else None)


@router.get("/overview")
async def get_overview():
    """Combined endpoint: market phase + fear & greed index."""
    from trading_bot.web.services.market_data import get_market_phase

    phase_data = get_market_phase(_engine.db if _engine else None)

    # Fear & greed
    fg_value = 50
    fg_label = "N/A"
    if _engine and _engine.data_manager:
        normalized = _engine.data_manager.get_fear_greed()
        fg_value = int(normalized * 50 + 50)
        if fg_value < 25:
            fg_label = "Extreme Fear"
        elif fg_value < 45:
            fg_label = "Fear"
        elif fg_value < 55:
            fg_label = "Neutral"
        elif fg_value < 75:
            fg_label = "Greed"
        else:
            fg_label = "Extreme Greed"

    return {
        "phase": phase_data.get("phase", "unknown"),
        "fear_greed": fg_value,
        "fear_greed_label": fg_label,
        "recommended_strategies": phase_data.get("recommended_strategies", []),
        "indicators": phase_data.get("indicators", {}),
    }


@router.get("/volumes")
async def get_volumes():
    """Return 24h notional volume per coin from asset contexts."""
    if not _engine or not _engine._asset_ctxs:
        return {}
    return {
        coin: ctx.day_ntl_vlm
        for coin, ctx in _engine._asset_ctxs.items()
        if ctx.day_ntl_vlm > 0
    }


@router.get("/digest")
async def get_digest():
    from trading_bot.web.services.digest import get_digest as _get_digest
    model = "claude-haiku-4-5-20251001"
    if _engine and _engine.config:
        model = getattr(_engine.config.sentiment, "claude_model", model)
    return _get_digest(model)


@router.post("/digest/refresh")
async def refresh_digest():
    from trading_bot.web.services.digest import clear_cache, get_digest as _get_digest
    clear_cache()
    model = "claude-haiku-4-5-20251001"
    if _engine and _engine.config:
        model = getattr(_engine.config.sentiment, "claude_model", model)
    return _get_digest(model)
