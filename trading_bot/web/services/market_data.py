import logging
import time

import httpx

log = logging.getLogger(__name__)

_client: httpx.Client | None = None
_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 120


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=15.0, verify=True)
    return _client


def get_global_market() -> dict:
    cached = _from_cache("global")
    if cached:
        return cached

    try:
        resp = _get_client().get("https://api.coingecko.com/api/v3/global")
        resp.raise_for_status()
        data = resp.json().get("data", {})

        result = {
            "btc_dominance": round(data.get("market_cap_percentage", {}).get("btc", 0), 2),
            "total_market_cap": round(data.get("total_market_cap", {}).get("usd", 0)),
            "total2": 0,
            "total3": 0,
        }

        total = data.get("total_market_cap", {}).get("usd", 0)
        btc_pct = data.get("market_cap_percentage", {}).get("btc", 0) / 100
        eth_pct = data.get("market_cap_percentage", {}).get("eth", 0) / 100
        result["total2"] = round(total * (1 - btc_pct))
        result["total3"] = round(total * (1 - btc_pct - eth_pct))

        _to_cache("global", result)
        return result
    except Exception:
        log.warning("Failed to fetch CoinGecko global data")
        return _from_cache("global") or {
            "btc_dominance": 0, "total_market_cap": 0, "total2": 0, "total3": 0
        }


def get_forex() -> dict:
    cached = _from_cache("forex")
    if cached:
        return cached

    try:
        resp = _get_client().get(
            "https://api.frankfurter.app/latest",
            params={"from": "USD", "to": "EUR,GBP,JPY,CHF"},
        )
        resp.raise_for_status()
        rates_raw = resp.json().get("rates", {})

        rates = {}
        if "EUR" in rates_raw and rates_raw["EUR"] > 0:
            rates["EUR/USD"] = round(1.0 / rates_raw["EUR"], 5)
        if "GBP" in rates_raw and rates_raw["GBP"] > 0:
            rates["GBP/USD"] = round(1.0 / rates_raw["GBP"], 5)
        if "JPY" in rates_raw:
            rates["USD/JPY"] = round(rates_raw["JPY"], 3)
        if "CHF" in rates_raw:
            rates["USD/CHF"] = round(rates_raw["CHF"], 5)

        result = {"rates": rates}
        _to_cache("forex", result)
        return result
    except Exception:
        log.warning("Failed to fetch Frankfurter forex data")
        return _from_cache("forex") or {"rates": {}}


def get_market_phase(db) -> dict:
    if not db:
        return {"phase": "unknown", "confidence": 0, "recommended_strategies": [], "indicators": {}}

    try:
        # Try 1h candles first
        rows = db.fetchall(
            "SELECT close FROM candles WHERE coin='BTC' AND interval='1h' "
            "ORDER BY time_open DESC LIMIT 100",
        )
        # Fallback: aggregate 5m candles to ~1h (12 bars per hour)
        if len(rows) < 50:
            rows_5m = db.fetchall(
                "SELECT close, time_open FROM candles WHERE coin='BTC' AND interval='5m' "
                "ORDER BY time_open DESC LIMIT 1200",
            )
            if len(rows_5m) >= 600:
                # Take every 12th close (hourly samples)
                reversed_5m = list(reversed(rows_5m))
                rows = [{"close": reversed_5m[i]["close"]} for i in range(11, len(reversed_5m), 12)]
                rows = list(reversed(rows))  # back to DESC order
        if len(rows) < 50:
            return {"phase": "unknown", "confidence": 0, "recommended_strategies": [], "indicators": {}}

        closes = [float(r["close"]) for r in reversed(rows)]

        sma20 = sum(closes[-20:]) / 20
        sma50 = sum(closes[-50:]) / 50
        current = closes[-1]

        tr_list = []
        for i in range(1, min(14, len(closes))):
            tr = abs(closes[i] - closes[i - 1])
            tr_list.append(tr)
        atr = sum(tr_list) / len(tr_list) if tr_list else 0
        atr_pct = atr / current * 100 if current > 0 else 0

        if current > sma20 > sma50 and atr_pct < 3:
            phase = "bull"
            confidence = 0.8
            strategies = ["trend_following", "breakout"]
        elif current < sma20 < sma50 and atr_pct < 3:
            phase = "bear"
            confidence = 0.8
            strategies = ["mean_reversion", "short_bias"]
        elif atr_pct > 4:
            phase = "high_vol"
            confidence = 0.7
            strategies = ["scalping", "grid"]
        else:
            phase = "range"
            confidence = 0.6
            strategies = ["mean_reversion", "grid"]

        return {
            "phase": phase,
            "confidence": confidence,
            "recommended_strategies": strategies,
            "indicators": {
                "sma20": round(sma20, 2),
                "sma50": round(sma50, 2),
                "atr_pct": round(atr_pct, 2),
                "price": round(current, 2),
            },
        }
    except Exception:
        log.exception("Error computing market phase")
        return {"phase": "unknown", "confidence": 0, "recommended_strategies": [], "indicators": {}}


def _from_cache(key: str) -> dict | None:
    if key in _cache:
        ts, data = _cache[key]
        if time.time() - ts < _CACHE_TTL:
            return data
    return None


def _to_cache(key: str, data: dict) -> None:
    _cache[key] = (time.time(), data)
