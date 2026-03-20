#!/usr/bin/env python3
"""
Telecharge les donnees 5m XRP/USDT et BNB/USDT depuis Binance Futures.
Periode: 2022-01-01 -> 2026-01-01 (4 ans, comme BTC/ETH/SOL).
Sauvegarde en parquet dans data/.
"""
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SYMBOLS = ["XRP/USDT:USDT", "BNB/USDT:USDT"]
TIMEFRAME = "5m"
SINCE = "2022-01-01"
UNTIL = "2026-03-20"  # aujourd'hui
LIMIT = 1500


def to_ms(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_symbol(exchange, symbol, since_str, until_str):
    since_ms = to_ms(since_str)
    until_ms = to_ms(until_str)
    all_candles = []

    log.info("Telechargement %s %s [%s -> %s]", symbol, TIMEFRAME, since_str, until_str)

    while since_ms < until_ms:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, TIMEFRAME, since=since_ms, limit=LIMIT
            )
        except ccxt.BaseError as e:
            log.error("Erreur ccxt: %s — retry dans 5s", e)
            time.sleep(5)
            continue

        if not candles:
            break

        all_candles.extend(candles)
        last_ts = candles[-1][0]

        if last_ts == since_ms:
            break
        since_ms = last_ts + 1

        if len(all_candles) % 10000 < LIMIT:
            dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
            log.info("  %s — %d bougies (-> %s)", symbol, len(all_candles), dt.strftime("%Y-%m-%d"))

        # Rate limit
        time.sleep(0.1)

    if not all_candles:
        log.error("Aucune bougie pour %s", symbol)
        return None

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Filtrer
    start_dt = pd.Timestamp(since_str, tz="UTC")
    end_dt = pd.Timestamp(until_str, tz="UTC")
    df = df.loc[start_dt:end_dt]

    return df


def main():
    exchange = ccxt.binanceusdm({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    exchange.load_markets()
    log.info("Binance Futures connecte — %d marches", len(exchange.markets))

    for symbol in SYMBOLS:
        if symbol not in exchange.markets:
            log.error("%s non disponible sur Binance Futures !", symbol)
            continue

        t0 = time.time()
        df = fetch_symbol(exchange, symbol, SINCE, UNTIL)
        elapsed = time.time() - t0

        if df is None or df.empty:
            continue

        # Sauvegarder — nom standardise (XRP_USDT, pas XRP_USDT:USDT)
        base = symbol.split(":")[0]  # "XRP/USDT:USDT" -> "XRP/USDT"
        safe = base.replace("/", "_")
        path = DATA_DIR / f"{safe}_{TIMEFRAME}_ohlcv.parquet"
        df.to_parquet(path)

        log.info(
            "%s termine — %d bougies [%s -> %s] en %.0fs -> %s",
            symbol, len(df),
            df.index[0].strftime("%Y-%m-%d"),
            df.index[-1].strftime("%Y-%m-%d"),
            elapsed, path,
        )

    log.info("Done !")


if __name__ == "__main__":
    main()
