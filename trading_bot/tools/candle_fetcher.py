import logging
import time
import argparse

import httpx

from trading_bot.db import Database

log = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
BATCH_SIZE = 1500
RATE_LIMIT_MS = 200
INTERVAL_5M_MS = 300_000


def fetch_candles(
    coin: str,
    db_path: str = "./data/trading_bot.db",
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> int:
    db = Database(db_path)
    db.open()

    try:
        symbol = f"{coin}USDT"
        total = 0

        if start_ms is None:
            last_time = db.get_max_candle_time(coin, "5m")
            if last_time:
                start_ms = last_time + INTERVAL_5M_MS
                log.info(f"Resuming from {start_ms}")
            else:
                start_ms = 1567296000000

        if end_ms is None:
            end_ms = int(time.time() * 1000)

        client = httpx.Client(timeout=30.0, verify=True)

        try:
            current_start = start_ms

            while current_start < end_ms:
                params = {
                    "symbol": symbol,
                    "interval": "5m",
                    "startTime": current_start,
                    "limit": BATCH_SIZE,
                }

                resp = client.get(BINANCE_KLINES_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

                if not data:
                    break

                rows = []
                for k in data:
                    rows.append((
                        coin,
                        "5m",
                        int(k[0]),
                        float(k[1]),
                        float(k[2]),
                        float(k[3]),
                        float(k[4]),
                        float(k[5]),
                        int(k[8]),
                    ))

                db.insert_candles(rows)
                total += len(rows)

                current_start = int(data[-1][0]) + INTERVAL_5M_MS

                if len(data) < BATCH_SIZE:
                    break

                time.sleep(RATE_LIMIT_MS / 1000.0)

                if total % 10000 == 0:
                    log.info(f"Fetched {total} candles for {coin}")

        finally:
            client.close()

        log.info(f"Total {total} candles fetched for {coin}")
        return total

    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Fetch 5m candles from Binance")
    parser.add_argument("coins", nargs="+", help="Coin symbols (BTC ETH SOL)")
    parser.add_argument("--db", default="./data/trading_bot.db", help="DB path")
    parser.add_argument("--start", type=int, default=None, help="Start timestamp ms")
    parser.add_argument("--end", type=int, default=None, help="End timestamp ms")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    for coin in args.coins:
        log.info(f"Fetching {coin}...")
        count = fetch_candles(coin, args.db, args.start, args.end)
        log.info(f"{coin}: {count} candles")


if __name__ == "__main__":
    main()
