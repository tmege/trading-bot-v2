import logging
import time
import argparse

import httpx

from trading_bot.db import Database

log = logging.getLogger(__name__)

BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
BATCH_SIZE = 1000
RATE_LIMIT_MS = 200


def fetch_funding(
    coin: str,
    db_path: str = "./data/trading_bot.db",
    start_ms: int | None = None,
) -> int:
    db = Database(db_path)
    db.open()

    try:
        symbol = f"{coin}USDT"
        total = 0

        last_time = db.get_max_funding_time(coin)
        if last_time and start_ms is None:
            start_ms = last_time + 1
            log.info(f"Resuming from {start_ms}")

        if start_ms is None:
            start_ms = int(time.time() * 1000) - 365 * 86400 * 1000

        client = httpx.Client(timeout=30.0, verify=True)

        try:
            current_start = start_ms

            while True:
                params = {
                    "symbol": symbol,
                    "startTime": current_start,
                    "limit": BATCH_SIZE,
                }

                resp = client.get(BINANCE_FUNDING_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

                if not data:
                    break

                rows = []
                for f in data:
                    mark_raw = f.get("markPrice", 0)
                    rows.append((
                        coin,
                        int(f["fundingTime"]),
                        float(f["fundingRate"]),
                        float(mark_raw) if mark_raw else 0.0,
                    ))

                db.insert_funding_rates(rows)
                total += len(rows)

                current_start = int(data[-1]["fundingTime"]) + 1

                if len(data) < BATCH_SIZE:
                    break

                time.sleep(RATE_LIMIT_MS / 1000.0)

        finally:
            client.close()

        log.info(f"Total {total} funding rates fetched for {coin}")
        return total

    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Fetch funding rates from Binance")
    parser.add_argument("coins", nargs="+", help="Coin symbols (BTC ETH SOL DOGE)")
    parser.add_argument("--db", default="./data/trading_bot.db", help="DB path")
    parser.add_argument("--start", type=int, default=None, help="Start timestamp ms")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    for coin in args.coins:
        log.info(f"Fetching funding for {coin}...")
        count = fetch_funding(coin, args.db, args.start)
        log.info(f"{coin}: {count} rates")


if __name__ == "__main__":
    main()
