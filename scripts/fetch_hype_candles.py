"""Fetch 5m candles for HYPE from Hyperliquid API and store in SQLite.

Hyperliquid candleSnapshot endpoint does not require authentication.
HYPE token is not on Binance, so we fetch directly from Hyperliquid.
"""
import logging
import sys
import time

import httpx

sys.path.insert(0, ".")
from trading_bot.db import Database

log = logging.getLogger(__name__)

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
INTERVAL = "5m"
INTERVAL_MS = 300_000
BATCH_MS = INTERVAL_MS * 500  # ~500 candles per request
RATE_LIMIT_S = 0.3


def fetch_hype_candles(db_path: str = "./data/trading_bot.db") -> int:
    db = Database(db_path)
    db.open()

    try:
        coin = "HYPE"
        total = 0

        # Resume from last candle if exists
        last_time = db.get_max_candle_time(coin, INTERVAL)
        if last_time:
            start_ms = last_time + INTERVAL_MS
            log.info(f"Resuming from {start_ms}")
        else:
            # HYPE token launched ~late Nov 2024
            start_ms = 1732492800000  # 2024-11-25 00:00:00 UTC
            log.info(f"Starting fresh from {start_ms}")

        end_ms = int(time.time() * 1000)

        client = httpx.Client(timeout=30.0, verify=True)
        try:
            current_start = start_ms

            while current_start < end_ms:
                current_end = min(current_start + BATCH_MS, end_ms)

                payload = {
                    "type": "candleSnapshot",
                    "req": {
                        "coin": coin,
                        "interval": INTERVAL,
                        "startTime": current_start,
                        "endTime": current_end,
                    },
                }

                resp = client.post(HL_INFO_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()

                if not isinstance(data, list) or not data:
                    log.info(f"No data at {current_start}, advancing...")
                    current_start = current_end
                    time.sleep(RATE_LIMIT_S)
                    continue

                rows = []
                for c in data:
                    rows.append((
                        coin,
                        INTERVAL,
                        int(c["t"]),
                        float(c["o"]),
                        float(c["h"]),
                        float(c["l"]),
                        float(c["c"]),
                        float(c["v"]),
                        int(c.get("n", 0)),
                    ))

                db.insert_candles(rows)
                total += len(rows)

                # Advance past last candle
                last_t = max(int(c["t"]) for c in data)
                current_start = last_t + INTERVAL_MS

                if total % 5000 == 0:
                    log.info(f"Fetched {total} candles for {coin}")

                time.sleep(RATE_LIMIT_S)

        finally:
            client.close()

        log.info(f"Total {total} candles fetched for {coin}")
        return total

    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    count = fetch_hype_candles()
    print(f"\nDone: {count} candles fetched for HYPE")
