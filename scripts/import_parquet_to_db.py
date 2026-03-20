#!/usr/bin/env python3
"""
Import parquet OHLCV data into the SQLite candles table.

Reads parquet files from crypto_bot/data/ and inserts candle data
into the trading_bot.db SQLite database.

Usage:
    python scripts/import_parquet_to_db.py
"""

import re
import sqlite3
import time
from pathlib import Path

import pyarrow.parquet as pq

# -- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PARQUET_DIR = BASE_DIR / "crypto_bot" / "data"
DB_PATH = BASE_DIR / "data" / "trading_bot.db"

# -- Target files ----------------------------------------------------------
TARGET_FILES = [
    "XRP_USDT_5m_ohlcv.parquet",
    "BNB_USDT_5m_ohlcv.parquet",
]

# -- Batch size for executemany --------------------------------------------
BATCH_SIZE = 10_000

# -- Filename pattern: COIN_USDT_INTERVAL_ohlcv.parquet --------------------
FILENAME_PATTERN = re.compile(r"^([A-Z]+)_USDT_(\d+[smhd])_ohlcv\.parquet$")


def parse_filename(filename: str) -> tuple[str, str]:
    """Extract coin and interval from the parquet filename.

    Example: XRP_USDT_5m_ohlcv.parquet -> ("XRP", "5m")
    """
    match = FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    return match.group(1), match.group(2)


def datetime_to_epoch_ms(dt) -> int:
    """Convert a datetime object to epoch milliseconds (integer)."""
    return int(dt.timestamp() * 1000)


def ensure_candles_table(conn: sqlite3.Connection) -> None:
    """Create the candles table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            coin TEXT NOT NULL,
            interval TEXT NOT NULL,
            time_open INTEGER NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            n_trades INTEGER,
            PRIMARY KEY (coin, interval, time_open)
        )
    """)
    conn.commit()


def import_parquet_file(conn: sqlite3.Connection, filepath: Path) -> int:
    """Import a single parquet file into the candles table.

    Returns the number of rows inserted.
    """
    coin, interval = parse_filename(filepath.name)
    print(f"\n[{coin}] Reading {filepath.name} ...")

    table = pq.read_table(str(filepath))
    num_rows = table.num_rows
    print(f"[{coin}] Found {num_rows:,} rows (interval={interval})")

    # Extract columns as pyarrow arrays
    timestamps = table.column("timestamp")
    opens = table.column("open")
    highs = table.column("high")
    lows = table.column("low")
    closes = table.column("close")
    volumes = table.column("volume")

    # Check if n_trades column exists
    has_n_trades = "n_trades" in table.column_names
    n_trades_col = table.column("n_trades") if has_n_trades else None

    insert_sql = """
        INSERT OR IGNORE INTO candles
            (coin, interval, time_open, open, high, low, close, volume, n_trades)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    cursor = conn.cursor()
    total_inserted = 0
    batch = []

    t_start = time.monotonic()

    for i in range(num_rows):
        ts = timestamps[i].as_py()
        time_open_ms = datetime_to_epoch_ms(ts)

        row = (
            coin,
            interval,
            time_open_ms,
            opens[i].as_py(),
            highs[i].as_py(),
            lows[i].as_py(),
            closes[i].as_py(),
            volumes[i].as_py(),
            n_trades_col[i].as_py() if has_n_trades else 0,
        )
        batch.append(row)

        if len(batch) >= BATCH_SIZE:
            cursor.executemany(insert_sql, batch)
            total_inserted += cursor.rowcount
            conn.commit()
            processed = i + 1
            pct = processed / num_rows * 100
            print(f"[{coin}] Progress: {processed:>10,} / {num_rows:,} ({pct:5.1f}%)")
            batch.clear()

    # Insert remaining rows
    if batch:
        cursor.executemany(insert_sql, batch)
        total_inserted += cursor.rowcount
        conn.commit()

    elapsed = time.monotonic() - t_start
    print(f"[{coin}] Done: {total_inserted:,} rows inserted in {elapsed:.1f}s "
          f"({num_rows - total_inserted:,} duplicates skipped)")

    return total_inserted


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("Parquet to SQLite Candle Importer")
    print("=" * 60)
    print(f"Parquet dir : {PARQUET_DIR}")
    print(f"Database    : {DB_PATH}")

    # Validate paths
    if not PARQUET_DIR.is_dir():
        raise FileNotFoundError(f"Parquet directory not found: {PARQUET_DIR}")
    if not DB_PATH.parent.is_dir():
        raise FileNotFoundError(f"Database directory not found: {DB_PATH.parent}")

    # Validate all target files exist before starting
    for filename in TARGET_FILES:
        filepath = PARQUET_DIR / filename
        if not filepath.is_file():
            raise FileNotFoundError(f"Parquet file not found: {filepath}")

    # Connect to database
    conn = sqlite3.connect(str(DB_PATH))
    try:
        ensure_candles_table(conn)

        grand_total = 0
        for filename in TARGET_FILES:
            filepath = PARQUET_DIR / filename
            inserted = import_parquet_file(conn, filepath)
            grand_total += inserted

        print("\n" + "=" * 60)
        print(f"Import complete. Total rows inserted: {grand_total:,}")
        print("=" * 60)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
