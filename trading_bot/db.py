import logging
import sqlite3
import threading
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = """
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
);

CREATE TABLE IF NOT EXISTS funding_rates (
    coin TEXT NOT NULL,
    time_ms INTEGER NOT NULL,
    rate REAL NOT NULL,
    mark_price REAL NOT NULL,
    PRIMARY KEY (coin, time_ms)
);

CREATE INDEX IF NOT EXISTS idx_fr_lookup ON funding_rates(coin, time_ms);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    oid INTEGER,
    tid INTEGER,
    coin TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL,
    fee REAL NOT NULL,
    closed_pnl REAL NOT NULL,
    strategy TEXT NOT NULL,
    time_ms INTEGER NOT NULL,
    hash TEXT
);

CREATE TABLE IF NOT EXISTS strategy_state (
    strategy TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    PRIMARY KEY (strategy, key)
);

CREATE TABLE IF NOT EXISTS order_strategy_map (
    oid INTEGER PRIMARY KEY,
    strategy TEXT NOT NULL,
    coin TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_history (
    run_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    coin TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    return_pct REAL,
    sharpe REAL,
    max_dd REAL,
    win_rate REAL,
    total_trades INTEGER,
    profit_factor REAL,
    verdict TEXT,
    config_json TEXT,
    result_json TEXT
);
"""


class Database:
    def __init__(self, db_path: str):
        resolved = Path(db_path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(resolved)
        self.conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def open(self) -> None:
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        # M-10: Auto-checkpoint WAL to prevent unbounded file growth
        self.conn.execute("PRAGMA wal_autocheckpoint=1000")
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        log.info(f"Database opened: {self.path}")

    def _create_tables(self) -> None:
        assert self.conn is not None
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
            log.info("Database closed")

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        assert self.conn is not None
        with self._lock:
            return self.conn.execute(sql, params)

    def executemany(self, sql: str, params_list: list[tuple]) -> None:
        assert self.conn is not None
        with self._lock:
            self.conn.executemany(sql, params_list)

    def commit(self) -> None:
        assert self.conn is not None
        with self._lock:
            self.conn.commit()

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        assert self.conn is not None
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        assert self.conn is not None
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    # --- Strategy state ---

    def save_state(self, strategy: str, key: str, value: str) -> None:
        self.execute(
            "INSERT OR REPLACE INTO strategy_state (strategy, key, value) VALUES (?, ?, ?)",
            (strategy, key, value)
        )
        self.commit()

    def load_state(self, strategy: str, key: str) -> str | None:
        row = self.fetchone(
            "SELECT value FROM strategy_state WHERE strategy=? AND key=?",
            (strategy, key)
        )
        return row["value"] if row else None

    # --- Order strategy map ---

    def map_order(self, oid: int, strategy: str, coin: str, created_ms: int) -> None:
        self.execute(
            "INSERT OR REPLACE INTO order_strategy_map (oid, strategy, coin, created_ms) VALUES (?, ?, ?, ?)",
            (oid, strategy, coin, created_ms)
        )
        self.commit()

    def get_order_strategy(self, oid: int) -> str | None:
        row = self.fetchone(
            "SELECT strategy FROM order_strategy_map WHERE oid=?",
            (oid,)
        )
        return row["strategy"] if row else None

    def cleanup_old_orders(self, max_age_ms: int, max_entries: int = 2048) -> None:
        import time
        cutoff = int(time.time() * 1000) - max_age_ms
        self.execute(
            "DELETE FROM order_strategy_map WHERE created_ms < ?",
            (cutoff,)
        )
        count_row = self.fetchone("SELECT COUNT(*) as cnt FROM order_strategy_map")
        if count_row and count_row["cnt"] > max_entries:
            self.execute(
                "DELETE FROM order_strategy_map WHERE oid NOT IN "
                "(SELECT oid FROM order_strategy_map ORDER BY created_ms DESC LIMIT ?)",
                (max_entries,)
            )
        self.commit()

    # --- Trades ---

    def insert_trade(
        self, oid: int, tid: int, coin: str, side: str,
        price: float, size: float, fee: float, closed_pnl: float,
        strategy: str, time_ms: int, hash_val: str = ""
    ) -> None:
        self.execute(
            "INSERT INTO trades (oid, tid, coin, side, price, size, fee, closed_pnl, strategy, time_ms, hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (oid, tid, coin, side, price, size, fee, closed_pnl, strategy, time_ms, hash_val)
        )
        self.commit()

    def get_daily_pnl(self, strategy: str, day_start_ms: int) -> float:
        row = self.fetchone(
            "SELECT COALESCE(SUM(closed_pnl - fee), 0) as pnl FROM trades "
            "WHERE strategy=? AND time_ms>=?",
            (strategy, day_start_ms)
        )
        return float(row["pnl"]) if row else 0.0

    def get_total_pnl(self, strategy: str) -> float:
        row = self.fetchone(
            "SELECT COALESCE(SUM(closed_pnl - fee), 0) as pnl FROM trades WHERE strategy=?",
            (strategy,)
        )
        return float(row["pnl"]) if row else 0.0

    # --- Candles ---

    def get_max_candle_time(self, coin: str, interval: str) -> int | None:
        row = self.fetchone(
            "SELECT MAX(time_open) as t FROM candles WHERE coin=? AND interval=?",
            (coin, interval)
        )
        return row["t"] if row and row["t"] is not None else None

    def insert_candles(self, rows: list[tuple]) -> None:
        self.executemany(
            "INSERT OR IGNORE INTO candles (coin, interval, time_open, open, high, low, close, volume, n_trades) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows
        )
        self.commit()

    def get_candles(
        self, coin: str, interval: str, limit: int = 300, end_ms: int | None = None
    ) -> list[sqlite3.Row]:
        if end_ms:
            return self.fetchall(
                "SELECT * FROM candles WHERE coin=? AND interval=? AND time_open<=? "
                "ORDER BY time_open DESC LIMIT ?",
                (coin, interval, end_ms, limit)
            )[::-1]
        return self.fetchall(
            "SELECT * FROM candles WHERE coin=? AND interval=? "
            "ORDER BY time_open DESC LIMIT ?",
            (coin, interval, limit)
        )[::-1]

    # --- Funding rates ---

    def get_max_funding_time(self, coin: str) -> int | None:
        row = self.fetchone(
            "SELECT MAX(time_ms) as t FROM funding_rates WHERE coin=?",
            (coin,)
        )
        return row["t"] if row and row["t"] is not None else None

    def insert_funding_rates(self, rows: list[tuple]) -> None:
        self.executemany(
            "INSERT OR IGNORE INTO funding_rates (coin, time_ms, rate, mark_price) "
            "VALUES (?, ?, ?, ?)",
            rows
        )
        self.commit()

    def get_funding_rate_at(self, coin: str, time_ms: int) -> tuple[float, float] | None:
        row = self.fetchone(
            "SELECT rate, mark_price FROM funding_rates "
            "WHERE coin=? AND time_ms<=? ORDER BY time_ms DESC LIMIT 1",
            (coin, time_ms)
        )
        return (float(row["rate"]), float(row["mark_price"])) if row else None
