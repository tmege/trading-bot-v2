"""
Module 1 — DataLoader
Télécharge les bougies 5min via ccxt (Binance Futures), cache en parquet.
Resample vers 1h et 4h. Ajoute colonnes dérivées.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DataLoader:
    """Télécharge, nettoie et cache les données OHLCV + funding rates."""

    # Mapping resample : timeframe cible → nombre de bougies 5min
    RESAMPLE_MAP = {
        "5m": 1,
        "1h": 12,
        "4h": 48,
    }

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.assets: list[str] = self.cfg["assets"]
        self.base_tf: str = self.cfg["base_timeframe"]
        self.timeframes: list[str] = self.cfg["timeframes"]
        self.period_start: str = self.cfg["period_start"]
        self.period_end: str = self.cfg["period_end"]

        self.cache_dir = Path(self.cfg["data"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_max_age_h: int = self.cfg["data"]["cache_max_age_hours"]
        self.max_gap: int = self.cfg["data"]["max_gap_candles"]

        self.funding_default: float = self.cfg["funding"]["default_rate"] / 100
        self.funding_interval_h: int = self.cfg["funding"]["interval_hours"]

        self.exchange = self._init_exchange()
        logger.info("DataLoader initialisé — %d assets, base_tf=%s",
                     len(self.assets), self.base_tf)

    # ── Exchange ──────────────────────────────────────────────

    def _init_exchange(self) -> ccxt.binance:
        exchange = ccxt.binanceusdm({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        try:
            exchange.load_markets()
            logger.info("Binance Futures connecté — %d marchés chargés",
                        len(exchange.markets))
        except ccxt.BaseError as e:
            logger.warning("Impossible de charger les marchés Binance: %s", e)
        return exchange

    # ── Timestamp helpers ─────────────────────────────────────

    @staticmethod
    def _to_ms(date_str: str) -> int:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        return int(dt.timestamp() * 1000)

    @staticmethod
    def _ms_to_dt(ms: int) -> datetime:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

    # ── Cache ─────────────────────────────────────────────────

    def _cache_path(self, symbol: str, timeframe: str,
                    kind: str = "ohlcv") -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self.cache_dir / f"{safe_symbol}_{timeframe}_{kind}.parquet"

    def _cache_is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age_h = (time.time() - path.stat().st_mtime) / 3600
        return age_h < self.cache_max_age_h

    # ── Téléchargement OHLCV ──────────────────────────────────

    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    since: str, until: str) -> pd.DataFrame:
        """Télécharge les bougies OHLCV via ccxt avec pagination.
        Cache en parquet. Retourne un DataFrame indexé par timestamp."""

        cache = self._cache_path(symbol, timeframe)
        if self._cache_is_fresh(cache):
            logger.info("Cache hit: %s", cache.name)
            return pd.read_parquet(cache)

        logger.info("Téléchargement %s %s [%s → %s]",
                     symbol, timeframe, since, until)

        since_ms = self._to_ms(since)
        until_ms = self._to_ms(until)
        all_candles: list[list] = []
        limit = 1500  # max Binance

        while since_ms < until_ms:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ms, limit=limit
                )
            except ccxt.BaseError as e:
                logger.error("Erreur ccxt fetch_ohlcv %s: %s", symbol, e)
                break

            if not candles:
                break

            all_candles.extend(candles)
            last_ts = candles[-1][0]

            if last_ts == since_ms:
                break
            since_ms = last_ts + 1

            # Log progression tous les 10k candles
            if len(all_candles) % 10000 < limit:
                logger.info("  %s %s — %d bougies téléchargées (→ %s)",
                            symbol, timeframe, len(all_candles),
                            self._ms_to_dt(last_ts).strftime("%Y-%m-%d"))

        if not all_candles:
            logger.warning("Aucune bougie récupérée pour %s %s", symbol, timeframe)
            return pd.DataFrame()

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()

        # Filtrer à la période demandée
        start_dt = pd.Timestamp(since, tz="UTC")
        end_dt = pd.Timestamp(until, tz="UTC")
        df = df.loc[start_dt:end_dt]

        # Supprimer doublons d'index
        df = df[~df.index.duplicated(keep="first")]

        df.to_parquet(cache)
        logger.info("Sauvegardé: %s — %d bougies", cache.name, len(df))
        return df

    # ── Funding Rates ─────────────────────────────────────────

    def fetch_funding_rates(self, symbol: str,
                            since: str, until: str) -> pd.Series:
        """Télécharge les funding rates historiques.
        Fallback: taux constant depuis la config."""

        cache = self._cache_path(symbol, "funding", kind="funding")
        if self._cache_is_fresh(cache):
            logger.info("Cache hit funding: %s", cache.name)
            df = pd.read_parquet(cache)
            return df["funding_rate"]

        logger.info("Téléchargement funding rates %s", symbol)

        since_ms = self._to_ms(since)
        until_ms = self._to_ms(until)
        all_rates: list[dict] = []

        try:
            while since_ms < until_ms:
                rates = self.exchange.fetch_funding_rate_history(
                    symbol, since=since_ms, limit=1000
                )
                if not rates:
                    break

                for r in rates:
                    all_rates.append({
                        "timestamp": pd.to_datetime(
                            r["timestamp"], unit="ms", utc=True
                        ),
                        "funding_rate": r["fundingRate"],
                    })

                last_ts = rates[-1]["timestamp"]
                if last_ts == since_ms:
                    break
                since_ms = last_ts + 1

        except (ccxt.BaseError, Exception) as e:
            logger.warning(
                "Funding rates indisponibles pour %s: %s — fallback constant",
                symbol, e
            )

        if all_rates:
            df = pd.DataFrame(all_rates).set_index("timestamp").sort_index()
            df = df[~df.index.duplicated(keep="first")]
            df.to_parquet(cache)
            logger.info("Funding rates sauvegardés: %s — %d entrées",
                        cache.name, len(df))
            return df["funding_rate"]

        # Fallback: taux constant
        logger.info("Fallback funding constant %.4f%% pour %s",
                     self.funding_default * 100, symbol)
        return self._generate_constant_funding(since, until)

    def _generate_constant_funding(self, since: str,
                                   until: str) -> pd.Series:
        """Génère une série de funding rate constant toutes les 8h."""
        idx = pd.date_range(
            start=since, end=until,
            freq=f"{self.funding_interval_h}h", tz="UTC"
        )
        return pd.Series(self.funding_default, index=idx, name="funding_rate")

    # ── Resample ──────────────────────────────────────────────

    @staticmethod
    def resample(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Resample un DataFrame 5min vers un timeframe supérieur.
        Agrégation OHLCV standard."""

        tf_map = {"1h": "1h", "4h": "4h", "1d": "1D"}
        freq = tf_map.get(target_tf)
        if freq is None:
            raise ValueError(f"Timeframe non supporté pour resample: {target_tf}")

        resampled = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["open"])

        return resampled

    # ── Nettoyage ─────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime doublons, interpole les gaps <= max_gap bougies,
        rejette les segments avec gaps > max_gap."""

        if df.empty:
            return df

        # Doublons
        before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        dupes = before - len(df)
        if dupes > 0:
            logger.info("  Supprimé %d doublons", dupes)

        # Détecter la fréquence attendue
        freq = pd.infer_freq(df.index[:100])
        if freq is None:
            diffs = df.index.to_series().diff().dropna()
            freq_td = diffs.mode().iloc[0] if len(diffs) > 0 else pd.Timedelta("5min")
        else:
            offset = pd.tseries.frequencies.to_offset(freq)
            # Convertir l'offset en Timedelta
            freq_td = pd.Timedelta(offset.nanos, unit="ns") if hasattr(offset, "nanos") else pd.Timedelta(freq)

        # Identifier les gaps
        diffs = df.index.to_series().diff()
        gap_mask = diffs > freq_td

        if gap_mask.any():
            gap_sizes = (diffs[gap_mask] / freq_td).astype(int)

            # Gaps > max_gap : on log mais on ne supprime pas les données
            # (on interpole seulement les petits gaps)
            large_gaps = gap_sizes[gap_sizes > self.max_gap]
            if len(large_gaps) > 0:
                logger.warning("  %d gaps > %d bougies détectés (non interpolés)",
                               len(large_gaps), self.max_gap)

            small_gaps = gap_sizes[gap_sizes <= self.max_gap]
            if len(small_gaps) > 0:
                logger.info("  Interpolation de %d petits gaps (≤ %d bougies)",
                            len(small_gaps), self.max_gap)

        # Reindex sur la grille complète et interpoler
        full_idx = pd.date_range(
            start=df.index[0], end=df.index[-1], freq=freq_td
        )
        df = df.reindex(full_idx)

        # Interpoler seulement les petits gaps
        df = df.interpolate(method="time", limit=self.max_gap)

        # Supprimer les lignes qui restent NaN (grands gaps)
        df = df.dropna(subset=["close"])

        logger.info("  Après nettoyage: %d bougies", len(df))
        return df

    # ── Colonnes dérivées ─────────────────────────────────────

    @staticmethod
    def add_derived_columns(df: pd.DataFrame,
                            base_tf_minutes: int = 5) -> pd.DataFrame:
        """Ajoute: log_return, volatility_7d, volatility_30d, volume_zscore.
        Les fenêtres sont adaptées au timeframe via base_tf_minutes."""

        df = df.copy()

        # Log return
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Fenêtres en nombre de bougies
        candles_per_day = (24 * 60) // base_tf_minutes
        w7d = 7 * candles_per_day
        w30d = 30 * candles_per_day

        df["volatility_7d"] = df["log_return"].rolling(
            window=w7d, min_periods=w7d // 2
        ).std()

        df["volatility_30d"] = df["log_return"].rolling(
            window=w30d, min_periods=w30d // 2
        ).std()

        # Volume z-score (fenêtre 20 périodes)
        vol_mean = df["volume"].rolling(20).mean()
        vol_std = df["volume"].rolling(20).std()
        df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std

        return df

    # ── Pipeline complète ─────────────────────────────────────

    def load_all(self) -> dict[str, dict[str, pd.DataFrame]]:
        """Pipeline complète:
        1. Télécharge 5min pour chaque asset
        2. Nettoie
        3. Ajoute colonnes dérivées sur le 5min
        4. Resample vers 1h et 4h
        5. Ajoute colonnes dérivées sur chaque timeframe

        Retourne: {symbol: {"5m": df, "1h": df, "4h": df}}
        """

        datasets: dict[str, dict[str, pd.DataFrame]] = {}

        for symbol in self.assets:
            logger.info("━" * 50)
            logger.info("Chargement %s", symbol)
            logger.info("━" * 50)

            # 1. Télécharger 5min
            df_5m = self.fetch_ohlcv(
                symbol, self.base_tf,
                self.period_start, self.period_end
            )

            if df_5m.empty:
                logger.error("Aucune donnée pour %s — skip", symbol)
                continue

            # 2. Nettoyer
            df_5m = self.clean(df_5m)

            # 3. Colonnes dérivées sur 5min
            df_5m = self.add_derived_columns(df_5m, base_tf_minutes=5)

            datasets[symbol] = {"5m": df_5m}

            # 4. Resample vers timeframes supérieurs
            for tf in self.timeframes:
                logger.info("  Resample %s → %s", self.base_tf, tf)
                df_tf = self.resample(df_5m, tf)

                # Minutes par bougie pour les colonnes dérivées
                tf_minutes = {"1h": 60, "4h": 240}.get(tf, 60)
                df_tf = self.add_derived_columns(df_tf, base_tf_minutes=tf_minutes)

                datasets[symbol][tf] = df_tf
                logger.info("  %s %s: %d bougies", symbol, tf, len(df_tf))

            logger.info("  %s complet — 5m:%d | 1h:%d | 4h:%d",
                        symbol,
                        len(datasets[symbol]["5m"]),
                        len(datasets[symbol].get("1h", [])),
                        len(datasets[symbol].get("4h", [])))

        return datasets

    def load_funding(self) -> dict[str, pd.Series]:
        """Télécharge les funding rates pour tous les assets.
        Retourne: {symbol: Series}"""

        funding_data: dict[str, pd.Series] = {}
        for symbol in self.assets:
            funding_data[symbol] = self.fetch_funding_rates(
                symbol, self.period_start, self.period_end
            )
        return funding_data


# ── Standalone test ───────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s"
    )

    config = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    loader = DataLoader(config)

    # Test avec un seul asset pour valider
    df = loader.fetch_ohlcv("BTC/USDT", "5m", "2024-01-01", "2024-01-02")
    if not df.empty:
        df = loader.clean(df)
        df = loader.add_derived_columns(df, base_tf_minutes=5)
        print(f"\nBTC/USDT 5m — {len(df)} bougies")
        print(df.head())
        print(f"\nColonnes: {list(df.columns)}")

        df_1h = loader.resample(df, "1h")
        print(f"\nResample 1h — {len(df_1h)} bougies")
        print(df_1h.head())
