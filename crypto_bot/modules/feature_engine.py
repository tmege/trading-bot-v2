"""
Module 2 — FeatureEngine
Calcule tous les indicateurs techniques sur chaque asset × timeframe.
Utilise pandas-ta. Les indicateurs sont calculés sur le timeframe de la
stratégie (1h, 4h), pas sur le 5min brut.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta
import yaml

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Pipeline d'indicateurs techniques : tendance, momentum,
    volatilité, volume, régime de marché, patterns."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        logger.info("FeatureEngine initialisé")

    # ── Tendance ──────────────────────────────────────────────

    @staticmethod
    def add_trend(df: pd.DataFrame) -> pd.DataFrame:
        """EMA 9/21/50/200, MACD(12,26,9), golden/death cross."""

        for period in (9, 21, 50, 200):
            df[f"EMA{period}"] = ta.ema(df["close"], length=period)

        # MACD
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df["MACD"] = macd.iloc[:, 0]         # MACD line
            df["MACD_signal"] = macd.iloc[:, 2]  # Signal line
            df["MACD_hist"] = macd.iloc[:, 1]    # Histogram

        # Golden cross : EMA9 croise EMA21 vers le haut
        ema9 = df["EMA9"]
        ema21 = df["EMA21"]
        df["golden_cross"] = (ema9 > ema21) & (ema9.shift(1) <= ema21.shift(1))
        df["death_cross"] = (ema9 < ema21) & (ema9.shift(1) >= ema21.shift(1))

        logger.debug("  Tendance ajoutée")
        return df

    # ── Momentum ──────────────────────────────────────────────

    @staticmethod
    def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
        """RSI(14), Stochastic(14,3), Williams %R(14)."""

        df["RSI_14"] = ta.rsi(df["close"], length=14)

        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
        if stoch is not None and not stoch.empty:
            df["STOCH_K"] = stoch.iloc[:, 0]
            df["STOCH_D"] = stoch.iloc[:, 1]

        df["WILLR_14"] = ta.willr(df["high"], df["low"], df["close"], length=14)

        logger.debug("  Momentum ajouté")
        return df

    # ── Volatilité ────────────────────────────────────────────

    @staticmethod
    def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands(20,2), ATR(14), ATR_percentile, compression."""

        # Bollinger Bands
        bb = ta.bbands(df["close"], length=20, std=2)
        if bb is not None and not bb.empty:
            df["BB_upper"] = bb.iloc[:, 2]    # BBU
            df["BB_middle"] = bb.iloc[:, 1]   # BBM
            df["BB_lower"] = bb.iloc[:, 0]    # BBL
            df["BB_bandwidth"] = bb.iloc[:, 3] if bb.shape[1] > 3 else (
                (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
            )
            # %B = (close - lower) / (upper - lower)
            span = df["BB_upper"] - df["BB_lower"]
            df["pct_B"] = np.where(
                span > 0,
                (df["close"] - df["BB_lower"]) / span,
                0.5
            )

        # ATR
        df["ATR_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # ATR percentile sur 30 dernières périodes
        df["ATR_percentile"] = df["ATR_14"].rolling(30).apply(
            lambda x: (x.rank(pct=True).iloc[-1]) * 100
            if len(x.dropna()) >= 10 else np.nan,
            raw=False
        )

        # Compression : ATR_percentile < 20 pendant N bougies consécutives
        low_vol = (df["ATR_percentile"] < 20).astype(int)
        # Compteur de bougies consécutives en compression
        groups = low_vol.ne(low_vol.shift()).cumsum()
        df["compression_count"] = low_vol.groupby(groups).cumsum()
        df["compression"] = df["compression_count"] > 0

        logger.debug("  Volatilité ajoutée")
        return df

    # ── Volume ────────────────────────────────────────────────

    @staticmethod
    def add_volume(df: pd.DataFrame) -> pd.DataFrame:
        """OBV, VWAP (reset daily), volume_ratio."""

        # OBV
        df["OBV"] = ta.obv(df["close"], df["volume"])

        # VWAP reset daily
        df["VWAP"] = FeatureEngine._compute_vwap_daily(df)

        # Volume ratio
        vol_ma20 = df["volume"].rolling(20).mean()
        df["volume_ratio"] = np.where(
            vol_ma20 > 0,
            df["volume"] / vol_ma20,
            1.0
        )

        logger.debug("  Volume ajouté")
        return df

    @staticmethod
    def _compute_vwap_daily(df: pd.DataFrame) -> pd.Series:
        """VWAP avec reset quotidien."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        tp_vol = typical_price * df["volume"]

        # Grouper par jour
        dates = df.index.date
        vwap = pd.Series(np.nan, index=df.index, dtype=float)

        for date in pd.unique(dates):
            mask = dates == date
            cum_tp_vol = tp_vol.loc[mask].cumsum()
            cum_vol = df["volume"].loc[mask].cumsum()
            vwap.loc[mask] = np.where(
                cum_vol > 0,
                cum_tp_vol / cum_vol,
                typical_price.loc[mask]
            )

        return vwap

    # ── Régime de marché ──────────────────────────────────────

    @staticmethod
    def add_regime(df: pd.DataFrame) -> pd.DataFrame:
        """Colonne catégorielle 'regime': bull | bear | ranging."""

        ema50 = df.get("EMA50")
        ema200 = df.get("EMA200")

        if ema50 is None or ema200 is None:
            logger.warning("EMA50/EMA200 manquantes — regime non calculé")
            df["regime"] = "unknown"
            return df

        conditions = [
            df["close"] > ema200,                              # bull
            df["close"] < ema200,                              # bear
        ]
        # Ranging : abs(close - EMA50) / EMA50 < 0.05
        ranging_mask = (
            (df["close"] - ema50).abs() / ema50.where(ema50 > 0, np.nan)
        ) < 0.05

        # Priorité : ranging écrase bull/bear si applicable
        regime = np.select(
            conditions,
            ["bull", "bear"],
            default="ranging"
        )
        # Override avec ranging
        regime = np.where(ranging_mask, "ranging", regime)

        df["regime"] = pd.Categorical(
            regime, categories=["bull", "bear", "ranging", "unknown"]
        )

        logger.debug("  Régime ajouté")
        return df

    # ── Patterns ──────────────────────────────────────────────

    @staticmethod
    def add_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """higher_high, lower_low, inside_bar, body_ratio."""

        # Higher high : close > max(close) des 5 dernières bougies
        roll_max = df["close"].rolling(5).max().shift(1)
        df["higher_high"] = df["close"] > roll_max

        # Lower low : close < min(close) des 5 dernières bougies
        roll_min = df["close"].rolling(5).min().shift(1)
        df["lower_low"] = df["close"] < roll_min

        # Inside bar : high < high[-1] AND low > low[-1]
        df["inside_bar"] = (
            (df["high"] < df["high"].shift(1)) &
            (df["low"] > df["low"].shift(1))
        )

        # Body ratio : |close - open| / (high - low)
        candle_range = df["high"] - df["low"]
        df["body_ratio"] = np.where(
            candle_range > 0,
            (df["close"] - df["open"]).abs() / candle_range,
            0.0
        )

        logger.debug("  Patterns ajoutés")
        return df

    # ── Features V2 (stratégies assouplies) ───────────────────

    @staticmethod
    def add_v2_features(df: pd.DataFrame) -> pd.DataFrame:
        """rsi_slope, price_position_bb, atr_pct.
        momentum_score dépend des params de chaque stratégie → calculé inline."""

        if "RSI_14" in df.columns:
            df["rsi_slope"] = df["RSI_14"] - df["RSI_14"].shift(3)

        if "BB_lower" in df.columns and "BB_upper" in df.columns:
            bb_span = df["BB_upper"] - df["BB_lower"]
            df["price_position_bb"] = np.where(
                bb_span > 0,
                (df["close"] - df["BB_lower"]) / bb_span,
                0.5,
            )

        if "ATR_14" in df.columns:
            df["atr_pct"] = np.where(
                df["close"] > 0,
                df["ATR_14"] / df["close"],
                0.0,
            )

        logger.debug("  Features V2 ajoutées")
        return df

    # ── Pipeline complète ─────────────────────────────────────

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique tous les indicateurs sur un DataFrame OHLCV."""

        if df.empty:
            return df

        df = df.copy()
        df = self.add_trend(df)
        df = self.add_momentum(df)
        df = self.add_volatility(df)
        df = self.add_volume(df)
        df = self.add_regime(df)
        df = self.add_patterns(df)
        df = self.add_v2_features(df)

        # Compter les colonnes ajoutées (hors OHLCV + dérivées du DataLoader)
        base_cols = {"open", "high", "low", "close", "volume",
                     "log_return", "volatility_7d", "volatility_30d",
                     "volume_zscore"}
        new_cols = [c for c in df.columns if c not in base_cols]
        logger.info("  %d indicateurs calculés", len(new_cols))

        return df

    def compute_all_datasets(
        self,
        datasets: dict[str, dict[str, pd.DataFrame]],
        skip_base_tf: bool = True
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Applique compute_all sur chaque asset × timeframe.

        Args:
            datasets: {symbol: {timeframe: DataFrame}}
            skip_base_tf: si True, ne calcule pas les indicateurs
                          sur le 5min (inutile pour les stratégies,
                          le 5min sert uniquement à la liquidation).
        """

        for symbol, tfs in datasets.items():
            for tf, df in tfs.items():
                if skip_base_tf and tf == "5m":
                    logger.info("  Skip indicateurs sur %s 5m (base_tf)", symbol)
                    continue

                logger.info("Indicateurs %s %s (%d bougies)", symbol, tf, len(df))
                datasets[symbol][tf] = self.compute_all(df)

        return datasets


# ── Standalone test ───────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s"
    )

    # Test avec des données synthétiques
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    price = 40000 + np.cumsum(np.random.randn(n) * 100)

    df = pd.DataFrame({
        "open": price,
        "high": price + np.abs(np.random.randn(n) * 50),
        "low": price - np.abs(np.random.randn(n) * 50),
        "close": price + np.random.randn(n) * 30,
        "volume": np.random.exponential(1000, n),
    }, index=dates)

    engine = FeatureEngine()
    df = engine.compute_all(df)

    print(f"\nColonnes ({len(df.columns)}):")
    for col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  {col:25s} — {non_null}/{n} valeurs")

    print(f"\nRégimes:\n{df['regime'].value_counts()}")
    print(f"\nGolden crosses: {df['golden_cross'].sum()}")
    print(f"Death crosses:  {df['death_cross'].sum()}")
    print(f"Inside bars:    {df['inside_bar'].sum()}")
