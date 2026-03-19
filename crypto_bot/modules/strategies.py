"""
Module 5 — Strategies
4 stratégies leveragées en isolated margin.
Chaque stratégie génère des signaux et définit ses variantes à tester.
Tout sizing en % du portfolio.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ── Base Strategy ─────────────────────────────────────────────

@dataclass
class TradeSignal:
    """Signal émis par une stratégie."""
    idx: int                    # index dans le DataFrame
    side: str                   # "long" ou "short"
    size_pct: float             # % du portfolio
    leverage: float
    sl_pct: float               # stop loss en % depuis entrée
    tp_pct: float               # take profit en % depuis entrée
    tp_partial: dict | None = None      # {"pct": 3.0, "close_ratio": 0.5}
    trailing_stop_pct: float | None = None
    exit_condition: str | None = None   # nom de la condition de sortie
    entry_order_type: str = "maker"     # "maker" (ALO limit) ou "taker" (market)
    exit_tp_order_type: str = "maker"   # TP peut être limit ALO → maker
    exit_sl_order_type: str = "taker"   # SL toujours market → taker
    metadata: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """Interface commune pour toutes les stratégies."""

    name: str = "base"
    leverage: float = 1.0
    size_pct: float = 10.0
    sl_pct: float = 2.0
    tp_pct: float = 5.0
    max_positions: int = 1
    preferred_timeframe: str = "1h"

    def __init__(self, config_path: str = "config.yaml", **overrides):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        # Appliquer les overrides (pour les variantes)
        for key, val in overrides.items():
            if hasattr(self, key):
                setattr(self, key, val)

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame,
                         df_5m: pd.DataFrame | None = None,
                         df_aux: dict[str, pd.DataFrame] | None = None
                         ) -> list[TradeSignal]:
        """Génère la liste des signaux de trading.

        Args:
            df: DataFrame avec indicateurs au timeframe de la stratégie
            df_5m: DataFrame 5min pour précision (optionnel)
            df_aux: DataFrames auxiliaires (ex: 4h pour strat C)
        """
        ...

    @classmethod
    @abstractmethod
    def get_variants(cls) -> list[dict]:
        """Retourne les combinaisons de paramètres à tester."""
        ...

    def __repr__(self) -> str:
        return (f"{self.name}(lev={self.leverage}x, size={self.size_pct}%, "
                f"sl={self.sl_pct}%, tp={self.tp_pct}%)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRAT A — Grid leveragée
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GridStrategy(BaseStrategy):
    """Grid leveragée ±8% autour EMA50.

    Activation : prix dans canal ±8% autour EMA50 depuis ≥ 48 bougies 1h.
    5 BUY espacés sous le prix, 5 SELL espacés au-dessus.
    Chaque ordre = 5% du portfolio.
    Si prix sort du canal ±15% → fermer toute la grille.
    Capital max grille : 30% du portfolio.
    """

    name = "grid"
    leverage = 3.0
    size_pct = 5.0          # par ordre de grille
    sl_pct = 15.0           # sortie si prix sort du canal ±15%
    tp_pct = 1.5            # chaque niveau de grille = espacement
    max_positions = 10      # 5 buy + 5 sell
    preferred_timeframe = "1h"

    # Paramètres spécifiques
    channel_pct: float = 8.0        # canal ±8% autour EMA50
    exit_channel_pct: float = 15.0  # sortie si ±15%
    grid_spacing_pct: float = 1.5   # espacement entre ordres
    n_levels: int = 5               # ordres par côté
    min_ranging_bars: int = 48      # bougies minimum en range
    max_capital_pct: float = 30.0   # capital max engagé

    def __init__(self, config_path: str = "config.yaml", **overrides):
        super().__init__(config_path, **overrides)
        for key in ("channel_pct", "exit_channel_pct", "grid_spacing_pct",
                     "n_levels", "min_ranging_bars", "max_capital_pct"):
            if key in overrides:
                setattr(self, key, overrides[key])

    def generate_signals(self, df: pd.DataFrame,
                         df_5m: pd.DataFrame | None = None,
                         df_aux: dict[str, pd.DataFrame] | None = None
                         ) -> list[TradeSignal]:
        signals = []

        if "EMA50" not in df.columns:
            logger.warning("GridStrategy: EMA50 manquante")
            return signals

        ema50 = df["EMA50"]
        close = df["close"]

        # Détecter les périodes de ranging : prix dans ±channel_pct% de EMA50
        in_channel = ((close - ema50).abs() / ema50 < self.channel_pct / 100)

        # Compteur de bougies consécutives dans le canal
        groups = in_channel.ne(in_channel.shift()).cumsum()
        consecutive = in_channel.groupby(groups).cumsum()

        # Activation : ≥ min_ranging_bars bougies dans le canal
        activation = consecutive >= self.min_ranging_bars

        # Capital déjà engagé (tracking simplifié — le backtester gère le réel)
        grid_active = False
        grid_entry_idx = None

        for i in range(len(df)):
            if not activation.iloc[i]:
                # Pas de range suffisant — si grille active, on vérifie la sortie
                if grid_active:
                    dist_from_ema = abs(close.iloc[i] - ema50.iloc[i]) / ema50.iloc[i] * 100
                    if dist_from_ema > self.exit_channel_pct:
                        grid_active = False
                        grid_entry_idx = None
                continue

            if grid_active:
                continue  # Grille déjà ouverte

            # Ouvrir une nouvelle grille
            grid_active = True
            grid_entry_idx = i
            current_price = close.iloc[i]

            # Capital max : limiter le nombre de niveaux
            max_orders = int(self.max_capital_pct / self.size_pct)
            n_buy = min(self.n_levels, max_orders // 2)
            n_sell = min(self.n_levels, max_orders - n_buy)

            # Ordres BUY sous le prix
            for level in range(1, n_buy + 1):
                offset_pct = level * self.grid_spacing_pct
                signals.append(TradeSignal(
                    idx=i,
                    side="long",
                    size_pct=self.size_pct,
                    leverage=self.leverage,
                    sl_pct=self.exit_channel_pct,
                    tp_pct=self.grid_spacing_pct,
                    metadata={
                        "grid_level": -level,
                        "target_entry_offset_pct": -offset_pct,
                    }
                ))

            # Ordres SELL au-dessus du prix
            for level in range(1, n_sell + 1):
                offset_pct = level * self.grid_spacing_pct
                signals.append(TradeSignal(
                    idx=i,
                    side="short",
                    size_pct=self.size_pct,
                    leverage=self.leverage,
                    sl_pct=self.exit_channel_pct,
                    tp_pct=self.grid_spacing_pct,
                    metadata={
                        "grid_level": level,
                        "target_entry_offset_pct": offset_pct,
                    }
                ))

        logger.info("GridStrategy — %d signaux générés", len(signals))
        return signals

    @classmethod
    def get_variants(cls) -> list[dict]:
        leverages = [2, 3, 5]
        spacings = [1.0, 1.5, 2.0]
        return [
            {"leverage": lev, "grid_spacing_pct": sp, "tp_pct": sp}
            for lev, sp in product(leverages, spacings)
        ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRAT B — Momentum futures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MomentumStrategy(BaseStrategy):
    """Momentum 4h avec TP partiel + trailing stop.

    LONG:  EMA9 > EMA21 > EMA50, RSI 55–70, volume_ratio > 1.8, regime bull
    SHORT: EMA9 < EMA21 < EMA50, RSI 30–45, volume_ratio > 1.8, regime bear

    Gestion: TP1 à +3% (ferme 50%), trailing stop 1.5% sur le reste.
    Max 2 positions par asset.
    """

    name = "momentum"
    leverage = 5.0
    size_pct = 20.0
    sl_pct = 2.0
    tp_pct = 6.0            # TP final (trailing)
    max_positions = 2
    preferred_timeframe = "4h"

    # Paramètres spécifiques
    rsi_long_min: float = 55.0
    rsi_long_max: float = 70.0
    rsi_short_min: float = 30.0
    rsi_short_max: float = 45.0
    volume_ratio_min: float = 1.8
    tp1_pct: float = 3.0
    tp1_close_ratio: float = 0.5
    trailing_pct: float = 1.5

    def __init__(self, config_path: str = "config.yaml", **overrides):
        super().__init__(config_path, **overrides)
        for key in ("rsi_long_min", "rsi_long_max", "rsi_short_min",
                     "rsi_short_max", "volume_ratio_min", "tp1_pct",
                     "tp1_close_ratio", "trailing_pct"):
            if key in overrides:
                setattr(self, key, overrides[key])

    def generate_signals(self, df: pd.DataFrame,
                         df_5m: pd.DataFrame | None = None,
                         df_aux: dict[str, pd.DataFrame] | None = None
                         ) -> list[TradeSignal]:
        signals = []
        required = {"EMA9", "EMA21", "EMA50", "RSI_14", "volume_ratio", "regime"}

        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            logger.warning("MomentumStrategy: colonnes manquantes: %s", missing)
            return signals

        open_positions = 0

        for i in range(1, len(df)):
            row = df.iloc[i]

            # LONG
            long_ema = (row["EMA9"] > row["EMA21"] > row["EMA50"])
            long_rsi = self.rsi_long_min <= row["RSI_14"] <= self.rsi_long_max
            long_vol = row["volume_ratio"] > self.volume_ratio_min
            long_regime = row["regime"] == "bull"

            if long_ema and long_rsi and long_vol and long_regime:
                if open_positions < self.max_positions:
                    signals.append(TradeSignal(
                        idx=i,
                        side="long",
                        size_pct=self.size_pct,
                        leverage=self.leverage,
                        sl_pct=self.sl_pct,
                        tp_pct=self.tp_pct,
                        tp_partial={
                            "pct": self.tp1_pct,
                            "close_ratio": self.tp1_close_ratio,
                        },
                        trailing_stop_pct=self.trailing_pct,
                    ))
                    open_positions += 1
                    continue

            # SHORT
            short_ema = (row["EMA9"] < row["EMA21"] < row["EMA50"])
            short_rsi = self.rsi_short_min <= row["RSI_14"] <= self.rsi_short_max
            short_vol = row["volume_ratio"] > self.volume_ratio_min
            short_regime = row["regime"] == "bear"

            if short_ema and short_rsi and short_vol and short_regime:
                if open_positions < self.max_positions:
                    signals.append(TradeSignal(
                        idx=i,
                        side="short",
                        size_pct=self.size_pct,
                        leverage=self.leverage,
                        sl_pct=self.sl_pct,
                        tp_pct=self.tp_pct,
                        tp_partial={
                            "pct": self.tp1_pct,
                            "close_ratio": self.tp1_close_ratio,
                        },
                        trailing_stop_pct=self.trailing_pct,
                    ))
                    open_positions += 1

            # Reset position counter (simplifié — le backtester gère le réel)
            # On assume que les positions se ferment avant le prochain signal
            if open_positions >= self.max_positions:
                open_positions = max(0, open_positions - 1)

        logger.info("MomentumStrategy — %d signaux générés", len(signals))
        return signals

    @classmethod
    def get_variants(cls) -> list[dict]:
        leverages = [3, 5, 7]
        stops = [1.5, 2.0, 3.0]
        return [
            {"leverage": lev, "sl_pct": sl,
             "tp1_pct": sl * 1.5, "tp_pct": sl * 3}
            for lev, sl in product(leverages, stops)
        ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRAT C — Mean reversion oversold
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion oversold 1h. Long uniquement.

    Entrée (toutes simultanées) :
      RSI(14) < 25
      close < BB lower
      body_ratio de la bougie précédente < 0.3
      volume_ratio < 0.8
      close(4h) > EMA200(4h) — ne pas shorter le trend majeur

    Sortie : RSI > 50 OU close > EMA21
    """

    name = "mean_reversion"
    leverage = 3.0
    size_pct = 15.0
    sl_pct = 3.0
    tp_pct = 10.0           # TP max (fallback si signal ne sort pas)
    max_positions = 1
    preferred_timeframe = "1h"

    # Paramètres spécifiques
    rsi_entry: float = 25.0
    rsi_exit: float = 50.0
    body_ratio_max: float = 0.3
    volume_ratio_max: float = 0.8

    def __init__(self, config_path: str = "config.yaml", **overrides):
        super().__init__(config_path, **overrides)
        for key in ("rsi_entry", "rsi_exit", "body_ratio_max",
                     "volume_ratio_max"):
            if key in overrides:
                setattr(self, key, overrides[key])

    def generate_signals(self, df: pd.DataFrame,
                         df_5m: pd.DataFrame | None = None,
                         df_aux: dict[str, pd.DataFrame] | None = None
                         ) -> list[TradeSignal]:
        signals = []
        required = {"RSI_14", "BB_lower", "body_ratio", "volume_ratio", "EMA21"}

        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            logger.warning("MeanReversionStrategy: colonnes manquantes: %s", missing)
            return signals

        # Vérifier le trend 4h si disponible
        has_4h = (df_aux is not None and "4h" in df_aux
                  and "EMA200" in df_aux["4h"].columns)

        in_position = False

        for i in range(2, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            if in_position:
                # Sortie : RSI > rsi_exit OU close > EMA21
                if row["RSI_14"] > self.rsi_exit or row["close"] > row["EMA21"]:
                    in_position = False
                continue

            # Conditions d'entrée
            rsi_ok = row["RSI_14"] < self.rsi_entry
            bb_ok = row["close"] < row["BB_lower"] if pd.notna(row.get("BB_lower")) else False
            body_ok = prev["body_ratio"] < self.body_ratio_max
            vol_ok = row["volume_ratio"] < self.volume_ratio_max

            # Trend 4h : close(4h) > EMA200(4h)
            trend_ok = True
            if has_4h:
                df_4h = df_aux["4h"]
                # Trouver la bougie 4h correspondante
                ts = df.index[i]
                valid_4h = df_4h.index[df_4h.index <= ts]
                if len(valid_4h) > 0:
                    last_4h = df_4h.loc[valid_4h[-1]]
                    if pd.notna(last_4h.get("EMA200")):
                        trend_ok = last_4h["close"] > last_4h["EMA200"]

            if rsi_ok and bb_ok and body_ok and vol_ok and trend_ok:
                # Exit condition : encode comme string, le backtester l'interprétera
                signals.append(TradeSignal(
                    idx=i,
                    side="long",
                    size_pct=self.size_pct,
                    leverage=self.leverage,
                    sl_pct=self.sl_pct,
                    tp_pct=self.tp_pct,
                    exit_condition="rsi_above_50_or_above_ema21",
                    metadata={
                        "rsi_at_entry": round(float(row["RSI_14"]), 2),
                    }
                ))
                in_position = True

        logger.info("MeanReversionStrategy — %d signaux générés", len(signals))
        return signals

    @classmethod
    def get_variants(cls) -> list[dict]:
        rsi_thresholds = [20, 25, 30]
        leverages = [2, 3, 5]
        return [
            {"rsi_entry": rsi, "leverage": lev}
            for rsi, lev in product(rsi_thresholds, leverages)
        ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRAT D — Breakout explosif
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BreakoutStrategy(BaseStrategy):
    """Breakout explosif 1h. Levier 10x sur petite taille.

    Conditions :
      ATR_percentile < 20 (compression)
      close dépasse le HIGH des 20 dernières bougies
      body_ratio > 0.7 (bougie de conviction)
      volume_ratio > 3.0

    Gestion :
      SL: -1.5% (= -15% de la marge à 10x)
      TP: +5%   (= +50% de la marge à 10x)
      Taille: 5% → levier 10x = 50% notionnel, mais seulement 5% risqué

    Max 1 position tous assets confondus.
    """

    name = "breakout"
    leverage = 10.0
    size_pct = 5.0
    sl_pct = 1.5
    tp_pct = 5.0
    max_positions = 1       # tous assets confondus
    preferred_timeframe = "1h"

    # Paramètres spécifiques
    atr_percentile_max: float = 20.0
    lookback_high: int = 20
    body_ratio_min: float = 0.7
    volume_ratio_min: float = 3.0

    def __init__(self, config_path: str = "config.yaml", **overrides):
        super().__init__(config_path, **overrides)
        for key in ("atr_percentile_max", "lookback_high",
                     "body_ratio_min", "volume_ratio_min"):
            if key in overrides:
                setattr(self, key, overrides[key])

    def generate_signals(self, df: pd.DataFrame,
                         df_5m: pd.DataFrame | None = None,
                         df_aux: dict[str, pd.DataFrame] | None = None
                         ) -> list[TradeSignal]:
        signals = []
        required = {"ATR_percentile", "body_ratio", "volume_ratio"}

        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            logger.warning("BreakoutStrategy: colonnes manquantes: %s", missing)
            return signals

        # High des N dernières bougies (rolling max)
        rolling_high = df["high"].rolling(self.lookback_high).max().shift(1)

        in_position = False

        for i in range(self.lookback_high + 1, len(df)):
            row = df.iloc[i]

            if in_position:
                # Cooldown simple : attendre au moins 10 bougies entre trades
                if i - last_signal_idx < 10:
                    continue
                in_position = False

            # Conditions
            compression = (pd.notna(row["ATR_percentile"])
                           and row["ATR_percentile"] < self.atr_percentile_max)
            breakout = row["close"] > rolling_high.iloc[i] if pd.notna(rolling_high.iloc[i]) else False
            body = row["body_ratio"] > self.body_ratio_min
            volume = row["volume_ratio"] > self.volume_ratio_min

            if compression and breakout and body and volume:
                signals.append(TradeSignal(
                    idx=i,
                    side="long",
                    size_pct=self.size_pct,
                    leverage=self.leverage,
                    sl_pct=self.sl_pct,
                    tp_pct=self.tp_pct,
                    metadata={
                        "atr_pct": round(float(row["ATR_percentile"]), 1),
                        "vol_ratio": round(float(row["volume_ratio"]), 2),
                        "body_ratio": round(float(row["body_ratio"]), 3),
                    }
                ))
                in_position = True
                last_signal_idx = i

        logger.info("BreakoutStrategy — %d signaux générés", len(signals))
        return signals

    @classmethod
    def get_variants(cls) -> list[dict]:
        leverages = [7, 10, 15]
        sizes = [3.0, 5.0, 8.0]
        return [
            {"leverage": lev, "size_pct": sz}
            for lev, sz in product(leverages, sizes)
        ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# V2 STRATEGIES — Assouplies pour plus de signaux
# Interface : __init__(config: dict), generate_signals(df) -> pd.Series
# Chaque strat retourne 1 (long), -1 (short), 0 (neutre)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _signal_frequency(signals: pd.Series, n_bars: int) -> dict:
    """Diagnostic de fréquence commun à toutes les strats V2."""
    active = signals[signals != 0]
    n_active = len(active)
    return {
        "total_bougies":    n_bars,
        "total_signaux":    n_active,
        "signal_rate_pct":  round(n_active / n_bars * 100, 2) if n_bars > 0 else 0,
        "signaux_par_mois": round(n_active / max(n_bars / 720, 0.01), 1),
        "verdict": "TESTER" if n_bars > 0 and n_active / n_bars >= 0.01 else "REJETER",
    }


class StratMomentumScore:
    """Score de momentum composite (somme de 4 conditions booléennes).

    LONG  : score passe de < threshold_low à >= threshold_high
    SHORT : score passe de > threshold_high à <= threshold_low
    """

    def __init__(self, config: dict):
        self.threshold_low  = config.get("threshold_low", 1)
        self.threshold_high = config.get("threshold_high", 3)
        self.sl_pct   = config.get("sl_pct", 2.0)
        self.tp_pct   = config.get("tp_pct", 4.0)
        self.max_hold = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ema21     = df.get("EMA21")
        rsi       = df.get("RSI_14")
        macd_hist = df.get("MACD_hist")
        vol_ratio = df.get("volume_ratio")

        if any(x is None for x in [ema21, rsi, macd_hist, vol_ratio]):
            return pd.Series(0, index=df.index)

        score = (
            (df["close"] > ema21).astype(int)
            + (rsi > 50).astype(int)
            + (macd_hist > 0).astype(int)
            + (vol_ratio > 1.2).astype(int)
        )
        prev = score.shift(1)

        signals = pd.Series(0, index=df.index)
        signals.loc[(prev < self.threshold_low) & (score >= self.threshold_high)] = 1
        signals.loc[(prev > self.threshold_high) & (score <= self.threshold_low)] = -1
        return signals

    def signal_frequency(self, df: pd.DataFrame) -> dict:
        return _signal_frequency(self.generate_signals(df), len(df))


class StratEmaCrossover:
    """Croisement d'EMA avec filtre de régime optionnel.

    LONG  : EMA_fast croise au-dessus de EMA_slow
    SHORT : EMA_fast croise en dessous de EMA_slow
    Filtre : long seulement en bull, short seulement en bear (si activé).
    """

    def __init__(self, config: dict):
        self.ema_fast          = config.get("ema_fast", 9)
        self.ema_slow          = config.get("ema_slow", 21)
        self.use_regime_filter = config.get("use_regime_filter", True)
        self.sl_buffer_pct     = config.get("sl_buffer_pct", 0.5)
        self.tp_pct            = config.get("tp_pct", 5.0)
        # SL estimé : distance EMA médiane + buffer
        self.sl_pct   = self.sl_buffer_pct + 1.5
        self.max_hold = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast_col = f"EMA{self.ema_fast}"
        slow_col = f"EMA{self.ema_slow}"

        ema_fast = df.get(fast_col)
        ema_slow = df.get(slow_col)

        # Calcul inline si colonnes absentes
        if ema_fast is None:
            ema_fast = df["close"].ewm(span=self.ema_fast, adjust=False).mean()
        if ema_slow is None:
            ema_slow = df["close"].ewm(span=self.ema_slow, adjust=False).mean()

        # Mise à jour SL depuis la distance EMA médiane
        ema_dist = ((ema_fast - ema_slow).abs() / df["close"]).median() * 100
        self.sl_pct = max(0.5, ema_dist + self.sl_buffer_pct)

        signals = pd.Series(0, index=df.index)

        long_cross  = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        short_cross = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

        signals.loc[long_cross]  = 1
        signals.loc[short_cross] = -1

        if self.use_regime_filter:
            regime = df.get("regime")
            if regime is not None:
                signals.loc[(signals == 1) & (regime != "bull")]  = 0
                signals.loc[(signals == -1) & (regime != "bear")] = 0

        return signals

    def signal_frequency(self, df: pd.DataFrame) -> dict:
        return _signal_frequency(self.generate_signals(df), len(df))


class StratBreakoutRelaxed:
    """Breakout assoupli avec filtre de compression optionnel.

    Breakout haut : close > HIGH(lookback) ET volume_ratio > vol_breakout_min
    Breakout bas  : close < LOW(lookback) ET volume_ratio > vol_breakout_min
    Direction     : long si close > EMA50, short sinon.
    """

    def __init__(self, config: dict):
        self.lookback         = config.get("lookback", 10)
        self.vol_breakout_min = config.get("vol_breakout_min", 1.5)
        self.use_compression  = config.get("use_compression", False)
        self.sl_pct           = config.get("sl_pct", 2.0)
        self.tp_pct           = config.get("tp_pct", 5.0)
        self.max_hold         = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close     = df["close"]
        vol_ratio = df.get("volume_ratio")
        ema50     = df.get("EMA50")

        if vol_ratio is None:
            return pd.Series(0, index=df.index)

        # Rolling high / low des lookback bougies (shift 1 = pas de look-ahead)
        rolling_high = df["high"].rolling(self.lookback).max().shift(1)
        rolling_low  = df["low"].rolling(self.lookback).min().shift(1)

        breakout_up   = (close > rolling_high) & (vol_ratio > self.vol_breakout_min)
        breakout_down = (close < rolling_low)  & (vol_ratio > self.vol_breakout_min)

        # Filtre compression optionnel : ATR < ATR.rolling(20).mean()
        if self.use_compression:
            atr_pct = df.get("atr_pct")
            if atr_pct is None:
                atr = df.get("ATR_14")
                if atr is not None:
                    atr_pct = atr / close
            if atr_pct is not None:
                compression = atr_pct < atr_pct.rolling(20).mean()
                breakout_up   = breakout_up   & compression
                breakout_down = breakout_down & compression

        signals = pd.Series(0, index=df.index)

        if ema50 is not None:
            signals.loc[breakout_up   & (close > ema50)] = 1
            signals.loc[breakout_down & (close < ema50)] = -1
        else:
            signals.loc[breakout_up]   = 1
            signals.loc[breakout_down] = -1

        return signals

    def signal_frequency(self, df: pd.DataFrame) -> dict:
        return _signal_frequency(self.generate_signals(df), len(df))


class StratMeanReversionBB:
    """Mean-reversion Bollinger Bands.

    LONG  : RSI < rsi_oversold AND pct_B < bb_entry_low AND regime != "bear"
    SHORT : RSI > rsi_overbought AND pct_B > bb_entry_high AND regime != "bull"
    """

    def __init__(self, config: dict):
        self.rsi_oversold   = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.bb_entry_low   = config.get("bb_entry_low", 0.05)
        self.bb_entry_high  = config.get("bb_entry_high", 0.95)
        self.sl_pct         = config.get("sl_pct", 2.0)
        self.tp_pct         = config.get("tp_pct", 4.0)
        self.max_hold       = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi    = df.get("RSI_14")
        pct_b  = df.get("pct_B")
        regime = df.get("regime")

        if rsi is None or pct_b is None:
            return pd.Series(0, index=df.index)

        signals = pd.Series(0, index=df.index)

        long_cond = (rsi < self.rsi_oversold) & (pct_b < self.bb_entry_low)
        short_cond = (rsi > self.rsi_overbought) & (pct_b > self.bb_entry_high)

        if regime is not None:
            long_cond = long_cond & (regime != "bear")
            short_cond = short_cond & (regime != "bull")

        signals.loc[long_cond] = 1
        signals.loc[short_cond] = -1
        return signals

    def signal_frequency(self, df: pd.DataFrame) -> dict:
        return _signal_frequency(self.generate_signals(df), len(df))


class StratStochReversal:
    """Stochastic K/D crossover dans les zones extrêmes avec volume.

    LONG  : K croise au-dessus de D AND both < oversold AND vol_ratio > vol_min
    SHORT : K croise en dessous de D AND both > overbought AND vol_ratio > vol_min
    """

    def __init__(self, config: dict):
        self.oversold   = config.get("oversold", 20)
        self.overbought = config.get("overbought", 80)
        self.vol_min    = config.get("vol_min", 1.0)
        self.sl_pct     = config.get("sl_pct", 2.0)
        self.tp_pct     = config.get("tp_pct", 4.0)
        self.max_hold   = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        k = df.get("STOCH_K")
        d = df.get("STOCH_D")
        vol_ratio = df.get("volume_ratio")

        if k is None or d is None:
            return pd.Series(0, index=df.index)

        signals = pd.Series(0, index=df.index)

        # Crossover : K was below D, now above
        k_cross_up = (k > d) & (k.shift(1) <= d.shift(1))
        k_cross_down = (k < d) & (k.shift(1) >= d.shift(1))

        long_cond = k_cross_up & (k < self.oversold) & (d < self.oversold)
        short_cond = k_cross_down & (k > self.overbought) & (d > self.overbought)

        if vol_ratio is not None:
            long_cond = long_cond & (vol_ratio > self.vol_min)
            short_cond = short_cond & (vol_ratio > self.vol_min)

        signals.loc[long_cond] = 1
        signals.loc[short_cond] = -1
        return signals

    def signal_frequency(self, df: pd.DataFrame) -> dict:
        return _signal_frequency(self.generate_signals(df), len(df))


class StratInsideBarBreakout:
    """Inside bar consolidation -> breakout directionnel.

    Signal : inside_bar à bar i, puis bar i+1 close > high[i] -> LONG,
             close < low[i] -> SHORT.
    Filtres optionnels : volume, trend EMA21, compression ATR.
    """

    def __init__(self, config: dict):
        self.vol_min      = config.get("vol_min", 1.0)
        self.trend_filter = config.get("trend_filter", False)
        self.atr_filter   = config.get("atr_filter", False)
        self.sl_pct       = config.get("sl_pct", 2.0)
        self.tp_pct       = config.get("tp_pct", 5.0)
        self.max_hold     = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        inside = df.get("inside_bar")
        vol_ratio = df.get("volume_ratio")

        if inside is None:
            return pd.Series(0, index=df.index)

        signals = pd.Series(0, index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Inside bar at i, breakout at i+1
        # Shift inside_bar forward: if inside_bar[i] was True, check bar i+1
        ib_prev = inside.shift(1).fillna(False)
        high_prev = high.shift(1)
        low_prev = low.shift(1)

        long_break = ib_prev & (close > high_prev)
        short_break = ib_prev & (close < low_prev)

        # Volume filter
        if vol_ratio is not None:
            vol_ok = vol_ratio > self.vol_min
            long_break = long_break & vol_ok
            short_break = short_break & vol_ok

        # Trend filter: long only above EMA21, short only below
        if self.trend_filter:
            ema21 = df.get("EMA21")
            if ema21 is not None:
                long_break = long_break & (close > ema21)
                short_break = short_break & (close < ema21)

        # ATR compression filter
        if self.atr_filter:
            compression = df.get("compression")
            if compression is not None:
                long_break = long_break & compression
                short_break = short_break & compression

        signals.loc[long_break] = 1
        signals.loc[short_break] = -1
        return signals

    def signal_frequency(self, df: pd.DataFrame) -> dict:
        return _signal_frequency(self.generate_signals(df), len(df))


class StratRegimeAdaptive:
    """Adapte selon le régime : momentum en trend, mean-reversion en ranging.

    Bull    : MACD_hist > 0 AND RSI > rsi_bull -> LONG
    Ranging : RSI < rsi_range_low -> LONG, RSI > rsi_range_high -> SHORT
    Bear    : MACD_hist < 0 AND RSI < rsi_bear -> SHORT
    Flag use_ranging_only : ne trade qu'en ranging (hypothèse ETH pure).
    """

    def __init__(self, config: dict):
        self.rsi_bull       = config.get("rsi_bull", 55)
        self.rsi_bear       = config.get("rsi_bear", 45)
        self.rsi_range_low  = config.get("rsi_range_low", 30)
        self.rsi_range_high = config.get("rsi_range_high", 70)
        self.use_ranging_only = config.get("use_ranging_only", False)
        self.sl_pct         = config.get("sl_pct", 2.0)
        self.tp_pct         = config.get("tp_pct", 4.0)
        self.max_hold       = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi       = df.get("RSI_14")
        macd_hist = df.get("MACD_hist")
        regime    = df.get("regime")

        if rsi is None or regime is None:
            return pd.Series(0, index=df.index)

        signals = pd.Series(0, index=df.index)

        is_bull    = regime == "bull"
        is_bear    = regime == "bear"
        is_ranging = regime == "ranging"

        if not self.use_ranging_only:
            # Bull regime: momentum long
            if macd_hist is not None:
                bull_long = is_bull & (macd_hist > 0) & (rsi > self.rsi_bull)
                signals.loc[bull_long] = 1

                # Bear regime: momentum short
                bear_short = is_bear & (macd_hist < 0) & (rsi < self.rsi_bear)
                signals.loc[bear_short] = -1

        # Ranging regime: mean-reversion
        range_long = is_ranging & (rsi < self.rsi_range_low)
        range_short = is_ranging & (rsi > self.rsi_range_high)
        signals.loc[range_long] = 1
        signals.loc[range_short] = -1

        return signals

    def signal_frequency(self, df: pd.DataFrame) -> dict:
        return _signal_frequency(self.generate_signals(df), len(df))


# ── V2 Strategy Registry ─────────────────────────────────────

V2_STRATEGY_REGISTRY: dict[str, type] = {
    "StratMomentumScore":    StratMomentumScore,
    "StratEmaCrossover":     StratEmaCrossover,
    "StratBreakoutRelaxed":  StratBreakoutRelaxed,
    "StratMeanReversionBB":  StratMeanReversionBB,
    "StratStochReversal":    StratStochReversal,
    "StratInsideBarBreakout": StratInsideBarBreakout,
    "StratRegimeAdaptive":   StratRegimeAdaptive,
}


# ── PARAM_GRID pour sweep ────────────────────────────────────

PARAM_GRID = {
    "StratMomentumScore": [
        {"threshold_low": 1, "threshold_high": 3, "sl_pct": 2.0, "tp_pct": 4.0},
        {"threshold_low": 1, "threshold_high": 3, "sl_pct": 1.5, "tp_pct": 3.0},
        {"threshold_low": 2, "threshold_high": 4, "sl_pct": 2.0, "tp_pct": 5.0},
    ],
    "StratEmaCrossover": [
        {"ema_fast": 9,  "ema_slow": 21, "use_regime_filter": True,  "sl_buffer_pct": 0.5, "tp_pct": 5.0},
        {"ema_fast": 9,  "ema_slow": 21, "use_regime_filter": False, "sl_buffer_pct": 0.5, "tp_pct": 5.0},
        {"ema_fast": 9,  "ema_slow": 50, "use_regime_filter": True,  "sl_buffer_pct": 1.0, "tp_pct": 8.0},
        {"ema_fast": 21, "ema_slow": 50, "use_regime_filter": True,  "sl_buffer_pct": 1.0, "tp_pct": 8.0},
    ],
    "StratBreakoutRelaxed": [
        {"lookback": 5,  "vol_breakout_min": 1.5, "use_compression": False, "sl_pct": 2.0, "tp_pct": 5.0},
        {"lookback": 10, "vol_breakout_min": 1.5, "use_compression": False, "sl_pct": 2.0, "tp_pct": 5.0},
        {"lookback": 10, "vol_breakout_min": 2.0, "use_compression": True,  "sl_pct": 2.0, "tp_pct": 5.0},
        {"lookback": 15, "vol_breakout_min": 2.5, "use_compression": True,  "sl_pct": 1.5, "tp_pct": 6.0},
    ],
    "StratMeanReversionBB": [
        {"rsi_oversold": 25, "rsi_overbought": 75, "bb_entry_low": 0.05, "bb_entry_high": 0.95, "sl_pct": 2.0, "tp_pct": 4.0},
        {"rsi_oversold": 30, "rsi_overbought": 70, "bb_entry_low": 0.10, "bb_entry_high": 0.90, "sl_pct": 1.5, "tp_pct": 3.0},
        {"rsi_oversold": 30, "rsi_overbought": 70, "bb_entry_low": 0.05, "bb_entry_high": 0.95, "sl_pct": 2.0, "tp_pct": 5.0},
        {"rsi_oversold": 35, "rsi_overbought": 65, "bb_entry_low": 0.10, "bb_entry_high": 0.90, "sl_pct": 2.5, "tp_pct": 6.0},
    ],
    "StratStochReversal": [
        {"oversold": 20, "overbought": 80, "vol_min": 1.0, "sl_pct": 2.0, "tp_pct": 4.0},
        {"oversold": 15, "overbought": 85, "vol_min": 1.2, "sl_pct": 1.5, "tp_pct": 3.0},
        {"oversold": 25, "overbought": 75, "vol_min": 0.8, "sl_pct": 2.0, "tp_pct": 5.0},
    ],
    "StratInsideBarBreakout": [
        {"vol_min": 1.0, "trend_filter": False, "atr_filter": False, "sl_pct": 2.0, "tp_pct": 5.0},
        {"vol_min": 1.5, "trend_filter": True,  "atr_filter": False, "sl_pct": 1.5, "tp_pct": 4.0},
        {"vol_min": 1.0, "trend_filter": True,  "atr_filter": True,  "sl_pct": 2.0, "tp_pct": 6.0},
        {"vol_min": 1.2, "trend_filter": False, "atr_filter": True,  "sl_pct": 2.5, "tp_pct": 8.0},
    ],
    "StratRegimeAdaptive": [
        {"rsi_bull": 55, "rsi_range_low": 30, "rsi_range_high": 70, "use_ranging_only": False, "sl_pct": 2.0, "tp_pct": 4.0},
        {"rsi_bull": 50, "rsi_range_low": 25, "rsi_range_high": 75, "use_ranging_only": False, "sl_pct": 1.5, "tp_pct": 3.0},
        {"rsi_bull": 55, "rsi_range_low": 30, "rsi_range_high": 70, "use_ranging_only": True,  "sl_pct": 2.0, "tp_pct": 5.0},
        {"rsi_bull": 60, "rsi_range_low": 35, "rsi_range_high": 65, "use_ranging_only": True,  "sl_pct": 2.5, "tp_pct": 6.0},
    ],
}


# ── Registry ──────────────────────────────────────────────────

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "grid": GridStrategy,
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
}


def get_all_variants(config_path: str = "config.yaml"
                     ) -> list[BaseStrategy]:
    """Instancie toutes les variantes de toutes les stratégies."""
    all_variants = []
    for name, cls in STRATEGY_REGISTRY.items():
        variants = cls.get_variants()
        for params in variants:
            strategy = cls(config_path=config_path, **params)
            all_variants.append(strategy)
        logger.info("%s — %d variantes", name, len(variants))

    logger.info("Total: %d variantes à tester", len(all_variants))
    return all_variants


# ── Standalone test ───────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s"
    )

    # Résumé des variantes
    print("=" * 60)
    print("STRATÉGIES ET VARIANTES")
    print("=" * 60)

    for name, cls in STRATEGY_REGISTRY.items():
        variants = cls.get_variants()
        print(f"\n{name.upper()} — {len(variants)} variantes:")
        for v in variants:
            strat = cls(**v)
            risk = strat.size_pct * strat.leverage * (strat.sl_pct / 100)
            print(f"  {strat}  → risque max par trade: {risk:.2f}% du portfolio")

    total = sum(len(cls.get_variants()) for cls in STRATEGY_REGISTRY.values())
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total} variantes × 4 assets × 2 timeframes = "
          f"{total * 4 * 2} backtests potentiels")

    # Test signaux avec données synthétiques
    print(f"\n{'=' * 60}")
    print("TEST SIGNAUX (données synthétiques)")
    print("=" * 60)

    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    trend = np.cumsum(np.random.randn(n) * 0.3)
    price = 40000 * np.exp(trend / 100)

    df = pd.DataFrame({
        "open": price * (1 + np.random.randn(n) * 0.001),
        "high": price * (1 + np.abs(np.random.randn(n)) * 0.005),
        "low": price * (1 - np.abs(np.random.randn(n)) * 0.005),
        "close": price,
        "volume": np.random.exponential(1000, n),
        "EMA9": price * (1 + np.random.randn(n) * 0.002),
        "EMA21": price * (1 + np.random.randn(n) * 0.003),
        "EMA50": price * (1 + np.random.randn(n) * 0.005),
        "EMA200": price * (1 - 0.02),
        "RSI_14": np.random.uniform(20, 80, n),
        "BB_lower": price * 0.97,
        "BB_upper": price * 1.03,
        "volume_ratio": np.random.exponential(1.0, n),
        "body_ratio": np.random.uniform(0, 1, n),
        "regime": np.random.choice(["bull", "bear", "ranging"], n, p=[0.5, 0.3, 0.2]),
        "ATR_percentile": np.random.uniform(0, 100, n),
        "compression": np.random.choice([True, False], n, p=[0.1, 0.9]),
    }, index=dates)

    for name, cls in STRATEGY_REGISTRY.items():
        strat = cls()
        signals = strat.generate_signals(df)
        print(f"  {strat.name}: {len(signals)} signaux")
        if signals:
            longs = sum(1 for s in signals if s.side == "long")
            shorts = sum(1 for s in signals if s.side == "short")
            print(f"    → {longs} long, {shorts} short")
