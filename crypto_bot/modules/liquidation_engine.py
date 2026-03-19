"""
Module 4 — LiquidationEngine
Simule le mécanisme exact des futures perpétuels en isolated margin.
Vérifie liquidation, funding, frais sur les bougies 5min pour une
précision maximale, même pour les stratégies 1h/4h.
Tout est exprimé en % du portfolio.
"""
from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Raison de sortie d'une position."""
    TAKE_PROFIT = "tp"
    TAKE_PROFIT_1 = "tp1"       # TP partiel (strat B momentum)
    STOP_LOSS = "sl"
    LIQUIDATION = "liquidation"
    SIGNAL = "signal"            # sortie sur condition technique
    TRAILING_STOP = "trailing"
    GRID_EXIT = "grid_exit"      # prix sort du canal (strat A)
    GLOBAL_STOP = "global_stop"  # portfolio < 50%
    END_OF_DATA = "end_of_data"


class LiquidationEngine:
    """Simule les positions futures perpétuels en isolated margin.
    Gère : liquidation, funding toutes les 8h, frais, slippage."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Frais en ratio (config en %, on convertit)
        self.fee_taker: float = self.cfg["fees"]["taker"] / 100
        self.fee_maker: float = self.cfg["fees"]["maker"] / 100
        self.slippage: float = self.cfg["fees"]["slippage"] / 100

        # Funding
        self.funding_default: float = self.cfg["funding"]["default_rate"] / 100
        self.funding_interval_h: int = self.cfg["funding"]["interval_hours"]

        # Margin
        self.maintenance_margin: float = self.cfg["margin"]["maintenance"] / 100

        # Stop global
        self.stop_global_pct: float = self.cfg["stop_global_portfolio"]

        logger.info(
            "LiquidationEngine initialisé — mm=%.4f, fee_taker=%.4f, "
            "funding=%.4f/8h, slippage=%.4f",
            self.maintenance_margin, self.fee_taker,
            self.funding_default, self.slippage
        )

    # ── Prix de liquidation ───────────────────────────────────

    def liquidation_price(self, entry: float, leverage: float,
                          side: str) -> float:
        """Calcule le prix de liquidation en isolated margin.

        Long:  liq = entry × (1 - 1/leverage + maintenance_margin)
        Short: liq = entry × (1 + 1/leverage - maintenance_margin)
        """
        if side == "long":
            return entry * (1 - 1 / leverage + self.maintenance_margin)
        elif side == "short":
            return entry * (1 + 1 / leverage - self.maintenance_margin)
        else:
            raise ValueError(f"side doit être 'long' ou 'short', reçu: {side}")

    # ── Vérification liquidation ──────────────────────────────

    @staticmethod
    def check_liquidation(candle_low: float, candle_high: float,
                          liq_price: float, side: str) -> bool:
        """Vérifie si une bougie touche le prix de liquidation.
        Long:  low  <= liq_price
        Short: high >= liq_price
        """
        if side == "long":
            return candle_low <= liq_price
        else:
            return candle_high >= liq_price

    # ── Frais ─────────────────────────────────────────────────

    def compute_fees(self, notional_pct: float,
                     fee_type: str = "taker") -> float:
        """Frais de transaction en % du portfolio.
        notional_pct = size_pct × leverage (exposition notionnelle).
        """
        rate = self.fee_taker if fee_type == "taker" else self.fee_maker
        return notional_pct * rate

    def compute_slippage(self, notional_pct: float) -> float:
        """Slippage en % du portfolio."""
        return notional_pct * self.slippage

    # ── Funding ───────────────────────────────────────────────

    def compute_funding_cost(self, notional_pct: float,
                             funding_rate: float,
                             side: str) -> float:
        """Coût funding en % du portfolio.

        Long  paie si funding_rate > 0 (marché normal)
        Short reçoit si funding_rate > 0
        Retour négatif = on reçoit.
        """
        if side == "long":
            return notional_pct * funding_rate
        else:
            return -notional_pct * funding_rate

    def _is_funding_time(self, timestamp: pd.Timestamp) -> bool:
        """Les funding rates s'appliquent toutes les 8h :
        00:00, 08:00, 16:00 UTC."""
        return timestamp.hour % self.funding_interval_h == 0 and timestamp.minute == 0

    def _get_funding_rate(self, timestamp: pd.Timestamp,
                          funding_rates: pd.Series | None) -> float:
        """Récupère le funding rate pour un timestamp donné.
        Fallback sur le taux par défaut."""
        if funding_rates is None or funding_rates.empty:
            return self.funding_default

        # Trouver le funding rate le plus proche <= timestamp
        valid = funding_rates.index[funding_rates.index <= timestamp]
        if len(valid) == 0:
            return self.funding_default

        return funding_rates.loc[valid[-1]]

    # ── Simulation complète d'une position ────────────────────

    def simulate_position(
        self,
        df_5m: pd.DataFrame,
        entry_idx: int,
        side: str,
        leverage: float,
        size_pct: float,
        sl_pct: float,
        tp_pct: float,
        funding_rates: pd.Series | None = None,
        tp_partial: dict | None = None,
        trailing_stop_pct: float | None = None,
        exit_condition: callable | None = None,
        entry_order_type: str = "maker",
        exit_tp_order_type: str = "maker",
        exit_sl_order_type: str = "taker",
    ) -> dict:
        """Simule une position bougie 5min par bougie 5min.

        Args:
            df_5m: DataFrame 5min avec OHLCV
            entry_idx: index numérique de la bougie d'entrée dans df_5m
            side: 'long' ou 'short'
            leverage: levier (ex: 5.0)
            size_pct: taille en % du portfolio (ex: 20.0 → 20%)
            sl_pct: stop loss en % depuis entrée (ex: 2.0 → -2%)
            tp_pct: take profit en % depuis entrée (ex: 3.0 → +3%)
            funding_rates: Series optionnelle de funding rates historiques
            tp_partial: dict optionnel pour TP partiel
                        {"pct": 3.0, "close_ratio": 0.5} → ferme 50% à +3%
            trailing_stop_pct: si défini, active un trailing stop après TP1
            exit_condition: callable(row) → bool, sortie sur signal technique
            entry_order_type: "maker" (ALO limit) ou "taker" (market)
            exit_tp_order_type: "maker" (TP en limit ALO) ou "taker"
            exit_sl_order_type: "taker" (SL toujours market) — rarement changé

        Retourne:
            entry_price, exit_price, exit_idx, exit_reason,
            pnl_pct (% du portfolio), fees_pct, funding_pct,
            duration_bars, duration_hours, peak_pnl_pct, trough_pnl_pct
        """
        if entry_idx >= len(df_5m):
            return self._empty_result(entry_idx)

        entry_row = df_5m.iloc[entry_idx]
        entry_price = float(entry_row["close"])

        # Slippage : maker (ALO) = pas de slippage, taker = slippage
        if entry_order_type == "taker":
            if side == "long":
                entry_price *= (1 + self.slippage)
            else:
                entry_price *= (1 - self.slippage)
        # ALO limit: pas de slippage, le prix est garanti

        # Notionnel en % du portfolio
        notional_pct = size_pct * leverage

        # Prix de liquidation
        liq_price = self.liquidation_price(entry_price, leverage, side)

        # Frais d'entrée : maker (ALO) = pas de slippage + frais maker
        entry_fees = self.compute_fees(notional_pct, entry_order_type)
        if entry_order_type == "taker":
            entry_fees += self.compute_slippage(notional_pct)

        # État de la position
        total_funding = 0.0
        remaining_ratio = 1.0  # 1.0 = position complète, 0.5 après TP partiel
        tp1_hit = False
        trailing_high = entry_price if side == "long" else entry_price
        trailing_low = entry_price if side == "short" else entry_price
        peak_pnl = 0.0
        trough_pnl = 0.0

        # Prix SL et TP
        if side == "long":
            sl_price = entry_price * (1 - sl_pct / 100)
            tp_price = entry_price * (1 + tp_pct / 100)
            tp1_price = entry_price * (1 + tp_partial["pct"] / 100) if tp_partial else None
        else:
            sl_price = entry_price * (1 + sl_pct / 100)
            tp_price = entry_price * (1 - tp_pct / 100)
            tp1_price = entry_price * (1 - tp_partial["pct"] / 100) if tp_partial else None

        # ── Boucle bougie par bougie ──────────────────────────

        exit_idx = len(df_5m) - 1
        exit_reason = ExitReason.END_OF_DATA
        exit_price = float(df_5m.iloc[-1]["close"])

        for i in range(entry_idx + 1, len(df_5m)):
            row = df_5m.iloc[i]
            low = float(row["low"])
            high = float(row["high"])
            close = float(row["close"])
            ts = df_5m.index[i]

            # 1. Vérifier liquidation (priorité maximale)
            if self.check_liquidation(low, high, liq_price, side):
                exit_idx = i
                exit_reason = ExitReason.LIQUIDATION
                exit_price = liq_price
                logger.debug("  LIQUIDATION à bar %d, prix=%.2f", i, liq_price)
                break

            # 2. Vérifier stop loss
            if side == "long" and low <= sl_price:
                exit_idx = i
                exit_reason = ExitReason.STOP_LOSS
                exit_price = sl_price * (1 - self.slippage)  # slippage défavorable
                break
            elif side == "short" and high >= sl_price:
                exit_idx = i
                exit_reason = ExitReason.STOP_LOSS
                exit_price = sl_price * (1 + self.slippage)
                break

            # 3. Vérifier TP partiel
            if tp_partial and not tp1_hit:
                if (side == "long" and high >= tp1_price) or \
                   (side == "short" and low <= tp1_price):
                    tp1_hit = True
                    remaining_ratio -= tp_partial.get("close_ratio", 0.5)
                    logger.debug("  TP1 partiel à bar %d", i)

                    # Si trailing stop activé après TP1
                    if trailing_stop_pct is not None:
                        if side == "long":
                            trailing_high = max(high, trailing_high)
                        else:
                            trailing_low = min(low, trailing_low)

            # 4. Trailing stop (si activé après TP1)
            if tp1_hit and trailing_stop_pct is not None:
                if side == "long":
                    trailing_high = max(high, trailing_high)
                    trail_sl = trailing_high * (1 - trailing_stop_pct / 100)
                    if low <= trail_sl:
                        exit_idx = i
                        exit_reason = ExitReason.TRAILING_STOP
                        exit_price = trail_sl * (1 - self.slippage)
                        break
                else:
                    trailing_low = min(low, trailing_low)
                    trail_sl = trailing_low * (1 + trailing_stop_pct / 100)
                    if high >= trail_sl:
                        exit_idx = i
                        exit_reason = ExitReason.TRAILING_STOP
                        exit_price = trail_sl * (1 + self.slippage)
                        break

            # 5. Take profit complet (si pas de TP partiel ou après TP1)
            if not tp_partial or tp1_hit:
                if (side == "long" and high >= tp_price) or \
                   (side == "short" and low <= tp_price):
                    exit_idx = i
                    exit_reason = ExitReason.TAKE_PROFIT
                    exit_price = tp_price
                    break

            # 6. Sortie sur condition technique
            if exit_condition is not None:
                try:
                    if exit_condition(row):
                        exit_idx = i
                        exit_reason = ExitReason.SIGNAL
                        exit_price = close
                        break
                except Exception:
                    pass

            # 7. Funding toutes les 8h
            if self._is_funding_time(ts):
                rate = self._get_funding_rate(ts, funding_rates)
                cost = self.compute_funding_cost(
                    notional_pct * remaining_ratio, rate, side
                )
                total_funding += cost

            # Tracker peak/trough PnL
            if side == "long":
                current_pnl = (close / entry_price - 1) * notional_pct * remaining_ratio
            else:
                current_pnl = (1 - close / entry_price) * notional_pct * remaining_ratio
            peak_pnl = max(peak_pnl, current_pnl)
            trough_pnl = min(trough_pnl, current_pnl)

        # ── Calcul du PnL final ───────────────────────────────

        # Frais de sortie — dépendent du type de sortie
        # TP = limit ALO possible (maker), SL/trailing/liquidation = market (taker)
        if exit_reason in (ExitReason.TAKE_PROFIT, ExitReason.TAKE_PROFIT_1):
            exit_fee_type = exit_tp_order_type
        else:
            exit_fee_type = exit_sl_order_type  # SL, trailing, signal = taker
        exit_fees = self.compute_fees(notional_pct * remaining_ratio, exit_fee_type)
        # Slippage seulement sur sorties market (taker)
        if exit_fee_type == "taker":
            exit_fees += self.compute_slippage(notional_pct * remaining_ratio)

        # Frais TP1 partiel (si applicable) — maker si ALO
        tp1_fees = 0.0
        if tp1_hit and tp_partial:
            tp1_ratio = tp_partial.get("close_ratio", 0.5)
            tp1_fees = self.compute_fees(notional_pct * tp1_ratio, exit_tp_order_type)

        if exit_reason == ExitReason.LIQUIDATION:
            # Liquidation = perte de 100% de la marge
            pnl_pct = -size_pct
            exit_fees = 0  # pas de frais supplémentaires sur liquidation
        else:
            # PnL brut en % du portfolio
            if side == "long":
                price_return = (exit_price / entry_price - 1)
            else:
                price_return = (1 - exit_price / entry_price)

            # PnL de la partie restante
            pnl_remaining = price_return * notional_pct * remaining_ratio

            # PnL de la partie TP1 (si applicable)
            pnl_tp1 = 0.0
            if tp1_hit and tp_partial:
                tp1_ratio = tp_partial.get("close_ratio", 0.5)
                if side == "long":
                    pnl_tp1 = (tp1_price / entry_price - 1) * notional_pct * tp1_ratio
                else:
                    pnl_tp1 = (1 - tp1_price / entry_price) * notional_pct * tp1_ratio

            pnl_pct = pnl_remaining + pnl_tp1

        # Total des coûts
        total_fees = entry_fees + exit_fees + tp1_fees
        net_pnl_pct = pnl_pct - total_fees - total_funding

        # Durée
        duration_bars = exit_idx - entry_idx
        entry_ts = df_5m.index[entry_idx]
        exit_ts = df_5m.index[min(exit_idx, len(df_5m) - 1)]
        duration_hours = (exit_ts - entry_ts).total_seconds() / 3600

        return {
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "entry_time": entry_ts,
            "exit_time": exit_ts,
            "entry_price": round(entry_price, 6),
            "exit_price": round(exit_price, 6),
            "liq_price": round(liq_price, 6),
            "side": side,
            "leverage": leverage,
            "size_pct": size_pct,
            "notional_pct": notional_pct,
            "exit_reason": exit_reason.value,
            "pnl_pct": round(net_pnl_pct, 6),
            "gross_pnl_pct": round(pnl_pct, 6),
            "fees_pct": round(total_fees, 6),
            "funding_pct": round(total_funding, 6),
            "duration_bars": duration_bars,
            "duration_hours": round(duration_hours, 2),
            "peak_pnl_pct": round(peak_pnl, 6),
            "trough_pnl_pct": round(trough_pnl, 6),
            "tp1_hit": tp1_hit,
        }

    # ── Vérification stop global ──────────────────────────────

    def check_global_stop(self, capital_pct: float) -> bool:
        """Retourne True si le capital est tombé sous le seuil global.
        capital_pct : capital actuel en % (100.0 = capital initial)."""
        triggered = capital_pct < self.stop_global_pct
        if triggered:
            logger.warning(
                "STOP GLOBAL PORTFOLIO — capital=%.2f%% < seuil=%.2f%%",
                capital_pct, self.stop_global_pct
            )
        return triggered

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _empty_result(entry_idx: int) -> dict:
        """Résultat vide pour une position qui ne peut pas être ouverte."""
        return {
            "entry_idx": entry_idx, "exit_idx": entry_idx,
            "entry_time": None, "exit_time": None,
            "entry_price": 0, "exit_price": 0, "liq_price": 0,
            "side": "", "leverage": 0, "size_pct": 0, "notional_pct": 0,
            "exit_reason": "invalid",
            "pnl_pct": 0, "gross_pnl_pct": 0, "fees_pct": 0, "funding_pct": 0,
            "duration_bars": 0, "duration_hours": 0,
            "peak_pnl_pct": 0, "trough_pnl_pct": 0, "tp1_hit": False,
        }

    def summarize_trades(self, trades: list[dict]) -> dict:
        """Résumé statistique d'une liste de trades.
        Tout en % du portfolio."""

        if not trades:
            return {
                "nb_trades": 0, "nb_liquidations": 0,
                "total_pnl_pct": 0, "total_fees_pct": 0,
                "total_funding_pct": 0, "win_rate": 0,
            }

        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "nb_trades": len(trades),
            "nb_liquidations": sum(1 for t in trades if t["exit_reason"] == "liquidation"),
            "total_pnl_pct": round(sum(pnls), 4),
            "total_fees_pct": round(sum(t["fees_pct"] for t in trades), 4),
            "total_funding_pct": round(sum(t["funding_pct"] for t in trades), 4),
            "win_rate": round(len(wins) / len(trades), 4) if trades else 0,
            "avg_win_pct": round(np.mean(wins), 4) if wins else 0,
            "avg_loss_pct": round(np.mean(losses), 4) if losses else 0,
            "best_trade_pct": round(max(pnls), 4),
            "worst_trade_pct": round(min(pnls), 4),
            "avg_duration_h": round(np.mean([t["duration_hours"] for t in trades]), 2),
        }


# ── Standalone test ───────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s"
    )

    engine = LiquidationEngine()

    # Test prix de liquidation
    print("=== Prix de liquidation ===")
    for lev in [2, 3, 5, 10, 15, 20]:
        liq_l = engine.liquidation_price(50000, lev, "long")
        liq_s = engine.liquidation_price(50000, lev, "short")
        dist_l = (50000 - liq_l) / 50000 * 100
        dist_s = (liq_s - 50000) / 50000 * 100
        print(f"  Levier {lev:2d}x — Long liq: {liq_l:>10.2f} (-{dist_l:.2f}%)"
              f"  |  Short liq: {liq_s:>10.2f} (+{dist_s:.2f}%)")

    # Test frais
    print("\n=== Frais sur position 20% × 5x ===")
    notional = 20.0 * 5
    entry_fee = engine.compute_fees(notional, "taker")
    slip = engine.compute_slippage(notional)
    funding = engine.compute_funding_cost(notional, 0.0001, "long")
    print(f"  Frais entrée: {entry_fee:.4f}% du portfolio")
    print(f"  Slippage:     {slip:.4f}% du portfolio")
    print(f"  Funding/8h:   {funding:.4f}% du portfolio")
    print(f"  Aller-retour: {(entry_fee + slip) * 2:.4f}% du portfolio")

    # Test avec données synthétiques
    print("\n=== Simulation position ===")
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    price = 50000 + np.cumsum(np.random.randn(n) * 10)

    df = pd.DataFrame({
        "open": price,
        "high": price + np.abs(np.random.randn(n) * 5),
        "low": price - np.abs(np.random.randn(n) * 5),
        "close": price + np.random.randn(n) * 3,
        "volume": np.random.exponential(100, n),
    }, index=dates)

    result = engine.simulate_position(
        df, entry_idx=10, side="long",
        leverage=5.0, size_pct=20.0,
        sl_pct=2.0, tp_pct=3.0,
    )
    print(f"  Entrée: {result['entry_price']:.2f}")
    print(f"  Sortie: {result['exit_price']:.2f} ({result['exit_reason']})")
    print(f"  PnL net: {result['pnl_pct']:+.4f}% du portfolio")
    print(f"  Frais: {result['fees_pct']:.4f}%")
    print(f"  Durée: {result['duration_hours']:.1f}h")
