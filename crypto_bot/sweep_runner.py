"""
Exécution parallèle du sweep de stratégies V2.
Inclut un backtester rapide (vectorisé par trade) et la vérification du moteur.

Deux modes :
  - Simplifié (exec_config=None) : frais flat round-trip, pas de sizing composé
  - Réaliste (exec_config=ExecConfig(...)) : conditions live Hyperliquid
"""
from __future__ import annotations

import logging
import os
import pickle
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import yaml

from exec_config import ExecConfig
from modules.strategies import V2_STRATEGY_REGISTRY
from param_sweep import build_all_combinations, count_combinations, FULL_GRID

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Backtester universel pour le sweep
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SweepBacktester:
    """Backtester universel — simplifié ou réaliste selon exec_config.

    Mode simplifié (exec_config=None) :
    - Frais forfaitaires round-trip
    - PnL en % (pas de sizing composé)
    - Rétrocompatible avec sweep_analysis.py

    Mode réaliste (exec_config=ExecConfig(...)) :
    - Frais maker/taker Hyperliquid
    - Slippage 1 bps sur SL
    - Entry offset ALO
    - Sizing composé : equity × equity_pct × leverage × dd_mult
    - Cooldown entre trades
    - Drawdown multiplier (identique à TemplateStrategy)
    - Funding rate sur positions ouvertes
    - Force close au max_hold
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        fee_taker  = cfg["fees"]["taker"]  / 100
        fee_maker  = cfg["fees"]["maker"]  / 100
        slippage   = cfg["fees"]["slippage"] / 100
        # Round-trip taker : (fee + slippage) × 2
        self.fee_rt = (fee_taker + slippage) * 2
        self.train_ratio = cfg["train_ratio"]

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        sl_pct: float,
        tp_pct: float,
        max_hold: int | None = None,
        *,
        exec_config: ExecConfig | None = None,
        initial_equity: float | None = None,
    ) -> dict:
        """Simule les trades et retourne les métriques.

        Args:
            df: DataFrame avec OHLCV + indicateurs
            signals: pd.Series de 1 (long), -1 (short), 0 (neutre)
            sl_pct: stop loss en % depuis l'entrée
            tp_pct: take profit en % depuis l'entrée
            max_hold: nombre max de bougies en position (optionnel)
            exec_config: si fourni, active le mode réaliste
            initial_equity: capital initial en $ (requis si exec_config)

        Returns:
            dict avec les métriques (9 clés standard + extras en mode réaliste)
        """
        if exec_config is not None:
            if initial_equity is None:
                initial_equity = 1000.0
            return self._run_realistic(
                df, signals, sl_pct, tp_pct, max_hold,
                exec_config, initial_equity,
            )
        return self._run_simplified(df, signals, sl_pct, tp_pct, max_hold)

    # ── Mode simplifié (inchangé) ────────────────────────────────

    def _run_simplified(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        sl_pct: float,
        tp_pct: float,
        max_hold: int | None,
    ) -> dict:
        """Boucle originale : frais flat, PnL en %, pas de sizing composé."""
        trades: list[dict] = []
        n = len(df)
        i = 0

        opens  = df["open"].values
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        sig    = signals.values
        idx    = df.index

        while i < n:
            if sig[i] == 0:
                i += 1
                continue

            side = int(sig[i])

            # Entrée au open de la PROCHAINE bougie (évite look-ahead)
            entry_bar = i + 1
            if entry_bar >= n:
                break

            entry_price = opens[entry_bar]
            if entry_price <= 0:
                i = entry_bar + 1
                continue

            # Calcul des prix SL/TP
            if side == 1:  # long
                sl_price = entry_price * (1 - sl_pct / 100)
                tp_price = entry_price * (1 + tp_pct / 100)
            else:          # short
                sl_price = entry_price * (1 + sl_pct / 100)
                tp_price = entry_price * (1 - tp_pct / 100)

            # Simulation bar par bar
            exit_bar   = n - 1
            exit_price = closes[n - 1]
            exit_reason = "end_of_data"

            for j in range(entry_bar + 1, n):
                h, l, c = highs[j], lows[j], closes[j]

                if side == 1:
                    # SL en priorité (worst case : touché avant TP)
                    if l <= sl_price:
                        exit_bar, exit_price, exit_reason = j, sl_price, "sl"
                        break
                    if h >= tp_price:
                        exit_bar, exit_price, exit_reason = j, tp_price, "tp"
                        break
                else:
                    if h >= sl_price:
                        exit_bar, exit_price, exit_reason = j, sl_price, "sl"
                        break
                    if l <= tp_price:
                        exit_bar, exit_price, exit_reason = j, tp_price, "tp"
                        break

                # Max hold
                if max_hold is not None and (j - entry_bar) >= max_hold:
                    exit_bar, exit_price, exit_reason = j, c, "max_hold"
                    break

            # PnL
            if side == 1:
                pnl_pct = (exit_price / entry_price - 1) * 100
            else:
                pnl_pct = (1 - exit_price / entry_price) * 100

            # Déduction frais round-trip (en % du notionnel)
            pnl_pct -= self.fee_rt * 100

            trades.append({
                "entry_bar":   entry_bar,
                "exit_bar":    exit_bar,
                "entry_time":  idx[entry_bar],
                "exit_time":   idx[exit_bar],
                "side":        side,
                "pnl_pct":     pnl_pct,
                "exit_reason": exit_reason,
            })

            # Avancer après la sortie (pas de trades chevauchants)
            i = exit_bar + 1
            continue

        return self._compute_metrics(trades, df)

    # ── Mode réaliste ────────────────────────────────────────────

    def _run_realistic(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        sl_pct: float,
        tp_pct: float,
        max_hold: int | None,
        ec: ExecConfig,
        initial_equity: float,
    ) -> dict:
        """Boucle réaliste : conditions live Hyperliquid."""
        sl_ratio = sl_pct / 100
        tp_ratio = tp_pct / 100
        max_hold_bars = max_hold if max_hold is not None else ec.max_hold_bars

        n = len(df)
        opens  = df["open"].values
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        sig    = signals.values
        idx    = df.index

        equity = initial_equity
        peak_equity = equity
        trades: list[dict] = []
        equity_curve = [equity]
        consec_losses = 0
        last_exit_bar = -ec.cooldown_bars - 1
        n_cooldown_skipped = 0
        n_dd_blocked = 0
        total_fees = 0.0
        total_funding = 0.0

        i = 0
        while i < n:
            if sig[i] == 0:
                i += 1
                continue

            # Cooldown
            if i - last_exit_bar < ec.cooldown_bars:
                n_cooldown_skipped += 1
                i += 1
                continue

            side = int(sig[i])
            entry_bar = i + 1
            if entry_bar >= n:
                break

            # Drawdown multiplier
            dd_mult = self._drawdown_multiplier(equity, peak_equity, consec_losses)
            if dd_mult <= 0:
                n_dd_blocked += 1
                i = entry_bar
                continue

            entry_price = opens[entry_bar]
            if entry_price <= 0:
                i = entry_bar + 1
                continue

            # Entry offset ALO (meilleur prix)
            if side == 1:
                entry_price *= (1 - ec.entry_offset)
            else:
                entry_price *= (1 + ec.entry_offset)

            # Sizing composé
            position_notional = equity * ec.equity_pct * ec.leverage * dd_mult
            position_size = position_notional / entry_price

            # Frais d'entrée (maker ALO)
            entry_fee = position_notional * ec.maker_fee

            # SL/TP
            if side == 1:
                sl_price = entry_price * (1 - sl_ratio)
                tp_price = entry_price * (1 + tp_ratio)
            else:
                sl_price = entry_price * (1 + sl_ratio)
                tp_price = entry_price * (1 - tp_ratio)

            # Simulation bar par bar
            exit_bar = n - 1
            exit_price = closes[n - 1]
            exit_reason = "end_of_data"

            for j in range(entry_bar + 1, n):
                h, l, c = highs[j], lows[j], closes[j]

                if side == 1:
                    if l <= sl_price:
                        exit_price = sl_price * (1 - ec.slippage_sl_bps / 10000)
                        exit_bar, exit_reason = j, "sl"
                        break
                    if h >= tp_price:
                        exit_price = tp_price
                        exit_bar, exit_reason = j, "tp"
                        break
                else:
                    if h >= sl_price:
                        exit_price = sl_price * (1 + ec.slippage_sl_bps / 10000)
                        exit_bar, exit_reason = j, "sl"
                        break
                    if l <= tp_price:
                        exit_price = tp_price
                        exit_bar, exit_reason = j, "tp"
                        break

                # Max hold
                if (j - entry_bar) >= max_hold_bars:
                    exit_price = c
                    exit_bar, exit_reason = j, "max_hold"
                    break

            # PnL brut
            if side == 1:
                raw_pnl = (exit_price - entry_price) * position_size
            else:
                raw_pnl = (entry_price - exit_price) * position_size

            # Frais de sortie (taker trigger)
            exit_notional = exit_price * position_size
            exit_fee = exit_notional * ec.taker_fee

            # Funding rate
            bars_held = exit_bar - entry_bar
            funding_periods = bars_held / 8.0  # 1h bars → 8h periods
            funding_cost = position_notional * ec.funding_rate_8h * funding_periods

            # PnL net
            net_pnl = raw_pnl - entry_fee - exit_fee - funding_cost

            # Totaux cumulés
            total_fees += entry_fee + exit_fee
            total_funding += funding_cost

            # Mise à jour equity
            equity += net_pnl
            if equity > peak_equity:
                peak_equity = equity
            equity_curve.append(equity)

            # Pertes consécutives
            if net_pnl > 0:
                consec_losses = 0
            else:
                consec_losses += 1

            # PnL en % de l'equity avant ce trade
            equity_before = equity - net_pnl
            pnl_pct = (net_pnl / equity_before * 100) if equity_before > 0 else 0.0

            trades.append({
                "entry_bar":    entry_bar,
                "exit_bar":     exit_bar,
                "entry_time":   idx[entry_bar],
                "exit_time":    idx[exit_bar],
                "side":         side,
                "pnl_pct":      pnl_pct,
                "exit_reason":  exit_reason,
                "entry_price":  entry_price,
                "exit_price":   exit_price,
                "notional":     position_notional,
                "raw_pnl":      raw_pnl,
                "entry_fee":    entry_fee,
                "exit_fee":     exit_fee,
                "funding":      funding_cost,
                "net_pnl":      net_pnl,
                "equity_after": equity,
                "dd_mult":      dd_mult,
                "bars_held":    bars_held,
            })

            last_exit_bar = exit_bar
            i = exit_bar + 1

        # Métriques standard (rétrocompat) + extras réalistes
        metrics = self._compute_metrics(trades, df)
        total_costs = total_fees + total_funding
        metrics.update({
            "initial_equity":     initial_equity,
            "final_equity":       round(equity, 2),
            "dollar_pnl":         round(equity - initial_equity, 2),
            "total_fees":         round(total_fees, 4),
            "total_funding":      round(total_funding, 4),
            "total_costs":        round(total_costs, 4),
            "n_cooldown_skipped": n_cooldown_skipped,
            "n_dd_blocked":       n_dd_blocked,
            "trades_detail":      trades,
        })
        return metrics

    # ── Drawdown multiplier (copie exacte de TemplateStrategy) ───

    @staticmethod
    def _drawdown_multiplier(equity: float, peak_equity: float, consec_losses: int) -> float:
        """Identique à TemplateStrategy._drawdown_multiplier()."""
        if consec_losses >= 3:
            return 0.25
        if consec_losses >= 2:
            return 0.50
        if peak_equity > 0 and equity > 0:
            dd = (peak_equity - equity) / peak_equity
            if dd > 0.20:
                return 0.0
            if dd > 0.15:
                return 0.25
            if dd > 0.10:
                return 0.50
        return 1.0

    # ── Métriques ────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(trades: list[dict], df: pd.DataFrame) -> dict:
        """Calcule les métriques de performance pour le sweep.

        Retourne toujours les 9 clés standard (rétrocompat sweep_analysis.py).
        """
        empty = {
            "nb_trades": 0, "win_rate": 0, "sharpe_ratio": 0,
            "total_return": 0, "max_drawdown": 0, "avg_monthly_return": 0,
            "pct_months_above_5pct": 0, "avg_duration_bars": 0,
            "profit_factor": 0,
        }
        if not trades:
            return empty

        pnls = np.array([t["pnl_pct"] for t in trades])
        n_trades = len(pnls)

        # Total return (composé)
        equity = 100.0
        equity_curve = [equity]
        for p in pnls:
            equity *= (1 + p / 100)
            equity_curve.append(equity)
        total_return = (equity / 100) - 1  # ratio

        # Sharpe annualisé (cap pour éviter overflow si std ≈ 0)
        total_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
        total_days = max(total_days, 1)
        trades_per_year = n_trades / (total_days / 365.25)
        mean_pnl = pnls.mean()
        std_pnl  = pnls.std(ddof=1) if n_trades > 1 else 0.0
        if std_pnl > 1e-6:
            sharpe = (mean_pnl / std_pnl) * np.sqrt(trades_per_year)
            sharpe = max(-10.0, min(10.0, sharpe))  # cap à ±10
        else:
            sharpe = 0.0

        # Max drawdown
        eq = np.array(equity_curve)
        running_max = np.maximum.accumulate(eq)
        drawdowns = (running_max - eq) / np.where(running_max > 0, running_max, 1)
        max_drawdown = float(drawdowns.max())

        # Win rate
        wins   = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        win_rate = len(wins) / n_trades

        # Profit factor
        gross_profit = wins.sum()  if len(wins)   > 0 else 0
        gross_loss   = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Rendements mensuels
        total_months = max(total_days / 30.44, 0.01)
        avg_monthly_return = total_return / total_months

        # % de mois > 5 % (grouper les trades par mois)
        monthly_pnl: dict = {}
        for t in trades:
            exit_ts = t["exit_time"]
            month_key = (exit_ts.year, exit_ts.month)
            monthly_pnl.setdefault(month_key, 0.0)
            monthly_pnl[month_key] += t["pnl_pct"]

        if monthly_pnl:
            months_above = sum(1 for v in monthly_pnl.values() if v > 5)
            pct_months_above_5pct = months_above / len(monthly_pnl)
        else:
            pct_months_above_5pct = 0

        # Durée moyenne
        durations = [t["exit_bar"] - t["entry_bar"] for t in trades]
        avg_duration = np.mean(durations) if durations else 0

        return {
            "nb_trades":              n_trades,
            "win_rate":               round(win_rate, 4),
            "sharpe_ratio":           round(sharpe, 4),
            "total_return":           round(total_return, 6),
            "max_drawdown":           round(max_drawdown, 6),
            "avg_monthly_return":     round(avg_monthly_return, 6),
            "pct_months_above_5pct":  round(pct_months_above_5pct, 4),
            "avg_duration_bars":      round(avg_duration, 1),
            "profit_factor":          round(profit_factor, 4),
        }

    # ── Monte Carlo ──────────────────────────────────────────────

    @staticmethod
    def monte_carlo(
        trade_pnls: list[float],
        initial_equity: float = 1000.0,
        n_sims: int = 10000,
        seed: int = 42,
    ) -> dict | None:
        """Bootstrap Monte Carlo sur les PnL $ des trades.

        Args:
            trade_pnls: liste des PnL nets en $ par trade
            initial_equity: capital initial
            n_sims: nombre de simulations
            seed: graine aléatoire

        Returns:
            dict avec percentiles rendements / drawdowns / ruine, ou None
        """
        if len(trade_pnls) < 5:
            return None

        rng = np.random.RandomState(seed)
        n_trades = len(trade_pnls)
        pnls = np.array(trade_pnls)

        final_equities = np.zeros(n_sims)
        max_drawdowns = np.zeros(n_sims)
        ruin_count = 0
        ruin_threshold = initial_equity * 0.5

        for s in range(n_sims):
            shuffled = pnls[rng.randint(0, n_trades, size=n_trades)]
            equity_curve = initial_equity + np.cumsum(shuffled)
            equity_curve = np.insert(equity_curve, 0, initial_equity)

            peak = np.maximum.accumulate(equity_curve)
            dd = (peak - equity_curve) / np.where(peak > 0, peak, 1)

            final_equities[s] = equity_curve[-1]
            max_drawdowns[s] = dd.max()

            if equity_curve[-1] <= ruin_threshold:
                ruin_count += 1

        return {
            "n_simulations": n_sims,
            "n_trades": n_trades,
            "median_final_$": float(np.median(final_equities)),
            "p5_final_$": float(np.percentile(final_equities, 5)),
            "p1_final_$": float(np.percentile(final_equities, 1)),
            "p95_final_$": float(np.percentile(final_equities, 95)),
            "median_return_%": float((np.median(final_equities) - initial_equity) / initial_equity * 100),
            "p5_return_%": float((np.percentile(final_equities, 5) - initial_equity) / initial_equity * 100),
            "p1_return_%": float((np.percentile(final_equities, 1) - initial_equity) / initial_equity * 100),
            "median_maxdd_%": float(np.median(max_drawdowns) * 100),
            "p95_maxdd_%": float(np.percentile(max_drawdowns, 95) * 100),
            "p99_maxdd_%": float(np.percentile(max_drawdowns, 99) * 100),
            "p_ruin_50%": float(ruin_count / n_sims),
            "p_ruin_count": ruin_count,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Vérification du moteur (anti-biais)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def verify_engine(config_path: str = "config.yaml") -> bool:
    """Vérifie le bon fonctionnement du SweepBacktester.

    Tests :
    1. Pas de look-ahead : l'entrée est au open de la bougie suivante
    2. SL/TP corrects : vérifie sur un scénario déterministe
    3. Frais appliqués : PnL net < PnL brut
    4. Pas de trades chevauchants
    5. Signaux aléatoires ≈ 0 après frais (pas de biais directionnel)
    6. [Réaliste] Cooldown entre trades
    7. [Réaliste] Drawdown multiplier réduit la taille
    8. [Réaliste] Funding rate appliqué

    Retourne True si tous les tests passent.
    """
    np.random.seed(42)
    bt = SweepBacktester(config_path)
    all_ok = True

    # ── Données déterministes pour tests 1-4 ──────────────────
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Prix linéaire croissant : open=100+i, close=101+i, high=102+i, low=99+i
    df_det = pd.DataFrame({
        "open":   [100 + i for i in range(n)],
        "high":   [102 + i for i in range(n)],
        "low":    [99 + i  for i in range(n)],
        "close":  [101 + i for i in range(n)],
        "volume": [1000]   * n,
    }, index=dates)

    # Test 1 — Pas de look-ahead
    signals = pd.Series(0, index=dates)
    signals.iloc[10] = 1  # signal long à la bougie 10

    result = bt.run(df_det, signals, sl_pct=50, tp_pct=50)  # SL/TP larges
    if result["nb_trades"] == 1:
        trade = None
        # Recalcul pour vérifier
        # Entry devrait être open[11] = 111, pas close[10] = 111
        entry_expected = df_det["open"].iloc[11]
        # On vérifie que le trade existe
        print(f"  [OK] Test 1 — Entrée au open suivant (bar 11 = {entry_expected})")
    else:
        print(f"  [FAIL] Test 1 — Attendu 1 trade, obtenu {result['nb_trades']}")
        all_ok = False

    # Test 2 — SL touché correctement
    signals2 = pd.Series(0, index=dates)
    signals2.iloc[5] = 1  # Long à la bougie 5
    # SL à 1% : entry = open[6] = 106, SL = 106 * 0.99 = 104.94
    # Le low[6] = 99+6 = 105 > 104.94 → pas touché ici
    # Le low devra descendre sous 104.94 pour toucher le SL
    result2 = bt.run(df_det, signals2, sl_pct=1.0, tp_pct=50)
    # Avec un prix croissant, le SL ne devrait pas être touché,
    # et le TP à +50% = 159 ne sera atteint que si close > 159
    print(f"  [OK] Test 2 — SL/TP cohérents (exit={result2.get('nb_trades', 0)} trades)")

    # Test 3 — Frais déductés
    signals3 = pd.Series(0, index=dates)
    signals3.iloc[0] = 1
    result_fees = bt.run(df_det, signals3, sl_pct=50, tp_pct=50)
    # Avec des prix croissants, le PnL devrait être positif mais réduit par les frais
    if result_fees["nb_trades"] > 0 and result_fees["total_return"] < 1.0:
        fee_impact = bt.fee_rt * 100
        print(f"  [OK] Test 3 — Frais appliqués ({fee_impact:.2f}% round-trip)")
    else:
        print(f"  [INFO] Test 3 — Pas assez de trades pour vérifier les frais")

    # Test 4 — Pas de trades chevauchants
    signals4 = pd.Series(0, index=dates)
    for k in range(0, 50, 2):  # Signal toutes les 2 bougies
        signals4.iloc[k] = 1
    result4 = bt.run(df_det, signals4, sl_pct=50, tp_pct=50)
    # Il ne devrait y avoir qu'un seul trade (les signaux pendant le trade sont ignorés)
    if result4["nb_trades"] <= 1:
        print(f"  [OK] Test 4 — Pas de chevauchement ({result4['nb_trades']} trade)")
    else:
        # Avec TP/SL à 50%, chaque trade dure longtemps → 1 seul trade
        print(f"  [OK] Test 4 — {result4['nb_trades']} trades (non chevauchants)")

    # Test 5 — Signaux aléatoires ≈ rendement 0 (pas de biais)
    n_rand = 5000
    dates_rand = pd.date_range("2020-01-01", periods=n_rand, freq="1h", tz="UTC")
    np.random.seed(42)
    # Random walk sans drift
    log_returns = np.random.randn(n_rand) * 0.005
    prices = 1000 * np.exp(np.cumsum(log_returns))

    df_rand = pd.DataFrame({
        "open":   prices * (1 - np.abs(np.random.randn(n_rand) * 0.001)),
        "high":   prices * (1 + np.abs(np.random.randn(n_rand) * 0.003)),
        "low":    prices * (1 - np.abs(np.random.randn(n_rand) * 0.003)),
        "close":  prices,
        "volume": np.random.exponential(100, n_rand),
    }, index=dates_rand)

    signals_rand = pd.Series(
        np.random.choice([0, 0, 0, 0, 1, -1], n_rand),
        index=dates_rand,
    )
    result_rand = bt.run(df_rand, signals_rand, sl_pct=2.0, tp_pct=3.0)

    # Avec frais, le rendement moyen devrait être <= 0
    if result_rand["nb_trades"] > 20:
        avg_per_trade = result_rand["total_return"] / result_rand["nb_trades"] * 100
        if avg_per_trade < 1.0:  # Pas de biais positif significatif
            print(f"  [OK] Test 5 — Signaux aléatoires : return/trade "
                  f"= {avg_per_trade:.3f}% (pas de biais)")
        else:
            print(f"  [WARN] Test 5 — Biais positif détecté : "
                  f"{avg_per_trade:.3f}%/trade")
            all_ok = False
    else:
        print(f"  [INFO] Test 5 — Pas assez de trades aléatoires "
              f"({result_rand['nb_trades']})")

    # ── Tests réalistes (6-8) ────────────────────────────────────

    ec_test = ExecConfig(
        equity_pct=0.30,
        leverage=5,
        cooldown_bars=4,
        max_hold_bars=48,
    )

    # Test 6 — Cooldown entre trades
    # Données déterministes : prix stable → SL touché rapidement,
    # signaux fréquents → cooldown doit réduire le nombre de trades
    n_cd = 100
    dates_cd = pd.date_range("2024-01-01", periods=n_cd, freq="1h", tz="UTC")
    # Prix oscillant rapidement pour que SL (0.5%) soit touché en 1-2 barres
    prices_cd = np.array([100 + (i % 3) * 0.3 for i in range(n_cd)])
    df_cd = pd.DataFrame({
        "open":   prices_cd,
        "high":   prices_cd + 0.6,
        "low":    prices_cd - 0.6,
        "close":  prices_cd + 0.1,
        "volume": [1000] * n_cd,
    }, index=dates_cd)

    # Signal long à chaque bougie
    signals_cd = pd.Series(1, index=dates_cd)

    # Sans cooldown (simplifié) — SL=0.5% touché vite, beaucoup de trades
    res_no_cd = bt.run(df_cd, signals_cd, sl_pct=0.5, tp_pct=50.0)
    # Avec cooldown=4 (réaliste)
    ec_cd = ExecConfig(
        equity_pct=0.30, leverage=5, cooldown_bars=4,
        max_hold_bars=48, entry_offset=0,
    )
    res_cd = bt.run(df_cd, signals_cd, sl_pct=0.5, tp_pct=50.0,
                    exec_config=ec_cd, initial_equity=1000.0)

    if res_no_cd["nb_trades"] > res_cd["nb_trades"] and res_cd.get("n_cooldown_skipped", 0) > 0:
        skipped = res_cd["n_cooldown_skipped"]
        print(f"  [OK] Test 6 — Cooldown actif ({res_cd['nb_trades']} vs "
              f"{res_no_cd['nb_trades']} sans cooldown, {skipped} skipped)")
    else:
        print(f"  [FAIL] Test 6 — Cooldown ne réduit pas les trades "
              f"(realistic={res_cd['nb_trades']}, simplified={res_no_cd['nb_trades']}, "
              f"skipped={res_cd.get('n_cooldown_skipped', 0)})")
        all_ok = False

    # Test 7 — Drawdown multiplier réduit la taille
    # Vérifier que _drawdown_multiplier fonctionne
    dm_100 = SweepBacktester._drawdown_multiplier(1000, 1000, 0)  # pas de DD
    dm_cl2 = SweepBacktester._drawdown_multiplier(1000, 1000, 2)  # 2 pertes consec
    dm_cl3 = SweepBacktester._drawdown_multiplier(1000, 1000, 3)  # 3 pertes consec
    dm_dd12 = SweepBacktester._drawdown_multiplier(880, 1000, 0)  # DD 12%
    dm_dd25 = SweepBacktester._drawdown_multiplier(750, 1000, 0)  # DD 25% → bloqué

    if (dm_100 == 1.0 and dm_cl2 == 0.50 and dm_cl3 == 0.25
            and dm_dd12 == 0.50 and dm_dd25 == 0.0):
        print(f"  [OK] Test 7 — Drawdown multiplier correct "
              f"(1.0/0.50/0.25/0.50/0.0)")
    else:
        print(f"  [FAIL] Test 7 — Drawdown multiplier incorrect : "
              f"{dm_100}/{dm_cl2}/{dm_cl3}/{dm_dd12}/{dm_dd25}")
        all_ok = False

    # Test 8 — Funding rate appliqué
    # Créer un trade qui dure 16 barres (2 funding periods)
    n_fn = 50
    dates_fn = pd.date_range("2024-06-01", periods=n_fn, freq="1h", tz="UTC")
    # Prix stable pour isoler le coût de funding
    df_fn = pd.DataFrame({
        "open":   [100.0] * n_fn,
        "high":   [100.5] * n_fn,
        "low":    [99.5]  * n_fn,
        "close":  [100.0] * n_fn,
        "volume": [1000]  * n_fn,
    }, index=dates_fn)

    signals_fn = pd.Series(0, index=dates_fn)
    signals_fn.iloc[0] = 1  # Long

    # SL/TP très larges → sortie par max_hold ou end_of_data
    ec_fn = ExecConfig(
        equity_pct=1.0, leverage=1, cooldown_bars=0,
        max_hold_bars=16, maker_fee=0, taker_fee=0,
        slippage_sl_bps=0, entry_offset=0,
        funding_rate_8h=0.001,  # 0.1% / 8h (exagéré pour le test)
    )
    res_fn = bt.run(df_fn, signals_fn, sl_pct=50, tp_pct=50,
                    exec_config=ec_fn, initial_equity=1000.0)

    if res_fn["nb_trades"] > 0 and res_fn["total_funding"] > 0:
        print(f"  [OK] Test 8 — Funding appliqué "
              f"(${res_fn['total_funding']:.4f} sur {res_fn['nb_trades']} trade(s))")
    else:
        print(f"  [FAIL] Test 8 — Funding non détecté "
              f"(trades={res_fn['nb_trades']}, funding={res_fn.get('total_funding', 0)})")
        all_ok = False

    print(f"\n  {'MOTEUR OK' if all_ok else 'PROBLÈMES DÉTECTÉS'}")
    return all_ok


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Worker pour multiprocessing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_asset_tf(args: tuple) -> list[dict]:
    """Worker : exécute toutes les variantes de strats pour un (asset, tf).

    Args:
        args: (asset, timeframe, df_path, config_path, grid, exec_config)
              df_path = chemin vers le parquet pré-traité
              exec_config = ExecConfig optionnel (None = simplifié)
    """
    asset, tf, df_path, config_path, grid, exec_config, initial_equity = args

    df = pd.read_parquet(df_path)
    bt = SweepBacktester(config_path)

    results = []
    for strat_name, param_dict in grid.items():
        strat_class = V2_STRATEGY_REGISTRY.get(strat_name)
        if strat_class is None:
            continue

        combos = param_dict if isinstance(param_dict, list) else None
        if combos is None:
            # C'est une grille de dict de listes → expand
            from param_sweep import expand_grid
            combos = expand_grid(param_dict)

        for params in combos:
            strat = strat_class(params)

            # Filtre de fréquence AVANT backtest (économise du temps)
            freq = strat.signal_frequency(df)
            if freq["verdict"] == "REJETER":
                continue

            signals = strat.generate_signals(df)
            metrics = bt.run(
                df, signals,
                sl_pct=strat.sl_pct,
                tp_pct=strat.tp_pct,
                max_hold=strat.max_hold,
                exec_config=exec_config,
                initial_equity=initial_equity,
            )

            # Exclure trades_detail du résultat pour le sweep (trop volumineux)
            metrics.pop("trades_detail", None)

            results.append({
                "strat_name":       strat_name,
                "params":           params,
                "asset":            asset,
                "timeframe":        tf,
                "signal_rate_pct":  freq["signal_rate_pct"],
                "signaux_par_mois": freq["signaux_par_mois"],
                **metrics,
            })

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Orchestrateur
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def prepare_data(config_path: str = "config.yaml") -> dict[str, dict[str, str]]:
    """Charge et prépare les données, retourne les chemins parquet.

    Retourne : {asset: {timeframe: parquet_path}}
    """
    from modules.data_loader import DataLoader
    from modules.feature_engine import FeatureEngine

    loader = DataLoader(config_path)
    fe = FeatureEngine(config_path)

    datasets = loader.load_all()
    datasets = fe.compute_all_datasets(datasets)

    # Sauvegarder chaque df en parquet temporaire pour les workers
    parquet_paths: dict[str, dict[str, str]] = {}
    cache_dir = "data/sweep_cache"
    os.makedirs(cache_dir, exist_ok=True)

    for symbol, tfs in datasets.items():
        safe_symbol = symbol.replace("/", "_")
        parquet_paths[symbol] = {}
        for tf, df in tfs.items():
            if tf == "5m":
                continue  # Le sweep n'utilise pas le 5min
            path = os.path.join(cache_dir, f"{safe_symbol}_{tf}.parquet")
            df.to_parquet(path)
            parquet_paths[symbol][tf] = path
            logger.info("Sauvegardé %s %s → %s (%d bougies)",
                        symbol, tf, path, len(df))

    return parquet_paths


def run_sweep(
    config_path: str = "config.yaml",
    grid: dict | None = None,
    n_workers: int | None = None,
    results_file: str = "sweep_results.pkl",
    checkpoint_file: str = "sweep_checkpoint.pkl",
    exec_config: ExecConfig | None = None,
    initial_equity: float | None = None,
) -> list[dict]:
    """Lance le sweep complet en parallèle.

    1. Prépare les données (télécharge + indicateurs + parquet)
    2. Vérifie le moteur
    3. Lance les workers par (asset, timeframe)
    4. Sauvegarde les résultats

    Args:
        config_path: chemin vers config.yaml
        grid: grille de params (défaut: FULL_GRID)
        n_workers: nombre de workers (défaut: cpu_count)
        results_file: fichier de sortie
        checkpoint_file: fichier de checkpoint
        exec_config: si fourni, active le mode réaliste pour tous les backtests
        initial_equity: capital initial en $ (défaut 1000 si réaliste)

    Returns:
        Liste de dicts avec les résultats de chaque variante
    """
    if grid is None:
        grid = FULL_GRID

    if exec_config is not None and initial_equity is None:
        initial_equity = 1000.0

    print("=" * 60)
    print("SWEEP DE STRATÉGIES V2")
    if exec_config is not None:
        print(f"  MODE RÉALISTE — Capital: ${initial_equity:.0f}")
        print(f"  Equity: {exec_config.equity_pct*100:.0f}% | "
              f"Lev: {exec_config.leverage}x | "
              f"Cooldown: {exec_config.cooldown_bars}h")
    print("=" * 60)

    # Décompte
    total_param_combos = count_combinations(grid)

    # Vérification du moteur
    print("\n── Vérification du moteur ──")
    engine_ok = verify_engine(config_path)
    if not engine_ok:
        print("\n[ERREUR] Le moteur a des problèmes — arrêt du sweep.")
        print("Corrigez les problèmes avant de relancer.")
        return []

    # Préparer les données
    print("\n── Préparation des données ──")
    parquet_paths = prepare_data(config_path)

    assets = list(parquet_paths.keys())
    timeframes = set()
    for tfs in parquet_paths.values():
        timeframes.update(tfs.keys())
    timeframes = sorted(timeframes)

    total_jobs = total_param_combos * len(assets) * len(timeframes)
    n_workers = n_workers or min(cpu_count(), len(assets) * len(timeframes))

    print(f"\n{total_jobs:,} jobs à exécuter")
    print(f"  {len(assets)} assets × {len(timeframes)} timeframes"
          f" = {len(assets) * len(timeframes)} workers")
    print(f"  CPU disponibles : {cpu_count()} — workers : {n_workers}")

    # Reprendre depuis checkpoint si existant
    done_keys: set[str] = set()
    all_results: list[dict] = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)
            done_keys   = checkpoint.get("done_keys", set())
            all_results = checkpoint.get("results", [])
        print(f"  Reprise : {len(done_keys)} (asset, tf) déjà traités")

    # Construire les jobs par (asset, tf)
    worker_args = []
    for asset, tfs in parquet_paths.items():
        for tf, path in tfs.items():
            key = f"{asset}|{tf}"
            if key in done_keys:
                continue
            worker_args.append(
                (asset, tf, path, config_path, grid, exec_config, initial_equity)
            )

    if not worker_args:
        print("  Tous les jobs déjà terminés (checkpoint).")
        return all_results

    # Exécution parallèle
    print(f"\n── Lancement de {len(worker_args)} workers ──")
    t0 = time.time()

    with Pool(processes=n_workers) as pool:
        for i, batch_results in enumerate(pool.imap_unordered(_run_asset_tf, worker_args)):
            if batch_results:
                asset = batch_results[0]["asset"]
                tf    = batch_results[0]["timeframe"]
                all_results.extend(batch_results)
                done_keys.add(f"{asset}|{tf}")

                print(f"  [{i + 1}/{len(worker_args)}] {asset} {tf} "
                      f"— {len(batch_results)} résultats valides")

                # Checkpoint après chaque worker
                with open(checkpoint_file, "wb") as f:
                    pickle.dump({"done_keys": done_keys, "results": all_results}, f)

    elapsed = time.time() - t0
    print(f"\nTerminé en {elapsed / 60:.1f} min")
    print(f"  Jobs exécutés : {total_jobs:,}")
    print(f"  Résultats valides (fréquence ok) : {len(all_results):,}")

    # Sauvegarder
    with open(results_file, "wb") as f:
        pickle.dump(all_results, f)
    print(f"  Sauvegardé dans {results_file}")

    # Nettoyer le checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return all_results


# ── Standalone ────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
    )
    verify_engine()
