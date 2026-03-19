"""
Configuration d'exécution réaliste pour le backtester.

Reproduit les conditions live Hyperliquid :
  - Frais maker/taker
  - Slippage sur SL
  - Entry offset ALO
  - Sizing composé avec drawdown multiplier
  - Cooldown entre trades
  - Funding rate
  - Max hold timeout
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecConfig:
    """Paramètres d'exécution live Hyperliquid."""

    equity_pct: float = 0.30          # % du capital par trade
    leverage: float = 5               # levier
    cooldown_bars: int = 4            # bougies entre trades
    max_hold_bars: int = 48           # hold max
    maker_fee: float = 0.00015        # 0.015% entrée ALO
    taker_fee: float = 0.00045        # 0.045% sortie trigger
    slippage_sl_bps: float = 1.0      # 1 bps slippage sur SL
    entry_offset: float = 0.0002      # 0.02% offset ALO
    funding_rate_8h: float = 0.0001   # 0.01% / 8h


# ── Presets pour les 4 stratégies live ───────────────────────

EXEC_CONFIGS: dict[str, ExecConfig] = {
    "SOL Safe": ExecConfig(
        equity_pct=0.30,
        leverage=5,
        cooldown_bars=6,
        max_hold_bars=48,
    ),
    "SOL Normal": ExecConfig(
        equity_pct=0.40,
        leverage=5,
        cooldown_bars=4,
        max_hold_bars=48,
    ),
    "SOL Aggressive": ExecConfig(
        equity_pct=0.50,
        leverage=7,
        cooldown_bars=2,
        max_hold_bars=48,
    ),
    "BTC Momentum": ExecConfig(
        equity_pct=0.35,
        leverage=5,
        cooldown_bars=4,
        max_hold_bars=72,
    ),
}
