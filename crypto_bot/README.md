# Crypto Strategy Research

Système de recherche de stratégies crypto à levier sur futures perpétuels. Combine analyse probabiliste historique et simulation de liquidation réaliste sur bougies 5min.

## Architecture

```
crypto_bot/
├── config.yaml              # Configuration (tout en % du portfolio)
├── main.py                  # Orchestrateur des 7 modules
├── requirements.txt
└── modules/
    ├── data_loader.py       # 1. Téléchargement OHLCV 5min + funding (ccxt/Binance)
    ├── feature_engine.py    # 2. 30+ indicateurs techniques (pandas-ta)
    ├── probability_engine.py # 3. Probabilités conditionnelles + corrélations laggées
    ├── liquidation_engine.py # 4. Simulation liquidation isolated margin (5min)
    ├── strategies.py        # 5. 4 stratégies × variantes = 36 configs
    ├── backtester.py        # 6. Backtest train/test + Kelly fractional
    └── reporter.py          # 7. Rapport HTML interactif + Markdown
```

## Installation

```bash
cd crypto_bot
pip install -r requirements.txt
```

### Dépendances

| Package | Usage |
|---------|-------|
| pandas | DataFrames, séries temporelles |
| numpy | Calculs numériques |
| pandas-ta | Indicateurs techniques |
| ccxt | Données Binance Futures (OHLCV, funding) |
| scipy | Tests statistiques (binomial) |
| plotly | Graphiques interactifs (equity curves, heatmaps) |
| pyyaml | Configuration |
| pyarrow | Cache parquet |
| jinja2 | Templates HTML |

## Utilisation

```bash
# Lancer la recherche complète
python main.py
```

Le script exécute dans l'ordre :

1. **Télécharge** les bougies 5min (4 ans × 4 assets) + funding rates → cache parquet
2. **Resample** vers 1h et 4h
3. **Calcule** 30+ indicateurs techniques sur 1h et 4h
4. **Scanne** 21 événements probabilistes, calcule les corrélations laggées
5. **Backteste** 36 variantes × 4 assets en phase train (2021–2023)
6. **Évalue** sur le test set (2024), applique Kelly fractional
7. **Génère** le rapport dans `reports/`

### Sortie

```
reports/
├── report_YYYYMMDD_HHMMSS.html   # Rapport interactif (plotly, dark theme)
└── report_YYYYMMDD_HHMMSS.md     # Version Markdown
```

## Configuration

Tout est dans `config.yaml`. Le portfolio est normalisé à **100%** (agnostique au capital).

### Paramètres clés

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `capital_initial` | 100.0 | 100% = portfolio complet |
| `stop_global_portfolio` | 50.0 | Arrêt si capital < 50% |
| `fees.taker` | 0.06 | 0.06% (market orders) |
| `fees.maker` | 0.02 | 0.02% (ALO limit orders) |
| `funding.default_rate` | 0.01 | 0.01% / 8h |
| `fees.slippage` | 0.05 | 0.05% (taker uniquement) |
| `margin.maintenance` | 0.4 | 0.4% (isolated) |
| `train_ratio` | 0.75 | 3 ans train / 1 an test |

### Optimisation des frais

Les stratégies utilisent par défaut des ordres **ALO (Add Liquidity Only)** pour l'entrée et les take-profits :

| Moment | Type d'ordre | Frais | Slippage |
|--------|-------------|-------|----------|
| Entrée | ALO limit (maker) | 0.02% | 0% |
| TP | ALO limit (maker) | 0.02% | 0% |
| SL | Market (taker) | 0.06% | 0.05% |

## Stratégies

### A — Grid leveragée (3x)
- **Timeframe** : 1h
- **Logique** : Range ±8% autour EMA50, grille 5 buy + 5 sell
- **Size** : 5% par ordre, max 30% engagé
- **Variantes** : levier {2x, 3x, 5x} × espacement {1%, 1.5%, 2%}

### B — Momentum futures (5x)
- **Timeframe** : 4h
- **Logique** : Alignement EMA + RSI + volume + régime
- **Gestion** : TP1 partiel à +3%, trailing stop 1.5%
- **Variantes** : levier {3x, 5x, 7x} × SL {1.5%, 2%, 3%}

### C — Mean reversion oversold (3x)
- **Timeframe** : 1h (long only)
- **Logique** : RSI<25 + BB lower + body + volume + trend 4h
- **Sortie** : RSI > 50 ou close > EMA21
- **Variantes** : RSI {20, 25, 30} × levier {2x, 3x, 5x}

### D — Breakout explosif (10x)
- **Timeframe** : 1h
- **Logique** : Compression ATR + breakout high20 + body + volume
- **Size** : 5% (risque limité malgré le levier élevé)
- **Variantes** : levier {7x, 10x, 15x} × taille {3%, 5%, 8%}

## Simulation de liquidation

La vérification de liquidation se fait **bougie 5min par bougie 5min**, même pour les stratégies 1h/4h. Cela capture les flash wicks intra-bougie :

```
Signal sur bougie 4h
  └─ Découpe en 48 bougies 5min
      └─ Pour chaque bougie 5min :
          1. Liquidation ? (low ≤ liq_price)
          2. Stop loss ?
          3. Take profit ?
          4. Funding toutes les 8h ?
```

### Prix de liquidation (isolated margin)

```
Long  : entry × (1 - 1/leverage + 0.004)
Short : entry × (1 + 1/leverage - 0.004)

Exemples (entry = 50000) :
  3x  → liq à -33.1% | 5x  → liq à -19.6%
  10x → liq à -9.6%  | 15x → liq à -6.3%
```

## Analyse probabiliste

21 événements prédéfinis scannés par asset × timeframe :
- Oversold bounces, overbought reversals
- Momentum entries (golden/death cross)
- Breakout patterns (compression + volume)
- Mean reversion, trend continuation
- Multi-indicator convergence

Chaque événement est validé par **test binomial** (N ≥ 30, p < 0.05) et sa **stabilité temporelle** est mesurée sur fenêtres glissantes de 6 mois.

## Kelly Criterion

Sizing conservateur (Kelly quart) appliqué en post-processing :

```
f* = W/a - (1-W)/b    (Kelly complet)
f_kelly = f* / 4       (Kelly quart, clampé 0–25%)
```

Le backtest est relancé avec `size = f_kelly` pour comparer le Sharpe avec/sans Kelly.
