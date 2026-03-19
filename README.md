# Trading Bot v2

Bot de trading automatise pour les perpetuels crypto sur **Hyperliquid** (DEX). Interface desktop native, multi-strategies avec hot-reload, paper trading, backtesting et analyse de sentiment via Claude AI.

## Fonctionnalites

- **Multi-strategies** : execution simultanee de strategies independantes sur differents coins
- **Hot-reload** : modification des strategies sans redemarrage (rechargement toutes les 5s)
- **Paper trading** : simulateur complet avec frais realistes (maker 0.015%, taker 0.045%)
- **Backtesting** : moteur de replay historique avec simulation Monte Carlo
- **Analyse de sentiment** : Claude AI analyse les news CryptoPanic + Fear & Greed Index
- **Risk management** : limites de perte journaliere, circuit breaker, controle de levier
- **50+ indicateurs techniques** : RSI, MACD, Bollinger, Ichimoku, Supertrend, etc.
- **Interface desktop** : GUI native via pywebview + API FastAPI locale
- **Persistence** : SQLite (WAL mode) pour trades, candles, etat des strategies

## Pre-requis

- Python 3.11+
- Compte Hyperliquid avec cle privee (wallet Ethereum)
- Cle API Anthropic (optionnel, pour le sentiment)
- Token CryptoPanic (optionnel, pour les news)

## Installation

```bash
# Cloner le projet
git clone <repo-url> trading-bot-v2
cd trading-bot-v2

# Creer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate

# Installer les dependances
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
# Editer .env avec vos cles
```

## Configuration

### Variables d'environnement (`.env`)

| Variable | Requis | Description |
|----------|--------|-------------|
| `TB_PRIVATE_KEY` | Oui | Cle privee Ethereum (format `0x...`) pour signer les ordres |
| `TB_WALLET_ADDRESS` | Oui | Adresse du wallet Hyperliquid |
| `ANTHROPIC_API_KEY` | Non | Cle API Claude pour l'analyse de sentiment |
| `CRYPTOPANIC_TOKEN` | Non | Token API CryptoPanic pour le flux de news |
| `TB_WEB_API_KEY` | Non | Cle API pour l'interface web (auto-generee si absente) |

### Configuration du bot (`config/bot_config.json`)

```jsonc
{
  "exchange": {
    "rest_url": "https://api.hyperliquid.xyz",  // URL REST Hyperliquid
    "ws_url": "wss://api.hyperliquid.xyz/ws",   // URL WebSocket
    "is_testnet": false,                         // true pour le testnet
    "vault_address": null                        // adresse du vault (optionnel)
  },
  "risk": {
    "daily_loss_pct": 6.0,        // stop trading apres -6% de perte journaliere
    "emergency_close_pct": 5.0,   // fermeture d'urgence a -5%
    "max_leverage": 10,           // levier max par ordre
    "max_position_pct": 700.0     // taille max de position (% du compte)
  },
  "strategies": {
    "dir": "./trading_bot/strategies",
    "reload_interval_sec": 5,
    "active": [
      {
        "file": "btc_sniper_1h.py",       // fichier de la strategie
        "role": "primary",                 // role (informatif)
        "coins": ["BTC"],                  // coins trades
        "paper_mode": true,                // true = paper trading
        "paper_balance": 500.0             // balance initiale paper
      }
    ]
  },
  "sentiment": {
    "enabled": true,                             // activer l'analyse de sentiment
    "claude_model": "claude-haiku-4-5-20251001", // modele Claude utilise
    "max_tokens_per_hour": 50000,                // limite de tokens/heure
    "cache_ttl_sec": 900,                        // cache du sentiment (15min)
    "weight": 0.3,                               // poids du sentiment dans les decisions
    "hard_block_threshold": -0.7                 // seuil de blocage (sentiment tres negatif)
  }
}
```

## Demarrage

```bash
# Demarrage direct
python main.py

# Ou via le script
./scripts/start.sh
```

Une fenetre desktop native s'ouvre avec le dashboard. Le bot demarre en etat **OFF** — utiliser l'interface pour le demarrer.

## Architecture

### 3 threads

```
Main Thread (pywebview)        Thread 1 (uvicorn)       Thread 2 (asyncio)
+------------------+          +------------------+     +------------------+
|  webview.start() |  HTTP    |  FastAPI          |     |  Engine          |
|  Native window   | <--------|  REST + WS        | <---|  Trading loop    |
|  localhost:8089   |          |  /api/*           |     |  Strategies      |
+------------------+          +------------------+     +------------------+
```

### Flux d'evenements

```
WebSocket Hyperliquid → Engine dispatch
    ├─ on_mids()   → mise a jour prix → strategy.on_tick()
    ├─ on_fills()  → trades executes  → strategy.on_fill()
    ├─ on_book()   → carnet d'ordres  → strategy.on_book()
    └─ on_timer()  → toutes les 60s   → strategy.on_timer()
```

### Routage des ordres

```
Strategy → StrategyAPI → OrderManager
    ├─ Paper mode → PaperExchange (simulateur local)
    └─ Live mode  → RestClient → Hyperliquid API (signe EIP-712)
```

## Structure du projet

```
trading-bot-v2/
├── main.py                             # Point d'entree (3 threads)
├── requirements.txt                    # Dependances Python
├── .env.example                        # Template variables d'environnement
├── config/
│   └── bot_config.json                 # Configuration principale
│
├── trading_bot/                        # Package principal
│   ├── __init__.py
│   ├── engine.py                       # Moteur de trading (lifecycle, dispatch)
│   ├── config.py                       # Dataclasses de configuration
│   ├── types.py                        # Types de donnees (Order, Fill, Position...)
│   ├── db.py                           # Couche SQLite (WAL mode)
│   ├── decimal_utils.py                # Arithmetique a precision fixe
│   ├── logging_config.py               # Configuration du logging
│   │
│   ├── exchange/                       # Integration exchange
│   │   ├── rest.py                     # Client REST Hyperliquid
│   │   ├── ws.py                       # Client WebSocket (prix, fills, ordres)
│   │   ├── signing.py                  # Signature EIP-712 des ordres
│   │   ├── order_manager.py            # Routage ordres (paper/live)
│   │   └── paper_exchange.py           # Simulateur paper trading
│   │
│   ├── strategy/                       # Framework de strategies
│   │   ├── api.py                      # StrategyAPI (interface principale)
│   │   ├── loader.py                   # Chargement dynamique + hot-reload
│   │   ├── base.py                     # Protocol de strategie
│   │   └── indicators.py              # 50+ indicateurs techniques
│   │
│   ├── strategies/                     # Implementations de strategies
│   │   ├── template.py                 # Classe de base TemplateStrategy
│   │   ├── btc_sniper_1h.py            # BTC sniper (RSI/MACD, 1h)
│   │   ├── doge_sniper_relaxed_1h.py   # DOGE sniper (trailing stops, 1h)
│   │   ├── sol_range_breakout_1h.py    # SOL range breakout (1h)
│   │   └── sol_test_1usd.py            # Strategie de test
│   │
│   ├── risk/
│   │   └── risk_manager.py             # Limites journalieres, circuit breaker
│   │
│   ├── data/
│   │   └── data_manager.py             # Sentiment (Claude AI + Fear & Greed)
│   │
│   ├── tools/                          # Utilitaires
│   │   ├── candle_fetcher.py           # Fetch candles historiques (Binance)
│   │   ├── funding_fetcher.py          # Historique funding rates
│   │   ├── regime_analyzer.py          # Analyse de regime de marche
│   │   └── signal_scanner.py           # Detection de signaux
│   │
│   ├── backtest/                       # Moteur de backtesting
│   │   ├── engine.py                   # Execution du backtest
│   │   ├── monte_carlo.py              # Simulation Monte Carlo
│   │   └── runner.py                   # CLI runner
│   │
│   ├── report/
│   │   └── dashboard.py               # Metriques dashboard temps reel
│   │
│   └── web/                            # Interface web (FastAPI)
│       ├── app.py                      # Creation app (auth, CORS, routes)
│       ├── routes/                     # Endpoints API
│       │   ├── bot.py                  # Start/stop/status du bot
│       │   ├── account.py              # Compte, positions, ordres
│       │   ├── strategies.py           # Gestion des strategies
│       │   ├── market.py               # Donnees de marche, sentiment
│       │   ├── trades.py               # Historique des trades
│       │   ├── backtest.py             # Execution de backtests
│       │   ├── settings.py             # Configuration
│       │   └── logs.py                 # Streaming de logs (SSE)
│       ├── services/                   # Logique metier
│       │   ├── market_data.py          # Aggregation donnees marche
│       │   ├── metrics.py              # Calcul de metriques
│       │   ├── digest.py               # Digest journalier
│       │   ├── alerts.py               # Systeme d'alertes
│       │   └── backtest_service.py     # Orchestration backtests
│       └── static/                     # Assets frontend
│           └── js/pages/              # Composants JS
│
├── scripts/
│   ├── start.sh                        # Script de demarrage
│   ├── backtest.sh                     # Lancement backtest
│   ├── fetch_candles.sh                # Recuperation de candles
│   ├── grid_search.sh                  # Recherche d'hyperparametres
│   └── run_backtests.py                # Runner backtest Python
│
├── docs/
│   ├── gui.md                          # Documentation de l'interface
│   └── SECURITY_AUDIT.md              # Audit de securite (13 vulns corrigees)
│
├── data/                               # Base SQLite (gitignore)
└── logs/                               # Logs d'execution (gitignore)
```

## API Endpoints

### Bot (`/api/bot`)

| Methode | Route | Description |
|---------|-------|-------------|
| GET | `/status` | Etat du bot (running, uptime, connexion WS) |
| POST | `/start` | Demarrer le bot |
| POST | `/stop` | Arreter le bot |

### Compte (`/api/account`)

| Methode | Route | Description |
|---------|-------|-------------|
| GET | `/value` | Valeur du compte |
| GET | `/positions` | Positions ouvertes |
| GET | `/open-orders` | Ordres en cours |

### Strategies (`/api/strategies`)

| Methode | Route | Description |
|---------|-------|-------------|
| GET | `/list` | Liste des strategies actives |
| GET | `/{name}/code` | Code source d'une strategie |
| PUT | `/{name}/disabled` | Activer/desactiver une strategie |

### Marche (`/api/market`)

| Methode | Route | Description |
|---------|-------|-------------|
| GET | `/mids` | Prix mid de tous les assets |
| GET | `/book/{coin}` | Carnet d'ordres |
| GET | `/sentiment` | Analyse de sentiment |

### Trades (`/api/trades`)

| Methode | Route | Description |
|---------|-------|-------------|
| GET | `/list` | Historique des trades |
| GET | `/export` | Export CSV |

### Backtest (`/api/backtest`)

| Methode | Route | Description |
|---------|-------|-------------|
| POST | `/run` | Lancer un backtest |

### Settings (`/api/settings`)

| Methode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Configuration actuelle |
| PUT | `/mode` | Basculer paper/live |
| PUT | `/` | Mettre a jour la configuration |

## Strategies incluses

| Strategie | Coin | Logique | TP/SL |
|-----------|------|---------|-------|
| `btc_sniper_1h` | BTC | RSI > 65 + MACD deceleration (long), RSI < 30 + MACD < 0 (short) | 2% / 2% |
| `doge_sniper_relaxed_1h` | DOGE | RSI < 35 (short dominant), trailing stops, pause apres 4 pertes | 4.5% / 1.5% |
| `sol_range_breakout_1h` | SOL | Bear mode (SMA200) + range breakout (volume spike) | 4.5-6% / 1.5-2% |

### Creer une strategie

1. Copier `trading_bot/strategies/template.py`
2. Implementer `on_init()`, `on_tick()`, `on_fill()`, `on_timer()`
3. Ajouter l'entree dans `config/bot_config.json` sous `strategies.active`
4. La strategie sera chargee automatiquement (hot-reload)

## Backtesting

```bash
# Via script
./scripts/backtest.sh btc_sniper_1h BTC 2025-01-01 2025-03-01

# Via Python
python -m trading_bot.backtest.runner \
  --strategy btc_sniper_1h \
  --coin BTC \
  --start 2025-01-01 \
  --end 2025-03-01
```

## Base de donnees

SQLite avec WAL mode. Tables principales :

| Table | Contenu |
|-------|---------|
| `candles` | Donnees OHLCV historiques |
| `trades` | Historique des trades executes |
| `strategy_state` | Etat persiste des strategies (JSON) |
| `funding_rates` | Historique des funding rates |
| `order_strategy_map` | Mapping ordre → strategie |
| `backtest_history` | Resultats de backtests |

## Securite

- Authentification par API key sur tous les endpoints
- CORS desactive (localhost only)
- Protection path traversal sur le chargement de strategies
- Echappement XSS sur le frontend
- Nettoyage ANSI sur le streaming de logs
- Audit complet : 13 vulnerabilites identifiees et corrigees (voir `docs/SECURITY_AUDIT.md`)

## Dependances

| Package | Version | Usage |
|---------|---------|-------|
| `httpx` | >= 0.27 | Client HTTP (API REST Hyperliquid) |
| `websockets` | >= 13.0 | Streaming temps reel |
| `msgpack` | >= 1.0 | Serialisation binaire (signature) |
| `eth-account` | >= 0.13 | Gestion compte Ethereum |
| `pycryptodome` | >= 3.20 | Fonctions cryptographiques |
| `anthropic` | >= 0.40 | API Claude (sentiment) |
| `fastapi` | >= 0.115 | Framework API web |
| `uvicorn` | >= 0.32 | Serveur ASGI |
| `pywebview` | >= 5.0 | Fenetre desktop native |
| `python-dotenv` | >= 1.0 | Chargement `.env` |
