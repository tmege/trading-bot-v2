# Trading Bot v2 - Instructions Claude Code

## Apercu du projet

Bot de trading crypto automatise pour **Hyperliquid** (perpetuels DEX). Architecture event-driven avec GUI desktop, multi-strategies hot-reloadable, paper trading, backtesting et analyse de sentiment.

**Stack** : Python 3.11+ | FastAPI | pywebview | SQLite | WebSocket | Hyperliquid API | Claude AI

~9400 lignes de code, 60 fichiers Python.

## Architecture

### 3 threads
- **Thread principal** : fenetre pywebview (GUI native)
- **Thread 1** : serveur FastAPI/uvicorn sur `127.0.0.1:8089`
- **Thread 2** : boucle asyncio avec le moteur de trading

### Flux d'evenements
```
WebSocket Hyperliquid → Engine._dispatch()
    ├─ _on_mids()    → prix       → strategy.on_tick()
    ├─ _on_fills()   → executions → strategy.on_fill()
    ├─ _on_book()    → carnet     → strategy.on_book()
    └─ timer (60s)                → strategy.on_timer()
```

### Routage des ordres
```
Strategy → StrategyAPI → OrderManager
    ├─ paper_mode=true  → PaperExchange (simulateur)
    └─ paper_mode=false → RestClient → Hyperliquid (EIP-712)
```

## Fichiers cles

### Core
- `main.py` : point d'entree, creation des 3 threads
- `trading_bot/engine.py` : moteur principal (lifecycle, dispatch, state persistence)
- `trading_bot/config.py` : dataclasses de config, chargement de `bot_config.json`
- `trading_bot/types.py` : types de donnees (Order, Fill, Position, Candle, Decimal...)
- `trading_bot/db.py` : couche SQLite avec WAL mode

### Exchange
- `trading_bot/exchange/rest.py` : client REST Hyperliquid (rate limit 1200 req/min)
- `trading_bot/exchange/ws.py` : client WebSocket (subscriptions temps reel)
- `trading_bot/exchange/signing.py` : signature EIP-712 des ordres
- `trading_bot/exchange/order_manager.py` : routage paper/live + mapping OID→strategie
- `trading_bot/exchange/paper_exchange.py` : simulateur complet avec frais

### Strategies
- `trading_bot/strategy/api.py` : **StrategyAPI** — interface principale utilisee par les strategies
- `trading_bot/strategy/loader.py` : chargement dynamique + hot-reload (5s)
- `trading_bot/strategy/base.py` : protocol (interface) de strategie
- `trading_bot/strategy/indicators.py` : 50+ indicateurs techniques
- `trading_bot/strategies/template.py` : classe de base `TemplateStrategy`
- `trading_bot/strategies/*.py` : implementations concretes

### Web
- `trading_bot/web/app.py` : creation FastAPI (auth API key, CORS, routes)
- `trading_bot/web/routes/*.py` : endpoints REST
- `trading_bot/web/services/*.py` : logique metier

### Autres
- `trading_bot/risk/risk_manager.py` : limites de perte, circuit breaker, controle de levier
- `trading_bot/data/data_manager.py` : sentiment (Claude AI + CryptoPanic + Fear & Greed)
- `trading_bot/backtest/engine.py` : moteur de backtesting avec Monte Carlo

## Conventions de code

### Patterns obligatoires
- **Async/await** dans le moteur et l'exchange (boucle asyncio dediee)
- **Dataclasses** pour les types de donnees (`@dataclass` dans `types.py`)
- **Protocol** pour l'interface de strategie (`base.py`)
- **Decimal custom** (`types.Decimal`) pour la precision des montants — ne PAS utiliser `float` pour les prix/tailles d'ordres
- **Requetes SQL parametrees** uniquement — jamais de concatenation

### Nommage
- Modules : `snake_case.py`
- Classes : `PascalCase`
- Fonctions/methodes : `snake_case`
- Constantes : `UPPER_SNAKE_CASE`
- Callbacks engine : `_on_mids()`, `_on_fills()`, `_on_book()`
- Routes API : prefixe `/api/`

### Configuration
- **Config** : `config/bot_config.json` (JSON)
- **Secrets** : `.env` uniquement (jamais dans le code, jamais commites)
- **Env requises** : `TB_PRIVATE_KEY`, `TB_WALLET_ADDRESS`
- **Env optionnelles** : `ANTHROPIC_API_KEY`, `CRYPTOPANIC_TOKEN`, `TB_WEB_API_KEY`

### Base de donnees
- SQLite WAL mode (`data/trading_bot.db`)
- Tables : `candles`, `trades`, `strategy_state`, `funding_rates`, `order_strategy_map`, `backtest_history`

## Regles de developpement

### Strategies
- Heriter de `TemplateStrategy` (`trading_bot/strategies/template.py`)
- Utiliser **uniquement** `StrategyAPI` pour interagir avec le moteur
- Ne PAS acceder directement a `Engine`, `RestClient` ou `DB` depuis une strategie
- Utiliser `api.save_state()` / `api.load_state()` pour la persistence d'etat
- Gerer les cooldowns et etats internes dans la strategie

### Risk management
- Chaque ordre passe par `risk_manager.check_order()` — ne pas contourner
- Circuit breaker : >7% de mouvement en 15min sur un coin
- Limites : daily_loss_pct=6%, emergency_close_pct=5%, max_leverage=10x

### Securite web
- API key obligatoire sur tous les endpoints (`X-API-Key` header)
- CORS desactive (allow_origins=[])
- Echapper les sorties frontend (`TB.utils.esc()`)
- Valider les noms de fichiers de strategies (regex + `Path.is_relative_to()`)
- Nettoyer les codes ANSI avant streaming SSE

### Paper trading
- `PaperExchange` simule les frais reels (maker 0.015%, taker 0.045%)
- Toute strategie doit etre testee en paper mode avant le live
- L'etat paper est persiste en DB et restaure au redemarrage

## Commandes utiles

```bash
# Demarrer le bot
python main.py

# Backtest
python -m trading_bot.backtest.runner --strategy btc_sniper_1h --coin BTC --start 2025-01-01 --end 2025-03-01

# Fetch candles historiques
./scripts/fetch_candles.sh
```

## Points d'attention

- Le bot demarre en etat OFF, il faut le demarrer via l'interface GUI
- Les strategies `paper_mode: true` utilisent `PaperExchange`, pas Hyperliquid
- Le sentiment a un poids de 0.3 et peut bloquer un trade si score < -0.7
- Rate limit Hyperliquid : 1200 req/min avec backoff exponentiel
- L'audit de securite complet est dans `docs/SECURITY_AUDIT.md` (13 vulns corrigees)
