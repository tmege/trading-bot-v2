import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ExchangeConfig:
    rest_url: str = "https://api.hyperliquid.xyz"
    ws_url: str = "wss://api.hyperliquid.xyz/ws"
    is_testnet: bool = False
    vault_address: str | None = None


@dataclass
class RiskConfig:
    daily_loss_pct: float = 6.0
    emergency_close_pct: float = 5.0
    max_leverage: int = 10
    max_position_pct: float = 700.0


@dataclass
class StrategyEntry:
    file: str = ""
    role: str = ""
    coins: list[str] = field(default_factory=list)
    paper_mode: bool | None = None
    vault_address: str | None = None


@dataclass
class StrategiesConfig:
    dir: str = "./strategies"
    reload_interval_sec: int = 5
    active: list[StrategyEntry] = field(default_factory=list)


@dataclass
class DatabaseConfig:
    path: str = "./data/trading_bot.db"


@dataclass
class LoggingConfig:
    dir: str = "./logs"
    level: int = 0


@dataclass
class ModeConfig:
    paper_trading: bool = False
    paper_initial_balance: float = 500.0


@dataclass
class SentimentConfig:
    enabled: bool = True
    claude_model: str = "claude-haiku-4-5-20251001"
    max_tokens_per_hour: int = 50000
    cache_ttl_sec: int = 900
    min_confidence: float = 0.5
    weight: float = 0.3
    hard_block_threshold: float = -0.7


@dataclass
class BotConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategies: StrategiesConfig = field(default_factory=StrategiesConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mode: ModeConfig = field(default_factory=ModeConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    private_key: str = ""
    wallet_address: str = ""


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _parse_strategy_entry(data: dict) -> StrategyEntry:
    return StrategyEntry(
        file=data.get("file", ""),
        role=data.get("role", ""),
        coins=data.get("coins", []),
        paper_mode=data.get("paper_mode"),
        vault_address=data.get("vault_address"),
    )


def load_config(config_path: str = "config/bot_config.json") -> BotConfig:
    resolved = Path(config_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Config not found: {resolved}")

    with open(resolved, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cfg = BotConfig()

    if "exchange" in raw:
        ex = raw["exchange"]
        cfg.exchange = ExchangeConfig(
            rest_url=ex.get("rest_url", cfg.exchange.rest_url),
            ws_url=ex.get("ws_url", cfg.exchange.ws_url),
            is_testnet=ex.get("is_testnet", False),
            vault_address=ex.get("vault_address"),
        )

    if "risk" in raw:
        r = raw["risk"]
        cfg.risk = RiskConfig(
            daily_loss_pct=_clamp(r.get("daily_loss_pct", 6.0), 1.0, 50.0),
            emergency_close_pct=r.get("emergency_close_pct", 5.0),
            max_leverage=int(_clamp(r.get("max_leverage", 10), 1, 50)),
            max_position_pct=_clamp(r.get("max_position_pct", 700.0), 10.0, 10000.0),
        )

    if "strategies" in raw:
        s = raw["strategies"]
        entries = [_parse_strategy_entry(e) for e in s.get("active", [])]
        cfg.strategies = StrategiesConfig(
            dir=s.get("dir", "./strategies"),
            reload_interval_sec=s.get("reload_interval_sec", 5),
            active=entries,
        )

    if "database" in raw:
        cfg.database = DatabaseConfig(path=raw["database"].get("path", "./data/trading_bot.db"))

    if "logging" in raw:
        cfg.logging = LoggingConfig(
            dir=raw["logging"].get("dir", "./logs"),
            level=raw["logging"].get("level", 0),
        )

    if "mode" in raw:
        cfg.mode = ModeConfig(
            paper_trading=raw["mode"].get("paper_trading", False),
            paper_initial_balance=raw["mode"].get("paper_initial_balance", 500.0),
        )

    if "sentiment" in raw:
        se = raw["sentiment"]
        # M-11: Validate sentiment config bounds
        cfg.sentiment = SentimentConfig(
            enabled=se.get("enabled", True),
            claude_model=se.get("claude_model", "claude-haiku-4-5-20251001"),
            max_tokens_per_hour=int(_clamp(se.get("max_tokens_per_hour", 50000), 1000, 200000)),
            cache_ttl_sec=int(_clamp(se.get("cache_ttl_sec", 900), 60, 86400)),
            min_confidence=_clamp(se.get("min_confidence", 0.5), 0.0, 1.0),
            weight=_clamp(se.get("weight", 0.3), 0.0, 1.0),
            hard_block_threshold=_clamp(se.get("hard_block_threshold", -0.7), -1.0, 0.0),
        )

    _load_env(cfg)
    _validate_config(cfg)

    return cfg


def _load_env(cfg: BotConfig) -> None:
    env_path = Path(".env").resolve()
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    value = value.strip()
                    # Strip surrounding quotes if present
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                        value = value[1:-1]
                    os.environ.setdefault(key.strip(), value)

    cfg.private_key = os.getenv("TB_PRIVATE_KEY", "")
    cfg.wallet_address = os.getenv("TB_WALLET_ADDRESS", "")


def _validate_config(cfg: BotConfig) -> None:
    coin_map: dict[str, str] = {}
    for entry in cfg.strategies.active:
        for coin in entry.coins:
            if coin in coin_map and not entry.paper_mode:
                # Only block live strategies sharing a coin
                # Paper mode strategies each have their own PaperExchange
                other = coin_map[coin]
                other_entry = next(
                    (e for e in cfg.strategies.active if e.file == other), None
                )
                if other_entry and not other_entry.paper_mode:
                    raise ValueError(
                        f"Coin {coin} assigned to both {other} and {entry.file} in live mode"
                    )
            coin_map[coin] = entry.file

    if not cfg.mode.paper_trading and not cfg.private_key:
        all_paper = all(
            e.paper_mode is True for e in cfg.strategies.active
        )
        if not all_paper:
            log.warning("No TB_PRIVATE_KEY set — live trading will fail")

    log.info(f"Config loaded: {len(cfg.strategies.active)} strategies, paper={cfg.mode.paper_trading}")
