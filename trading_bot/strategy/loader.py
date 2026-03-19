import importlib.util
import logging
import os
from pathlib import Path

from trading_bot.strategy.api import StrategyAPI

log = logging.getLogger(__name__)


class StrategyInfo:
    def __init__(self, file_path: str, coins: list[str]):
        self.file_path = str(Path(file_path).resolve())
        self.coins = coins
        self.instance: object | None = None
        self.api: StrategyAPI | None = None
        self.mtime: float = 0.0
        self.errored: bool = False
        self.disabled: bool = False
        self.name: str = Path(file_path).stem


class StrategyLoader:
    def __init__(self, strategy_dir: str = "./strategies"):
        self._dir = str(Path(strategy_dir).resolve())
        self._loaded: dict[str, StrategyInfo] = {}

    def load_strategy(
        self, file_name: str, coins: list[str], api: StrategyAPI,
    ) -> StrategyInfo:
        file_path = os.path.join(self._dir, file_name)
        resolved = str(Path(file_path).resolve())

        if not Path(resolved).is_relative_to(Path(self._dir)):
            raise ValueError(f"Path traversal blocked: {file_name}")

        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Strategy file not found: {resolved}")

        info = StrategyInfo(resolved, coins)
        info.api = api
        info.mtime = os.path.getmtime(resolved)

        instance = self._load_module(resolved, info.name)
        instance.coin = coins[0] if coins else ""
        instance.errored = False

        if not hasattr(instance, "name"):
            instance.name = info.name

        instance.on_init(api)
        info.instance = instance

        self._loaded[info.name] = info
        log.info(f"Strategy loaded: {info.name} → {coins}")
        return info

    def check_reload(self) -> list[str]:
        reloaded = []
        for name, info in self._loaded.items():
            if not os.path.exists(info.file_path):
                continue
            current_mtime = os.path.getmtime(info.file_path)
            if current_mtime > info.mtime:
                log.info(f"Hot-reloading strategy: {name}")
                try:
                    instance = self._load_module(info.file_path, name)
                    instance.coin = info.coins[0] if info.coins else ""
                    instance.errored = False
                    if not hasattr(instance, "name"):
                        instance.name = name
                    instance.on_init(info.api)
                    info.instance = instance
                    info.mtime = current_mtime
                    info.errored = False
                    reloaded.append(name)
                except Exception:
                    log.exception(f"Failed to hot-reload {name}")
                    info.errored = True
        return reloaded

    def get_all(self) -> list[StrategyInfo]:
        return list(self._loaded.values())

    def get(self, name: str) -> StrategyInfo | None:
        return self._loaded.get(name)

    @staticmethod
    def _load_module(file_path: str, name: str) -> object:
        spec = importlib.util.spec_from_file_location(f"strategy_{name}", file_path)
        if not spec or not spec.loader:
            raise ImportError(f"Cannot load strategy: {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        strategy_cls = None
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (isinstance(obj, type) and
                    hasattr(obj, "on_init") and
                    hasattr(obj, "on_tick") and
                    attr_name != "Strategy"):
                strategy_cls = obj
                break

        if not strategy_cls:
            raise ImportError(f"No strategy class found in {file_path}")

        return strategy_cls()
