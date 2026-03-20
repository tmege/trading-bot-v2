import ast
import importlib.util
import logging
import os
from pathlib import Path

from trading_bot.strategy.api import StrategyAPI

log = logging.getLogger(__name__)

# C-01: Whitelist of allowed imports in strategy files
_ALLOWED_IMPORTS = frozenset({
    "math", "statistics", "collections", "dataclasses", "typing", "enum",
    "decimal", "fractions", "datetime", "time", "logging", "json", "re",
    "functools", "itertools", "operator", "copy", "abc",
    "numpy", "pandas", "pandas_ta", "ta", "talib",
    "trading_bot", "trading_bot.strategy", "trading_bot.strategy.api",
    "trading_bot.strategy.base", "trading_bot.strategy.indicators",
    "trading_bot.strategies.template", "trading_bot.types",
})

_BLOCKED_IMPORTS = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib", "socket", "http",
    "urllib", "requests", "httpx", "aiohttp", "ftplib", "smtplib",
    "pickle", "shelve", "marshal", "ctypes", "importlib",
    "signal", "multiprocessing", "threading", "asyncio",
    "code", "codeop", "compile", "compileall",
    "webbrowser", "tempfile", "glob", "fnmatch",
})


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

        # C-01/M-01: Block symlinks to prevent escape from strategies dir
        if os.path.islink(file_path):
            raise ValueError(f"Symlinks not allowed: {file_name}")

        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Strategy file not found: {resolved}")

        # C-01: AST inspection before loading
        _audit_strategy_file(resolved)

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
            # M-01: Block symlinks on reload too
            if os.path.islink(info.file_path):
                log.warning(f"Symlink detected on reload, skipping: {name}")
                info.errored = True
                continue
            current_mtime = os.path.getmtime(info.file_path)
            if current_mtime > info.mtime:
                log.info(f"Hot-reloading strategy: {name}")
                try:
                    _audit_strategy_file(info.file_path)
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


def _audit_strategy_file(file_path: str) -> None:
    """C-01: AST-based security inspection — block dangerous imports and calls."""
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as e:
        raise ImportError(f"Syntax error in strategy {file_path}: {e}") from e

    for node in ast.walk(tree):
        # Check import statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module(alias.name, file_path)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                _check_module(node.module, file_path)
        # Block eval/exec/compile/__import__
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("eval", "exec", "compile", "__import__", "open"):
                raise ImportError(
                    f"Blocked dangerous call '{func.id}()' in {file_path}"
                )
            if isinstance(func, ast.Attribute) and func.attr in ("system", "popen", "exec_module"):
                raise ImportError(
                    f"Blocked dangerous call '.{func.attr}()' in {file_path}"
                )


def _check_module(module_name: str, file_path: str) -> None:
    top = module_name.split(".")[0]
    if top in _BLOCKED_IMPORTS:
        raise ImportError(
            f"Blocked import '{module_name}' in {file_path} — "
            f"strategies may only use: trading_bot.strategy.*, math, numpy, pandas, etc."
        )
