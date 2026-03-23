import json
import logging
import os
import re
from pathlib import Path

from fastapi import APIRouter

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategies", tags=["strategies"])

_engine = None

# Allowed group names (alphanumeric, hyphens, underscores)
_GROUP_RE = re.compile(r"^[a-zA-Z0-9_-]{1,50}$")


def init(engine):
    global _engine
    _engine = engine


def _get_group_for_strategy(idx: int) -> str:
    """Get group name for strategy at config index."""
    if _engine and _engine.config and idx < len(_engine.config.strategies.active):
        return _engine.config.strategies.active[idx].group or ""
    return ""


def _build_coin_group_map() -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Return (groups: {group_name: [strat_names]}, strat_coins: {strat_name: {coins}})."""
    groups: dict[str, list[str]] = {}
    strat_coins: dict[str, set[str]] = {}
    if not _engine or not _engine.config:
        return groups, strat_coins
    for entry in _engine.config.strategies.active:
        name = entry.file.replace(".py", "")
        strat_coins[name] = set(entry.coins)
        if entry.group:
            groups.setdefault(entry.group, []).append(name)
    return groups, strat_coins


def _find_coin_conflicts(strat_name: str, groups: dict[str, list[str]],
                         strat_coins: dict[str, set[str]]) -> set[str]:
    """Find strategies from OTHER groups that share coins with strat_name."""
    # Find this strategy's group
    my_group = ""
    for g, members in groups.items():
        if strat_name in members:
            my_group = g
            break
    if not my_group:
        return set()

    my_coins = strat_coins.get(strat_name, set())
    if not my_coins:
        return set()

    conflicts = set()
    for g, members in groups.items():
        if g == my_group:
            continue
        for member in members:
            if strat_coins.get(member, set()) & my_coins:
                conflicts.add(member)
    return conflicts


@router.get("")
async def get_strategies():
    if not _engine:
        return []

    # If engine has loaded strategies, use them
    if _engine._strategies:
        result = []
        for idx, info in enumerate(_engine._strategies):
            inst = info.instance

            trades = getattr(inst, "trade_count", 0) if inst else 0
            wins = getattr(inst, "win_count", 0) if inst else 0
            wr = round(wins / trades * 100, 1) if trades > 0 else None
            consec = getattr(inst, "consec_losses", 0) if inst else 0

            pos_side = None
            if inst and getattr(inst, "in_position", False):
                pos_side = getattr(inst, "position_side", "").upper() or None

            entry = None
            role = ""
            paper_mode = False
            coins_cfg = info.coins
            if _engine.config and idx < len(_engine.config.strategies.active):
                entry = _engine.config.strategies.active[idx]
                role = entry.role or ""
                paper_mode = entry.paper_mode if entry.paper_mode is not None else _engine.config.mode.paper_trading

            total_pnl = 0.0
            if _engine.db and _engine.db.conn:
                total_pnl = _engine.db.get_total_pnl(info.name)

            wr_per_coin = {}
            if _engine.db and _engine.db.conn and trades > 0:
                for coin in info.coins:
                    rows = _engine.db.fetchone(
                        "SELECT COUNT(CASE WHEN closed_pnl > 0 THEN 1 END) as w, "
                        "COUNT(CASE WHEN closed_pnl != 0 THEN 1 END) as t "
                        "FROM trades WHERE strategy=? AND coin=?",
                        (info.name, coin),
                    )
                    if rows and rows["t"] > 0:
                        wr_per_coin[coin] = round(rows["w"] / rows["t"] * 100, 1)

            equity_pct = getattr(inst, "equity_pct", None) if inst else None
            leverage = getattr(inst, "leverage", None) if inst else None

            result.append({
                "name": info.name,
                "file": Path(info.file_path).name,
                "coins": coins_cfg,
                "role": role,
                "group": _get_group_for_strategy(idx),
                "status": "DISABLED" if info.disabled else ("ERRORED" if info.errored else "OK"),
                "paper_mode": paper_mode,
                "trades": trades,
                "win_rate": wr,
                "win_rate_per_coin": wr_per_coin,
                "pnl": round(total_pnl, 4),
                "position": pos_side,
                "consec_losses": consec,
                "equity_pct": equity_pct,
                "leverage": leverage,
            })
        return result

    # Engine stopped — show strategies from config
    if _engine.config:
        # Load disabled list from DB
        disabled_names = set()
        if _engine.db and _engine.db.conn:
            raw = _engine.db.load_state("__engine__", "state")
            if raw:
                try:
                    state = json.loads(raw)
                    disabled_names = set(state.get("disabled_strategies", []))
                except (json.JSONDecodeError, KeyError):
                    pass

        result = []
        for entry in _engine.config.strategies.active:
            name = entry.file.replace(".py", "")
            paper_mode = entry.paper_mode if entry.paper_mode is not None else _engine.config.mode.paper_trading
            total_pnl = 0.0
            if _engine.db and _engine.db.conn:
                total_pnl = _engine.db.get_total_pnl(name)
            is_disabled = name in disabled_names
            result.append({
                "name": name,
                "file": entry.file,
                "coins": entry.coins,
                "role": entry.role or "",
                "group": entry.group or "",
                "status": "DISABLED" if is_disabled else "STOPPED",
                "paper_mode": paper_mode,
                "trades": 0,
                "win_rate": None,
                "win_rate_per_coin": {},
                "pnl": round(total_pnl, 4),
                "position": None,
                "consec_losses": 0,
            })
        return result

    return []


@router.get("/{name}/code")
async def get_strategy_code(name: str):
    if not _engine or not _engine.config:
        return {"error": "not initialized"}

    strategies_dir = str(Path(_engine.config.strategies.dir).resolve())

    info = None
    for s in _engine._strategies:
        if s.name == name:
            info = s
            break

    if not info:
        return {"error": "strategy not found"}

    resolved = Path(info.file_path).resolve()
    if not resolved.is_relative_to(Path(strategies_dir)):
        return {"error": "access denied"}
    resolved = str(resolved)

    try:
        with open(resolved, "r", encoding="utf-8") as f:
            code = f.read()
        lines = code.count("\n") + 1
        return {"name": name, "file": Path(resolved).name, "code": code, "lines": lines}
    except Exception:
        return {"error": "failed to read file"}


@router.post("/group/{group_name}/toggle")
async def toggle_group(group_name: str):
    if not _engine:
        return {"error": "not initialized"}

    if not _GROUP_RE.match(group_name):
        return {"error": "invalid group name"}

    if not _engine.config:
        return {"error": "no config loaded"}

    groups, _ = _build_coin_group_map()

    if group_name not in groups:
        return {"error": "group not found"}

    target_names = set(groups[group_name])
    other_groups = {g for g in groups if g != group_name}

    # Engine running: toggle in-memory
    if _engine._strategies:
        # Check if the target group is currently active (at least one strategy enabled)
        target_active = any(
            not info.disabled
            for info in _engine._strategies
            if info.name in target_names
        )

        if target_active:
            # Disable the target group
            for info in _engine._strategies:
                if info.name in target_names:
                    info.disabled = True
            action = "disabled"
        else:
            # Enable target group, disable other groups
            other_names = set()
            for g in other_groups:
                other_names.update(groups[g])

            for info in _engine._strategies:
                if info.name in target_names:
                    info.disabled = False
                    info.errored = False
                elif info.name in other_names:
                    info.disabled = True

            action = "enabled"

        _engine.update_traded_coins()
        _engine._save_engine_state()

        return {"group": group_name, "status": action}

    # Engine stopped: toggle in DB
    if _engine.db:
        raw = _engine.db.load_state("__engine__", "state")
        state = json.loads(raw) if raw else {}
        disabled_list = set(state.get("disabled_strategies", []))

        target_active = not target_names.issubset(disabled_list)

        if target_active:
            # Disable all strategies in target group
            disabled_list.update(target_names)
            action = "disabled"
        else:
            # Enable target group, disable other groups
            disabled_list -= target_names
            for g in other_groups:
                disabled_list.update(groups[g])
            action = "enabled"

        state["disabled_strategies"] = sorted(disabled_list)
        _engine.db.save_state("__engine__", "state", json.dumps(state))

        return {"group": group_name, "status": action}

    return {"error": "no strategies loaded"}


@router.post("/{name}/toggle")
async def toggle_strategy(name: str):
    if not _engine:
        return {"error": "not initialized"}

    groups, strat_coins = _build_coin_group_map()

    # Engine running: toggle in-memory + persist
    for info in _engine._strategies:
        if info.name == name:
            info.disabled = not info.disabled
            if not info.disabled:
                info.errored = False
                # Disable strategies from other groups that share the same coin(s)
                conflicts = _find_coin_conflicts(name, groups, strat_coins)
                for other in _engine._strategies:
                    if other.name in conflicts and not other.disabled:
                        other.disabled = True
                        log.info("Auto-disabled %s (coin conflict with %s)", other.name, name)

            _engine.update_traded_coins()
            _engine._save_engine_state()
            return {
                "status": "disabled" if info.disabled else "enabled",
                "restart_required": False,
            }

    # Engine stopped: toggle in DB directly so it takes effect on next start
    if _engine.db and _engine.config:
        strat_name = name.replace(".py", "")
        known = [e.file.replace(".py", "") for e in _engine.config.strategies.active]
        if strat_name not in known:
            return {"error": "strategy not found"}

        raw = _engine.db.load_state("__engine__", "state")
        state = json.loads(raw) if raw else {}
        disabled_list = set(state.get("disabled_strategies", []))

        if strat_name in disabled_list:
            disabled_list.discard(strat_name)
            new_status = "enabled"
            # Disable strategies from other groups that share the same coin(s)
            conflicts = _find_coin_conflicts(strat_name, groups, strat_coins)
            disabled_list.update(conflicts)
        else:
            disabled_list.add(strat_name)
            new_status = "disabled"

        state["disabled_strategies"] = sorted(disabled_list)
        _engine.db.save_state("__engine__", "state", json.dumps(state))

        return {
            "status": new_status,
            "restart_required": False,
        }

    return {"error": "strategy not found"}
