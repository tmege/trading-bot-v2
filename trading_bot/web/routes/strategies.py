import logging
import os
from pathlib import Path

from fastapi import APIRouter

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategies", tags=["strategies"])

_engine = None


def init(engine):
    global _engine
    _engine = engine


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
        import json

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


@router.post("/{name}/toggle")
async def toggle_strategy(name: str):
    if not _engine:
        return {"error": "not initialized"}

    # Engine running: toggle in-memory + persist
    for info in _engine._strategies:
        if info.name == name:
            info.disabled = not info.disabled
            if not info.disabled:
                info.errored = False
            _engine._save_engine_state()
            return {
                "status": "disabled" if info.disabled else "enabled",
                "restart_required": False,
            }

    # Engine stopped: toggle in DB directly so it takes effect on next start
    if _engine.db and _engine.config:
        import json
        strat_name = name.replace(".py", "")
        known = [e.file.replace(".py", "") for e in _engine.config.strategies.active]
        if strat_name not in known:
            return {"error": "strategy not found"}

        raw = _engine.db.load_state("__engine__", "state")
        state = json.loads(raw) if raw else {}
        disabled_list = state.get("disabled_strategies", [])

        if strat_name in disabled_list:
            disabled_list.remove(strat_name)
            new_status = "enabled"
        else:
            disabled_list.append(strat_name)
            new_status = "disabled"

        state["disabled_strategies"] = disabled_list
        _engine.db.save_state("__engine__", "state", json.dumps(state))

        return {
            "status": new_status,
            "restart_required": False,
        }

    return {"error": "strategy not found"}
