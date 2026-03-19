import json
import logging
import os
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Request

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])

_engine = None
_config_path = "config/bot_config.json"


def init(engine, config_path: str = "config/bot_config.json"):
    global _engine, _config_path
    _engine = engine
    _config_path = config_path


@router.get("")
async def get_settings():
    if not _engine or not _engine.config:
        return {"error": "not initialized"}

    cfg = _engine.config
    strategies = []
    for entry in cfg.strategies.active:
        strategies.append({
            "file": entry.file,
            "role": entry.role,
            "coins": entry.coins,
            "paper_mode": entry.paper_mode,
            "paper_balance": entry.paper_balance,
        })

    return {
        "risk": {
            "daily_loss_pct": cfg.risk.daily_loss_pct,
            "emergency_close_pct": cfg.risk.emergency_close_pct,
            "max_position_pct": cfg.risk.max_position_pct,
            "max_leverage": cfg.risk.max_leverage,
        },
        "strategies": strategies,
    }


@router.put("")
async def update_settings(request: Request):
    if not _engine or not _engine.config:
        return {"error": "not initialized"}

    body = await request.json()

    resolved = Path(_config_path).resolve()
    if not resolved.exists():
        return {"error": "config file not found"}

    try:
        with open(resolved, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception:
        return {"error": "failed to read config"}

    restart_required = False

    if "risk" in body:
        r = body["risk"]
        if "daily_loss_pct" in r:
            v = _clamp(float(r["daily_loss_pct"]), 1, 50)
            config_data.setdefault("risk", {})["daily_loss_pct"] = v
        if "emergency_close_pct" in r:
            v = _clamp(float(r["emergency_close_pct"]), 1, 50)
            config_data.setdefault("risk", {})["emergency_close_pct"] = v
        if "max_position_pct" in r:
            v = _clamp(float(r["max_position_pct"]), 10, 10000)
            config_data.setdefault("risk", {})["max_position_pct"] = v
        if "max_leverage" in r:
            v = int(_clamp(float(r["max_leverage"]), 1, 50))
            config_data.setdefault("risk", {})["max_leverage"] = v
        restart_required = True

    if "strategies" in body:
        new_strategies = []
        all_coins: dict[str, str] = {}
        _VALID_ROLES = {"primary", "secondary", ""}

        for s in body["strategies"]:
            # V-05: Validate strategy file name
            file_name = s.get("file", "")
            if file_name and not re.match(r'^[a-zA-Z0-9_-]+\.py$', file_name):
                return {"error": f"Invalid strategy filename: {file_name}", "status": "error"}

            # V-07: Validate role against whitelist
            role = s.get("role", "")
            if role not in _VALID_ROLES:
                return {"error": f"Invalid role: {role}. Must be one of: primary, secondary, or empty", "status": "error"}

            coins = s.get("coins", [])
            for coin in coins:
                if not re.match(r"^[A-Z]{2,10}$", coin):
                    return {"error": f"Invalid coin format: {coin}", "status": "error"}
                if coin in all_coins:
                    return {
                        "error": f"Coin {coin} assigned to both {all_coins[coin]} and {file_name}",
                        "status": "error",
                    }
                all_coins[coin] = file_name

            entry = {
                "file": file_name,
                "role": role,
                "coins": coins,
            }
            if "paper_mode" in s and s["paper_mode"] is not None:
                entry["paper_mode"] = bool(s["paper_mode"])
            if "paper_balance" in s:
                entry["paper_balance"] = max(0, float(s["paper_balance"]))

            new_strategies.append(entry)

        config_data.setdefault("strategies", {})["active"] = new_strategies
        restart_required = True

    try:
        dir_path = resolved.parent
        fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        os.replace(tmp_path, str(resolved))
    except Exception:
        log.exception("Failed to save config")
        return {"error": "failed to save config", "status": "error"}

    return {
        "status": "ok",
        "restart_required": restart_required,
        "message": "Configuration saved" + (" — restart bot to apply" if restart_required else ""),
    }


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))
