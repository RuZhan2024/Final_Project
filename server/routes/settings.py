from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter, Query

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

from ..core import (
    _derive_ops_params_from_yaml,
    apply_settings_update_inmem,
)
from ..db import get_conn
from ..repositories.settings_repository import load_settings_snapshot, persist_settings_update
from ..schemas import SettingsUpdatePayload
from ..services.settings_service import (
    apply_yaml_override as _service_apply_yaml_override,
    base_settings_snapshot,
    build_settings_response,
)

router = APIRouter()

def _apply_yaml_override(system: Dict[str, object]) -> None:
    _service_apply_yaml_override(system, _derive_ops_params_from_yaml)


@router.get("/api/settings")
@router.get("/api/v1/settings")
def get_settings(resident_id: int = Query(1, description="Resident id")) -> Dict[str, object]:
    system, deploy = base_settings_snapshot(resident_id)
    system.setdefault("camera_source", "webcam")
    db_available = False

    try:
        with get_conn() as conn:
            db_available = True
            load_settings_snapshot(conn, resident_id, system, deploy)
    except (MySQLError, RuntimeError, OSError, TypeError, ValueError):
        db_available = False

    _apply_yaml_override(system)
    return build_settings_response(resident_id, system, deploy, db_available=db_available)


@router.put("/api/settings")
@router.put("/api/v1/settings")
def update_settings(payload: SettingsUpdatePayload, resident_id: Optional[int] = None) -> Dict[str, object]:
    rid = int(resident_id or 1)

    if payload.fall_threshold is not None:
        try:
            value = float(payload.fall_threshold)
            if 1.0 < value <= 100.0:
                payload.fall_threshold = value / 100.0
        except (TypeError, ValueError):
            pass

    try:
        with get_conn() as conn:
            persist_settings_update(conn, rid, payload)
            return {"ok": True, "persisted": True}
    except (MySQLError, RuntimeError, OSError, TypeError, ValueError):
        apply_settings_update_inmem(payload, resident_id=rid)
        return {"ok": True, "persisted": False, "reason": "db_unavailable"}
