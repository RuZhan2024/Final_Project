from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter, Query

from ..db import get_conn
from ..deploy_ops import derive_ops_params_from_yaml
from ..inmemory_state import apply_settings_update_inmem
from ..repositories.settings_repository import load_settings_snapshot, persist_settings_update
from ..schemas import SettingsUpdatePayload
from ..services.settings_service import (
    build_settings_snapshot_response,
    persist_settings_update_with_fallback,
)

router = APIRouter()

_derive_ops_params_from_yaml = derive_ops_params_from_yaml


@router.get("/api/settings")
@router.get("/api/v1/settings")
def get_settings(resident_id: int = Query(1, description="Resident id")) -> Dict[str, object]:
    return build_settings_snapshot_response(
        resident_id=resident_id,
        get_conn=get_conn,
        load_settings_snapshot=load_settings_snapshot,
        derive_ops_params_from_yaml=derive_ops_params_from_yaml,
    )


@router.put("/api/settings")
@router.put("/api/v1/settings")
def update_settings(payload: SettingsUpdatePayload, resident_id: Optional[int] = None) -> Dict[str, object]:
    rid = int(resident_id or 1)
    return persist_settings_update_with_fallback(
        payload=payload,
        resident_id=rid,
        get_conn=get_conn,
        persist_settings_update=persist_settings_update,
        apply_settings_update_inmem=apply_settings_update_inmem,
    )
