from __future__ import annotations

"""Settings API routes and compatibility aliases.

This module keeps the HTTP contract for reading and updating monitor settings.
Persistence, YAML-derived operating-point overlays, and in-memory fallback live
in neighbouring services/repositories; the route layer only wires FastAPI
inputs to those lower-level contracts and preserves legacy path aliases.
"""

from typing import Dict, Optional

from fastapi import APIRouter, Query

from ..code_normalization import norm_op_code, normalize_dataset_code, normalize_model_code
from ..db import get_conn
from ..deploy_ops import derive_ops_params_from_yaml
from ..inmemory_state import apply_settings_update_inmem
from ..repositories.settings_repository import load_settings_snapshot, persist_settings_update
from ..schemas import SettingsUpdatePayload
from ..services.settings_service import (
    apply_yaml_override,
    build_settings_snapshot_response,
    persist_settings_update_with_fallback,
)

router = APIRouter()

# Route-level aliases keep older tests and monkeypatch-based contract checks
# pointed at the same deploy/YAML override path after the service split.
_derive_ops_params_from_yaml = derive_ops_params_from_yaml


def _apply_yaml_override(system: Dict[str, object]) -> None:
    """Apply deploy YAML overrides through the route-level alias seam.

    Tests patch ``_derive_ops_params_from_yaml`` on this module, so the wrapper
    preserves that injection point after the logic moved into a service.
    """
    apply_yaml_override(system, _derive_ops_params_from_yaml)


@router.get("/api/settings")
@router.get("/api/v1/settings")
def get_settings(resident_id: int = Query(1, description="Resident id")) -> Dict[str, object]:
    """Return the merged settings snapshot for one resident.

    The response combines in-memory defaults, database-backed values when the
    DB is reachable, and deploy-time YAML overlays for operating-point UI state.
    """
    return build_settings_snapshot_response(
        resident_id=resident_id,
        get_conn=get_conn,
        load_settings_snapshot=load_settings_snapshot,
        derive_ops_params_from_yaml=_derive_ops_params_from_yaml,
    )


@router.put("/api/settings")
@router.put("/api/v1/settings")
def update_settings(payload: SettingsUpdatePayload, resident_id: Optional[int] = None) -> Dict[str, object]:
    """Persist settings updates or fall back to in-memory runtime state.

    The route keeps the historical ``ok`` response contract even during DB
    outages. Durability is communicated via the returned ``persisted`` flag.
    """
    rid = int(resident_id or 1)
    return persist_settings_update_with_fallback(
        payload=payload,
        resident_id=rid,
        get_conn=get_conn,
        persist_settings_update=persist_settings_update,
        apply_settings_update_inmem=apply_settings_update_inmem,
        normalize_model_code=normalize_model_code,
        normalize_dataset_code=normalize_dataset_code,
        norm_op_code=norm_op_code,
    )
