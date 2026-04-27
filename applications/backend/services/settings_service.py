from __future__ import annotations

"""Settings snapshot and update helpers.

These helpers sit between the FastAPI route layer and storage/runtime state.
They normalize settings payloads, merge deploy-time operating-point overlays,
and preserve the frontend contract when persistence falls back to in-memory
state during DB outages.
"""

from typing import Any, Dict

from ..code_normalization import norm_op_code, normalize_dataset_code, normalize_model_code
from ..inmemory_state import get_inmem_settings


def apply_yaml_override(system: Dict[str, Any], derive_ops_params_from_yaml) -> None:
    """Overlay deploy-time operating-point UI values onto a settings snapshot.

    The YAML layer is advisory rather than authoritative: failures are swallowed
    so settings endpoints can still serve DB/in-memory defaults when deploy
    assets are absent or temporarily inconsistent.
    """
    try:
        system.setdefault("active_dataset_code", "caucafall")
        system.setdefault("active_model_code", "TCN")
        system.setdefault("active_op_code", "OP-2")
        dp = derive_ops_params_from_yaml(
            dataset_code=normalize_dataset_code(str(system.get("active_dataset_code") or "caucafall")),
            model_code=str(system.get("active_model_code") or "TCN"),
            op_code=str(system.get("active_op_code") or "OP-2"),
        )
        system["deploy_params"] = dp
        ui = dp.get("ui") or {}
        system["fall_threshold"] = float(ui.get("tau_high", system.get("fall_threshold", 0.71)))
        system["tau_low"] = float(ui.get("tau_low", system.get("tau_low", 0.0)))
        system["active_op_code"] = norm_op_code(str(ui.get("op_code", system.get("active_op_code", "OP-2"))))
        system["alert_cooldown_sec"] = int(round(float(ui.get("cooldown_s", system.get("alert_cooldown_sec", 3)))))
    except (RuntimeError, OSError, TypeError, ValueError, KeyError):
        return


def build_settings_response(resident_id: int, system: Dict[str, Any], deploy: Dict[str, Any], *, db_available: bool) -> Dict[str, Any]:
    """Normalize the mixed DB/deploy snapshot into the frontend settings contract.

    The returned payload intentionally duplicates some top-level fields from the
    nested ``system`` object because older frontend code reads both shapes.
    """
    system["active_dataset_code"] = normalize_dataset_code(system.get("active_dataset_code"))
    system["active_model_code"] = normalize_model_code(system.get("active_model_code"))
    system["active_op_code"] = norm_op_code(str(system.get("active_op_code") or "OP-2"))
    # The privacy toggle exposed in the UI is a derived capability: storing
    # anonymized data only makes sense when clip persistence itself is enabled.
    system["store_anonymized_data"] = bool(
        system.get("store_event_clips", False) and system.get("anonymize_skeleton_data", True)
    )

    return {
        "system": system,
        "deploy": deploy,
        "privacy": {
            "store_anonymized_data": system.get("store_anonymized_data", False),
            "store_event_clips": system.get("store_event_clips", False),
            "anonymize_skeleton_data": system.get("anonymize_skeleton_data", True),
        },
        "db_available": bool(db_available),
        "monitoring_enabled": system.get("monitoring_enabled", False),
        "active_model_code": system.get("active_model_code", "TCN"),
        "active_operating_point": system.get("active_operating_point"),
        "active_op_code": system.get("active_op_code"),
        "fall_threshold": system.get("fall_threshold", 0.71),
        "alert_cooldown_sec": system.get("alert_cooldown_sec", 3),
    }


def base_settings_snapshot(resident_id: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Return mutable baseline settings dictionaries for one resident.

    Callers are expected to enrich these dicts in place with DB and deploy
    values before serializing them back to the route response.
    """
    base = get_inmem_settings(resident_id)
    return base["system"], base["deploy"]


def normalize_settings_payload_threshold(payload) -> None:
    """Accept UI thresholds expressed either as probability or percentage."""
    if payload.fall_threshold is None:
        return
    try:
        value = float(payload.fall_threshold)
        if 1.0 < value <= 100.0:
            payload.fall_threshold = value / 100.0
    except (TypeError, ValueError):
        return


def build_settings_snapshot_response(
    *,
    resident_id: int,
    get_conn,
    load_settings_snapshot,
    derive_ops_params_from_yaml,
) -> Dict[str, object]:
    """Load the read-only settings snapshot with DB fallback and deploy overlays.

    Resolution order is: in-memory defaults -> DB snapshot when reachable ->
    YAML-derived operating-point UI fields. The DB availability flag is kept
    explicit so the frontend can message degraded persistence honestly.
    """
    system, deploy = base_settings_snapshot(resident_id)
    system.setdefault("camera_source", "webcam")
    db_available = False

    try:
        with get_conn() as conn:
            db_available = True
            load_settings_snapshot(conn, resident_id, system, deploy)
    except Exception:
        db_available = False

    apply_yaml_override(system, derive_ops_params_from_yaml)
    return build_settings_response(resident_id, system, deploy, db_available=db_available)


def persist_settings_update_with_fallback(
    *,
    payload,
    resident_id: int,
    get_conn,
    persist_settings_update,
    apply_settings_update_inmem,
    normalize_model_code,
    normalize_dataset_code,
    norm_op_code,
) -> Dict[str, object]:
    """Persist settings when possible, otherwise fall back to in-memory state.

    This preserves the route contract during SQLite/MySQL outages: callers still
    get an `ok` response plus an explicit `persisted` flag describing durability.
    """
    normalize_settings_payload_threshold(payload)

    try:
        with get_conn() as conn:
            persist_settings_update(conn, resident_id, payload)
            return {"ok": True, "persisted": True}
    except Exception:
        # Apply the same normalized payload to the live in-memory store so the
        # UI reflects the user's intent immediately even without durable storage.
        apply_settings_update_inmem(
            payload,
            resident_id=resident_id,
            normalize_model_code=normalize_model_code,
            normalize_dataset_code=normalize_dataset_code,
            norm_op_code=norm_op_code,
        )
        return {"ok": True, "persisted": False, "reason": "db_unavailable"}
