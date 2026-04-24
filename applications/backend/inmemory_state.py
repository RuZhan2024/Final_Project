from __future__ import annotations

"""In-memory fallback state for settings/caregiver flows without a live DB."""

import json

from datetime import datetime, timezone
from typing import Any, Dict, List

from .schemas import CaregiverUpsertPayload, SettingsUpdatePayload

_DEFAULT_SYSTEM_SETTINGS: Dict[str, Any] = {
    "monitoring_enabled": False,
    "api_online": True,
    "alert_cooldown_sec": 3,
    "notify_on_every_fall": True,
    "notify_sms": False,
    "notify_phone": False,
    "fall_threshold": 0.71,
    "store_event_clips": False,
    "anonymize_skeleton_data": True,
    "store_anonymized_data": False,
    "active_model_code": "TCN",
    "active_operating_point": None,
    "active_dataset_code": "caucafall",
    "active_op_code": "OP-2",
    "mc_enabled": False,
    "mc_M": 10,
    "mc_M_confirm": 25,
}

_DEFAULT_DEPLOY_SETTINGS: Dict[str, Any] = {
    "fps": 30,
    "window": {"W": 48, "S": 12},
    "mc": {"M": 10, "M_confirm": 25},
}

_INMEM_SETTINGS: Dict[int, Dict[str, Any]] = {}
_INMEM_CAREGIVERS: Dict[int, List[Dict[str, Any]]] = {}


def get_inmem_settings(resident_id: int = 1) -> Dict[str, Any]:
    """Return a deep-copied settings snapshot for the requested resident."""
    rid = int(resident_id or 1)
    if rid not in _INMEM_SETTINGS:
        _INMEM_SETTINGS[rid] = {
            "system": dict(_DEFAULT_SYSTEM_SETTINGS),
            "deploy": json.loads(json.dumps(_DEFAULT_DEPLOY_SETTINGS)),
        }
    return {
        "system": dict(_INMEM_SETTINGS[rid]["system"]),
        "deploy": json.loads(json.dumps(_INMEM_SETTINGS[rid]["deploy"])),
    }


def apply_settings_update_inmem(
    payload: SettingsUpdatePayload,
    *,
    resident_id: int = 1,
    normalize_model_code,
    normalize_dataset_code,
    norm_op_code,
) -> None:
    """Apply a settings patch to the in-memory fallback store."""
    rid = int(resident_id or 1)
    cur = get_inmem_settings(rid)
    system = cur["system"]
    deploy = cur["deploy"]

    if payload.fall_threshold is not None:
        try:
            v = float(payload.fall_threshold)
            if 1.0 < v <= 100.0:
                v = v / 100.0
            system["fall_threshold"] = max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            pass

    if payload.store_anonymized_data is not None:
        v = bool(payload.store_anonymized_data)
        # Legacy UI toggles still treat this as the combined clip+anonymize switch.
        system["store_event_clips"] = v
        system["anonymize_skeleton_data"] = v
        system["store_anonymized_data"] = v

    for key in [
        "monitoring_enabled",
        "api_online",
        "notify_on_every_fall",
        "notify_sms",
        "notify_phone",
        "store_event_clips",
        "anonymize_skeleton_data",
        "mc_enabled",
    ]:
        v = getattr(payload, key, None)
        if v is not None:
            system[key] = bool(v)

    if payload.notify_on_every_fall is False:
        system["notify_sms"] = False
        system["notify_phone"] = False

    system["store_anonymized_data"] = bool(
        system.get("store_event_clips", False) and system.get("anonymize_skeleton_data", True)
    )

    if payload.alert_cooldown_sec is not None:
        try:
            system["alert_cooldown_sec"] = int(payload.alert_cooldown_sec)
        except (TypeError, ValueError):
            pass

    if payload.active_model_code is not None:
        system["active_model_code"] = normalize_model_code(payload.active_model_code, default="TCN")
    if payload.active_operating_point is not None:
        system["active_operating_point"] = payload.active_operating_point
    if payload.active_dataset_code is not None:
        system["active_dataset_code"] = normalize_dataset_code(payload.active_dataset_code, default="caucafall")
    if payload.active_op_code is not None:
        system["active_op_code"] = norm_op_code(payload.active_op_code)
    if payload.mc_M is not None:
        try:
            system["mc_M"] = int(payload.mc_M)
            deploy.setdefault("mc", {})["M"] = int(payload.mc_M)
        except (TypeError, ValueError):
            pass
    if payload.mc_M_confirm is not None:
        try:
            system["mc_M_confirm"] = int(payload.mc_M_confirm)
            deploy.setdefault("mc", {})["M_confirm"] = int(payload.mc_M_confirm)
        except (TypeError, ValueError):
            pass

    _INMEM_SETTINGS[rid] = {"system": system, "deploy": deploy}


def get_inmem_caregivers(resident_id: int = 1) -> List[Dict[str, Any]]:
    """Return a copy of caregiver rows from the in-memory fallback store."""
    rid = int(resident_id or 1)
    rows = _INMEM_CAREGIVERS.get(rid, [])
    return [dict(r) for r in rows]


def upsert_inmem_caregiver(payload: CaregiverUpsertPayload) -> Dict[str, Any]:
    """Insert or update the fallback caregiver row for one resident."""
    rid = int(payload.resident_id or 1)
    rows = _INMEM_CAREGIVERS.setdefault(rid, [])
    target_id = int(payload.id) if payload.id else (rows[0]["id"] if rows else 1)

    row = None
    for candidate in rows:
        if int(candidate.get("id", 0)) == target_id:
            row = candidate
            break

    now = datetime.now(timezone.utc).isoformat()
    if row is None:
        # Fallback mode still keeps timestamps so responses match DB-backed shape.
        row = {
            "id": target_id,
            "resident_id": rid,
            "name": None,
            "email": None,
            "phone": None,
            "telegram_chat_id": None,
            "created_at": now,
            "updated_at": now,
        }
        rows.append(row)

    if payload.name is not None:
        row["name"] = payload.name
    if payload.email is not None:
        row["email"] = payload.email
    if payload.phone is not None:
        row["phone"] = payload.phone
    if payload.telegram_chat_id is not None:
        row["telegram_chat_id"] = payload.telegram_chat_id
    row["updated_at"] = now
    return dict(row)
