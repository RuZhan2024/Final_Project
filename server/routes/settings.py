from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Query

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

from ..core import (
    SettingsUpdatePayload,
    apply_settings_update_inmem,
    get_inmem_settings,
    _col_exists,
    _derive_ops_params_from_yaml,
    _ensure_system_settings_schema,
    _safe_get,
    _table_exists,
)
from ..db import get_conn

router = APIRouter()


def _apply_yaml_override(system: Dict[str, Any]) -> None:
    """Override UI thresholds/cooldown using YAML-derived deploy params (best-effort)."""
    try:
        system.setdefault("active_op_code", "OP-2")
        dp = _derive_ops_params_from_yaml(
            dataset_code=str(system.get("active_dataset_code") or "muvim"),
            model_code=str(system.get("active_model_code") or "HYBRID"),
            op_code=str(system.get("active_op_code") or "OP-2"),
        )
        system["deploy_params"] = dp
        ui = dp.get("ui") or {}
        system["fall_threshold"] = float(ui.get("tau_high", system.get("fall_threshold", 0.85)))
        system["tau_low"] = float(ui.get("tau_low", system.get("tau_low", 0.0)))
        system["active_op_code"] = str(ui.get("op_code", system.get("active_op_code", "OP-2")))
        system["alert_cooldown_sec"] = int(round(float(ui.get("cooldown_s", system.get("alert_cooldown_sec", 3)))))
    except (RuntimeError, OSError, TypeError, ValueError, KeyError):
        return


@router.get("/api/settings")
@router.get("/api/v1/settings")
def get_settings(resident_id: int = Query(1, description="Resident id")) -> Dict[str, Any]:
    """Return UI settings (nested + legacy flat fields).

    Works with DB when available; otherwise falls back to in-memory defaults.
    """
    base = get_inmem_settings(resident_id)
    system: Dict[str, Any] = base["system"]
    deploy: Dict[str, Any] = base["deploy"]

    # UI legacy field (not persisted by default)
    system.setdefault("camera_source", "webcam")

    db_available = False
    try:
        with get_conn() as conn:
            db_available = True
            _ensure_system_settings_schema(conn)

            if _table_exists(conn, "settings"):
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM settings WHERE resident_id=%s LIMIT 1", (resident_id,))
                    row = cur.fetchone() or {}

                system.update(
                    {
                        "monitoring_enabled": bool(_safe_get(row, "monitoring_enabled", system.get("monitoring_enabled", 0))),
                        "active_model_code": _safe_get(row, "active_model_code", system.get("active_model_code", "HYBRID")),
                        "active_operating_point": _safe_get(row, "active_operating_point", system.get("active_operating_point")),
                        "active_op_code": _safe_get(row, "active_op_code", system.get("active_op_code", "OP-2")),
                        "fall_threshold": float(_safe_get(row, "fall_threshold", system.get("fall_threshold", 0.85)) or 0.85),
                        "alert_cooldown_sec": int(_safe_get(row, "alert_cooldown_sec", system.get("alert_cooldown_sec", 3)) or 3),
                        "store_event_clips": bool(_safe_get(row, "store_event_clips", system.get("store_event_clips", 0))),
                        "anonymize_skeleton_data": bool(_safe_get(row, "anonymize_skeleton_data", system.get("anonymize_skeleton_data", 1))),
                        "require_confirmation": bool(_safe_get(row, "require_confirmation", system.get("require_confirmation", 0))),
                        "notify_on_every_fall": bool(_safe_get(row, "notify_on_every_fall", system.get("notify_on_every_fall", 1))),
                    }
                )
                deploy.update(
                    {
                        "fps": int(_safe_get(row, "fps", deploy.get("fps", 30)) or 30),
                        "window": {
                            "W": int(_safe_get(row, "window_size", deploy.get("window", {}).get("W", 48)) or 48),
                            "S": int(_safe_get(row, "stride", deploy.get("window", {}).get("S", 12)) or 12),
                        },
                    }
                )
            elif _table_exists(conn, "system_settings"):
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM system_settings WHERE resident_id=%s ORDER BY id ASC LIMIT 1",
                        (resident_id,),
                    )
                    sys_row = cur.fetchone() or {}

                if isinstance(sys_row, dict) and "monitoring_enabled" in sys_row:
                    system["monitoring_enabled"] = bool(sys_row.get("monitoring_enabled", system.get("monitoring_enabled", 0)))

                if isinstance(sys_row, dict) and sys_row.get("camera_source"):
                    system["camera_source"] = sys_row.get("camera_source")

                if isinstance(sys_row, dict) and sys_row.get("active_model_code"):
                    system["active_model_code"] = sys_row.get("active_model_code") or system.get("active_model_code", "HYBRID")

                # optional columns (best effort)
                for col, setter in [
                    ("p_fall_threshold", lambda v: system.__setitem__("fall_threshold", float(v))),
                    ("fall_threshold", lambda v: system.__setitem__("fall_threshold", float(v))),
                    ("alert_cooldown_sec", lambda v: system.__setitem__("alert_cooldown_sec", int(v))),
                    ("store_event_clips", lambda v: system.__setitem__("store_event_clips", bool(v))),
                    ("anonymize_skeleton_data", lambda v: system.__setitem__("anonymize_skeleton_data", bool(v))),
                    ("require_confirmation", lambda v: system.__setitem__("require_confirmation", bool(v))),
                    ("notify_on_every_fall", lambda v: system.__setitem__("notify_on_every_fall", bool(v))),
                    ("fps", lambda v: deploy.__setitem__("fps", int(v))),
                    ("window_size", lambda v: deploy.setdefault("window", {}).__setitem__("W", int(v))),
                    ("stride", lambda v: deploy.setdefault("window", {}).__setitem__("S", int(v))),
                    ("mc_M", lambda v: deploy.setdefault("mc", {}).__setitem__("M", int(v))),
                    ("mc_M_confirm", lambda v: deploy.setdefault("mc", {}).__setitem__("M_confirm", int(v))),
                    ("active_dataset_code", lambda v: system.__setitem__("active_dataset_code", str(v).lower())),
                    ("active_op_code", lambda v: system.__setitem__("active_op_code", str(v).upper())),
                    ("mc_enabled", lambda v: system.__setitem__("mc_enabled", bool(int(v)) if str(v).isdigit() else bool(v))),
                ]:
                    if isinstance(sys_row, dict) and col in sys_row and sys_row.get(col) is not None:
                        try:
                            setter(sys_row[col])
                        except (TypeError, ValueError):
                            pass

                system["active_operating_point"] = (
                    sys_row.get("active_operating_point_id") or sys_row.get("active_operating_point") or system.get("active_operating_point")
                )

                # keep consistent with variants setting schema if needed
    except (MySQLError, RuntimeError, OSError, TypeError, ValueError):
        db_available = False

    _apply_yaml_override(system)

    return {
        "system": system,
        "deploy": deploy,
        "privacy": {
            "store_event_clips": system.get("store_event_clips", False),
            "anonymize_skeleton_data": system.get("anonymize_skeleton_data", True),
        },
        "db_available": bool(db_available),
        # legacy flat keys
        "monitoring_enabled": system.get("monitoring_enabled", False),
        "active_model_code": system.get("active_model_code", "HYBRID"),
        "active_operating_point": system.get("active_operating_point"),
        "active_op_code": system.get("active_op_code"),
        "fall_threshold": system.get("fall_threshold", 0.85),
        "alert_cooldown_sec": system.get("alert_cooldown_sec", 3),
    }


@router.put("/api/settings")
@router.put("/api/v1/settings")
def update_settings(payload: SettingsUpdatePayload, resident_id: Optional[int] = None) -> Dict[str, Any]:
    """Update settings.

    Prefers DB persistence, but falls back to in-memory updates if DB is unavailable.
    """
    rid = int(resident_id or 1)

    # Accept either 0-1 or 0-100.
    if payload.fall_threshold is not None:
        try:
            v = float(payload.fall_threshold)
            if 1.0 < v <= 100.0:
                payload.fall_threshold = v / 100.0
        except (TypeError, ValueError):
            pass

    try:
        with get_conn() as conn:
            _ensure_system_settings_schema(conn)

            if _table_exists(conn, "settings"):
                sets = []
                vals = []

                if payload.monitoring_enabled is not None:
                    sets.append("monitoring_enabled=%s")
                    vals.append(1 if payload.monitoring_enabled else 0)

                if payload.fall_threshold is not None:
                    sets.append("fall_threshold=%s")
                    vals.append(payload.fall_threshold)

                if payload.alert_cooldown_sec is not None:
                    sets.append("alert_cooldown_sec=%s")
                    vals.append(payload.alert_cooldown_sec)

                if payload.store_event_clips is not None:
                    sets.append("store_event_clips=%s")
                    vals.append(1 if payload.store_event_clips else 0)

                if payload.anonymize_skeleton_data is not None:
                    sets.append("anonymize_skeleton_data=%s")
                    vals.append(1 if payload.anonymize_skeleton_data else 0)

                if payload.require_confirmation is not None:
                    sets.append("require_confirmation=%s")
                    vals.append(1 if payload.require_confirmation else 0)

                if payload.notify_on_every_fall is not None:
                    sets.append("notify_on_every_fall=%s")
                    vals.append(1 if payload.notify_on_every_fall else 0)

                if payload.active_model_code is not None:
                    sets.append("active_model_code=%s")
                    vals.append(payload.active_model_code)

                if payload.active_operating_point is not None:
                    sets.append("active_operating_point=%s")
                    vals.append(payload.active_operating_point)

                if payload.active_op_code is not None and _col_exists(conn, "settings", "active_op_code"):
                    sets.append("active_op_code=%s")
                    vals.append(str(payload.active_op_code).upper())

                if sets:
                    sql = "UPDATE settings SET " + ", ".join(sets) + ", updated_at=NOW() WHERE resident_id=%s"
                    vals.append(rid)
                    with conn.cursor() as cur:
                        cur.execute(sql, tuple(vals))
                    conn.commit()

                return {"ok": True, "persisted": True}

            # v2: system_settings
            sets = []
            vals = []

            def add(col: str, expr: str, value: Any) -> None:
                if _col_exists(conn, "system_settings", col):
                    sets.append(expr)
                    vals.append(value)

            if payload.monitoring_enabled is not None:
                add("monitoring_enabled", "monitoring_enabled=%s", 1 if payload.monitoring_enabled else 0)

            if payload.fall_threshold is not None:
                if _col_exists(conn, "system_settings", "p_fall_threshold"):
                    sets.append("p_fall_threshold=%s")
                    vals.append(payload.fall_threshold)
                else:
                    add("fall_threshold", "fall_threshold=%s", payload.fall_threshold)

            if payload.alert_cooldown_sec is not None:
                add("alert_cooldown_sec", "alert_cooldown_sec=%s", payload.alert_cooldown_sec)

            if payload.store_event_clips is not None:
                add("store_event_clips", "store_event_clips=%s", 1 if payload.store_event_clips else 0)

            if payload.anonymize_skeleton_data is not None:
                add("anonymize_skeleton_data", "anonymize_skeleton_data=%s", 1 if payload.anonymize_skeleton_data else 0)

            if payload.require_confirmation is not None:
                add("require_confirmation", "require_confirmation=%s", 1 if payload.require_confirmation else 0)

            if payload.notify_on_every_fall is not None:
                add("notify_on_every_fall", "notify_on_every_fall=%s", 1 if payload.notify_on_every_fall else 0)

            if payload.active_dataset_code is not None:
                add("active_dataset_code", "active_dataset_code=%s", (payload.active_dataset_code or "").lower())

            if payload.active_op_code is not None and _col_exists(conn, "system_settings", "active_op_code"):
                add("active_op_code", "active_op_code=%s", str(payload.active_op_code).upper())

            if payload.mc_enabled is not None:
                add("mc_enabled", "mc_enabled=%s", 1 if payload.mc_enabled else 0)

            if payload.mc_M is not None:
                add("mc_M", "mc_M=%s", int(payload.mc_M))

            if payload.mc_M_confirm is not None:
                add("mc_M_confirm", "mc_M_confirm=%s", int(payload.mc_M_confirm))

            # Active model selection
            if payload.active_model_code is not None:
                if _col_exists(conn, "system_settings", "active_model_code"):
                    add("active_model_code", "active_model_code=%s", payload.active_model_code)
                if _col_exists(conn, "system_settings", "active_model_id") and _table_exists(conn, "models"):
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT id FROM models WHERE model_code=%s OR code=%s LIMIT 1",
                            (payload.active_model_code, payload.active_model_code),
                        )
                        mrow = cur.fetchone()
                    if mrow and isinstance(mrow, dict) and mrow.get("id") is not None:
                        sets.append("active_model_id=%s")
                        vals.append(int(mrow["id"]))

            if payload.active_operating_point is not None:
                if _col_exists(conn, "system_settings", "active_operating_point_id"):
                    sets.append("active_operating_point_id=%s")
                    vals.append(payload.active_operating_point)
                else:
                    add("active_operating_point", "active_operating_point=%s", payload.active_operating_point)

            if sets:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM system_settings WHERE resident_id=%s ORDER BY id ASC LIMIT 1",
                        (rid,),
                    )
                    row = cur.fetchone()
                    if not row:
                        cur.execute("INSERT INTO system_settings (resident_id) VALUES (%s)", (rid,))
                        conn.commit()

                sql = "UPDATE system_settings SET " + ", ".join(sets) + ", updated_at=NOW() WHERE resident_id=%s"
                vals.append(rid)
                with conn.cursor() as cur:
                    cur.execute(sql, tuple(vals))
                conn.commit()

            return {"ok": True, "persisted": True}

    except (MySQLError, RuntimeError, OSError, TypeError, ValueError):
        apply_settings_update_inmem(payload, resident_id=rid)
        return {"ok": True, "persisted": False, "reason": "db_unavailable"}
