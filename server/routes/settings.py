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
    normalize_dataset_code,
    normalize_model_code,
    _col_exists,
    _derive_ops_params_from_yaml,
    _ensure_system_settings_schema,
    _safe_get,
    _table_exists,
)
from ..db import get_conn

router = APIRouter()


def _updated_at_expr(conn) -> str:
    return "CURRENT_TIMESTAMP" if str(getattr(conn, "db_backend", "mysql")).lower() == "sqlite" else "NOW()"


def _apply_yaml_override(system: Dict[str, Any]) -> None:
    """Override UI thresholds/cooldown using YAML-derived deploy params (best-effort)."""
    try:
        system.setdefault("active_op_code", "OP-2")
        dp = _derive_ops_params_from_yaml(
            dataset_code=normalize_dataset_code(str(system.get("active_dataset_code") or "caucafall")),
            model_code=str(system.get("active_model_code") or "TCN"),
            op_code=str(system.get("active_op_code") or "OP-2"),
        )
        system["deploy_params"] = dp
        ui = dp.get("ui") or {}
        system["fall_threshold"] = float(ui.get("tau_high", system.get("fall_threshold", 0.71)))
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
                        "active_model_code": _safe_get(row, "active_model_code", system.get("active_model_code", "TCN")),
                        "active_operating_point": _safe_get(row, "active_operating_point", system.get("active_operating_point")),
                        "active_op_code": _safe_get(row, "active_op_code", system.get("active_op_code", "OP-2")),
                        "fall_threshold": float(_safe_get(row, "fall_threshold", system.get("fall_threshold", 0.71)) or 0.71),
                        "alert_cooldown_sec": int(_safe_get(row, "alert_cooldown_sec", system.get("alert_cooldown_sec", 3)) or 3),
                        "store_event_clips": bool(_safe_get(row, "store_event_clips", system.get("store_event_clips", 0))),
                        "anonymize_skeleton_data": bool(_safe_get(row, "anonymize_skeleton_data", system.get("anonymize_skeleton_data", 1))),
                        "notify_on_every_fall": bool(_safe_get(row, "notify_on_every_fall", system.get("notify_on_every_fall", 1))),
                        "notify_sms": bool(_safe_get(row, "notify_sms", system.get("notify_sms", 0))),
                        "notify_phone": bool(_safe_get(row, "notify_phone", system.get("notify_phone", 0))),
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
                    system["active_model_code"] = sys_row.get("active_model_code") or system.get("active_model_code", "TCN")

                # optional columns (best effort)
                for col, setter in [
                    ("p_fall_threshold", lambda v: system.__setitem__("fall_threshold", float(v))),
                    ("fall_threshold", lambda v: system.__setitem__("fall_threshold", float(v))),
                    ("alert_cooldown_sec", lambda v: system.__setitem__("alert_cooldown_sec", int(v))),
                    ("store_event_clips", lambda v: system.__setitem__("store_event_clips", bool(v))),
                    ("anonymize_skeleton_data", lambda v: system.__setitem__("anonymize_skeleton_data", bool(v))),
                    ("notify_on_every_fall", lambda v: system.__setitem__("notify_on_every_fall", bool(v))),
                    ("notify_sms", lambda v: system.__setitem__("notify_sms", bool(v))),
                    ("notify_phone", lambda v: system.__setitem__("notify_phone", bool(v))),
                    ("fps", lambda v: deploy.__setitem__("fps", int(v))),
                    ("window_size", lambda v: deploy.setdefault("window", {}).__setitem__("W", int(v))),
                    ("stride", lambda v: deploy.setdefault("window", {}).__setitem__("S", int(v))),
                    ("mc_M", lambda v: deploy.setdefault("mc", {}).__setitem__("M", int(v))),
                    ("mc_M_confirm", lambda v: deploy.setdefault("mc", {}).__setitem__("M_confirm", int(v))),
                    ("active_dataset_code", lambda v: system.__setitem__("active_dataset_code", normalize_dataset_code(v))),
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

    system["active_dataset_code"] = normalize_dataset_code(system.get("active_dataset_code"))
    system["active_model_code"] = normalize_model_code(system.get("active_model_code"))
    system["store_anonymized_data"] = bool(
        system.get("store_event_clips", False) and system.get("anonymize_skeleton_data", True)
    )
    _apply_yaml_override(system)

    return {
        "system": system,
        "deploy": deploy,
        "privacy": {
            "store_anonymized_data": system.get("store_anonymized_data", False),
            "store_event_clips": system.get("store_event_clips", False),
            "anonymize_skeleton_data": system.get("anonymize_skeleton_data", True),
        },
        "db_available": bool(db_available),
        # legacy flat keys
        "monitoring_enabled": system.get("monitoring_enabled", False),
        "active_model_code": system.get("active_model_code", "TCN"),
        "active_operating_point": system.get("active_operating_point"),
        "active_op_code": system.get("active_op_code"),
        "fall_threshold": system.get("fall_threshold", 0.71),
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
                store_anonymized_data = payload.store_anonymized_data

                if payload.monitoring_enabled is not None:
                    sets.append("monitoring_enabled=%s")
                    vals.append(1 if payload.monitoring_enabled else 0)

                if payload.fall_threshold is not None:
                    sets.append("fall_threshold=%s")
                    vals.append(payload.fall_threshold)

                if payload.alert_cooldown_sec is not None:
                    sets.append("alert_cooldown_sec=%s")
                    vals.append(payload.alert_cooldown_sec)

                if store_anonymized_data is not None:
                    sets.append("store_event_clips=%s")
                    vals.append(1 if store_anonymized_data else 0)
                    sets.append("anonymize_skeleton_data=%s")
                    vals.append(1 if store_anonymized_data else 0)

                if payload.store_event_clips is not None:
                    sets.append("store_event_clips=%s")
                    vals.append(1 if payload.store_event_clips else 0)

                if payload.anonymize_skeleton_data is not None:
                    sets.append("anonymize_skeleton_data=%s")
                    vals.append(1 if payload.anonymize_skeleton_data else 0)

                if payload.notify_on_every_fall is not None:
                    sets.append("notify_on_every_fall=%s")
                    vals.append(1 if payload.notify_on_every_fall else 0)
                    if payload.notify_on_every_fall is False:
                        if _col_exists(conn, "settings", "notify_sms"):
                            sets.append("notify_sms=%s")
                            vals.append(0)
                        if _col_exists(conn, "settings", "notify_phone"):
                            sets.append("notify_phone=%s")
                            vals.append(0)

                if payload.notify_sms is not None and _col_exists(conn, "settings", "notify_sms"):
                    sets.append("notify_sms=%s")
                    vals.append(1 if payload.notify_sms else 0)

                if payload.notify_phone is not None and _col_exists(conn, "settings", "notify_phone"):
                    sets.append("notify_phone=%s")
                    vals.append(1 if payload.notify_phone else 0)

                if payload.active_model_code is not None:
                    sets.append("active_model_code=%s")
                    vals.append(normalize_model_code(payload.active_model_code))

                if payload.active_operating_point is not None:
                    sets.append("active_operating_point=%s")
                    vals.append(payload.active_operating_point)

                if payload.active_op_code is not None and _col_exists(conn, "settings", "active_op_code"):
                    sets.append("active_op_code=%s")
                    vals.append(str(payload.active_op_code).upper())

                if sets:
                    sql = "UPDATE settings SET " + ", ".join(sets) + f", updated_at={_updated_at_expr(conn)} WHERE resident_id=%s"
                    vals.append(rid)
                    with conn.cursor() as cur:
                        cur.execute(sql, tuple(vals))
                    conn.commit()

                return {"ok": True, "persisted": True}

            # v2: system_settings
            sets = []
            vals = []
            store_anonymized_data = payload.store_anonymized_data

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

            if store_anonymized_data is not None:
                add("store_event_clips", "store_event_clips=%s", 1 if store_anonymized_data else 0)
                add("anonymize_skeleton_data", "anonymize_skeleton_data=%s", 1 if store_anonymized_data else 0)

            if payload.store_event_clips is not None:
                add("store_event_clips", "store_event_clips=%s", 1 if payload.store_event_clips else 0)

            if payload.anonymize_skeleton_data is not None:
                add("anonymize_skeleton_data", "anonymize_skeleton_data=%s", 1 if payload.anonymize_skeleton_data else 0)

            if payload.notify_on_every_fall is not None:
                add("notify_on_every_fall", "notify_on_every_fall=%s", 1 if payload.notify_on_every_fall else 0)
                if payload.notify_on_every_fall is False:
                    add("notify_sms", "notify_sms=%s", 0)
                    add("notify_phone", "notify_phone=%s", 0)

            if payload.notify_sms is not None:
                add("notify_sms", "notify_sms=%s", 1 if payload.notify_sms else 0)

            if payload.notify_phone is not None:
                add("notify_phone", "notify_phone=%s", 1 if payload.notify_phone else 0)

            if payload.active_dataset_code is not None:
                add("active_dataset_code", "active_dataset_code=%s", normalize_dataset_code(payload.active_dataset_code))

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
                    add("active_model_code", "active_model_code=%s", normalize_model_code(payload.active_model_code))
                if _col_exists(conn, "system_settings", "active_model_id") and _table_exists(conn, "models"):
                    norm_mc = normalize_model_code(payload.active_model_code)
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT id FROM models WHERE model_code=%s OR code=%s LIMIT 1",
                            (norm_mc, norm_mc),
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

                sql = "UPDATE system_settings SET " + ", ".join(sets) + f", updated_at={_updated_at_expr(conn)} WHERE resident_id=%s"
                vals.append(rid)
                with conn.cursor() as cur:
                    cur.execute(sql, tuple(vals))
                conn.commit()

            return {"ok": True, "persisted": True}

    except (MySQLError, RuntimeError, OSError, TypeError, ValueError):
        apply_settings_update_inmem(payload, resident_id=rid)
        return {"ok": True, "persisted": False, "reason": "db_unavailable"}
