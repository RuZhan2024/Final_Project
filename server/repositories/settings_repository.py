from __future__ import annotations

from typing import Any, Dict, List

from ..core import (
    SettingsUpdatePayload,
    _norm_op_code,
    _col_exists,
    _ensure_system_settings_schema,
    _safe_get,
    _table_exists,
    normalize_dataset_code,
    normalize_model_code,
)


def _updated_at_expr(conn: Any) -> str:
    return "CURRENT_TIMESTAMP" if str(getattr(conn, "db_backend", "mysql")).lower() == "sqlite" else "NOW()"


def load_settings_snapshot(conn: Any, resident_id: int, system: Dict[str, Any], deploy: Dict[str, Any]) -> None:
    _ensure_system_settings_schema(conn)

    if not _table_exists(conn, "system_settings"):
        return

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
        ("active_op_code", lambda v: system.__setitem__("active_op_code", _norm_op_code(v))),
        ("mc_enabled", lambda v: system.__setitem__("mc_enabled", bool(int(v)) if str(v).isdigit() else bool(v))),
    ]:
        if isinstance(sys_row, dict) and col in sys_row and sys_row.get(col) is not None:
            try:
                setter(sys_row[col])
            except (TypeError, ValueError):
                pass

    system["active_operating_point"] = (
        sys_row.get("active_operating_point_id")
        or sys_row.get("active_operating_point")
        or system.get("active_operating_point")
    )


def persist_settings_update(conn: Any, resident_id: int, payload: SettingsUpdatePayload) -> bool:
    _ensure_system_settings_schema(conn)

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
        add("active_op_code", "active_op_code=%s", _norm_op_code(payload.active_op_code))

    if payload.mc_enabled is not None:
        add("mc_enabled", "mc_enabled=%s", 1 if payload.mc_enabled else 0)

    if payload.mc_M is not None:
        add("mc_M", "mc_M=%s", int(payload.mc_M))

    if payload.mc_M_confirm is not None:
        add("mc_M_confirm", "mc_M_confirm=%s", int(payload.mc_M_confirm))

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
                model_row = cur.fetchone()
            if model_row and isinstance(model_row, dict) and model_row.get("id") is not None:
                sets.append("active_model_id=%s")
                vals.append(int(model_row["id"]))

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
                (resident_id,),
            )
            row = cur.fetchone()
            if not row:
                cur.execute("INSERT INTO system_settings (resident_id) VALUES (%s)", (resident_id,))
                conn.commit()

        sql = (
            "UPDATE system_settings SET "
            + ", ".join(sets)
            + f", updated_at={_updated_at_expr(conn)} WHERE resident_id=%s"
        )
        vals.append(resident_id)
        with conn.cursor() as cur:
            cur.execute(sql, tuple(vals))
        conn.commit()

    return True
