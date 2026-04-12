from __future__ import annotations

from typing import Any, Dict, Optional


def caregiver_select_sql(conn: Any, *, col_exists) -> str:
    cols = ["id", "resident_id", "name", "email", "phone"]
    if col_exists(conn, "caregivers", "telegram_chat_id"):
        cols.append("telegram_chat_id")
    if col_exists(conn, "caregivers", "created_at"):
        cols.append("created_at")
    if col_exists(conn, "caregivers", "updated_at"):
        cols.append("updated_at")
    return ", ".join(cols)


def fetch_caregivers(conn: Any, *, resident_id: int, select_cols: str):
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {select_cols} FROM caregivers WHERE resident_id=%s ORDER BY id ASC",
            (resident_id,),
        )
        return cur.fetchall() or []


def resolve_target_caregiver_id(conn: Any, *, resident_id: int, payload_id: Optional[int]) -> Optional[int]:
    if payload_id is not None:
        return int(payload_id)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM caregivers WHERE resident_id=%s ORDER BY id ASC LIMIT 1",
            (resident_id,),
        )
        row = cur.fetchone() or {}
    if isinstance(row, dict) and row.get("id"):
        return int(row["id"])
    return None


def upsert_caregiver_row(
    conn: Any,
    *,
    resident_id: int,
    target_id: Optional[int],
    fields: Dict[str, Any],
    select_cols: str,
    telegram_supported: bool,
) -> Dict[str, Any]:
    with conn.cursor() as cur:
        if target_id is not None:
            if fields:
                sets = ", ".join([f"`{k}`=%s" for k in fields.keys()])
                cur.execute(
                    f"UPDATE caregivers SET {sets} WHERE id=%s AND resident_id=%s",
                    (*fields.values(), target_id, resident_id),
                )
            cur.execute(f"SELECT {select_cols} FROM caregivers WHERE id=%s", (target_id,))
            out = cur.fetchone() or {}
        else:
            cur.execute(
                "INSERT INTO caregivers (resident_id, name, email, phone, telegram_chat_id) VALUES (%s,%s,%s,%s,%s)",
                (
                    resident_id,
                    fields.get("name"),
                    fields.get("email"),
                    fields.get("phone"),
                    fields.get("telegram_chat_id") if telegram_supported else None,
                ),
            )
            new_id = cur.lastrowid
            cur.execute(f"SELECT {select_cols} FROM caregivers WHERE id=%s", (new_id,))
            out = cur.fetchone() or {}
    conn.commit()
    return out
