from __future__ import annotations

from typing import Any, Dict

from ..repositories.caregivers_repository import (
    caregiver_select_sql,
    fetch_caregivers,
    resolve_target_caregiver_id,
    upsert_caregiver_row,
)


def build_caregivers_list_response(
    conn: Any,
    *,
    resident_id: int,
    col_exists,
    jsonable,
) -> Dict[str, Any]:
    select_cols = caregiver_select_sql(conn, col_exists=col_exists)
    rows = fetch_caregivers(conn, resident_id=resident_id, select_cols=select_cols)
    return {"resident_id": resident_id, "caregivers": jsonable(rows)}


def build_upsert_caregiver_response(
    conn: Any,
    *,
    payload,
    col_exists,
    jsonable,
) -> Dict[str, Any]:
    resident_id = int(payload.resident_id or 1)
    telegram_supported = bool(col_exists(conn, "caregivers", "telegram_chat_id"))
    select_cols = caregiver_select_sql(conn, col_exists=col_exists)
    target_id = resolve_target_caregiver_id(
        conn,
        resident_id=resident_id,
        payload_id=int(payload.id) if payload.id else None,
    )

    fields: Dict[str, Any] = {}
    if payload.name is not None:
        fields["name"] = payload.name
    if payload.email is not None:
        fields["email"] = payload.email
    if payload.phone is not None:
        fields["phone"] = payload.phone
    if payload.telegram_chat_id is not None and telegram_supported:
        fields["telegram_chat_id"] = payload.telegram_chat_id

    out = upsert_caregiver_row(
        conn,
        resident_id=resident_id,
        target_id=target_id,
        fields=fields,
        select_cols=select_cols,
        telegram_supported=telegram_supported,
    )
    return {"caregiver": jsonable(out)}
