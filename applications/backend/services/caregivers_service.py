from __future__ import annotations

"""Caregiver settings service helpers.

These helpers sit between route handlers and caregiver repositories. They keep
schema-sensitive column selection and optional Telegram support in one place so
the route contract can remain stable across DB variants.
"""

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
    """Return the caregiver list using the schema-compatible select projection.

    The service resolves the select column set first so older schemas can omit
    newer fields without forcing the route layer to special-case them.
    """
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
    """Upsert one caregiver while respecting optional schema features.

    Telegram fields are included only when the active schema supports them so a
    newer frontend payload can still round-trip against older databases.
    """
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
    # Gate Telegram writes here so repository code can stay focused on SQL shape
    # instead of re-deciding feature support for every caller.
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
