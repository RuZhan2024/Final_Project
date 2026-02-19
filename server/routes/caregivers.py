from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from ..core import CaregiverUpsertPayload, _ensure_caregivers_table, _jsonable, _table_exists
from ..db import get_conn_optional


router = APIRouter()


@router.get("/api/caregivers")
def get_caregivers(resident_id: int = Query(1, description="Resident ID")) -> Dict[str, Any]:
    with get_conn_optional() as conn:
        if conn is None:
            return {"resident_id": resident_id, "caregivers": [], "db_available": False}
        _ensure_caregivers_table(conn)
        if not _table_exists(conn, "caregivers"):
            return {"resident_id": resident_id, "caregivers": []}
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, resident_id, name, email, phone, created_at, updated_at "
                "FROM caregivers WHERE resident_id=%s ORDER BY id ASC",
                (resident_id,),
            )
            rows = cur.fetchall() or []
        return {"resident_id": resident_id, "caregivers": _jsonable(rows)}


@router.put("/api/caregivers")
@router.post("/api/caregivers")
def upsert_caregiver(payload: CaregiverUpsertPayload = Body(...)) -> Dict[str, Any]:
    resident_id = int(payload.resident_id or 1)
    with get_conn_optional() as conn:
        if conn is None:
            raise HTTPException(status_code=503, detail="DB not available")
        _ensure_caregivers_table(conn)
        if not _table_exists(conn, "caregivers"):
            raise HTTPException(status_code=500, detail="caregivers table not available")

        target_id: Optional[int] = int(payload.id) if payload.id else None
        if target_id is None:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM caregivers WHERE resident_id=%s ORDER BY id ASC LIMIT 1",
                    (resident_id,),
                )
                row = cur.fetchone() or {}
                if isinstance(row, dict) and row.get("id"):
                    target_id = int(row["id"])

        fields: Dict[str, Any] = {}
        if payload.name is not None:
            fields["name"] = payload.name
        if payload.email is not None:
            fields["email"] = payload.email
        if payload.phone is not None:
            fields["phone"] = payload.phone

        with conn.cursor() as cur:
            if target_id is not None:
                if fields:
                    sets = ", ".join([f"`{k}`=%s" for k in fields.keys()])
                    cur.execute(
                        f"UPDATE caregivers SET {sets} WHERE id=%s AND resident_id=%s",
                        (*fields.values(), target_id, resident_id),
                    )
                cur.execute(
                    "SELECT id, resident_id, name, email, phone, created_at, updated_at FROM caregivers WHERE id=%s",
                    (target_id,),
                )
                out = cur.fetchone() or {}
            else:
                cur.execute(
                    "INSERT INTO caregivers (resident_id, name, email, phone) VALUES (%s,%s,%s,%s)",
                    (resident_id, payload.name, payload.email, payload.phone),
                )
                new_id = cur.lastrowid
                cur.execute(
                    "SELECT id, resident_id, name, email, phone, created_at, updated_at FROM caregivers WHERE id=%s",
                    (new_id,),
                )
                out = cur.fetchone() or {}
        conn.commit()
        return {"caregiver": _jsonable(out)}
