from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Query
try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

from ..core import CaregiverUpsertPayload, _col_exists, _ensure_caregivers_table, _jsonable, _table_exists
from ..core import get_inmem_caregivers, upsert_inmem_caregiver
from ..db import get_conn_optional


router = APIRouter()
logger = logging.getLogger(__name__)


def _caregiver_select_sql(conn) -> str:
    cols = ["id", "resident_id", "name", "email", "phone"]
    if _col_exists(conn, "caregivers", "created_at"):
        cols.append("created_at")
    if _col_exists(conn, "caregivers", "updated_at"):
        cols.append("updated_at")
    return ", ".join(cols)


@router.get("/api/caregivers")
@router.get("/api/v1/caregivers")
def get_caregivers(resident_id: int = Query(1, description="Resident ID")) -> Dict[str, Any]:
    try:
        with get_conn_optional() as conn:
            if conn is None:
                return {"resident_id": resident_id, "caregivers": get_inmem_caregivers(resident_id), "db_available": False}
            _ensure_caregivers_table(conn)
            if not _table_exists(conn, "caregivers"):
                return {"resident_id": resident_id, "caregivers": get_inmem_caregivers(resident_id), "db_available": False}
            select_cols = _caregiver_select_sql(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {select_cols} FROM caregivers WHERE resident_id=%s ORDER BY id ASC",
                    (resident_id,),
                )
                rows = cur.fetchall() or []
            return {"resident_id": resident_id, "caregivers": _jsonable(rows)}
    except Exception as e:
        logger.exception("caregivers.get failed; falling back to in-memory")
        return {
            "resident_id": resident_id,
            "caregivers": get_inmem_caregivers(resident_id),
            "db_available": False,
            "error": str(e),
        }


@router.put("/api/caregivers")
@router.put("/api/v1/caregivers")
@router.post("/api/caregivers")
@router.post("/api/v1/caregivers")
def upsert_caregiver(payload: CaregiverUpsertPayload = Body(...)) -> Dict[str, Any]:
    resident_id = int(payload.resident_id or 1)
    try:
        with get_conn_optional() as conn:
            if conn is None:
                return {"caregiver": _jsonable(upsert_inmem_caregiver(payload)), "db_available": False}
            _ensure_caregivers_table(conn)
            if not _table_exists(conn, "caregivers"):
                return {"caregiver": _jsonable(upsert_inmem_caregiver(payload)), "db_available": False}
            select_cols = _caregiver_select_sql(conn)

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
                        f"SELECT {select_cols} FROM caregivers WHERE id=%s",
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
                        f"SELECT {select_cols} FROM caregivers WHERE id=%s",
                        (new_id,),
                    )
                    out = cur.fetchone() or {}
            conn.commit()
            return {"caregiver": _jsonable(out)}
    except Exception as e:
        logger.exception("caregivers.upsert failed; falling back to in-memory")
        return {
            "caregiver": _jsonable(upsert_inmem_caregiver(payload)),
            "db_available": False,
            "error": str(e),
        }
