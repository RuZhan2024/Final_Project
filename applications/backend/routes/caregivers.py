from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, Query
try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

from ..db import get_conn_optional
from ..db_schema import col_exists, ensure_caregivers_table, table_exists
from ..inmemory_state import get_inmem_caregivers, upsert_inmem_caregiver
from ..json_utils import jsonable as _jsonable
from ..schemas import CaregiverUpsertPayload
from ..services.caregivers_service import build_caregivers_list_response, build_upsert_caregiver_response


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/api/caregivers")
@router.get("/api/v1/caregivers")
def get_caregivers(resident_id: int = Query(1, description="Resident ID")) -> Dict[str, Any]:
    try:
        with get_conn_optional() as conn:
            if conn is None:
                return {"resident_id": resident_id, "caregivers": get_inmem_caregivers(resident_id), "db_available": False}
            ensure_caregivers_table(conn)
            if not table_exists(conn, "caregivers"):
                return {"resident_id": resident_id, "caregivers": get_inmem_caregivers(resident_id), "db_available": False}
            return build_caregivers_list_response(
                conn,
                resident_id=resident_id,
                col_exists=col_exists,
                jsonable=_jsonable,
            )
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
    try:
        with get_conn_optional() as conn:
            if conn is None:
                return {"caregiver": _jsonable(upsert_inmem_caregiver(payload)), "db_available": False}
            ensure_caregivers_table(conn)
            if not table_exists(conn, "caregivers"):
                return {"caregiver": _jsonable(upsert_inmem_caregiver(payload)), "db_available": False}
            return build_upsert_caregiver_response(
                conn,
                payload=payload,
                col_exists=col_exists,
                jsonable=_jsonable,
            )
    except Exception as e:
        logger.exception("caregivers.upsert failed; falling back to in-memory")
        return {
            "caregiver": _jsonable(upsert_inmem_caregiver(payload)),
            "db_available": False,
            "error": str(e),
        }
