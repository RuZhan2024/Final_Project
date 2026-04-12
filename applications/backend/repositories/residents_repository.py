from __future__ import annotations

from fastapi import HTTPException


def one_resident_id(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM residents ORDER BY id ASC LIMIT 1")
        row = cur.fetchone()
        if not row:
            raise HTTPException(
                status_code=500,
                detail="No residents found. Did you run create_db.sql + seed data?",
            )
        return int(row["id"])


def resident_exists(conn, resident_id: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 AS ok FROM residents WHERE id=%s LIMIT 1", (resident_id,))
        return cur.fetchone() is not None
