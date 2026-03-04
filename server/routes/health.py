from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter


router = APIRouter()


@router.get("/api/health")
@router.get("/api/v1/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ts": datetime.utcnow().isoformat()}
