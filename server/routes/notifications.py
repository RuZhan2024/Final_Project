from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body


router = APIRouter()


@router.post("/api/notifications/test")
@router.post("/api/v1/notifications/test")
def test_notification(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Minimal endpoint used by the monitor UI fallback test action."""
    return {"ok": True, "accepted": True, "payload": payload if isinstance(payload, dict) else {}}
