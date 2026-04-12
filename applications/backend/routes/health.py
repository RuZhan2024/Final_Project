from __future__ import annotations

import os

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Request


router = APIRouter()


@router.get("/api/health")
@router.get("/api/v1/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


def _mysql_env_ready() -> bool:
    required = ("DB_HOST", "DB_PORT", "DB_USER", "DB_PASS", "DB_NAME")
    return all(str(os.getenv(name, "")).strip() for name in required)


def _sqlite_ready(sqlite_path: Path) -> bool:
    parent = sqlite_path.parent
    return parent.exists() and parent.is_dir()


@router.get("/api/ready")
@router.get("/api/v1/ready")
def ready(request: Request) -> Dict[str, Any]:
    cfg = request.app.state.app_config
    if cfg.db_backend == "sqlite":
        ready_ok = _sqlite_ready(cfg.sqlite_path)
        detail = f"sqlite_path={cfg.sqlite_path}"
    else:
        ready_ok = _mysql_env_ready()
        detail = "mysql_env=present" if ready_ok else "mysql_env=missing"

    return {
        "ok": ready_ok,
        "db_backend": cfg.db_backend,
        "detail": detail,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
