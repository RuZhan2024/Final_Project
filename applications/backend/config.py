from __future__ import annotations

import os

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .env import load_local_env_files

_DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://fall-detection-frontend.onrender.com",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_repo_root() / path).resolve()


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class AppConfig:
    cors_allowed_origins: tuple[str, ...]
    db_backend: str
    sqlite_path: Path
    event_clips_dir: Path
    notification_sqlite_path: Path


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    load_local_env_files()

    raw_origins = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    origins = tuple(_split_csv(raw_origins)) if raw_origins else _DEFAULT_ALLOWED_ORIGINS

    raw_backend = str(os.getenv("DB_BACKEND", "mysql")).strip().lower()
    db_backend = "sqlite" if raw_backend == "sqlite" else "mysql"

    sqlite_path = _resolve_repo_path(str(os.getenv("SQLITE_PATH", "applications/backend/cloud_demo.sqlite3")).strip())
    event_clips_dir = _resolve_repo_path(str(os.getenv("EVENT_CLIPS_DIR", "applications/backend/event_clips")).strip())
    notification_sqlite_path = _resolve_repo_path(
        str(os.getenv("SAFE_GUARD_SQLITE_PATH", "applications/backend/safe_guard_notifications.sqlite3")).strip()
    )

    return AppConfig(
        cors_allowed_origins=origins,
        db_backend=db_backend,
        sqlite_path=sqlite_path,
        event_clips_dir=event_clips_dir,
        notification_sqlite_path=notification_sqlite_path,
    )
