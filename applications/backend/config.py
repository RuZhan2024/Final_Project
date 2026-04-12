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
    return Path(__file__).resolve().parent.parent.parent


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_repo_root() / path).resolve()


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def get_env_str(name: str, default: str = "") -> str:
    load_local_env_files()
    raw = os.getenv(name)
    return str(raw if raw is not None else default).strip()


def get_env_bool(name: str, default: bool) -> bool:
    raw = get_env_str(name, "")
    if not raw:
        return default
    val = raw.lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return default


def get_env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    try:
        value = int(get_env_str(name, str(default)))
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None:
        value = max(minimum, value)
    return value


def get_env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    try:
        value = float(get_env_str(name, str(default)))
    except (TypeError, ValueError):
        value = float(default)
    if minimum is not None:
        value = max(minimum, value)
    return value


@dataclass(frozen=True)
class AppConfig:
    cors_allowed_origins: tuple[str, ...]
    db_backend: str
    sqlite_path: Path
    event_clips_dir: Path
    notification_sqlite_path: Path
    app_base_url: str
    app_timezone: str
    session_ttl_s: int
    session_max_states: int


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    load_local_env_files()

    raw_origins = get_env_str("CORS_ALLOWED_ORIGINS", "")
    origins = tuple(_split_csv(raw_origins)) if raw_origins else _DEFAULT_ALLOWED_ORIGINS

    raw_backend = get_env_str("DB_BACKEND", "mysql").lower()
    db_backend = "sqlite" if raw_backend == "sqlite" else "mysql"

    sqlite_path = _resolve_repo_path(get_env_str("SQLITE_PATH", "applications/backend/cloud_demo.sqlite3"))
    event_clips_dir = _resolve_repo_path(get_env_str("EVENT_CLIPS_DIR", "applications/backend/event_clips"))
    notification_sqlite_path = _resolve_repo_path(
        get_env_str("SAFE_GUARD_SQLITE_PATH", "applications/backend/safe_guard_notifications.sqlite3")
    )
    app_base_url = get_env_str("APP_BASE_URL", "http://127.0.0.1:3000").rstrip("/")
    app_timezone = get_env_str("APP_TIMEZONE", "Europe/London") or "Europe/London"
    session_ttl_s = get_env_int("SESSION_TTL_S", 1800, minimum=60)
    session_max_states = get_env_int("SESSION_MAX_STATES", 1000, minimum=10)

    return AppConfig(
        cors_allowed_origins=origins,
        db_backend=db_backend,
        sqlite_path=sqlite_path,
        event_clips_dir=event_clips_dir,
        notification_sqlite_path=notification_sqlite_path,
        app_base_url=app_base_url,
        app_timezone=app_timezone,
        session_ttl_s=session_ttl_s,
        session_max_states=session_max_states,
    )
