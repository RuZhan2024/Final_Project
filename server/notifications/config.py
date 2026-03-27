from __future__ import annotations

import os

from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


@dataclass(frozen=True)
class NotificationConfig:
    safe_guard_enabled: bool
    sqlite_path: str
    queue_size: int
    worker_poll_interval_s: float
    retry_count: int
    http_timeout_s: float
    high_conf_margin: float
    low_uncertainty_threshold: float
    high_uncertainty_threshold: float
    alert_cooldown_seconds: int
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_from_phone: str
    caregiver_phone: str
    resend_api_key: str
    email_from: str
    caregiver_email: str
    app_base_url: str
    ai_provider: str
    openai_api_key: str
    openai_model: str
    gemini_api_key: str
    gemini_model: str
    openai_timeout_s: float
    ai_reports_enabled: bool


def load_notification_config() -> NotificationConfig:
    return NotificationConfig(
        safe_guard_enabled=_get_bool("SAFE_GUARD_ENABLED", False),
        sqlite_path=os.getenv("SAFE_GUARD_SQLITE_PATH", "server/safe_guard_notifications.sqlite3"),
        queue_size=max(10, _get_int("SAFE_GUARD_WORKER_QUEUE_SIZE", 256)),
        worker_poll_interval_s=max(0.05, _get_float("SAFE_GUARD_WORKER_POLL_INTERVAL_S", 0.25)),
        retry_count=max(0, _get_int("SAFE_GUARD_RETRY_COUNT", 2)),
        http_timeout_s=max(1.0, _get_float("SAFE_GUARD_HTTP_TIMEOUT_S", 8.0)),
        high_conf_margin=_get_float("HIGH_CONF_MARGIN", 0.08),
        low_uncertainty_threshold=_get_float("LOW_UNCERTAINTY_THRESHOLD", 0.05),
        high_uncertainty_threshold=_get_float("HIGH_UNCERTAINTY_THRESHOLD", 0.15),
        alert_cooldown_seconds=max(0, _get_int("ALERT_COOLDOWN_SECONDS", 60)),
        twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", "").strip(),
        twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", "").strip(),
        twilio_from_phone=os.getenv("TWILIO_FROM_PHONE", "").strip(),
        caregiver_phone=os.getenv("CAREGIVER_PHONE", "").strip(),
        resend_api_key=os.getenv("RESEND_API_KEY", "").strip(),
        email_from=os.getenv("EMAIL_FROM", "").strip(),
        caregiver_email=os.getenv("CAREGIVER_EMAIL", "").strip(),
        app_base_url=os.getenv("APP_BASE_URL", "http://127.0.0.1:3000").rstrip("/"),
        ai_provider=os.getenv("AI_PROVIDER", "openai").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip(),
        openai_timeout_s=max(3.0, _get_float("OPENAI_TIMEOUT_S", 12.0)),
        ai_reports_enabled=_get_bool("AI_REPORTS_ENABLED", True),
    )
