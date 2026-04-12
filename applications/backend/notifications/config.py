from __future__ import annotations

import os

from dataclasses import dataclass

from ..config import get_app_config, get_env_bool, get_env_float, get_env_int, get_env_str


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
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_api_base: str
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
    app_config = get_app_config()
    return NotificationConfig(
        safe_guard_enabled=get_env_bool("SAFE_GUARD_ENABLED", False),
        sqlite_path=str(app_config.notification_sqlite_path),
        queue_size=get_env_int("SAFE_GUARD_WORKER_QUEUE_SIZE", 256, minimum=10),
        worker_poll_interval_s=get_env_float("SAFE_GUARD_WORKER_POLL_INTERVAL_S", 0.25, minimum=0.05),
        retry_count=get_env_int("SAFE_GUARD_RETRY_COUNT", 2, minimum=0),
        http_timeout_s=get_env_float("SAFE_GUARD_HTTP_TIMEOUT_S", 8.0, minimum=1.0),
        high_conf_margin=get_env_float("HIGH_CONF_MARGIN", 0.08),
        low_uncertainty_threshold=get_env_float("LOW_UNCERTAINTY_THRESHOLD", 0.05),
        high_uncertainty_threshold=get_env_float("HIGH_UNCERTAINTY_THRESHOLD", 0.15),
        alert_cooldown_seconds=get_env_int("ALERT_COOLDOWN_SECONDS", 60, minimum=0),
        telegram_bot_token=get_env_str("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=get_env_str("CAREGIVER_TELEGRAM_CHAT_ID", ""),
        telegram_api_base=get_env_str("TELEGRAM_API_BASE", "https://api.telegram.org").rstrip("/"),
        twilio_account_sid=get_env_str("TWILIO_ACCOUNT_SID", ""),
        twilio_auth_token=get_env_str("TWILIO_AUTH_TOKEN", ""),
        twilio_from_phone=get_env_str("TWILIO_FROM_PHONE", ""),
        caregiver_phone=get_env_str("CAREGIVER_PHONE", ""),
        resend_api_key=get_env_str("RESEND_API_KEY", ""),
        email_from=get_env_str("EMAIL_FROM", ""),
        caregiver_email=get_env_str("CAREGIVER_EMAIL", ""),
        app_base_url=app_config.app_base_url,
        ai_provider=get_env_str("AI_PROVIDER", "openai").lower(),
        openai_api_key=get_env_str("OPENAI_API_KEY", ""),
        openai_model=get_env_str("OPENAI_MODEL", "gpt-4.1-mini"),
        gemini_api_key=get_env_str("GEMINI_API_KEY", ""),
        gemini_model=get_env_str("GEMINI_MODEL", "gemini-2.0-flash"),
        openai_timeout_s=get_env_float("OPENAI_TIMEOUT_S", 12.0, minimum=3.0),
        ai_reports_enabled=get_env_bool("AI_REPORTS_ENABLED", True),
    )
