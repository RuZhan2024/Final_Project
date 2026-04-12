from __future__ import annotations

from datetime import datetime, timezone

from applications.backend.notifications.classifier import EventClassifier
from applications.backend.notifications.config import NotificationConfig
from applications.backend.notifications.models import NotificationPreferences, SafeGuardEvent, SafeGuardTier


def _cfg() -> NotificationConfig:
    return NotificationConfig(
        safe_guard_enabled=True,
        sqlite_path="/tmp/safe_guard_test.sqlite3",
        queue_size=32,
        worker_poll_interval_s=0.1,
        retry_count=0,
        http_timeout_s=2.0,
        high_conf_margin=0.08,
        low_uncertainty_threshold=0.05,
        high_uncertainty_threshold=0.15,
        alert_cooldown_seconds=30,
        telegram_bot_token="",
        telegram_chat_id="",
        telegram_api_base="https://api.telegram.org",
        twilio_account_sid="",
        twilio_auth_token="",
        twilio_from_phone="",
        caregiver_phone="",
        resend_api_key="",
        email_from="",
        caregiver_email="",
        app_base_url="http://localhost:3000",
        ai_provider="openai",
        openai_api_key="",
        openai_model="gpt-4.1-mini",
        gemini_api_key="",
        gemini_model="gemini-2.0-flash",
        openai_timeout_s=12.0,
        ai_reports_enabled=True,
    )


def _event(**overrides) -> SafeGuardEvent:
    base = dict(
        event_id="evt-1",
        resident_id=1,
        location="living_room",
        probability=0.93,
        uncertainty=0.03,
        threshold=0.80,
        margin=0.13,
        triage_state="fall",
        safe_alert=True,
        recall_alert=True,
        model_code="TCN",
        dataset_code="caucafall",
        op_code="OP-2",
        timestamp=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return SafeGuardEvent(**base)


def test_classifier_tier1_high_confidence():
    decision = EventClassifier(_cfg()).classify(
        _event(),
        NotificationPreferences(telegram_enabled=True),
    )
    assert decision.tier == SafeGuardTier.TIER1
    assert decision.actions["telegram"] is True


def test_classifier_tier2_borderline():
    decision = EventClassifier(_cfg()).classify(
        _event(probability=0.82, margin=0.02, uncertainty=0.18),
        NotificationPreferences(telegram_enabled=True),
    )
    assert decision.tier == SafeGuardTier.TIER2
    assert decision.actions["telegram"] is True


def test_classifier_tier3_non_alert():
    decision = EventClassifier(_cfg()).classify(
        _event(triage_state="not_fall", safe_alert=False, recall_alert=False),
        NotificationPreferences(telegram_enabled=True),
    )
    assert decision.tier == SafeGuardTier.TIER3
    assert decision.actions["telegram"] is False
