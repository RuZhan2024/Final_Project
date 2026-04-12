from __future__ import annotations

from datetime import datetime, timezone
from applications.backend.notifications.config import NotificationConfig
from applications.backend.notifications.manager import NotificationManager
from applications.backend.notifications.models import DeliveryResult, NotificationPreferences, SafeGuardEvent


def _cfg() -> NotificationConfig:
    return NotificationConfig(
        safe_guard_enabled=True,
        sqlite_path="/tmp/safe_guard_manager_test.sqlite3",
        queue_size=8,
        worker_poll_interval_s=0.1,
        retry_count=0,
        http_timeout_s=2.0,
        high_conf_margin=0.08,
        low_uncertainty_threshold=0.05,
        high_uncertainty_threshold=0.15,
        alert_cooldown_seconds=30,
        telegram_bot_token="bot-token",
        telegram_chat_id="12345",
        telegram_api_base="https://api.telegram.org",
        twilio_account_sid="sid",
        twilio_auth_token="token",
        twilio_from_phone="+10000000000",
        caregiver_phone="+19999999999",
        resend_api_key="re_test_key",
        email_from="noreply@example.com",
        caregiver_email="default@example.com",
        app_base_url="http://localhost:3000",
        ai_provider="openai",
        openai_api_key="",
        openai_model="gpt-4.1-mini",
        gemini_api_key="",
        gemini_model="gemini-2.0-flash",
        openai_timeout_s=12.0,
        ai_reports_enabled=True,
    )


def _event() -> SafeGuardEvent:
    return SafeGuardEvent(
        event_id="evt-123",
        resident_id=1,
        location="living_room",
        probability=0.92,
        uncertainty=0.02,
        threshold=0.71,
        margin=0.21,
        triage_state="fall",
        safe_alert=True,
        recall_alert=True,
        model_code="TCN",
        dataset_code="caucafall",
        op_code="OP-2",
        timestamp=datetime.now(timezone.utc),
    )


def test_notification_manager_dispatch_prefers_caregiver_chat_id(monkeypatch):
    manager = NotificationManager(config=_cfg())
    sent = {"chat_id": None, "text": None}

    def _fake_send_message(*, chat_id: str, text: str):
        sent["chat_id"] = chat_id
        sent["text"] = text
        return DeliveryResult(channel="telegram", attempted=True, status="sent", detail="")

    monkeypatch.setattr(manager._telegram, "send_message", _fake_send_message)

    decision = manager._classifier.classify(
        _event(),
        NotificationPreferences(
            telegram_enabled=True,
            caregiver_name="Alice",
            caregiver_telegram_chat_id="987654321",
        ),
    )

    manager._dispatch(
        _event(),
        decision,
        NotificationPreferences(
            telegram_enabled=True,
            caregiver_name="Alice",
            caregiver_telegram_chat_id="987654321",
        ),
    )

    assert sent["chat_id"] == "987654321"
    assert "Safe Guard alert" in (sent["text"] or "")
    assert "Ref: evt-123" in (sent["text"] or "")


def test_notification_manager_handle_event_returns_dispatch_acceptance(monkeypatch):
    manager = NotificationManager(config=_cfg())
    monkeypatch.setattr(manager._worker, "submit", lambda job: True)
    monkeypatch.setattr(manager._store, "should_enqueue", lambda *args, **kwargs: True)

    dispatch = manager.handle_event(
        _event(),
        NotificationPreferences(
            telegram_enabled=True,
            caregiver_name="Alice",
            caregiver_telegram_chat_id="987654321",
        ),
    )

    assert dispatch.enabled is True
    assert dispatch.tier.startswith("tier")
    assert dispatch.actions.get("telegram") is True
    assert dispatch.enqueued is True
    assert dispatch.state == "enqueued"
    assert dispatch.audit_backend == "sqlite"
