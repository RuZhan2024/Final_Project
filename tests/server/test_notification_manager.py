from __future__ import annotations

from datetime import datetime, timezone
from email.message import EmailMessage

from server.notifications.config import NotificationConfig
from server.notifications.manager import NotificationManager
from server.notifications.models import DeliveryResult, NotificationPreferences, SafeGuardEvent


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
        twilio_account_sid="sid",
        twilio_auth_token="token",
        twilio_from_phone="+10000000000",
        caregiver_phone="+19999999999",
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_username="",
        smtp_password="",
        email_from="noreply@example.com",
        caregiver_email="default@example.com",
        app_base_url="http://localhost:3000",
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


def test_notification_manager_dispatch_prefers_caregiver_contacts(monkeypatch):
    manager = NotificationManager(config=_cfg())
    sent = {"email_to": None, "sms_to": None, "phone_to": None}

    def _fake_email_send(msg: EmailMessage):
        sent["email_to"] = msg["To"]
        return DeliveryResult(channel="email", attempted=True, status="sent", detail="")

    def _fake_sms(*, to_phone: str, message: str):
        sent["sms_to"] = to_phone
        return DeliveryResult(channel="sms", attempted=True, status="sent", detail=message)

    def _fake_call(*, to_phone: str, message: str):
        sent["phone_to"] = to_phone
        return DeliveryResult(channel="phone", attempted=True, status="sent", detail=message)

    monkeypatch.setattr(manager._email, "send", _fake_email_send)
    monkeypatch.setattr(manager._twilio, "sms", _fake_sms)
    monkeypatch.setattr(manager._twilio, "call", _fake_call)

    decision = manager._classifier.classify(
        _event(),
        NotificationPreferences(
            phone_enabled=True,
            sms_enabled=True,
            email_enabled=True,
            caregiver_name="Alice",
            caregiver_phone="+447700900123",
            caregiver_email="alice@example.com",
        ),
    )

    manager._dispatch(
        _event(),
        decision,
        NotificationPreferences(
            phone_enabled=True,
            sms_enabled=True,
            email_enabled=True,
            caregiver_name="Alice",
            caregiver_phone="+447700900123",
            caregiver_email="alice@example.com",
        ),
    )

    assert sent["email_to"] == "alice@example.com"
    assert sent["sms_to"] == "+447700900123"
    assert sent["phone_to"] == "+447700900123"
