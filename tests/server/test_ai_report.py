from __future__ import annotations

from datetime import datetime, timezone
import io
import json

import pytest

from server.notifications import ai_report
from server.notifications.ai_report import generate_event_ai_report
from server.notifications.config import NotificationConfig
from server.notifications.models import SafeGuardEvent, SafeGuardTier, TierDecision


def _cfg(**overrides) -> NotificationConfig:
    base = dict(
        safe_guard_enabled=True,
        sqlite_path="/tmp/safe_guard_ai_test.sqlite3",
        queue_size=8,
        worker_poll_interval_s=0.1,
        retry_count=0,
        http_timeout_s=2.0,
        high_conf_margin=0.08,
        low_uncertainty_threshold=0.05,
        high_uncertainty_threshold=0.15,
        alert_cooldown_seconds=30,
        twilio_account_sid="",
        twilio_auth_token="",
        twilio_from_phone="",
        caregiver_phone="",
        resend_api_key="re_test_key",
        email_from="noreply@example.com",
        caregiver_email="care@example.com",
        app_base_url="http://localhost:3000",
        ai_provider="openai",
        openai_api_key="",
        openai_model="gpt-4.1-mini",
        gemini_api_key="",
        gemini_model="gemini-2.0-flash",
        openai_timeout_s=12.0,
        ai_reports_enabled=True,
    )
    base.update(overrides)
    return NotificationConfig(**base)


def _event() -> SafeGuardEvent:
    return SafeGuardEvent(
        event_id="evt-ai-1",
        resident_id=1,
        location="bedroom",
        probability=0.93,
        uncertainty=0.04,
        threshold=0.71,
        margin=0.22,
        triage_state="fall",
        safe_alert=True,
        recall_alert=True,
        model_code="TCN",
        dataset_code="caucafall",
        op_code="OP-2",
        timestamp=datetime.now(timezone.utc),
    )


def test_ai_report_falls_back_without_openai_key():
    decision = TierDecision(
        tier=SafeGuardTier.TIER1,
        reason="strong_margin_low_uncertainty",
        actions={"email": True, "sms": True, "phone": True},
        recommendation="Check the resident immediately.",
    )
    out = generate_event_ai_report(_event(), decision, _cfg(openai_api_key=""))
    assert "fall-like event" in out.lower()
    assert "recommended action" in out.lower()


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def test_ai_report_uses_openai_provider(monkeypatch):
    captured = {}

    def _fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        captured["timeout"] = timeout
        body = json.loads(req.data.decode("utf-8"))
        captured["model"] = body["model"]
        payload = {"output_text": "What happened\nTest output"}
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(ai_report.urllib.request, "urlopen", _fake_urlopen)

    decision = TierDecision(
        tier=SafeGuardTier.TIER1,
        reason="strong_margin_low_uncertainty",
        actions={"email": True, "sms": False, "phone": False},
        recommendation="Check the resident immediately.",
    )

    out = generate_event_ai_report(_event(), decision, _cfg(openai_api_key="sk-test"))
    assert out == "What happened\nTest output"
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["model"] == "gpt-4.1-mini"
    assert captured["timeout"] == 12.0


def test_ai_report_uses_gemini_provider(monkeypatch):
    captured = {}

    def _fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        body = json.loads(req.data.decode("utf-8"))
        captured["text"] = body["contents"][0]["parts"][0]["text"]
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "What happened\nGemini output"
                            }
                        ]
                    }
                }
            ]
        }
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(ai_report.urllib.request, "urlopen", _fake_urlopen)

    decision = TierDecision(
        tier=SafeGuardTier.TIER1,
        reason="strong_margin_low_uncertainty",
        actions={"email": True, "sms": False, "phone": False},
        recommendation="Check the resident immediately.",
    )

    out = generate_event_ai_report(
        _event(),
        decision,
        _cfg(ai_provider="gemini", gemini_api_key="gem-key", openai_api_key=""),
    )
    assert out == "What happened\nGemini output"
    assert captured["url"].startswith("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=gem-key")
    assert "event_id: evt-ai-1" in captured["text"]
    assert captured["timeout"] == 12.0
