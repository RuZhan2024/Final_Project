from __future__ import annotations

from datetime import datetime, timezone

from applications.backend.notifications.models import SafeGuardEvent, SafeGuardTier, TierDecision
from applications.backend.notifications.templates import build_telegram_message


def test_build_telegram_message_formats_timestamp_in_app_timezone(monkeypatch):
    monkeypatch.setenv("APP_TIMEZONE", "Europe/London")

    event = SafeGuardEvent(
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
        timestamp=datetime(2026, 4, 11, 16, 59, 14, tzinfo=timezone.utc),
    )
    decision = TierDecision(
        tier=SafeGuardTier.TIER2,
        reason="ambiguous_fall",
        actions={"telegram": True},
        recommendation="Check immediately",
    )

    text = build_telegram_message(event, decision, "summary")

    assert "2026-04-11 17:59:14 BST" in text
    assert "Ref: evt-123" in text
