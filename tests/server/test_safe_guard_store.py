from __future__ import annotations

from datetime import datetime, timezone

from applications.backend.notifications.models import DeliveryResult, NotificationPreferences, SafeGuardEvent, SafeGuardTier, TierDecision
from applications.backend.notifications.sqlite_store import SQLiteNotificationStore


def _event(event_id: str) -> SafeGuardEvent:
    return SafeGuardEvent(
        event_id=event_id,
        resident_id=1,
        location="hallway",
        probability=0.86,
        uncertainty=0.04,
        threshold=0.80,
        margin=0.06,
        triage_state="fall",
        safe_alert=True,
        recall_alert=True,
        model_code="GCN",
        dataset_code="caucafall",
        op_code="OP-2",
        timestamp=datetime.now(timezone.utc),
    )


def test_sqlite_store_event_roundtrip_and_feedback(tmp_path):
    store = SQLiteNotificationStore(str(tmp_path / "safe_guard.sqlite3"))
    event = _event("evt-123")
    decision = TierDecision(
        tier=SafeGuardTier.TIER2,
        reason="borderline_margin_or_policy_promoted",
        actions={"telegram": True},
        recommendation="Review the live stream or recent clip.",
    )
    store.upsert_event(event, decision, NotificationPreferences(telegram_enabled=True))
    assert store.should_enqueue("evt-123", cooldown_seconds=30, resident_id=1) is True
    store.record_delivery("evt-123", DeliveryResult(channel="telegram", attempted=True, status="sent", detail=""))
    assert store.should_enqueue("evt-123", cooldown_seconds=30, resident_id=1) is False
    store.mark_feedback("evt-123", resident_id=1, reply_code="2", mapped_status="confirmed_fall")
    assert store.get_most_recent_unresolved_event_id(1) == "evt-123"
    store.resolve_event("evt-123")
    assert store.get_most_recent_unresolved_event_id(1) is None
