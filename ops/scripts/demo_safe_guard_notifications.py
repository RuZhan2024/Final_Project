#!/usr/bin/env python3
"""Minimal Safe Guard demo for Tier 1 / Tier 2 / Tier 3 event handling."""

from __future__ import annotations

import time

from dataclasses import replace
from datetime import datetime, timezone

from applications.backend.notifications.classifier import EventClassifier
from applications.backend.notifications.config import NotificationConfig
from applications.backend.notifications.manager import NotificationManager
from applications.backend.notifications.models import NotificationPreferences, SafeGuardEvent


def build_demo_config(sqlite_path: str) -> NotificationConfig:
    return NotificationConfig(
        safe_guard_enabled=True,
        sqlite_path=sqlite_path,
        queue_size=32,
        worker_poll_interval_s=0.1,
        retry_count=0,
        http_timeout_s=3.0,
        high_conf_margin=0.08,
        low_uncertainty_threshold=0.05,
        high_uncertainty_threshold=0.15,
        alert_cooldown_seconds=0,
        twilio_account_sid="",
        twilio_auth_token="",
        twilio_from_phone="",
        caregiver_phone="",
        smtp_host="",
        smtp_port=587,
        smtp_username="",
        smtp_password="",
        email_from="",
        caregiver_email="",
        app_base_url="http://127.0.0.1:3000",
    )


def build_base_event() -> SafeGuardEvent:
    return SafeGuardEvent(
        event_id="demo-tier1",
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
        source="demo",
    )


def main() -> None:
    cfg = build_demo_config("applications/backend/safe_guard_demo.sqlite3")
    manager = NotificationManager(cfg)
    classifier = EventClassifier(cfg)
    prefs = NotificationPreferences(phone_enabled=True, sms_enabled=True, email_enabled=True)

    tier1 = build_base_event()
    tier2 = replace(
        build_base_event(),
        event_id="demo-tier2",
        probability=0.82,
        uncertainty=0.18,
        margin=0.02,
    )
    tier3 = replace(
        build_base_event(),
        event_id="demo-tier3",
        triage_state="not_fall",
        safe_alert=False,
        recall_alert=False,
        probability=0.41,
        uncertainty=0.02,
        margin=-0.39,
    )

    events = [tier1, tier2, tier3]
    for ev in events:
        decision = classifier.classify(ev, prefs)
        print(f"event_id={ev.event_id} tier={decision.tier.value} actions={decision.actions} reason={decision.reason}")
        manager.handle_event(ev, prefs)

    # Give the background worker time to write audit entries.
    time.sleep(0.6)
    print(f"demo sqlite written to: {cfg.sqlite_path}")


if __name__ == "__main__":
    main()
