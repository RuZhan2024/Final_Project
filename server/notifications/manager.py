from __future__ import annotations

import atexit
import logging
import threading
import time

from typing import Optional

from .classifier import EventClassifier
from .config import NotificationConfig, load_notification_config
from .email_client import EmailClient
from .models import NotificationPreferences, SafeGuardEvent, TierDecision
from .queue_worker import DispatchJob, NotificationQueueWorker
from .sqlite_store import SQLiteNotificationStore
from .templates import build_email_message, build_phone_message, build_sms_message
from .twilio_client import TwilioClient


logger = logging.getLogger(__name__)
_MANAGER_LOCK = threading.Lock()
_MANAGER: Optional["NotificationManager"] = None


class NotificationManager:
    """Main non-blocking interface for Safe Guard event handling."""

    def __init__(self, config: Optional[NotificationConfig] = None):
        self._cfg = config or load_notification_config()
        self._classifier = EventClassifier(self._cfg)
        self._store = SQLiteNotificationStore(self._cfg.sqlite_path)
        self._twilio = TwilioClient(self._cfg)
        self._email = EmailClient(self._cfg)
        self._worker = NotificationQueueWorker(
            maxsize=self._cfg.queue_size,
            poll_interval_s=self._cfg.worker_poll_interval_s,
        )
        atexit.register(self._worker.stop)

    @property
    def store(self) -> SQLiteNotificationStore:
        return self._store

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.safe_guard_enabled)

    def handle_event(self, event: SafeGuardEvent, prefs: NotificationPreferences) -> None:
        """Classify and enqueue notification work without blocking inference."""
        if not self.enabled:
            return

        decision = self._classifier.classify(event, prefs)
        self._store.upsert_event(event, decision, prefs)

        if decision.tier.value == "tier3_silent":
            logger.info("safe_guard silent event event_id=%s", event.event_id)
            return

        if not self._store.should_enqueue(event.event_id, self._cfg.alert_cooldown_seconds, event.resident_id):
            logger.info("safe_guard dedup suppressed event_id=%s", event.event_id)
            return

        accepted = self._worker.submit(
            DispatchJob(
                event_id=event.event_id,
                fn=lambda: self._dispatch(event, decision),
            )
        )
        if not accepted:
            logger.warning("safe_guard enqueue failed event_id=%s", event.event_id)

    def _dispatch(self, event: SafeGuardEvent, decision: TierDecision) -> None:
        results = []
        if decision.actions.get("email", False):
            email_msg = build_email_message(
                event,
                decision,
                caregiver_email=self._cfg.caregiver_email,
                email_from=self._cfg.email_from,
                app_base_url=self._cfg.app_base_url,
            )
            results.append(self._with_retry(lambda: self._email.send(email_msg)))

        if decision.actions.get("sms", False):
            results.append(
                self._with_retry(
                    lambda: self._twilio.sms(
                        to_phone=self._cfg.caregiver_phone,
                        message=build_sms_message(event, decision),
                    )
                )
            )

        if decision.actions.get("phone", False):
            results.append(
                self._with_retry(
                    lambda: self._twilio.call(
                        to_phone=self._cfg.caregiver_phone,
                        message=build_phone_message(event),
                    )
                )
            )

        for result in results:
            self._store.record_delivery(event.event_id, result)

    def _with_retry(self, fn):
        last = None
        for attempt in range(self._cfg.retry_count + 1):
            result = fn()
            last = result
            if result.status == "sent" or result.status.startswith("skipped"):
                return result
            if attempt < self._cfg.retry_count:
                time.sleep(0.3 * (attempt + 1))
        return last


def get_notification_manager() -> NotificationManager:
    global _MANAGER
    if _MANAGER is not None:
        return _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = NotificationManager()
    return _MANAGER
