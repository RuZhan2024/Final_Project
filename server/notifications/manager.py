from __future__ import annotations

import atexit
import logging
import threading
import time

from typing import Optional

from .classifier import EventClassifier
from .ai_report import generate_event_ai_report
from .config import NotificationConfig, load_notification_config
from .models import DispatchAcceptance, NotificationPreferences, SafeGuardEvent, TierDecision
from .queue_worker import DispatchJob, NotificationQueueWorker
from .sqlite_store import SQLiteNotificationStore
from .telegram_client import TelegramClient
from .templates import build_telegram_message


logger = logging.getLogger(__name__)
_MANAGER_LOCK = threading.Lock()
_MANAGER: Optional["NotificationManager"] = None


class NotificationManager:
    """Main non-blocking interface for Safe Guard event handling."""

    def __init__(self, config: Optional[NotificationConfig] = None):
        self._cfg = config or load_notification_config()
        self._classifier = EventClassifier(self._cfg)
        self._store = SQLiteNotificationStore(self._cfg.sqlite_path)
        self._telegram = TelegramClient(self._cfg)
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

    def handle_event(self, event: SafeGuardEvent, prefs: NotificationPreferences) -> DispatchAcceptance:
        """Classify and enqueue notification work without blocking inference."""
        if not self.enabled:
            return DispatchAcceptance(
                enabled=False,
                tier="disabled",
                reason="safe_guard_disabled",
                actions={},
                enqueued=False,
                state="disabled",
            )

        decision = self._classifier.classify(event, prefs)
        # Persist the decision first so the audit trail remains the source of truth
        # even when the outbound channel is later suppressed or fails.
        self._store.upsert_event(event, decision, prefs)

        if decision.tier.value == "tier3_silent":
            logger.info("safe_guard silent event event_id=%s", event.event_id)
            return DispatchAcceptance(
                enabled=True,
                tier=decision.tier.value,
                reason=decision.reason,
                actions=dict(decision.actions),
                enqueued=False,
                state="silent",
            )

        if not self._store.should_enqueue(event.event_id, self._cfg.alert_cooldown_seconds, event.resident_id):
            logger.info("safe_guard dedup suppressed event_id=%s", event.event_id)
            return DispatchAcceptance(
                enabled=True,
                tier=decision.tier.value,
                reason=decision.reason,
                actions=dict(decision.actions),
                enqueued=False,
                state="dedup_suppressed",
            )

        accepted = self._worker.submit(
            DispatchJob(
                event_id=event.event_id,
                fn=lambda: self._dispatch(event, decision, prefs),
            )
        )
        if not accepted:
            logger.warning("safe_guard enqueue failed event_id=%s", event.event_id)
        return DispatchAcceptance(
            enabled=True,
            tier=decision.tier.value,
            reason=decision.reason,
            actions=dict(decision.actions),
            enqueued=bool(accepted),
            state="enqueued" if accepted else "queue_full",
        )

    def _dispatch(self, event: SafeGuardEvent, decision: TierDecision, prefs: NotificationPreferences) -> None:
        results = []
        caregiver_chat_id = str(prefs.caregiver_telegram_chat_id or self._cfg.telegram_chat_id or "").strip()
        if decision.actions.get("telegram", False):
            analysis_report = generate_event_ai_report(event, decision, self._cfg)
            results.append(
                self._with_retry(
                    lambda: self._telegram.send_message(
                        chat_id=caregiver_chat_id,
                        text=build_telegram_message(event, decision, analysis_report),
                    )
                )
            )

        # Delivery attempts are recorded in the Safe Guard store, not the older
        # queue-log surface, so operator-visible history matches actual dispatch.
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
