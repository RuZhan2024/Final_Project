from __future__ import annotations

import json
import sqlite3
import threading

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from .models import DeliveryResult, NotificationPreferences, SafeGuardEvent, TierDecision


class SQLiteNotificationStore:
    """Local SQLite audit store for Safe Guard notifications."""

    def __init__(self, db_path: str):
        self._path = Path(db_path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._path), timeout=10.0)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS notification_events (
            event_id TEXT PRIMARY KEY,
            resident_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            location TEXT NOT NULL,
            is_fall_like_event INTEGER NOT NULL,
            probability REAL NOT NULL,
            threshold REAL NOT NULL,
            margin REAL NOT NULL,
            uncertainty REAL NOT NULL,
            alert_tier TEXT NOT NULL,
            telegram_attempted INTEGER NOT NULL DEFAULT 0,
            phone_attempted INTEGER NOT NULL DEFAULT 0,
            sms_attempted INTEGER NOT NULL DEFAULT 0,
            email_attempted INTEGER NOT NULL DEFAULT 0,
            telegram_status TEXT,
            phone_status TEXT,
            sms_status TEXT,
            email_status TEXT,
            caregiver_feedback TEXT,
            notes TEXT,
            event_payload_json TEXT,
            unresolved INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS notification_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            channel TEXT NOT NULL,
            attempted INTEGER NOT NULL,
            status TEXT NOT NULL,
            detail TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS caregiver_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT,
            resident_id INTEGER,
            reply_code TEXT NOT NULL,
            mapped_status TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_notification_events_resident_time
            ON notification_events(resident_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_notification_events_unresolved
            ON notification_events(unresolved, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_notification_attempts_event
            ON notification_attempts(event_id, created_at DESC);
        """
        with self._lock, self._connect() as conn:
            conn.executescript(schema)
            cols = {
                str(row["name"])
                for row in (conn.execute("PRAGMA table_info(notification_events)").fetchall() or [])
            }
            if "telegram_attempted" not in cols:
                conn.execute("ALTER TABLE notification_events ADD COLUMN telegram_attempted INTEGER NOT NULL DEFAULT 0")
            if "telegram_status" not in cols:
                conn.execute("ALTER TABLE notification_events ADD COLUMN telegram_status TEXT")

    def upsert_event(self, event: SafeGuardEvent, decision: TierDecision, prefs: NotificationPreferences) -> None:
        payload_json = json.dumps(
            {
                "event_id": event.event_id,
                "resident_id": event.resident_id,
                "location": event.location,
                "triage_state": event.triage_state,
                "safe_alert": event.safe_alert,
                "recall_alert": event.recall_alert,
                "model_code": event.model_code,
                "dataset_code": event.dataset_code,
                "op_code": event.op_code,
                "source": event.source,
                "meta": event.meta,
                "preferences": {
                    "telegram_enabled": prefs.telegram_enabled,
                    "caregiver_name": prefs.caregiver_name,
                    "caregiver_telegram_chat_id": prefs.caregiver_telegram_chat_id,
                },
            }
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO notification_events (
                    event_id, resident_id, timestamp, location, is_fall_like_event,
                    probability, threshold, margin, uncertainty, alert_tier,
                    notes, event_payload_json, unresolved, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(event_id) DO UPDATE SET
                    resident_id=excluded.resident_id,
                    timestamp=excluded.timestamp,
                    location=excluded.location,
                    is_fall_like_event=excluded.is_fall_like_event,
                    probability=excluded.probability,
                    threshold=excluded.threshold,
                    margin=excluded.margin,
                    uncertainty=excluded.uncertainty,
                    alert_tier=excluded.alert_tier,
                    notes=excluded.notes,
                    event_payload_json=excluded.event_payload_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    event.event_id,
                    int(event.resident_id),
                    event.timestamp.isoformat(),
                    event.location,
                    1 if event.alert_worthy else 0,
                    float(event.probability),
                    float(event.threshold),
                    float(event.margin),
                    float(event.uncertainty),
                    decision.tier.value,
                    event.notes,
                    payload_json,
                ),
            )

    def should_enqueue(self, event_id: str, cooldown_seconds: int, resident_id: int) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT event_id FROM notification_events
                WHERE event_id = ?
                  AND (telegram_attempted = 1 OR phone_attempted = 1 OR sms_attempted = 1 OR email_attempted = 1)
                LIMIT 1
                """,
                (event_id,),
            ).fetchone()
            if row:
                return False
            if cooldown_seconds <= 0:
                return True
            row = conn.execute(
                """
                SELECT timestamp FROM notification_events
                WHERE resident_id = ?
                  AND alert_tier IN ('tier1_high_confidence_fall', 'tier2_ambiguous_fall')
                  AND (telegram_attempted = 1 OR phone_attempted = 1 OR sms_attempted = 1 OR email_attempted = 1)
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (int(resident_id),),
            ).fetchone()
            if not row:
                return True
            ts_txt = str(row["timestamp"])
        prev = datetime.fromisoformat(ts_txt)
        now = datetime.now(timezone.utc)
        if prev.tzinfo is None:
            prev = prev.replace(tzinfo=timezone.utc)
        return (now - prev).total_seconds() >= float(cooldown_seconds)

    def record_delivery(self, event_id: str, result: DeliveryResult) -> None:
        col_attempt = f"{result.channel}_attempted"
        col_status = f"{result.channel}_status"
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO notification_attempts (event_id, channel, attempted, status, detail)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_id, result.channel, 1 if result.attempted else 0, result.status, result.detail),
            )
            conn.execute(
                f"""
                UPDATE notification_events
                SET {col_attempt} = ?, {col_status} = ?, updated_at = CURRENT_TIMESTAMP
                WHERE event_id = ?
                """,
                (1 if result.attempted else 0, result.status, event_id),
            )

    def mark_feedback(self, event_id: Optional[str], resident_id: int, reply_code: str, mapped_status: str, notes: str = "") -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO caregiver_feedback (event_id, resident_id, reply_code, mapped_status, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_id, int(resident_id), reply_code, mapped_status, notes),
            )
            if event_id:
                conn.execute(
                    """
                    UPDATE notification_events
                    SET caregiver_feedback = ?, unresolved = CASE WHEN ? IN ('resolved', 'assistance_provided') THEN 0 ELSE unresolved END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE event_id = ?
                    """,
                    (mapped_status, mapped_status, event_id),
                )

    def resolve_event(self, event_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE notification_events SET unresolved = 0, updated_at = CURRENT_TIMESTAMP WHERE event_id = ?",
                (event_id,),
            )

    def get_most_recent_unresolved_event_id(self, resident_id: int) -> Optional[str]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT event_id FROM notification_events
                WHERE resident_id = ? AND unresolved = 1
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (int(resident_id),),
            ).fetchone()
        return str(row["event_id"]) if row and row["event_id"] is not None else None
