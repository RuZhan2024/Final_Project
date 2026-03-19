from __future__ import annotations

from datetime import timezone
from email.message import EmailMessage

from .models import SafeGuardEvent, SafeGuardTier, TierDecision


def _fmt_ts(event: SafeGuardEvent) -> str:
    ts = event.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def build_phone_message(event: SafeGuardEvent) -> str:
    return f"Emergency alert. A fall has been detected at {event.location}. Please check immediately."


def build_sms_message(event: SafeGuardEvent, decision: TierDecision) -> str:
    ts_txt = _fmt_ts(event)
    if decision.tier == SafeGuardTier.TIER2:
        return (
            f"Safe Guard alert: possible fall at {event.location} on {ts_txt}. "
            f"Event is ambiguous. Please check the live stream. Ref:{event.event_id}"
        )
    return f"Safe Guard alert: fall detected at {event.location} on {ts_txt}. Please check immediately. Ref:{event.event_id}"


def build_email_message(
    event: SafeGuardEvent,
    decision: TierDecision,
    *,
    caregiver_email: str,
    email_from: str,
    app_base_url: str,
) -> EmailMessage:
    msg = EmailMessage()
    msg["To"] = caregiver_email
    msg["From"] = email_from
    msg["Subject"] = f"Safe Guard {decision.tier.value}: event {event.event_id}"
    event_url = f"{app_base_url}/events"
    body = "\n".join(
        [
            "Safe Guard detailed event report",
            "",
            f"event_id: {event.event_id}",
            f"timestamp: {_fmt_ts(event)}",
            f"location: {event.location}",
            f"triage_state: {event.triage_state}",
            f"probability: {event.probability:.4f}",
            f"threshold: {event.threshold:.4f}",
            f"margin: {event.margin:.4f}",
            f"uncertainty: {event.uncertainty:.4f}",
            f"alert_tier: {decision.tier.value}",
            f"notification_actions: {decision.actions}",
            f"interpretation: {decision.reason}",
            f"recommendation: {decision.recommendation}",
            f"event_history_url: {event_url}",
        ]
    )
    msg.set_content(body)
    return msg
