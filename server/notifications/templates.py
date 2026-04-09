from __future__ import annotations

from datetime import timezone
from .models import SafeGuardEvent, SafeGuardTier, TierDecision


def _fmt_ts(event: SafeGuardEvent) -> str:
    ts = event.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def build_telegram_message(event: SafeGuardEvent, decision: TierDecision, analysis_report: str) -> str:
    ts_txt = _fmt_ts(event)
    if decision.tier == SafeGuardTier.TIER2:
        header = f"Safe Guard alert: possible fall at {event.location} on {ts_txt}."
    else:
        header = f"Safe Guard alert: fall detected at {event.location} on {ts_txt}."
    return "\n".join(
        [
            header,
            f"Recommendation: {decision.recommendation}",
            f"Ref: {event.event_id}",
            "",
            analysis_report.strip() if analysis_report.strip() else "AI analysis unavailable.",
        ]
    ).strip()
