from __future__ import annotations

from .models import SafeGuardEvent, SafeGuardTier, TierDecision
from ..time_utils import format_local_event_timestamp


def _fmt_ts(event: SafeGuardEvent) -> str:
    return format_local_event_timestamp(event.timestamp)


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
            analysis_report.strip() if analysis_report.strip() else "Generated summary unavailable.",
        ]
    ).strip()
