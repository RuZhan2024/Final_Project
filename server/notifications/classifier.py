from __future__ import annotations

from .config import NotificationConfig
from .models import NotificationPreferences, SafeGuardEvent, SafeGuardTier, TierDecision


class EventClassifier:
    """Threshold-aware and policy-aware Safe Guard tier classifier."""

    def __init__(self, config: NotificationConfig):
        self._cfg = config

    def classify(self, event: SafeGuardEvent, prefs: NotificationPreferences) -> TierDecision:
        if not event.alert_worthy:
            return TierDecision(
                tier=SafeGuardTier.TIER3,
                reason="not_alert_worthy",
                actions={"telegram": False},
                recommendation="No caregiver action required. Keep event for audit only.",
            )

        if (
            event.margin >= self._cfg.high_conf_margin
            and event.uncertainty < self._cfg.low_uncertainty_threshold
            and str(event.triage_state or "").lower() == "fall"
        ):
            return TierDecision(
                tier=SafeGuardTier.TIER1,
                reason="strong_margin_low_uncertainty",
                actions={
                    "telegram": bool(prefs.telegram_enabled),
                },
                recommendation="High-confidence fall. Escalate immediately and check the resident.",
            )

        return TierDecision(
            tier=SafeGuardTier.TIER2,
            reason=(
                "high_uncertainty"
                if event.uncertainty >= self._cfg.high_uncertainty_threshold
                else "borderline_margin_or_policy_promoted"
            ),
            actions={
                "telegram": bool(prefs.telegram_enabled),
            },
            recommendation="Borderline or ambiguous fall event. Review the live stream or recent clip.",
        )
