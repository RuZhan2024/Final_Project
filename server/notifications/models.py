from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class SafeGuardTier(str, Enum):
    TIER1 = "tier1_high_confidence_fall"
    TIER2 = "tier2_ambiguous_fall"
    TIER3 = "tier3_silent"


@dataclass(frozen=True)
class NotificationPreferences:
    telegram_enabled: bool = True
    caregiver_name: str = ""
    caregiver_telegram_chat_id: str = ""


@dataclass(frozen=True)
class SafeGuardEvent:
    event_id: str
    resident_id: int
    location: str
    probability: float
    uncertainty: float
    threshold: float
    margin: float
    triage_state: str
    safe_alert: bool
    recall_alert: bool
    model_code: str
    dataset_code: str
    op_code: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "monitor"
    notes: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def alert_worthy(self) -> bool:
        triage = str(self.triage_state or "").lower()
        return bool(self.safe_alert or self.recall_alert or triage in {"fall", "uncertain"})


@dataclass(frozen=True)
class TierDecision:
    tier: SafeGuardTier
    reason: str
    actions: Dict[str, bool]
    recommendation: str


@dataclass(frozen=True)
class DeliveryResult:
    channel: str
    attempted: bool
    status: str
    detail: str = ""
