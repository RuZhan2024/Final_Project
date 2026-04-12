from __future__ import annotations

from typing import Optional


def get_last_pred_latency_ms() -> Optional[float]:
    from . import core

    return core.LAST_PRED_LATENCY_MS


def get_last_pred_p_fall() -> Optional[float]:
    from . import core

    return core.LAST_PRED_P_FALL


def get_last_pred_decision() -> Optional[str]:
    from . import core

    return core.LAST_PRED_DECISION
