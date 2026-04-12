from __future__ import annotations

from typing import Any, Dict, Optional


def get_last_pred_latency_ms() -> Optional[float]:
    from . import core

    return core.LAST_PRED_LATENCY_MS


def get_last_pred_p_fall() -> Optional[float]:
    from . import core

    return core.LAST_PRED_P_FALL


def get_last_pred_decision() -> Optional[str]:
    from . import core

    return core.LAST_PRED_DECISION


def get_session_store() -> Dict[str, Dict[str, Any]]:
    from . import core

    return core._SESSION_STATE


def set_last_prediction_snapshot(
    *,
    latency_ms: Optional[float],
    p_fall: Optional[float],
    decision: Optional[str],
    model_code: Optional[str],
    ts_iso: Optional[str],
) -> None:
    from . import core

    core.LAST_PRED_LATENCY_MS = latency_ms
    core.LAST_PRED_P_FALL = p_fall
    core.LAST_PRED_DECISION = decision
    core.LAST_PRED_MODEL_CODE = model_code
    core.LAST_PRED_TS_ISO = ts_iso
