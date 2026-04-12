from __future__ import annotations

from typing import Any, Dict, Optional

from .config import get_app_config


SESSION_STATE: Dict[str, Dict[str, Any]] = {}

LAST_PRED_LATENCY_MS: Optional[float] = None
LAST_PRED_P_FALL: Optional[float] = None
LAST_PRED_DECISION: Optional[str] = None
LAST_PRED_MODEL_CODE: Optional[str] = None
LAST_PRED_TS_ISO: Optional[str] = None


def get_last_pred_latency_ms() -> Optional[float]:
    return LAST_PRED_LATENCY_MS


def get_last_pred_p_fall() -> Optional[float]:
    return LAST_PRED_P_FALL


def get_last_pred_decision() -> Optional[str]:
    return LAST_PRED_DECISION


def get_session_store() -> Dict[str, Dict[str, Any]]:
    return SESSION_STATE


def touch_session_state(session_id: str, now_s: Optional[float] = None) -> Dict[str, Any]:
    import time

    t_s = float(now_s if now_s is not None else time.time())
    st = SESSION_STATE.setdefault(str(session_id), {})
    st["last_seen_s"] = t_s
    return st


def prune_session_state(
    now_s: Optional[float] = None,
    *,
    ttl_s: Optional[int] = None,
    max_states: Optional[int] = None,
) -> int:
    import time

    if not SESSION_STATE:
        return 0

    t_s = float(now_s if now_s is not None else time.time())
    cfg = get_app_config()
    effective_ttl = max(60, int(ttl_s if ttl_s is not None else cfg.session_ttl_s))
    effective_cap = max(10, int(max_states if max_states is not None else cfg.session_max_states))
    cutoff = t_s - float(effective_ttl)
    removed = 0

    stale_ids = []
    for sid, st in SESSION_STATE.items():
        try:
            last_seen = float((st or {}).get("last_seen_s", 0.0) or 0.0)
        except Exception:
            last_seen = 0.0
        if last_seen < cutoff:
            stale_ids.append(sid)

    for sid in stale_ids:
        if SESSION_STATE.pop(sid, None) is not None:
            removed += 1

    if len(SESSION_STATE) > effective_cap:
        ordered = sorted(
            SESSION_STATE.items(),
            key=lambda kv: float((kv[1] or {}).get("last_seen_s", 0.0) or 0.0),
        )
        overflow = len(SESSION_STATE) - effective_cap
        for sid, _ in ordered[:overflow]:
            if SESSION_STATE.pop(sid, None) is not None:
                removed += 1

    return removed


def set_last_prediction_snapshot(
    *,
    latency_ms: Optional[float],
    p_fall: Optional[float],
    decision: Optional[str],
    model_code: Optional[str],
    ts_iso: Optional[str],
) -> None:
    global LAST_PRED_LATENCY_MS, LAST_PRED_P_FALL, LAST_PRED_DECISION, LAST_PRED_MODEL_CODE, LAST_PRED_TS_ISO
    LAST_PRED_LATENCY_MS = latency_ms
    LAST_PRED_P_FALL = p_fall
    LAST_PRED_DECISION = decision
    LAST_PRED_MODEL_CODE = model_code
    LAST_PRED_TS_ISO = ts_iso
