from __future__ import annotations

from typing import Any, Dict


def reset_monitor_session(session_store: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    session_store.pop(str(session_id), None)
    return {"ok": True, "session_id": str(session_id)}


def get_monitor_session_state(session_store: Dict[str, Any], session_id: str, now_s: float) -> Dict[str, Any]:
    state = session_store.setdefault(str(session_id), {})
    state.setdefault("session_start_t_s", float(now_s))
    state.setdefault("trackers", {})
    state.setdefault("trackers_cfg", {})
    return state
