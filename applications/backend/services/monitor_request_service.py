from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

from ..runtime_state import get_session_store
from .monitor_context_service import load_monitor_request_context
from .monitor_session_service import get_monitor_session_state


@dataclass(frozen=True)
class MonitorPreparedRequest:
    compact_response: bool
    session_id: str
    input_source: str
    is_replay: bool
    requested_mode: str
    mode: str
    dataset_code: str
    op_code: str
    requested_use_mc: bool
    requested_mc_M: int
    effective_use_mc: bool
    effective_mc_M: int
    persist: bool
    target_T: int
    raw_t_ms: Any
    raw_stats: Dict[str, Any]
    xy: List[Any]
    conf: List[Any]
    cap_fps_est: Optional[float]
    resident_id: int
    active_model_code: str
    cooldown_sec: int
    runtime: Any
    event_location: str
    expected_fps: float
    effective_fps: float
    session_state: Dict[str, Any]
    current_t_s: float
    specs: Dict[str, Any]
    tcn_key: str
    gcn_key: str
    guard_spec_key: str
    primary_spec_key: str
    primary_model_key: str
    window_end_t_ms: Optional[float]
    window_seq: Any


def prepare_monitor_request(
    *,
    payload,
    logger: logging.Logger,
    get_conn,
    normalize_dataset_code,
    coerce_bool,
    decode_quantized_raw_window,
    raw_window_stats,
    resolve_runtime_fps,
    resolve_monitor_specs,
    get_pose_preprocess_cfg,
    resample_pose_window,
    preprocess_online_raw_window,
    direct_window_stats,
    get_deploy_specs,
    ensure_system_settings_schema,
    detect_variants,
    table_exists,
) -> MonitorPreparedRequest:
    payload_d = payload.model_dump()
    compact_response = coerce_bool(payload_d.get("compact_response"), False)

    session_id = str(payload_d.get("session_id") or "default")
    input_source = str(payload_d.get("input_source") or "").strip().lower()
    is_replay = input_source in {"video", "replay", "file"}

    requested_mode = str(payload_d.get("mode") or "tcn").lower().strip()
    mode = requested_mode
    if mode in {"hyb", "hybrid", "dual"}:
        mode = "hybrid"
    elif mode not in {"tcn", "gcn", "hybrid"}:
        mode = "tcn"

    dataset_code = normalize_dataset_code(payload_d.get("dataset_code") or payload_d.get("dataset"))
    op_code = str(payload_d.get("op_code") or payload_d.get("op") or "").upper().strip()
    requested_use_mc = payload_d.get("use_mc")
    requested_mc_M = payload_d.get("mc_M")
    persist = coerce_bool(payload_d.get("persist", False), False)

    target_T = int(payload_d.get("target_T") or 48)
    raw_xy = payload_d.get("raw_xy")
    raw_conf = payload_d.get("raw_conf")
    raw_xy_q = payload_d.get("raw_xy_q")
    raw_conf_q = payload_d.get("raw_conf_q")
    raw_shape = payload_d.get("raw_shape")
    raw_t_ms = payload_d.get("raw_t_ms")
    window_end_t_ms = payload_d.get("window_end_t_ms", None)
    window_seq = payload_d.get("window_seq", None)
    if raw_xy is None and raw_xy_q is not None:
        dec_xy, dec_conf = decode_quantized_raw_window(raw_xy_q, raw_conf_q, raw_shape)
        if dec_xy is not None:
            raw_xy = dec_xy
            raw_conf = dec_conf

    raw_stats = raw_window_stats(raw_t_ms, raw_xy, raw_conf)

    xy: List[Any] = []
    conf: List[Any] = []
    cap_fps_est: Optional[float] = None

    resident_id = int(payload_d.get("resident_id") or 1)
    active_model_code = mode.upper()
    request_context = None
    try:
        with get_conn() as conn:
            request_context = load_monitor_request_context(
                conn=conn,
                resident_id=resident_id,
                input_source=input_source,
                mode=mode,
                dataset_code=dataset_code,
                op_code=op_code,
                event_location=payload_d.get("location"),
                active_model_code=active_model_code,
                requested_use_mc=requested_use_mc,
                requested_mc_M=requested_mc_M,
                is_replay=is_replay,
                ensure_system_settings_schema=ensure_system_settings_schema,
                detect_variants=detect_variants,
                table_exists=table_exists,
            )
    except (MySQLError, RuntimeError, OSError, TypeError, ValueError) as exc:
        logger.warning(
            "monitor.prepare_request: failed to load DB defaults (resident_id=%s, session_id=%s, mode=%s, dataset=%s): %s",
            resident_id,
            session_id,
            mode,
            dataset_code or "unset",
            exc,
        )

    if request_context is None:
        request_context = load_monitor_request_context(
            conn=None,
            resident_id=resident_id,
            input_source=input_source,
            mode=mode,
            dataset_code=dataset_code,
            op_code=op_code,
            event_location=payload_d.get("location"),
            active_model_code=active_model_code,
            requested_use_mc=requested_use_mc,
            requested_mc_M=requested_mc_M,
            is_replay=is_replay,
            ensure_system_settings_schema=lambda _conn: None,
            detect_variants=lambda _conn: {},
            table_exists=lambda _conn, _table: False,
        )

    event_location = request_context.event_location
    dataset_code = request_context.dataset_code
    op_code = request_context.op_code
    active_model_code = request_context.active_model_code
    cooldown_sec = request_context.cooldown_sec
    requested_use_mc = request_context.requested_use_mc
    requested_mc_M = request_context.requested_mc_M
    effective_use_mc = request_context.effective_use_mc
    effective_mc_M = request_context.effective_mc_M
    runtime = request_context.runtime

    raw_fps_est = raw_stats.get("raw_fps_est")
    expected_fps, effective_fps = resolve_runtime_fps(
        dataset_code=dataset_code,
        payload_d=payload_d,
        raw_fps_est=raw_fps_est,
        is_replay=is_replay,
    )
    current_t_s = time.time()
    session_state = get_monitor_session_state(get_session_store(), session_id, current_t_s)

    specs = get_deploy_specs()
    spec_selection = resolve_monitor_specs(
        specs=specs,
        dataset_code=dataset_code,
        mode=mode,
        payload_d=payload_d,
    )
    mode = spec_selection["mode"]
    tcn_key = spec_selection["tcn_key"]
    gcn_key = spec_selection["gcn_key"]
    guard_spec_key = spec_selection["guard_spec_key"]
    primary_spec_key = spec_selection["primary_spec_key"]
    primary_model_key = spec_selection["primary_model_key"]

    raw_preproc_cfg = None
    if raw_xy is not None and raw_t_ms is not None:
        try:
            raw_preproc_cfg = get_pose_preprocess_cfg(primary_spec_key, enrich_from_checkpoint=False)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            raw_preproc_cfg = None

    if raw_xy is not None and raw_t_ms is not None:
        xy, conf, _, _, cap_fps_est = resample_pose_window(
            raw_t_ms=raw_t_ms,
            raw_xy=raw_xy,
            raw_conf=raw_conf,
            target_fps=float(effective_fps),
            target_T=target_T,
            window_end_t_ms=float(window_end_t_ms) if window_end_t_ms is not None else None,
        )
        if xy:
            xy, conf = preprocess_online_raw_window(xy, conf, cfg=raw_preproc_cfg)

    if not xy:
        xy = payload_d.get("xy") or []
        conf = payload_d.get("conf") or []

    if (raw_xy is None or raw_t_ms is None) and isinstance(xy, list) and len(xy) > 0 and (raw_stats.get("raw_len") or 0) <= 0:
        raw_stats = direct_window_stats(xy, conf, effective_fps=float(effective_fps))

    if not xy:
        raise HTTPException(status_code=400, detail="payload must include raw_* (preferred) or xy")

    return MonitorPreparedRequest(
        compact_response=bool(compact_response),
        session_id=session_id,
        input_source=input_source,
        is_replay=bool(is_replay),
        requested_mode=requested_mode,
        mode=mode,
        dataset_code=dataset_code,
        op_code=op_code,
        requested_use_mc=bool(requested_use_mc),
        requested_mc_M=int(requested_mc_M),
        effective_use_mc=bool(effective_use_mc),
        effective_mc_M=int(effective_mc_M),
        persist=bool(persist),
        target_T=target_T,
        raw_t_ms=raw_t_ms,
        raw_stats=raw_stats,
        xy=xy,
        conf=conf,
        cap_fps_est=cap_fps_est,
        resident_id=resident_id,
        active_model_code=active_model_code,
        cooldown_sec=int(cooldown_sec),
        runtime=runtime,
        event_location=event_location,
        expected_fps=float(expected_fps),
        effective_fps=float(effective_fps),
        session_state=session_state,
        current_t_s=float(current_t_s),
        specs=specs,
        tcn_key=tcn_key,
        gcn_key=gcn_key,
        guard_spec_key=guard_spec_key,
        primary_spec_key=primary_spec_key,
        primary_model_key=primary_model_key,
        window_end_t_ms=float(window_end_t_ms) if window_end_t_ms is not None else None,
        window_seq=window_seq,
    )
