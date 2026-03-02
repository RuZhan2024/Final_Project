from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Body, HTTPException, Query

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

from .. import core
from ..core import MonitorPredictPayload, _detect_variants, _ensure_system_settings_schema, _table_exists
from ..db import get_conn
from ..deploy_runtime import (
    fuse_hybrid as _fuse_hybrid,
    get_specs as _get_deploy_specs,
    predict_spec as _predict_spec,
)
from ..online_alert import OnlineAlertTracker


router = APIRouter()
logger = logging.getLogger(__name__)



def _resample_pose_window(
    *,
    raw_t_ms: Any,
    raw_xy: Any,
    raw_conf: Any = None,
    target_fps: float = 30.0,
    target_T: int = 48,
    window_end_t_ms: Optional[float] = None,
) -> Tuple[List[Any], List[Any], float, float, Optional[float]]:
    """Resample variable-FPS pose frames to a fixed FPS + fixed length window."""

    if (
        not isinstance(raw_t_ms, list)
        or not isinstance(raw_xy, list)
        or len(raw_t_ms) != len(raw_xy)
        or len(raw_xy) < 1
    ):
        return [], [], 0.0, 0.0, None

    if isinstance(raw_conf, list) and len(raw_conf) == len(raw_xy):
        use_conf = True
    else:
        use_conf = False
        raw_conf = [None] * len(raw_xy)

    t: List[float] = []
    xy: List[Any] = []
    conf: List[Any] = []
    last_t: Optional[float] = None
    for ti, xyi, ci in zip(raw_t_ms, raw_xy, raw_conf):
        try:
            tf = float(ti)
        except (TypeError, ValueError):
            continue
        if last_t is not None and tf <= last_t:
            continue
        if not isinstance(xyi, list):
            continue
        t.append(tf)
        xy.append(xyi)
        conf.append(ci if use_conf else None)
        last_t = tf

    if len(t) < 1:
        return [], [], 0.0, 0.0, None

    cap_fps: Optional[float] = None
    if len(t) >= 2:
        dt_s = (t[-1] - t[0]) / 1000.0
        if dt_s > 1e-6:
            cap_fps = (len(t) - 1) / dt_s

    target_fps = float(target_fps) if target_fps and float(target_fps) > 0 else 30.0
    target_T = int(target_T) if target_T and int(target_T) > 1 else 48
    dt_ms = 1000.0 / target_fps

    end_t_ms = float(window_end_t_ms) if window_end_t_ms is not None else t[-1]
    start_t_ms = end_t_ms - (target_T - 1) * dt_ms

    j = 0
    xy_out: List[Any] = []
    conf_out: List[Any] = []

    def lerp(a: float, b: float, alpha: float) -> float:
        return a + (b - a) * alpha

    def interp_frame(frame0: Any, frame1: Any, alpha: float, is_xy: bool) -> Any:
        if not isinstance(frame0, list):
            return frame1
        if not isinstance(frame1, list):
            return frame0
        out = []
        if is_xy:
            n = min(len(frame0), len(frame1))
            for k in range(n):
                p0 = frame0[k] if isinstance(frame0[k], list) and len(frame0[k]) >= 2 else [0.0, 0.0]
                p1 = frame1[k] if isinstance(frame1[k], list) and len(frame1[k]) >= 2 else [0.0, 0.0]
                try:
                    x0, y0 = float(p0[0]), float(p0[1])
                    x1, y1 = float(p1[0]), float(p1[1])
                except (TypeError, ValueError):
                    x0 = y0 = x1 = y1 = 0.0
                out.append([lerp(x0, x1, alpha), lerp(y0, y1, alpha)])
            if len(frame0) > n:
                out.extend(frame0[n:])
            elif len(frame1) > n:
                out.extend(frame1[n:])
            return out

        n = min(len(frame0), len(frame1))
        for k in range(n):
            try:
                v0 = float(frame0[k])
            except (TypeError, ValueError):
                v0 = 0.0
            try:
                v1 = float(frame1[k])
            except (TypeError, ValueError):
                v1 = v0
            out.append(lerp(v0, v1, alpha))
        if len(frame0) > n:
            out.extend(frame0[n:])
        elif len(frame1) > n:
            out.extend(frame1[n:])
        return out

    for k in range(target_T):
        tw = start_t_ms + k * dt_ms

        if tw <= t[0]:
            xy_out.append(xy[0])
            conf_out.append(
                conf[0]
                if use_conf and isinstance(conf[0], list)
                else [1.0] * (len(xy[0]) if isinstance(xy[0], list) else 0)
            )
            continue
        if tw >= t[-1]:
            xy_out.append(xy[-1])
            conf_out.append(
                conf[-1]
                if use_conf and isinstance(conf[-1], list)
                else [1.0] * (len(xy[-1]) if isinstance(xy[-1], list) else 0)
            )
            continue

        while j + 1 < len(t) and t[j + 1] < tw:
            j += 1

        t0, t1 = t[j], t[j + 1]
        alpha = 0.0 if t1 <= t0 else (tw - t0) / (t1 - t0)

        xy_out.append(interp_frame(xy[j], xy[j + 1], alpha, True))

        if use_conf and isinstance(conf[j], list) and isinstance(conf[j + 1], list):
            conf_out.append(interp_frame(conf[j], conf[j + 1], alpha, False))
        else:
            conf_out.append([1.0] * (len(xy_out[-1]) if isinstance(xy_out[-1], list) else 0))

    return xy_out, conf_out, start_t_ms, end_t_ms, cap_fps


@router.post("/api/monitor/reset_session")
@router.post("/api/v1/monitor/reset_session")
def reset_session(session_id: str = Query(...)) -> Dict[str, Any]:
    core._SESSION_STATE.pop(session_id, None)
    return {"ok": True, "session_id": session_id}


@router.post("/api/monitor/predict_window")
@router.post("/api/v1/monitor/predict_window")
def predict_window(payload: MonitorPredictPayload = Body(...)) -> Dict[str, Any]:
    """Score one window from the live monitor UI."""
    payload_d = payload.model_dump()

    t0 = time.time()

    # -------------
    # Inputs
    # -------------
    session_id = str(payload_d.get("session_id") or "default")

    mode = str(payload_d.get("mode") or "hybrid").lower().strip()
    if mode in {"hyb", "hybrid", "dual"}:
        mode = "dual"
    elif mode not in {"tcn", "gcn"}:
        mode = "dual"
    requested_mode = mode

    dataset_code = str(payload_d.get("dataset_code") or payload_d.get("dataset") or "").lower().strip()
    op_code = str(payload_d.get("op_code") or payload_d.get("op") or "").upper().strip()

    use_mc = payload_d.get("use_mc")
    mc_M = payload_d.get("mc_M")

    persist = bool(payload_d.get("persist", False))

    target_T = int(payload_d.get("target_T") or 48)
    raw_xy = payload_d.get("raw_xy")
    raw_conf = payload_d.get("raw_conf")
    raw_t_ms = payload_d.get("raw_t_ms")
    window_end_t_ms = payload_d.get("window_end_t_ms", None)

    xy: List[Any] = []
    conf: List[Any] = []
    cap_fps_est: Optional[float] = None

    # -------------
    # Defaults from DB (if available)
    # -------------
    resident_id = int(payload_d.get("resident_id") or 1)
    active_model_code = "HYBRID" if mode == "dual" else mode.upper()

    try:
        with get_conn() as conn:
            _ensure_system_settings_schema(conn)
            variants = _detect_variants(conn)

            sys_row = None
            with conn.cursor() as cur:
                if variants.get("settings") == "v2" and _table_exists(conn, "system_settings"):
                    cur.execute("SELECT * FROM system_settings WHERE resident_id=%s LIMIT 1", (resident_id,))
                    sys_row = cur.fetchone()
                elif _table_exists(conn, "settings"):
                    cur.execute("SELECT * FROM settings WHERE resident_id=%s LIMIT 1", (resident_id,))
                    sys_row = cur.fetchone()

            if isinstance(sys_row, dict):
                if not dataset_code and sys_row.get("active_dataset_code"):
                    dataset_code = str(sys_row.get("active_dataset_code") or "").lower()
                if use_mc is None and sys_row.get("mc_enabled") is not None:
                    use_mc = (
                        bool(int(sys_row.get("mc_enabled")))
                        if str(sys_row.get("mc_enabled")).isdigit()
                        else bool(sys_row.get("mc_enabled"))
                    )
                if mc_M is None and sys_row.get("mc_M") is not None:
                    mc_M = int(sys_row.get("mc_M"))
                if sys_row.get("alert_cooldown_sec") is not None:
                    cooldown_sec = int(sys_row.get("alert_cooldown_sec"))
                if sys_row.get("active_model_code"):
                    active_model_code = str(sys_row.get("active_model_code") or active_model_code)

                if (not op_code) and sys_row.get("active_op_code"):
                    op_code = str(sys_row.get("active_op_code") or "").upper().strip()

                op_id = None
                for k in ("active_operating_point", "active_operating_point_id"):
                    if sys_row.get(k) is not None:
                        op_id = sys_row.get(k)
                        break
                if (not op_code) and op_id and _table_exists(conn, "operating_points"):
                    with conn.cursor() as cur:
                        cur.execute("SELECT code FROM operating_points WHERE id=%s LIMIT 1", (int(op_id),))
                        r = cur.fetchone() or {}
                        if isinstance(r, dict) and r.get("code"):
                            op_code = str(r.get("code") or "").upper()
    except (MySQLError, RuntimeError, OSError, TypeError, ValueError) as exc:
        logger.warning(
            "monitor.predict_window: failed to load DB defaults (resident_id=%s, session_id=%s, mode=%s, dataset=%s): %s",
            resident_id,
            session_id,
            mode,
            dataset_code or "unset",
            exc,
        )

    if not dataset_code:
        dataset_code = "muvim"
    if not op_code:
        op_code = "OP-2"
    if use_mc is None:
        use_mc = True
    if mc_M is None:
        mc_M = 10

    expected_fps = {
        "le2i": 25,
        "urfd": 30,
        "caucafall": 23,
        "muvim": 30,
    }.get(dataset_code, int(payload_d.get("target_fps") or payload_d.get("fps") or 30))

    if raw_xy is not None and raw_t_ms is not None:
        xy, conf, _, _, cap_fps_est = _resample_pose_window(
            raw_t_ms=raw_t_ms,
            raw_xy=raw_xy,
            raw_conf=raw_conf,
            target_fps=float(expected_fps),
            target_T=target_T,
            window_end_t_ms=float(window_end_t_ms) if window_end_t_ms is not None else None,
        )

    if not xy:
        xy = payload_d.get("xy") or []
        conf = payload_d.get("conf") or []

    if not xy:
        raise HTTPException(status_code=400, detail="payload must include raw_* (preferred) or xy")

    try:
        if window_end_t_ms is not None:
            _now_ms = float(window_end_t_ms)
        elif raw_t_ms is not None and len(raw_t_ms) > 0:
            _now_ms = float(raw_t_ms[-1])
        else:
            _now_ms = time.time() * 1000.0
    except (TypeError, ValueError):
        _now_ms = time.time() * 1000.0
    _t_s = float(_now_ms) / 1000.0

    st = core._SESSION_STATE.setdefault(session_id, {})
    st_trackers = st.setdefault("trackers", {})
    st_trackers_cfg = st.setdefault("trackers_cfg", {})
    started_tcn = False
    started_gcn = False

    # -------------
    # Inference
    # -------------
    specs = _get_deploy_specs()

    def resolve_spec_key(arch: str, preferred: str) -> str:
        if preferred in specs:
            return preferred
        ds_prefix = f"{dataset_code}_"
        suffix = f"_{arch}"
        candidates = [k for k in specs.keys() if k.startswith(ds_prefix) and k.endswith(suffix)]
        if not candidates:
            return preferred
        candidates.sort(key=lambda x: (len(x), x))
        return candidates[0]

    def spec_key_for(arch: str) -> str:
        return f"{dataset_code}_{arch}".lower()

    tcn_key = resolve_spec_key("tcn", str(payload_d.get("model_tcn") or spec_key_for("tcn")).lower())
    gcn_key = resolve_spec_key("gcn", str(payload_d.get("model_gcn") or spec_key_for("gcn")).lower())
    has_tcn = tcn_key in specs
    has_gcn = gcn_key in specs
    if mode == "tcn":
        if not has_tcn:
            raise HTTPException(status_code=404, detail=f"No TCN deploy spec found for dataset '{dataset_code}'.")
    elif mode == "gcn":
        if not has_gcn:
            raise HTTPException(status_code=404, detail=f"No GCN deploy spec found for dataset '{dataset_code}'.")
    else:
        if not has_tcn and not has_gcn:
            raise HTTPException(status_code=404, detail=f"No deploy specs found for dataset '{dataset_code}'.")
        if has_tcn and not has_gcn:
            mode = "tcn"
        elif has_gcn and not has_tcn:
            mode = "gcn"

    models_out: Dict[str, Any] = {}
    tri_tcn = None
    tri_gcn = None

    if mode in {"tcn", "dual"}:
        out_tcn = _predict_spec(
            spec_key=tcn_key,
            joints_xy=xy,
            conf=conf,
            fps=float(expected_fps),
            target_T=target_T,
            op_code=op_code,
            use_mc=bool(use_mc),
            mc_M=int(mc_M),
        )

        cfg_tcn = out_tcn.get("alert_cfg") or {}
        trk = st_trackers.get(tcn_key)
        if trk is None or st_trackers_cfg.get(tcn_key) != cfg_tcn:
            trk = OnlineAlertTracker(cfg_tcn)
            st_trackers[tcn_key] = trk
            st_trackers_cfg[tcn_key] = cfg_tcn
        r = trk.step(
            p=float(out_tcn.get("mu") if out_tcn.get("mu") is not None else out_tcn.get("p_det", 0.0)),
            t_s=_t_s,
        )
        out_tcn["triage"] = {
            "state": r.triage_state,
            "ps": r.ps,
            "p_in": r.p_in,
            "tau_low": float(cfg_tcn.get("tau_low", out_tcn.get("tau_low", 0.0))),
            "tau_high": float(cfg_tcn.get("tau_high", out_tcn.get("tau_high", 0.0))),
            "ema_alpha": float(cfg_tcn.get("ema_alpha", 0.0)),
            "k": int(cfg_tcn.get("k", 2)),
            "n": int(cfg_tcn.get("n", 3)),
            "cooldown_s": float(cfg_tcn.get("cooldown_s", 0.0)),
            "cooldown_remaining_s": r.cooldown_remaining_s,
        }
        models_out["tcn"] = out_tcn
        tri_tcn = r.triage_state
        started_tcn = bool(r.started_event)

    if mode in {"gcn", "dual"}:
        out_gcn = _predict_spec(
            spec_key=gcn_key,
            joints_xy=xy,
            conf=conf,
            fps=float(expected_fps),
            target_T=target_T,
            op_code=op_code,
            use_mc=bool(use_mc),
            mc_M=int(mc_M),
        )
        models_out["gcn"] = out_gcn

        cfg_gcn = out_gcn.get("alert_cfg") or {}
        trk = st_trackers.get(gcn_key)
        if trk is None or st_trackers_cfg.get(gcn_key) != cfg_gcn:
            trk = OnlineAlertTracker(cfg_gcn)
            st_trackers[gcn_key] = trk
            st_trackers_cfg[gcn_key] = cfg_gcn
        res = trk.step(p=float(out_gcn.get("mu") or out_gcn.get("p_det") or 0.0), t_s=_t_s)
        out_gcn["triage"] = {
            "state": res.triage_state,
            "ps": res.ps,
            "p_in": res.p_in,
            "tau_low": float(cfg_gcn.get("tau_low", out_gcn.get("tau_low", 0.0))),
            "tau_high": float(cfg_gcn.get("tau_high", out_gcn.get("tau_high", 0.0))),
            "ema_alpha": float(cfg_gcn.get("ema_alpha", 0.0)),
            "k": int(cfg_gcn.get("k", 2)),
            "n": int(cfg_gcn.get("n", 3)),
            "cooldown_s": float(cfg_gcn.get("cooldown_s", 0.0)),
            "cooldown_remaining_s": res.cooldown_remaining_s,
        }
        started_gcn = bool(res.started_event)
        tri_gcn = res.triage_state

    if mode == "tcn":
        triage_state = tri_tcn or "not_fall"
        p_display = float(models_out.get("tcn", {}).get("mu", 0.0))
    elif mode == "gcn":
        triage_state = tri_gcn or "not_fall"
        p_display = float(models_out.get("gcn", {}).get("mu", 0.0))
    else:
        triage_state = _fuse_hybrid(tri_tcn=str(tri_tcn or "not_fall"), tri_gcn=str(tri_gcn or "not_fall"))
        mu_t = float(models_out.get("tcn", {}).get("mu", 0.0))
        mu_g = float(models_out.get("gcn", {}).get("mu", 0.0))
        sig_t = float(models_out.get("tcn", {}).get("sigma", 0.0))
        sig_g = float(models_out.get("gcn", {}).get("sigma", 0.0))
        models_out["hybrid"] = {
            "mu": float(min(mu_t, mu_g)),
            "sigma": float(max(sig_t, sig_g)),
            "triage": {"state": triage_state},
            "components": {"tcn": tri_tcn, "gcn": tri_gcn},
        }
        p_display = float(min(mu_t, mu_g))

    saved_event_id = None

    if mode == "tcn":
        started_event = started_tcn
    elif mode == "gcn":
        started_event = started_gcn
    else:
        prev_hyb = str(st.get("last_hybrid_state") or "not_fall")
        started_event = (triage_state == "fall") and (prev_hyb != "fall")
        st["last_hybrid_state"] = triage_state

    if persist and started_event and triage_state == "fall":
        meta = {
            "dataset": dataset_code,
            "mode": mode,
            "op_code": op_code,
            "use_mc": bool(use_mc),
            "mc_M": int(mc_M),
            "expected_fps": expected_fps,
            "capture_fps_est": cap_fps_est,
            "models": models_out,
        }
        try:
            with get_conn() as conn:
                if _table_exists(conn, "events"):
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO events (resident_id, type, severity, model_code, operating_point_id, score, meta) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                            (
                                resident_id,
                                "fall",
                                "high",
                                active_model_code,
                                None,
                                float(p_display),
                                json.dumps(meta),
                            ),
                        )
                        saved_event_id = cur.lastrowid
                    conn.commit()
        except (MySQLError, RuntimeError, OSError, TypeError, ValueError) as exc:
            logger.warning(
                "monitor.predict_window: failed to persist event (resident_id=%s, session_id=%s, mode=%s, dataset=%s): %s",
                resident_id,
                session_id,
                mode,
                dataset_code,
                exc,
            )

    latency_ms = int((time.time() - t0) * 1000)
    core.LAST_PRED_LATENCY_MS = latency_ms
    core.LAST_PRED_P_FALL = float(p_display)
    core.LAST_PRED_DECISION = str(triage_state)
    core.LAST_PRED_MODEL_CODE = str(active_model_code)
    core.LAST_PRED_TS_ISO = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

    return {
        "triage_state": triage_state,
        "models": models_out,
        "latency_ms": latency_ms,
        "capture_fps_est": cap_fps_est,
        "target_fps": expected_fps,
        "target_T": target_T,
        "dataset_code": dataset_code,
        "requested_mode": requested_mode,
        "effective_mode": mode,
        "op_code": op_code,
        "use_mc": bool(use_mc),
        "event_id": saved_event_id,
    }
