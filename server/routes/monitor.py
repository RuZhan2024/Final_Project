from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Body, HTTPException, Query

from .. import core
from ..core import _detect_variants, _ensure_system_settings_schema, _table_exists
from ..db import get_conn
from ..deploy_runtime import (
    fuse_hybrid as _fuse_hybrid,
    get_specs as _get_deploy_specs,
    predict_spec as _predict_spec,
)
from ..online_alert import OnlineAlertTracker


router = APIRouter()
logger = logging.getLogger(__name__)

_RUNTIME_DEFAULTS_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}
_RUNTIME_DEFAULTS_TTL_S = 2.0
_RESAMPLE_TW_CACHE: Dict[Tuple[int, int], np.ndarray] = {}
_RESAMPLE_TW_CACHE_MAX = 32
_MAX_RAW_SRC_FRAMES = 4096
_MAX_RAW_JOINTS = 64
_EXPECTED_FPS_BY_DATASET: Dict[str, int] = {
    "le2i": 25,
    "urfd": 30,
    "caucafall": 23,
    "muvim": 30,
}


def _compact_models_meta(models: Dict[str, Any]) -> Dict[str, Any]:
    """Keep persisted monitor-event payload small to reduce request-thread overhead."""
    out: Dict[str, Any] = {}
    if not isinstance(models, dict):
        return out
    for name, item in models.items():
        if not isinstance(item, dict):
            continue
        tri = item.get("triage") if isinstance(item.get("triage"), dict) else {}
        comp = item.get("components") if isinstance(item.get("components"), dict) else None
        out[str(name)] = {
            "spec_key": item.get("spec_key"),
            "dataset": item.get("dataset"),
            "arch": item.get("arch"),
            "mu": item.get("mu"),
            "sigma": item.get("sigma"),
            "p_det": item.get("p_det"),
            "mc_n_used": item.get("mc_n_used"),
            "triage_state": tri.get("state"),
            "components": comp,
        }
    return out


def _read_runtime_defaults_db(resident_id: int) -> Dict[str, Any]:
    """Read deploy defaults from DB once for monitor runtime."""
    out: Dict[str, Any] = {}
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

        if not isinstance(sys_row, dict):
            return out

        if sys_row.get("active_dataset_code"):
            out["dataset_code"] = str(sys_row.get("active_dataset_code") or "").lower()
        if sys_row.get("mc_enabled") is not None:
            mc_enabled_raw = sys_row.get("mc_enabled")
            out["use_mc"] = (
                bool(int(mc_enabled_raw))
                if str(mc_enabled_raw).isdigit()
                else bool(mc_enabled_raw)
            )
        if sys_row.get("mc_M") is not None:
            out["mc_M"] = int(sys_row.get("mc_M"))
        if sys_row.get("active_model_code"):
            out["active_model_code"] = str(sys_row.get("active_model_code") or "")
        if sys_row.get("active_op_code"):
            out["op_code"] = str(sys_row.get("active_op_code") or "").upper().strip()

        if "op_code" not in out:
            op_id = None
            for k in ("active_operating_point", "active_operating_point_id"):
                if sys_row.get(k) is not None:
                    op_id = sys_row.get(k)
                    break
            if op_id and _table_exists(conn, "operating_points"):
                with conn.cursor() as cur:
                    cur.execute("SELECT code FROM operating_points WHERE id=%s LIMIT 1", (int(op_id),))
                    r = cur.fetchone() or {}
                    if isinstance(r, dict) and r.get("code"):
                        out["op_code"] = str(r.get("code") or "").upper()

    return out


def _get_runtime_defaults_cached(resident_id: int) -> Dict[str, Any]:
    now = time.time()
    rec = _RUNTIME_DEFAULTS_CACHE.get(int(resident_id))
    if rec is not None:
        ts, payload = rec
        if (now - float(ts)) <= float(_RUNTIME_DEFAULTS_TTL_S):
            return dict(payload)
    payload = _read_runtime_defaults_db(int(resident_id))
    _RUNTIME_DEFAULTS_CACHE[int(resident_id)] = (now, dict(payload))
    return payload



def _resample_pose_window(
    *,
    raw_t_ms: Any,
    raw_xy: Any,
    raw_conf: Any = None,
    raw_xy_flat: Any = None,
    raw_conf_flat: Any = None,
    raw_joints: Optional[int] = None,
    target_fps: float = 30.0,
    target_T: int = 48,
    window_end_t_ms: Optional[float] = None,
    generate_default_conf: bool = True,
    prevalidated_raw: bool = False,
) -> Tuple[Any, Any, float, float, Optional[float]]:
    """Resample variable-FPS pose frames to a fixed FPS + fixed length window."""

    def _clip01_if_needed(a: np.ndarray, *, make_copy: bool = False) -> np.ndarray:
        if a.size < 1:
            return a.copy() if make_copy else a
        out = a.copy() if make_copy else a
        # Single-pass clip is typically faster than min/max pre-check scans.
        np.clip(out, 0.0, 1.0, out=out)
        return out

    try:
        raw_t_arr = np.asarray(raw_t_ms, dtype=np.float32).reshape(-1)
    except Exception:
        return [], [], 0.0, 0.0, None
    if int(raw_t_arr.size) < 1:
        return [], [], 0.0, 0.0, None
    raw_t_src = raw_t_arr

    keep_idx: Optional[np.ndarray] = None
    t_np: Optional[np.ndarray] = None
    xy_np_fast: Optional[np.ndarray] = None
    conf_np_fast: Optional[np.ndarray] = None
    use_conf = False

    # Compact payload path: flattened XY/CONF arrays from app.
    if raw_xy_flat is not None:
        try:
            t0 = raw_t_arr
            n_src = int(t0.size)
            n_joints = int(raw_joints) if raw_joints is not None else 33
            if n_src > 0 and n_joints > 0:
                if isinstance(raw_xy_flat, np.ndarray) and raw_xy_flat.dtype == np.float32:
                    xy_flat = raw_xy_flat.reshape(-1)
                else:
                    xy_flat = np.asarray(raw_xy_flat, dtype=np.float32).reshape(-1)
                if xy_flat.size == (n_src * n_joints * 2):
                    x0 = xy_flat.reshape(n_src, n_joints, 2)
                    conf_flat = None
                    conf0 = None
                    if raw_conf_flat is not None:
                        if isinstance(raw_conf_flat, np.ndarray) and raw_conf_flat.dtype == np.float32:
                            conf_flat = raw_conf_flat.reshape(-1)
                        else:
                            conf_flat = np.asarray(raw_conf_flat, dtype=np.float32).reshape(-1)
                        if conf_flat.size == (n_src * n_joints):
                            conf0 = conf_flat.reshape(n_src, n_joints)
                    if prevalidated_raw:
                        t_np = t0
                        xy_np_fast = x0
                        if conf0 is not None:
                            conf_np_fast = conf0
                            use_conf = True
                    elif n_src == 1:
                        if np.isfinite(t0[0]):
                            t_np = t0
                            xy_np_fast = x0
                            if conf0 is not None:
                                conf_np_fast = conf0
                                use_conf = True
                    else:
                        fin_t = np.isfinite(t0)
                        dt_t = np.diff(t0)
                        finite_all = bool(fin_t.all())
                        inc_all = bool(np.all(dt_t > 0.0))
                        if finite_all and inc_all:
                            t_np = t0
                            xy_np_fast = x0
                            if conf0 is not None:
                                conf_np_fast = conf0
                                use_conf = True
                        else:
                            keep = fin_t.copy()
                            keep[1:] &= dt_t > 0.0
                            if keep.any():
                                keep_idx = np.flatnonzero(keep).astype(np.int64, copy=False)
                                t_np = t0[keep]
                                xy_np_fast = x0[keep]
                                if conf0 is not None:
                                    conf_np_fast = conf0[keep]
                                    use_conf = True
        except Exception:
            t_np = None
            xy_np_fast = None
            conf_np_fast = None
            use_conf = False

    if isinstance(raw_conf, list) and isinstance(raw_xy, list) and len(raw_conf) == len(raw_xy):
        use_conf = True
    if xy_np_fast is None:
        try:
            t0 = raw_t_arr
            x0 = np.asarray(raw_xy, dtype=np.float32)
            if x0.ndim == 3 and x0.shape[0] == t0.shape[0] and x0.shape[2] >= 2:
                x0 = x0[..., :2]
                n_src = int(t0.size)
                c0 = None
                if use_conf:
                    c0 = np.asarray(raw_conf, dtype=np.float32)
                if prevalidated_raw:
                    t_np = t0
                    xy_np_fast = x0
                    if c0 is not None:
                        if c0.ndim == 2 and c0.shape[0] == t0.shape[0]:
                            conf_np_fast = c0
                        else:
                            conf_np_fast = None
                            use_conf = False
                elif n_src == 1:
                    if np.isfinite(t0[0]):
                        t_np = t0
                        xy_np_fast = x0
                        if c0 is not None:
                            if c0.ndim == 2 and c0.shape[0] == t0.shape[0]:
                                conf_np_fast = c0
                            else:
                                conf_np_fast = None
                                use_conf = False
                    else:
                        t_np = None
                else:
                    fin_t = np.isfinite(t0)
                    dt_t = np.diff(t0)
                    finite_all = bool(fin_t.all())
                    inc_all = bool(np.all(dt_t > 0.0))
                    if finite_all and inc_all:
                        t_np = t0
                        xy_np_fast = x0
                        if c0 is not None:
                            if c0.ndim == 2 and c0.shape[0] == t0.shape[0]:
                                conf_np_fast = c0
                            else:
                                conf_np_fast = None
                                use_conf = False
                    else:
                        keep = fin_t.copy()
                        keep[1:] &= dt_t > 0.0
                        if keep.any():
                            keep_idx = np.flatnonzero(keep).astype(np.int64, copy=False)
                            t_np = t0[keep]
                            xy_np_fast = x0[keep]
                            if c0 is not None:
                                if c0.ndim == 2 and c0.shape[0] == t0.shape[0]:
                                    conf_np_fast = c0[keep]
                                else:
                                    conf_np_fast = None
                                    use_conf = False
                        else:
                            t_np = None
        except Exception:
            t_np = None

    if t_np is None or xy_np_fast is None or t_np.size < 1:
        if (
            not isinstance(raw_xy, list)
            or len(raw_xy) < 1
            or int(raw_t_arr.size) != len(raw_xy)
        ):
            return [], [], 0.0, 0.0, None
        use_conf = bool(isinstance(raw_conf, list) and len(raw_conf) == len(raw_xy))
        t: List[float] = []
        xy: List[Any] = []
        keep_src_idx: List[int] = []
        conf: Optional[List[Any]] = [] if use_conf else None
        last_t: Optional[float] = None
        if use_conf:
            for src_i, (ti, xyi, ci) in enumerate(zip(raw_t_src, raw_xy, raw_conf)):
                try:
                    tf = float(ti)
                except Exception:
                    continue
                if last_t is not None and tf <= last_t:
                    continue
                if not isinstance(xyi, list):
                    continue
                t.append(tf)
                xy.append(xyi)
                keep_src_idx.append(int(src_i))
                if conf is not None:
                    conf.append(ci)
                last_t = tf
        else:
            for src_i, (ti, xyi) in enumerate(zip(raw_t_src, raw_xy)):
                try:
                    tf = float(ti)
                except Exception:
                    continue
                if last_t is not None and tf <= last_t:
                    continue
                if not isinstance(xyi, list):
                    continue
                t.append(tf)
                xy.append(xyi)
                keep_src_idx.append(int(src_i))
                last_t = tf
        if len(t) < 1:
            return [], [], 0.0, 0.0, None
        keep_idx = np.asarray(keep_src_idx, dtype=np.int64)
        t_np = np.asarray(t, dtype=np.float32)
        try:
            xy_np_fast = np.asarray(xy, dtype=np.float32)
            if xy_np_fast.ndim == 3 and xy_np_fast.shape[2] >= 2:
                xy_np_fast = xy_np_fast[..., :2]
            else:
                xy_np_fast = None
        except Exception:
            xy_np_fast = None
        if use_conf and conf is not None and conf:
            try:
                conf0 = np.asarray(conf, dtype=np.float32)
                conf_np_fast = conf0 if conf0.ndim == 2 else None
            except Exception:
                conf_np_fast = None
            if conf_np_fast is None:
                use_conf = False
        else:
            conf_np_fast = None

    cap_fps: Optional[float] = None
    if t_np.size >= 2:
        dt_s = float((t_np[-1] - t_np[0]) / 1000.0)
        if dt_s > 1e-6:
            cap_fps = float((t_np.size - 1) / dt_s)

    target_fps = float(target_fps) if target_fps and float(target_fps) > 0 else 30.0
    target_T = int(target_T) if target_T and int(target_T) > 1 else 48
    dt_ms = 1000.0 / target_fps

    end_t_ms = float(window_end_t_ms) if window_end_t_ms is not None else float(t_np[-1])
    start_t_ms = end_t_ms - (target_T - 1) * dt_ms

    # Fast path: fixed-shape arrays (normal app flow). This avoids nested Python loops.
    try:
        xy_np = xy_np_fast if xy_np_fast is not None else np.asarray(raw_xy, dtype=np.float32)
        if xy_np.ndim != 3 or xy_np.shape[2] < 2:
            raise ValueError("xy must be [N,J,2] for vectorized resampling")
        xy_np = xy_np[..., :2]
        xy_np = np.nan_to_num(xy_np, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        n_src, n_joints, _ = xy_np.shape
        if n_src < 1 or n_joints < 1:
            return [], [], start_t_ms, end_t_ms, cap_fps
        conf_np: Optional[np.ndarray] = None
        if use_conf:
            conf_np = conf_np_fast if conf_np_fast is not None else np.asarray(raw_conf, dtype=np.float32)
            if conf_np.ndim != 2:
                raise ValueError("conf must be [N,J] for vectorized resampling")
            if conf_np.shape[0] != n_src:
                raise ValueError("conf length mismatch")
            if conf_np.shape[1] != n_joints:
                conf_np = conf_np[:, :n_joints]
            conf_np = np.nan_to_num(conf_np, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        start_t32 = np.float32(start_t_ms)
        fps_key = int(round(target_fps * 1000.0))
        cache_key = (int(target_T), fps_key)
        tw_offsets = _RESAMPLE_TW_CACHE.get(cache_key)
        if tw_offsets is None:
            tw_offsets = (np.arange(target_T, dtype=np.float32) * np.float32(dt_ms))
            if len(_RESAMPLE_TW_CACHE) >= _RESAMPLE_TW_CACHE_MAX:
                # Keep cache bounded for long-lived processes.
                _RESAMPLE_TW_CACHE.pop(next(iter(_RESAMPLE_TW_CACHE)))
            _RESAMPLE_TW_CACHE[cache_key] = tw_offsets
        tw = start_t32 + tw_offsets
        # Fast path: source frames already align with the target time grid.
        if n_src == target_T:
            tol_ms = max(1e-3, 0.25 * dt_ms)
            if np.all(np.abs(t_np - tw) <= tol_ms):
                xy_out_np = _clip01_if_needed(xy_np, make_copy=False)
                if use_conf:
                    assert conf_np is not None
                    conf_out_np = _clip01_if_needed(conf_np, make_copy=False)
                else:
                    conf_out_np = np.ones((target_T, n_joints), dtype=np.float32) if generate_default_conf else None
                return (
                    xy_out_np,
                    conf_out_np,
                    start_t_ms,
                    end_t_ms,
                    cap_fps,
                )
        if n_src == 1:
            xy_out_np = np.broadcast_to(xy_np[0], (target_T, n_joints, 2)).astype(np.float32, copy=True)
            xy_out_np = _clip01_if_needed(xy_out_np, make_copy=False)
            if use_conf:
                assert conf_np is not None
                conf_src = conf_np[0]
                if conf_src.shape[0] != n_joints:
                    conf_src = conf_src[:n_joints]
                conf_out_np = np.broadcast_to(conf_src, (target_T, n_joints)).astype(np.float32, copy=True)
                conf_out_np = _clip01_if_needed(conf_out_np, make_copy=False)
            else:
                conf_out_np = np.ones((target_T, n_joints), dtype=np.float32) if generate_default_conf else None
            return xy_out_np, conf_out_np, start_t_ms, end_t_ms, cap_fps

        # Fully vectorized linear interpolation for [T,J,2] and [T,J] using source-time indices.
        right = np.searchsorted(t_np, tw, side="left")
        np.clip(right, 1, n_src - 1, out=right)
        left = right - 1
        t_left = t_np[left]
        t_right = t_np[right]
        denom = np.maximum(t_right - t_left, 1e-9)
        alpha = ((tw - t_left) / denom).astype(np.float32, copy=False)

        xy_left = xy_np[left]   # [target_T,J,2]
        xy_right = xy_np[right] # [target_T,J,2]
        xy_out_np = xy_left + (xy_right - xy_left) * alpha[:, None, None]
        xy_out_np = _clip01_if_needed(xy_out_np, make_copy=False)

        if use_conf:
            assert conf_np is not None
            conf_left = conf_np[left]   # [target_T,J]
            conf_right = conf_np[right] # [target_T,J]
            conf_out_np = conf_left + (conf_right - conf_left) * alpha[:, None]
            conf_out_np = _clip01_if_needed(conf_out_np, make_copy=False)
        else:
            conf_out_np = np.ones((target_T, n_joints), dtype=np.float32) if generate_default_conf else None

        return (
            xy_out_np,
            conf_out_np,
            start_t_ms,
            end_t_ms,
            cap_fps,
        )
    except Exception:
        # Fallback: robust interpolation for ragged/malformed frames.
        idx = keep_idx if keep_idx is not None else np.arange(int(raw_t_arr.size), dtype=np.int64)
        n_idx = int(idx.size)
        t: List[float] = [0.0] * n_idx
        xy: List[Any] = [None] * n_idx
        conf: Optional[List[Any]] = ([None] * n_idx) if use_conf else None
        n_keep = 0
        for ii_raw in idx:
            ii = int(ii_raw)
            try:
                ti = float(raw_t_arr[ii])
            except Exception:
                continue
            t[n_keep] = ti
            xy[n_keep] = raw_xy[ii]
            if conf is not None:
                conf[n_keep] = raw_conf[ii]
            n_keep += 1
        if n_keep < 1:
            return [], [], start_t_ms, end_t_ms, cap_fps
        if n_keep != n_idx:
            t = t[:n_keep]
            xy = xy[:n_keep]
            if conf is not None:
                conf = conf[:n_keep]
        j = 0
        target_T_i = int(target_T)
        xy_out: List[Any] = [None] * target_T_i
        need_conf_out = bool(generate_default_conf or (conf is not None))
        conf_out: Optional[List[Any]] = ([None] * target_T_i) if need_conf_out else None
        n_t = len(t)
        t0_first = float(t[0])
        t_last = float(t[-1])
        xy_first = xy[0]
        xy_last = xy[-1]
        conf_first = conf[0] if conf is not None else None
        conf_last = conf[-1] if conf is not None else None
        ones_first = [1.0] * (len(xy_first) if isinstance(xy_first, list) else 0) if generate_default_conf else None
        ones_last = [1.0] * (len(xy_last) if isinstance(xy_last, list) else 0) if generate_default_conf else None
        ones_cache: Dict[int, List[float]] = {}
        dt_ms_f = float(dt_ms)

        def lerp(a: float, b: float, alpha: float) -> float:
            return a + (b - a) * alpha

        def interp_frame(frame0: Any, frame1: Any, alpha: float, is_xy: bool) -> Any:
            if not isinstance(frame0, list):
                return frame1
            if not isinstance(frame1, list):
                return frame0
            if alpha <= 0.0:
                return frame0
            if alpha >= 1.0:
                return frame1
            if is_xy:
                n = min(len(frame0), len(frame1))
                out = [[0.0, 0.0] for _ in range(n)]
                for k in range(n):
                    p0 = frame0[k]
                    p1 = frame1[k]
                    try:
                        x0 = float(p0[0]) if isinstance(p0, list) and len(p0) >= 2 else 0.0
                        y0 = float(p0[1]) if isinstance(p0, list) and len(p0) >= 2 else 0.0
                        x1 = float(p1[0]) if isinstance(p1, list) and len(p1) >= 2 else 0.0
                        y1 = float(p1[1]) if isinstance(p1, list) and len(p1) >= 2 else 0.0
                    except Exception:
                        x0 = y0 = x1 = y1 = 0.0
                    out[k][0] = lerp(x0, x1, alpha)
                    out[k][1] = lerp(y0, y1, alpha)
                return out

            n = min(len(frame0), len(frame1))
            out = [0.0] * n
            for k in range(n):
                try:
                    v0 = float(frame0[k])
                except Exception:
                    v0 = 0.0
                try:
                    v1 = float(frame1[k])
                except Exception:
                    v1 = v0
                out[k] = lerp(v0, v1, alpha)
            return out

        def _ones_for_len(n: int) -> Optional[List[float]]:
            if not generate_default_conf:
                return None
            nn = int(n) if int(n) > 0 else 0
            out = ones_cache.get(nn)
            if out is None:
                out = [1.0] * nn
                ones_cache[nn] = out
            return out

        tw = float(start_t_ms)
        for k in range(target_T_i):

            if tw <= t0_first:
                xy_out[k] = xy_first
                if conf_out is not None:
                    conf_out[k] = conf_first if conf is not None and isinstance(conf_first, list) else (
                        ones_first
                    )
                tw += dt_ms_f
                continue
            if tw >= t_last:
                xy_out[k] = xy_last
                if conf_out is not None:
                    conf_out[k] = conf_last if conf is not None and isinstance(conf_last, list) else (
                        ones_last
                    )
                tw += dt_ms_f
                continue

            while j + 1 < n_t and t[j + 1] < tw:
                j += 1

            t0, t1 = t[j], t[j + 1]
            alpha = 0.0 if t1 <= t0 else (tw - t0) / (t1 - t0)

            xy_interp = interp_frame(xy[j], xy[j + 1], alpha, True)
            xy_out[k] = xy_interp
            if conf_out is not None:
                if conf is not None and isinstance(conf[j], list) and isinstance(conf[j + 1], list):
                    conf_out[k] = interp_frame(conf[j], conf[j + 1], alpha, False)
                else:
                    conf_out[k] = _ones_for_len(len(xy_interp) if isinstance(xy_interp, list) else 0)
            tw += dt_ms_f

        # Robustly pack potentially ragged fallback frames to dense arrays.
        n_joints_out = None
        for fr in xy_out:
            if not isinstance(fr, list):
                continue
            lfr = len(fr)
            if n_joints_out is None or lfr < n_joints_out:
                n_joints_out = lfr
        if n_joints_out is None:
            return [], [], start_t_ms, end_t_ms, cap_fps
        n_joints_out = max(int(n_joints_out), 0)

        xy_out_np = np.zeros((target_T_i, n_joints_out, 2), dtype=np.float32)
        for ti, fr in enumerate(xy_out):
            if not isinstance(fr, list):
                continue
            upto = min(n_joints_out, len(fr))
            row_xy = xy_out_np[ti]
            for ji in range(upto):
                p = fr[ji]
                if not (isinstance(p, list) and len(p) >= 2):
                    continue
                try:
                    row_xy[ji, 0] = float(p[0])
                    row_xy[ji, 1] = float(p[1])
                except Exception:
                    continue
        xy_out_np = _clip01_if_needed(xy_out_np, make_copy=False)

        if conf_out is None:
            conf_out_np = None
        else:
            default_conf = 1.0 if generate_default_conf else 0.0
            conf_out_np = np.full((target_T_i, n_joints_out), np.float32(default_conf), dtype=np.float32)
            for ti, fr in enumerate(conf_out):
                if not isinstance(fr, list):
                    continue
                upto = min(n_joints_out, len(fr))
                row_conf = conf_out_np[ti]
                for ji in range(upto):
                    try:
                        row_conf[ji] = float(fr[ji])
                    except Exception:
                        continue
            conf_out_np = _clip01_if_needed(conf_out_np, make_copy=False)

        return (
            xy_out_np,
            conf_out_np,
            start_t_ms,
            end_t_ms,
            cap_fps,
        )


@router.post("/api/monitor/reset_session")
def reset_session(session_id: str = Query(...)) -> Dict[str, Any]:
    core._SESSION_STATE.pop(session_id, None)
    return {"ok": True, "session_id": session_id}


@router.post("/api/monitor/predict_window")
def predict_window(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Score one window from the live monitor UI."""

    t0 = time.time()

    def _as_int(
        v: Any,
        *,
        field: str,
        default: int,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> int:
        if v is None:
            out = int(default)
        else:
            try:
                out = int(v)
            except Exception:
                raise HTTPException(status_code=400, detail=f"{field} must be an integer")
        if min_value is not None and out < min_value:
            raise HTTPException(status_code=400, detail=f"{field} must be >= {min_value}")
        if max_value is not None and out > max_value:
            raise HTTPException(status_code=400, detail=f"{field} must be <= {max_value}")
        return out

    # -------------
    # Inputs
    # -------------
    session_id = str(payload.get("session_id") or "default")

    mode = str(payload.get("mode") or "hybrid").lower().strip()
    if mode in {"hyb", "hybrid", "dual"}:
        mode = "dual"
    elif mode not in {"tcn", "gcn"}:
        mode = "dual"

    dataset_code = str(payload.get("dataset_code") or payload.get("dataset") or "").lower().strip()
    op_code = str(payload.get("op_code") or payload.get("op") or "").upper().strip()

    use_mc = payload.get("use_mc")
    mc_M = payload.get("mc_M")
    mc_sigma_tol_raw = payload.get("mc_sigma_tol")
    mc_se_tol_raw = payload.get("mc_se_tol")

    persist = bool(payload.get("persist", False))

    target_T = _as_int(payload.get("target_T"), field="target_T", default=48, min_value=2, max_value=512)
    raw_xy = payload.get("raw_xy")
    raw_conf = payload.get("raw_conf")
    raw_xy_flat = payload.get("raw_xy_flat")
    raw_conf_flat = payload.get("raw_conf_flat")
    raw_joints = payload.get("raw_joints")
    raw_t_ms = payload.get("raw_t_ms")
    window_end_t_ms = payload.get("window_end_t_ms", None)
    raw_xy_flat_arr: Optional[np.ndarray] = None
    raw_conf_flat_arr: Optional[np.ndarray] = None

    if raw_joints is not None:
        raw_joints_i = _as_int(raw_joints, field="raw_joints", default=33, min_value=1, max_value=_MAX_RAW_JOINTS)
    else:
        raw_joints_i = None
    if isinstance(raw_t_ms, list):
        try:
            raw_t_ms_arr = np.asarray(raw_t_ms, dtype=np.float32).reshape(-1)
        except Exception:
            raise HTTPException(status_code=400, detail="raw_t_ms must be a numeric array")
        n_src = int(raw_t_ms_arr.size)
        if n_src > _MAX_RAW_SRC_FRAMES:
            raise HTTPException(
                status_code=413,
                detail=f"raw_t_ms too long ({n_src}); max {_MAX_RAW_SRC_FRAMES} source frames",
            )
        if not bool(np.isfinite(raw_t_ms_arr).all()):
            raise HTTPException(status_code=400, detail="raw_t_ms contains non-finite values")
        if n_src > 1 and not bool(np.all(raw_t_ms_arr[1:] > raw_t_ms_arr[:-1])):
            raise HTTPException(status_code=400, detail="raw_t_ms must be strictly increasing")
        raw_t_ms = raw_t_ms_arr
        if raw_xy_flat is not None and raw_joints_i is not None:
            try:
                if isinstance(raw_xy_flat, np.ndarray) and raw_xy_flat.dtype == np.float32:
                    raw_xy_flat_arr = raw_xy_flat.reshape(-1)
                else:
                    raw_xy_flat_arr = np.asarray(raw_xy_flat, dtype=np.float32).reshape(-1)
                flat_len = int(raw_xy_flat_arr.size)
                max_flat = int(n_src * raw_joints_i * 2)
                if flat_len > max_flat:
                    raise HTTPException(
                        status_code=413,
                        detail=f"raw_xy_flat too long ({flat_len}); expected <= {max_flat}",
                    )
                if flat_len != max_flat:
                    raise HTTPException(
                        status_code=400,
                        detail=f"raw_xy_flat length mismatch ({flat_len}); expected exactly {max_flat}",
                    )
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(status_code=400, detail="raw_xy_flat must be a flat numeric array")
        if raw_conf_flat is not None and raw_joints_i is not None:
            try:
                if isinstance(raw_conf_flat, np.ndarray) and raw_conf_flat.dtype == np.float32:
                    raw_conf_flat_arr = raw_conf_flat.reshape(-1)
                else:
                    raw_conf_flat_arr = np.asarray(raw_conf_flat, dtype=np.float32).reshape(-1)
                conf_flat_len = int(raw_conf_flat_arr.size)
                max_conf_flat = int(n_src * raw_joints_i)
                if conf_flat_len > max_conf_flat:
                    raise HTTPException(
                        status_code=413,
                        detail=f"raw_conf_flat too long ({conf_flat_len}); expected <= {max_conf_flat}",
                    )
                if conf_flat_len != max_conf_flat:
                    raise HTTPException(
                        status_code=400,
                        detail=f"raw_conf_flat length mismatch ({conf_flat_len}); expected exactly {max_conf_flat}",
                    )
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(status_code=400, detail="raw_conf_flat must be a flat numeric array")

    xy: Optional[np.ndarray] = None
    conf: Optional[np.ndarray] = None
    cap_fps_est: Optional[float] = None
    used_raw_path = False
    sanitized_from_resample = False

    # -------------
    # Defaults from DB (if available)
    # -------------
    resident_id = _as_int(payload.get("resident_id"), field="resident_id", default=1, min_value=1)
    active_model_code = "HYBRID" if mode == "dual" else mode.upper()
    need_db_defaults = (not dataset_code) or (not op_code) or (use_mc is None) or (mc_M is None)
    if need_db_defaults:
        try:
            db_defaults = _get_runtime_defaults_cached(resident_id)
            if not dataset_code and db_defaults.get("dataset_code"):
                dataset_code = str(db_defaults.get("dataset_code") or "").lower()
            if use_mc is None and ("use_mc" in db_defaults):
                use_mc = bool(db_defaults.get("use_mc"))
            if mc_M is None and ("mc_M" in db_defaults):
                mc_M = _as_int(db_defaults.get("mc_M"), field="mc_M", default=10, min_value=1, max_value=200)
            if db_defaults.get("active_model_code"):
                active_model_code = str(db_defaults.get("active_model_code") or active_model_code)
            if (not op_code) and db_defaults.get("op_code"):
                op_code = str(db_defaults.get("op_code") or "").upper().strip()
        except Exception:
            logger.debug("Falling back to payload/default monitor settings (DB unavailable)", exc_info=True)

    if not dataset_code:
        dataset_code = "muvim"
    if not op_code:
        op_code = "OP-2"
    if use_mc is None:
        use_mc = True
    if mc_M is None:
        mc_M = 10
    mc_M = _as_int(mc_M, field="mc_M", default=10, min_value=1, max_value=200)
    mc_sigma_tol: Optional[float] = None
    if mc_sigma_tol_raw is not None:
        try:
            mc_sigma_tol = float(mc_sigma_tol_raw)
            if not (mc_sigma_tol > 0.0):
                mc_sigma_tol = None
        except Exception:
            raise HTTPException(status_code=400, detail="mc_sigma_tol must be a positive number")
    mc_se_tol: Optional[float] = None
    if mc_se_tol_raw is not None:
        try:
            mc_se_tol = float(mc_se_tol_raw)
            if not (mc_se_tol > 0.0):
                mc_se_tol = None
        except Exception:
            raise HTTPException(status_code=400, detail="mc_se_tol must be a positive number")

    expected_fps = _EXPECTED_FPS_BY_DATASET.get(
        dataset_code,
        int(payload.get("target_fps") or payload.get("fps") or 30),
    )

    if raw_t_ms is not None and (raw_xy is not None or raw_xy_flat is not None):
        used_raw_path = True
        has_raw_conf = (raw_conf_flat is not None) or (isinstance(raw_conf, list) and len(raw_conf) > 0)
        xy, conf, _, _, cap_fps_est = _resample_pose_window(
            raw_t_ms=raw_t_ms,
            raw_xy=raw_xy,
            raw_conf=raw_conf,
            raw_xy_flat=raw_xy_flat_arr if raw_xy_flat_arr is not None else raw_xy_flat,
            raw_conf_flat=raw_conf_flat_arr if raw_conf_flat_arr is not None else raw_conf_flat,
            raw_joints=raw_joints_i,
            target_fps=float(expected_fps),
            target_T=target_T,
            window_end_t_ms=float(window_end_t_ms) if window_end_t_ms is not None else None,
            generate_default_conf=False,
            prevalidated_raw=True,
        )
        sanitized_from_resample = isinstance(xy, np.ndarray) and (conf is None or isinstance(conf, np.ndarray))
        # If source stream did not provide confidence, keep conf=None so the ML
        # pipeline can use its no-conf fast/default path without carrying dense
        # synthetic confidence tensors through inference.
        if not has_raw_conf:
            conf = None

    if not (isinstance(xy, np.ndarray) and int(xy.size) > 0):
        xy_in = payload.get("xy") or []
        conf_in = payload.get("conf") or []
        try:
            xy = np.asarray(xy_in, dtype=np.float32)
        except Exception:
            raise HTTPException(status_code=400, detail="xy must be a numeric array")
        if conf_in:
            try:
                conf = np.asarray(conf_in, dtype=np.float32)
            except Exception:
                raise HTTPException(status_code=400, detail="conf must be a numeric array")
        else:
            conf = None

    if not (isinstance(xy, np.ndarray) and int(xy.size) > 0):
        raise HTTPException(status_code=400, detail="payload must include raw_* (preferred) or xy")
    if xy.ndim != 3 or int(xy.shape[2]) < 2:
        raise HTTPException(status_code=400, detail="xy must be shaped [T,J,2]")
    xy = xy[..., :2]
    if not sanitized_from_resample:
        xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        np.clip(xy, 0.0, 1.0, out=xy)
    if conf is not None:
        if conf.ndim != 2:
            raise HTTPException(status_code=400, detail="conf must be shaped [T,J]")
        t_xy = int(xy.shape[0])
        j_xy = int(xy.shape[1])
        if conf.shape != (t_xy, j_xy):
            if int(conf.size) == int(t_xy * j_xy):
                conf = conf.reshape(t_xy, j_xy)
            else:
                raise HTTPException(status_code=400, detail="conf shape mismatch for xy")
        if not sanitized_from_resample:
            conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            np.clip(conf, 0.0, 1.0, out=conf)
    if not used_raw_path and int(xy.shape[0]) != int(target_T):
        raise HTTPException(
            status_code=400,
            detail=f"xy time length mismatch ({int(xy.shape[0])}); expected target_T={int(target_T)}",
        )

    expected_fps_f = float(expected_fps)
    use_mc_b = bool(use_mc)
    mc_M_i = int(mc_M)

    try:
        if window_end_t_ms is not None:
            _now_ms = float(window_end_t_ms)
        elif raw_t_ms is not None and len(raw_t_ms) > 0:
            _now_ms = float(raw_t_ms[-1])
        else:
            _now_ms = time.time() * 1000.0
    except Exception:
        _now_ms = time.time() * 1000.0
    _t_s = float(_now_ms) / 1000.0

    core.prune_session_state(now_s=_t_s)
    st = core.touch_session_state(session_id, now_s=_t_s)
    st_trackers = st.setdefault("trackers", {})
    st_trackers_cfg = st.setdefault("trackers_cfg", {})
    started_tcn = False
    started_gcn = False

    # -------------
    # Inference
    # -------------
    specs = _get_deploy_specs()

    def spec_key_for(arch: str) -> str:
        return f"{dataset_code}_{arch}".lower()

    tcn_key = str(payload.get("model_tcn") or spec_key_for("tcn")).lower()
    gcn_key = str(payload.get("model_gcn") or spec_key_for("gcn")).lower()
    if tcn_key not in specs or gcn_key not in specs:
        raise HTTPException(status_code=404, detail=f"No deploy specs found for dataset '{dataset_code}'.")

    models_out: Dict[str, Any] = {}
    tri_tcn = None
    tri_gcn = None
    feature_cache: Optional[Dict[Any, Any]] = {} if mode == "dual" else None
    def _predict_spec_safe(*, spec_key_local: str) -> Dict[str, Any]:
        try:
            return _predict_spec(
                spec_key=spec_key_local,
                joints_xy=xy,
                conf=conf,
                fps=expected_fps_f,
                target_T=target_T,
                op_code=op_code,
                use_mc=use_mc_b,
                mc_M=mc_M_i,
                mc_sigma_tol=mc_sigma_tol,
                mc_se_tol=mc_se_tol,
                feature_cache=feature_cache,
                assume_sanitized_inputs=True,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"invalid inference input: {e}") from e

    if mode in {"tcn", "dual"}:
        out_tcn = _predict_spec_safe(spec_key_local=tcn_key)

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
        out_gcn = _predict_spec_safe(spec_key_local=gcn_key)
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
            "mc_M": mc_M_i,
            "mc_sigma_tol": mc_sigma_tol,
            "mc_se_tol": mc_se_tol,
            "expected_fps": expected_fps,
            "capture_fps_est": cap_fps_est,
            "models": _compact_models_meta(models_out),
        }
        try:
            with get_conn() as conn:
                if _table_exists(conn, "events"):
                    cols = core._cols(conn, "events")
                    if "type" not in cols:
                        raise RuntimeError("events table missing required `type` column")

                    model_id = None
                    if "model_id" in cols:
                        model_id = core._resolve_model_id(conn, active_model_code)

                    op_id = None
                    if "operating_point_id" in cols and _table_exists(conn, "operating_points"):
                        with conn.cursor() as cur:
                            if model_id is not None:
                                cur.execute(
                                    "SELECT id FROM operating_points WHERE model_id=%s AND code=%s LIMIT 1",
                                    (int(model_id), str(op_code).upper()),
                                )
                                orow = cur.fetchone() or {}
                                if isinstance(orow, dict):
                                    op_id = orow.get("id")
                            if op_id is None:
                                cur.execute(
                                    "SELECT id FROM operating_points WHERE code=%s ORDER BY id ASC LIMIT 1",
                                    (str(op_code).upper(),),
                                )
                                orow = cur.fetchone() or {}
                                if isinstance(orow, dict):
                                    op_id = orow.get("id")

                    time_col = core._event_time_col(conn)
                    prob_col = core._event_prob_col(conn)
                    insert_cols: List[str] = ["resident_id", "type"]
                    params: List[Any] = [resident_id, "fall"]

                    def add_if(col: str, val: Any) -> None:
                        if col in cols:
                            insert_cols.append(col)
                            params.append(val)

                    add_if(time_col, datetime.utcnow())
                    add_if("severity", "high")
                    add_if("status", "unreviewed")
                    add_if("model_code", active_model_code)
                    add_if("model_id", model_id)
                    add_if("operating_point_id", op_id)
                    if prob_col is not None:
                        add_if(prob_col, float(p_display))
                    meta_json = None
                    if "meta" in cols or "payload_json" in cols:
                        try:
                            meta_json = json.dumps(meta, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
                        except Exception:
                            # Fallback for unexpected non-finite values in telemetry payloads.
                            meta_json = json.dumps(meta, separators=(",", ":"), ensure_ascii=False)
                    if "meta" in cols:
                        add_if("meta", meta_json)
                    elif "payload_json" in cols:
                        add_if("payload_json", meta_json)

                    with conn.cursor() as cur:
                        cur.execute(
                            f"INSERT INTO events ({', '.join('`'+c+'`' for c in insert_cols)}) "
                            f"VALUES ({', '.join(['%s'] * len(insert_cols))})",
                            tuple(params),
                        )
                        saved_event_id = cur.lastrowid
                    conn.commit()
        except Exception:
            logger.warning(
                "Failed to persist monitor event (resident_id=%s, mode=%s, dataset=%s)",
                resident_id,
                mode,
                dataset_code,
                exc_info=True,
            )

    latency_ms = int((time.time() - t0) * 1000)
    core.LAST_PRED_LATENCY_MS = latency_ms
    core.LAST_PRED_P_FALL = float(p_display)
    core.LAST_PRED_DECISION = str(triage_state)
    core.LAST_PRED_MODEL_CODE = str(active_model_code)
    core.LAST_PRED_TS_ISO = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    mc_n_used_map = {
        k: int(v.get("mc_n_used"))
        for k, v in models_out.items()
        if isinstance(v, dict) and v.get("mc_n_used") is not None
    }

    return {
        "triage_state": triage_state,
        "models": models_out,
        "mc_n_used": mc_n_used_map,
        "latency_ms": latency_ms,
        "capture_fps_est": cap_fps_est,
        "target_fps": expected_fps,
        "target_T": target_T,
        "dataset_code": dataset_code,
        "op_code": op_code,
        "use_mc": use_mc_b,
        "mc_sigma_tol": mc_sigma_tol,
        "mc_se_tol": mc_se_tol,
        "event_id": saved_event_id,
    }
