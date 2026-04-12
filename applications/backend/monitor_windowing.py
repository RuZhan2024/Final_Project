from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fall_detection.pose.preprocess_config import normalize_pose_preprocess_cfg
from fall_detection.pose.preprocess_pose_npz import (
    linear_fill_small_gaps,
    normalize_body_centric,
    smooth_weighted_moving_average,
    standardize_missing,
)


LIVE_EFFECTIVE_FPS_MIN = 10.0
MONITOR_Q_SCALE = 1000.0


def window_motion_score(xy: Any) -> Optional[float]:
    if not isinstance(xy, list) or len(xy) < 2:
        return None

    def torso_center_scale(frame: Any) -> Optional[Tuple[float, float, float]]:
        if not isinstance(frame, list):
            return None
        points: Dict[int, Tuple[float, float]] = {}
        for idx in (11, 12, 23, 24):
            if idx >= len(frame):
                continue
            point = frame[idx]
            if not isinstance(point, list) or len(point) < 2:
                continue
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                points[idx] = (x, y)
        if len(points) < 2:
            return None
        values = list(points.values())
        sx = sum(point[0] for point in values) / len(values)
        sy = sum(point[1] for point in values) / len(values)

        scales: List[float] = []
        if 11 in points and 12 in points:
            dx = points[12][0] - points[11][0]
            dy = points[12][1] - points[11][1]
            scales.append(math.hypot(dx, dy))
        if 23 in points and 24 in points:
            dx = points[24][0] - points[23][0]
            dy = points[24][1] - points[23][1]
            scales.append(math.hypot(dx, dy))
        if 11 in points and 23 in points:
            dx = points[23][0] - points[11][0]
            dy = points[23][1] - points[11][1]
            scales.append(math.hypot(dx, dy))
        if 12 in points and 24 in points:
            dx = points[24][0] - points[12][0]
            dy = points[24][1] - points[12][1]
            scales.append(math.hypot(dx, dy))

        scale = max(1e-6, (sum(scales) / len(scales)) if scales else 0.08)
        return (sx, sy, scale)

    speeds: List[float] = []
    for i in range(1, len(xy)):
        c0 = torso_center_scale(xy[i - 1])
        c1 = torso_center_scale(xy[i])
        if c0 is None or c1 is None:
            continue
        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        if math.isfinite(dx) and math.isfinite(dy):
            disp = math.hypot(dx, dy)
            scale = max(1e-6, 0.5 * (c0[2] + c1[2]))
            speeds.append(disp / scale)

    if not speeds:
        return None
    speeds.sort()
    mid = len(speeds) // 2
    if len(speeds) % 2 == 1:
        return float(speeds[mid])
    return float((speeds[mid - 1] + speeds[mid]) * 0.5)


def raw_window_stats(raw_t_ms: Any, raw_xy: Any, raw_conf: Any) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "raw_len": 0,
        "raw_fps_est": None,
        "raw_duration_s": None,
        "joints_per_frame_med": None,
        "conf_mean": None,
    }
    if not isinstance(raw_t_ms, list) or not isinstance(raw_xy, list):
        return stats
    n = min(len(raw_t_ms), len(raw_xy))
    if n <= 0:
        return stats
    stats["raw_len"] = int(n)

    t_vals: List[float] = []
    joint_counts: List[int] = []
    conf_vals: List[float] = []
    for i in range(n):
        try:
            t_vals.append(float(raw_t_ms[i]))
        except (TypeError, ValueError):
            continue
        frame = raw_xy[i]
        if isinstance(frame, list):
            joint_counts.append(len(frame))
        cfr = raw_conf[i] if isinstance(raw_conf, list) and i < len(raw_conf) else None
        if isinstance(cfr, list):
            for value in cfr:
                try:
                    fv = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv):
                    conf_vals.append(fv)

    if len(t_vals) >= 2:
        dt_s = (t_vals[-1] - t_vals[0]) / 1000.0
        if dt_s > 1e-6:
            stats["raw_duration_s"] = float(dt_s)
            stats["raw_fps_est"] = float((len(t_vals) - 1) / dt_s)

    if joint_counts:
        joint_counts.sort()
        stats["joints_per_frame_med"] = int(joint_counts[len(joint_counts) // 2])
    if conf_vals:
        stats["conf_mean"] = float(sum(conf_vals) / max(1, len(conf_vals)))
    return stats


def direct_window_stats(xy: Any, conf: Any, *, effective_fps: float) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "raw_len": 0,
        "raw_fps_est": None,
        "raw_duration_s": None,
        "joints_per_frame_med": None,
        "conf_mean": None,
    }
    if not isinstance(xy, list) or len(xy) <= 0:
        return stats

    n = len(xy)
    stats["raw_len"] = int(n)
    fps_eff = float(effective_fps) if effective_fps and float(effective_fps) > 0 else None
    if fps_eff is not None:
        stats["raw_fps_est"] = fps_eff
        stats["raw_duration_s"] = float((n - 1) / fps_eff) if n >= 2 else 0.0

    joint_counts: List[int] = []
    conf_vals: List[float] = []
    for i, frame in enumerate(xy):
        if isinstance(frame, list):
            joint_counts.append(len(frame))
        cfr = conf[i] if isinstance(conf, list) and i < len(conf) else None
        if isinstance(cfr, list):
            for value in cfr:
                try:
                    fv = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv):
                    conf_vals.append(fv)

    if joint_counts:
        joint_counts.sort()
        stats["joints_per_frame_med"] = int(joint_counts[len(joint_counts) // 2])
    if conf_vals:
        stats["conf_mean"] = float(sum(conf_vals) / max(1, len(conf_vals)))
    return stats


def decode_quantized_raw_window(
    raw_xy_q: Any,
    raw_conf_q: Any,
    raw_shape: Any,
) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    try:
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) < 2:
            return None, None
        n = int(raw_shape[0])
        j = int(raw_shape[1])
        if n <= 0 or j <= 0:
            return None, None
    except (TypeError, ValueError):
        return None, None

    if not isinstance(raw_xy_q, list) or len(raw_xy_q) != n * j * 2:
        return None, None
    if raw_conf_q is not None and (not isinstance(raw_conf_q, list) or len(raw_conf_q) != n * j):
        return None, None

    xy_out: List[Any] = []
    conf_out: List[Any] = []
    qi = 0
    ci = 0
    for _ in range(n):
        fxy: List[Any] = []
        fconf: List[Any] = []
        for _ in range(j):
            try:
                x = float(raw_xy_q[qi]) / MONITOR_Q_SCALE
                y = float(raw_xy_q[qi + 1]) / MONITOR_Q_SCALE
            except (TypeError, ValueError, IndexError):
                x = 0.0
                y = 0.0
            qi += 2
            fxy.append([x, y])
            if raw_conf_q is not None:
                try:
                    c = float(raw_conf_q[ci]) / MONITOR_Q_SCALE
                except (TypeError, ValueError, IndexError):
                    c = 0.0
                ci += 1
            else:
                c = 1.0
            fconf.append(c)
        xy_out.append(fxy)
        conf_out.append(fconf)
    return xy_out, conf_out


def window_quality_block(
    *,
    raw_stats: Dict[str, Any],
    expected_fps: float,
    effective_fps: float,
    target_T: int,
    dataset_code: str,
    live_guard_by_dataset: Dict[str, Dict[str, Any]],
    live_guard_global: Dict[str, Any],
    min_fps_ratio_override: Optional[float] = None,
    min_frames_ratio_override: Optional[float] = None,
    min_coverage_ratio_override: Optional[float] = None,
) -> Dict[str, Any]:
    need_s = max(1e-6, (max(2, int(target_T)) - 1) / max(1e-6, float(effective_fps)))
    raw_fps = raw_stats.get("raw_fps_est")
    raw_len = int(raw_stats.get("raw_len") or 0)
    raw_dur = raw_stats.get("raw_duration_s")

    fps_ratio = (float(raw_fps) / float(expected_fps)) if raw_fps is not None and expected_fps > 0 else None
    frame_ratio = (float(raw_len) / float(target_T)) if target_T > 0 else None
    coverage_ratio = (float(raw_dur) / float(need_s)) if raw_dur is not None and need_s > 0 else None

    ds_defaults = live_guard_by_dataset.get(dataset_code, {})
    min_fps_ratio = float(min_fps_ratio_override if min_fps_ratio_override is not None else ds_defaults.get("min_fps_ratio", 0.75))
    min_frames_ratio = float(
        min_frames_ratio_override if min_frames_ratio_override is not None else live_guard_global.get("min_frames_ratio", 0.60)
    )
    min_coverage_ratio = float(
        min_coverage_ratio_override
        if min_coverage_ratio_override is not None
        else live_guard_global.get("min_coverage_ratio", 0.85)
    )
    low_fps = bool(fps_ratio is not None and fps_ratio < min_fps_ratio)
    low_frames = bool(frame_ratio is not None and frame_ratio < min_frames_ratio)
    low_coverage = bool(coverage_ratio is not None and coverage_ratio < min_coverage_ratio)

    return {
        "need_duration_s": float(need_s),
        "fps_ratio": float(fps_ratio) if fps_ratio is not None else None,
        "frame_ratio": float(frame_ratio) if frame_ratio is not None else None,
        "coverage_ratio": float(coverage_ratio) if coverage_ratio is not None else None,
        "min_fps_ratio": float(min_fps_ratio),
        "min_frames_ratio": float(min_frames_ratio),
        "min_coverage_ratio": float(min_coverage_ratio),
        "low_fps": low_fps,
        "low_frames": low_frames,
        "low_coverage": low_coverage,
        "low_quality_block": bool(low_fps or low_frames or low_coverage),
    }


def effective_target_fps(*, expected_fps: float, raw_fps_est: Optional[float]) -> float:
    exp = float(expected_fps) if expected_fps and float(expected_fps) > 0 else 23.0
    if raw_fps_est is None or not math.isfinite(float(raw_fps_est)) or float(raw_fps_est) <= 0:
        return exp
    raw = float(raw_fps_est)
    lo = min(LIVE_EFFECTIVE_FPS_MIN, exp)
    hi = exp
    return float(max(lo, min(raw, hi)))


def resolve_runtime_fps(
    *,
    dataset_code: str,
    payload_d: Dict[str, Any],
    raw_fps_est: Optional[float],
    is_replay: bool,
) -> Tuple[float, float]:
    expected_fps = {
        "le2i": 25,
        "caucafall": 23,
    }.get(dataset_code, int(payload_d.get("target_fps") or payload_d.get("fps") or 23))
    if is_replay and raw_fps_est is not None and math.isfinite(float(raw_fps_est)) and float(raw_fps_est) > 0:
        measured_fps = float(raw_fps_est)
        return measured_fps, measured_fps
    effective_fps = (
        float(expected_fps)
        if is_replay
        else effective_target_fps(expected_fps=float(expected_fps), raw_fps_est=raw_fps_est)
    )
    return float(expected_fps), float(effective_fps)


def preprocess_online_raw_window(
    xy: Any,
    conf: Any,
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Any], List[Any]]:
    if not isinstance(xy, list) or len(xy) <= 0:
        return [], []

    xy_np = np.asarray(xy, dtype=np.float32)
    conf_np = np.asarray(conf, dtype=np.float32) if isinstance(conf, list) and len(conf) == len(xy) else None
    if conf_np is None:
        conf_np = np.ones((xy_np.shape[0], xy_np.shape[1]), dtype=np.float32)

    if xy_np.ndim != 3 or xy_np.shape[-1] != 2 or conf_np.ndim != 2:
        return xy, conf if isinstance(conf, list) else []

    cfg_norm = normalize_pose_preprocess_cfg(cfg)
    xy_np, conf_np = standardize_missing(xy_np, conf_np)
    xy_np, conf_np, _ = linear_fill_small_gaps(
        xy_np,
        conf_np,
        conf_thr=float(cfg_norm["conf_thr"]),
        max_gap=int(cfg_norm["max_gap"]),
        fill_conf=str(cfg_norm["fill_conf"]),
    )
    xy_np = smooth_weighted_moving_average(
        xy_np,
        conf_np,
        conf_thr=float(cfg_norm["conf_thr"]),
        k=int(cfg_norm["smooth_k"]),
    )
    xy_np, _ = normalize_body_centric(
        xy_np,
        conf_np,
        conf_thr=float(cfg_norm["conf_thr"]),
        mode=str(cfg_norm["normalize"]),
        pelvis_fill=str(cfg_norm["pelvis_fill"]),
        rotate=str(cfg_norm["rotate"]),
    )
    xy_np = np.nan_to_num(xy_np, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    conf_np = np.nan_to_num(conf_np, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return xy_np.tolist(), conf_np.tolist()


def resample_pose_window(
    *,
    raw_t_ms: Any,
    raw_xy: Any,
    raw_conf: Any = None,
    target_fps: float = 30.0,
    target_T: int = 48,
    window_end_t_ms: Optional[float] = None,
) -> Tuple[List[Any], List[Any], float, float, Optional[float]]:
    if not isinstance(raw_t_ms, list) or not isinstance(raw_xy, list) or len(raw_t_ms) != len(raw_xy) or len(raw_xy) < 1:
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
            conf_out.append(conf[0] if use_conf and isinstance(conf[0], list) else [1.0] * (len(xy[0]) if isinstance(xy[0], list) else 0))
            continue
        if tw >= t[-1]:
            xy_out.append(xy[-1])
            conf_out.append(conf[-1] if use_conf and isinstance(conf[-1], list) else [1.0] * (len(xy[-1]) if isinstance(xy[-1], list) else 0))
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
