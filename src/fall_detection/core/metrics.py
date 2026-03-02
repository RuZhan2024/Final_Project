#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/metrics.py

Threshold sweeps and window/event-level metrics helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


def prf_fpr_at_threshold(probs: np.ndarray, y_true: np.ndarray, thr: float) -> Dict[str, float]:
    p = np.asarray(probs).reshape(-1)
    y = np.asarray(y_true).reshape(-1).astype(np.int32)
    pred = (p >= float(thr)).astype(np.int32)

    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())

    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = (2.0 * P * R / (P + R)) if (P + R) > 0 else 0.0
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {"thr": float(thr), "precision": float(P), "recall": float(R), "f1": float(F1), "fpr": float(FPR)}


def sweep_thresholds(probs: np.ndarray, y_true: np.ndarray, thr_min: float = 0.05, thr_max: float = 0.95, thr_step: float = 0.01) -> Dict[str, List[float]]:
    thr_values = np.arange(float(thr_min), float(thr_max) + 1e-12, float(thr_step), dtype=np.float32)
    out = {"thr": [], "precision": [], "recall": [], "f1": [], "fpr": []}
    for thr in thr_values:
        m = prf_fpr_at_threshold(probs, y_true, float(thr))
        out["thr"].append(m["thr"])
        out["precision"].append(m["precision"])
        out["recall"].append(m["recall"])
        out["f1"].append(m["f1"])
        out["fpr"].append(m["fpr"])
    return out


def best_threshold_by_f1(probs: np.ndarray, y_true: np.ndarray, thr_min: float = 0.05, thr_max: float = 0.95, thr_step: float = 0.01) -> Dict[str, float]:
    sweep = sweep_thresholds(probs, y_true, thr_min, thr_max, thr_step)
    f1 = np.asarray(sweep["f1"], dtype=float)
    if f1.size == 0:
        return {"thr": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0, "fpr": 0.0}
    i = int(f1.argmax())
    return {
        "thr": float(sweep["thr"][i]),
        "f1": float(sweep["f1"][i]),
        "precision": float(sweep["precision"][i]),
        "recall": float(sweep["recall"][i]),
        "fpr": float(sweep["fpr"][i]),
    }


def ap_auc(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    p = np.asarray(probs).reshape(-1)
    y = np.asarray(y_true).reshape(-1)
    out = {"ap": float("nan"), "auc": float("nan")}
    if average_precision_score is not None:
        try:
            out["ap"] = float(average_precision_score(y, p))
        except Exception:
            pass
    if roc_auc_score is not None:
        try:
            # requires both classes present
            if np.unique(y).size >= 2:
                out["auc"] = float(roc_auc_score(y, p))
        except Exception:
            pass
    return out


# -------- FA/24h estimation (window -> event) --------

@dataclass
class SweepMeta:
    fps_default: float = 30.0
    stride_frames: Optional[int] = None  # if known, used for event merging
    fa24h_method: str = "approx"         # "pose_npz" | "approx"


def _group_fp_events_for_video(starts: np.ndarray, ends: np.ndarray, stride_frames: int) -> int:
    """Count false-positive *events* from predicted-positive windows (already filtered to y==0).

    If two predicted windows overlap in time or are separated by <= stride_frames,
    treat them as the same event.
    """
    if starts.size == 0:
        return 0
    order = np.argsort(starts)
    starts = starts[order]
    ends = ends[order]
    n_events = 1
    cur_end = int(ends[0])
    gap = int(stride_frames)
    for s, e in zip(starts[1:], ends[1:]):
        s = int(s); e = int(e)
        if s <= cur_end + gap:
            cur_end = max(cur_end, e)
        else:
            n_events += 1
            cur_end = e
    return int(n_events)


def sweep_with_fa24h(
    probs: np.ndarray,
    y_true: np.ndarray,
    video_ids: Sequence[str],
    w_start: Sequence[int],
    w_end: Sequence[int],
    fps: Sequence[float],
    *,
    thr_min: float = 0.05,
    thr_max: float = 0.95,
    thr_step: float = 0.01,
    pose_npz_dir: Optional[str] = None,
    stride_frames_hint: Optional[int] = None,
    fps_default: float = 30.0,
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """Return (sweep, meta) where sweep includes fa24h.

    Duration estimation:
      - if pose_npz_dir is provided, we load each sequence npz and use its true length.
      - otherwise, use (max_end-min_start+1)/fps per video.
    """
    p = np.asarray(probs).reshape(-1)
    y = np.asarray(y_true).reshape(-1).astype(np.int32)
    vids = np.asarray(list(video_ids))
    ws = np.asarray(w_start).astype(np.int64)
    we = np.asarray(w_end).astype(np.int64)
    fps_arr = np.asarray(fps).astype(np.float32)

    thr_values = np.arange(float(thr_min), float(thr_max) + 1e-12, float(thr_step), dtype=np.float32)

    out = {"thr": [], "precision": [], "recall": [], "f1": [], "fpr": [], "fa24h": []}

    # duration per video
    unique_vids = list(dict.fromkeys([str(v) for v in vids.tolist()]))
    duration_sec: Dict[str, float] = {}

    fa24h_method = "approx"
    if pose_npz_dir:
        import os, glob
        fa24h_method = "pose_npz"
        idx = {}
        for pp in glob.glob(os.path.join(pose_npz_dir, "**", "*.npz"), recursive=True):
            stem = os.path.splitext(os.path.basename(pp))[0]
            if stem not in idx:
                idx[stem] = pp
        for v in unique_vids:
            pp = idx.get(v)
            if not pp:
                continue
            try:
                with np.load(pp, allow_pickle=False) as z:
                    # most pose npz store xy/conf with first dim as frames
                    if "xy" in z.files:
                        T = int(np.array(z["xy"]).shape[0])
                    elif "conf" in z.files:
                        T = int(np.array(z["conf"]).shape[0])
                    else:
                        continue
                    fps_v = float(np.array(z["fps"]).reshape(-1)[0]) if "fps" in z.files else fps_default
                    duration_sec[v] = float(T) / max(1e-6, float(fps_v))
            except Exception:
                continue

    # fallback approx
    for v in unique_vids:
        if v in duration_sec:
            continue
        m = vids == v
        if not m.any():
            continue
        s0 = int(ws[m].min())
        e0 = int(we[m].max())
        fps_v = float(np.median(fps_arr[m])) if np.isfinite(fps_arr[m]).any() else float(fps_default)
        duration_sec[v] = float(e0 - s0 + 1) / max(1e-6, fps_v)

    total_sec = float(sum(duration_sec.values()))
    total_days = total_sec / 86400.0 if total_sec > 0 else float("nan")

    # stride estimate for event grouping
    stride_frames = int(stride_frames_hint) if (stride_frames_hint and stride_frames_hint > 0) else None
    if stride_frames is None:
        # estimate from median difference of consecutive starts across all windows
        if ws.size >= 2:
            d = np.diff(np.sort(ws))
            d = d[d > 0]
            stride_frames = int(np.median(d)) if d.size else 1
        else:
            stride_frames = 1

    # threshold loop
    for thr in thr_values:
        pred = (p >= float(thr)).astype(np.int32)
        # window-level metrics
        m = prf_fpr_at_threshold(p, y, float(thr))
        out["thr"].append(m["thr"])
        out["precision"].append(m["precision"])
        out["recall"].append(m["recall"])
        out["f1"].append(m["f1"])
        out["fpr"].append(m["fpr"])

        # fp events per video (only y==0)
        fp_events = 0
        for v in unique_vids:
            mv = vids == v
            if not mv.any():
                continue
            fp = mv & (y == 0) & (pred == 1)
            if not fp.any():
                continue
            fp_events += _group_fp_events_for_video(ws[fp], we[fp], stride_frames)

        fa24h = float(fp_events / total_days) if (total_days and total_days > 0) else float("nan")
        out["fa24h"].append(fa24h)

    meta = {
        "fa24h_method": fa24h_method,
        "total_duration_sec": total_sec,
        "total_videos": len(unique_vids),
        "stride_frames_used": stride_frames,
    }
    return out, meta


def pareto_frontier(recall: Sequence[float], x: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return indices of the Pareto frontier (minimise x, maximise recall)."""
    r = np.asarray(recall, dtype=float)
    xx = np.asarray(x, dtype=float)
    valid = np.isfinite(r) & np.isfinite(xx)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return np.asarray([], dtype=int), r, xx
    # sort by x ascending
    order = idx[np.argsort(xx[idx])]
    best_r = -1.0
    keep = []
    for i in order:
        if r[i] > best_r + 1e-12:
            keep.append(i)
            best_r = r[i]
    return np.asarray(keep, dtype=int), r, xx
