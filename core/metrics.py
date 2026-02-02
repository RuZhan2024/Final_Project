#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/metrics.py

Threshold sweeps + simple classification metrics used across the repo.

Why this module exists
----------------------
Many scripts in this repo need the same metric logic:
- training scripts (pick best threshold, report AP/AUC)
- evaluation scripts (threshold sweeps)
- operating-point fitting (choose thresholds meeting FA/day constraints)

If different scripts compute metrics differently, you can "optimize" a threshold
offline that doesn't behave the same during replay/deployment evaluation.

Key conventions
---------------
- We treat labels as binary:
    y = 1 -> fall
    y = 0 -> non-fall
  Any other value (e.g., -1 for unlabeled) is ignored by default.

- Probability:
    probs in [0,1], higher means "more likely fall"
  Prediction:
    pred = (probs >= threshold)

- FPR:
    false_positive_rate = FP / (FP + TN)

- AP and AUC use sklearn if available. If sklearn is not installed, we return NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


# ============================================================
# 1) Basic window-level metrics at ONE threshold
# ============================================================
def _filter_binary_labels(probs: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only samples where y is a valid binary label (0 or 1).

    Why:
    - Some parts of the pipeline can include y=-1 (unlabeled windows)
    - Some datasets might contain other numeric labels by accident
    - Metrics must be computed on a clean binary subset
    """
    p = np.asarray(probs).reshape(-1).astype(np.float32)
    y = np.asarray(y_true).reshape(-1).astype(np.int32)

    m = (y == 0) | (y == 1)
    if not np.any(m):
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.int32)

    return p[m], y[m]


def prf_fpr_at_threshold(probs: np.ndarray, y_true: np.ndarray, thr: float) -> Dict[str, float]:
    """
    Compute precision/recall/F1/FPR for a single threshold.

    Inputs:
      probs: predicted probabilities (higher => more likely fall)
      y_true: ground truth labels (0/1). Other values are ignored.
      thr: threshold in [0,1]

    Output keys:
      thr, precision, recall, f1, fpr

    Notes:
      - If there are no positives in predictions, precision becomes 0 by convention.
      - If there are no positives in ground truth, recall becomes 0 by convention.
    """
    p, y = _filter_binary_labels(probs, y_true)
    if p.size == 0:
        return {"thr": float(thr), "precision": 0.0, "recall": 0.0, "f1": 0.0, "fpr": 0.0}

    pred = (p >= float(thr)).astype(np.int32)

    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "thr": float(thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
    }


# ============================================================
# 2) Threshold sweeps + selecting best threshold
# ============================================================
def sweep_thresholds(
    probs: np.ndarray,
    y_true: np.ndarray,
    thr_min: float = 0.05,
    thr_max: float = 0.95,
    thr_step: float = 0.01,
) -> Dict[str, List[float]]:
    """
    Evaluate prf/fpr for many thresholds.

    Returns a dict of lists:
      {"thr": [...], "precision": [...], "recall": [...], "f1": [...], "fpr": [...]}
    """
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


def best_threshold_by_f1(
    probs: np.ndarray,
    y_true: np.ndarray,
    thr_min: float = 0.05,
    thr_max: float = 0.95,
    thr_step: float = 0.01,
) -> Dict[str, float]:
    """
    Find the threshold with the best F1 score.

    Tie-breaking (important!):
      1) higher F1
      2) higher recall        (prefer catching falls)
      3) lower FPR            (prefer fewer false alarms)
      4) lower threshold      (small preference; often gives earlier warning)

    This gives more stable behavior than argmax(F1) alone.
    """
    sweep = sweep_thresholds(probs, y_true, thr_min, thr_max, thr_step)
    if not sweep["thr"]:
        return {"thr": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0, "fpr": 0.0}

    thr = np.asarray(sweep["thr"], dtype=float)
    f1 = np.asarray(sweep["f1"], dtype=float)
    rec = np.asarray(sweep["recall"], dtype=float)
    fpr = np.asarray(sweep["fpr"], dtype=float)

    # Build a sorting key: we want max f1, max rec, min fpr, min thr
    # So we sort by (-f1, -rec, fpr, thr)
    order = np.lexsort((thr, fpr, -rec, -f1))
    i = int(order[0])

    return {
        "thr": float(sweep["thr"][i]),
        "f1": float(sweep["f1"][i]),
        "precision": float(sweep["precision"][i]),
        "recall": float(sweep["recall"][i]),
        "fpr": float(sweep["fpr"][i]),
    }


# ============================================================
# 3) Threshold-free metrics: AP / ROC-AUC
# ============================================================
def ap_auc(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """
    Compute AP (Average Precision) and ROC-AUC using sklearn if available.

    Important:
    - ROC-AUC requires BOTH classes present. If only one class is present,
      we return NaN for auc.
    - We ignore non-binary labels automatically.
    """
    p, y = _filter_binary_labels(probs, y_true)

    out = {"ap": float("nan"), "auc": float("nan")}
    if p.size == 0:
        return out

    if average_precision_score is not None:
        try:
            out["ap"] = float(average_precision_score(y, p))
        except Exception:
            pass

    if roc_auc_score is not None:
        try:
            if np.unique(y).size >= 2:
                out["auc"] = float(roc_auc_score(y, p))
        except Exception:
            pass

    return out


# ============================================================
# 4) False Alarms per 24 hours (FA/24h)
# ============================================================
@dataclass
class SweepMeta:
    """
    Metadata returned by sweep_with_fa24h.

    fps_default:
      fallback FPS if per-window fps is missing

    stride_frames:
      used for merging predicted-positive windows into "events"

    fa24h_method:
      - "pose_npz" if duration uses pose sequence NPZ true length
      - "approx"   if duration uses windows' min/max indices as approximation
    """

    fps_default: float = 30.0
    stride_frames: Optional[int] = None
    fa24h_method: str = "approx"


def _group_fp_events_for_video(starts: np.ndarray, ends: np.ndarray, stride_frames: int) -> int:
    """
    Count false-positive EVENTS from predicted-positive windows.

    Input assumption:
      starts/ends contain only windows that are:
        y_true == 0  AND  pred == 1   (i.e., false positive windows)

    Event grouping rule (simple, effective):
      If two positive windows overlap OR the gap between them is <= stride_frames,
      treat them as the same false alarm event.

    This approximates "alarm fatigue" better than counting FP windows.
    """
    if starts.size == 0:
        return 0

    order = np.argsort(starts)
    starts = starts[order].astype(np.int64)
    ends = ends[order].astype(np.int64)

    n_events = 1
    cur_end = int(ends[0])
    gap = int(max(0, stride_frames))

    for s, e in zip(starts[1:], ends[1:]):
        s = int(s)
        e = int(e)
        if s <= cur_end + gap:
            cur_end = max(cur_end, e)
        else:
            n_events += 1
            cur_end = e

    return int(n_events)


def _estimate_stride_frames_per_video(vids: np.ndarray, ws: np.ndarray) -> int:
    """
    Estimate stride (in frames) from window start indices.

    Why this improved version:
    - If you compute stride from all starts across all videos together,
      you can get huge gaps between videos that distort the median.
    - So we compute median stride within EACH video then take median of medians.
    """
    uniq = list(dict.fromkeys([str(v) for v in vids.tolist()]))
    medians: List[int] = []

    for v in uniq:
        m = vids == v
        if m.sum() < 2:
            continue
        s = np.sort(ws[m])
        d = np.diff(s)
        d = d[d > 0]
        if d.size:
            medians.append(int(np.median(d)))

    if medians:
        return max(1, int(np.median(np.asarray(medians))))
    return 1


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
    """
    Threshold sweep that also estimates False Alarms per 24 hours (FA/24h).

    Returns:
      (sweep, meta)

    sweep keys:
      thr, precision, recall, f1, fpr, fa24h

    Duration estimation:
      - If pose_npz_dir is given:
          load each sequence NPZ and use true length / fps for duration.
          (best accuracy if video_ids match NPZ stems)
      - Else:
          approximate duration per video as:
              (max(w_end) - min(w_start) + 1) / fps

    stride_frames:
      used to merge FP windows into FP events.
      If not provided, estimated from window starts per video.
    """
    # Filter to binary labels only
    p, y = _filter_binary_labels(probs, y_true)
    if p.size == 0:
        empty = {"thr": [], "precision": [], "recall": [], "f1": [], "fpr": [], "fa24h": []}
        meta = {"fa24h_method": "none", "total_duration_sec": 0.0, "total_videos": 0, "stride_frames_used": None}
        return empty, meta

    # IMPORTANT: When filtering labels, we must also filter all aligned arrays.
    vids_all = np.asarray(list(video_ids), dtype=object).reshape(-1)
    ws_all = np.asarray(w_start, dtype=np.int64).reshape(-1)
    we_all = np.asarray(w_end, dtype=np.int64).reshape(-1)
    fps_all = np.asarray(fps, dtype=np.float32).reshape(-1)

    # Build mask for "binary-labeled" samples in the original arrays.
    y0 = np.asarray(y_true).reshape(-1).astype(np.int32)
    m_bin = (y0 == 0) | (y0 == 1)

    vids = vids_all[m_bin].astype(object)
    ws = ws_all[m_bin]
    we = we_all[m_bin]
    fps_arr = fps_all[m_bin]

    # Threshold grid
    thr_values = np.arange(float(thr_min), float(thr_max) + 1e-12, float(thr_step), dtype=np.float32)

    out = {"thr": [], "precision": [], "recall": [], "f1": [], "fpr": [], "fa24h": []}

    # ------------------------------------------------------------
    # Duration per video
    # ------------------------------------------------------------
    unique_vids = list(dict.fromkeys([str(v) for v in vids.tolist()]))
    duration_sec: Dict[str, float] = {}

    fa24h_method = "approx"
    if pose_npz_dir:
        # Build index of pose npz stems -> file path (first match wins)
        import glob
        fa24h_method = "pose_npz"

        idx: Dict[str, str] = {}
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
                    # Most sequence NPZ store xy/conf with first dim = frames
                    if "xy" in z.files:
                        T = int(np.array(z["xy"]).shape[0])
                    elif "conf" in z.files:
                        T = int(np.array(z["conf"]).shape[0])
                    else:
                        continue
                    fps_v = float(np.array(z["fps"]).reshape(-1)[0]) if "fps" in z.files else float(fps_default)
                    duration_sec[v] = float(T) / max(1e-6, fps_v)
            except Exception:
                continue

    # Approx fallback (for any videos not covered by pose_npz)
    for v in unique_vids:
        if v in duration_sec:
            continue
        mv = np.array([str(x) == v for x in vids.tolist()], dtype=bool)
        if not mv.any():
            continue
        s0 = int(ws[mv].min())
        e0 = int(we[mv].max())
        fps_v = float(np.median(fps_arr[mv])) if np.isfinite(fps_arr[mv]).any() else float(fps_default)
        duration_sec[v] = float(e0 - s0 + 1) / max(1e-6, fps_v)

    total_sec = float(sum(duration_sec.values()))
    total_days = total_sec / 86400.0 if total_sec > 0 else float("nan")

    # ------------------------------------------------------------
    # Stride estimation (for FP event grouping)
    # ------------------------------------------------------------
    if stride_frames_hint and stride_frames_hint > 0:
        stride_frames = int(stride_frames_hint)
    else:
        stride_frames = _estimate_stride_frames_per_video(vids.astype(object), ws)

    # Pre-group per video to avoid repeated masks inside the threshold loop
    per_video = []
    for v in unique_vids:
        mv = np.array([str(x) == v for x in vids.tolist()], dtype=bool)
        if not mv.any():
            continue
        per_video.append(
            (
                v,
                p[mv],
                y[mv],
                ws[mv],
                we[mv],
            )
        )

    # ------------------------------------------------------------
    # Threshold loop
    # ------------------------------------------------------------
    for thr in thr_values:
        # window-level metrics
        m = prf_fpr_at_threshold(p, y, float(thr))
        out["thr"].append(m["thr"])
        out["precision"].append(m["precision"])
        out["recall"].append(m["recall"])
        out["f1"].append(m["f1"])
        out["fpr"].append(m["fpr"])

        # FP events (only y==0 predicted positive)
        fp_events = 0
        for _, pv, yv, sv, ev in per_video:
            pred = (pv >= float(thr))
            fp = (yv == 0) & pred
            if not np.any(fp):
                continue
            fp_events += _group_fp_events_for_video(sv[fp], ev[fp], stride_frames)

        fa24h = float(fp_events / total_days) if (total_days and total_days > 0) else float("nan")
        out["fa24h"].append(fa24h)

    meta = {
        "fa24h_method": fa24h_method,
        "total_duration_sec": total_sec,
        "total_videos": len(unique_vids),
        "stride_frames_used": stride_frames,
        "fps_default": float(fps_default),
    }
    return out, meta


# ============================================================
# 5) Pareto frontier helper
# ============================================================
def pareto_frontier(recall: Sequence[float], x: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Pareto frontier for:
      - maximize recall
      - minimize x (often FA/24h or FPR)

    Returns:
      keep_indices, recall_array, x_array

    keep_indices are indices i such that:
      there is no other j with (x[j] <= x[i]) AND (recall[j] >= recall[i])
      with at least one strict improvement.
    """
    r = np.asarray(recall, dtype=float)
    xx = np.asarray(x, dtype=float)

    valid = np.isfinite(r) & np.isfinite(xx)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return np.asarray([], dtype=int), r, xx

    # sort by x ascending
    order = idx[np.argsort(xx[idx])]

    best_r = -1.0
    keep: List[int] = []
    for i in order:
        if r[i] > best_r + 1e-12:
            keep.append(int(i))
            best_r = float(r[i])

    return np.asarray(keep, dtype=int), r, xx
