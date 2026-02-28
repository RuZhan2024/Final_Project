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
    if isinstance(probs, np.ndarray) and probs.ndim == 1:
        p = probs
    else:
        p = np.asarray(probs).reshape(-1)
    if isinstance(y_true, np.ndarray) and y_true.ndim == 1 and y_true.dtype == np.int32:
        y = y_true
    else:
        y = np.asarray(y_true).reshape(-1).astype(np.int32, copy=False)
    pred = (p >= float(thr))
    y1 = (y == 1)
    y0 = (y == 0)

    tp = float((pred & y1).sum())
    fp = float((pred & y0).sum())
    fn = float(((~pred) & y1).sum())
    tn = float(((~pred) & y0).sum())

    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = (2.0 * P * R / (P + R)) if (P + R) > 0 else 0.0
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {"thr": float(thr), "precision": float(P), "recall": float(R), "f1": float(F1), "fpr": float(FPR)}


def sweep_thresholds(probs: np.ndarray, y_true: np.ndarray, thr_min: float = 0.05, thr_max: float = 0.95, thr_step: float = 0.01) -> Dict[str, List[float]]:
    if isinstance(probs, np.ndarray) and probs.ndim == 1:
        p = probs
    else:
        p = np.asarray(probs).reshape(-1)
    if isinstance(y_true, np.ndarray) and y_true.ndim == 1 and y_true.dtype == np.int32:
        y = y_true
    else:
        y = np.asarray(y_true).reshape(-1).astype(np.int32, copy=False)
    thr_values = np.arange(float(thr_min), float(thr_max) + 1e-12, float(thr_step), dtype=np.float32)
    if p.size == 0 or thr_values.size == 0:
        return {"thr": [], "precision": [], "recall": [], "f1": [], "fpr": []}

    y1 = (y == 1).astype(np.int64, copy=False)
    y0 = (y == 0).astype(np.int64, copy=False)
    pos_total = int(y1.sum())
    neg_total = int(y0.sum())

    # Sort once by score ascending, then use cumulative counts and searchsorted.
    # This avoids allocating [N,K] boolean matrices during sweeps.
    order = np.argsort(p, kind="mergesort")
    p_asc = p[order]
    y1_asc = y1[order]
    y0_asc = y0[order]
    tp_suf = np.cumsum(y1_asc[::-1], dtype=np.int64)[::-1]
    fp_suf = np.cumsum(y0_asc[::-1], dtype=np.int64)[::-1]

    # idx_left[i] = first index with p >= thr_values[i].
    idx_left = np.searchsorted(p_asc, thr_values, side="left").astype(np.int64, copy=False)
    has_pos = idx_left < p_asc.size
    tp = np.zeros_like(thr_values, dtype=np.float64)
    fp = np.zeros_like(thr_values, dtype=np.float64)
    if has_pos.any():
        idx = idx_left[has_pos]
        tp[has_pos] = tp_suf[idx]
        fp[has_pos] = fp_suf[idx]
    fn = float(pos_total) - tp
    tn = float(neg_total) - fp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    fpr = np.divide(fp, fp + tn, out=np.zeros_like(tp), where=(fp + tn) > 0)

    return {
        "thr": thr_values.astype(float).tolist(),
        "precision": precision.astype(float).tolist(),
        "recall": recall.astype(float).tolist(),
        "f1": f1.astype(float).tolist(),
        "fpr": fpr.astype(float).tolist(),
    }


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
    if isinstance(probs, np.ndarray) and probs.ndim == 1:
        p = probs
    else:
        p = np.asarray(probs).reshape(-1)
    if isinstance(y_true, np.ndarray) and y_true.ndim == 1:
        y = y_true
    else:
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


def _group_fp_events_for_video(
    starts: np.ndarray,
    ends: np.ndarray,
    stride_frames: int,
    *,
    assume_sorted: bool = False,
) -> int:
    """Count false-positive *events* from predicted-positive windows (already filtered to y==0).

    If two predicted windows overlap in time or are separated by <= stride_frames,
    treat them as the same event.
    """
    if starts.size == 0:
        return 0
    if not assume_sorted:
        order = np.argsort(starts)
        starts = starts[order]
        ends = ends[order]
    if starts.size == 1:
        return 1

    if isinstance(starts, np.ndarray) and starts.dtype == np.int64:
        starts_i = starts
    else:
        starts_i = starts.astype(np.int64, copy=False)
    if isinstance(ends, np.ndarray) and ends.dtype == np.int64:
        ends_i = ends
    else:
        ends_i = ends.astype(np.int64, copy=False)
    gap = int(stride_frames)

    # A new event starts at i>0 iff start_i is beyond the running merged end of prior windows.
    prev_merged_end = np.maximum.accumulate(ends_i[:-1])
    new_event = starts_i[1:] > (prev_merged_end + gap)
    return int(1 + int(new_event.sum()))


def _group_fp_events_for_video_from_scores(
    p_sorted: np.ndarray,
    starts_sorted: np.ndarray,
    ends_sorted: np.ndarray,
    thr: float,
    stride_frames: int,
) -> int:
    """Count FP events for sorted windows without allocating masked ws/we slices."""
    if p_sorted.size == 0:
        return 0
    if starts_sorted.size != p_sorted.size or ends_sorted.size != p_sorted.size:
        raise ValueError("p_sorted, starts_sorted, ends_sorted must have the same length")

    gap = int(stride_frames)
    thr_f = float(thr)
    active = (p_sorted >= thr_f)
    n_active = int(np.count_nonzero(active))
    if n_active == 0:
        return 0
    if n_active == 1:
        return 1
    if n_active == int(active.size):
        return _group_fp_events_for_video(
            starts_sorted,
            ends_sorted,
            gap,
            assume_sorted=True,
        )
    starts_sel = starts_sorted[active]
    ends_sel = ends_sorted[active]
    if isinstance(starts_sel, np.ndarray) and starts_sel.dtype == np.int64:
        starts_a = starts_sel
    else:
        starts_a = starts_sel.astype(np.int64, copy=False)
    if isinstance(ends_sel, np.ndarray) and ends_sel.dtype == np.int64:
        ends_a = ends_sel
    else:
        ends_a = ends_sel.astype(np.int64, copy=False)
    if starts_a.size == 1:
        return 1
    prev_merged_end = np.maximum.accumulate(ends_a[:-1])
    new_event = starts_a[1:] > (prev_merged_end + gap)
    return int(1 + int(new_event.sum()))


def _build_video_groups(vids: np.ndarray, ws: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Precompute per-video sorted indices once."""
    vids_arr = np.asarray(vids).reshape(-1)
    ws_arr = np.asarray(ws)
    if vids_arr.size < 1:
        return []

    uniq, first_idx, inv = np.unique(vids_arr, return_index=True, return_inverse=True)
    uniq_order = np.argsort(first_idx, kind="mergesort")
    inv_i64 = inv.astype(np.int64, copy=False)
    by_group = np.argsort(inv_i64, kind="mergesort")
    counts = np.bincount(inv_i64, minlength=uniq.size)
    offsets = np.empty((counts.size + 1,), dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    groups: List[Tuple[str, np.ndarray]] = []
    for u in uniq_order:
        uu = int(u)
        a = int(offsets[uu])
        b = int(offsets[uu + 1])
        idx = by_group[a:b]
        if idx.size < 1:
            continue
        sort_idx = np.argsort(ws_arr[idx], kind="mergesort")
        groups.append((str(uniq[u]), idx[sort_idx]))
    return groups


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
    if isinstance(probs, np.ndarray) and probs.ndim == 1:
        p = probs
    else:
        p = np.asarray(probs).reshape(-1)
    if isinstance(y_true, np.ndarray) and y_true.ndim == 1 and y_true.dtype == np.int32:
        y = y_true
    else:
        y = np.asarray(y_true).reshape(-1).astype(np.int32, copy=False)
    vids = np.asarray(video_ids)
    if isinstance(w_start, np.ndarray) and w_start.ndim == 1 and w_start.dtype == np.int64:
        ws = w_start
    else:
        ws = np.asarray(w_start).astype(np.int64, copy=False)
    if isinstance(w_end, np.ndarray) and w_end.ndim == 1 and w_end.dtype == np.int64:
        we = w_end
    else:
        we = np.asarray(w_end).astype(np.int64, copy=False)
    if isinstance(fps, np.ndarray) and fps.ndim == 1 and fps.dtype == np.float32:
        fps_arr = fps
    else:
        fps_arr = np.asarray(fps).astype(np.float32, copy=False)

    thr_values = np.arange(float(thr_min), float(thr_max) + 1e-12, float(thr_step), dtype=np.float32)
    base = sweep_thresholds(p, y, thr_min=float(thr_min), thr_max=float(thr_max), thr_step=float(thr_step))
    out = {
        "thr": list(base["thr"]),
        "precision": list(base["precision"]),
        "recall": list(base["recall"]),
        "f1": list(base["f1"]),
        "fpr": list(base["fpr"]),
        "fa24h": [],
    }

    # duration per video
    groups = _build_video_groups(vids, ws)
    unique_vids = [v for v, _idx in groups]
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
    for v, idx_full in groups:
        if v in duration_sec:
            continue
        if idx_full.size < 1:
            continue
        s0 = int(ws[idx_full].min())
        e0 = int(we[idx_full].max())
        fps_slice = fps_arr[idx_full]
        fps_v = float(np.median(fps_slice)) if np.isfinite(fps_slice).any() else float(fps_default)
        if fps_v <= 0:
            fps_v = float(fps_default)
        duration_sec[v] = float(e0 - s0 + 1) / max(1e-6, fps_v)

    total_sec = float(sum(duration_sec.values()))
    total_days = total_sec / 86400.0 if total_sec > 0 else float("nan")
    inv_total_days = (86400.0 / total_sec) if total_sec > 0 else float("nan")

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

    # Cache per-video negative windows once (already sorted by w_start).
    neg_groups: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]] = []
    empty_p = np.asarray([], dtype=np.float32)
    empty_w = np.asarray([], dtype=np.int64)
    for _v, idx_full in groups:
        if idx_full.size < 1:
            neg_groups.append(
                (
                    empty_p,
                    empty_w,
                    empty_w,
                    float("-inf"),
                    float("inf"),
                    0,
                )
            )
            continue
        idx_neg = idx_full[y[idx_full] == 0]
        if idx_neg.size < 1:
            neg_groups.append(
                (
                    empty_p,
                    empty_w,
                    empty_w,
                    float("-inf"),
                    float("inf"),
                    0,
                )
            )
            continue
        p_neg = p[idx_neg].astype(np.float32, copy=False)
        ws_neg = ws[idx_neg].astype(np.int64, copy=False)
        we_neg = we[idx_neg].astype(np.int64, copy=False)
        full_events = _group_fp_events_for_video(
            ws_neg,
            we_neg,
            stride_frames,
            assume_sorted=True,
        )
        neg_groups.append((p_neg, ws_neg, we_neg, float(np.max(p_neg)), float(np.min(p_neg)), int(full_events)))

    stride_i = int(stride_frames)
    # threshold loop
    for thr in thr_values:
        thr_f = float(thr)

        # fp events per video (only y==0)
        fp_events = 0
        for p_neg, ws_neg, we_neg, p_max, p_min, full_events in neg_groups:
            if p_neg.size < 1:
                continue
            if p_max < thr_f:
                continue
            if thr_f <= p_min:
                fp_events += int(full_events)
                continue
            fp_events += _group_fp_events_for_video_from_scores(
                p_neg,
                ws_neg,
                we_neg,
                thr_f,
                stride_i,
            )

        fa24h = float(fp_events * inv_total_days) if np.isfinite(inv_total_days) else float("nan")
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
