#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval/metrics.py

Evaluate a trained checkpoint on a windows directory, producing a JSON report.

This version evaluates REAL deployment behavior:
  - Threshold sweep is under the FULL alert policy (EMA + k-of-n + hysteresis + cooldown)
  - FA/24h counts FALSE alert events only (alerts not overlapping GT fall events)

Also supports evaluating OP-1/OP-2/OP-3 operating points.

IMPORTANT CONVENTION:
  - w_end is INCLUSIVE (last frame index of the window).
  - Any duration computed from window indices MUST use:
        duration_s = (w_end - w_start + 1) / fps
  - For timestamps used in alert policy / event merging, we use core.alerting.times_from_windows:
        center time = (w_start + w_end) / 2 / fps
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import yaml
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.alerting import (
    AlertCfg,
    times_from_windows,
    event_metrics_from_windows,
    gt_events_from_windows,
    sweep_alert_policy_from_windows,
    detect_alert_events_from_smoothed,
    ema_smooth,
)
from core.ckpt import get_cfg, load_ckpt
from core.features import FeatCfg, read_window_npz, build_tcn_input, build_canonical_input, split_gcn_two_stream
from core.confirm import confirm_scores_window
from core.models import build_model, logits_1d, pick_device, validate_model_input_dims
from core.metrics import ap_auc



from dataclasses import asdict, is_dataclass

def _event_metrics_to_compat_dict(em: Any) -> Dict[str, Any]:
    """Coerce core.alerting.EventMetrics (dataclass) or legacy dict into a JSON-safe dict.

    Also adds legacy alias keys expected by older eval/plot code:
      - recall <- event_recall
      - fa_per_day/fa24h <- false_alerts_per_day
      - fa_per_hour <- false_alerts_per_hour
      - precision <- event_precision
      - f1 <- event_f1
    """
    if em is None:
        d: Dict[str, Any] = {}
    elif isinstance(em, dict):
        d = dict(em)
    elif hasattr(em, "to_dict") and callable(getattr(em, "to_dict")):
        try:
            d = dict(em.to_dict())  # type: ignore[attr-defined]
        except Exception:
            d = dict(getattr(em, "__dict__", {}))
    elif is_dataclass(em):
        d = asdict(em)
    else:
        d = dict(getattr(em, "__dict__", {}))

    # aliases (best-effort)
    if "event_recall" in d and "recall" not in d:
        d["recall"] = d.get("event_recall")
    if "false_alerts_per_day" in d:
        if "fa_per_day" not in d:
            d["fa_per_day"] = d.get("false_alerts_per_day")
        if "fa24h" not in d:
            d["fa24h"] = d.get("false_alerts_per_day")
    if "false_alerts_per_hour" in d and "fa_per_hour" not in d:
        d["fa_per_hour"] = d.get("false_alerts_per_hour")
    if "event_precision" in d and "precision" not in d:
        d["precision"] = d.get("event_precision")
    if "event_f1" in d and "f1" not in d:
        d["f1"] = d.get("event_f1")

    # make JSON-safe scalars
    for k, v in list(d.items()):
        if isinstance(v, (np.floating, np.integer)):
            d[k] = v.item()
    return d


def _em_get(em: Any, key: str, default: Any = None) -> Any:
    """Safe getter for EventMetrics dataclass or dict.

    Works for:
      - dict (possibly legacy keys)
      - core.alerting.EventMetrics dataclass

    Also supports legacy alias keys used in older eval/plot code.
    """
    if em is None:
        return default

    aliases = {
        'recall': 'event_recall',
        'fa_per_day': 'false_alerts_per_day',
        'fa24h': 'false_alerts_per_day',
        'fa_per_hour': 'false_alerts_per_hour',
        'precision': 'event_precision',
        'f1': 'event_f1',
    }

    # dict-like
    if isinstance(em, dict):
        if key in em:
            return em.get(key)
        alt = aliases.get(key)
        if alt is not None and alt in em:
            return em.get(alt)
        return default

    # object / dataclass attributes
    if hasattr(em, key):
        return getattr(em, key)
    alt = aliases.get(key)
    if alt is not None and hasattr(em, alt):
        return getattr(em, alt)
    return default


def _build_video_groups(vids_arr: np.ndarray, ws_arr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Precompute per-video sorted window indices.

    This avoids repeated O(N) masking/sorting work across threshold sweeps.
    """
    vids_obj = np.asarray(vids_arr, dtype=object).reshape(-1)
    if vids_obj.size < 1:
        return []
    ws = np.asarray(ws_arr)
    uniq, first_idx, inv = np.unique(vids_obj, return_index=True, return_inverse=True)
    order = np.argsort(first_idx, kind="mergesort")
    inv_i64 = inv.astype(np.int64, copy=False)
    by_group = np.argsort(inv_i64, kind="mergesort")
    counts = np.bincount(inv_i64, minlength=uniq.size)
    offsets = np.empty((counts.size + 1,), dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    groups: List[Tuple[str, np.ndarray]] = []
    for u in order:
        uu = int(u)
        idx = by_group[int(offsets[uu]) : int(offsets[uu + 1])]
        if idx.size < 1:
            continue
        sort_idx = np.argsort(ws[idx], kind="mergesort")
        groups.append((str(uniq[u]), idx[sort_idx]))
    return groups


def _prepare_video_cache(
    probs: np.ndarray,
    vids_arr: np.ndarray,
    ws_arr: np.ndarray,
    we_arr: np.ndarray,
    fps_arr: np.ndarray,
    y_true: np.ndarray,
    *,
    fps_default: float,
    time_mode: str,
    merge_gap_s: float,
    ls_arr: Optional[np.ndarray] = None,
    ms_arr: Optional[np.ndarray] = None,
    video_groups: Optional[List[Tuple[str, np.ndarray]]] = None,
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, float, float, Optional[np.ndarray], Optional[np.ndarray], List[Tuple[float, float]]]]:
    groups = video_groups if video_groups is not None else _build_video_groups(vids_arr, ws_arr)
    cache = []
    for v, idx_full in groups:
        if idx_full.size < 1:
            continue
        p_v = probs[idx_full]
        y_v = y_true[idx_full]
        ws_v = ws_arr[idx_full]
        we_v = we_arr[idx_full]
        ls_v = ls_arr[idx_full] if ls_arr is not None else None
        ms_v = ms_arr[idx_full] if ms_arr is not None else None
        fps_slice = fps_arr[idx_full]
        fps_v = float(np.median(fps_slice)) if np.isfinite(fps_slice).any() else float(fps_default)
        if fps_v <= 0:
            fps_v = float(fps_default)
        t_v = times_from_windows(ws_v, we_v, fps_v, mode=time_mode)
        duration_s = float((we_v.max() - ws_v.min() + 1) / max(1e-6, fps_v))
        gt_v = gt_events_from_windows(t_v, y_v, merge_gap_s=float(merge_gap_s))
        cache.append((v, p_v, y_v, t_v, duration_s, fps_v, ls_v, ms_v, gt_v))
    return cache


# ---------------- ops yaml helpers ----------------

def _load_ops_yaml(path: str) -> dict:
    """Best-effort loader for fit_ops-style YAML files.

    Supports a few common schemas:
      - {policy: {...}, ops: {op1: {...}, op2: {...}, op3: {...}}}
      - {op1: {...}, op2: {...}, op3: {...}, policy: {...}}
      - {operating_points: [{name: op1, ...}, ...], policy: {...}}

    Missing keys are OK; caller should fall back to CLI defaults.
    """
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _extract_policy_and_ops(d: dict) -> tuple[dict, dict]:
    """Return (policy_dict, ops_dict) from a variety of YAML schemas.

    Supported top-level shapes:
      - {policy: {...}, ops: {op1: {...}, op2: {...}, op3: {...}}}
      - {alert_cfg: {...}, ops: {OP1: {...}, OP2: {...}, OP3: {...}}}   (legacy fit_ops output)
      - {operating_points: [{name: OP1, ...}, ...]}
      - {op1: {...}, op2: {...}, op3: {...}}
    Keys OP1/OP2/OP3 are treated case-insensitively.
    """
    policy: dict = {}
    ops: dict = {}

    if not isinstance(d, dict):
        return policy, ops

    # Policy schema: prefer explicit policy, otherwise fall back to legacy alert_cfg/alert_base
    if isinstance(d.get("policy"), dict):
        policy = dict(d.get("policy"))
    elif isinstance(d.get("alert_cfg"), dict):
        policy = dict(d.get("alert_cfg"))
    elif isinstance(d.get("alert_base"), dict):
        policy = dict(d.get("alert_base"))

    # ops can be under 'ops' or directly as op1/op2/op3. Accept OP1/OP2/OP3.
    def _maybe_add_ops(name: str, blob):
        if not isinstance(blob, dict):
            return
        n = str(name).strip().lower()
        if n in {"op1", "op2", "op3"}:
            ops[n] = dict(blob)

    if isinstance(d.get("ops"), dict):
        for k, v in d.get("ops", {}).items():
            _maybe_add_ops(k, v)
    else:
        for k in ("op1", "op2", "op3", "OP1", "OP2", "OP3"):  # tolerate case
            if isinstance(d.get(k), dict):
                _maybe_add_ops(k, d.get(k))

    # list-style schema
    if isinstance(d.get("operating_points"), list):
        for item in d.get("operating_points", []):
            if not isinstance(item, dict):
                continue
            name = item.get("name", item.get("id", ""))
            _maybe_add_ops(str(name), item)

    return policy, ops


# ---------------- dataset ----------------

class LabeledWindows(Dataset):
    def __init__(
        self,
        win_dir: str,
        *,
        feat_cfg: FeatCfg,
        fps_default: float,
        arch: str,
        two_stream: bool,
        compute_confirm_scores: bool = True,
    ):
        self.win_dir = win_dir
        self.feat_cfg = feat_cfg
        self.fps_default = float(fps_default)
        self.arch = str(arch).lower()
        self.two_stream = bool(two_stream)
        self.compute_confirm_scores = bool(compute_confirm_scores)

        self.files = sorted(glob.glob(os.path.join(win_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz windows found under: {win_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        fp = self.files[i]
        joints, motion, conf, mask, fps, meta = read_window_npz(fp, fps_default=self.fps_default)

        # Canonical features + derived mask (mask includes conf gating and/or precomputed mask).
        Xc, mask_used = build_canonical_input(
            joints_xy=joints,
            motion_xy=motion,
            conf=conf,
            mask=mask,
            fps=fps,
            feat_cfg=self.feat_cfg,
        )

        # Confirm-stage heuristic signals (used by alert policy when enabled).
        # WindowMeta is a plain dataclass (no slots), so we can attach attributes.
        lying_score = float(getattr(meta, "lying_score", 0.0))
        motion_score = float(getattr(meta, "motion_score", 0.0))
        if self.compute_confirm_scores and (lying_score == 0.0 and motion_score == 0.0):
            try:
                ls, ms = confirm_scores_window(joints, mask_used, float(fps))
                lying_score = float(ls) if np.isfinite(ls) else 0.0
                motion_score = float(ms) if np.isfinite(ms) else float("inf")
            except Exception:
                pass
        setattr(meta, "lying_score", lying_score)
        setattr(meta, "motion_score", motion_score)

        if self.arch == "tcn":
            X = torch.from_numpy(build_tcn_input(Xc, self.feat_cfg))
        else:
            X = torch.from_numpy(Xc)
            if self.two_stream:
                xj, xm = split_gcn_two_stream(Xc, self.feat_cfg)
                X = (torch.from_numpy(xj), torch.from_numpy(xm))

        return X, meta



def _collate(batch):
    Xs, metas = zip(*batch)
    if Xs and isinstance(Xs[0], tuple):
        xj_arr, xm_arr = zip(*Xs)
        xj = torch.stack(xj_arr, 0)
        xm = torch.stack(xm_arr, 0)
        return (xj, xm), list(metas)
    xb = torch.stack(Xs, 0)
    return xb, list(metas)


@torch.inference_mode()
def infer_probs(model, loader, device, arch: str, two_stream: bool, temperature: float = 1.0):
    # Temperature scaling (must match eval/fit_ops.py)
    try:
        temp = float(temperature)
        if not (temp > 0.0):
            temp = 1.0
    except Exception:
        temp = 1.0
    use_temp = abs(temp - 1.0) > 1e-12
    inv_temp = (1.0 / temp) if use_temp else 1.0
    n_total = len(loader.dataset) if hasattr(loader, "dataset") else 0
    use_prealloc = n_total > 0
    probs_buf = np.empty((n_total,), dtype=np.float32) if use_prealloc else None
    vids_buf = np.empty((n_total,), dtype=object) if use_prealloc else None
    ws_buf = np.empty((n_total,), dtype=np.int32) if use_prealloc else None
    we_buf = np.empty((n_total,), dtype=np.int32) if use_prealloc else None
    fps_buf = np.empty((n_total,), dtype=np.float64) if use_prealloc else None
    y_buf = np.empty((n_total,), dtype=np.int32) if use_prealloc else None
    ls_buf = np.empty((n_total,), dtype=np.float32) if use_prealloc else None
    ms_buf = np.empty((n_total,), dtype=np.float32) if use_prealloc else None
    ptr = 0

    probs: List[np.ndarray] = []
    vids: List[str] = []
    ws: List[int] = []
    we: List[int] = []
    fps: List[float] = []
    y_true: List[int] = []
    ls_list: List[float] = []
    ms_list: List[float] = []
    non_blocking = (getattr(device, "type", "cpu") in {"cuda", "mps"})

    for Xb, metas in loader:
        if arch == "tcn":
            xb = Xb.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            logits = logits_1d(model(xb))
        else:
            if two_stream:
                xj, xm = Xb
                xj = xj.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                xm = xm.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                logits = logits_1d(model(xj, xm))
            else:
                xb = Xb.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                logits = logits_1d(model(xb))

        logits_eff = (logits * inv_temp) if use_temp else logits
        p = torch.sigmoid(logits_eff).cpu().numpy().reshape(-1)
        bsz = p.size
        if (
            use_prealloc
            and probs_buf is not None
            and vids_buf is not None
            and ws_buf is not None
            and we_buf is not None
            and fps_buf is not None
            and y_buf is not None
            and ls_buf is not None
            and ms_buf is not None
            and (ptr + bsz) <= n_total
        ):
            probs_buf[ptr:ptr + bsz] = p
            vids_buf[ptr:ptr + bsz] = np.asarray([m.video_id for m in metas], dtype=object)
            ws_buf[ptr:ptr + bsz] = np.fromiter((int(m.w_start) for m in metas), dtype=np.int32, count=bsz)
            we_buf[ptr:ptr + bsz] = np.fromiter((int(m.w_end) for m in metas), dtype=np.int32, count=bsz)
            fps_buf[ptr:ptr + bsz] = np.fromiter((float(m.fps) for m in metas), dtype=np.float64, count=bsz)
            y_buf[ptr:ptr + bsz] = np.fromiter((int(m.y) for m in metas), dtype=np.int32, count=bsz)
            ls_buf[ptr:ptr + bsz] = np.fromiter(
                (float(getattr(m, "lying_score", 0.0)) for m in metas),
                dtype=np.float32,
                count=bsz,
            )
            ms_buf[ptr:ptr + bsz] = np.fromiter(
                (float(getattr(m, "motion_score", 0.0)) for m in metas),
                dtype=np.float32,
                count=bsz,
            )
            ptr += bsz
        else:
            probs.append(p)
            vids.extend(m.video_id for m in metas)
            ws.extend(int(m.w_start) for m in metas)
            we.extend(int(m.w_end) for m in metas)
            fps.extend(float(m.fps) for m in metas)
            y_true.extend(int(m.y) for m in metas)
            ls_list.extend(float(getattr(m, "lying_score", 0.0)) for m in metas)
            ms_list.extend(float(getattr(m, "motion_score", 0.0)) for m in metas)

    if use_prealloc and ptr > 0 and not probs:
        probs_out = probs_buf[:ptr]
        vids_out = vids_buf[:ptr]
        ws_out = ws_buf[:ptr]
        we_out = we_buf[:ptr]
        fps_out = fps_buf[:ptr]
        y_out = y_buf[:ptr]
        ls_out = ls_buf[:ptr]
        ms_out = ms_buf[:ptr]
    elif use_prealloc and ptr > 0:
        vids_tail = np.asarray(vids, dtype=object)
        ws_tail = np.asarray(ws, dtype=np.int32)
        we_tail = np.asarray(we, dtype=np.int32)
        fps_tail = np.asarray(fps, dtype=float)
        y_tail = np.asarray(y_true, dtype=np.int32)
        ls_tail = np.asarray(ls_list, dtype=np.float32)
        ms_tail = np.asarray(ms_list, dtype=np.float32)
        probs_out = np.concatenate((probs_buf[:ptr], *probs), axis=0)
        vids_out = np.concatenate([vids_buf[:ptr], vids_tail], axis=0)
        ws_out = np.concatenate([ws_buf[:ptr], ws_tail], axis=0)
        we_out = np.concatenate([we_buf[:ptr], we_tail], axis=0)
        fps_out = np.concatenate([fps_buf[:ptr], fps_tail], axis=0)
        y_out = np.concatenate([y_buf[:ptr], y_tail], axis=0)
        ls_out = np.concatenate([ls_buf[:ptr], ls_tail], axis=0)
        ms_out = np.concatenate([ms_buf[:ptr], ms_tail], axis=0)
    else:
        probs_out = np.concatenate(probs, axis=0) if probs else np.array([], dtype=np.float32)
        vids_out = np.asarray(vids, dtype=object)
        ws_out = np.asarray(ws, dtype=np.int32)
        we_out = np.asarray(we, dtype=np.int32)
        fps_out = np.asarray(fps, dtype=float)
        y_out = np.asarray(y_true, dtype=np.int32)
        ls_out = np.asarray(ls_list, dtype=np.float32)
        ms_out = np.asarray(ms_list, dtype=np.float32)

    return (
        probs_out,
        vids_out,
        ws_out,
        we_out,
        fps_out,
        y_out,
        ls_out,
        ms_out,
    )


def _aggregate_event_metrics(
    probs: np.ndarray,
    vids_arr: np.ndarray,
    ws_arr: np.ndarray,
    we_arr: np.ndarray,
    fps_arr: np.ndarray,
    y_true: np.ndarray,
    *,
    fps_default: float,
    alert_cfg: AlertCfg,
    merge_gap_s: float,
    overlap_slack_s: float,
    time_mode: str,
    ls_arr: Optional[np.ndarray] = None,
    ms_arr: Optional[np.ndarray] = None,
    video_groups: Optional[List[Tuple[str, np.ndarray]]] = None,
    video_cache: Optional[List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, float, float, Optional[np.ndarray], Optional[np.ndarray], List[Tuple[float, float]]]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Aggregate per-video event metrics into totals + rich per-video details."""
    groups = video_groups if video_groups is not None else _build_video_groups(vids_arr, ws_arr)
    cache = video_cache

    totals = {
        "n_videos": 0,
        "n_windows": 0,
        "n_gt_events": 0,
        "n_alert_events": 0,
        "n_true_alerts": 0,
        "n_false_alerts": 0,
        "sum_recall": 0.0,
        "sum_recall_count": 0,
        "sum_delay_s": 0.0,
        "sum_delay_s_count": 0,
        "total_duration_s": 0.0,
    }
    state_totals = {
        "n_windows": 0,
        "clear": 0,
        "suspect": 0,
        "alert": 0,
        "suspect_time_s": 0.0,
        "alert_time_s": 0.0,
    }

    per_video: Dict[str, Any] = {}

    if cache is None:
        cache = _prepare_video_cache(
            probs,
            vids_arr,
            ws_arr,
            we_arr,
            fps_arr,
            y_true,
            fps_default=fps_default,
            time_mode=time_mode,
            merge_gap_s=float(merge_gap_s),
            ls_arr=ls_arr,
            ms_arr=ms_arr,
            video_groups=groups,
        )

    for v, p_v, y_v, t_v, duration_s, fps_v, ls_v, ms_v, gt_v in cache:
        totals["total_duration_s"] += max(0.0, duration_s)

        ps_v = ema_smooth(p_v, alert_cfg.ema_alpha)
        alert_mask, alert_events = detect_alert_events_from_smoothed(
            ps_v,
            t_v,
            alert_cfg,
            lying_score=ls_v,
            motion_score=ms_v,
        )
        em, detail = event_metrics_from_windows(
            p_v,
            y_v,
            t_v,
            alert_cfg,
            lying_score=ls_v,
            motion_score=ms_v,
            alert_events=alert_events,
            gt_events=gt_v,
            duration_s=duration_s,
            merge_gap_s=float(merge_gap_s),
            overlap_slack_s=float(overlap_slack_s),
        )
        em = _event_metrics_to_compat_dict(em)

        # Reuse the same policy pass as event metrics for state counts.
        suspect_mask = (~alert_mask) & (ps_v >= float(alert_cfg.tau_low)) & (ps_v < float(alert_cfg.tau_high))
        clear_mask = (~alert_mask) & (ps_v < float(alert_cfg.tau_low))

        # Approximate time occupancy with median dt.
        dt = float(np.median(np.diff(t_v))) if t_v.size >= 2 else 0.0
        n_windows = int(t_v.size)
        n_clear = int(np.sum(clear_mask))
        n_suspect = int(np.sum(suspect_mask))
        n_alert = int(np.sum(alert_mask))

        state_totals["n_windows"] += n_windows
        state_totals["clear"] += n_clear
        state_totals["suspect"] += n_suspect
        state_totals["alert"] += n_alert
        state_totals["suspect_time_s"] += float(n_suspect * dt)
        state_totals["alert_time_s"] += float(n_alert * dt)

        totals["n_videos"] += 1
        totals["n_windows"] += int(len(p_v))
        totals["n_gt_events"] += int(_em_get(em, "n_gt_events", 0))
        totals["n_alert_events"] += int(_em_get(em, "n_alert_events", 0))
        totals["n_true_alerts"] += int(_em_get(em, "n_true_alerts", 0))
        totals["n_false_alerts"] += int(_em_get(em, "n_false_alerts", 0))
        rec_v = _em_get(em, "recall", float('nan'))
        n_gt = int(_em_get(em, "n_gt_events", 0))
        if n_gt > 0 and np.isfinite(float(rec_v)):
            totals["sum_recall"] += float(rec_v)
            totals["sum_recall_count"] += 1

        md = _em_get(em, "mean_delay_s", None)
        if md is not None and np.isfinite(float(md)):
            totals["sum_delay_s"] += float(md)
            totals["sum_delay_s_count"] += 1

        per_video[str(v)] = {
            "event_metrics": em,
            "state_counts": {
                "n_windows": n_windows,
                "clear": n_clear,
                "suspect": n_suspect,
                "alert": n_alert,
                "suspect_frac": float(n_suspect / n_windows) if n_windows > 0 else float("nan"),
                "alert_frac": float(n_alert / n_windows) if n_windows > 0 else float("nan"),
                "suspect_time_s": float(n_suspect * dt),
                "alert_time_s": float(n_alert * dt),
            },
            "detail": detail,
            "fps": float(fps_v),
            "duration_s": float(duration_s),
        }

    # Derived totals
    n_vid = int(totals["n_videos"])
    avg_recall = float(totals["sum_recall"] / totals["sum_recall_count"]) if totals["sum_recall_count"] > 0 else float("nan")
    mean_delay = float(totals["sum_delay_s"] / totals["sum_delay_s_count"]) if totals["sum_delay_s_count"] > 0 else float("nan")

    dur_s = float(totals["total_duration_s"])
    fa_per_hour = float(totals["n_false_alerts"] / (dur_s / 3600.0)) if dur_s > 0 else float("nan")
    fa_per_day = float(totals["n_false_alerts"] / (dur_s / 86400.0)) if dur_s > 0 else float("nan")

    out_totals = {
        **totals,
        "avg_recall": avg_recall,
        "mean_delay_s": mean_delay,
        "fa_per_hour": fa_per_hour,
        "fa_per_day": fa_per_day,
        "fa24h": fa_per_day,
        "state_counts": {
            **state_totals,
            "suspect_frac": float(state_totals["suspect"] / max(1, state_totals["n_windows"])),
            "alert_frac": float(state_totals["alert"] / max(1, state_totals["n_windows"])),
        },
    }

    out_detail = {"per_video": per_video}
    return out_totals, out_detail



def _aggregate_event_counts(
    probs: np.ndarray,
    vids_arr: np.ndarray,
    ws_arr: np.ndarray,
    we_arr: np.ndarray,
    fps_arr: np.ndarray,
    y_true: np.ndarray,
    *,
    fps_default: float,
    alert_cfg: AlertCfg,
    merge_gap_s: float,
    overlap_slack_s: float,
    time_mode: str,
    ls_arr: Optional[np.ndarray] = None,
    ms_arr: Optional[np.ndarray] = None,
    video_groups: Optional[List[Tuple[str, np.ndarray]]] = None,
    video_cache: Optional[List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, float, float, Optional[np.ndarray], Optional[np.ndarray], List[Tuple[float, float]]]]] = None,
) -> Dict[str, Any]:
    """Totals-only aggregation (faster, no per-video details/state counts).

    Used for threshold sweeps.
    """
    groups = video_groups if video_groups is not None else _build_video_groups(vids_arr, ws_arr)
    cache = video_cache

    totals = {
        "n_videos": 0,
        "n_windows": 0,
        "n_gt_events": 0,
        "n_alert_events": 0,
        "n_true_alerts": 0,
        "n_false_alerts": 0,
        "sum_recall": 0.0,
        "sum_recall_count": 0,
        "sum_delay_s": 0.0,
        "sum_delay_s_count": 0,
        "total_duration_s": 0.0,
    }

    if cache is None:
        cache = _prepare_video_cache(
            probs,
            vids_arr,
            ws_arr,
            we_arr,
            fps_arr,
            y_true,
            fps_default=fps_default,
            time_mode=time_mode,
            merge_gap_s=float(merge_gap_s),
            ls_arr=ls_arr,
            ms_arr=ms_arr,
            video_groups=groups,
        )

    for _v, p_v, y_v, t_v, duration_s, _fps_v, ls_v, ms_v, gt_v in cache:
        totals["total_duration_s"] += max(0.0, duration_s)

        ps_v = ema_smooth(p_v, alert_cfg.ema_alpha)
        _alert_mask, alert_events = detect_alert_events_from_smoothed(
            ps_v,
            t_v,
            alert_cfg,
            lying_score=ls_v,
            motion_score=ms_v,
        )
        em, _detail = event_metrics_from_windows(
            p_v,
            y_v,
            t_v,
            alert_cfg,
            lying_score=ls_v,
            motion_score=ms_v,
            alert_events=alert_events,
            gt_events=gt_v,
            duration_s=duration_s,
            merge_gap_s=float(merge_gap_s),
            overlap_slack_s=float(overlap_slack_s),
        )
        em = _event_metrics_to_compat_dict(em)

        totals["n_videos"] += 1
        totals["n_windows"] += int(len(p_v))
        totals["n_gt_events"] += int(_em_get(em, "n_gt_events", 0))
        totals["n_alert_events"] += int(_em_get(em, "n_alert_events", 0))
        totals["n_true_alerts"] += int(_em_get(em, "n_true_alerts", 0))
        totals["n_false_alerts"] += int(_em_get(em, "n_false_alerts", 0))
        rec_v = _em_get(em, "recall", float('nan'))
        n_gt = int(_em_get(em, "n_gt_events", 0))
        if n_gt > 0 and np.isfinite(float(rec_v)):
            totals["sum_recall"] += float(rec_v)
            totals["sum_recall_count"] += 1

        md = _em_get(em, "mean_delay_s", None)
        if md is not None and np.isfinite(float(md)):
            totals["sum_delay_s"] += float(md)
            totals["sum_delay_s_count"] += 1

    n_vid = int(totals["n_videos"])
    avg_recall = float(totals["sum_recall"] / totals["sum_recall_count"]) if totals["sum_recall_count"] > 0 else float("nan")
    mean_delay = float(totals["sum_delay_s"] / totals["sum_delay_s_count"]) if totals["sum_delay_s_count"] > 0 else float("nan")

    dur_s = float(totals["total_duration_s"])
    fa_per_day = float(totals["n_false_alerts"] / (dur_s / 86400.0)) if dur_s > 0 else float("nan")

    # Event-level precision/recall/F1 (across all videos)
    n_true = float(totals["n_true_alerts"])
    n_false = float(totals["n_false_alerts"])
    n_gt = float(totals["n_gt_events"])

    precision = float(n_true / max(1e-9, (n_true + n_false)))
    recall_ev = float(n_true / max(1e-9, n_gt))
    f1 = float(2 * precision * recall_ev / max(1e-9, (precision + recall_ev)))

    return {
        **totals,
        "avg_recall": avg_recall,
        "mean_delay_s": mean_delay,
        "fa_per_day": fa_per_day,
        "fa24h": fa_per_day,
        "precision": precision,
        "recall": recall_ev,
        "f1": f1,
    }




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)

    # Optional: load operating points + policy from fit_ops output
    ap.add_argument("--ops_yaml", default=None)

    # Optional: override fps_default (otherwise read from checkpoint cfg)
    ap.add_argument("--fps_default", type=float, default=None)

    # Optional: threshold sweep (kept for Makefile compatibility)
    ap.add_argument("--thr_min", type=float, default=None)
    ap.add_argument("--thr_max", type=float, default=None)
    ap.add_argument("--thr_step", type=float, default=None)

    # When sweeping: tau_low = tau_high * tau_low_ratio
    ap.add_argument("--tau_low_ratio", type=float, default=0.78)

    # alert cfg
    ap.add_argument("--ema_alpha", type=float, default=0.20)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tau_high", type=float, default=0.90)
    ap.add_argument("--tau_low", type=float, default=0.70)
    ap.add_argument("--cooldown_s", type=float, default=30.0)

    # confirm (optional)
    ap.add_argument("--confirm", type=int, default=0)
    ap.add_argument("--confirm_s", type=float, default=2.0)
    ap.add_argument("--confirm_min_lying", type=float, default=0.65)
    ap.add_argument("--confirm_max_motion", type=float, default=0.08)
    ap.add_argument("--confirm_require_low", type=int, default=1)

    # event + time params
    ap.add_argument("--merge_gap_s", type=float, default=1.0)
    ap.add_argument("--overlap_slack_s", type=float, default=0.0)
    ap.add_argument("--time_mode", type=str, default="center", choices=["start", "center", "end"])

    # optional FA-only dir
    ap.add_argument("--fa_dir", default=None, help="Optional dir of ADL-only windows for better FA/24h estimate.")

    args = ap.parse_args()

    # Load ops/policy early so inference can apply the same calibration/confirm settings.
    ops_blob = _load_ops_yaml(str(args.ops_yaml) if args.ops_yaml else "")
    policy_d, ops_d = _extract_policy_and_ops(ops_blob)
    confirm_enabled = bool(int(args.confirm))
    if "confirm" in policy_d and policy_d["confirm"] is not None:
        try:
            confirm_enabled = bool(int(policy_d["confirm"]))
        except Exception:
            pass
    confirm_s = float(args.confirm_s)
    if "confirm_s" in policy_d and policy_d["confirm_s"] is not None:
        try:
            confirm_s = float(policy_d["confirm_s"])
        except Exception:
            pass
    need_confirm_scores = bool(confirm_enabled and confirm_s > 0.0)

    # Calibration: fit_ops writes calibration.T into ops_yaml; evaluation must apply it.
    temperature = 1.0
    try:
        cal = ops_blob.get("calibration", {}) if isinstance(ops_blob, dict) else {}
        if isinstance(cal, dict) and cal.get("T", None) is not None:
            temperature = float(cal.get("T"))
    except Exception:
        temperature = 1.0
    if not (temperature > 0.0):
        temperature = 1.0

    device = pick_device()
    bundle = load_ckpt(args.ckpt, map_location=device)
    arch_ck, model_cfg_d, feat_cfg_d, data_cfg_d = get_cfg(bundle)

    arch = str(arch_ck).lower()
    feat_cfg = FeatCfg.from_dict(feat_cfg_d)
    fps_default = float(data_cfg_d.get("fps_default", 30.0))
    if args.fps_default is not None:
        fps_default = float(args.fps_default)

    two_stream = bool(model_cfg_d.get("two_stream", False))
    nw = int(args.num_workers)
    if device.type == "mps":
        nw = 0
    loader_kwargs = {
        "num_workers": nw,
        "pin_memory": bool(device.type == "cuda"),
        "persistent_workers": bool(nw > 0),
        "collate_fn": _collate,
    }
    if nw > 0:
        loader_kwargs["prefetch_factor"] = 2

    model = build_model(arch, model_cfg_d, feat_cfg, fps_default=fps_default).to(device)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()

    ds = LabeledWindows(
        args.win_dir,
        feat_cfg=feat_cfg,
        fps_default=fps_default,
        arch=arch,
        two_stream=two_stream,
        compute_confirm_scores=need_confirm_scores,
    )
    if len(ds) > 0:
        x0, _m0 = ds[0]
        if arch == "gcn" and two_stream and isinstance(x0, tuple):
            validate_model_input_dims(
                "gcn",
                model_cfg_d,
                xj=np.asarray(x0[0]),
                xm=np.asarray(x0[1]),
            )
        elif arch == "gcn":
            validate_model_input_dims("gcn", model_cfg_d, x=np.asarray(x0))
        else:
            validate_model_input_dims("tcn", model_cfg_d, x=np.asarray(x0))
    loader = DataLoader(ds, batch_size=int(args.batch), shuffle=False, **loader_kwargs)

    probs, vids_arr, ws_arr, we_arr, fps_arr, y_true, ls_arr, ms_arr = infer_probs(model, loader, device, arch, two_stream, temperature=temperature)
    video_groups_main = _build_video_groups(vids_arr, ws_arr)
    merge_gap_main = float(args.merge_gap_s)
    video_cache_main = _prepare_video_cache(
        probs,
        vids_arr,
        ws_arr,
        we_arr,
        fps_arr,
        y_true,
        fps_default=fps_default,
        time_mode=str(args.time_mode),
        merge_gap_s=merge_gap_main,
        ls_arr=ls_arr,
        ms_arr=ms_arr,
        video_groups=video_groups_main,
    )


    # Optional: run inference on FA-only windows for a better FA/24h estimate.
    fa_payload = None
    if args.fa_dir:
        ds_fa = LabeledWindows(
            args.fa_dir,
            feat_cfg=feat_cfg,
            fps_default=fps_default,
            arch=arch,
            two_stream=two_stream,
            compute_confirm_scores=need_confirm_scores,
        )
        loader_fa = DataLoader(ds_fa, batch_size=int(args.batch), shuffle=False, **loader_kwargs)
        probs_fa, vids_fa, ws_fa, we_fa, fps_fa, y_fa, ls_fa, ms_fa = infer_probs(model, loader_fa, device, arch, two_stream, temperature=temperature)

        # Treat FA windows as negative-only (best-effort).
        y_fa = np.zeros_like(y_fa, dtype=np.int32)

        fa_groups = _build_video_groups(vids_fa, ws_fa)
        fa_payload = {
            "probs": probs_fa,
            "vids": vids_fa,
            "ws": ws_fa,
            "we": we_fa,
            "fps": fps_fa,
            "y": y_fa,
            "ls": ls_fa,
            "ms": ms_fa,
            "groups": fa_groups,
            "cache": _prepare_video_cache(
                probs_fa,
                vids_fa,
                ws_fa,
                we_fa,
                fps_fa,
                y_fa,
                fps_default=fps_default,
                time_mode=str(args.time_mode),
                merge_gap_s=merge_gap_main,
                ls_arr=ls_fa,
                ms_arr=ms_fa,
                video_groups=fa_groups,
            ),
        }
    # Start from CLI defaults...
    base_alert_cfg = AlertCfg(
        ema_alpha=float(args.ema_alpha),
        k=int(args.k),
        n=int(args.n),
        tau_high=float(args.tau_high),
        tau_low=float(args.tau_low),
        cooldown_s=float(args.cooldown_s),
        confirm=bool(int(args.confirm)),
        confirm_s=float(args.confirm_s),
        confirm_min_lying=float(args.confirm_min_lying),
        confirm_max_motion=float(args.confirm_max_motion),
        confirm_require_low=bool(int(args.confirm_require_low)),
    )

    # If ops_yaml includes a policy section, override base cfg (keep CLI as fallback).
    def _maybe(policy_key: str, cast, cur):
        if policy_key in policy_d and policy_d[policy_key] is not None:
            try:
                return cast(policy_d[policy_key])
            except Exception:
                return cur
        return cur

    tau_low_ratio = float(args.tau_low_ratio)
    # Prefer explicit tau_low_ratio in policy; otherwise fall back to fit_ops sweep_cfg.
    if "tau_low_ratio" in policy_d:
        try:
            tau_low_ratio = float(policy_d["tau_low_ratio"])
        except Exception:
            pass
    elif isinstance(ops_blob, dict) and isinstance(ops_blob.get("sweep_cfg"), dict) and ops_blob.get("sweep_cfg", {}).get("tau_low_ratio", None) is not None:
        try:
            tau_low_ratio = float(ops_blob.get("sweep_cfg", {}).get("tau_low_ratio"))
        except Exception:
            pass

    base_alert_cfg = AlertCfg(
        ema_alpha=_maybe("ema_alpha", float, base_alert_cfg.ema_alpha),
        k=_maybe("k", int, base_alert_cfg.k),
        n=_maybe("n", int, base_alert_cfg.n),
        tau_high=base_alert_cfg.tau_high,
        tau_low=base_alert_cfg.tau_low,
        cooldown_s=_maybe("cooldown_s", float, base_alert_cfg.cooldown_s),
        confirm=bool(int(_maybe("confirm", int, int(base_alert_cfg.confirm)))),
        confirm_s=_maybe("confirm_s", float, base_alert_cfg.confirm_s),
        confirm_min_lying=_maybe("confirm_min_lying", float, base_alert_cfg.confirm_min_lying),
        confirm_max_motion=_maybe("confirm_max_motion", float, base_alert_cfg.confirm_max_motion),
        confirm_require_low=bool(int(_maybe("confirm_require_low", int, int(base_alert_cfg.confirm_require_low)))),
    )

    # Optional: threshold sweep (needed by Makefile plot targets).
    sweep: List[Dict[str, Any]] = []
    if args.thr_min is not None and args.thr_max is not None and args.thr_step is not None:
        thr_min = float(args.thr_min)
        thr_max = float(args.thr_max)
        thr_step = float(args.thr_step)
        taus = np.arange(thr_min, thr_max + 1e-12, thr_step, dtype=float)
        taus = taus[(taus > 0.0) & (taus < 1.0)]
        for tau_h in taus:
            tau_l = min(float(tau_h * tau_low_ratio), float(tau_h - 1e-6))
            cfg_i = AlertCfg(
                ema_alpha=base_alert_cfg.ema_alpha,
                k=base_alert_cfg.k,
                n=base_alert_cfg.n,
                tau_high=float(tau_h),
                tau_low=float(max(0.0, tau_l)),
                cooldown_s=base_alert_cfg.cooldown_s,
                confirm=base_alert_cfg.confirm,
                confirm_s=base_alert_cfg.confirm_s,
                confirm_min_lying=base_alert_cfg.confirm_min_lying,
                confirm_max_motion=base_alert_cfg.confirm_max_motion,
                confirm_require_low=base_alert_cfg.confirm_require_low,
            )
            tot_i = _aggregate_event_counts(
                probs,
                vids_arr,
                ws_arr,
                we_arr,
                fps_arr,
                y_true,
                fps_default=fps_default,
                alert_cfg=cfg_i,
                merge_gap_s=float(args.merge_gap_s),
                overlap_slack_s=float(args.overlap_slack_s),
                time_mode=str(args.time_mode),
                ls_arr=ls_arr,
                ms_arr=ms_arr,
                video_groups=video_groups_main,
                video_cache=video_cache_main,
            )
            sweep.append({"tau_high": float(tau_h), "tau_low": float(cfg_i.tau_low), "fa_per_24h": tot_i.get("fa24h"), **tot_i})

    # Optional: evaluate ops from YAML (OP1/OP2/OP3) and attach their results.
    ops_results: Dict[str, Any] = {}
    for name, od in (ops_d or {}).items():
        if not isinstance(od, dict):
            continue
        tau_h = od.get("tau_high", od.get("tau", None))
        if tau_h is None:
            continue
        try:
            tau_h = float(tau_h)
        except Exception:
            continue
        tau_l = od.get("tau_low", None)
        if tau_l is None:
            tau_l = min(float(tau_h * tau_low_ratio), float(tau_h - 1e-6))
        try:
            tau_l = float(tau_l)
        except Exception:
            tau_l = min(float(tau_h * tau_low_ratio), float(tau_h - 1e-6))

        cfg_op = AlertCfg(
            ema_alpha=base_alert_cfg.ema_alpha,
            k=base_alert_cfg.k,
            n=base_alert_cfg.n,
            tau_high=float(tau_h),
            tau_low=float(max(0.0, min(tau_l, tau_h - 1e-6))),
            cooldown_s=base_alert_cfg.cooldown_s,
            confirm=base_alert_cfg.confirm,
            confirm_s=base_alert_cfg.confirm_s,
            confirm_min_lying=base_alert_cfg.confirm_min_lying,
            confirm_max_motion=base_alert_cfg.confirm_max_motion,
            confirm_require_low=base_alert_cfg.confirm_require_low,
        )

        tot_op = _aggregate_event_counts(
            probs,
            vids_arr,
            ws_arr,
            we_arr,
            fps_arr,
            y_true,
            fps_default=fps_default,
            alert_cfg=cfg_op,
            merge_gap_s=float(args.merge_gap_s),
            overlap_slack_s=float(args.overlap_slack_s),
            time_mode=str(args.time_mode),
            ls_arr=ls_arr,
            ms_arr=ms_arr,
            video_groups=video_groups_main,
            video_cache=video_cache_main,
        )
        ops_results[str(name)] = {"tau_high": cfg_op.tau_high, "tau_low": cfg_op.tau_low, "fa_per_24h": tot_op.get("fa24h"), **tot_op}

    # Pick the "main" config to compute full details for.
    selected = None
    if "op2" in ops_results:
        selected = ("op2", ops_results["op2"]["tau_high"], ops_results["op2"]["tau_low"])
    elif sweep:
        # best by F1 (event-level)
        best = max(sweep, key=lambda r: float(r.get("f1", -1e9)))
        selected = ("best_f1", float(best["tau_high"]), float(best["tau_low"]))

    if selected is None:
        selected = ("cli", float(base_alert_cfg.tau_high), float(base_alert_cfg.tau_low))

    sel_name, sel_tau_h, sel_tau_l = selected
    alert_cfg = AlertCfg(
        ema_alpha=base_alert_cfg.ema_alpha,
        k=base_alert_cfg.k,
        n=base_alert_cfg.n,
        tau_high=float(sel_tau_h),
        tau_low=float(sel_tau_l),
        cooldown_s=base_alert_cfg.cooldown_s,
        confirm=base_alert_cfg.confirm,
        confirm_s=base_alert_cfg.confirm_s,
        confirm_min_lying=base_alert_cfg.confirm_min_lying,
        confirm_max_motion=base_alert_cfg.confirm_max_motion,
        confirm_require_low=base_alert_cfg.confirm_require_low,
    )

    totals, detail = _aggregate_event_metrics(
        probs,
        vids_arr,
        ws_arr,
        we_arr,
        fps_arr,
        y_true,
        fps_default=fps_default,
        alert_cfg=alert_cfg,
        merge_gap_s=float(args.merge_gap_s),
        overlap_slack_s=float(args.overlap_slack_s),
        time_mode=str(args.time_mode),
        ls_arr=ls_arr,
        ms_arr=ms_arr,
        video_groups=video_groups_main,
        video_cache=video_cache_main,
    )

    # AP/AUC (robust to NaNs / empty) - independent of thresholds
    apauc = ap_auc(probs, y_true)

    # If FA-only windows are provided, compute FA/24h under the selected policy and each OP.
    fa_eval = None
    if fa_payload is not None:
        def _fa_counts_for_cfg(cfg: AlertCfg) -> Dict[str, Any]:
            tot_fa = _aggregate_event_counts(
                fa_payload["probs"],
                fa_payload["vids"],
                fa_payload["ws"],
                fa_payload["we"],
                fa_payload["fps"],
                fa_payload["y"],
                fps_default=fps_default,
                alert_cfg=cfg,
                merge_gap_s=float(args.merge_gap_s),
                overlap_slack_s=float(args.overlap_slack_s),
                time_mode=str(args.time_mode),
                ls_arr=fa_payload["ls"],
                ms_arr=fa_payload["ms"],
                video_groups=fa_payload.get("groups"),
                video_cache=fa_payload.get("cache"),
            )
            return {
                "fa_per_24h": tot_fa.get("fa24h"),
                "fa_per_day": tot_fa.get("fa_per_day"),
                "n_false_alerts_fa": tot_fa.get("n_false_alerts"),
                "n_alert_events_fa": tot_fa.get("n_alert_events"),
                "n_videos_fa": tot_fa.get("n_videos"),
                "total_duration_s_fa": tot_fa.get("total_duration_s"),
            }

        fa_eval = {
            "fa_dir": str(args.fa_dir),
            "selected": _fa_counts_for_cfg(alert_cfg),
            "ops": {},
        }
        for _opname, _od in ops_results.items():
            try:
                _tau_h = float(_od.get("tau_high"))
                _tau_l = float(_od.get("tau_low"))
            except Exception:
                continue
            _cfg = AlertCfg(
                ema_alpha=base_alert_cfg.ema_alpha,
                k=base_alert_cfg.k,
                n=base_alert_cfg.n,
                tau_high=float(_tau_h),
                tau_low=float(_tau_l),
                cooldown_s=base_alert_cfg.cooldown_s,
                confirm=base_alert_cfg.confirm,
                confirm_s=base_alert_cfg.confirm_s,
                confirm_min_lying=base_alert_cfg.confirm_min_lying,
                confirm_max_motion=base_alert_cfg.confirm_max_motion,
                confirm_require_low=base_alert_cfg.confirm_require_low,
            )
            fa_eval["ops"][_opname] = _fa_counts_for_cfg(_cfg)
            # Expose FA/24h on the OP result (keep other event metrics from main eval).
            if _opname in ops_results and isinstance(fa_eval["ops"][_opname], dict):
                ops_results[_opname]["fa_per_24h"] = fa_eval["ops"][_opname].get("fa_per_24h")
    report: Dict[str, Any] = {
        "arch": arch,
        "meta": {
            "overlap_slack_s": float(args.overlap_slack_s),
            "merge_gap_s": float(args.merge_gap_s),
            "time_mode": str(args.time_mode),
            "tau_low_ratio": float(tau_low_ratio),
            "n_fa_videos": (fa_eval.get("selected", {}).get("n_videos_fa") if fa_eval else 0),
            "total_duration_s_fa": (fa_eval.get("selected", {}).get("total_duration_s_fa") if fa_eval else 0.0),
        },
        "ckpt": args.ckpt,
        "win_dir": args.win_dir,
        "selected": {"name": str(sel_name), "tau_high": float(alert_cfg.tau_high), "tau_low": float(alert_cfg.tau_low)},
        "alert_cfg": alert_cfg.to_dict(),
        "event_cfg": {
            "merge_gap_s": float(args.merge_gap_s),
            "overlap_slack_s": float(args.overlap_slack_s),
            "time_mode": str(args.time_mode),
        },
        "policy": {"tau_low_ratio": float(tau_low_ratio), **(policy_d if isinstance(policy_d, dict) else {})},
        "ops": ops_results,
        "sweep": sweep,
        "totals": totals,
        "ap_auc": apauc,
        "detail": detail,
        "fa_eval": fa_eval,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[ok] wrote: {args.out_json}")


if __name__ == "__main__":
    main()
