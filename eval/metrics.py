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
    sweep_alert_policy_from_windows,
    classify_states,
)
from core.ckpt import get_cfg, load_ckpt
from core.features import FeatCfg, read_window_npz, build_tcn_input, build_canonical_input, split_gcn_two_stream
from core.confirm import confirm_scores_window
from core.models import build_model, logits_1d, pick_device
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
    def __init__(self, win_dir: str, *, feat_cfg: FeatCfg, fps_default: float, arch: str, two_stream: bool):
        self.win_dir = win_dir
        self.feat_cfg = feat_cfg
        self.fps_default = float(fps_default)
        self.arch = str(arch).lower()
        self.two_stream = bool(two_stream)

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
        try:
            ls, ms = confirm_scores_window(joints, mask_used, float(fps))
            setattr(meta, "lying_score", float(ls))
            setattr(meta, "motion_score", float(ms))
        except Exception:
            pass

        if self.arch == "tcn":
            X = build_tcn_input(Xc, self.feat_cfg)
        else:
            X = Xc
            if self.two_stream:
                X = split_gcn_two_stream(X, self.feat_cfg)

        return X, meta



def _collate(batch):
    Xs, metas = zip(*batch)
    return list(Xs), list(metas)


@torch.no_grad()
def infer_probs(model, loader, device, arch: str, two_stream: bool):
    probs = []
    vids, ws, we, fps = [], [], [], []
    y_true = []
    ls_list, ms_list = [], []

    for Xs, metas in loader:
        if arch == "tcn":
            xb = torch.from_numpy(np.stack(Xs, axis=0)).to(device)
            logits = logits_1d(model(xb))
        else:
            if two_stream:
                xj = torch.from_numpy(np.stack([x[0] for x in Xs], axis=0)).to(device)
                xm = torch.from_numpy(np.stack([x[1] for x in Xs], axis=0)).to(device)
                logits = logits_1d(model(xj, xm))
            else:
                xb = torch.from_numpy(np.stack(Xs, axis=0)).to(device)
                logits = logits_1d(model(xb))

        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs.append(p)

        for m in metas:
            vids.append(m.video_id)
            ws.append(int(m.w_start))
            we.append(int(m.w_end))
            fps.append(float(m.fps))
            y_true.append(int(m.y))
            ls_list.append(float(getattr(m, "lying_score", 0.0)))
            ms_list.append(float(getattr(m, "motion_score", 0.0)))

    return (
        (np.concatenate(probs) if probs else np.array([])),
        np.asarray(vids),
        np.asarray(ws),
        np.asarray(we),
        np.asarray(fps, dtype=float),
        np.asarray(y_true, dtype=np.int32),
        np.asarray(ls_list, dtype=np.float32),
        np.asarray(ms_list, dtype=np.float32),
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
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Aggregate per-video event metrics into totals + rich per-video details."""
    unique_vids = list(dict.fromkeys(list(vids_arr)))

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

    for v in unique_vids:
        mv = vids_arr == v
        if not mv.any():
            continue

        idx = np.argsort(ws_arr[mv])
        p_v = probs[mv][idx]
        y_v = y_true[mv][idx]
        ws_v = ws_arr[mv][idx]
        we_v = we_arr[mv][idx]

        ls_v = ls_arr[mv][idx] if ls_arr is not None else None
        ms_v = ms_arr[mv][idx] if ms_arr is not None else None

        fps_v = float(np.median(fps_arr[mv])) if np.isfinite(fps_arr[mv]).any() else float(fps_default)
        if fps_v <= 0:
            fps_v = float(fps_default)

        # Timestamps used for alert policy + merging.
        # NOTE: w_end is inclusive; core.alerting.times_from_windows uses (ws+we)/2 for center mode.
        t_v = times_from_windows(ws_v, we_v, fps_v, mode=time_mode)

        # Duration MUST honor inclusive w_end: +1 frame.
        duration_s = float((we_v.max() - ws_v.min() + 1) / max(1e-6, fps_v))
        totals["total_duration_s"] += max(0.0, duration_s)

        em, detail = event_metrics_from_windows(
            p_v,
            y_v,
            t_v,
            alert_cfg,
            lying_score=ls_v,
            motion_score=ms_v,
            duration_s=duration_s,
            merge_gap_s=float(merge_gap_s),
            overlap_slack_s=float(overlap_slack_s),
        )
        em = _event_metrics_to_compat_dict(em)

        # ✅ FIX: keep state counts consistent with confirm stage (lying/motion).
        st = classify_states(p_v, t_v, alert_cfg, lying_score=ls_v, motion_score=ms_v)

        # Approximate time occupancy with median dt.
        dt = float(np.median(np.diff(t_v))) if t_v.size >= 2 else 0.0
        n_windows = int(t_v.size)
        n_clear = int(np.sum(st["clear"]))
        n_suspect = int(np.sum(st["suspect"]))
        n_alert = int(np.sum(st["alert"]))

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
) -> Dict[str, Any]:
    """Totals-only aggregation (faster, no per-video details/state counts).

    Used for threshold sweeps.
    """
    unique_vids = list(dict.fromkeys(list(vids_arr)))

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

    for v in unique_vids:
        mv = vids_arr == v
        if not mv.any():
            continue

        idx = np.argsort(ws_arr[mv])
        p_v = probs[mv][idx]
        y_v = y_true[mv][idx]
        ws_v = ws_arr[mv][idx]
        we_v = we_arr[mv][idx]

        ls_v = ls_arr[mv][idx] if ls_arr is not None else None
        ms_v = ms_arr[mv][idx] if ms_arr is not None else None

        fps_v = float(np.median(fps_arr[mv])) if np.isfinite(fps_arr[mv]).any() else float(fps_default)
        if fps_v <= 0:
            fps_v = float(fps_default)

        t_v = times_from_windows(ws_v, we_v, fps_v, mode=time_mode)
        duration_s = float((we_v.max() - ws_v.min() + 1) / max(1e-6, fps_v))
        totals["total_duration_s"] += max(0.0, duration_s)

        em, _detail = event_metrics_from_windows(
            p_v,
            y_v,
            t_v,
            alert_cfg,
            lying_score=ls_v,
            motion_score=ms_v,
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

    device = pick_device()
    bundle = load_ckpt(args.ckpt, map_location=device)
    arch_ck, model_cfg_d, feat_cfg_d, data_cfg_d = get_cfg(bundle)

    arch = str(arch_ck).lower()
    feat_cfg = FeatCfg.from_dict(feat_cfg_d)
    fps_default = float(data_cfg_d.get("fps_default", 30.0))
    if args.fps_default is not None:
        fps_default = float(args.fps_default)

    two_stream = bool(model_cfg_d.get("two_stream", False))

    model = build_model(arch, model_cfg_d, feat_cfg, fps_default=fps_default).to(device)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()

    ds = LabeledWindows(args.win_dir, feat_cfg=feat_cfg, fps_default=fps_default, arch=arch, two_stream=two_stream)
    loader = DataLoader(ds, batch_size=int(args.batch), shuffle=False, num_workers=0, collate_fn=_collate)

    probs, vids_arr, ws_arr, we_arr, fps_arr, y_true, ls_arr, ms_arr = infer_probs(model, loader, device, arch, two_stream)

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

    ops_blob = _load_ops_yaml(str(args.ops_yaml) if args.ops_yaml else "")
    policy_d, ops_d = _extract_policy_and_ops(ops_blob)

    # If ops_yaml includes a policy section, override base cfg (keep CLI as fallback).
    def _maybe(policy_key: str, cast, cur):
        if policy_key in policy_d and policy_d[policy_key] is not None:
            try:
                return cast(policy_d[policy_key])
            except Exception:
                return cur
        return cur

    tau_low_ratio = float(args.tau_low_ratio)
    if "tau_low_ratio" in policy_d:
        try:
            tau_low_ratio = float(policy_d["tau_low_ratio"])
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
            )
            sweep.append({"tau_high": float(tau_h), "tau_low": float(cfg_i.tau_low), **tot_i})

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
        )
        ops_results[str(name)] = {"tau_high": cfg_op.tau_high, "tau_low": cfg_op.tau_low, **tot_op}

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
    )

    # AP/AUC (robust to NaNs / empty) - independent of thresholds
    apauc = ap_auc(probs, y_true)
    report: Dict[str, Any] = {
        "arch": arch,
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
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[ok] wrote: {args.out_json}")


if __name__ == "__main__":
    main()
