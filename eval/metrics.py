#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/metrics.py

Evaluate a trained checkpoint on a windows directory, producing a JSON report.

This script answers 2 questions:
1) Window-level: "How good are the probabilities?" (AP / AUC)
2) Deployment-level: "How good is the alert system?" (event recall, FA/24h, delay)

Key features supported:
- prefer_ema (default ON)
- temperature scaling (cli > calibration_yaml > ops_yaml.calibration.temperature)
- full alert policy: EMA smoothing, hysteresis, k-of-n, confirm, cooldown
- quality/motion/lying signals are fed into alert policy (when enabled)

Output JSON contains:
- window_metrics: threshold-free (AP/AUC)
- event_metrics: deployment-style event recall / precision / FA/day / delay
- per_video: event metrics + state counts per clip
"""

from __future__ import annotations

# -------------------------
# Path bootstrap: allow `python eval/metrics.py ...` from repo root
# -------------------------
import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.alerting import AlertCfg, classify_states, event_metrics_from_windows, times_from_windows
from core.calibration import apply_temperature, load_temperature
from core.ckpt import get_cfg, get_state_dict, load_ckpt
from core.features import FeatCfg, build_gcn_input, build_tcn_input, read_window_npz
from core.metrics import ap_auc
from core.models import build_model, logits_1d, pick_device
from core.signals import compute_window_signals
from core.yamlio import yaml_load_simple


# ============================================================
# 0) JSON sanitization helper (convert NaN/Inf -> null)
# ============================================================
def _json_sanitize(obj: Any):
    """Recursively convert values into JSON-safe Python types.

    - float NaN/Inf -> None (serialized as null)
    - numpy scalars -> Python scalars
    - numpy arrays -> lists
    """
    # numpy scalar
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_json_sanitize(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj

# ============================================================
# 1) Small numerical helper
# ============================================================
def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    """
    Stable sigmoid for numpy.

    Why we need a "stable" sigmoid:
    - If logits are large (e.g., 100), exp(-x) underflows/overflows.
    - Clipping keeps exp in a safe range.

    Output: float32 array in [0,1]
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


# ============================================================
# 2) Dataset record for traceability
# ============================================================
@dataclass
class MetaRow:
    """
    Metadata for one window sample.

    path:
      file path of this window NPZ (for debugging / traceability)

    video_id:
      which original clip this window belongs to (used for per-video grouping)

    w_start, w_end:
      inclusive frame indices of the window in the original clip (used for time mapping)

    fps:
      fps for this window (after preprocessing / resample)

    y:
      label: 1=fall, 0=non-fall, -1=unlabeled (dataset dependent)
    """
    path: str
    video_id: str
    w_start: int
    w_end: int
    fps: float
    y: int


class Windows(Dataset):
    """
    Loads window NPZ files and builds model inputs.

    Returns:
      X: model input
         - TCN: np.ndarray [T, C]
         - GCN: np.ndarray [T, V, F]
         - GCN two-stream: tuple(xj, xm) with:
              xj [T,V,Fj], xm [T,V,2]
      y: int64
      meta: MetaRow
      quality, lying, motion: float32 signals for alert policy
    """

    def __init__(self, root: str, feat_cfg: FeatCfg, fps_default: float, arch: str, two_stream: bool):
        # Use recursive glob so passing a split folder or shard folder still works.
        self.files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No .npz under: {root}")

        self.feat_cfg = feat_cfg
        self.fps_default = float(fps_default)
        self.arch = str(arch).lower().strip()
        self.two_stream = bool(two_stream)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]

        # read_window_npz standardizes the NPZ schema for windows
        joints, motion, conf, mask, fps, meta = read_window_npz(p, fps_default=self.fps_default)

        # Keep y exactly as stored (can be -1 for unlabeled)
        y = int(meta.y) if meta.y is not None else -1

        # Compute signals from the same representation the model uses
        sig = compute_window_signals(joints, motion, conf, mask, float(fps), self.feat_cfg)

        # ---- Build model input ----
        if self.arch == "tcn":
            X, _ = build_tcn_input(joints, motion, conf, mask, fps=float(fps), feat_cfg=self.feat_cfg)  # [T,C]
        else:
            X, _ = build_gcn_input(joints, motion, conf, mask, fps=float(fps), feat_cfg=self.feat_cfg)  # [T,V,F]

            # Optional: two-stream split (joint stream + motion stream)
            if self.two_stream:
                xy = X[..., 0:2]  # always xy channels

                # Joint stream optionally includes confidence channel as the last feature
                if self.feat_cfg.use_conf_channel:
                    c = X[..., -1:]  # [T,V,1]
                    xj = np.concatenate([xy, c], axis=-1)  # [T,V,3]
                else:
                    xj = xy  # [T,V,2]

                # Motion stream is channels 2:4 only if motion is enabled
                if self.feat_cfg.use_motion and X.shape[-1] >= 4:
                    xm = X[..., 2:4]  # [T,V,2]
                else:
                    xm = np.zeros_like(xy, dtype=np.float32)

                X = (xj.astype(np.float32), xm.astype(np.float32))

        # ---- Build MetaRow ----
        # video_id is used to group windows into a clip (for event/FA metrics).
        # We accept multiple possible meta fields.
        video_id = (
            meta.video_id
            or meta.seq_id
            or getattr(meta, "clip_id", None)
            or os.path.splitext(os.path.basename(p))[0]
        )

        m = MetaRow(
            path=str(p),
            video_id=str(video_id),
            w_start=int(meta.w_start),
            w_end=int(meta.w_end),
            fps=float(fps),
            y=int(y),
        )

        return X, np.int64(y), m, np.float32(sig.quality), np.float32(sig.lying), np.float32(sig.motion)


def _collate(batch):
    """
    Custom collate:
    - We keep X as a Python list because it may contain tuples for two-stream.
    - Other outputs are stacked into numpy arrays.
    """
    Xs, ys, metas, qs, ls, ms = zip(*batch)
    return (
        list(Xs),
        np.asarray(ys, dtype=np.int64),
        list(metas),
        np.asarray(qs, dtype=np.float32),
        np.asarray(ls, dtype=np.float32),
        np.asarray(ms, dtype=np.float32),
    )


# ============================================================
# 3) Model inference helper
# ============================================================
@torch.no_grad()
def infer_logits(model, loader, device, arch: str, two_stream: bool):
    """
    Run inference on a loader and return:
      logits: [N]
      y_true: [N]
      vids, ws, we, fps: list metadata (length N)
      qs, ls, ms: signals aligned with N windows
    """
    logits_all: List[np.ndarray] = []
    y_true_all: List[np.ndarray] = []

    vids: List[str] = []
    ws: List[int] = []
    we: List[int] = []
    fps_list: List[float] = []

    q_all: List[np.ndarray] = []
    l_all: List[np.ndarray] = []
    m_all: List[np.ndarray] = []

    arch = str(arch).lower().strip()

    for Xs, ys, metas, qs, ls, ms in loader:
        # ---- Build batch tensors ----
        if arch == "tcn":
            # Xs is a list of [T,C] arrays -> stack into [B,T,C]
            xb = torch.from_numpy(np.stack(Xs, axis=0)).to(device)
            logits = logits_1d(model(xb))
        else:
            if two_stream:
                # Xs is a list of tuples: (xj[T,V,Fj], xm[T,V,2])
                xj = torch.from_numpy(np.stack([x[0] for x in Xs], axis=0)).to(device)
                xm = torch.from_numpy(np.stack([x[1] for x in Xs], axis=0)).to(device)
                logits = logits_1d(model(xj, xm))
            else:
                xb = torch.from_numpy(np.stack(Xs, axis=0)).to(device)  # [B,T,V,F]
                logits = logits_1d(model(xb))

        # ---- Store outputs ----
        logits_all.append(logits.detach().cpu().numpy().reshape(-1))
        y_true_all.append(ys.reshape(-1))

        q_all.append(qs.reshape(-1))
        l_all.append(ls.reshape(-1))
        m_all.append(ms.reshape(-1))

        for meta in metas:
            vids.append(meta.video_id)
            ws.append(int(meta.w_start))
            we.append(int(meta.w_end))
            fps_list.append(float(meta.fps))

    return (
        np.concatenate(logits_all) if logits_all else np.array([], dtype=np.float32),
        np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int32),
        vids,
        ws,
        we,
        fps_list,
        np.concatenate(q_all) if q_all else np.array([], dtype=np.float32),
        np.concatenate(l_all) if l_all else np.array([], dtype=np.float32),
        np.concatenate(m_all) if m_all else np.array([], dtype=np.float32),
    )


# ============================================================
# 4) Event-level aggregation (per-video loop)
# ============================================================
def aggregate_event_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    vids: List[str],
    ws: List[int],
    we: List[int],
    fps_list: List[float],
    qs: np.ndarray,
    ls: np.ndarray,
    ms: np.ndarray,
    *,
    alert_cfg: AlertCfg,
    merge_gap_s: Optional[float],
    overlap_slack_s: float,
    time_mode: str,
    fps_default: float,
) -> Dict[str, Any]:
    """
    Convert per-window probabilities into per-video alert events and summarize.

    Important concept:
    - Event metrics are computed PER VIDEO, then summed/averaged.
    - FA/24h must use per-video duration, not mixed global windows.

    merge_gap_s:
      If None, we auto-estimate a reasonable merge gap based on median window time step.
    """
    vids_arr = np.asarray(vids, dtype=str)
    ws_arr = np.asarray(ws, dtype=np.int32)
    we_arr = np.asarray(we, dtype=np.int32)
    fps_arr = np.asarray(fps_list, dtype=np.float32)

    # Keep first-seen order (stable)
    unique_vids = list(dict.fromkeys(list(vids_arr)))

    total_duration_s = 0.0

    gt_total = 0
    matched_gt_total = 0
    alert_total = 0
    true_alert_total = 0
    false_alert_total = 0
    delays: List[float] = []

    # Count how often the alert state machine is in each state
    state_totals = {"n_windows": 0, "clear": 0, "suspect": 0, "pending": 0, "alert": 0}

    # ------------------------------------------------------------
    # Auto-estimate merge gap if not provided
    # ------------------------------------------------------------
    if merge_gap_s is None:
        gaps: List[float] = []
        for v in unique_vids:
            mv = vids_arr == v
            if not mv.any():
                continue
            idx = np.argsort(ws_arr[mv])

            fps_v = float(np.median(fps_arr[mv])) if np.isfinite(fps_arr[mv]).any() else float(fps_default)
            fps_v = fps_v if fps_v > 0 else float(fps_default)

            t_v = times_from_windows(ws_arr[mv][idx], we_arr[mv][idx], fps_v, mode=time_mode)
            if t_v.size >= 2:
                gaps.append(float(np.median(np.diff(t_v))))

        # If we cannot estimate, fall back to 0.5s
        med_gap = float(np.median(gaps)) if gaps else 0.5

        # Heuristic: merge gap should be larger than step size so adjacent alerts merge
        merge_gap_s = max(0.25, 2.0 * med_gap)

    per_video_details: Dict[str, Any] = {}

    # ------------------------------------------------------------
    # Per-video aggregation loop
    # ------------------------------------------------------------
    for v in unique_vids:
        mv = vids_arr == v
        if not mv.any():
            continue

        # Sort windows by time order (w_start)
        idx = np.argsort(ws_arr[mv])

        p_v = probs[mv][idx]
        y_v = y_true[mv][idx]
        ws_v = ws_arr[mv][idx]
        we_v = we_arr[mv][idx]

        fps_v = float(np.median(fps_arr[mv])) if np.isfinite(fps_arr[mv]).any() else float(fps_default)
        fps_v = fps_v if fps_v > 0 else float(fps_default)

        t_v = times_from_windows(ws_v, we_v, fps_v, mode=time_mode)

        # Duration estimate:
        # - If windows cover the entire clip with stride S, this is accurate.
        # - If windows are subsampled (e.g., mined windows only), this is an underestimate.
        duration_s = float((we_v.max() - ws_v.min() + 1) / max(1e-6, fps_v))
        total_duration_s += max(0.0, duration_s)

        # Align signals
        q_v = qs[mv][idx] if qs is not None else None
        l_v = ls[mv][idx] if ls is not None else None
        m_v = ms[mv][idx] if ms is not None else None

        # Compute event-level metrics and raw event details
        em, detail = event_metrics_from_windows(
            p_v,
            y_v,
            t_v,
            alert_cfg,
            duration_s=duration_s,
            merge_gap_s=float(merge_gap_s),
            overlap_slack_s=float(overlap_slack_s),
            lying_score=l_v,
            motion_score=m_v,
            quality_score=q_v,
        )

        # Also compute state occupancy (clear/suspect/alert per window)
        st = classify_states(
            p_v,
            t_v,
            alert_cfg,
            lying_score=l_v,
            motion_score=m_v,
            quality_score=q_v,
        )

        per_video_details[v] = {
            "event_metrics": em.to_dict(),
            "detail": detail,
            "state_counts": {
                "n_windows": int(t_v.size),
                "clear": int(np.sum(st["clear"])),
                "suspect": int(np.sum(st["suspect"])),
                "pending": int(np.sum(st.get("pending", np.zeros_like(st["alert"], dtype=bool)))),
                "alert": int(np.sum(st["alert"])),
            },
        }

        # Sum totals
        gt_total += int(em.n_gt_events)
        matched_gt_total += int(em.n_matched_gt)
        alert_total += int(em.n_alert_events)
        true_alert_total += int(em.n_true_alerts)
        false_alert_total += int(em.n_false_alerts)

        if np.isfinite(em.mean_delay_s):
            delays.append(float(em.mean_delay_s))

        state_totals["n_windows"] += int(t_v.size)
        state_totals["clear"] += int(np.sum(st["clear"]))
        state_totals["suspect"] += int(np.sum(st["suspect"]))
        state_totals["pending"] += int(np.sum(st.get("pending", np.zeros_like(st["alert"], dtype=bool))))
        state_totals["alert"] += int(np.sum(st["alert"]))

    # ------------------------------------------------------------
    # Final aggregate summary
    # ------------------------------------------------------------
    recall = float(matched_gt_total / gt_total) if gt_total > 0 else float("nan")
    precision = float(true_alert_total / alert_total) if alert_total > 0 else float("nan")
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )

    dur_h = total_duration_s / 3600.0 if total_duration_s > 0 else float("nan")
    dur_d = total_duration_s / 86400.0 if total_duration_s > 0 else float("nan")
    fa_h = float(false_alert_total / dur_h) if np.isfinite(dur_h) and dur_h > 0 else float("nan")
    fa_d = float(false_alert_total / dur_d) if np.isfinite(dur_d) and dur_d > 0 else float("nan")

    return {
        "summary": {
            "event_recall": recall,
            "event_precision": precision,
            "event_f1": f1,
            "false_alerts_per_hour": fa_h,
            "false_alerts_per_day": fa_d,
            "mean_delay_s": float(np.mean(delays)) if delays else float("nan"),
            "median_delay_s": float(np.median(delays)) if delays else float("nan"),
            "n_gt_events": int(gt_total),
            "n_alert_events": int(alert_total),
            "n_true_alerts": int(true_alert_total),
            "n_false_alerts": int(false_alert_total),
            "total_duration_s": float(total_duration_s),
        },
        "state_totals": state_totals,
        "per_video": per_video_details,
        "merge_gap_s": float(merge_gap_s) if merge_gap_s is not None else None,
    }


# ============================================================
# 5) Main
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--arch", choices=["tcn", "gcn"], required=True)
    ap.add_argument("--windows_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_json", required=True)

    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--prefer_ema", type=int, default=1)

    # Optional: ops yaml from eval/fit_ops.py
    ap.add_argument("--ops_yaml", default="", help="Optional ops yaml from eval/fit_ops.py")
    ap.add_argument("--op", choices=["op1", "op2", "op3"], default="op2")

    # Calibration overrides
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--calibration_yaml", default="")

    # Time mapping & event merge settings
    ap.add_argument("--time_mode", choices=["start", "center", "end"], default="center")
    ap.add_argument("--merge_gap_s", type=float, default=-1.0)  # <=0 => auto
    ap.add_argument("--overlap_slack_s", type=float, default=0.0)

    # Fallback fps if metadata is missing/broken
    ap.add_argument("--fps_default", type=float, default=30.0)

    args = ap.parse_args()

    arch = str(args.arch).lower().strip()

    # ------------------------------------------------------------
    # Load ops yaml (optional)
    # ------------------------------------------------------------
    ops = None
    if str(args.ops_yaml).strip():
        ops = yaml_load_simple(args.ops_yaml)

    # ------------------------------------------------------------
    # Load checkpoint bundle and rebuild model
    # ------------------------------------------------------------
    bundle = load_ckpt(args.ckpt, map_location="cpu")
    model_cfg = get_cfg(bundle, "model_cfg", default={})

    raw_feat = get_cfg(bundle, "feat_cfg", default={})
    if hasattr(raw_feat, "to_dict"):
        try:
            raw_feat = raw_feat.to_dict()
        except Exception:
            pass
    feat_cfg = FeatCfg.from_dict(raw_feat if isinstance(raw_feat, dict) else {})

    # two-stream is only meaningful for GCN arch
    two_stream = bool(model_cfg.get("two_stream", False)) if arch == "gcn" else False

    # Keep fps_default consistent with training if stored in checkpoint
    fps_default = float(get_cfg(bundle, "data_cfg", default={}).get("fps_default", args.fps_default))

    device = pick_device()
    model = build_model(arch, model_cfg, feat_cfg, fps_default=fps_default).to(device)

    # EMA-safe loading strategy:
    # - base state_dict contains parameters + buffers (BatchNorm running stats)
    # - EMA state_dict often contains ONLY parameters
    # So we load base first, then overwrite parameters with EMA params.
    sd = get_state_dict(bundle, prefer_ema=False)  # base params + buffers
    if bool(int(args.prefer_ema)):
        ema_sd = bundle.get("ema_state_dict", None) or bundle.get("ema", None)
        if isinstance(ema_sd, dict) and len(ema_sd) > 0:
            sd = dict(sd)      # copy so we don't mutate bundle
            sd.update(ema_sd)  # overwrite params with EMA params
    model.load_state_dict(sd, strict=True)
    model.eval()

    # ------------------------------------------------------------
    # Temperature scaling priority
    # ------------------------------------------------------------
    if float(args.temperature) > 0:
        T = float(args.temperature)
        cal_source = "cli"
    elif str(args.calibration_yaml).strip():
        T = load_temperature(args.calibration_yaml, default=1.0)
        cal_source = "yaml"
    elif isinstance(ops, dict) and isinstance(ops.get("calibration"), dict) and "temperature" in ops["calibration"]:
        T = float(ops["calibration"]["temperature"])
        cal_source = "ops_yaml"
    else:
        T = 1.0
        cal_source = "none"

    # ------------------------------------------------------------
    # Alert configuration
    # ------------------------------------------------------------
    if isinstance(ops, dict) and isinstance(ops.get("alert_cfg"), dict):
        alert_cfg = AlertCfg.from_dict(ops["alert_cfg"])

        # Overwrite thresholds using the selected operating point if present
        if isinstance(ops.get("ops"), dict) and isinstance(ops["ops"].get(args.op), dict):
            oc = ops["ops"][args.op]
            d = alert_cfg.to_dict()
            d["tau_high"] = float(oc.get("tau_high", alert_cfg.tau_high))
            d["tau_low"] = float(oc.get("tau_low", alert_cfg.tau_low))
            alert_cfg = AlertCfg.from_dict(d)
    else:
        # fallback defaults
        alert_cfg = AlertCfg()

    merge_gap_s = None if float(args.merge_gap_s) <= 0 else float(args.merge_gap_s)

    # ------------------------------------------------------------
    # Load windows dataset and run inference
    # ------------------------------------------------------------
    ds = Windows(
        args.windows_dir,
        feat_cfg=feat_cfg,
        fps_default=fps_default,
        arch=arch,
        two_stream=two_stream,
    )
    loader = DataLoader(ds, batch_size=int(args.batch), shuffle=False, num_workers=0, collate_fn=_collate)

    logits, y_true, vids, ws, we, fps_list, qs, ls, ms = infer_logits(model, loader, device, arch, two_stream)

    # Convert logits -> calibrated probs
    probs = _sigmoid_np(apply_temperature(logits, float(T)))

    # Window-level metrics: ap_auc will ignore any labels not in {0,1}
    window_metrics = ap_auc(probs, y_true)

    # Event-level metrics under full alert policy
    event_report = aggregate_event_metrics(
        probs,
        y_true,
        vids,
        ws,
        we,
        fps_list,
        qs,
        ls,
        ms,
        alert_cfg=alert_cfg,
        merge_gap_s=merge_gap_s,
        overlap_slack_s=float(args.overlap_slack_s),
        time_mode=str(args.time_mode),
        fps_default=float(fps_default),
    )

    out = {
        "arch": arch,
        "ckpt": str(args.ckpt),
        "windows_dir": str(args.windows_dir),
        "prefer_ema": bool(int(args.prefer_ema)),
        "calibration": {"temperature": float(T), "source": str(cal_source)},
        "feat_cfg": feat_cfg.to_dict(),
        "alert_cfg": alert_cfg.to_dict(),
        "window_metrics": window_metrics,
        "event_metrics": event_report["summary"],
        "state_totals": event_report["state_totals"],
        "merge_gap_s": event_report["merge_gap_s"],
        "total_duration_s": event_report["summary"].get("total_duration_s"),
        "per_video": event_report["per_video"],
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(out), f, indent=2, allow_nan=False)
    print(f"[ok] wrote: {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
