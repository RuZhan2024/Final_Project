#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/replay_eval.py

Replay evaluation (deployment-matching) using window NPZ files.

What this script is for
-----------------------
In deployment, you don't care about "window-level accuracy" alone.
You care about:
- Did an alert trigger during a real fall event? (event recall)
- How many false alarms happen per day? (FA/day)
- How long after fall starts do we alert? (delay)

This script simulates deployment behavior by:
1) grouping windows by video_id (clip)
2) sorting windows in time order (w_start)
3) producing probabilities p(t) for each window
4) applying the alert policy (core/alerting.detect_alert_events)
5) computing event-level metrics per video and aggregated.

Inputs
------
- windows_dir: directory containing *.npz windows (recursive)
  Each window NPZ should contain:
    joints or xy, conf (optional), motion (optional), mask (optional),
    y (0/1 or -1), fps, w_start, w_end, and a video identifier.

- ckpt: a checkpoint BUNDLE (.pt) from training (TCN or GCN)

Optional:
- ops_yaml + op: load tau_high/tau_low from fitted operating points

Outputs
-------
- out_json: aggregated metrics + per-video breakdown

IMPORTANT NOTE ABOUT "REPLAY"
-----------------------------
This script assumes your windows_dir contains windows generated using your
deploy schedule (W,S) across the full clip (not mined/random windows).
If you run it on mined windows, the "duration" and "FA/day" can be misleading.
"""

from __future__ import annotations

# ------------------------------------------------------------
# Path bootstrap so running directly works:
#   python eval/replay_eval.py ...
# ------------------------------------------------------------
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.alerting import AlertCfg, classify_states, event_metrics_from_windows, times_from_windows
from core.calibration import apply_temperature, load_temperature
from core.ckpt import get_cfg, get_state_dict, load_ckpt
from core.features import FeatCfg, build_gcn_input, build_tcn_input, read_window_npz
from core.models import build_model, logits_1d, pick_device
from core.signals import compute_window_signals
from core.yamlio import yaml_load_simple


# ============================================================
# 1) Stable sigmoid for numpy
# ============================================================
def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    """
    Stable sigmoid:
      sigmoid(x) = 1 / (1 + exp(-x))

    We clip x to avoid overflow in exp for large logits.
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


# ============================================================
# 2) Dataset: load windows and build model inputs
# ============================================================
@dataclass
class MetaRow:
    """Metadata per window used for grouping/time mapping."""
    path: str
    video_id: str
    w_start: int
    w_end: int
    fps: float
    y: int


class WindowNPZDataset(Dataset):
    """
    Reads window NPZ files and returns:
      X: model input (TCN / GCN / Two-stream GCN)
      y: int label (0/1/-1)
      meta: MetaRow
      q/l/m: signals (quality, lying, motion) used by alert policy
    """

    def __init__(self, windows_dir: str, feat_cfg: FeatCfg, fps_default: float, arch: str, two_stream: bool):
        self.files = sorted(glob.glob(os.path.join(windows_dir, "**", "*.npz"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No .npz under: {windows_dir}")

        self.feat_cfg = feat_cfg
        self.fps_default = float(fps_default)
        self.arch = str(arch).lower().strip()
        self.two_stream = bool(two_stream)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]

        joints, motion, conf, mask, fps, meta = read_window_npz(p, fps_default=self.fps_default)
        y = int(meta.y) if meta.y is not None else -1

        # Signals computed on the same representation the model consumes
        sig = compute_window_signals(joints, motion, conf, mask, float(fps), self.feat_cfg)

        # Build model input
        if self.arch == "tcn":
            X, _ = build_tcn_input(joints, motion, conf, mask, float(fps), self.feat_cfg)  # [T,C]
        else:
            X, _ = build_gcn_input(joints, motion, conf, mask, float(fps), self.feat_cfg)  # [T,V,F]

            if self.two_stream:
                xy = X[..., 0:2]  # always first 2 channels
                if self.feat_cfg.use_conf_channel:
                    c = X[..., -1:]              # [T,V,1]
                    xj = np.concatenate([xy, c], axis=-1)  # [T,V,3]
                else:
                    xj = xy

                if self.feat_cfg.use_motion and X.shape[-1] >= 4:
                    xm = X[..., 2:4]  # [T,V,2]
                else:
                    xm = np.zeros_like(xy, dtype=np.float32)

                X = (xj.astype(np.float32), xm.astype(np.float32))

        # video_id fallback chain:
        vid = meta.video_id or meta.seq_id or os.path.splitext(os.path.basename(p))[0]

        m = MetaRow(
            path=str(p),
            video_id=str(vid),
            w_start=int(meta.w_start),
            w_end=int(meta.w_end),
            fps=float(fps),
            y=int(y),
        )

        return X, np.int64(y), m, np.float32(sig.quality), np.float32(sig.lying), np.float32(sig.motion)


def _collate(batch):
    """
    Keep X as list (because it may contain tuples for two-stream).
    Stack other items into numpy arrays.
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
# 3) Inference helper
# ============================================================
@torch.no_grad()
def infer_logits(model, loader, device, arch: str, two_stream: bool):
    """
    Returns:
      logits: [N]
      y_true: [N]
      vids/ws/we/fps: metadata lists length N
      qs/ls/ms: signals arrays length N
    """
    arch = str(arch).lower().strip()

    logits_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    vids: List[str] = []
    ws: List[int] = []
    we: List[int] = []
    fps_list: List[float] = []

    q_all: List[np.ndarray] = []
    l_all: List[np.ndarray] = []
    m_all: List[np.ndarray] = []

    for Xs, ys, metas, qs, ls, ms in loader:
        if arch == "tcn":
            xb = torch.from_numpy(np.stack(Xs, axis=0)).to(device)  # [B,T,C]
            logits = logits_1d(model(xb))
        else:
            if two_stream:
                xj = torch.from_numpy(np.stack([x[0] for x in Xs], axis=0)).to(device)
                xm = torch.from_numpy(np.stack([x[1] for x in Xs], axis=0)).to(device)
                logits = logits_1d(model(xj, xm))
            else:
                xb = torch.from_numpy(np.stack(Xs, axis=0)).to(device)  # [B,T,V,F]
                logits = logits_1d(model(xb))

        logits_all.append(logits.detach().cpu().numpy().reshape(-1))
        y_all.append(ys.reshape(-1))

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
        np.concatenate(y_all) if y_all else np.array([], dtype=np.int32),
        vids,
        ws,
        we,
        fps_list,
        np.concatenate(q_all) if q_all else np.array([], dtype=np.float32),
        np.concatenate(l_all) if l_all else np.array([], dtype=np.float32),
        np.concatenate(m_all) if m_all else np.array([], dtype=np.float32),
    )


# ============================================================
# 4) Replay aggregation: per-video policy + summary
# ============================================================
def replay_aggregate(
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
    time_mode: str,
    merge_gap_s: Optional[float],
    overlap_slack_s: float,
    fps_default: float,
) -> Dict[str, Any]:
    """
    Apply alert policy per video and compute:
      - per-video event metrics
      - aggregated totals and FA/day

    Why per-video:
    - replay is a streaming process within ONE clip at a time.
    - you must not mix time indices across different videos.
    """
    vids_arr = np.asarray(vids, dtype=str)
    ws_arr = np.asarray(ws, dtype=np.int32)
    we_arr = np.asarray(we, dtype=np.int32)
    fps_arr = np.asarray(fps_list, dtype=np.float32)

    unique_vids = list(dict.fromkeys(list(vids_arr)))

    # If merge_gap_s is None, estimate from median time step across videos
    if merge_gap_s is None:
        gaps: List[float] = []
        for v in unique_vids:
            mv = vids_arr == v
            if mv.sum() < 2:
                continue
            idx = np.argsort(ws_arr[mv])

            fps_v = float(np.median(fps_arr[mv])) if np.isfinite(fps_arr[mv]).any() else float(fps_default)
            fps_v = fps_v if fps_v > 0 else float(fps_default)

            t_v = times_from_windows(ws_arr[mv][idx], we_arr[mv][idx], fps_v, mode=time_mode)
            if t_v.size >= 2:
                gaps.append(float(np.median(np.diff(t_v))))
        med_gap = float(np.median(gaps)) if gaps else 0.5
        merge_gap_s = max(0.25, 2.0 * med_gap)

    total_duration_s = 0.0

    gt_total = 0
    matched_gt_total = 0
    alert_total = 0
    true_alert_total = 0
    false_alert_total = 0
    delays: List[float] = []

    state_totals = {"n_windows": 0, "clear": 0, "suspect": 0, "alert": 0}
    per_video: Dict[str, Any] = {}

    for v in unique_vids:
        mv = vids_arr == v
        if not mv.any():
            continue

        idx = np.argsort(ws_arr[mv])  # time order

        p_v = probs[mv][idx]
        y_v = y_true[mv][idx]
        ws_v = ws_arr[mv][idx]
        we_v = we_arr[mv][idx]

        fps_v = float(np.median(fps_arr[mv])) if np.isfinite(fps_arr[mv]).any() else float(fps_default)
        fps_v = fps_v if fps_v > 0 else float(fps_default)

        t_v = times_from_windows(ws_v, we_v, fps_v, mode=time_mode)

        # Duration estimate from coverage of windows
        duration_s = float((we_v.max() - ws_v.min() + 1) / max(1e-6, fps_v))
        total_duration_s += max(0.0, duration_s)

        q_v = qs[mv][idx] if qs is not None else None
        l_v = ls[mv][idx] if ls is not None else None
        m_v = ms[mv][idx] if ms is not None else None

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

        st = classify_states(
            p_v,
            t_v,
            alert_cfg,
            lying_score=l_v,
            motion_score=m_v,
            quality_score=q_v,
        )

        per_video[v] = {
            "duration_s": float(duration_s),
            "event_metrics": em.to_dict(),
            "state_counts": {
                "n_windows": int(t_v.size),
                "clear": int(np.sum(st["clear"])),
                "suspect": int(np.sum(st["suspect"])),
                "alert": int(np.sum(st["alert"])),
            },
            "detail": detail,
        }

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
        state_totals["alert"] += int(np.sum(st["alert"]))

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

    summary = {
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
    }

    return {
        "summary": summary,
        "state_totals": state_totals,
        "merge_gap_s": float(merge_gap_s) if merge_gap_s is not None else None,
        "per_video": per_video,
    }


# ============================================================
# 5) Main CLI
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--arch", choices=["tcn", "gcn"], required=True)
    ap.add_argument("--windows_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_json", required=True)

    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--prefer_ema", type=int, default=1)

    # Ops YAML (optional)
    ap.add_argument("--ops_yaml", default="")
    ap.add_argument("--op", choices=["op1", "op2", "op3"], default="op2")

    # Calibration
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--calibration_yaml", default="")

    # Alert config overrides (if you don't use ops_yaml)
    ap.add_argument("--tau_high", type=float, default=0.0, help="If >0, override tau_high")
    ap.add_argument("--tau_low", type=float, default=0.0, help="If >0, override tau_low")
    ap.add_argument("--tau_low_ratio", type=float, default=0.80)

    ap.add_argument("--ema_alpha", type=float, default=0.20)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--cooldown_s", type=float, default=30.0)

    ap.add_argument("--confirm", type=int, default=0)
    ap.add_argument("--confirm_s", type=float, default=2.0)
    ap.add_argument("--confirm_min_lying", type=float, default=0.65)
    ap.add_argument("--confirm_max_motion", type=float, default=0.08)
    ap.add_argument("--confirm_require_low", type=int, default=1)

    ap.add_argument("--quality_adapt", type=int, default=0)
    ap.add_argument("--quality_min", type=float, default=0.0)
    ap.add_argument("--quality_boost", type=float, default=0.15)
    ap.add_argument("--quality_boost_low", type=float, default=0.05)

    # Time mapping
    ap.add_argument("--time_mode", choices=["start", "center", "end"], default="center")
    ap.add_argument("--merge_gap_s", type=float, default=-1.0)  # <=0 => auto
    ap.add_argument("--overlap_slack_s", type=float, default=0.0)

    ap.add_argument("--fps_default", type=float, default=30.0)

    args = ap.parse_args()
    arch = str(args.arch).lower().strip()

    # ------------------------------------------------------------
    # Load ops yaml if provided
    # ------------------------------------------------------------
    ops = None
    if str(args.ops_yaml).strip():
        ops = yaml_load_simple(args.ops_yaml)

    # ------------------------------------------------------------
    # Load checkpoint and rebuild model
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

    two_stream = bool(model_cfg.get("two_stream", False)) if arch == "gcn" else False

    fps_default = float(get_cfg(bundle, "data_cfg", default={}).get("fps_default", args.fps_default))

    device = pick_device()
    model = build_model(arch, model_cfg, feat_cfg, fps_default=fps_default).to(device)

    # EMA-safe loading (base buffers + optional EMA params)
    sd = get_state_dict(bundle, prefer_ema=False)
    if bool(int(args.prefer_ema)):
        ema_sd = bundle.get("ema_state_dict", None) or bundle.get("ema", None)
        if isinstance(ema_sd, dict) and len(ema_sd) > 0:
            sd = dict(sd)
            sd.update(ema_sd)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # ------------------------------------------------------------
    # Temperature scaling priority
    # ------------------------------------------------------------
    if float(args.temperature) > 0:
        T = float(args.temperature)
        cal_source = "cli"
    elif str(args.calibration_yaml).strip():
        T = load_temperature(str(args.calibration_yaml).strip(), default=1.0)
        cal_source = "yaml"
    elif isinstance(ops, dict) and isinstance(ops.get("calibration"), dict) and "temperature" in ops["calibration"]:
        T = float(ops["calibration"]["temperature"])
        cal_source = "ops_yaml"
    else:
        T = 1.0
        cal_source = "none"

    # ------------------------------------------------------------
    # Build alert_cfg (ops_yaml > cli overrides > defaults)
    # ------------------------------------------------------------
    if isinstance(ops, dict) and isinstance(ops.get("alert_cfg"), dict):
        alert_cfg = AlertCfg.from_dict(ops["alert_cfg"])
    else:
        alert_cfg = AlertCfg(
            ema_alpha=float(args.ema_alpha),
            k=int(args.k),
            n=int(args.n),
            tau_high=0.90,
            tau_low=0.70,
            cooldown_s=float(args.cooldown_s),
            quality_adapt=bool(int(args.quality_adapt)),
            quality_min=float(args.quality_min),
            quality_boost=float(args.quality_boost),
            quality_boost_low=float(args.quality_boost_low),
            confirm=bool(int(args.confirm)),
            confirm_s=float(args.confirm_s),
            confirm_min_lying=float(args.confirm_min_lying),
            confirm_max_motion=float(args.confirm_max_motion),
            confirm_require_low=bool(int(args.confirm_require_low)),
        )

    # If ops_yaml contains selected operating point, override tau_high/tau_low
    if isinstance(ops, dict) and isinstance(ops.get("ops"), dict) and isinstance(ops["ops"].get(args.op), dict):
        oc = ops["ops"][args.op]
        d = alert_cfg.to_dict()
        d["tau_high"] = float(oc.get("tau_high", d.get("tau_high")))
        d["tau_low"] = float(oc.get("tau_low", d.get("tau_low")))
        alert_cfg = AlertCfg.from_dict(d)

    # CLI explicit overrides (highest priority)
    if float(args.tau_high) > 0:
        d = alert_cfg.to_dict()
        d["tau_high"] = float(args.tau_high)
        # If tau_low not explicitly set, derive it by ratio for consistency
        if float(args.tau_low) <= 0:
            d["tau_low"] = float(max(0.0, min(d["tau_high"], d["tau_high"] * float(args.tau_low_ratio))))
        alert_cfg = AlertCfg.from_dict(d)

    if float(args.tau_low) > 0:
        d = alert_cfg.to_dict()
        d["tau_low"] = float(args.tau_low)
        alert_cfg = AlertCfg.from_dict(d)

    merge_gap_s = None if float(args.merge_gap_s) <= 0 else float(args.merge_gap_s)

    # ------------------------------------------------------------
    # Run inference on windows
    # ------------------------------------------------------------
    ds = WindowNPZDataset(args.windows_dir, feat_cfg, fps_default, arch, two_stream)
    loader = DataLoader(ds, batch_size=int(args.batch), shuffle=False, num_workers=0, collate_fn=_collate)

    logits, y_true, vids, ws, we, fps_list, qs, ls, ms = infer_logits(model, loader, device, arch, two_stream)

    # logits -> calibrated probs
    probs = _sigmoid_np(apply_temperature(logits, float(T)))

    # ------------------------------------------------------------
    # Replay aggregation under alert policy
    # ------------------------------------------------------------
    rep = replay_aggregate(
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
        time_mode=str(args.time_mode),
        merge_gap_s=merge_gap_s,
        overlap_slack_s=float(args.overlap_slack_s),
        fps_default=float(fps_default),
    )

    out = {
        "arch": arch,
        "ckpt": str(args.ckpt),
        "windows_dir": str(args.windows_dir),
        "prefer_ema": bool(int(args.prefer_ema)),
        "calibration": {"temperature": float(T), "source": str(cal_source)},
        "alert_cfg": alert_cfg.to_dict(),
        "time_mode": str(args.time_mode),
        "merge_gap_s": rep["merge_gap_s"],
        "summary": rep["summary"],
        "state_totals": rep["state_totals"],
        "per_video": rep["per_video"],
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[ok] wrote replay report: {args.out_json}")

    # Quick console summary
    s = rep["summary"]
    print(
        f"[summary] recall={s['event_recall']:.3f} FA/day={s['false_alerts_per_day']:.3f} "
        f"delay_med={s['median_delay_s']:.2f}s n_gt={s['n_gt_events']} n_alert={s['n_alert_events']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
