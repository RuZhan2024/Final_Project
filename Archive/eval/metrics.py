#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval/metrics.py

Evaluate a trained checkpoint on a windows directory, producing a JSON report.

This version evaluates REAL deployment behavior:
  - Threshold sweep is under the FULL alert policy (EMA + k-of-n + hysteresis + cooldown)
  - FA/24h counts FALSE alert events only (alerts not overlapping GT fall events)

Also supports evaluating OP-1/OP-2/OP-3 from an ops YAML (produced by eval/fit_ops.py).
"""


from __future__ import annotations

# -------------------------
# Path bootstrap (so `from core.*` works when running as a script)
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

from core.ckpt import load_ckpt, get_cfg
from core.features import FeatCfg, read_window_npz, build_tcn_input, build_gcn_input
from core.metrics import ap_auc
from core.models import build_model, pick_device, logits_1d
from core.yamlio import yaml_load_simple
from core.alerting import AlertCfg, times_from_windows, event_metrics_from_windows, sweep_alert_policy_from_windows, classify_states


@dataclass
class MetaRow:
    path: str
    video_id: str
    w_start: int
    w_end: int
    fps: float
    y: int


class Windows(Dataset):
    def __init__(self, root: str, feat_cfg: FeatCfg, fps_default: float, arch: str, two_stream: bool):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz in {root}")
        self.feat_cfg = feat_cfg
        self.fps_default = float(fps_default)
        self.arch = str(arch).lower()
        self.two_stream = bool(two_stream)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        joints, motion, conf, mask, fps, meta = read_window_npz(p, fps_default=self.fps_default)
        y = int(meta.y) if meta.y is not None else int(meta.label) if meta.label is not None else 0

        if self.arch == "tcn":
            X, _ = build_tcn_input(joints, motion, conf, mask, fps=float(fps), feat_cfg=self.feat_cfg)
        else:
            X, _ = build_gcn_input(joints, motion, conf, mask, fps=float(fps), feat_cfg=self.feat_cfg)
            if self.two_stream:
                xy = X[..., 0:2]
                if self.feat_cfg.use_conf_channel:
                    c = X[..., -1:]
                    xj = np.concatenate([xy, c], axis=-1)
                else:
                    xj = xy

                if self.feat_cfg.use_motion and X.shape[-1] >= 4:
                    xm = X[..., 2:4]
                else:
                    xm = np.zeros_like(xy)

                X = (xj, xm)

        m = MetaRow(
            path=str(p),
            video_id=str(meta.video_id or meta.seq_id or os.path.splitext(os.path.basename(p))[0]),
            w_start=int(meta.w_start),
            w_end=int(meta.w_end),
            fps=float(fps),
            y=int(y),
        )
        return X, np.int64(y), m


def _collate(batch):
    Xs, ys, metas = zip(*batch)
    return list(Xs), np.asarray(ys, dtype=np.int64), list(metas)


@torch.no_grad()
def infer_probs(model, loader, device, arch: str, two_stream: bool):
    probs = []
    y_true = []
    vids, ws, we, fps = [], [], [], []
    for Xs, ys, metas in loader:
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
        y_true.append(ys)

        for m in metas:
            vids.append(m.video_id)
            ws.append(int(m.w_start))
            we.append(int(m.w_end))
            fps.append(float(m.fps))

    return (
        np.concatenate(probs) if probs else np.array([], dtype=np.float32),
        np.concatenate(y_true) if y_true else np.array([], dtype=np.int32),
        vids, ws, we, fps
    )


def _aggregate_event_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    vids: List[str],
    ws: List[int],
    we: List[int],
    fps_list: List[float],
    *,
    alert_cfg: AlertCfg,
    merge_gap_s: Optional[float],
    overlap_slack_s: float,
    time_mode: str,
    fps_default: float,
) -> Dict[str, Any]:
    vids_arr = np.asarray(vids).astype(str)
    ws_arr = np.asarray(ws, dtype=np.int32)
    we_arr = np.asarray(we, dtype=np.int32)
    fps_arr = np.asarray(fps_list, dtype=np.float32)

    unique_vids = list(dict.fromkeys(list(vids_arr)))
    per_video: Dict[str, Any] = {}

    total_duration_s = 0.0
    gt_total = 0
    matched_gt_total = 0
    alert_total = 0
    true_alert_total = 0
    false_alert_total = 0
    delays: List[float] = []
    state_totals = {"n_windows": 0, "clear": 0, "suspect": 0, "alert": 0}

    # auto merge gap if not provided
    if merge_gap_s is None:
        gaps = []
        for v in unique_vids:
            mv = vids_arr == v
            if not mv.any():
                continue
            idx = np.argsort(ws_arr[mv])
            fps_v = float(np.median(fps_arr[mv])) if np.isfinite(fps_arr[mv]).any() else float(fps_default)
            if fps_v <= 0:
                fps_v = float(fps_default)
            t_v = times_from_windows(ws_arr[mv][idx], we_arr[mv][idx], fps_v, mode=time_mode)
            if t_v.size >= 2:
                gaps.append(float(np.median(np.diff(t_v))))
        med_gap = float(np.median(gaps)) if gaps else 0.5
        merge_gap_s = max(0.25, 2.0 * med_gap)

    for v in unique_vids:
        mv = vids_arr == v
        if not mv.any():
            continue
        idx = np.argsort(ws_arr[mv])
        p_v = probs[mv][idx]
        y_v = y_true[mv][idx]
        ws_v = ws_arr[mv][idx]
        we_v = we_arr[mv][idx]
        fps_v = float(np.median(fps_arr[mv])) if np.isfinite(fps_arr[mv]).any() else float(fps_default)
        if fps_v <= 0:
            fps_v = float(fps_default)
        t_v = times_from_windows(ws_v, we_v, fps_v, mode=time_mode)
        duration_s = float((we_v.max() - ws_v.min() + 1) / max(1e-6, fps_v))
        total_duration_s += max(0.0, duration_s)

        em, detail = event_metrics_from_windows(
            p_v, y_v, t_v, alert_cfg,
            duration_s=duration_s,
            merge_gap_s=float(merge_gap_s),
            overlap_slack_s=float(overlap_slack_s),
        )
        st = classify_states(p_v, t_v, alert_cfg)
        n_windows = int(t_v.size)
        n_clear = int(np.sum(st["clear"]))
        n_suspect = int(np.sum(st["suspect"]))
        n_alert = int(np.sum(st["alert"]))
        state_totals["n_windows"] += n_windows
        state_totals["clear"] += n_clear
        state_totals["suspect"] += n_suspect
        state_totals["alert"] += n_alert

        per_video[v] = {
            "event_metrics": em.to_dict(),
            "detail": detail,
            "state_counts": {
                "n_windows": n_windows,
                "clear": n_clear,
                "suspect": n_suspect,
                "alert": n_alert,
                "suspect_frac": float(n_suspect / n_windows) if n_windows > 0 else float("nan"),
                "alert_frac": float(n_alert / n_windows) if n_windows > 0 else float("nan"),
            },
        }

        gt_total += int(em.n_gt_events)
        matched_gt_total += int(em.n_matched_gt)
        alert_total += int(em.n_alert_events)
        true_alert_total += int(em.n_true_alerts)
        false_alert_total += int(em.n_false_alerts)
        if np.isfinite(em.mean_delay_s):
            delays.append(float(em.mean_delay_s))

    dur_h = float(total_duration_s) / 3600.0 if total_duration_s > 0 else float("nan")
    dur_d = float(total_duration_s) / 86400.0 if total_duration_s > 0 else float("nan")

    recall = float(matched_gt_total / gt_total) if gt_total > 0 else float("nan")
    precision = float(true_alert_total / alert_total) if alert_total > 0 else float("nan")
    f1 = float(2.0 * precision * recall / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else float("nan")
    fa_h = float(false_alert_total / dur_h) if np.isfinite(dur_h) and dur_h > 0 else float("nan")
    fa_d = float(false_alert_total / dur_d) if np.isfinite(dur_d) and dur_d > 0 else float("nan")

    return {
        "alert_cfg": alert_cfg.to_dict(),
        "micro_event_recall": recall,
        "micro_event_precision": precision,
        "micro_event_f1": f1,
        "false_alerts_per_hour": fa_h,
        "fa24h": fa_d,
        "mean_delay_s": float(np.mean(delays)) if delays else float("nan"),
        "n_gt_events": int(gt_total),
        "n_alert_events": int(alert_total),
        "n_true_alerts": int(true_alert_total),
        "n_false_alerts": int(false_alert_total),
        "total_duration_s": float(total_duration_s),
        "merge_gap_s": float(merge_gap_s),
        "state_counts": {
            **state_totals,
            "suspect_frac": float(state_totals["suspect"] / max(1, state_totals["n_windows"])),
            "alert_frac": float(state_totals["alert"] / max(1, state_totals["n_windows"])),
        },
        "per_video": per_video,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    # NOTE: For Makefile compatibility we accept both --win_dir (preferred)
    # and the legacy flag name --test_dir (same meaning).
    ap.add_argument("--win_dir", default="", help="e.g. data/processed/le2i/windows_W32_S8/test")
    ap.add_argument("--fa_dir", default="", help="Optional: windows dir used only to estimate FA/24h (long unlabeled/negative streams).")
    ap.add_argument("--test_dir", dest="win_dir", default="", help=argparse.SUPPRESS)
    # Backward-compat aliases sometimes used by Make targets / notebooks.
    ap.add_argument("--eval_dir", dest="win_dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--arch", type=str, default="", help=argparse.SUPPRESS)
    ap.add_argument("--dataset_name", type=str, default="", help=argparse.SUPPRESS)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_json", required=True)

    # Optional feature flags (accepted for backwards compatibility with older Makefiles).
    # Normally, we use the checkpoint's feat_cfg. If the checkpoint has no feat_cfg, we
    # will fall back to these CLI values (if provided).
    ap.add_argument("--center", type=str, default=None)
    ap.add_argument("--use_motion", type=int, default=None)
    ap.add_argument("--use_conf_channel", type=int, default=None)
    ap.add_argument("--motion_scale_by_fps", type=int, default=None)
    ap.add_argument("--conf_gate", type=float, default=None)
    ap.add_argument("--use_precomputed_mask", type=int, default=None)
    ap.add_argument("--batch", type=int, default=256)

    # Sweep range for tau_high
    ap.add_argument("--thr_min", type=float, default=0.001)
    ap.add_argument("--thr_max", type=float, default=0.999)
    ap.add_argument("--thr_step", type=float, default=0.01)

    # Alert policy base (used for sweep if no ops_yaml, or as fallback)
    ap.add_argument("--ema_alpha", type=float, default=0.20)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--cooldown_s", type=float, default=30.0)
    ap.add_argument("--tau_low_ratio", type=float, default=0.80)

    ap.add_argument("--time_mode", choices=["start", "center", "end"], default="center")
    ap.add_argument("--merge_gap_s", type=float, default=-1.0, help="<=0 => auto from data")
    ap.add_argument("--overlap_slack_s", type=float, default=0.0)

    ap.add_argument("--ops_yaml", default="", help="ops YAML from eval/fit_ops.py; evaluates OPs if provided")
    ap.add_argument("--ops", dest="ops_yaml", default="", help=argparse.SUPPRESS)
    ap.add_argument("--fps_default", type=float, default=30.0)

    # Extra flags sometimes passed by Make targets (not needed for metrics,
    # but we accept them so users can keep a single 'eval' command template).
    ap.add_argument("--pose_npz_dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--stride_frames_hint", default="", help=argparse.SUPPRESS)

    args = ap.parse_args()
    if not str(args.win_dir).strip():
        ap.error("the following arguments are required: --win_dir")

    bundle = load_ckpt(args.ckpt, map_location="cpu")
    model_cfg = get_cfg(bundle, "model_cfg", default={})
    raw_feat = get_cfg(bundle, "feat_cfg", default={})
    cli_feat = {}
    for k in ["center","use_motion","use_conf_channel","motion_scale_by_fps","conf_gate","use_precomputed_mask"]:
        v = getattr(args, k, None)
        if v is not None:
            cli_feat[k] = v
    if hasattr(raw_feat, "to_dict"):
        try:
            raw_feat = raw_feat.to_dict()
        except Exception:
            pass
    if (not raw_feat) and cli_feat:
        raw_feat = cli_feat
    elif raw_feat and cli_feat and isinstance(raw_feat, dict):
        for k, v in cli_feat.items():
            if k in raw_feat and str(raw_feat.get(k)) != str(v):
                print(f"[warn] CLI feat flag {k}={v} ignored (ckpt has {raw_feat.get(k)}).")
    feat_cfg = FeatCfg.from_dict(raw_feat)
    # Determine architecture.
    ckpt_arch = get_cfg(bundle, "arch", default=model_cfg.get("arch", "")) or ""
    ckpt_arch = str(ckpt_arch).lower().strip()
    cli_arch = str(args.arch or "").lower().strip()
    arch = ckpt_arch or cli_arch or "tcn"
    if cli_arch and ckpt_arch and cli_arch != ckpt_arch:
        print(f"[warn] CLI --arch={cli_arch} ignored (ckpt arch is {ckpt_arch}).")

    two_stream = bool(model_cfg.get("two_stream", False)) if arch == "gcn" else False
    fps_default = float(get_cfg(bundle, "data_cfg", default={}).get("fps_default", args.fps_default))

    device = pick_device()
    model = build_model(arch, model_cfg, feat_cfg, fps_default=fps_default).to(device)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()

    ds = Windows(args.win_dir, feat_cfg=feat_cfg, fps_default=fps_default, arch=arch, two_stream=two_stream)
    loader = DataLoader(ds, batch_size=int(args.batch), shuffle=False, num_workers=0, collate_fn=_collate)

    probs, y_true, vids, ws, we, fps_list = infer_probs(model, loader, device, arch, two_stream)
    fa_probs = fa_vids = fa_ws = fa_we = fa_fps_list = None
    if str(args.fa_dir).strip():
        ds_fa = Windows(args.fa_dir, feat_cfg=feat_cfg, fps_default=fps_default, arch=arch, two_stream=two_stream)
        loader_fa = DataLoader(ds_fa, batch_size=int(args.batch), shuffle=False, num_workers=0, collate_fn=_collate)
        fa_probs, _fa_y, fa_vids, fa_ws, fa_we, fa_fps_list = infer_probs(model, loader_fa, device, arch, two_stream)
        print(f"[info] FA-estimation set: {len(ds_fa)} windows from {args.fa_dir}")


    # Base metrics (window-level): AP + ROC-AUC (if available)
    base = {
        "arch": arch,
        "ckpt": str(args.ckpt),
        "win_dir": str(args.win_dir),
        "fa_dir": str(args.fa_dir) if str(args.fa_dir).strip() else "",
        "n_windows": int(probs.size),
        "feat_cfg": feat_cfg.to_dict(),
    }

    # core.metrics.ap_auc returns a dict {"ap": ..., "auc": ...}
    # (do NOT unpack it, otherwise you'd unpack dict keys: "ap", "auc").
    apm = ap_auc(probs, y_true)
    ap_val = apm.get("ap", None) if isinstance(apm, dict) else None
    auc_val = apm.get("auc", None) if isinstance(apm, dict) else None

    # Convert NaN to None for cleaner JSON.
    if ap_val is None or (isinstance(ap_val, float) and not np.isfinite(ap_val)):
        base["window_ap"] = None
    else:
        base["window_ap"] = float(ap_val)

    if auc_val is None or (isinstance(auc_val, float) and not np.isfinite(auc_val)):
        base["window_auc"] = None
    else:
        base["window_auc"] = float(auc_val)

    # Load ops yaml if provided (also sets base alert policy for sweep)
    ops = None
    alert_base = None
    if args.ops_yaml:
        ops = yaml_load_simple(args.ops_yaml)
        alert_base = AlertCfg.from_dict(ops.get("alert_cfg") or {})

    if alert_base is None:
        alert_base = AlertCfg(
            ema_alpha=float(args.ema_alpha),
            k=int(args.k),
            n=int(args.n),
            tau_high=0.5,
            tau_low=0.4,
            cooldown_s=float(args.cooldown_s),
        )

    merge_gap_s = None if float(args.merge_gap_s) <= 0 else float(args.merge_gap_s)

    sweep, sweep_meta = sweep_alert_policy_from_windows(
        probs, y_true, vids, ws, we, fps_list,
        alert_base=alert_base,
        thr_min=float(args.thr_min),
        thr_max=float(args.thr_max),
        thr_step=float(args.thr_step),
        tau_low_ratio=float(args.tau_low_ratio),
        merge_gap_s=merge_gap_s,
        overlap_slack_s=float(args.overlap_slack_s),
        time_mode=str(args.time_mode),
        fa_probs=fa_probs,
        fa_video_ids=fa_vids,
        fa_w_start=fa_ws,
        fa_w_end=fa_we,
        fa_fps=fa_fps_list,
        fps_default=float(fps_default),
    )

    report: Dict[str, Any] = {**base, "sweep": sweep, "sweep_meta": sweep_meta}

    # Optional: evaluate OP-1/OP-2/OP-3 from YAML (policy metrics under each OP thresholds)
    if ops:
        report["ops_yaml"] = str(args.ops_yaml)
        report["ops_eval"] = {}
        base_cfg = AlertCfg.from_dict(ops.get("alert_cfg") or {})
        ops_points = ops.get("ops") or {}
        for name, op in ops_points.items():
            cfg = AlertCfg(
                ema_alpha=float(base_cfg.ema_alpha),
                k=int(base_cfg.k),
                n=int(base_cfg.n),
                tau_high=float(op.get("tau_high", base_cfg.tau_high)),
                tau_low=float(op.get("tau_low", base_cfg.tau_low)),
                cooldown_s=float(base_cfg.cooldown_s),
                confirm=bool(base_cfg.confirm),
                confirm_s=float(base_cfg.confirm_s),
                confirm_min_lying=float(base_cfg.confirm_min_lying),
                confirm_max_motion=float(base_cfg.confirm_max_motion),
                confirm_require_low=bool(base_cfg.confirm_require_low),
            )
            ev = _aggregate_event_metrics(
                probs, y_true, vids, ws, we, fps_list,
                alert_cfg=cfg,
                merge_gap_s=merge_gap_s,
                overlap_slack_s=float(args.overlap_slack_s),
                time_mode=str(args.time_mode),
                fps_default=float(fps_default),
            )
            report["ops_eval"][name] = ev

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] wrote report → {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
