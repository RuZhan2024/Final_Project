#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval/score_unlabeled_alert_rate.py

Score an *unlabeled* windows directory and estimate false alert rate using
the real-time alert policy (EMA + k-of-n + hysteresis + cooldown).

Output: JSON summary with FA/hour and FA/day per video, plus totals.

Typical:
  python eval/score_unlabeled_alert_rate.py \
    --win_dir data/processed/le2i/windows_W48_S12/test_unlabeled \
    --ckpt outputs/le2i_tcn_W48S12/best.pt \
    --out_json reports/le2i_unlabeled_alerts.json
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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.ckpt import load_ckpt, get_cfg
from core.features import FeatCfg, read_window_npz, build_tcn_input, build_gcn_input
from core.models import build_model, pick_device, logits_1d
from core.alerting import AlertCfg, detect_alert_events, classify_states


class UnlabeledWindows(Dataset):
    def __init__(self, root: str, feat_cfg: FeatCfg, fps_default: float, arch: str, two_stream: bool):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No npz in {root}")
        self.feat_cfg = feat_cfg
        self.fps_default = float(fps_default)
        self.arch = str(arch).lower()
        self.two_stream = bool(two_stream)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        joints, motion, conf, mask, fps, meta = read_window_npz(p, fps_default=self.fps_default)
        if self.arch == "tcn":
            X, _ = build_tcn_input(joints, motion, conf, mask, fps, self.feat_cfg)
            return X.astype(np.float32), meta
        X, _ = build_gcn_input(joints, motion, conf, mask, fps, self.feat_cfg)
        if self.two_stream:
            xy = X[..., 0:2]
            conf1 = X[..., -1:] if (self.feat_cfg.use_conf_channel) else None
            xj = np.concatenate([xy, conf1], axis=-1) if conf1 is not None else xy
            xm = X[..., 2:4] if self.feat_cfg.use_motion else np.zeros_like(xy, dtype=np.float32)
            return (xj.astype(np.float32), xm.astype(np.float32)), meta
        return X.astype(np.float32), meta


def _collate(batch):
    Xs, metas = [], []
    for x, m in batch:
        Xs.append(x)
        metas.append(m)
    return Xs, metas


def _times_from_windows(ws: np.ndarray, we: np.ndarray, fps: float) -> np.ndarray:
    center = 0.5 * (ws.astype(np.float32) + we.astype(np.float32) + 1.0)
    return center / max(1e-6, float(fps))


@torch.no_grad()
def infer_probs(model, loader, device, arch: str, two_stream: bool):
    probs = []
    vids, ws, we, fps = [], [], [], []
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
        # Always flatten to 1-D (defensive against (B,1) logits).
        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs.append(p)
        for m in metas:
            vids.append(m.video_id or os.path.splitext(os.path.basename(m.path))[0])
            ws.append(int(m.w_start))
            we.append(int(m.w_end))
            fps.append(float(m.fps))
    return (np.concatenate(probs) if probs else np.array([]), vids, np.asarray(ws), np.asarray(we), np.asarray(fps, dtype=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--batch", type=int, default=256)

    # alert cfg
    ap.add_argument("--ema_alpha", type=float, default=0.20)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tau_high", type=float, default=0.90)
    ap.add_argument("--tau_low", type=float, default=0.70)
    ap.add_argument("--cooldown_s", type=float, default=30.0)

    # Optional confirm args (Makefile may pass them). These map to core.alerting.AlertCfg.
    ap.add_argument("--confirm", type=int, default=0)
    ap.add_argument("--confirm_s", type=float, default=2.0)
    ap.add_argument("--confirm_min_lying", type=float, default=0.65)
    ap.add_argument("--confirm_max_motion", type=float, default=0.08)
    ap.add_argument("--confirm_require_low", type=int, default=1)

    args = ap.parse_args()

    device = pick_device()
    bundle = load_ckpt(args.ckpt, map_location=device)
    arch_ck, model_cfg_d, feat_cfg_d, data_cfg_d = get_cfg(bundle)

    arch = str(arch_ck).lower()
    feat_cfg = FeatCfg.from_dict(feat_cfg_d)
    fps_default = float(data_cfg_d.get("fps_default", 30.0))
    two_stream = bool(model_cfg_d.get("two_stream", False))

    # build model dims from one sample
    sample_files = sorted(glob.glob(os.path.join(args.win_dir, "*.npz")))
    if not sample_files:
        raise SystemExit(f"No npz in {args.win_dir}")

    joints, motion, conf, mask, fps, meta = read_window_npz(sample_files[0], fps_default=fps_default)
    if arch == "tcn":
        X0, _ = build_tcn_input(joints, motion, conf, mask, fps, feat_cfg)
        model = build_model("tcn", model_cfg_d, in_ch=int(X0.shape[1])).to(device)
    else:
        X0, _ = build_gcn_input(joints, motion, conf, mask, fps, feat_cfg)
        if two_stream:
            xy = X0[..., 0:2]
            conf1 = X0[..., -1:] if feat_cfg.use_conf_channel else None
            xj = np.concatenate([xy, conf1], axis=-1) if conf1 is not None else xy
            xm = X0[..., 2:4] if feat_cfg.use_motion else np.zeros_like(xy, dtype=np.float32)
            model = build_model("gcn", model_cfg_d, in_ch=0, num_joints=int(X0.shape[1]), in_feats=0, in_feats_j=int(xj.shape[-1]), in_feats_m=int(xm.shape[-1])).to(device)
        else:
            model = build_model("gcn", model_cfg_d, in_ch=0, num_joints=int(X0.shape[1]), in_feats=int(X0.shape[-1])).to(device)

    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()

    ds = UnlabeledWindows(args.win_dir, feat_cfg=feat_cfg, fps_default=fps_default, arch=arch, two_stream=two_stream)
    loader = DataLoader(ds, batch_size=int(args.batch), shuffle=False, num_workers=0, collate_fn=_collate)

    probs, vids, ws, we, fps_arr = infer_probs(model, loader, device, arch, two_stream)

    alert_cfg = AlertCfg(
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

    # per-video alerts
    vids_arr = np.asarray(vids)
    out = {"arch": arch, "ckpt": args.ckpt, "win_dir": args.win_dir, "alert_cfg": alert_cfg.to_dict(), "per_video": {}, "total": {}}

    total_alerts = 0
    total_dur_s = 0.0
    total_state = {"n_windows": 0, "clear": 0, "suspect": 0, "alert": 0, "suspect_time_s": 0.0, "alert_time_s": 0.0}

    for v in list(dict.fromkeys(vids)):
        mv = vids_arr == v
        idx = np.argsort(ws[mv])
        p_v = probs[mv][idx]
        ws_v = ws[mv][idx]
        we_v = we[mv][idx]
        fps_v = float(np.median(fps_arr[mv])) if np.isfinite(fps_arr[mv]).any() else fps_default
        t_v = _times_from_windows(ws_v, we_v, fps_v)
        duration_s = float((we_v.max() - ws_v.min() + 1) / max(1e-6, fps_v))

        st = classify_states(p_v, t_v, alert_cfg)
        _alert_mask, events = detect_alert_events(p_v, t_v, alert_cfg)
        n = int(len(events))
        fa_hour = float(n / (duration_s / 3600.0)) if duration_s > 0 else float("nan")
        fa_day = float(n / (duration_s / 86400.0)) if duration_s > 0 else float("nan")

        # Approximate time in each state using median step in t_v.
        dt = float(np.median(np.diff(t_v))) if t_v.size >= 2 else 0.0
        n_clear = int(np.sum(st["clear"]))
        n_suspect = int(np.sum(st["suspect"]))
        n_alert = int(np.sum(st["alert"]))
        tot = int(t_v.size)

        out["per_video"][v] = {
            "n_alert_events": n,
            "duration_s": duration_s,
            "fa_per_hour": fa_hour,
            "fa_per_day": fa_day,
            "state_counts": {
                "n_windows": tot,
                "clear": n_clear,
                "suspect": n_suspect,
                "alert": n_alert,
                "suspect_frac": float(n_suspect / tot) if tot > 0 else float("nan"),
                "alert_frac": float(n_alert / tot) if tot > 0 else float("nan"),
                "suspect_time_s": float(n_suspect * dt),
                "alert_time_s": float(n_alert * dt),
            },
            "first_3_events": [e.to_dict() for e in events[:3]],
        }

        total_alerts += n
        total_dur_s += duration_s
        total_state["n_windows"] += tot
        total_state["clear"] += n_clear
        total_state["suspect"] += n_suspect
        total_state["alert"] += n_alert
        total_state["suspect_time_s"] += float(n_suspect * dt)
        total_state["alert_time_s"] += float(n_alert * dt)

    out["total"] = {
        "n_alert_events": int(total_alerts),
        "duration_s": float(total_dur_s),
        "fa_per_hour": float(total_alerts / (total_dur_s / 3600.0)) if total_dur_s > 0 else float("nan"),
        "fa_per_day": float(total_alerts / (total_dur_s / 86400.0)) if total_dur_s > 0 else float("nan"),
        "state_counts": {
            **total_state,
            "suspect_frac": float(total_state["suspect"] / max(1, total_state["n_windows"])),
            "alert_frac": float(total_state["alert"] / max(1, total_state["n_windows"])),
        },
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[ok] wrote: {args.out_json}")


if __name__ == "__main__":
    main()
