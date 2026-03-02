#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval/score_unlabeled_alert_rate.py

Score an *unlabeled* windows directory and estimate false alert rate using
the real-time alert policy (EMA + k-of-n + hysteresis + cooldown).

Output: JSON summary with FA/hour and FA/day per video, plus totals.

IMPORTANT CONVENTION:
  - w_end is INCLUSIVE (last frame index of the window).
  - Any duration computed from indices MUST use:
        duration_s = (w_end - w_start + 1) / fps
  - For timestamps used in alert policy, we match metrics.py exactly by using:
        core.alerting.times_from_windows(mode="center")  => (ws+we)/2/fps
"""

from __future__ import annotations
import os as _os
import sys as _sys

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fall_detection.core.ckpt import load_ckpt, get_cfg
from fall_detection.core.features import FeatCfg, read_window_npz, build_tcn_input, build_canonical_input, split_gcn_two_stream
from fall_detection.core.models import build_model, pick_device, logits_1d
from fall_detection.core.confirm import confirm_scores_window
from fall_detection.core.alerting import AlertCfg, detect_alert_events, classify_states, times_from_windows


@dataclass
class MetaLite:
    video_id: str
    w_start: int
    w_end: int
    fps: float
    lying_score: float
    motion_score: float


class UnlabeledWindows(Dataset):
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

        # Build canonical features + derived mask.
        Xc, mask_used = build_canonical_input(
            joints_xy=joints,
            motion_xy=motion,
            conf=conf,
            mask=mask,
            fps=fps,
            feat_cfg=self.feat_cfg,
        )

        if self.arch == "tcn":
            X = build_tcn_input(Xc, self.feat_cfg)
        else:
            X = Xc
            if self.two_stream:
                X = split_gcn_two_stream(X, self.feat_cfg)

        # Confirm scores (computed from window signal; used by alert policy if enabled).
        lying_score = float(getattr(meta, "lying_score", 0.0))
        motion_score = float(getattr(meta, "motion_score", 0.0))
        if lying_score == 0.0 and motion_score == 0.0:
            try:
                ls, ms = confirm_scores_window(joints, mask_used, fps=float(fps))
                lying_score = float(ls) if np.isfinite(ls) else 0.0
                motion_score = float(ms) if np.isfinite(ms) else 0.0
            except Exception:
                # Keep zeros; alert policy will fall back to probability-only confirm if needed.
                pass

        m = MetaLite(
            video_id=str(meta.video_id),
            w_start=int(meta.w_start),
            w_end=int(meta.w_end),
            fps=float(meta.fps),
            lying_score=lying_score,
            motion_score=motion_score,
        )
        return X, m



def _collate(batch):
    Xs, metas = zip(*batch)
    return list(Xs), list(metas)


def _times_from_windows(ws: np.ndarray, we: np.ndarray, fps: float) -> np.ndarray:
    """Window timestamps aligned with metrics.py (w_end is INCLUSIVE).

    We use the same implementation as core.alerting.times_from_windows with mode='center':
      t = (w_start + w_end) / 2 / fps
    """
    return times_from_windows(ws, we, float(fps), mode="center")


@torch.no_grad()
def infer_probs(model, loader, device, arch: str, two_stream: bool):
    probs = []
    vids, ws, we, fps = [], [], [], []
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
            ls_list.append(float(m.lying_score))
            ms_list.append(float(m.motion_score))

    return (
        (np.concatenate(probs) if probs else np.array([])),
        vids,
        np.asarray(ws),
        np.asarray(we),
        np.asarray(fps, dtype=float),
        np.asarray(ls_list, dtype=np.float32),
        np.asarray(ms_list, dtype=np.float32),
    )


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

    model = build_model(arch, model_cfg_d, feat_cfg, fps_default=fps_default).to(device)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()

    ds = UnlabeledWindows(args.win_dir, feat_cfg=feat_cfg, fps_default=fps_default, arch=arch, two_stream=two_stream)
    loader = DataLoader(ds, batch_size=int(args.batch), shuffle=False, num_workers=0, collate_fn=_collate)

    probs, vids, ws, we, fps_arr, ls_arr, ms_arr = infer_probs(model, loader, device, arch, two_stream)

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

    vids_arr = np.asarray(vids)
    out: Dict[str, Any] = {
        "arch": arch,
        "ckpt": args.ckpt,
        "win_dir": args.win_dir,
        "alert_cfg": alert_cfg.to_dict(),
        "per_video": {},
        "total": {},
    }

    total_alerts = 0
    total_dur_s = 0.0
    total_state = {"n_windows": 0, "clear": 0, "suspect": 0, "alert": 0, "suspect_time_s": 0.0, "alert_time_s": 0.0}

    for v in list(dict.fromkeys(vids)):
        mv = vids_arr == v
        idx = np.argsort(ws[mv])
        p_v = probs[mv][idx]
        ws_v = ws[mv][idx]
        we_v = we[mv][idx]
        ls_v = ls_arr[mv][idx]
        ms_v = ms_arr[mv][idx]
        fps_v = float(np.median(fps_arr[mv])) if np.isfinite(fps_arr[mv]).any() else fps_default

        t_v = _times_from_windows(ws_v, we_v, fps_v)

        # Duration MUST honor inclusive w_end: +1 frame.
        duration_s = float((we_v.max() - ws_v.min() + 1) / max(1e-6, fps_v))

        # Duration guard: skip degenerate videos to avoid infinite/unstable FA rates.
        if not np.isfinite(duration_s) or duration_s < 1.0:
            out["per_video"][v] = {
                "skipped": True,
                "reason": f"duration_s too small ({duration_s})",
                "n_windows": int(len(ws_v)),
            }
            continue

        st = classify_states(p_v, t_v, alert_cfg, lying_score=ls_v, motion_score=ms_v)
        _alert_mask, events = detect_alert_events(p_v, t_v, alert_cfg, lying_score=ls_v, motion_score=ms_v)

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
