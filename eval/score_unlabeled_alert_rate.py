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
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.ckpt import load_ckpt, get_cfg
from core.features import FeatCfg, read_window_npz, build_tcn_input, build_canonical_input, split_gcn_two_stream
from core.models import build_model, pick_device, logits_1d, validate_model_input_dims
from core.confirm import confirm_scores_window
from core.alerting import AlertCfg, detect_alert_events_from_smoothed, ema_smooth, times_from_windows


@dataclass
class MetaLite:
    video_id: str
    w_start: int
    w_end: int
    fps: float
    lying_score: float
    motion_score: float


class UnlabeledWindows(Dataset):
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
            X = torch.from_numpy(build_tcn_input(Xc, self.feat_cfg))
        else:
            X = torch.from_numpy(Xc)
            if self.two_stream:
                xj, xm = split_gcn_two_stream(Xc, self.feat_cfg)
                X = (torch.from_numpy(xj), torch.from_numpy(xm))

        # Confirm scores (computed from window signal; used by alert policy if enabled).
        lying_score = float(getattr(meta, "lying_score", 0.0))
        motion_score = float(getattr(meta, "motion_score", 0.0))
        if self.compute_confirm_scores and lying_score == 0.0 and motion_score == 0.0:
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
    first = Xs[0]
    if isinstance(first, tuple):
        feats = Xs
        xj, xm = zip(*feats)
        xj = torch.stack(xj, 0)
        xm = torch.stack(xm, 0)
        return (xj, xm), list(metas)
    xb = torch.stack(Xs, 0)
    return xb, list(metas)


def _times_from_windows(ws: np.ndarray, we: np.ndarray, fps: float) -> np.ndarray:
    """Window timestamps aligned with metrics.py (w_end is INCLUSIVE).

    We use the same implementation as core.alerting.times_from_windows with mode='center':
      t = (w_start + w_end) / 2 / fps
    """
    return times_from_windows(ws, we, float(fps), mode="center")


def _build_video_groups(vids: Sequence[str], ws: np.ndarray) -> list[tuple[str, np.ndarray]]:
    vids_arr = np.asarray(vids, dtype=object)
    if vids_arr.size == 0:
        return []
    uniq, first_idx, inv = np.unique(vids_arr, return_index=True, return_inverse=True)
    order = np.argsort(first_idx, kind="mergesort")
    inv_i64 = inv.astype(np.int64, copy=False)
    by_group = np.argsort(inv_i64, kind="mergesort")
    counts = np.bincount(inv_i64, minlength=uniq.size)
    offsets = np.empty((counts.size + 1,), dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    groups: list[tuple[str, np.ndarray]] = []
    for u in order:
        uu = int(u)
        idx = by_group[int(offsets[uu]) : int(offsets[uu + 1])]
        if idx.size == 0:
            continue
        sort_idx = np.argsort(ws[idx], kind="mergesort")
        groups.append((str(uniq[u]), idx[sort_idx]))
    return groups


@torch.inference_mode()
def infer_probs(model, loader, device, arch: str, two_stream: bool):
    n_total = len(loader.dataset) if hasattr(loader, "dataset") else 0
    use_prealloc = n_total > 0
    probs_buf = np.empty((n_total,), dtype=np.float32) if use_prealloc else None
    vids_buf = np.empty((n_total,), dtype=object) if use_prealloc else None
    ws_buf = np.empty((n_total,), dtype=np.int32) if use_prealloc else None
    we_buf = np.empty((n_total,), dtype=np.int32) if use_prealloc else None
    fps_buf = np.empty((n_total,), dtype=np.float64) if use_prealloc else None
    ls_buf = np.empty((n_total,), dtype=np.float32) if use_prealloc else None
    ms_buf = np.empty((n_total,), dtype=np.float32) if use_prealloc else None
    ptr = 0

    probs: list[np.ndarray] = []
    vids: list[str] = []
    ws: list[int] = []
    we: list[int] = []
    fps: list[float] = []
    ls_list: list[float] = []
    ms_list: list[float] = []
    use_non_blocking = isinstance(device, torch.device) and device.type in {"cuda", "mps"}

    for Xs, metas in loader:
        if arch == "tcn":
            xb = Xs.to(device=device, dtype=torch.float32, non_blocking=use_non_blocking)
            logits = logits_1d(model(xb))
        else:
            if two_stream:
                xj, xm = Xs
                xj = xj.to(device=device, dtype=torch.float32, non_blocking=use_non_blocking)
                xm = xm.to(device=device, dtype=torch.float32, non_blocking=use_non_blocking)
                logits = logits_1d(model(xj, xm))
            else:
                xb = Xs.to(device=device, dtype=torch.float32, non_blocking=use_non_blocking)
                logits = logits_1d(model(xb))

        p = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        bsz = p.size
        if (
            use_prealloc
            and probs_buf is not None
            and vids_buf is not None
            and ws_buf is not None
            and we_buf is not None
            and fps_buf is not None
            and ls_buf is not None
            and ms_buf is not None
            and (ptr + bsz) <= n_total
        ):
            probs_buf[ptr:ptr + bsz] = p
            vids_buf[ptr:ptr + bsz] = np.asarray([m.video_id for m in metas], dtype=object)
            ws_buf[ptr:ptr + bsz] = np.fromiter((int(m.w_start) for m in metas), dtype=np.int32, count=bsz)
            we_buf[ptr:ptr + bsz] = np.fromiter((int(m.w_end) for m in metas), dtype=np.int32, count=bsz)
            fps_buf[ptr:ptr + bsz] = np.fromiter((float(m.fps) for m in metas), dtype=np.float64, count=bsz)
            ls_buf[ptr:ptr + bsz] = np.fromiter((float(m.lying_score) for m in metas), dtype=np.float32, count=bsz)
            ms_buf[ptr:ptr + bsz] = np.fromiter((float(m.motion_score) for m in metas), dtype=np.float32, count=bsz)
            ptr += bsz
        else:
            probs.append(p)
            vids.extend(m.video_id for m in metas)
            ws.extend(int(m.w_start) for m in metas)
            we.extend(int(m.w_end) for m in metas)
            fps.extend(float(m.fps) for m in metas)
            ls_list.extend(float(m.lying_score) for m in metas)
            ms_list.extend(float(m.motion_score) for m in metas)

    if use_prealloc and ptr > 0 and not probs:
        probs_out = probs_buf[:ptr]
        vids_out = vids_buf[:ptr]
        ws_out = ws_buf[:ptr]
        we_out = we_buf[:ptr]
        fps_out = fps_buf[:ptr]
        ls_out = ls_buf[:ptr]
        ms_out = ms_buf[:ptr]
    elif use_prealloc and ptr > 0:
        vids_tail = np.asarray(vids, dtype=object)
        ws_tail = np.asarray(ws, dtype=np.int32)
        we_tail = np.asarray(we, dtype=np.int32)
        fps_tail = np.asarray(fps, dtype=float)
        ls_tail = np.asarray(ls_list, dtype=np.float32)
        ms_tail = np.asarray(ms_list, dtype=np.float32)
        probs_out = np.concatenate((probs_buf[:ptr], *probs), axis=0)
        vids_out = np.concatenate([vids_buf[:ptr], vids_tail], axis=0)
        ws_out = np.concatenate([ws_buf[:ptr], ws_tail], axis=0)
        we_out = np.concatenate([we_buf[:ptr], we_tail], axis=0)
        fps_out = np.concatenate([fps_buf[:ptr], fps_tail], axis=0)
        ls_out = np.concatenate([ls_buf[:ptr], ls_tail], axis=0)
        ms_out = np.concatenate([ms_buf[:ptr], ms_tail], axis=0)
    else:
        probs_out = np.concatenate(probs, axis=0) if probs else np.array([], dtype=np.float32)
        vids_out = np.asarray(vids, dtype=object)
        ws_out = np.asarray(ws, dtype=np.int32)
        we_out = np.asarray(we, dtype=np.int32)
        fps_out = np.asarray(fps, dtype=float)
        ls_out = np.asarray(ls_list, dtype=np.float32)
        ms_out = np.asarray(ms_list, dtype=np.float32)

    return (
        probs_out,
        vids_out,
        ws_out,
        we_out,
        fps_out,
        ls_out,
        ms_out,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)

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

    need_confirm_scores = bool(int(args.confirm)) and (float(args.confirm_s) > 0.0)
    ds = UnlabeledWindows(
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
    nw = max(0, int(args.num_workers))
    pin_memory = isinstance(device, torch.device) and device.type == "cuda"
    persistent_workers = nw > 0
    if isinstance(device, torch.device) and device.type == "mps":
        persistent_workers = False
    loader_kwargs: Dict[str, Any] = {
        "batch_size": int(args.batch),
        "shuffle": False,
        "num_workers": nw,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "collate_fn": _collate,
    }
    if nw > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(ds, **loader_kwargs)

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
    groups = _build_video_groups(vids_arr, ws)
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

    for v, idx_full in groups:
        if idx_full.size < 1:
            continue
        p_v = probs[idx_full]
        ws_v = ws[idx_full]
        we_v = we[idx_full]
        ls_v = ls_arr[idx_full]
        ms_v = ms_arr[idx_full]
        fps_slice = fps_arr[idx_full]
        fps_v = float(np.median(fps_slice)) if np.isfinite(fps_slice).any() else fps_default

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

        ps_v = ema_smooth(p_v, alert_cfg.ema_alpha)
        alert_mask, events = detect_alert_events_from_smoothed(
            ps_v, t_v, alert_cfg, lying_score=ls_v, motion_score=ms_v
        )
        suspect_mask = (~alert_mask) & (ps_v >= float(alert_cfg.tau_low)) & (ps_v < float(alert_cfg.tau_high))
        clear_mask = (~alert_mask) & (ps_v < float(alert_cfg.tau_low))

        n = int(len(events))
        fa_hour = float(n / (duration_s / 3600.0)) if duration_s > 0 else float("nan")
        fa_day = float(n / (duration_s / 86400.0)) if duration_s > 0 else float("nan")

        # Approximate time in each state using median step in t_v.
        dt = float(np.median(np.diff(t_v))) if t_v.size >= 2 else 0.0
        n_clear = int(np.sum(clear_mask))
        n_suspect = int(np.sum(suspect_mask))
        n_alert = int(np.sum(alert_mask))
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
