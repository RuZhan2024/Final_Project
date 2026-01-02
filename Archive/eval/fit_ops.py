#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval/fit_ops.py

Fit operating points for REAL-TIME deployment on a validation windows split.

Key difference vs older versions:
  - Threshold sweep is done under the FULL alert policy:
      sigmoid -> EMA -> k-of-n -> hysteresis -> cooldown
  - FA/24h counts FALSE alerts only (alerts not overlapping GT fall events).

Output YAML is consumed by eval/metrics.py and deployment.

YAML schema (compact):
  arch: tcn|gcn
  ckpt: path
  feat_cfg: {...}              # from checkpoint bundle
  alert_cfg: {...}             # base policy (ema/k/n/cooldown); tau from OP2 by default
  ops:
    op1: {tau_high, tau_low, precision, recall, f1, fa24h, mean_delay_s}
    op2: ...
    op3: ...
  sweep_meta: {...}
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
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.ckpt import load_ckpt, get_cfg
from core.features import FeatCfg, read_window_npz, build_tcn_input, build_gcn_input
from core.models import build_model, pick_device, logits_1d
from core.metrics import pareto_frontier
from core.alerting import AlertCfg, sweep_alert_policy_from_windows
from core.yamlio import yaml_dump_simple


@dataclass
class MetaRow:
    path: str
    video_id: str
    w_start: int
    w_end: int
    fps: float


class ValWindows(Dataset):
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
        # y: labeled windows should store y=0/1
        y = int(meta.y) if meta.y is not None else int(meta.label) if meta.label is not None else 0

        if self.arch == "tcn":
            X, _ = build_tcn_input(joints, motion, conf, mask, fps=float(fps), feat_cfg=self.feat_cfg)
        else:
            X, _ = build_gcn_input(joints, motion, conf, mask, fps=float(fps), feat_cfg=self.feat_cfg)
            if self.two_stream:
                # Two-stream GCN expects separate inputs:
                #   joints stream: (x,y[,conf])
                #   motion stream: (dx,dy) (or zeros if motion is disabled)
                xy = X[..., 0:2]

                if getattr(self.feat_cfg, 'use_motion', False) and X.shape[-1] >= 4:
                    xm = X[..., 2:4]
                else:
                    xm = np.zeros_like(xy)

                if getattr(self.feat_cfg, 'use_conf_channel', False) and X.shape[-1] >= 3:
                    # conf is always the last channel when present
                    conf_ch = X[..., -1:]
                    xj = np.concatenate([xy, conf_ch], axis=-1)
                else:
                    xj = xy

                X = (xj, xm)

        m = MetaRow(
            path=str(p),
            video_id=str(meta.video_id or meta.seq_id or os.path.splitext(os.path.basename(p))[0]),
            w_start=int(meta.w_start),
            w_end=int(meta.w_end),
            fps=float(fps),
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


def _pick_op1(sweep: Dict[str, List[float]], *, target_recall: float) -> int:
    r = np.asarray(sweep["recall"], dtype=float)
    fa = np.asarray(sweep["fa24h"], dtype=float)
    ok = np.isfinite(r) & np.isfinite(fa) & (r >= float(target_recall))
    if ok.any():
        idx = np.where(ok)[0]
        # among those, pick minimum FA; tie-breaker: higher recall
        best = idx[np.lexsort((-r[idx], fa[idx]))][0]
        return int(best)
    # fallback: pick max recall, tie: min FA
    ok2 = np.isfinite(r) & np.isfinite(fa)
    if ok2.any():
        idx = np.where(ok2)[0]
        best = idx[np.lexsort((fa[idx], -r[idx]))][0]
        return int(best)
    # If recall is all-NaN (e.g., no GT positives in val), fall back to most conservative threshold.
    ok3 = np.isfinite(fa)
    if ok3.any():
        idx = np.where(ok3)[0]
        best = idx[np.argmin(fa[idx])]
        return int(best)
    return 0


def _pick_op2(sweep: Dict[str, List[float]]) -> int:
    f1 = np.asarray(sweep.get("f1", []), dtype=float)
    fa = np.asarray(sweep.get("fa24h", []), dtype=float)
    r = np.asarray(sweep.get("recall", []), dtype=float)
    ok = np.isfinite(f1) & np.isfinite(fa)
    if ok.any():
        idx = np.where(ok)[0]
        # max f1; tie min FA; tie max recall
        best = idx[np.lexsort((-r[idx], fa[idx], -f1[idx]))][0]
        return int(best)
    # fallback: max recall, tie min FA
    ok2 = np.isfinite(r) & np.isfinite(fa)
    if ok2.any():
        idx = np.where(ok2)[0]
        best = idx[np.lexsort((fa[idx], -r[idx]))][0]
        return int(best)
    return 0


def _pick_op3(sweep: Dict[str, List[float]], *, max_fa24h: float) -> int:
    r = np.asarray(sweep["recall"], dtype=float)
    fa = np.asarray(sweep["fa24h"], dtype=float)
    ok = np.isfinite(r) & np.isfinite(fa) & (fa <= float(max_fa24h))
    if ok.any():
        idx = np.where(ok)[0]
        # max recall; tie min FA
        best = idx[np.lexsort((fa[idx], -r[idx]))][0]
        return int(best)
    # fallback: min FA (most conservative)
    ok2 = np.isfinite(fa)
    if ok2.any():
        idx = np.where(ok2)[0]
        best = idx[np.argmin(fa[idx])]
        return int(best)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["tcn", "gcn"], required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--fa_dir", default="", help="Optional: windows dir used only to estimate FA/24h (long unlabeled/negative streams).")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=256)

    # Optional feature flags (accepted for backwards compatibility with older Makefiles).
    # Normally, we use the checkpoint's feat_cfg. If the checkpoint has no feat_cfg, we
    # will fall back to these CLI values (if provided).
    ap.add_argument("--center", type=str, default=None)
    ap.add_argument("--use_motion", type=int, default=None)
    ap.add_argument("--use_conf_channel", type=int, default=None)
    ap.add_argument("--motion_scale_by_fps", type=int, default=None)
    ap.add_argument("--conf_gate", type=float, default=None)
    ap.add_argument("--use_precomputed_mask", type=int, default=None)

    # Sweep range for tau_high (kept as thr_* names for backwards compat)
    ap.add_argument("--thr_min", type=float, default=0.001)
    ap.add_argument("--thr_max", type=float, default=0.999)
    ap.add_argument("--thr_step", type=float, default=0.01)

    # Real-time policy (base); tau_high/tau_low are swept/picked per OP
    ap.add_argument("--ema_alpha", type=float, default=0.20)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--cooldown_s", type=float, default=30.0)
    ap.add_argument("--tau_low_ratio", type=float, default=0.80, help="tau_low = tau_high * ratio")

    # Optional confirmation stage (accepted because Makefile may pass these).
    # These are used by core.alerting.AlertCfg when detect_alert_events(...) runs.
    ap.add_argument("--confirm", type=int, default=0, help="1 => enable confirmation stage")
    ap.add_argument("--confirm_s", type=float, default=2.0)
    ap.add_argument("--confirm_min_lying", type=float, default=0.65)
    ap.add_argument("--confirm_max_motion", type=float, default=0.08)
    ap.add_argument("--confirm_require_low", type=int, default=1)

    # Time mapping + GT/alert overlap definition
    ap.add_argument("--time_mode", choices=["start", "center", "end"], default="center")
    ap.add_argument("--merge_gap_s", type=float, default=-1.0, help="GT merge gap seconds; <=0 => auto from data")
    ap.add_argument("--overlap_slack_s", type=float, default=0.0)

    # OP selection policy
    ap.add_argument("--op1_recall", type=float, default=0.95)
    ap.add_argument("--op3_fa24h", type=float, default=1.0)

    # Info-only / legacy args (ignored but accepted for old Makefiles)
    ap.add_argument("--pose_npz_dir", default="")
    ap.add_argument("--stride_frames_hint", type=int, default=0)
    ap.add_argument("--fps_default", type=float, default=30.0)

    args = ap.parse_args()

    bundle = load_ckpt(args.ckpt, map_location="cpu")
    arch = str(args.arch).lower()

    model_cfg = get_cfg(bundle, "model_cfg", default={})
    raw_feat = get_cfg(bundle, "feat_cfg", default={})
    # CLI feature flags (if provided). Used only when checkpoint lacks feat_cfg.
    cli_feat = {}
    for k in ["center","use_motion","use_conf_channel","motion_scale_by_fps","conf_gate","use_precomputed_mask"]:
        v = getattr(args, k, None)
        if v is not None:
            cli_feat[k] = v
    # Normalize raw_feat to dict if possible
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
    two_stream = bool(model_cfg.get("two_stream", False)) if arch == "gcn" else False

    fps_default = float(get_cfg(bundle, "data_cfg", default={}).get("fps_default", args.fps_default))
    device = pick_device()
    model = build_model(arch, model_cfg, feat_cfg, fps_default=fps_default).to(device)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()

    ds = ValWindows(args.val_dir, feat_cfg=feat_cfg, fps_default=fps_default, arch=arch, two_stream=two_stream)
    loader = DataLoader(ds, batch_size=int(args.batch), shuffle=False, num_workers=0, collate_fn=_collate)

    probs, y_true, vids, ws, we, fps_list = infer_probs(model, loader, device, arch, two_stream)

    fa_probs = fa_vids = fa_ws = fa_we = fa_fps_list = None
    if str(args.fa_dir).strip():
        ds_fa = ValWindows(args.fa_dir, feat_cfg=feat_cfg, fps_default=fps_default, arch=arch, two_stream=two_stream)
        loader_fa = DataLoader(ds_fa, batch_size=int(args.batch), shuffle=False, num_workers=0, collate_fn=_collate)
        fa_probs, _fa_y, fa_vids, fa_ws, fa_we, fa_fps_list = infer_probs(model, loader_fa, device, arch, two_stream)
        print(f"[info] FA-estimation set: {len(ds_fa)} windows from {args.fa_dir}")

    # Sanity: if validation contains no positives, recall is not identifiable; still write ops but warn.
    if int(np.sum(np.asarray(y_true, dtype=np.int64) > 0)) == 0:
        print('[warn] Validation split has 0 positive windows. OP-1/OP-2 based on recall are not meaningful; fix your split/windowing.')


    alert_base = AlertCfg(
        ema_alpha=float(args.ema_alpha),
        k=int(args.k),
        n=int(args.n),
        tau_high=0.5,  # placeholder; swept
        tau_low=0.4,
        cooldown_s=float(args.cooldown_s),
        confirm=bool(int(args.confirm)),
        confirm_s=float(args.confirm_s),
        confirm_min_lying=float(args.confirm_min_lying),
        confirm_max_motion=float(args.confirm_max_motion),
        confirm_require_low=bool(int(args.confirm_require_low)),
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

    # Pareto frontier (minimise FA, maximise recall)
    # core.metrics.pareto_frontier signature is (recall, x) and it returns (idx, recall_arr, x_arr)
    fa = np.asarray(sweep["fa24h"], dtype=float)
    rec = np.asarray(sweep["recall"], dtype=float)
    pf_idx, _, _ = pareto_frontier(rec, fa)

    i1 = _pick_op1(sweep, target_recall=float(args.op1_recall))
    i2 = _pick_op2(sweep)
    i3 = _pick_op3(sweep, max_fa24h=float(args.op3_fa24h))

    def _op_at(i: int) -> Dict[str, Any]:
        th = float(sweep["thr"][i])
        tl = float(sweep["tau_low"][i])
        return {
            "tau_high": float(sweep["thr"][i]),
            "tau_low": float(sweep["tau_low"][i]),
            "uncertain_band": {"low": float(tl), "high": float(th)},
            "precision": float(sweep["precision"][i]),
            "recall": float(sweep["recall"][i]),
            "f1": float(sweep["f1"][i]),
            "fa24h": float(sweep["fa24h"][i]),
            "fa_per_hour": float(sweep["fa_per_hour"][i]),
            "mean_delay_s": float(sweep["mean_delay_s"][i]),
            "median_delay_s": float(sweep["median_delay_s"][i]),
        }

    ops = {"op1": _op_at(i1), "op2": _op_at(i2), "op3": _op_at(i3)}

    # Warn if any selected tau_high hits sweep boundary (usually means sweep range too narrow).
    eps = 1e-9
    for _k, _op in ops.items():
        th = float(_op.get("tau_high", float('nan')))
        hit = (abs(th - float(args.thr_min)) < eps) or (abs(th - float(args.thr_max)) < eps)
        _op["hits_boundary"] = bool(hit)
        if hit:
            print(f"[warn] {_k} selected tau_high={th:.4f} hits sweep boundary [{args.thr_min:.4f},{args.thr_max:.4f}] — consider lowering --thr_min or raising --thr_max")


    # base alert_cfg stored for convenience; set tau to OP2
    alert_cfg = AlertCfg(
        ema_alpha=float(args.ema_alpha),
        k=int(args.k),
        n=int(args.n),
        tau_high=float(ops["op2"]["tau_high"]),
        tau_low=float(ops["op2"]["tau_low"]),
        cooldown_s=float(args.cooldown_s),
        confirm=bool(int(args.confirm)),
        confirm_s=float(args.confirm_s),
        confirm_min_lying=float(args.confirm_min_lying),
        confirm_max_motion=float(args.confirm_max_motion),
        confirm_require_low=bool(int(args.confirm_require_low)),
    )

    out = {
        "arch": arch,
        "ckpt": str(args.ckpt),
        "val_dir": str(args.val_dir),
        "fa_dir": str(args.fa_dir) if str(args.fa_dir).strip() else "",
        "feat_cfg": feat_cfg.to_dict(),
        "alert_cfg": alert_cfg.to_dict(),
        "ops": ops,
        "sweep_meta": sweep_meta,
        "pareto_idx": [int(i) for i in pf_idx],
    }

    # yaml_dump_simple writes to a path (no PyYAML dependency). Older helper
    # versions returned a string; call the current signature explicitly.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    yaml_dump_simple(out, str(args.out))
    print(f"[OK] wrote ops yaml → {args.out}")
    print(f"     OP1: recall={ops['op1']['recall']:.3f} fa24h={ops['op1']['fa24h']:.3f} tau={ops['op1']['tau_high']:.2f}/{ops['op1']['tau_low']:.2f}")
    print(f"     OP2: recall={ops['op2']['recall']:.3f} fa24h={ops['op2']['fa24h']:.3f} tau={ops['op2']['tau_high']:.2f}/{ops['op2']['tau_low']:.2f}")
    print(f"     OP3: recall={ops['op3']['recall']:.3f} fa24h={ops['op3']['fa24h']:.3f} tau={ops['op3']['tau_high']:.2f}/{ops['op3']['tau_low']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
