#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/fit_ops.py

Fit Operating Points (OPs) for deployment-style alerting.

What this script does
---------------------
Given:
- a trained checkpoint (TCN or GCN)
- a validation windows folder (val_dir)
- optionally a long negative/unlabeled folder for FA estimation (fa_dir)

It will:
1) Run inference -> logits per window
2) Convert logits -> calibrated probabilities (temperature scaling)
3) Run the REAL alert policy (core/alerting) for a sweep of tau_high thresholds
4) Compute event-level metrics for each tau_high (recall, precision, f1, FA/24h, delay)
5) Choose three operating points:
   OP-1 High Recall: best FA/24h subject to recall >= target (default 0.95)
   OP-2 Balanced:    best event-level F1 (tie-break: higher recall, lower FA)
   OP-3 Low Alarm:   best recall subject to FA/24h <= target (default 1/day)
6) Write a YAML file that eval/metrics.py (and later server runtime) can consume.

Why OP fitting is NOT done on window-level AP
---------------------------------------------
AP/AUC measure probability quality, but deployment uses:
- persistence (k-of-n)
- hysteresis (tau_low)
- cooldown
- optional confirmation using (lying, motion, quality)

So we fit thresholds under the *same policy* used in deployment.

Output format
-------------
The YAML file contains:
- deploy: {fps, W, S}
- calibration: {temperature}
- alert_cfg: (base alert config)
- ops: {op1/op2/op3} with tau_high/tau_low and metrics
- sweep: list of all sweep points (for analysis/plotting)
"""

from __future__ import annotations

# ------------------------------------------------------------
# Path bootstrap so `python eval/fit_ops.py ...` works
# ------------------------------------------------------------
import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import glob
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.alerting import AlertCfg, sweep_alert_policy_from_windows, times_from_windows
from core.calibration import apply_temperature, load_temperature
from core.ckpt import get_cfg, get_state_dict, load_ckpt
from core.features import FeatCfg, build_gcn_input, build_tcn_input, read_window_npz
from core.models import build_model, logits_1d, pick_device
from core.signals import compute_window_signals
from core.yamlio import yaml_dump_simple


# ============================================================
# 1) Stable sigmoid (avoid overflow)
# ============================================================
def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    """
    Stable sigmoid for numpy arrays.

    x: logits
    returns: probs in [0,1]
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


# ============================================================
# 2) Dataset for val/fa windows
# ============================================================
@dataclass
class MetaRow:
    """Minimal meta we need for alert evaluation."""
    path: str
    video_id: str
    w_start: int
    w_end: int
    fps: float
    y: int


class ValWindows(Dataset):
    """
    Loads window NPZ files and produces model inputs + meta + signals.

    Returns:
      X:
        - for TCN: [T,C]
        - for GCN single: [T,V,F]
        - for GCN two-stream: (xj[T,V,Fj], xm[T,V,2])
      y: int label (0/1/-1)
      meta: MetaRow
      quality/lying/motion: signals used by alert policy
    """

    def __init__(self, root: str, feat_cfg: FeatCfg, fps_default: float, arch: str, two_stream: bool):
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

        joints, motion, conf, mask, fps, meta = read_window_npz(p, fps_default=self.fps_default)
        y = int(meta.y) if meta.y is not None else -1

        # Signals are computed from the SAME representation the model consumes.
        sig = compute_window_signals(joints, motion, conf, mask, float(fps), self.feat_cfg)

        # Build model input
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
                    xm = np.zeros_like(xy, dtype=np.float32)

                X = (xj.astype(np.float32), xm.astype(np.float32))

        # video_id best-effort fallback chain:
        # - explicit video_id, seq_id in NPZ meta
        # - otherwise fallback to file stem
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
    Keep X as list because it may contain tuples for two-stream.
    Stack other arrays.
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
# 3) Inference: windows -> logits + aligned metadata
# ============================================================
@torch.no_grad()
def infer_logits(model, loader, device, arch: str, two_stream: bool):
    logits_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    vids: List[str] = []
    ws: List[int] = []
    we: List[int] = []
    fps_list: List[float] = []

    q_all: List[np.ndarray] = []
    l_all: List[np.ndarray] = []
    m_all: List[np.ndarray] = []

    arch = str(arch).lower().strip()

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
# 4) Operating point pick rules (OP-1 / OP-2 / OP-3)
# ============================================================
def _pick_op1(sweep: Dict[str, List[float]], target_recall: float) -> int:
    """
    OP-1 High Recall:
    Choose the lowest FA/24h among thresholds that achieve recall >= target.
    Tie-break: higher recall.
    """
    r = np.asarray(sweep["recall"], dtype=float)
    fa = np.asarray(sweep["fa24h"], dtype=float)

    ok = np.isfinite(r) & np.isfinite(fa) & (r >= float(target_recall))
    if ok.any():
        idx = np.where(ok)[0]
        # sort by (fa asc, recall desc)
        best = idx[np.lexsort((-r[idx], fa[idx]))][0]
        return int(best)

    # fallback: pick minimal FA/24h with best recall
    ok2 = np.isfinite(r) & np.isfinite(fa)
    if ok2.any():
        idx = np.where(ok2)[0]
        best = idx[np.lexsort((fa[idx], -r[idx]))][0]
        return int(best)

    return 0


def _pick_op2(sweep: Dict[str, List[float]]) -> int:
    """
    OP-2 Balanced:
    Choose maximum F1.
    Tie-break: higher recall, lower FA/24h.

    Robustness:
    - If there are GT events in the labeled sweep, prefer configurations that
      actually produce at least one TRUE alert (n_true_alerts > 0). This prevents
      the degenerate "silent OP" where FA=0 but recall=0.
    - If *no* sweep point produces any true alerts, fall back to the old rule but
      emit a warning (usually confirm/quality gating is too strict).
    """
    f1 = np.asarray(sweep["f1"], dtype=float)
    r = np.asarray(sweep["recall"], dtype=float)
    fa = np.asarray(sweep["fa24h"], dtype=float)

    n_true = np.asarray(sweep.get("n_true_alerts", [0] * len(f1)), dtype=float)
    n_alert_events = np.asarray(sweep.get("n_alert_events", [0] * len(f1)), dtype=float)
    n_gt = np.asarray(sweep.get("n_gt_events", [0] * len(f1)), dtype=float)

    ok = np.isfinite(f1) & np.isfinite(r) & np.isfinite(fa)

    has_gt = (np.nanmax(n_gt) > 0) if n_gt.size > 0 else False
    if has_gt:
        ok_live = ok & (n_true > 0)
        if ok_live.any():
            idx = np.where(ok_live)[0]
            best = idx[np.lexsort((fa[idx], -r[idx], -f1[idx]))][0]
            return int(best)

        ok_relax = ok & (r > 0) & (n_alert_events > 0)
        if ok_relax.any():
            idx = np.where(ok_relax)[0]
            best = idx[np.lexsort((fa[idx], -r[idx], -f1[idx]))][0]
            return int(best)

        print(
            "[warn] OP-2: no sweep point produced any true alerts on labeled data. "
            "Confirm/quality gating may be too strict. Falling back to best recall.",
            file=_sys.stderr,
        )

    if ok.any():
        idx = np.where(ok)[0]
        best = idx[np.lexsort((fa[idx], -r[idx], -f1[idx]))][0]
        return int(best)

    ok2 = np.isfinite(r) & np.isfinite(fa)
    if ok2.any():
        idx = np.where(ok2)[0]
        best = idx[np.lexsort((fa[idx], -r[idx]))][0]
        return int(best)

    return 0

def _pick_op3(sweep: Dict[str, List[float]], target_fa24h: float) -> int:
    """
    OP-3 Low Alarm:
    Choose the highest recall subject to FA/24h <= target_fa24h.
    Tie-break: lower FA.
    """
    r = np.asarray(sweep["recall"], dtype=float)
    fa = np.asarray(sweep["fa24h"], dtype=float)

    ok = np.isfinite(r) & np.isfinite(fa) & (fa <= float(target_fa24h))
    if ok.any():
        idx = np.where(ok)[0]
        best = idx[np.lexsort((fa[idx], -r[idx]))][0]
        return int(best)

    # fallback: minimal FA
    ok2 = np.isfinite(fa)
    if ok2.any():
        idx = np.where(ok2)[0]
        best = idx[np.argmin(fa[idx])]
        return int(best)

    return 0


# ============================================================
# 5) Deploy spec inference (fps/W/S)
# ============================================================
def _infer_deploy_spec(val_dir: str, deploy_fps: float, deploy_w: int, deploy_s: int) -> Dict[str, Any]:
    """
    Try to infer W and S from folder name like:
      windows_W48_S12
      windows_W32_S8
    """
    name = os.path.basename(os.path.abspath(val_dir))
    mW = re.search(r"_W(\d+)", name)
    mS = re.search(r"_S(\d+)", name)

    W = int(deploy_w) if int(deploy_w) > 0 else (int(mW.group(1)) if mW else None)
    S = int(deploy_s) if int(deploy_s) > 0 else (int(mS.group(1)) if mS else None)

    fps = float(deploy_fps) if float(deploy_fps) > 0 else 30.0

    return {"fps": float(fps), "W": int(W) if W is not None else None, "S": int(S) if S is not None else None}


# ============================================================
# 6) Main
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()

    # Required inputs
    ap.add_argument("--arch", choices=["tcn", "gcn"], required=True)
    ap.add_argument("--val_dir", required=True, help="Validation windows folder (labeled).")
    ap.add_argument("--ckpt", required=True, help="Checkpoint bundle .pt from training.")
    ap.add_argument("--out", required=True, help="Output YAML file.")

    # Optional FA-only windows folder (unlabeled or negative streams)
    ap.add_argument("--fa_dir", default="", help="Optional windows folder used only for FA/24h estimation.")

    # Runtime
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--prefer_ema", type=int, default=1, help="1 => use EMA weights if present in checkpoint bundle")

    # Calibration
    ap.add_argument("--temperature", type=float, default=0.0, help="If >0, override temperature scaling")
    ap.add_argument("--calibration_yaml", default="", help="YAML containing temperature: <T>")

    # Feature overrides (backwards compatibility)
    ap.add_argument("--center", type=str, default=None)
    ap.add_argument("--use_motion", type=int, default=None)
    ap.add_argument("--use_conf_channel", type=int, default=None)
    ap.add_argument("--motion_scale_by_fps", type=int, default=None)
    ap.add_argument("--conf_gate", type=float, default=None)
    ap.add_argument("--use_precomputed_mask", type=int, default=None)

    # Alert policy parameters (base)
    ap.add_argument("--ema_alpha", type=float, default=0.20)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tau_low_ratio", type=float, default=0.80)
    ap.add_argument("--cooldown_s", type=float, default=30.0)

    # Confirmation
    ap.add_argument("--confirm", type=int, default=0)
    ap.add_argument("--confirm_s", type=float, default=2.0)
    ap.add_argument("--confirm_min_lying", type=float, default=0.65)
    ap.add_argument("--confirm_max_motion", type=float, default=0.08)
    ap.add_argument("--confirm_require_low", type=int, default=1)

    # Quality adaptation
    ap.add_argument("--quality_adapt", type=int, default=0)
    ap.add_argument("--quality_min", type=float, default=0.0)
    ap.add_argument("--quality_boost", type=float, default=0.15)
    ap.add_argument("--quality_boost_low", type=float, default=0.05)

    # Time mapping and overlap definition
    ap.add_argument("--time_mode", choices=["start", "center", "end"], default="center")
    ap.add_argument("--merge_gap_s", type=float, default=-1.0, help="GT merge gap seconds; <=0 => auto from data")
    ap.add_argument("--overlap_slack_s", type=float, default=0.0)

    # Deploy spec (stored in yaml for server/runtime)
    ap.add_argument("--deploy_fps", type=float, default=0.0)
    ap.add_argument("--deploy_w", type=int, default=0)
    ap.add_argument("--deploy_s", type=int, default=0)

    # OP selection policy
    ap.add_argument("--op1_recall", type=float, default=0.95)
    ap.add_argument("--op3_fa24h", type=float, default=1.0)

    # Legacy fallback
    ap.add_argument("--fps_default", type=float, default=30.0)

    args = ap.parse_args()

    # ------------------------------------------------------------
    # Load checkpoint bundle and rebuild model
    # ------------------------------------------------------------
    bundle = load_ckpt(args.ckpt, map_location="cpu")
    arch = str(args.arch).lower()

    model_cfg = get_cfg(bundle, "model_cfg", default={})
    raw_feat = get_cfg(bundle, "feat_cfg", default={})

    # Allow CLI to override feature flags if user explicitly provides them
    cli_feat: Dict[str, Any] = {}
    for k in ["center", "use_motion", "use_conf_channel", "motion_scale_by_fps", "conf_gate", "use_precomputed_mask"]:
        v = getattr(args, k, None)
        if v is not None:
            cli_feat[k] = v

    if hasattr(raw_feat, "to_dict"):
        try:
            raw_feat = raw_feat.to_dict()
        except Exception:
            pass
    if isinstance(raw_feat, dict):
        raw_feat = dict(raw_feat)
        raw_feat.update(cli_feat)
    elif cli_feat:
        raw_feat = cli_feat

    feat_cfg = FeatCfg.from_dict(raw_feat if isinstance(raw_feat, dict) else {})

    two_stream = bool(model_cfg.get("two_stream", False)) if arch == "gcn" else False

    # Keep fps_default consistent with training if stored
    fps_default = float(get_cfg(bundle, "data_cfg", default={}).get("fps_default", args.fps_default))

    device = pick_device()
    model = build_model(arch, model_cfg, feat_cfg, fps_default=fps_default).to(device)

    # EMA-safe loading:
    # 1) load base state_dict (includes BN buffers)
    # 2) if prefer_ema and ema_state_dict exists, overwrite param tensors
    sd = get_state_dict(bundle, prefer_ema=False)
    if bool(int(args.prefer_ema)):
        ema_sd = bundle.get("ema_state_dict", None) or bundle.get("ema", None)
        if isinstance(ema_sd, dict) and len(ema_sd) > 0:
            sd = dict(sd)
            sd.update(ema_sd)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # ------------------------------------------------------------
    # Temperature selection priority
    # ------------------------------------------------------------
    if float(args.temperature) > 0:
        T = float(args.temperature)
        cal_source = "cli"
    elif str(args.calibration_yaml).strip():
        T = load_temperature(str(args.calibration_yaml).strip(), default=1.0)
        cal_source = "yaml"
    else:
        T = 1.0
        cal_source = "none"

    # ------------------------------------------------------------
    # Build validation dataset and infer logits
    # ------------------------------------------------------------
    ds = ValWindows(args.val_dir, feat_cfg=feat_cfg, fps_default=fps_default, arch=arch, two_stream=two_stream)
    loader = DataLoader(ds, batch_size=int(args.batch), shuffle=False, num_workers=0, collate_fn=_collate)

    logits, y_true, vids, ws, we, fps_list, qs, ls, ms = infer_logits(model, loader, device, arch, two_stream)

    # Logits -> calibrated probabilities
    probs = _sigmoid_np(apply_temperature(logits, float(T)))

    # ------------------------------------------------------------
    # Optional FA-only stream for more realistic FA/24h
    # ------------------------------------------------------------
    fa_probs = fa_vids = fa_ws = fa_we = fa_fps_list = None
    fa_ls = fa_ms = fa_qs = None

    if str(args.fa_dir).strip():
        ds_fa = ValWindows(str(args.fa_dir).strip(), feat_cfg=feat_cfg, fps_default=fps_default, arch=arch, two_stream=two_stream)
        loader_fa = DataLoader(ds_fa, batch_size=int(args.batch), shuffle=False, num_workers=0, collate_fn=_collate)

        fa_logits, fa_y, fa_vids, fa_ws, fa_we, fa_fps_list, fa_qs, fa_ls, fa_ms = infer_logits(
            model, loader_fa, device, arch, two_stream
        )

        # Some FA dirs contain y=-1; for FA estimation we treat them as y=0 (no GT events).
        fa_probs = _sigmoid_np(apply_temperature(fa_logits, float(T)))

    # ------------------------------------------------------------
    # Alert base configuration (tau_high is swept later)
    # ------------------------------------------------------------
    alert_base = AlertCfg(
        ema_alpha=float(args.ema_alpha),
        k=int(args.k),
        n=int(args.n),
        tau_high=0.5,  # placeholder (swept)
        tau_low=0.4,   # placeholder (derived from tau_low_ratio per sweep point)
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

    merge_gap_s = None if float(args.merge_gap_s) <= 0 else float(args.merge_gap_s)

    # ------------------------------------------------------------
    # Sweep tau_high under the real policy
    # ------------------------------------------------------------
    sweep, meta = sweep_alert_policy_from_windows(
        probs=probs,
        y_true=y_true,
        video_ids=vids,
        w_start=ws,
        w_end=we,
        fps=fps_list,
        alert_base=alert_base,
        lying_score=ls,
        motion_score=ms,
        quality_score=qs,
        thr_min=0.05,
        thr_max=0.95,
        thr_step=0.01,
        tau_low_ratio=float(args.tau_low_ratio),
        merge_gap_s=merge_gap_s,
        overlap_slack_s=float(args.overlap_slack_s),
        time_mode=str(args.time_mode),
        fps_default=float(fps_default),

        # Optional FA-only stream
        fa_probs=fa_probs,
        fa_video_ids=fa_vids,
        fa_w_start=fa_ws,
        fa_w_end=fa_we,
        fa_fps=fa_fps_list,
        fa_lying_score=fa_ls,
        fa_motion_score=fa_ms,
        fa_quality_score=fa_qs,
    )

    # ------------------------------------------------------------
    # Pick OP indices
    # ------------------------------------------------------------
    i1 = _pick_op1(sweep, target_recall=float(args.op1_recall))
    i2 = _pick_op2(sweep)
    i3 = _pick_op3(sweep, target_fa24h=float(args.op3_fa24h))

    def op_row(i: int, code: str) -> Dict[str, Any]:
        """Extract one row from sweep arrays as a dict."""
        return {
            "code": code,
            "tau_high": float(sweep["thr"][i]),
            "tau_low": float(sweep["tau_low"][i]),
            "tau_low_ratio": float(args.tau_low_ratio),
            "recall": float(sweep["recall"][i]),
            "precision": float(sweep["precision"][i]),
            "f1": float(sweep["f1"][i]),
            "fa24h": float(sweep["fa24h"][i]),
            "fa_per_hour": float(sweep["fa_per_hour"][i]),
            "mean_delay_s": float(sweep["mean_delay_s"][i]),
            "median_delay_s": float(sweep["median_delay_s"][i]),
            "n_gt_events": int(sweep["n_gt_events"][i]),
            "n_alert_events": int(sweep["n_alert_events"][i]),
            "n_true_alerts": int(sweep["n_true_alerts"][i]),
            "n_false_alerts": int(sweep["n_false_alerts"][i]),
        }

    op1 = op_row(i1, "OP-1")
    op2 = op_row(i2, "OP-2")
    op3 = op_row(i3, "OP-3")

    # ------------------------------------------------------------
    # Deploy spec (used later by server/runtime)
    # ------------------------------------------------------------
    deploy = _infer_deploy_spec(args.val_dir, float(args.deploy_fps), int(args.deploy_w), int(args.deploy_s))

    # ------------------------------------------------------------
    # Build output YAML object
    # ------------------------------------------------------------
    sweep_rows = []
    for i in range(len(sweep["thr"])):
        sweep_rows.append(
            {
                "tau_high": float(sweep["thr"][i]),
                "tau_low": float(sweep["tau_low"][i]),
                "recall": float(sweep["recall"][i]),
                "precision": float(sweep["precision"][i]),
                "f1": float(sweep["f1"][i]),
                "fa24h": float(sweep["fa24h"][i]),
                "mean_delay_s": float(sweep["mean_delay_s"][i]),
            }
        )

    out = {
        "schema": "ops_yaml_v2",
        "arch": arch,
        "ckpt": str(args.ckpt),
        "deploy": deploy,
        "calibration": {"temperature": float(T), "source": str(cal_source)},
        "feat_cfg": feat_cfg.to_dict(),
        "alert_cfg": alert_base.to_dict(),
        "ops": {"op1": op1, "op2": op2, "op3": op3},
        "selection_policy": {"op1_recall": float(args.op1_recall), "op3_fa24h": float(args.op3_fa24h)},
        "meta": meta,
        "sweep": sweep_rows,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    yaml_dump_simple(out, args.out)

    print(f"[ok] wrote ops yaml: {args.out}")
    print(f"[op1] {op1}")
    print(f"[op2] {op2}")
    print(f"[op3] {op3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
