#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/calibrate_temperature.py

Fit temperature scaling for binary classification (fall vs non-fall).

What "temperature scaling" means
--------------------------------
Your model outputs logits z (unbounded real numbers).
We calibrate probabilities by dividing logits by a scalar T:

    z_cal = z / T
    p_cal = sigmoid(z_cal)

- If T > 1: probabilities become LESS confident (softer)
- If T < 1: probabilities become MORE confident (sharper)

Important:
- Temperature scaling does NOT change ranking much.
- It mostly improves probability calibration, which helps:
  - threshold selection
  - alert policy stability
  - combining models (hybrid) and uncertainty gating

What this script does
---------------------
1) Load checkpoint bundle and rebuild model consistently
2) Run inference on a labeled validation windows folder -> logits and labels
3) Fit T by minimizing NLL (negative log likelihood) on probabilities:
     NLL(T) = mean( BCEWithLogits(z / T, y) )
4) Save a YAML file:
     temperature: <bestT>
     calibration: { temperature: <bestT> , ... }
     meta: { ... }

We use a simple, robust grid + local refinement approach (no scipy dependency).

NOTE
----
- Only windows with y in {0,1} are used for calibration.
- If the validation set has only one class, we fall back to T=1.
"""

from __future__ import annotations

# ------------------------------------------------------------
# Path bootstrap
# ------------------------------------------------------------
import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

# ------------------------------------------------------------
# Stdlib
# ------------------------------------------------------------
import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------
# Third-party
# ------------------------------------------------------------
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------
# Project core
# ------------------------------------------------------------
from core.calibration import apply_temperature
from core.ckpt import get_cfg, get_state_dict, load_ckpt
from core.features import FeatCfg, build_gcn_input, build_tcn_input, read_window_npz
from core.models import build_model, logits_1d, pick_device
from core.yamlio import yaml_dump_simple


# ============================================================
# 1) Dataset for labeled windows
# ============================================================
@dataclass
class MetaRow:
    path: str
    y: int


class LabeledWindows(Dataset):
    """
    Loads windows and produces model input + label.

    We use the same feature pipeline as training (core/features.py),
    so logits are consistent.

    Returns:
      kind: "tcn" | "gcn" | "gcn2"
      X: model input
      y: int label
    """

    def __init__(self, windows_dir: str, feat_cfg: FeatCfg, fps_default: float, arch: str, two_stream: bool):
        self.files = sorted(glob.glob(os.path.join(windows_dir, "**", "*.npz"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No .npz under: {windows_dir}")

        self.feat_cfg = feat_cfg
        self.fps_default = float(fps_default)
        self.arch = str(arch).lower().strip()
        self.two_stream = bool(two_stream) and (self.arch == "gcn")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        joints, motion, conf, mask, fps, meta = read_window_npz(p, fps_default=self.fps_default)

        y = int(meta.y)

        # Build model input
        if self.arch == "tcn":
            X, _ = build_tcn_input(joints, motion, conf, mask, float(fps), self.feat_cfg)
            return "tcn", X.astype(np.float32), y

        X, _ = build_gcn_input(joints, motion, conf, mask, float(fps), self.feat_cfg)

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

            return "gcn2", (xj.astype(np.float32), xm.astype(np.float32)), y

        return "gcn", X.astype(np.float32), y


def collate(batch):
    """
    Custom collate because gcn2 returns tuples.

    Output:
      kind
      X_batch
      y_batch (np.int64)
    """
    kind = batch[0][0]

    if kind == "gcn2":
        xj_list, xm_list, ys = [], [], []
        for _k, (xj, xm), y in batch:
            xj_list.append(xj)
            xm_list.append(xm)
            ys.append(y)
        return kind, (np.stack(xj_list, axis=0), np.stack(xm_list, axis=0)), np.asarray(ys, dtype=np.int64)

    xs, ys = [], []
    for _k, x, y in batch:
        xs.append(x)
        ys.append(y)
    return kind, np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)


# ============================================================
# 2) Inference: get logits and labels
# ============================================================
@torch.no_grad()
def infer_logits(model, loader, device, arch: str, two_stream: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      logits: [N]
      y_true: [N] int labels
    """
    arch = str(arch).lower().strip()

    logits_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    for kind, X, y in loader:
        # IMPORTANT:
        # 'kind' may be 'gcn2' even though arch is 'gcn'.
        # That depends on model_cfg.two_stream.
        if kind == "tcn":
            xb = torch.from_numpy(X).to(device)
            logits = logits_1d(model(xb))
        elif kind == "gcn2":
            xj, xm = X
            xj = torch.from_numpy(xj).to(device)
            xm = torch.from_numpy(xm).to(device)
            logits = logits_1d(model(xj, xm))
        else:
            xb = torch.from_numpy(X).to(device)
            logits = logits_1d(model(xb))

        logits_all.append(logits.detach().cpu().numpy().reshape(-1))
        y_all.append(y.reshape(-1))

    return (
        np.concatenate(logits_all) if logits_all else np.array([], dtype=np.float32),
        np.concatenate(y_all) if y_all else np.array([], dtype=np.int64),
    )


# ============================================================
# 3) NLL objective for temperature scaling
# ============================================================
def bce_with_logits_np(logits: np.ndarray, y: np.ndarray) -> float:
    """
    Compute mean BCEWithLogits loss using numpy (stable).

    logits: [N]
    y:      [N] in {0,1}

    Formula (stable):
      loss = max(z,0) - z*y + log(1 + exp(-|z|))
    """
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    # Stable BCE with logits
    # ref: standard "softplus" formulation
    loss = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float(np.mean(loss))


def nll_for_T(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    """
    Evaluate negative log-likelihood at a given temperature T.

    We apply:
      logits_scaled = logits / T
      loss = BCEWithLogits(logits_scaled, y)
    """
    if not np.isfinite(T) or T <= 0:
        return float("inf")
    z = apply_temperature(logits, float(T))
    return bce_with_logits_np(z, y)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Fit temperature T by minimizing NLL on validation data.

    We avoid scipy dependency by doing:
    1) coarse grid search in log-space (T from 0.05 to 10)
    2) local refinement around the best point

    Returns:
      {"T": best_T, "nll": best_nll, "grid": [...]} for debug.
    """
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    # Only use y in {0,1}
    m = (y == 0) | (y == 1)
    logits = logits[m]
    y = y[m]

    # If only one class exists, calibration is not meaningful.
    if logits.size == 0 or np.unique(y).size < 2:
        return {"T": 1.0, "nll": float("nan"), "note": "insufficient class diversity"}

    # ---- 1) coarse grid in log space ----
    # logspace gives good coverage for T across multiple magnitudes.
    Ts = np.logspace(np.log10(0.05), np.log10(10.0), num=60)
    nlls = [nll_for_T(logits, y, float(T)) for T in Ts]

    i0 = int(np.argmin(nlls))
    best_T = float(Ts[i0])
    best_nll = float(nlls[i0])

    # ---- 2) local refinement around best ----
    # We refine by searching a narrower logspace around the best.
    lo = max(0.01, best_T / 2.0)
    hi = min(50.0, best_T * 2.0)

    Ts2 = np.logspace(np.log10(lo), np.log10(hi), num=60)
    nlls2 = [nll_for_T(logits, y, float(T)) for T in Ts2]

    i1 = int(np.argmin(nlls2))
    best_T2 = float(Ts2[i1])
    best_nll2 = float(nlls2[i1])

    if best_nll2 < best_nll:
        best_T, best_nll = best_T2, best_nll2

    return {
        "T": float(best_T),
        "nll": float(best_nll),
        "grid": [{"T": float(T), "nll": float(n)} for T, n in zip(Ts.tolist(), nlls)],
        "grid_refine": [{"T": float(T), "nll": float(n)} for T, n in zip(Ts2.tolist(), nlls2)],
    }


# ============================================================
# 4) Main CLI
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--arch", choices=["tcn", "gcn"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--out_yaml", required=True)

    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--prefer_ema", type=int, default=1)
    ap.add_argument("--fps_default", type=float, default=30.0)

    # Optional extra report for debugging
    ap.add_argument("--out_json", default="", help="If set, save debug details to JSON.")

    args = ap.parse_args()
    arch = str(args.arch).lower().strip()

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

    # two-stream only for GCN
    two_stream = bool(model_cfg.get("two_stream", False)) if arch == "gcn" else False

    fps_default = float(get_cfg(bundle, "data_cfg", default={}).get("fps_default", args.fps_default))

    device = pick_device()
    model = build_model(arch, model_cfg, feat_cfg, fps_default=fps_default).to(device)

    # EMA-safe loading
    sd = get_state_dict(bundle, prefer_ema=False)
    if bool(int(args.prefer_ema)):
        ema_sd = bundle.get("ema_state_dict", None) or bundle.get("ema", None)
        if isinstance(ema_sd, dict) and len(ema_sd) > 0:
            sd = dict(sd)
            sd.update(ema_sd)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # ------------------------------------------------------------
    # Dataset + inference
    # ------------------------------------------------------------
    ds = LabeledWindows(args.val_dir, feat_cfg, fps_default, arch, two_stream)
    dl = DataLoader(ds, batch_size=int(args.batch), shuffle=False, num_workers=0, collate_fn=collate)

    logits, y_true = infer_logits(model, dl, device, arch, two_stream)

    # Fit T
    fit = fit_temperature(logits, y_true)

    T = float(fit.get("T", 1.0))
    nll = float(fit.get("nll", float("nan")))

    # ------------------------------------------------------------
    # Write YAML output (simple schema used by other scripts)
    # ------------------------------------------------------------
    out = {
        "temperature": float(T),
        "calibration": {
            "temperature": float(T),
            "objective": "bce_with_logits_nll",
            "nll": float(nll),
            "prefer_ema": bool(int(args.prefer_ema)),
        },
        "meta": {
            "arch": arch,
            "ckpt": str(args.ckpt),
            "val_dir": str(args.val_dir),
            "fps_default": float(fps_default),
            "num_samples": int(np.sum(((y_true == 0) | (y_true == 1)))),
        },
    }

    os.makedirs(os.path.dirname(args.out_yaml) or ".", exist_ok=True)
    yaml_dump_simple(out, args.out_yaml)
    print(f"[ok] wrote calibration yaml: {args.out_yaml}")
    print(f"[summary] T={T:.4f} NLL={nll:.6f}")

    # Optional debug JSON
    if str(args.out_json).strip():
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(fit, f, indent=2)
        print(f"[ok] wrote debug json: {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
