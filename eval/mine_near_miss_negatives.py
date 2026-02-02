#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/mine_near_miss_negatives.py

Goal
----
Mine "near-miss negatives": windows that are NEGATIVE (or unlabeled) but whose
predicted probability is close to the *decision boundary*.

Why near-miss negatives matter
------------------------------
Hard negatives are the big mistakes (p very high).
Near-miss negatives are the borderline cases:
- model is uncertain
- model frequently goes into "suspect" band but doesn't fully alert
- these samples teach the model to push probability DOWN on borderline ADLs

Near-miss mining is especially useful when:
- you already reduced major false alarms with hard negatives
- you want to reduce alert fatigue further (lower FA/day) without losing recall

Two mining modes (supported here)
---------------------------------
Mode A: Probability band
  - pick windows with prob in [low, high], e.g. [0.40, 0.70]
  - typical use: mine near your tau_low/tau_high bands

Mode B: Distance-to-threshold
  - given a target threshold tau (e.g., tau_high of OP-2)
  - mine windows with smallest |prob - tau| (closest to the boundary)

This script supports BOTH.

Outputs
-------
- TXT: paths (or prob<TAB>path) so you can feed it into training as hard_neg_list
- CSV: structured output with clip_id, indices, prob, distance, etc.
- JSONL: structured output per line

Notes about labels
------------------
By default we mine:
- y == 0 (negative)
Optionally include:
- y == -1 (unlabeled) treated as negative
We exclude y == 1 unless you explicitly ask to include positives (rare).

Compatibility
-------------
- Works for TCN and GCN (including GCN two-stream)
- Supports EMA-safe checkpoint loading
- Supports temperature scaling
- Can read tau_high from ops_yaml (OP-1/2/3) so mining aligns with your deployment OP
"""

from __future__ import annotations

# ------------------------------------------------------------
# Path bootstrap (so direct runs work)
# ------------------------------------------------------------
import os as _os
import sys as _sys

_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

# ------------------------------------------------------------
# Standard library
# ------------------------------------------------------------
import argparse
import csv
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
# Project (core)
# ------------------------------------------------------------
from core.calibration import apply_temperature, load_temperature
from core.ckpt import get_cfg, get_state_dict, load_ckpt
from core.features import FeatCfg, build_gcn_input, build_tcn_input, read_window_npz
from core.models import build_model, logits_1d, pick_device
from core.yamlio import yaml_load_simple


# ============================================================
# 1) Stable sigmoid
# ============================================================
def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    """
    Convert logits -> probability in [0,1] safely.

    sigmoid(x) = 1 / (1 + exp(-x))

    Clipping prevents overflow/underflow in exp.
    """
    x = np.asarray(logits, dtype=np.float32)
    x = np.clip(x, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


# ============================================================
# 2) Metadata record for one window
# ============================================================
@dataclass
class MetaRow:
    path: str
    video_id: str
    w_start: int
    w_end: int
    fps: float
    y: int


# ============================================================
# 3) Dataset: load windows -> model input + meta
# ============================================================
class WindowsForNearMiss(Dataset):
    """
    Returns:
      kind: "tcn" | "gcn" | "gcn2"
      X: model input
      meta: MetaRow
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
        path = self.files[idx]

        joints, motion, conf, mask, fps, meta = read_window_npz(path, fps_default=self.fps_default)
        y = int(meta.y)

        # Determine a stable per-video id for grouping/human review
        video_id = meta.video_id or meta.seq_id or os.path.splitext(os.path.basename(path))[0]

        m = MetaRow(
            path=str(path),
            video_id=str(video_id),
            w_start=int(meta.w_start),
            w_end=int(meta.w_end),
            fps=float(fps),
            y=int(y),
        )

        if self.arch == "tcn":
            X, _ = build_tcn_input(joints, motion, conf, mask, float(fps), self.feat_cfg)
            return "tcn", X.astype(np.float32), m

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

            return "gcn2", (xj.astype(np.float32), xm.astype(np.float32)), m

        return "gcn", X.astype(np.float32), m


def collate(batch):
    """
    Custom batch collate because two-stream uses tuples.

    Output:
      kind
      X_batch:
        - for tcn/gcn: np.ndarray [B, ...]
        - for gcn2: (xj_batch, xm_batch)
      metas: list[MetaRow]
    """
    kind = batch[0][0]

    if kind == "gcn2":
        xj_list, xm_list, metas = [], [], []
        for _k, (xj, xm), m in batch:
            xj_list.append(xj)
            xm_list.append(xm)
            metas.append(m)
        return kind, (np.stack(xj_list, axis=0), np.stack(xm_list, axis=0)), metas

    xs, metas = [], []
    for _k, x, m in batch:
        xs.append(x)
        metas.append(m)
    return kind, np.stack(xs, axis=0), metas


# ============================================================
# 4) Main
# ============================================================
@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()

    # Model inputs
    ap.add_argument("--arch", choices=["tcn", "gcn"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows_dir", required=True)

    # Outputs
    ap.add_argument("--out_txt", default="")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--out_jsonl", default="")

    # Optional: save all scored windows
    ap.add_argument("--out_all_csv", default="")
    ap.add_argument("--out_all_jsonl", default="")

    # Inference
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--prefer_ema", type=int, default=1)

    # Calibration
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--calibration_yaml", default="")

    # Operating point yaml (optional): to align mining with deployment thresholds
    ap.add_argument("--ops_yaml", default="")
    ap.add_argument("--op", choices=["op1", "op2", "op3"], default="op2")
    ap.add_argument("--tau", type=float, default=0.0, help="If >0, override target threshold tau for distance mining.")

    # Mining mode selection
    ap.add_argument(
        "--mode",
        choices=["band", "distance"],
        default="band",
        help="band: keep prob in [low, high]. distance: keep smallest |prob - tau|.",
    )

    # Band mode parameters
    ap.add_argument("--low", type=float, default=0.30, help="Lower bound for prob band.")
    ap.add_argument("--high", type=float, default=0.60, help="Upper bound for prob band.")

    # Distance mode parameters
    ap.add_argument("--max_dist", type=float, default=0.10, help="Keep only if |prob-tau| <= max_dist (distance mode).")

    # Label inclusion policy
    ap.add_argument("--include_unlabeled", action="store_true")
    ap.add_argument("--include_positives", action="store_true")

    # Selection caps
    ap.add_argument("--top_k", type=int, default=1000)
    ap.add_argument("--txt_with_prob", action="store_true")

    ap.add_argument("--fps_default", type=float, default=30.0)

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
    # Temperature scaling selection
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
    # Choose target threshold tau (for distance mode)
    # Priority:
    #   1) --tau cli
    #   2) ops_yaml selected op's tau_high
    #   3) fallback tau = 0.5
    # ------------------------------------------------------------
    tau = 0.0
    tau_source = "none"

    if float(args.tau) > 0:
        tau = float(args.tau)
        tau_source = "cli"
    elif str(args.ops_yaml).strip():
        ops = yaml_load_simple(str(args.ops_yaml).strip())
        if isinstance(ops, dict) and isinstance(ops.get("ops"), dict) and isinstance(ops["ops"].get(args.op), dict):
            tau = float(ops["ops"][args.op].get("tau_high", 0.0))
            if tau > 0:
                tau_source = f"ops_yaml:{args.op}"

    if tau <= 0:
        tau = 0.5
        tau_source = "default"

    # ------------------------------------------------------------
    # Build dataset + dataloader
    # ------------------------------------------------------------
    ds = WindowsForNearMiss(args.windows_dir, feat_cfg, fps_default, arch, two_stream)
    dl = DataLoader(ds, batch_size=int(args.batch), shuffle=False, num_workers=0, collate_fn=collate)

    # We will store:
    # - rows_all: every window with its probability (for debugging)
    # - rows_keep: only negative/unlabeled candidates to select from
    rows_all: List[Dict[str, Any]] = []
    rows_keep: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------
    for kind, X, metas in dl:
        # Convert batch data into torch tensors on the chosen device.
        if kind == "tcn":
            xb = torch.from_numpy(X).to(device)  # [B,T,C]
            logits = logits_1d(model(xb)).detach().cpu().numpy()
        elif kind == "gcn2":
            xj, xm = X
            xj = torch.from_numpy(xj).to(device)
            xm = torch.from_numpy(xm).to(device)
            logits = logits_1d(model(xj, xm)).detach().cpu().numpy()
        else:
            xb = torch.from_numpy(X).to(device)  # [B,T,V,F]
            logits = logits_1d(model(xb)).detach().cpu().numpy()

        logits = logits.reshape(-1)

        # Apply temperature scaling to logits then sigmoid
        logits_scaled = apply_temperature(logits, float(T))
        probs = sigmoid_np(logits_scaled)

        # Build row per window
        for p, m in zip(probs.tolist(), metas):
            prob = float(p)
            dist = float(abs(prob - float(tau)))

            rec = {
                "prob": prob,
                "dist_to_tau": dist,
                "tau": float(tau),
                "tau_source": str(tau_source),
                "clip_id": str(m.video_id),
                "video_id": str(m.video_id),
                "w_start": int(m.w_start),
                "w_end": int(m.w_end),
                "fps": float(m.fps),
                "y": int(m.y),
                "path": str(m.path),
            }
            rows_all.append(rec)

            # Candidate policy:
            is_pos = (int(m.y) == 1)
            is_neg = (int(m.y) == 0)
            is_unl = (int(m.y) < 0)

            if (not args.include_positives) and is_pos:
                continue
            if is_neg:
                rows_keep.append(rec)
                continue
            if args.include_unlabeled and is_unl:
                rows_keep.append(rec)
                continue

    # ------------------------------------------------------------
    # Selection logic
    # ------------------------------------------------------------
    mode = str(args.mode).lower().strip()

    if mode == "band":
        low = float(args.low)
        high = float(args.high)

        # Ensure low <= high even if user passes weird values
        if low > high:
            low, high = high, low

        selected = [r for r in rows_keep if (float(r["prob"]) >= low and float(r["prob"]) <= high)]

        # In band mode, we usually want the "most suspicious" within band first
        selected.sort(key=lambda r: float(r["prob"]), reverse=True)

        selection_info = {"mode": "band", "low": low, "high": high}

    else:
        # distance mode: keep smallest distance to tau (closest to boundary)
        max_dist = float(args.max_dist)
        selected = [r for r in rows_keep if float(r["dist_to_tau"]) <= max_dist]

        # Sort ascending by distance (closest first),
        # tie-break by probability descending (more dangerous first)
        selected.sort(key=lambda r: (float(r["dist_to_tau"]), -float(r["prob"])))

        selection_info = {"mode": "distance", "tau": float(tau), "max_dist": max_dist, "tau_source": str(tau_source)}

    # Top-K cap
    top_k = int(args.top_k) if int(args.top_k) > 0 else len(selected)
    selected = selected[:top_k]

    # ------------------------------------------------------------
    # Output writers
    # ------------------------------------------------------------
    for p in [args.out_txt, args.out_csv, args.out_jsonl, args.out_all_csv, args.out_all_jsonl]:
        if str(p).strip():
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

    # TXT: easiest for feeding back into training
    if str(args.out_txt).strip():
        with open(args.out_txt, "w", encoding="utf-8") as f:
            for r in selected:
                if args.txt_with_prob:
                    f.write(f"{float(r['prob']):.6f}\t{r['path']}\n")
                else:
                    f.write(f"{r['path']}\n")
        print(f"[ok] wrote txt: {args.out_txt}")

    # CSV: structured output
    if str(args.out_csv).strip():
        fieldnames = ["prob", "dist_to_tau", "tau", "tau_source", "clip_id", "video_id", "w_start", "w_end", "fps", "y", "path"]
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in selected:
                w.writerow({k: r.get(k) for k in fieldnames})
        print(f"[ok] wrote csv: {args.out_csv}")

    # JSONL: structured output
    if str(args.out_jsonl).strip():
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for r in selected:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[ok] wrote jsonl: {args.out_jsonl}")

    # Optional: all-scored outputs for debugging
    if str(args.out_all_csv).strip():
        fieldnames = ["prob", "dist_to_tau", "tau", "tau_source", "clip_id", "video_id", "w_start", "w_end", "fps", "y", "path"]
        with open(args.out_all_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows_all:
                w.writerow({k: r.get(k) for k in fieldnames})
        print(f"[ok] wrote all csv: {args.out_all_csv}")

    if str(args.out_all_jsonl).strip():
        with open(args.out_all_jsonl, "w", encoding="utf-8") as f:
            for r in rows_all:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[ok] wrote all jsonl: {args.out_all_jsonl}")

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------
    print(
        f"[summary] device={pick_device().type} arch={arch} two_stream={two_stream} "
        f"temperature={float(T):.4g} (source={cal_source})\n"
        f"[summary] scanned={len(rows_all)} candidates={len(rows_keep)} selected={len(selected)} "
        f"selection={selection_info}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
