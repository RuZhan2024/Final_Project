#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/mine_hard_negatives.py

Goal
----
Mine "hard negatives": windows that the model predicts as FALL with high probability
inside a directory that is supposed to contain NEGATIVE windows (ADL / non-fall clips)
or UNLABELED windows (treat as negative for mining purposes).

Why this is useful
------------------
Hard negatives are the mistakes the model is currently making, for example:
- sitting down quickly looks like a fall
- bending / picking up items
- camera jitter / pose failure causing false positives

If you add these windows back into training (as negatives), the model can learn
to stop triggering on them, reducing false alarms.

What this script does (step-by-step)
------------------------------------
1) Load checkpoint bundle (.pt) and rebuild model consistently
2) Load each window NPZ and convert to model input tensors (TCN or GCN)
3) Run inference -> logits -> (optional) temperature scaling -> probability
4) Filter to candidates (default: y==0, optionally include y==-1)
5) Sort by probability descending
6) Output top-K:
    - TXT  (paths or "prob\\tpath")
    - CSV  (prob + metadata)
    - JSONL (prob + metadata)

Important behavior choices
--------------------------
- We keep inference fast by batching windows (DataLoader).
- We keep model rebuilding correct by using cfg stored in the checkpoint.
- We keep EMA safe:
    load base state_dict first (has BatchNorm buffers), then overwrite with EMA params.
"""

from __future__ import annotations

# ------------------------------------------------------------
# Path bootstrap (so `python eval/mine_hard_negatives.py ...` works)
# ------------------------------------------------------------
import os as _os
import sys as _sys

_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

# ------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------
import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------
# Project imports (core is your “single source of truth”)
# ------------------------------------------------------------
from core.calibration import apply_temperature, load_temperature
from core.ckpt import get_cfg, get_state_dict, load_ckpt
from core.features import FeatCfg, build_gcn_input, build_tcn_input, read_window_npz
from core.models import build_model, logits_1d, pick_device


# ============================================================
# 1) Numerics: stable sigmoid
# ============================================================
def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    """
    Stable sigmoid for numpy.

    Why clip?
    - exp(1000) overflows.
    - logits from a model can occasionally be large.

    Returns:
      probs in [0, 1], float32
    """
    x = np.asarray(logits, dtype=np.float32)
    x = np.clip(x, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


# ============================================================
# 2) Metadata container for each window
# ============================================================
@dataclass
class MetaRow:
    """
    Metadata we need to write useful mining outputs.

    path:
      Full path to the window NPZ on disk.

    video_id:
      Which original clip this window belongs to (used for grouping / human review).

    w_start, w_end:
      Window's frame indices inside the original clip.
      (In your pipeline, w_end is inclusive.)

    fps:
      FPS for this window (after resampling to deploy fps).

    y:
      label:
        1 -> fall
        0 -> non-fall
       -1 -> unlabeled (depends on your dataset)
    """
    path: str
    video_id: str
    w_start: int
    w_end: int
    fps: float
    y: int


# ============================================================
# 3) Dataset: windows -> model input + metadata
# ============================================================
class WindowsForMining(Dataset):
    """
    Loads windows (*.npz) recursively from windows_dir and returns model inputs.

    Important:
    - We do NOT apply augmentation here. Mining should reflect raw model behavior.
    - We use core/features.py to build inputs so it matches training/eval.

    Returns a tuple:
      kind: "tcn" or "gcn" or "gcn2"
      X:
        - "tcn"  : np.ndarray [T, C]
        - "gcn"  : np.ndarray [T, V, F]
        - "gcn2" : tuple(xj, xm) where:
            xj: [T,V,Fj] (xy + optional conf)
            xm: [T,V,2]  (motion)
      meta: MetaRow
    """

    def __init__(self, windows_dir: str, feat_cfg: FeatCfg, fps_default: float, arch: str, two_stream: bool):
        # Collect files: supports nested folders and “sharded directories”
        self.files = sorted(glob.glob(os.path.join(windows_dir, "**", "*.npz"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No .npz found under: {windows_dir}")

        self.feat_cfg = feat_cfg
        self.fps_default = float(fps_default)

        # arch controls whether we produce TCN-style or GCN-style input
        self.arch = str(arch).lower().strip()

        # two_stream only makes sense when arch == "gcn"
        self.two_stream = bool(two_stream) and (self.arch == "gcn")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]

        # read_window_npz provides a consistent schema regardless of legacy keys
        joints, motion, conf, mask, fps, meta = read_window_npz(path, fps_default=self.fps_default)

        # Decide a stable video id for grouping / human review.
        # We try meta fields first; otherwise fallback to file stem.
        video_id = meta.video_id or meta.seq_id or os.path.splitext(os.path.basename(path))[0]

        m = MetaRow(
            path=str(path),
            video_id=str(video_id),
            w_start=int(meta.w_start),
            w_end=int(meta.w_end),
            fps=float(fps),
            y=int(meta.y),
        )

        # Build model input from raw arrays using the same feature rules as training.
        if self.arch == "tcn":
            # TCN input shape: [T, C]
            X, _ = build_tcn_input(joints, motion, conf, mask, float(fps), self.feat_cfg)
            return "tcn", X.astype(np.float32), m

        # Otherwise: GCN
        # Single-stream GCN input shape: [T, V, F]
        X, _ = build_gcn_input(joints, motion, conf, mask, float(fps), self.feat_cfg)

        if self.two_stream:
            # Two-stream split:
            # - joints stream uses xy (+ optional conf)
            # - motion stream uses dxdy
            xy = X[..., 0:2]  # [T,V,2]

            if self.feat_cfg.use_conf_channel:
                conf1 = X[..., -1:]  # [T,V,1]
                xj = np.concatenate([xy, conf1], axis=-1)  # [T,V,3]
            else:
                xj = xy  # [T,V,2]

            if self.feat_cfg.use_motion and X.shape[-1] >= 4:
                xm = X[..., 2:4]  # [T,V,2]
            else:
                # If motion is disabled, still provide a valid motion tensor of zeros.
                xm = np.zeros_like(xy, dtype=np.float32)

            return "gcn2", (xj.astype(np.float32), xm.astype(np.float32)), m

        return "gcn", X.astype(np.float32), m


def collate(batch):
    """
    Custom collate function for DataLoader.

    Why custom?
    - Some batches contain X as arrays, others contain X as tuples (xj, xm).
    - Default collate can't stack tuples cleanly for our use case.

    Output:
      kind: "tcn" | "gcn" | "gcn2"
      X_batch:
        - for tcn/gcn: np.ndarray [B, ...]
        - for gcn2: tuple(np.ndarray, np.ndarray) for xj and xm
      metas: list[MetaRow] length B
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
# 4) Main mining routine
# ============================================================
@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()

    # --- Model / data selection ---
    ap.add_argument("--arch", choices=["tcn", "gcn"], required=True, help="Which architecture to build.")
    ap.add_argument("--ckpt", required=True, help="Checkpoint bundle from training (best.pt).")
    ap.add_argument("--windows_dir", required=True, help="Directory containing window NPZ files to mine from.")

    # --- Output files (you can provide any subset) ---
    ap.add_argument("--out_txt", default="", help="TXT output (paths only or prob\\tpath).")
    ap.add_argument("--out_csv", default="", help="CSV output with structured fields.")
    ap.add_argument("--out_jsonl", default="", help="JSONL output with structured fields.")

    # Optional: write ALL scored windows (not only selected)
    ap.add_argument("--out_all_csv", default="", help="If set, write ALL scored windows to CSV.")
    ap.add_argument("--out_all_jsonl", default="", help="If set, write ALL scored windows to JSONL.")

    # --- Selection rules ---
    ap.add_argument("--top_k", type=int, default=500, help="How many windows to keep after sorting by prob.")
    ap.add_argument("--min_prob", type=float, default=0.50, help="Keep only windows with prob >= this value.")

    # Which labels are allowed to be mined?
    ap.add_argument(
        "--include_unlabeled",
        action="store_true",
        help="If set, treat y==-1 windows as negative candidates to mine.",
    )
    ap.add_argument(
        "--include_positives",
        action="store_true",
        help="If set, allow y==1 windows in mining candidates (usually you want False).",
    )

    # --- Inference control ---
    ap.add_argument("--batch", type=int, default=256, help="Batch size for inference.")
    ap.add_argument("--prefer_ema", type=int, default=1, help="1 => use EMA weights if present.")

    # --- Calibration / temperature scaling ---
    ap.add_argument("--temperature", type=float, default=0.0, help="If >0, override temperature scaling.")
    ap.add_argument("--calibration_yaml", default="", help="Optional YAML that contains: temperature: <T>")

    # Fallback FPS if NPZ metadata missing
    ap.add_argument("--fps_default", type=float, default=30.0)

    # TXT format control
    ap.add_argument(
        "--txt_with_prob",
        action="store_true",
        help="If set, write 'prob<TAB>path' lines. Otherwise write only 'path'.",
    )

    args = ap.parse_args()

    arch = str(args.arch).lower().strip()

    # ------------------------------------------------------------
    # Load checkpoint bundle and rebuild model consistently
    # ------------------------------------------------------------
    bundle = load_ckpt(args.ckpt, map_location="cpu")

    # model_cfg contains architecture hyperparameters saved during training
    model_cfg = get_cfg(bundle, "model_cfg", default={})

    # feat_cfg controls how we build features from window NPZ
    raw_feat = get_cfg(bundle, "feat_cfg", default={})
    if hasattr(raw_feat, "to_dict"):
        # if raw_feat is a dataclass-like object
        try:
            raw_feat = raw_feat.to_dict()
        except Exception:
            pass
    feat_cfg = FeatCfg.from_dict(raw_feat if isinstance(raw_feat, dict) else {})

    # two-stream is only relevant for GCN
    two_stream = bool(model_cfg.get("two_stream", False)) if arch == "gcn" else False

    # keep fps_default consistent with checkpoint if stored
    fps_default = float(get_cfg(bundle, "data_cfg", default={}).get("fps_default", args.fps_default))

    # Pick device (mps/cuda/cpu)
    device = pick_device()

    # Build model architecture
    model = build_model(arch, model_cfg, feat_cfg, fps_default=fps_default).to(device)

    # EMA-safe loading:
    # 1) load base state_dict (params + BN buffers)
    # 2) if prefer_ema and ema_state_dict exists, overwrite params with EMA params
    sd = get_state_dict(bundle, prefer_ema=False)
    if bool(int(args.prefer_ema)):
        ema_sd = bundle.get("ema_state_dict", None) or bundle.get("ema", None)
        if isinstance(ema_sd, dict) and len(ema_sd) > 0:
            sd = dict(sd)
            sd.update(ema_sd)

    model.load_state_dict(sd, strict=True)
    model.eval()

    # ------------------------------------------------------------
    # Temperature scaling (priority: CLI > yaml > none)
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
    # Dataset + loader
    # ------------------------------------------------------------
    ds = WindowsForMining(args.windows_dir, feat_cfg, fps_default, arch, two_stream)
    dl = DataLoader(
        ds,
        batch_size=int(args.batch),
        shuffle=False,     # do not shuffle; we want deterministic output ordering before sorting
        num_workers=0,     # keep simple; inference is often I/O bound
        pin_memory=False,
        collate_fn=collate,
    )

    # ------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------
    rows_all: List[Dict[str, Any]] = []   # stores ALL windows (if user wants out_all_*)
    rows_keep: List[Dict[str, Any]] = []  # stores only candidate negatives for selection

    for kind, X, metas in dl:
        # kind tells us how to feed the model (tcn vs gcn vs gcn2)
        if kind == "tcn":
            xb = torch.from_numpy(X).to(device)                 # [B,T,C]
            logits = logits_1d(model(xb)).detach().cpu().numpy()  # [B]
        elif kind == "gcn2":
            xj, xm = X
            xj = torch.from_numpy(xj).to(device)  # [B,T,V,Fj]
            xm = torch.from_numpy(xm).to(device)  # [B,T,V,2]
            logits = logits_1d(model(xj, xm)).detach().cpu().numpy()
        else:
            xb = torch.from_numpy(X).to(device)                 # [B,T,V,F]
            logits = logits_1d(model(xb)).detach().cpu().numpy()  # [B]

        logits = logits.reshape(-1)

        # Apply temperature scaling to logits: logits_scaled = logits / T
        logits_scaled = apply_temperature(logits, float(T))

        # Convert logits -> probability of fall
        probs = sigmoid_np(logits_scaled)

        # Build records per sample
        for pr, m in zip(probs.tolist(), metas):
            # clip_id is what humans want to read; video_id is the grouping id
            clip_id = str(m.video_id or os.path.splitext(os.path.basename(m.path))[0])

            rec = {
                "prob": float(pr),
                "clip_id": clip_id,
                "video_id": str(m.video_id),
                "w_start": int(m.w_start),
                "w_end": int(m.w_end),
                "fps": float(m.fps),
                "y": int(m.y),
                "path": str(m.path),
            }

            rows_all.append(rec)

            # Candidate filter logic:
            # - Most mining is intended for negatives only (y==0)
            # - Some datasets produce y==-1 for unlabeled; user may include those too.
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
    # Select hard negatives: filter by min_prob then take top_k
    # ------------------------------------------------------------
    min_prob = float(args.min_prob)
    cand = [r for r in rows_keep if float(r["prob"]) >= min_prob]

    # Sort descending by probability (largest mistakes first)
    cand.sort(key=lambda r: float(r["prob"]), reverse=True)

    # Top-K selection
    top_k = int(args.top_k) if int(args.top_k) > 0 else len(cand)
    selected = cand[:top_k]

    # ------------------------------------------------------------
    # Output writers
    # ------------------------------------------------------------
    # Make output dirs if needed
    for p in [args.out_txt, args.out_csv, args.out_jsonl, args.out_all_csv, args.out_all_jsonl]:
        if str(p).strip():
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

    # TXT: easiest for feeding back into training as "hard_neg_list"
    if str(args.out_txt).strip():
        with open(args.out_txt, "w", encoding="utf-8") as f:
            for r in selected:
                if args.txt_with_prob:
                    f.write(f"{float(r['prob']):.6f}\t{r['path']}\n")
                else:
                    f.write(f"{r['path']}\n")
        print(f"[ok] wrote txt: {args.out_txt}")

    # CSV: structured output (easy to open in Excel)
    if str(args.out_csv).strip():
        fieldnames = ["prob", "clip_id", "video_id", "w_start", "w_end", "fps", "y", "path"]
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in selected:
                w.writerow({k: r.get(k) for k in fieldnames})
        print(f"[ok] wrote csv: {args.out_csv}")

    # JSONL: structured, line-delimited JSON (great for later tooling)
    if str(args.out_jsonl).strip():
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for r in selected:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[ok] wrote jsonl: {args.out_jsonl}")

    # Optional: write all-scored windows (useful for debugging)
    if str(args.out_all_csv).strip():
        fieldnames = ["prob", "clip_id", "video_id", "w_start", "w_end", "fps", "y", "path"]
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

    # Console summary so you can sanity check quickly
    print(
        f"[summary] device={device.type} arch={arch} two_stream={two_stream} "
        f"T={float(T):.4g} (source={cal_source})\n"
        f"[summary] scanned={len(rows_all)} candidates={len(rows_keep)} "
        f"min_prob={min_prob} kept={len(cand)} selected(top_k)={len(selected)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
