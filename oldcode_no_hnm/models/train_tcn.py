#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/train_tcn.py  (redesigned + rewritten)

This version is aligned with the *new window schema* produced by windows/make_windows.py:

Window NPZ keys (preferred)
- joints : [T,33,2]   (float32)
- motion : [T,33,2]   (float32)  (optional; if missing we compute from joints)
- conf   : [T,33]     (float32)  (optional)
- mask   : [T,33]     (uint8/bool) (optional)
- fps    : scalar     (optional)
- y      : 0/1        (int/float) (required for labeled windows; -1 for unlabeled)

What’s new vs your old trainer
- Uses joints + motion (+ optional conf) from new windows
- Pelvis-relative option (center per-frame) + motion recompute in that space (recommended)
- DropGraph-style augmentation during training:
    * random joint masking (mask_joint_p)
    * random frame masking (mask_frame_p)
- Better imbalance handling:
    * --pos_weight auto|none|float
    * --balanced_sampler
- Clean reporting:
    * saves best.pt by best val F1 (threshold sweep)
    * writes train_report.json + per-epoch history.jsonl

Usage (typical)
  python models/train_tcn.py \
    --train_dir data/processed/le2i/windows_W32_S8/train \
    --val_dir   data/processed/le2i/windows_W32_S8/val \
    --epochs 50 --batch 128 --lr 1e-3 --seed 33724876 \
    --save_dir outputs/le2i_tcn_W32S8 \
    --pos_weight auto --balanced_sampler \
    --mask_joint_p 0.15 --mask_frame_p 0.10

"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)
from tqdm import tqdm


# -------------------------
# Repro / device
# -------------------------
def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def pick_device() -> torch.device:
    # macOS MPS first if available
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------
# Label parsing (robust)
# -------------------------
_POS_STR = {"1", "true", "fall", "pos", "positive", "yes", "y", "t"}
_NEG_STR = {"0", "false", "adl", "neg", "negative", "no", "n", "f", "nonfall", "nofall", "normal"}


def _label_from_npz(d) -> Optional[float]:
    if "y" in d.files:
        y = d["y"]
        if isinstance(y, np.ndarray):
            y = y.item() if y.shape == () else y.ravel()[0]
        try:
            y = float(y)
        except Exception:
            return None
        if y < 0:
            return None
        return 1.0 if y >= 0.5 else 0.0

    for k in ("label", "y_label", "target"):
        if k not in d.files:
            continue
        lab = d[k]
        if isinstance(lab, bytes):
            lab = lab.decode()
        elif isinstance(lab, np.ndarray):
            try:
                lab = lab.item()
            except Exception:
                lab = str(lab)

        # numeric?
        try:
            v = float(lab)
            if v < 0:
                return None
            return 1.0 if v >= 0.5 else 0.0
        except Exception:
            pass

        s = str(lab).strip().lower()
        if s in _POS_STR:
            return 1.0
        if s in _NEG_STR:
            return 0.0
        return None

    return None


# -------------------------
# Feature building
# -------------------------
def _safe_fps(d, fps_default: float) -> float:
    if "fps" in d.files:
        try:
            return float(np.array(d["fps"]).reshape(-1)[0])
        except Exception:
            return float(fps_default)
    return float(fps_default)


def _pelvis_center(joints: np.ndarray) -> np.ndarray:
    # MediaPipe hips: 23 (L), 24 (R)
    if joints.shape[1] > 24:
        return 0.5 * (joints[:, 23:24, :] + joints[:, 24:25, :])
    return joints[:, 23:24, :]


def _derive_mask(joints: np.ndarray, conf: Optional[np.ndarray], conf_gate: float) -> np.ndarray:
    finite = np.isfinite(joints[..., 0]) & np.isfinite(joints[..., 1])
    if conf is None or conf_gate <= 0:
        return finite
    return finite & (conf >= conf_gate)


def _apply_mask(joints: np.ndarray, motion: np.ndarray, conf: Optional[np.ndarray], mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    m = mask.astype(np.float32)[..., None]  # [T,V,1]
    joints = joints * m
    motion = motion * m
    if conf is not None:
        conf = conf * mask.astype(np.float32)
    return joints, motion, conf


def _compute_motion(joints: np.ndarray, fps: float, scale_by_fps: bool) -> np.ndarray:
    motion = np.zeros_like(joints, dtype=np.float32)
    motion[1:] = joints[1:] - joints[:-1]
    if scale_by_fps:
        motion = motion * float(fps)
    motion[0] = 0.0
    return motion


def build_features_from_window(
    joints: np.ndarray,           # [T,V,2]
    motion: Optional[np.ndarray], # [T,V,2] or None
    conf: Optional[np.ndarray],   # [T,V]   or None
    mask: np.ndarray,             # [T,V] bool/0-1
    fps: float,
    center: str,
    use_motion: bool,
    use_conf_channel: bool,
    motion_scale_by_fps: bool,
) -> np.ndarray:
    joints = np.nan_to_num(joints, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if conf is not None:
        conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    if center == "pelvis":
        pel = _pelvis_center(joints)                 # [T,1,2]
        joints = joints - pel                        # pelvis-relative

    if use_motion:
        if motion is None:
            motion = _compute_motion(joints, fps, motion_scale_by_fps)
        else:
            motion = np.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            # If we centered joints, better to recompute motion in that same space
            if center == "pelvis":
                motion = _compute_motion(joints, fps, motion_scale_by_fps)
            elif motion_scale_by_fps:
                motion = motion * float(fps)
    else:
        motion = np.zeros_like(joints, dtype=np.float32)

    joints, motion, conf = _apply_mask(joints, motion, conf, mask)

    parts = [joints]  # [T,V,2]
    if use_motion:
        parts.append(motion)      # [T,V,2]
    if use_conf_channel and conf is not None:
        parts.append(conf[..., None].astype(np.float32))  # [T,V,1]

    feats = np.concatenate(parts, axis=-1).astype(np.float32)  # [T,V,F]
    return feats


# -------------------------
# DropGraph augmentation (train only)
# -------------------------
def augment_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
    mask_joint_p: float,
    mask_frame_p: float,
    ensure_nonempty: bool = True,
) -> np.ndarray:
    """
    mask: [T,V] bool/0-1
    Applies:
      - joint dropout: drop some joints across all frames
      - frame dropout: drop some frames across all joints
    """
    m = mask.copy().astype(bool)

    T, V = m.shape
    if mask_joint_p > 0:
        drop_j = rng.random(V) < float(mask_joint_p)
        if drop_j.any():
            m[:, drop_j] = False
    if mask_frame_p > 0:
        drop_t = rng.random(T) < float(mask_frame_p)
        if drop_t.any():
            m[drop_t, :] = False

    if ensure_nonempty and not m.any():
        # keep one random joint on one random frame
        t = int(rng.integers(0, T))
        v = int(rng.integers(0, V))
        m[t, v] = True

    return m


# -------------------------
# Dataset
# -------------------------
class WindowNPZTCN(Dataset):
    """
    Returns:
      x: [T, C] float32
      y: [1] float32
    """
    def __init__(
        self,
        root: str,
        *,
        split: str,
        skip_unlabeled: bool = True,
        fps_default: float = 30.0,
        conf_gate: float = 0.20,
        use_precomputed_mask: bool = True,
        center: str = "pelvis",
        use_motion: bool = True,
        use_conf_channel: bool = True,
        motion_scale_by_fps: bool = True,
        # augmentation
        mask_joint_p: float = 0.0,
        mask_frame_p: float = 0.0,
        seed: int = 33724876,
    ):
        self.root = str(root)
        self.split = str(split)
        self.files = sorted(glob.glob(os.path.join(self.root, "**", "*.npz"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No .npz windows found under: {root}")

        kept: List[str] = []
        ys: List[int] = []
        skipped = 0
        for p in self.files:
            try:
                with np.load(p, allow_pickle=False) as d:
                    y = _label_from_npz(d)
                if skip_unlabeled and y is None:
                    skipped += 1
                    continue
                kept.append(p)
                ys.append(int(y if y is not None else 0))
            except Exception:
                skipped += 1

        if not kept:
            raise FileNotFoundError(f"All windows under {root} were unlabeled or unreadable.")
        if skipped:
            print(f"[WindowNPZTCN] skipped {skipped} unlabeled/unreadable windows under {root}")

        self.files = kept
        self.labels = np.asarray(ys, dtype=np.int64)

        self.fps_default = float(fps_default)
        self.conf_gate = float(conf_gate)
        self.use_precomputed_mask = bool(use_precomputed_mask)

        self.center = str(center)
        self.use_motion = bool(use_motion)
        self.use_conf_channel = bool(use_conf_channel)
        self.motion_scale_by_fps = bool(motion_scale_by_fps)

        # aug only for train
        self.mask_joint_p = float(mask_joint_p) if self.split == "train" else 0.0
        self.mask_frame_p = float(mask_frame_p) if self.split == "train" else 0.0

        self._rng = np.random.default_rng(int(seed) + (0 if self.split == "train" else 991))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        with np.load(p, allow_pickle=False) as d:
            y = _label_from_npz(d)
            if y is None:
                y = 0.0

            # Prefer new schema
            if "joints" in d.files:
                joints = d["joints"].astype(np.float32, copy=False)
            else:
                joints = d["xy"].astype(np.float32, copy=False)

            motion = d["motion"].astype(np.float32, copy=False) if ("motion" in d.files) else None
            conf = d["conf"].astype(np.float32, copy=False) if ("conf" in d.files) else None
            fps = _safe_fps(d, self.fps_default)

            if self.use_precomputed_mask and ("mask" in d.files):
                mask = np.array(d["mask"]).astype(bool, copy=False)
            else:
                mask = _derive_mask(joints, conf, self.conf_gate)

        # Augment mask (train only)
        if self.mask_joint_p > 0 or self.mask_frame_p > 0:
            mask = augment_mask(mask, self._rng, self.mask_joint_p, self.mask_frame_p, ensure_nonempty=True)

        feats = build_features_from_window(
            joints=joints,
            motion=motion,
            conf=conf,
            mask=mask,
            fps=fps,
            center=self.center,
            use_motion=self.use_motion,
            use_conf_channel=self.use_conf_channel,
            motion_scale_by_fps=self.motion_scale_by_fps,
        )  # [T,V,F]

        x = feats.reshape(feats.shape[0], -1)  # [T, V*F]
        return torch.from_numpy(x).float(), torch.tensor([float(y)], dtype=torch.float32)


# -------------------------
# Model: Residual TCN
# -------------------------
class ResTCNBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3, dilation: int = 1, p: float = 0.30):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.relu(self.bn(self.conv(x))))
        return x + out


class TCN(nn.Module):
    def __init__(self, in_ch: int, hid: int = 128, p: float = 0.30):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_ch, hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(hid),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResTCNBlock(hid, dilation=1, p=p),
            ResTCNBlock(hid, dilation=2, p=p),
            ResTCNBlock(hid, dilation=4, p=p),
            ResTCNBlock(hid, dilation=8, p=p),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C] -> [B,C,T]
        x = x.permute(0, 2, 1)
        h = self.blocks(self.conv_in(x))
        h = self.pool(h).squeeze(-1)
        return self.head(h)  # [B,1]


# -------------------------
# Metrics / threshold sweep
# -------------------------
def _fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


@torch.no_grad()
def collect_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).squeeze(-1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y = yb.detach().cpu().numpy().reshape(-1).astype(int)
        probs_list.append(probs)
        y_list.append(y)
    probs = np.concatenate(probs_list, axis=0) if probs_list else np.array([])
    y = np.concatenate(y_list, axis=0) if y_list else np.array([])
    return probs, y


def best_f1_threshold(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    best = {"thr": 0.5, "P": 0.0, "R": 0.0, "F1": 0.0, "FPR": 0.0}
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (probs >= thr).astype(int)
        P, R, F1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        if F1 > best["F1"]:
            best.update({"thr": float(thr), "P": float(P), "R": float(R), "F1": float(F1), "FPR": float(_fpr(y_true, pred))})
    return best


def extra_scores(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    out = {"AP": 0.0, "AUC": 0.0}
    if y_true.size and (y_true.min() != y_true.max()):
        out["AP"] = float(average_precision_score(y_true, probs))
        try:
            out["AUC"] = float(roc_auc_score(y_true, probs))
        except Exception:
            out["AUC"] = 0.0
    return out


# -------------------------
# Imbalance helpers
# -------------------------
def compute_pos_weight_from_labels(y: np.ndarray) -> Optional[torch.Tensor]:
    pos = int(y.sum())
    total = int(y.size)
    neg = total - pos
    if total == 0 or pos == 0 or neg <= 0:
        return None
    w = neg / pos
    print(f"[info] class balance: total={total}, pos={pos}, neg={neg}, pos_weight={w:.2f}")
    return torch.tensor([w], dtype=torch.float32)


def make_balanced_sampler(y: np.ndarray) -> WeightedRandomSampler:
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    w_pos = 1.0 / max(pos, 1)
    w_neg = 1.0 / max(neg, 1)
    weights = np.where(y == 1, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=len(weights), replacement=True)


# -------------------------
# Train
# -------------------------
@dataclass
class TrainConfig:
    train_dir: str
    val_dir: str
    test_dir: Optional[str]
    epochs: int
    batch: int
    lr: float
    seed: int
    save_dir: str
    grad_clip: float
    patience: int
    fps_default: float

    hid: int
    dropout: float

    # features
    center: str
    use_motion: bool
    use_conf_channel: bool
    motion_scale_by_fps: bool
    conf_gate: float
    use_precomputed_mask: bool

    # aug
    mask_joint_p: float
    mask_frame_p: float

    # imbalance
    pos_weight: str
    balanced_sampler: bool

    # reporting
    report_json: Optional[str]
    report_dataset_name: Optional[str]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a redesigned TCN on skeleton windows (new schema).")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--test_dir", default=None)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=33724876)

    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--fps_default", type=float, default=30.0)

    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.30)

    # features
    ap.add_argument("--center", choices=["pelvis", "none"], default="pelvis")
    ap.add_argument("--use_motion", type=int, default=1)  # keep Makefile-friendly
    ap.add_argument("--use_conf_channel", type=int, default=1)
    ap.add_argument("--motion_scale_by_fps", type=int, default=1)
    ap.add_argument("--conf_gate", type=float, default=0.20)
    ap.add_argument("--use_precomputed_mask", type=int, default=1)

    # augmentation
    ap.add_argument("--mask_joint_p", type=float, default=0.15)
    ap.add_argument("--mask_frame_p", type=float, default=0.10)

    # imbalance
    ap.add_argument("--pos_weight", default="auto", help="auto | none | <float>")
    ap.add_argument("--balanced_sampler", action="store_true")

    # optional report output (used by Makefile in some versions)
    ap.add_argument("--report_json", default=None)
    ap.add_argument("--report_dataset_name", default=None)

    args = ap.parse_args()

    cfg = TrainConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir if args.test_dir else None,
        epochs=int(args.epochs),
        batch=int(args.batch),
        lr=float(args.lr),
        seed=int(args.seed),
        save_dir=str(args.save_dir),
        grad_clip=float(args.grad_clip),
        patience=int(args.patience),
        fps_default=float(args.fps_default),
        hid=int(args.hid),
        dropout=float(args.dropout),
        center=str(args.center),
        use_motion=bool(int(args.use_motion)),
        use_conf_channel=bool(int(args.use_conf_channel)),
        motion_scale_by_fps=bool(int(args.motion_scale_by_fps)),
        conf_gate=float(args.conf_gate),
        use_precomputed_mask=bool(int(args.use_precomputed_mask)),
        mask_joint_p=float(args.mask_joint_p),
        mask_frame_p=float(args.mask_frame_p),
        pos_weight=str(args.pos_weight),
        balanced_sampler=bool(args.balanced_sampler),
        report_json=str(args.report_json) if args.report_json else None,
        report_dataset_name=str(args.report_dataset_name) if args.report_dataset_name else None,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Save config
    with open(os.path.join(cfg.save_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = pick_device()
    print(f"[info] device: {device.type}")

    train_ds = WindowNPZTCN(
        cfg.train_dir,
        split="train",
        skip_unlabeled=True,
        fps_default=cfg.fps_default,
        conf_gate=cfg.conf_gate,
        use_precomputed_mask=cfg.use_precomputed_mask,
        center=cfg.center,
        use_motion=cfg.use_motion,
        use_conf_channel=cfg.use_conf_channel,
        motion_scale_by_fps=cfg.motion_scale_by_fps,
        mask_joint_p=cfg.mask_joint_p,
        mask_frame_p=cfg.mask_frame_p,
        seed=cfg.seed,
    )
    val_ds = WindowNPZTCN(
        cfg.val_dir,
        split="val",
        skip_unlabeled=True,
        fps_default=cfg.fps_default,
        conf_gate=cfg.conf_gate,
        use_precomputed_mask=cfg.use_precomputed_mask,
        center=cfg.center,
        use_motion=cfg.use_motion,
        use_conf_channel=cfg.use_conf_channel,
        motion_scale_by_fps=cfg.motion_scale_by_fps,
        mask_joint_p=0.0,
        mask_frame_p=0.0,
        seed=cfg.seed,
    )

    x0, y0 = train_ds[0]
    T, C = x0.shape
    print(f"[info] window shape: T={T}, C={C}; first y={float(y0.item())}")

    model = TCN(in_ch=C, hid=cfg.hid, p=cfg.dropout).to(device)

    # Loss
    pos_w = None
    pw = cfg.pos_weight.strip().lower()
    if pw == "auto":
        pos_w = compute_pos_weight_from_labels(train_ds.labels)
    elif pw == "none":
        pos_w = None
    else:
        try:
            pos_w = torch.tensor([float(cfg.pos_weight)], dtype=torch.float32)
        except Exception:
            pos_w = None

    if pos_w is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3, verbose=True, min_lr=1e-6)

    pin = torch.cuda.is_available()
    if cfg.balanced_sampler:
        sampler = make_balanced_sampler(train_ds.labels)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch, sampler=sampler, shuffle=False, num_workers=0, pin_memory=pin)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, num_workers=0, pin_memory=pin)

    best_f1 = -1.0
    best_path = os.path.join(cfg.save_dir, "best.pt")
    epochs_no_improve = 0

    history_path = os.path.join(cfg.save_dir, "history.jsonl")
    if os.path.exists(history_path):
        os.remove(history_path)

    def log_history(row: Dict[str, Any]) -> None:
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train TCN ep{ep}", leave=False)

        running = 0.0
        n_seen = 0

        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)  # [B,1]
            loss = criterion(logits, yb)
            loss.backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()

            running += float(loss.detach().cpu()) * int(xb.shape[0])
            n_seen += int(xb.shape[0])
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        train_loss = running / max(1, n_seen)

        # Validation
        probs, y_true = collect_probs(model, val_loader, device)
        if probs.size == 0:
            print("[val] empty val loader?")
            continue

        best = best_f1_threshold(probs, y_true)
        extras = extra_scores(probs, y_true)
        lr_now = float(opt.param_groups[0]["lr"])

        row = {
            "epoch": ep,
            "train_loss": train_loss,
            "val_F1": best["F1"],
            "val_P": best["P"],
            "val_R": best["R"],
            "val_FPR": best["FPR"],
            "val_thr": best["thr"],
            "val_AP": extras["AP"],
            "val_AUC": extras["AUC"],
            "lr": lr_now,
        }
        log_history(row)

        print(
            f"[val] ep={ep:03d} "
            f"loss={train_loss:.4f}  "
            f"F1={best['F1']:.3f} P={best['P']:.3f} R={best['R']:.3f} FPR={best['FPR']:.3f} thr={best['thr']:.2f}  "
            f"AP={extras['AP']:.3f} AUC={extras['AUC']:.3f}  lr={lr_now:g}"
        )

        scheduler.step(best["F1"])

        if best["F1"] > best_f1 + 1e-6:
            best_f1 = best["F1"]
            epochs_no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "in_ch": C,
                    "best_thr": best["thr"],
                    "best_val": {**best, **extras},
                    "cfg": asdict(cfg),
                },
                best_path,
            )
            print(f"[save] {best_path} (best val F1={best_f1:.3f})")
        else:
            epochs_no_improve += 1
            if cfg.patience > 0 and epochs_no_improve >= cfg.patience:
                print(f"[early stop] patience={cfg.patience} reached at epoch {ep}")
                break

    report = {
        "arch": "tcn",
        "best_val_F1": float(best_f1),
        "best_ckpt": best_path,
        "train_dir": cfg.train_dir,
        "val_dir": cfg.val_dir,
        "seed": cfg.seed,
    }
    report_path = os.path.join(cfg.save_dir, "train_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Optional external report path (Makefile sometimes wants it)
    if cfg.report_json:
        os.makedirs(os.path.dirname(cfg.report_json), exist_ok=True)
        with open(cfg.report_json, "w", encoding="utf-8") as f:
            json.dump({**report, "dataset": cfg.report_dataset_name or ""}, f, indent=2)

    print(f"[done] best val F1={best_f1:.3f}  ckpt={best_path}")
    print(f"[ok] wrote: {report_path} and {history_path}")


if __name__ == "__main__":
    main()
