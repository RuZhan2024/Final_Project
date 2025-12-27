#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/train_gcn.py  (fixed)

This is a corrected version of the file you attached. fileciteturn37file0

Main fixes
- Removed/repair all indentation errors (the root cause of the crash you saw).
- Fixed two-stream forward blocks in train/val loops (xj/xm were referenced when undefined).
- Added explicit TrainConfig fields for threshold sweep + min_epochs with sane defaults.
- Made validation probability collection consistent for two_stream vs single_stream.
- Added NaN/Inf sanitising for inputs/logits (optional but prevents silent metric collapse).

The script trains a two-stream GCN (joints stream + motion stream, optional conf) on window .npz files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm


# -------------------------
# Small safety helpers
# -------------------------
def _ensure_finite_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
    """Replace NaN/Inf with zeros (prevents silent collapse)."""
    if not torch.isfinite(x).all():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _bce_shapes(logits: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalise logits/targets to matching 1D shapes for BCEWithLogitsLoss."""
    if isinstance(logits, torch.Tensor) and logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)  # (B,)
    if isinstance(y, torch.Tensor):
        y = y.float().view(-1)      # (B,)
    return logits, y


# -------------------------
# Imbalance helpers
# -------------------------
def compute_pos_weight(labels) -> Optional[torch.Tensor]:
    """BCEWithLogitsLoss pos_weight = (#neg / #pos). Returns tensor([w]) or None."""
    if labels is None:
        return None
    if isinstance(labels, torch.Tensor):
        y = labels.detach().cpu().numpy()
    else:
        y = np.asarray(labels)
    y = y.reshape(-1)
    pos = float((y > 0.5).sum())
    n = float(y.size)
    neg = n - pos
    if pos <= 0 or neg <= 0:
        return None
    return torch.tensor([neg / pos], dtype=torch.float32)


def make_balanced_sampler(labels) -> WeightedRandomSampler:
    """Inverse-frequency sampler."""
    if isinstance(labels, torch.Tensor):
        y = labels.detach().cpu().numpy()
    else:
        y = np.asarray(labels)
    y = y.reshape(-1)
    n = len(y)
    pos = (y > 0.5).astype(np.float32)
    n_pos = float(pos.sum())
    n_neg = float(n - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        w = np.ones(n, dtype=np.float32)
    else:
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)
        w = np.where(y > 0.5, w_pos, w_neg).astype(np.float32)
    return WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=n, replacement=True)


# -------------------------
# Validation metrics helpers
# -------------------------
def best_metric_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    thr_min: float = 0.05,
    thr_max: float = 0.95,
    thr_step: float = 0.01,
) -> Dict[str, float]:
    """Pick threshold maximising F1 (tie-break: higher P, higher R, lower FPR)."""
    p = np.asarray(probs).reshape(-1)
    y = np.asarray(y_true).reshape(-1).astype(np.int32)
    if p.size == 0 or y.size == 0:
        return {"thr": 0.5, "F1": 0.0, "P": 0.0, "R": 0.0, "FPR": 0.0}

    thr_values = np.arange(thr_min, thr_max + 1e-12, thr_step, dtype=np.float32)

    best = {"thr": float(thr_values[0]), "F1": 0.0, "P": 0.0, "R": 0.0, "FPR": 1.0}
    best_tuple = (-1.0, -1.0, -1.0, -1.0)  # maximise (F1, P, R, -FPR)

    for thr in thr_values:
        pred = (p >= float(thr)).astype(np.int32)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = float(cm[0, 0]), float(cm[0, 1]), float(cm[1, 0]), float(cm[1, 1])

        P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        F1 = (2.0 * P * R / (P + R)) if (P + R) > 0 else 0.0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tup = (F1, P, R, -FPR)
        if tup > best_tuple:
            best_tuple = tup
            best = {"thr": float(thr), "F1": float(F1), "P": float(P), "R": float(R), "FPR": float(FPR)}

    return best


def extra_scores(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    p = np.asarray(probs).reshape(-1)
    y = np.asarray(y_true).reshape(-1)
    out = {"AP": float("nan"), "AUC": float("nan")}
    try:
        out["AP"] = float(average_precision_score(y, p))
    except Exception:
        pass
    try:
        out["AUC"] = float(roc_auc_score(y, p))
    except Exception:
        pass
    return out


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
    if joints.shape[1] > 24:
        return 0.5 * (joints[:, 23:24, :] + joints[:, 24:25, :])
    return joints[:, 23:24, :]


def _derive_mask(joints: np.ndarray, conf: Optional[np.ndarray], conf_gate: float) -> np.ndarray:
    finite = np.isfinite(joints[..., 0]) & np.isfinite(joints[..., 1])
    if conf is None or conf_gate <= 0:
        return finite
    return finite & (conf >= conf_gate)


def _apply_mask(
    joints: np.ndarray,
    motion: np.ndarray,
    conf: Optional[np.ndarray],
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    m = mask.astype(np.float32)[..., None]
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
    joints: np.ndarray,
    motion: Optional[np.ndarray],
    conf: Optional[np.ndarray],
    mask: np.ndarray,
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
        joints = joints - _pelvis_center(joints)

    if use_motion:
        if motion is None:
            motion = _compute_motion(joints, fps, motion_scale_by_fps)
        else:
            motion = np.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            if center == "pelvis":
                motion = _compute_motion(joints, fps, motion_scale_by_fps)
            elif motion_scale_by_fps:
                motion = motion * float(fps)
    else:
        motion = np.zeros_like(joints, dtype=np.float32)

    joints, motion, conf = _apply_mask(joints, motion, conf, mask)

    parts = [joints]
    if use_motion:
        parts.append(motion)
    if use_conf_channel and conf is not None:
        parts.append(conf[..., None].astype(np.float32))
    return np.concatenate(parts, axis=-1).astype(np.float32)  # [T,V,F]


# -------------------------
# DropGraph augmentation
# -------------------------
def augment_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
    mask_joint_p: float,
    mask_frame_p: float,
    ensure_nonempty: bool = True,
) -> np.ndarray:
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
        t = int(rng.integers(0, T))
        v = int(rng.integers(0, V))
        m[t, v] = True

    return m


# -------------------------
# Dataset
# -------------------------
class WindowNPZGraph(Dataset):
    """Returns (x:[T,V,F], y:[1])."""
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
        mask_joint_p: float = 0.0,
        mask_frame_p: float = 0.0,
        seed: int = 33724876,
    ):
        self.root = str(root)
        self.split = str(split)
        files = sorted([str(p) for p in Path(self.root).glob("*.npz")])
        if not files:
            files = sorted([str(p) for p in Path(self.root).glob("**/*.npz")])
        if not files:
            raise FileNotFoundError(f"No .npz windows found under: {root}")

        kept: List[str] = []
        ys: List[int] = []
        skipped = 0
        for p in files:
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
            print(f"[WindowNPZGraph] skipped {skipped} unlabeled/unreadable windows under {root}")

        self.files = kept
        self.labels = np.asarray(ys, dtype=np.int64)

        self.fps_default = float(fps_default)
        self.conf_gate = float(conf_gate)
        self.use_precomputed_mask = bool(use_precomputed_mask)
        self.center = str(center)
        self.use_motion = bool(use_motion)
        self.use_conf_channel = bool(use_conf_channel)
        self.motion_scale_by_fps = bool(motion_scale_by_fps)

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

            joints = d["joints"].astype(np.float32, copy=False) if ("joints" in d.files) else d["xy"].astype(np.float32, copy=False)
            motion = d["motion"].astype(np.float32, copy=False) if ("motion" in d.files) else None
            conf = d["conf"].astype(np.float32, copy=False) if ("conf" in d.files) else None
            fps = _safe_fps(d, self.fps_default)

            if self.use_precomputed_mask and ("mask" in d.files):
                mask = np.array(d["mask"]).astype(bool, copy=False)
            else:
                mask = _derive_mask(joints, conf, self.conf_gate)

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

        return torch.from_numpy(feats).float(), torch.tensor([float(y)], dtype=torch.float32)


# -------------------------
# Graph adjacency (MediaPipe Pose approx)
# -------------------------
def build_mediapipe_adjacency(num_joints: int = 33) -> np.ndarray:
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12),
        (11, 23), (12, 24),
        (23, 24),
        (11, 13), (13, 15), (15, 17),
        (12, 14), (14, 16), (16, 18),
        (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
        (7, 28), (8, 27),
    ]
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        if 0 <= i < num_joints and 0 <= j < num_joints:
            A[i, j] = 1.0
            A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)
    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-6))
    return D_inv_sqrt @ A @ D_inv_sqrt


# -------------------------
# Blocks
# -------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation over channel dimension."""
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        mid = max(4, ch // r)
        self.fc1 = nn.Linear(ch, mid)
        self.fc2 = nn.Linear(mid, ch)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,V,C]
        b, t, v, c = x.shape
        z = x.mean(dim=(1, 2))     # [B,C]
        z = self.relu(self.fc1(z))
        s = self.sig(self.fc2(z)).view(b, 1, 1, c)
        return x * s


class GCNBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, dropout: float = 0.30):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)
        self.ln = nn.LayerNorm(out_feats)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # x: [B,T,V,Cin]
        x = torch.einsum("vw,btwc->btvc", A_hat, x)
        x = self.lin(x)
        x = self.drop(self.relu(self.ln(x)))
        return x


class GCNTemporalEncoder(nn.Module):
    """Produces embedding per sample: input [B,T,V,F] -> [B,E]."""
    def __init__(
        self,
        num_joints: int,
        in_feats: int,
        gcn_hidden: int = 96,
        tcn_hidden: int = 192,
        dropout: float = 0.35,
        use_se: bool = True,
    ):
        super().__init__()
        A_hat = normalize_adjacency(build_mediapipe_adjacency(num_joints))
        self.register_buffer("A_hat", torch.from_numpy(A_hat))

        self.g1 = GCNBlock(in_feats, gcn_hidden, dropout=dropout)
        self.g2 = GCNBlock(gcn_hidden, gcn_hidden, dropout=dropout)

        self.use_se = bool(use_se)
        self.se = SEBlock(gcn_hidden) if self.use_se else nn.Identity()

        self.temporal = nn.Sequential(
            nn.Conv1d(gcn_hidden, tcn_hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(tcn_hidden, tcn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.g1(x, self.A_hat)
        h = self.g2(h, self.A_hat)
        h = self.se(h)
        h = h.mean(dim=2)        # pool joints -> [B,T,C]
        h = h.permute(0, 2, 1)   # [B,C,T]
        h = self.temporal(h).squeeze(-1)  # [B,E]
        return h


class TwoStreamGCN(nn.Module):
    """Two-stream encoder: joints(+conf) and motion(+conf), fused by concat or sum."""
    def __init__(
        self,
        num_joints: int,
        in_feats_joint: int,
        in_feats_motion: int,
        gcn_hidden: int = 96,
        tcn_hidden: int = 192,
        dropout: float = 0.35,
        use_se: bool = True,
        fuse: str = "concat",
    ):
        super().__init__()
        self.enc_j = GCNTemporalEncoder(num_joints, in_feats_joint, gcn_hidden, tcn_hidden, dropout, use_se)
        self.enc_m = GCNTemporalEncoder(num_joints, in_feats_motion, gcn_hidden, tcn_hidden, dropout, use_se)
        self.fuse = str(fuse)

        if self.fuse == "sum":
            self.head = nn.Linear(tcn_hidden, 1)
        else:
            self.head = nn.Linear(tcn_hidden * 2, 1)

    def forward(self, xj: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        ej = self.enc_j(xj)
        em = self.enc_m(xm)
        if self.fuse == "sum":
            e = 0.5 * (ej + em)
        else:
            e = torch.cat([ej, em], dim=-1)
        return self.head(e)


# -------------------------
# Collate helpers
# -------------------------
def collate_two_stream(batch):
    """Split [T,V,F] into (xj, xm) + y for two-stream model.

    F=4 => joints(2) + motion(2)
    F=5 => joints(2) + motion(2) + conf(1)
    """
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # [B,T,V,F]
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.cat(ys, dim=0).view(-1)  # [B]

    if x.ndim != 4:
        raise ValueError(f"Expected x [B,T,V,F], got {tuple(x.shape)}")
    F = x.shape[-1]
    if F == 5:
        conf = x[..., 4:5]
        xj = torch.cat([x[..., 0:2], conf], dim=-1)
        xm = torch.cat([x[..., 2:4], conf], dim=-1)
    elif F == 4:
        xj = x[..., 0:2]
        xm = x[..., 2:4]
    else:
        raise ValueError(f"Two-stream expects F=4 or F=5, got F={F}. Check window feature flags.")
    return (xj, xm), y


# -------------------------
# Train config
# -------------------------
@dataclass
class TrainConfig:
    # data
    train_dir: str
    val_dir: str
    test_dir: Optional[str]

    # optim
    epochs: int
    batch: int
    lr: float
    seed: int
    fps_default: float
    save_dir: str
    grad_clip: float
    patience: int

    # model
    gcn_hidden: int
    tcn_hidden: int
    dropout: float
    use_se: bool
    two_stream: bool
    fuse: str

    # feature flags (must match windows + fit/eval)
    center: str
    use_motion: bool
    use_conf_channel: bool
    motion_scale_by_fps: bool
    conf_gate: float
    use_precomputed_mask: bool

    # augmentation / imbalance
    mask_joint_p: float
    mask_frame_p: float
    pos_weight: str
    balanced_sampler: bool

    # threshold sweep (val) for picking best F1 threshold
    thr_min: float = 0.05
    thr_max: float = 0.95
    thr_step: float = 0.05
    min_epochs: int = 5

    # regularisation / selection
    weight_decay: float = 1e-4
    monitor: str = "F1"          # "F1" | "AP" | "AUC"
    label_smoothing: float = 0.0
    num_workers: int = 0

    # reporting
    report_json: Optional[str] = None
    report_dataset_name: Optional[str] = None


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Train a redesigned GCN temporal model on skeleton windows (new schema).")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--test_dir", default=None)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=33724876)

    ap.add_argument("--fps_default", type=float, default=30.0)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=12)

    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--monitor", choices=["F1", "AP", "AUC"], default="F1")
    ap.add_argument("--label_smoothing", type=float, default=0.0)

    ap.add_argument("--thr_min", type=float, default=0.05)
    ap.add_argument("--thr_max", type=float, default=0.95)
    ap.add_argument("--thr_step", type=float, default=0.05)
    ap.add_argument("--min_epochs", type=int, default=5)

    # model knobs
    ap.add_argument("--gcn_hidden", type=int, default=96)
    ap.add_argument("--tcn_hidden", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--use_se", type=int, default=1)
    ap.add_argument("--two_stream", type=int, default=1)
    ap.add_argument("--fuse", choices=["concat", "sum"], default="concat")

    # features
    ap.add_argument("--center", choices=["pelvis", "none"], default="pelvis")
    ap.add_argument("--use_motion", type=int, default=1)
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

    ap.add_argument("--report_json", default=None)
    ap.add_argument("--report_dataset_name", default=None)

    args = ap.parse_args()

    cfg = TrainConfig(
        train_dir=str(args.train_dir),
        val_dir=str(args.val_dir),
        test_dir=str(args.test_dir) if args.test_dir else None,
        epochs=int(args.epochs),
        batch=int(args.batch),
        lr=float(args.lr),
        seed=int(args.seed),
        fps_default=float(args.fps_default),
        save_dir=str(args.save_dir),
        grad_clip=float(args.grad_clip),
        patience=int(args.patience),
        gcn_hidden=int(args.gcn_hidden),
        tcn_hidden=int(args.tcn_hidden),
        dropout=float(args.dropout),
        use_se=bool(int(args.use_se)),
        two_stream=bool(int(args.two_stream)),
        fuse=str(args.fuse),
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
        weight_decay=float(args.weight_decay),
        monitor=str(args.monitor),
        label_smoothing=float(args.label_smoothing),
        num_workers=int(args.num_workers),
        thr_min=float(args.thr_min),
        thr_max=float(args.thr_max),
        thr_step=float(args.thr_step),
        min_epochs=int(args.min_epochs),
        report_json=str(args.report_json) if args.report_json else None,
        report_dataset_name=str(args.report_dataset_name) if args.report_dataset_name else None,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = pick_device()
    print(f"[info] device: {device.type}")

    train_ds = WindowNPZGraph(
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
    val_ds = WindowNPZGraph(
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
    T, V, F_full = x0.shape
    print(f"[info] window shape: T={T}, V={V}, F={F_full}; first y={float(y0.item())}")

    if cfg.two_stream and (not cfg.use_motion):
        raise SystemExit("[error] --two_stream requires --use_motion=1.")

    # build model
    if cfg.two_stream:
        has_conf = (F_full == 5)
        in_j = 3 if has_conf else 2
        in_m = 3 if has_conf else 2
        model: nn.Module = TwoStreamGCN(
            num_joints=V,
            in_feats_joint=in_j,
            in_feats_motion=in_m,
            gcn_hidden=cfg.gcn_hidden,
            tcn_hidden=cfg.tcn_hidden,
            dropout=cfg.dropout,
            use_se=cfg.use_se,
            fuse=cfg.fuse,
        ).to(device)
        collate_fn = collate_two_stream
    else:
        backbone = GCNTemporalEncoder(
            num_joints=V,
            in_feats=F_full,
            gcn_hidden=cfg.gcn_hidden,
            tcn_hidden=cfg.tcn_hidden,
            dropout=cfg.dropout,
            use_se=cfg.use_se,
        ).to(device)
        model = nn.Sequential(backbone, nn.Linear(cfg.tcn_hidden, 1)).to(device)
        collate_fn = None

    # loss
    pos_w = None
    pw = cfg.pos_weight.strip().lower()
    if pw == "auto":
        pos_w = compute_pos_weight(train_ds.labels)
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

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    pin = torch.cuda.is_available()
    if cfg.balanced_sampler:
        sampler = make_balanced_sampler(train_ds.labels)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch, sampler=sampler, shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=pin, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=pin, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin, collate_fn=collate_fn)

    best_metric = -1.0
    best_path = os.path.join(cfg.save_dir, "best.pt")
    epochs_no_improve = 0

    history_path = os.path.join(cfg.save_dir, "history.jsonl")
    if os.path.exists(history_path):
        os.remove(history_path)

    def log_history(row: Dict[str, Any]) -> None:
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @torch.no_grad()
    def collect_probs(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        probs_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        for xb, yb in loader:
            if cfg.two_stream:
                xj, xm = xb
                xj = _ensure_finite_tensor(xj.to(device), "xj")
                xm = _ensure_finite_tensor(xm.to(device), "xm")
                logits = model(xj, xm)  # [B,1]
            else:
                x = _ensure_finite_tensor(xb.to(device), "xb")
                logits = model(x)       # [B,1]
            logits = _ensure_finite_tensor(logits, "logits")
            logits, yb_t = _bce_shapes(logits, yb.to(device))
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            y = yb_t.detach().cpu().numpy().reshape(-1).astype(int)
            probs_list.append(probs)
            y_list.append(y)
        probs_all = np.concatenate(probs_list, axis=0) if probs_list else np.array([])
        y_all = np.concatenate(y_list, axis=0) if y_list else np.array([])
        return probs_all, y_all

    # training loop
    for ep in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train GCN ep{ep}", leave=False)
        running = 0.0
        n_seen = 0

        for xb, yb in pbar:
            if cfg.two_stream:
                xj, xm = xb
                xj = _ensure_finite_tensor(xj.to(device), "xj")
                xm = _ensure_finite_tensor(xm.to(device), "xm")
                logits = model(xj, xm)  # [B,1]
            else:
                x = _ensure_finite_tensor(xb.to(device), "xb")
                logits = model(x)       # [B,1]

            yb = yb.to(device)
            logits = _ensure_finite_tensor(logits, "logits")
            logits, yb = _bce_shapes(logits, yb)

            # label smoothing (optional)
            if cfg.label_smoothing and cfg.label_smoothing > 0:
                eps = float(cfg.label_smoothing)
                yb = yb * (1.0 - eps) + 0.5 * eps

            opt.zero_grad(set_to_none=True)
            loss = criterion(logits, yb)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss encountered")

            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            bs = int(yb.shape[0])
            running += float(loss.detach().cpu()) * bs
            n_seen += bs
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        train_loss = running / max(1, n_seen)

        probs, y_true = collect_probs(val_loader)
        if probs.size == 0:
            print("[val] empty val loader?")
            continue

        best = best_metric_threshold(probs, y_true, thr_min=cfg.thr_min, thr_max=cfg.thr_max, thr_step=cfg.thr_step)
        extras = extra_scores(probs, y_true)
        lr_now = float(opt.param_groups[0]["lr"])

        monitor = (cfg.monitor or "F1").upper()
        if monitor == "AP":
            monitor_value = float(extras["AP"])
        elif monitor == "AUC":
            monitor_value = float(extras["AUC"])
        else:
            monitor_value = float(best["F1"])

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
            "val_monitor": monitor_value,
            "monitor": monitor,
            "lr": lr_now,
        }
        log_history(row)

        print(
            f"[val] ep={ep:03d} loss={train_loss:.4f}  "
            f"F1={best['F1']:.3f} P={best['P']:.3f} R={best['R']:.3f} FPR={best['FPR']:.3f} thr={best['thr']:.2f}  "
            f"AP={extras['AP']:.3f} AUC={extras['AUC']:.3f}  monitor({monitor})={monitor_value:.3f}  lr={lr_now:g}"
        )

        scheduler.step(monitor_value)

        if monitor_value > best_metric + 1e-6:
            best_metric = monitor_value
            epochs_no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_joints": V,
                    "in_feats": F_full,
                    "best_thr": best["thr"],
                    "best_val": {**best, **extras, "monitor": monitor, "val_monitor": monitor_value},
                    "cfg": asdict(cfg),
                    "model_type": "TwoStreamGCN" if cfg.two_stream else "GCNTemporal+Linear",
                },
                best_path,
            )
            print(f"[save] {best_path} (best val {monitor}={best_metric:.3f})")
        else:
            epochs_no_improve += 1
            if cfg.patience > 0 and epochs_no_improve >= cfg.patience and ep >= cfg.min_epochs:
                print(f"[early stop] patience={cfg.patience} reached at epoch {ep}")
                break

    report = {
        "arch": "gcn",
        "monitor": str(cfg.monitor).upper(),
        "best_val_monitor": float(best_metric),
        "best_ckpt": best_path,
        "train_dir": cfg.train_dir,
        "val_dir": cfg.val_dir,
        "seed": cfg.seed,
        "two_stream": bool(cfg.two_stream),
        "fuse": cfg.fuse,
    }
    report_path = os.path.join(cfg.save_dir, "train_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if cfg.report_json:
        os.makedirs(os.path.dirname(cfg.report_json), exist_ok=True)
        with open(cfg.report_json, "w", encoding="utf-8") as f:
            json.dump({**report, "dataset": cfg.report_dataset_name or ""}, f, indent=2)

    print(f"[done] best val {str(cfg.monitor).upper()}={best_metric:.3f}  ckpt={best_path}")
    print(f"[ok] wrote: {report_path} and {history_path}")


if __name__ == "__main__":
    main()
