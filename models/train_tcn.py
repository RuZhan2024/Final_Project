#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/train_tcn.py

A clean, reproducible TCN trainer that uses core/* as the single source of truth.

What this script does (end-to-end):
1) Load window .npz files from train/val dirs.
2) Convert each window into a TCN input tensor using core/features.py.
3) Train a temporal convolution model to predict fall/non-fall.
4) Evaluate each epoch (F1 + AP/AUC).
5) Save the best checkpoint BUNDLE (core/ckpt.py) + training logs.

Key concepts you should understand:
- "Window": a short fixed-length clip (T frames) used as one training sample.
- "TCN input": shape [T, C], where C is the number of features per frame.
- "Logits": raw model output BEFORE sigmoid.
- "Probability": sigmoid(logits) in [0, 1].
"""

from __future__ import annotations

# ============================================================
# 0) Minimal path bootstrap
# ============================================================
# Purpose: allow "from core.* import ..." when you run this file directly:
#   python models/train_tcn.py ...
#
# It inserts the repo root into sys.path one time (no duplicates).
import os as _os
import sys as _sys


def _ensure_repo_root_on_syspath() -> None:
    repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    if repo_root not in _sys.path:
        _sys.path.insert(0, repo_root)


_ensure_repo_root_on_syspath()

# ============================================================
# 1) Imports
# ============================================================
import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from core.augment import AugCfg, apply_augmentations
from core.ckpt import get_cfg, load_ckpt, save_ckpt
from core.ema import EMA
from core.features import FeatCfg, build_tcn_input, read_window_npz
from core.losses import BinaryFocalLossWithLogits
from core.metrics import ap_auc, best_threshold_by_f1
from core.models import TCN, TCNConfig, logits_1d, pick_device


# ============================================================
# 2) Small utilities (repro + reporting)
# ============================================================
def set_seed(seed: int) -> None:
    """
    Reproducibility helper.

    Why:
    - Training uses random sampling (shuffling, augmentation).
    - Setting seeds helps you get repeatable results across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    """
    Safe JSON writer.

    - If path is None/empty, do nothing.
    - Creates parent folders automatically.
    """
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"[warn] failed to write json at {path}: {e}")


def _apply_label_smoothing(y: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Label smoothing for binary labels.

    y is expected to be float tensor containing {0,1}.
    eps in [0, 1).

    Formula:
      y_smooth = y*(1-eps) + 0.5*eps

    Why it helps:
    - Reduces overconfidence.
    - Sometimes improves generalization for noisy labels.
    """
    if eps <= 0:
        return y
    return y * (1.0 - eps) + 0.5 * eps


# ============================================================
# 3) Dataset: load window NPZ -> build TCN features
# ============================================================
class WindowDatasetTCN(Dataset):
    """
    Each item is ONE window file (.npz).

    What __getitem__ returns:
      X: FloatTensor [T, C]  (TCN input)
      y: FloatTensor [1]     (binary label: 0 non-fall, 1 fall)

    Important inputs:
    - feat_cfg tells build_tcn_input() which channels to include:
      * use_motion
      * use_conf_channel
      * conf_gate
      * etc.
    """

    def __init__(
        self,
        root: str,
        *,
        split: str,
        feat_cfg: FeatCfg,
        fps_default: float,
        skip_unlabeled: bool = True,
        hard_neg_files: Optional[List[str]] = None,
        hard_neg_mult: int = 0,
        aug_cfg: Optional[AugCfg] = None,
        seed: int = 33724876,
    ):
        import glob

        self.root = str(root)
        self.split = str(split)
        self.feat_cfg = feat_cfg
        self.fps_default = float(fps_default)

        # Only train split uses augmentation
        self.aug_cfg = aug_cfg if self.split == "train" else None

        # Base seed used to create per-sample RNGs (stable augmentations)
        self.base_seed = int(seed)

        # Discover all window files
        files = sorted(glob.glob(os.path.join(self.root, "**", "*.npz"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No .npz under: {self.root}")

        kept: List[str] = []
        labels: List[int] = []
        skipped = 0

        # Validate each NPZ quickly (and optionally skip unlabeled windows)
        for p in files:
            try:
                _, _, _, _, _, meta = read_window_npz(p, fps_default=self.fps_default)
                if skip_unlabeled and meta.y < 0:
                    skipped += 1
                    continue
                kept.append(p)
                labels.append(int(meta.y if meta.y >= 0 else 0))
            except Exception:
                skipped += 1

        if not kept:
            raise FileNotFoundError(f"All windows under {self.root} were unlabeled/unreadable.")
        if skipped:
            print(f"[dataset] skipped {skipped} unlabeled/unreadable windows in {self.root}")

        self.files = kept
        self.labels = np.asarray(labels, dtype=np.int64)

        # Optional: oversample hard negatives (paths listed in a file).
        # This increases the frequency of difficult negatives during training.
        if hard_neg_files:
            extra: List[str] = []
            for p in hard_neg_files:
                p = str(p).strip()
                if not p or not os.path.exists(p):
                    continue

                # Avoid accidentally including positive windows as "hard negatives"
                _, _, _, _, _, meta2 = read_window_npz(p, fps_default=self.fps_default)
                if meta2.y == 1:
                    continue
                extra.append(p)

            if extra:
                mult = int(hard_neg_mult) if int(hard_neg_mult) > 0 else 1
                extra_rep = extra * mult
                self.files.extend(extra_rep)
                self.labels = np.concatenate(
                    [self.labels, np.zeros(len(extra_rep), dtype=np.int64)], axis=0
                )

    def _make_rng(self, idx: int) -> np.random.Generator:
        """
        Creates a deterministic RNG per sample index, even in multi-worker dataloading.

        Why:
        - If num_workers > 0, each worker has its own process.
        - get_worker_info() gives worker id so we can offset the seed.
        """
        try:
            from torch.utils.data import get_worker_info

            wi = get_worker_info()
            wid = int(wi.id) if wi is not None else 0
        except Exception:
            wid = 0
        return np.random.default_rng(self.base_seed + int(idx) + wid * 1_000_003)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        # p is the file path of one window NPZ
        p = self.files[idx]

        # Read raw arrays from the NPZ (joints, motion, conf, mask, fps, meta)
        joints, motion, conf, mask, fps, meta = read_window_npz(
            p, fps_default=self.fps_default
        )

        # Apply augmentation only for training:
        # apply_augmentations can flip, jitter, occlude, mask, time-shift, etc.
        if self.aug_cfg is not None:
            rng = self._make_rng(idx)
            joints, conf, mask = apply_augmentations(
                joints=joints,
                conf=conf,
                mask=mask,
                fps=float(fps),
                feat_conf_gate=float(self.feat_cfg.conf_gate),
                rng=rng,
                cfg=self.aug_cfg,
                training=True,
            )
            # Motion must be recomputed after augmentations (we set motion=None)
            motion = None

        # Convert arrays -> the TCN input features: X shape [T, C]
        X, _ = build_tcn_input(joints, motion, conf, mask, fps, self.feat_cfg)

        # Binary label: meta.y is -1 for unlabeled; we treat it as 0 if present
        y = float(meta.y if meta.y >= 0 else 0.0)

        return torch.from_numpy(X).float(), torch.tensor([y], dtype=torch.float32)


# ============================================================
# 4) Class imbalance tools
# ============================================================
def compute_pos_weight(y: np.ndarray) -> Optional[torch.Tensor]:
    """
    Computes pos_weight = (#neg / #pos)

    Used by BCEWithLogitsLoss(pos_weight=...).

    Why:
    - If positives are rare, the model can get "lazy" predicting all negatives.
    - pos_weight pushes the loss to care more about positives.
    """
    y = np.asarray(y).reshape(-1)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos <= 0 or neg <= 0:
        return None
    w = neg / pos
    print(f"[info] class balance: pos={int(pos)} neg={int(neg)} pos_weight={w:.2f}")
    return torch.tensor([w], dtype=torch.float32)


def make_balanced_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """
    Alternative imbalance strategy: balanced sampling.

    It creates sampling weights so positives/negatives appear roughly equally often.
    """
    y = np.asarray(y).reshape(-1)
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    w_pos = 1.0 / pos
    w_neg = 1.0 / neg
    weights = np.where(y == 1, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights), num_samples=len(weights), replacement=True
    )


# ============================================================
# 5) Evaluation helper: collect probs on a loader
# ============================================================
@torch.no_grad()
def collect_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the model on loader and returns:
      probs: sigmoid(logits) -> predicted probabilities, shape [N]
      y_true: int labels, shape [N]
    """
    model.eval()
    ps: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device)

        # logits_1d ensures output shape is [B]
        logits = logits_1d(model(xb))

        p = torch.sigmoid(logits).detach().cpu().numpy()
        y = yb.detach().cpu().numpy().reshape(-1).astype(int)

        ps.append(p)
        ys.append(y)

    return (np.concatenate(ps) if ps else np.array([])), (np.concatenate(ys) if ys else np.array([]))


# ============================================================
# 6) Training configuration dataclass
# ============================================================
@dataclass
class TrainCfg:
    # Data
    train_dir: str
    val_dir: str
    save_dir: str
    test_dir: Optional[str] = None

    # Resume & hard negatives
    resume: Optional[str] = None
    hard_neg_list: Optional[str] = None
    hard_neg_mult: int = 0

    # Optimization
    epochs: int = 50
    batch: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Repro & training control
    seed: int = 33724876
    patience: int = 12
    min_epochs: int = 0

    # LR scheduler ReduceLROnPlateau
    lr_plateau_patience: int = 5
    lr_plateau_factor: float = 0.5
    lr_plateau_min_lr: float = 1e-6

    # Data loading
    num_workers: int = 0
    fps_default: float = 30.0

    # Monitoring metric
    monitor: str = "ap"  # "ap" or "f1"

    # Imbalance strategy (use ONLY ONE: balanced_sampler OR pos_weight)
    pos_weight: str = "auto"  # "auto" | "none" | "<float>"
    balanced_sampler: bool = False

    # Loss choice
    loss: str = "bce"  # "bce" | "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

    # EMA (Exponential Moving Average of weights)
    ema_decay: float = 0.999  # 0 disables

    # Augmentation
    mask_joint_p: float = 0.15
    mask_frame_p: float = 0.10
    aug_hflip_p: float = 0.50
    aug_jitter_std: float = 0.008
    aug_jitter_conf_scaled: int = 1
    aug_occ_p: float = 0.30
    aug_occ_min_len: int = 3
    aug_occ_max_len: int = 10
    aug_time_shift: int = 1

    # TCN architecture
    hidden: int = 128
    dropout: float = 0.30
    num_blocks: int = 4
    kernel: int = 3

    # Feature flags (passed into core/features.py)
    center: str = "pelvis"
    use_motion: int = 1
    use_conf_channel: int = 1
    motion_scale_by_fps: int = 1
    conf_gate: float = 0.20
    use_precomputed_mask: int = 1


# ============================================================
# 7) Main training routine
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()

    # ---- Required paths ----
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--test_dir", default=None)
    ap.add_argument("--save_dir", required=True)

    # ---- Optional reporting ----
    ap.add_argument("--report_json", default=None, help="Write a training summary JSON to this path.")
    ap.add_argument("--report_dataset_name", default=None, help="Dataset name stored in report JSON.")

    # ---- Resume & hard negatives ----
    ap.add_argument("--resume", default=None, help="Path to a checkpoint bundle to fine-tune from.")
    ap.add_argument("--hard_neg_list", default=None, help="TXT file containing NPZ paths of hard negatives.")
    ap.add_argument("--hard_neg_mult", type=int, default=0, help="Oversample multiplier for hard negatives (e.g., 5).")

    # ---- Optimization ----
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # ---- Training control ----
    ap.add_argument("--seed", type=int, default=33724876)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min_epochs", type=int, default=0)

    # ---- Scheduler knobs ----
    ap.add_argument("--lr_plateau_patience", type=int, default=5)
    ap.add_argument("--lr_plateau_factor", type=float, default=0.5)
    ap.add_argument("--lr_plateau_min_lr", type=float, default=1e-6)

    # ---- Loader ----
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--fps_default", type=float, default=30.0)

    # ---- Monitoring ----
    ap.add_argument("--monitor", type=lambda s: str(s).strip().lower(), choices=["f1", "ap"], default="ap")

    # ---- Imbalance ----
    ap.add_argument("--pos_weight", default="auto", help="auto | none | <float>")
    ap.add_argument("--balanced_sampler", action="store_true")

    # ---- Loss ----
    ap.add_argument("--loss", type=lambda s: str(s).strip().lower(), choices=["bce", "focal"], default="bce")
    ap.add_argument("--focal_alpha", type=float, default=0.25)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)

    # ---- EMA ----
    ap.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay; 0 disables.")

    # ---- Augmentation ----
    ap.add_argument("--mask_joint_p", type=float, default=0.15)
    ap.add_argument("--mask_frame_p", type=float, default=0.10)
    ap.add_argument("--aug_hflip_p", type=float, default=0.50)
    ap.add_argument("--aug_jitter_std", type=float, default=0.008)
    ap.add_argument("--aug_jitter_conf_scaled", type=int, default=1)
    ap.add_argument("--aug_occ_p", type=float, default=0.30)
    ap.add_argument("--aug_occ_min_len", type=int, default=3)
    ap.add_argument("--aug_occ_max_len", type=int, default=10)
    ap.add_argument("--aug_time_shift", type=int, default=1)

    # ---- TCN architecture ----
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--num_blocks", type=int, default=4)
    ap.add_argument("--kernel", type=int, default=3)

    # ---- Feature flags ----
    ap.add_argument("--center", choices=["pelvis", "none"], default="pelvis")
    ap.add_argument("--scale", default=None, help="(deprecated) ignored; scaling is handled in core/features.py")
    ap.add_argument("--use_motion", type=int, default=1)
    ap.add_argument("--use_conf_channel", type=int, default=1)
    ap.add_argument("--motion_scale_by_fps", type=int, default=1)
    ap.add_argument("--conf_gate", type=float, default=0.20)
    ap.add_argument("--use_precomputed_mask", type=int, default=1)

    args = ap.parse_args()

    # Warn about deprecated flags (safe, keeps Makefile compatibility)
    if getattr(args, "scale", None) not in (None, "", "none"):
        print(f"[warn] --scale={args.scale!r} is deprecated and ignored.")

    # Build config object (single source of truth for later logging)
    cfg = TrainCfg(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir if args.test_dir else None,
        save_dir=args.save_dir,
        resume=args.resume,
        hard_neg_list=args.hard_neg_list,
        hard_neg_mult=int(args.hard_neg_mult),
        epochs=int(args.epochs),
        batch=int(args.batch),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        seed=int(args.seed),
        patience=int(args.patience),
        min_epochs=int(args.min_epochs),
        lr_plateau_patience=int(args.lr_plateau_patience),
        lr_plateau_factor=float(args.lr_plateau_factor),
        lr_plateau_min_lr=float(args.lr_plateau_min_lr),
        num_workers=int(args.num_workers),
        fps_default=float(args.fps_default),
        monitor=str(args.monitor),
        pos_weight=str(args.pos_weight),
        balanced_sampler=bool(args.balanced_sampler),
        loss=str(args.loss),
        focal_alpha=float(args.focal_alpha),
        focal_gamma=float(args.focal_gamma),
        label_smoothing=float(args.label_smoothing),
        ema_decay=float(args.ema_decay),
        mask_joint_p=float(args.mask_joint_p),
        mask_frame_p=float(args.mask_frame_p),
        aug_hflip_p=float(args.aug_hflip_p),
        aug_jitter_std=float(args.aug_jitter_std),
        aug_jitter_conf_scaled=int(args.aug_jitter_conf_scaled),
        aug_occ_p=float(args.aug_occ_p),
        aug_occ_min_len=int(args.aug_occ_min_len),
        aug_occ_max_len=int(args.aug_occ_max_len),
        aug_time_shift=int(args.aug_time_shift),
        hidden=int(args.hidden),
        dropout=float(args.dropout),
        num_blocks=int(args.num_blocks),
        kernel=int(args.kernel),
        center=str(args.center),
        use_motion=int(args.use_motion),
        use_conf_channel=int(args.use_conf_channel),
        motion_scale_by_fps=int(args.motion_scale_by_fps),
        conf_gate=float(args.conf_gate),
        use_precomputed_mask=int(args.use_precomputed_mask),
    )

    # Enforce: use only ONE imbalance strategy
    if cfg.balanced_sampler and str(cfg.pos_weight).strip().lower() not in {"none", "0", "0.0"}:
        if str(cfg.pos_weight).strip().lower() == "auto":
            print("[warn] --balanced_sampler and --pos_weight=auto both set. Using balanced_sampler only.")
            cfg.pos_weight = "none"

    # Repro + output folder
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Save the config so you can reproduce runs later
    _write_json(os.path.join(cfg.save_dir, "train_config.json"), asdict(cfg))

    device = pick_device()
    print(f"[info] device: {device.type}")

    # ------------------------------------------------------------
    # Resume handling
    # ------------------------------------------------------------
    bundle = None
    if cfg.resume:
        bundle = load_ckpt(cfg.resume, map_location="cpu")
        arch0, model_cfg_d0, feat_cfg_d0, data_cfg0 = get_cfg(bundle)

        # Safety: make sure you don’t load TCN weights into a non-TCN model
        if arch0 and arch0 != "tcn":
            raise SystemExit(f"[err] resume arch mismatch: expected 'tcn', got {arch0}")

        # Important: for consistent train/eval, reuse feature flags from checkpoint
        cfg.center = str(feat_cfg_d0.get("center", cfg.center))
        cfg.use_motion = int(feat_cfg_d0.get("use_motion", cfg.use_motion))
        cfg.use_conf_channel = int(feat_cfg_d0.get("use_conf_channel", cfg.use_conf_channel))
        cfg.motion_scale_by_fps = int(feat_cfg_d0.get("motion_scale_by_fps", cfg.motion_scale_by_fps))
        cfg.conf_gate = float(feat_cfg_d0.get("conf_gate", cfg.conf_gate))
        cfg.use_precomputed_mask = int(feat_cfg_d0.get("use_precomputed_mask", cfg.use_precomputed_mask))

        # Keep architecture consistent when resuming
        cfg.hidden = int(model_cfg_d0.get("hidden", cfg.hidden))
        cfg.dropout = float(model_cfg_d0.get("dropout", cfg.dropout))
        cfg.num_blocks = int(model_cfg_d0.get("num_blocks", cfg.num_blocks))
        cfg.kernel = int(model_cfg_d0.get("kernel", cfg.kernel))

        # Keep fps_default consistent (important for motion scaling)
        cfg.fps_default = float(data_cfg0.get("fps_default", cfg.fps_default))

        print(f"[info] resume: loaded cfg from {cfg.resume}")

    # Build feature configuration used by core/features.py
    feat_cfg = FeatCfg(
        center=cfg.center,
        use_motion=bool(cfg.use_motion),
        use_conf_channel=bool(cfg.use_conf_channel),
        motion_scale_by_fps=bool(cfg.motion_scale_by_fps),
        conf_gate=float(cfg.conf_gate),
        use_precomputed_mask=bool(cfg.use_precomputed_mask),
    )

    # Load hard negative paths (optional)
    hard_neg_files: Optional[List[str]] = None
    if cfg.hard_neg_list:
        try:
            with open(cfg.hard_neg_list, "r", encoding="utf-8") as f:
                hard_neg_files = [ln.strip() for ln in f.readlines() if ln.strip()]
        except Exception:
            hard_neg_files = None

    # Build augmentation config (used only for train split)
    aug_cfg = AugCfg(
        hflip_p=float(cfg.aug_hflip_p),
        jitter_std=float(cfg.aug_jitter_std),
        jitter_conf_scaled=bool(int(cfg.aug_jitter_conf_scaled)),
        mask_joint_p=float(cfg.mask_joint_p),
        mask_frame_p=float(cfg.mask_frame_p),
        occ_p=float(cfg.aug_occ_p),
        occ_min_len=int(cfg.aug_occ_min_len),
        occ_max_len=int(cfg.aug_occ_max_len),
        time_shift=int(cfg.aug_time_shift),
        conf_gate=float(cfg.conf_gate),
    )

    # Create datasets
    train_ds = WindowDatasetTCN(
        cfg.train_dir,
        split="train",
        feat_cfg=feat_cfg,
        fps_default=cfg.fps_default,
        skip_unlabeled=True,
        hard_neg_files=hard_neg_files,
        hard_neg_mult=int(cfg.hard_neg_mult),
        aug_cfg=aug_cfg,
        seed=cfg.seed,
    )
    val_ds = WindowDatasetTCN(
        cfg.val_dir,
        split="val",
        feat_cfg=feat_cfg,
        fps_default=cfg.fps_default,
        skip_unlabeled=True,
        aug_cfg=None,  # no augmentation on validation
        seed=cfg.seed,
    )

    # Infer input shape from first training sample
    x0, _ = train_ds[0]
    T, C = x0.shape
    print(f"[info] window shape: T={T}, C={C}")

    # Build the model
    model_cfg = TCNConfig(hidden=cfg.hidden, dropout=cfg.dropout, num_blocks=cfg.num_blocks, kernel=cfg.kernel)
    model = TCN(
        in_ch=C,
        hidden=model_cfg.hidden,
        dropout=model_cfg.dropout,
        num_blocks=model_cfg.num_blocks,
        kernel=model_cfg.kernel,
    ).to(device)

    # Load weights if resuming
    if bundle is not None:
        model.load_state_dict(bundle["state_dict"], strict=True)
        print(f"[info] resumed weights from: {cfg.resume}")

    # EMA tracks a smoothed version of model weights for more stable validation
    ema = None
    if float(cfg.ema_decay) > 0:
        ema = EMA(model, decay=float(cfg.ema_decay))
        ema.update(model)  # initialize shadow weights

    # -------------------------
    # Loss function selection
    # -------------------------
    pos_w = None
    pw = str(cfg.pos_weight).strip().lower()
    if pw == "auto":
        pos_w = compute_pos_weight(train_ds.labels)
    elif pw == "none":
        pos_w = None
    else:
        try:
            pos_w = torch.tensor([float(cfg.pos_weight)], dtype=torch.float32)
        except Exception:
            pos_w = None

    if cfg.loss == "focal":
        # Focal loss is often helpful when positives are rare and “hard examples” matter.
        # Note: we do not apply pos_weight together with focal to avoid double-boosting.
        if pos_w is not None:
            print("[warn] focal loss ignores pos_weight; using focal only.")
        criterion = BinaryFocalLossWithLogits(
            alpha=float(cfg.focal_alpha),
            gamma=float(cfg.focal_gamma),
            reduction="mean",
        )
    else:
        # BCEWithLogitsLoss expects logits (not probabilities)
        if pos_w is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
        else:
            criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    # Scheduler: Reduce LR when validation score plateaus
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=float(cfg.lr_plateau_factor),
        patience=int(cfg.lr_plateau_patience),
        min_lr=float(cfg.lr_plateau_min_lr),
    )

    # -------------------------
    # Data loaders
    # -------------------------
    pin = torch.cuda.is_available()

    if cfg.balanced_sampler:
        sampler = make_balanced_sampler(train_ds.labels)
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg.batch),
            sampler=sampler,
            shuffle=False,  # sampler and shuffle are mutually exclusive
            num_workers=int(cfg.num_workers),
            pin_memory=pin,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg.batch),
            shuffle=True,
            num_workers=int(cfg.num_workers),
            pin_memory=pin,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=pin,
    )

    # -------------------------
    # Training loop bookkeeping
    # -------------------------
    best_score = -1.0
    best_path = os.path.join(cfg.save_dir, "best.pt")
    history_path = os.path.join(cfg.save_dir, "history.jsonl")

    # Reset history file each run for clarity
    if os.path.exists(history_path):
        os.remove(history_path)

    def log_row(row: Dict[str, Any]) -> None:
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    no_improve = 0

    # ============================================================
    # 8) Epoch loop
    # ============================================================
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        pbar = tqdm(train_loader, desc=f"train TCN ep{ep}", leave=False)

        for xb, yb in pbar:
            xb = xb.to(device)

            # yb: shape [B, 1] -> reshape to [B]
            yb = yb.to(device).view(-1)

            # Optional label smoothing
            yb = _apply_label_smoothing(yb, float(cfg.label_smoothing))

            # Forward -> logits
            logits = logits_1d(model(xb))

            # Loss on logits (not sigmoid)
            loss = criterion(logits, yb)

            # Backprop
            opt.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping prevents exploding gradients
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))

            opt.step()

            # Update EMA after optimizer update
            if ema is not None:
                ema.update(model)

            # Track loss
            bs = int(xb.shape[0])
            running_loss += float(loss.detach().cpu()) * bs
            n_seen += bs

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        train_loss = running_loss / max(1, n_seen)

        # -------------------------
        # Validation evaluation
        # -------------------------
        if ema is not None:
            # Evaluate using EMA-smoothed weights for more stable metrics
            with ema.average_parameters(model):
                probs, y_true = collect_probs(model, val_loader, device)
        else:
            probs, y_true = collect_probs(model, val_loader, device)

        if probs.size == 0:
            print("[val] empty validation set?")
            continue

        # Sweep threshold to maximize F1
        sweep_best = best_threshold_by_f1(probs, y_true, thr_min=0.05, thr_max=0.95, thr_step=0.01)

        # Compute AP/AUC as threshold-free metrics
        extras = ap_auc(probs, y_true)

        # Select which metric to monitor for early stopping / best checkpoint
        score = float(sweep_best["f1"]) if cfg.monitor == "f1" else float(extras.get("ap", float("nan")))
        lr_now = float(opt.param_groups[0]["lr"])

        row = {
            "epoch": ep,
            "train_loss": float(train_loss),
            "val_f1": float(sweep_best["f1"]),
            "val_precision": float(sweep_best["precision"]),
            "val_recall": float(sweep_best["recall"]),
            "val_fpr": float(sweep_best["fpr"]),
            "val_thr": float(sweep_best["thr"]),
            "val_ap": float(extras.get("ap", float("nan"))),
            "val_auc": float(extras.get("auc", float("nan"))),
            "monitor": cfg.monitor,
            "monitor_score": float(score),
            "lr": lr_now,
        }
        log_row(row)

        print(
            f"[val] ep={ep:03d} loss={train_loss:.4f} "
            f"F1={row['val_f1']:.3f} P={row['val_precision']:.3f} R={row['val_recall']:.3f} "
            f"FPR={row['val_fpr']:.3f} thr={row['val_thr']:.2f} "
            f"AP={row['val_ap']:.3f} AUC={row['val_auc']:.3f} lr={lr_now:g}"
        )

        # Scheduler step ONLY if the metric is finite (avoid NaN breaking training)
        metric = float(score)
        if np.isfinite(metric):
            sched.step(metric)
        else:
            print("[warn] monitor score is NaN/inf; skipping ReduceLROnPlateau.step()")

        # -------------------------
        # Best checkpoint + early stop
        # -------------------------
        if score > best_score + 1e-6:
            best_score = float(score)
            no_improve = 0

            save_ckpt(
                best_path,
                arch="tcn",
                state_dict=model.state_dict(),
                ema_state_dict=(ema.state_dict() if ema is not None else None),
                ema_decay=(float(cfg.ema_decay) if ema is not None else 0.0),
                model_cfg=model_cfg.to_dict(),
                feat_cfg=feat_cfg.to_dict(),
                data_cfg={"fps_default": cfg.fps_default},
                best={"val_best": row, "best_thr": float(row["val_thr"])},
                meta={"monitor": cfg.monitor},
            )
            print(f"[save] {best_path} (best {cfg.monitor}={best_score:.4f})")
        else:
            # Only start counting patience after min_epochs
            if ep >= int(cfg.min_epochs):
                no_improve += 1
                if cfg.patience > 0 and no_improve >= cfg.patience:
                    print(f"[early stop] patience={cfg.patience} reached at ep={ep}")
                    break

    # Final report (always written)
    report = {
        "arch": "tcn",
        "best_ckpt": best_path,
        "monitor": cfg.monitor,
        "best_score": float(best_score),
        "dataset": args.report_dataset_name,
    }
    _write_json(os.path.join(cfg.save_dir, "train_report.json"), report)
    _write_json(args.report_json, report)

    print(f"[done] ckpt={best_path}")
    print(f"[ok] wrote history: {history_path}")


if __name__ == "__main__":
    main()