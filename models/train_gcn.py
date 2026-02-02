#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/train_gcn.py

GCN trainer (single-stream or two-stream) using core/* as the single source of truth.

Supported model types
---------------------
1) Single-stream GCN:
   Input X: [B, T, V, F]
   Where F includes xy (+ motion + conf depending on FeatCfg flags).

2) Two-stream GCN:
   Joint stream  xj: [B, T, V, Fj]  (xy + optional conf)
   Motion stream xm: [B, T, V, 2]   (dxdy)
   Streams are fused ("concat" or "sum") inside TwoStreamGCN.

Important rules for correctness
-------------------------------
- If you apply augmentation to joints, you MUST recompute motion afterwards.
  We do that by setting motion=None so core/features.py recomputes motion consistently.
- Use exactly ONE class-imbalance strategy:
    either --balanced_sampler OR --pos_weight (auto/float).
  Using both tends to over-correct and hurts precision / FA/24h.

Checkpoint format
-----------------
Saved as a BUNDLE using core/ckpt.py so evaluation scripts can rebuild the model:
  {
    "arch": "gcn",
    "state_dict": ...,
    "ema_state_dict": ... (optional),
    "model_cfg": ...,
    "feat_cfg": ...,
    "data_cfg": ...,
    "best": ...,
    "meta": ...
  }
"""

from __future__ import annotations

# ============================================================
# 0) Path bootstrap (so `from core.*` works when run directly)
# ============================================================
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
from core.features import FeatCfg, build_gcn_input, read_window_npz
from core.losses import BinaryFocalLossWithLogits
from core.metrics import ap_auc, best_threshold_by_f1
from core.models import GCN, GCNConfig, TwoStreamGCN, logits_1d, pick_device


# ============================================================
# 2) Reproducibility helpers
# ============================================================
def set_seed(seed: int) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch so your training run is reproducible.

    Note:
    - True determinism can still vary depending on device backend (CUDA/MPS).
    - But this dramatically reduces randomness across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    """Write JSON safely (create parent dirs; ignore if path is None)."""
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
    Binary label smoothing:
      y_smooth = y*(1-eps) + 0.5*eps
    """
    if eps <= 0:
        return y
    return y * (1.0 - eps) + 0.5 * eps


# ============================================================
# 3) Dataset: window NPZ -> GCN tensors
# ============================================================
class WindowDatasetGCN(Dataset):
    """
    Loads window NPZ files and builds model input tensors.

    Returns for single-stream:
      X: [T, V, F]
      y: [1]

    Returns for two-stream:
      xj: [T, V, Fj]  (xy (+conf))
      xm: [T, V, 2]   (motion)
      y:  [1]
    """

    def __init__(
        self,
        root: str,
        *,
        split: str,
        feat_cfg: FeatCfg,
        fps_default: float,
        two_stream: bool,
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
        self.two_stream = bool(two_stream)

        # Only apply augmentation on training split
        self.aug_cfg = aug_cfg if self.split == "train" else None

        # Base seed used for per-sample RNG (deterministic augmentation)
        self.base_seed = int(seed)

        # Collect all NPZ files under root
        files = sorted(glob.glob(os.path.join(self.root, "**", "*.npz"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No .npz under: {self.root}")

        kept: List[str] = []
        labels: List[int] = []
        skipped = 0

        # Keep only readable windows; optionally skip unlabeled windows (y=-1)
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

        # Optional: oversample hard negatives by duplicating paths.
        # This increases the frequency of difficult negatives.
        if hard_neg_files:
            extra: List[str] = []
            for p in hard_neg_files:
                p = str(p).strip()
                if not p or not os.path.exists(p):
                    continue

                try:
                    _, _, _, _, _, meta2 = read_window_npz(p, fps_default=self.fps_default)
                    # Never include positives as "hard negatives"
                    if meta2.y == 1:
                        continue
                    extra.append(p)
                except Exception:
                    continue

            if extra:
                mult = int(hard_neg_mult) if int(hard_neg_mult) > 0 else 1
                extra_rep = extra * mult
                self.files.extend(extra_rep)
                self.labels = np.concatenate(
                    [self.labels, np.zeros(len(extra_rep), dtype=np.int64)],
                    axis=0,
                )

    def _make_rng(self, idx: int) -> np.random.Generator:
        """
        Create deterministic RNG for this sample index, even under multi-worker loading.
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
        p = self.files[idx]

        joints, motion, conf, mask, fps, meta = read_window_npz(p, fps_default=self.fps_default)

        # Apply augmentation only during training.
        # IMPORTANT: when joints change, motion must be recomputed.
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
            motion = None  # force recompute motion in build_gcn_input()

        # Build the canonical feature tensor the model sees: X [T,V,F]
        X, _ = build_gcn_input(joints, motion, conf, mask, float(fps), self.feat_cfg)

        y = float(meta.y if meta.y >= 0 else 0.0)

        # If two-stream, split X into two inputs:
        # X layout from core/features.py:
        #   [xy(2), motion(2) if use_motion, conf(1) if use_conf_channel]
        if self.two_stream:
            xy = X[..., 0:2]  # [T,V,2]

            # conf is the last channel only if use_conf_channel is True
            if self.feat_cfg.use_conf_channel:
                conf1 = X[..., -1:]  # [T,V,1]
                xj = np.concatenate([xy, conf1], axis=-1)  # [T,V,3]
            else:
                xj = xy  # [T,V,2]

            # motion channels exist only if use_motion True; otherwise use zeros
            if self.feat_cfg.use_motion and X.shape[-1] >= 4:
                xm = X[..., 2:4]  # [T,V,2]
            else:
                xm = np.zeros_like(xy, dtype=np.float32)

            return (
                torch.from_numpy(xj).float(),
                torch.from_numpy(xm).float(),
                torch.tensor([y], dtype=torch.float32),
            )

        return torch.from_numpy(X).float(), torch.tensor([y], dtype=torch.float32)


# ============================================================
# 4) Class imbalance helpers
# ============================================================
def compute_pos_weight(y: np.ndarray) -> Optional[torch.Tensor]:
    """
    pos_weight = (#neg / #pos) as PyTorch expects (shape [1]).

    Used by BCEWithLogitsLoss(pos_weight=...).
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
    Balanced sampling: positives and negatives appear with similar frequency.

    This is an alternative to pos_weight.
    """
    y = np.asarray(y).reshape(-1)
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    w_pos = 1.0 / pos
    w_neg = 1.0 / neg
    weights = np.where(y == 1, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )


# ============================================================
# 5) Validation: collect probabilities on a loader
# ============================================================
@torch.no_grad()
def collect_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    two_stream: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model on loader and return:
      probs: sigmoid(logits) -> [N]
      y_true: int labels -> [N]
    """
    model.eval()
    ps: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for batch in loader:
        if two_stream:
            xj, xm, yb = batch
            xj = xj.to(device)
            xm = xm.to(device)
            logits = logits_1d(model(xj, xm))
        else:
            xb, yb = batch
            xb = xb.to(device)
            logits = logits_1d(model(xb))

        p = torch.sigmoid(logits).detach().cpu().numpy()
        y = yb.detach().cpu().numpy().reshape(-1).astype(int)

        ps.append(p)
        ys.append(y)

    return (np.concatenate(ps) if ps else np.array([])), (np.concatenate(ys) if ys else np.array([]))


# ============================================================
# 6) Training config (FIXED indentation)
# ============================================================
@dataclass
class TrainCfg:
    # Data paths
    train_dir: str
    val_dir: str
    save_dir: str
    test_dir: Optional[str] = None

    # Resume + hard negatives
    resume: Optional[str] = None
    hard_neg_list: Optional[str] = None
    hard_neg_mult: int = 0

    # Optimization
    epochs: int = 60
    batch: int = 64
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Repro + early stopping
    seed: int = 33724876
    patience: int = 12
    min_epochs: int = 0  # early stopping only counts after this epoch

    # Scheduler (ReduceLROnPlateau)
    lr_plateau_patience: int = 5
    lr_plateau_factor: float = 0.5
    lr_plateau_min_lr: float = 1e-6

    # Loader
    num_workers: int = 0
    fps_default: float = 30.0

    # Monitoring metric for "best"
    monitor: str = "ap"  # "ap" or "f1"

    # Imbalance strategy (use ONLY ONE)
    pos_weight: str = "auto"  # auto | none | <float>
    balanced_sampler: bool = False

    # Loss selection
    loss: str = "focal"  # focal | bce
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

    # EMA (0 disables)
    ema_decay: float = 0.999

    # Augmentation (GCN benefits from realistic missingness)
    mask_joint_p: float = 0.15
    mask_frame_p: float = 0.10
    aug_hflip_p: float = 0.50
    aug_jitter_std: float = 0.008
    aug_jitter_conf_scaled: int = 1
    aug_occ_p: float = 0.30
    aug_occ_min_len: int = 3
    aug_occ_max_len: int = 10
    aug_time_shift: int = 1

    # Model architecture (GCN)
    two_stream: bool = False
    fuse: str = "concat"  # concat | sum
    gcn_hidden: int = 96
    tcn_hidden: int = 192
    dropout: float = 0.35
    use_se: bool = True

    # Feature flags (must match training/eval)
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

    # ---- Paths ----
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--test_dir", default=None)
    ap.add_argument("--save_dir", required=True)

    ap.add_argument("--report_json", default=None)
    ap.add_argument("--report_dataset_name", default=None)

    # ---- Resume & hard negatives ----
    ap.add_argument("--resume", default=None)
    ap.add_argument("--hard_neg_list", default=None)
    ap.add_argument("--hard_neg_mult", type=int, default=0)

    # ---- Optimization ----
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # ---- Repro + early stop ----
    ap.add_argument("--seed", type=int, default=33724876)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min_epochs", type=int, default=0)

    # ---- Scheduler ----
    ap.add_argument("--lr_plateau_patience", type=int, default=5)
    ap.add_argument("--lr_plateau_factor", type=float, default=0.5)
    ap.add_argument("--lr_plateau_min_lr", type=float, default=1e-6)

    # ---- Loader ----
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--fps_default", type=float, default=30.0)

    # ---- Monitoring ----
    ap.add_argument("--monitor", type=lambda s: str(s).strip().lower(), choices=["f1", "ap"], default="ap")

    # ---- Imbalance ----
    ap.add_argument("--pos_weight", default="auto")
    ap.add_argument("--balanced_sampler", action="store_true")

    # ---- Loss ----
    ap.add_argument("--loss", type=lambda s: str(s).strip().lower(), choices=["focal", "bce"], default="focal")
    ap.add_argument("--focal_alpha", type=float, default=0.25)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)

    # ---- EMA ----
    ap.add_argument("--ema_decay", type=float, default=0.999)

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

    # ---- GCN architecture ----
    ap.add_argument("--two_stream", action="store_true")
    ap.add_argument("--fuse", choices=["concat", "sum"], default="concat")
    ap.add_argument("--gcn_hidden", type=int, default=96)
    ap.add_argument("--tcn_hidden", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--use_se", type=int, default=1)

    # ---- Feature flags ----
    ap.add_argument("--center", choices=["pelvis", "none"], default="pelvis")
    ap.add_argument("--use_motion", type=int, default=1)
    ap.add_argument("--use_conf_channel", type=int, default=1)
    ap.add_argument("--motion_scale_by_fps", type=int, default=1)
    ap.add_argument("--conf_gate", type=float, default=0.20)
    ap.add_argument("--use_precomputed_mask", type=int, default=1)

    args = ap.parse_args()

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
        two_stream=bool(args.two_stream),
        fuse=str(args.fuse),
        gcn_hidden=int(args.gcn_hidden),
        tcn_hidden=int(args.tcn_hidden),
        dropout=float(args.dropout),
        use_se=bool(int(args.use_se)),
        center=str(args.center),
        use_motion=int(args.use_motion),
        use_conf_channel=int(args.use_conf_channel),
        motion_scale_by_fps=int(args.motion_scale_by_fps),
        conf_gate=float(args.conf_gate),
        use_precomputed_mask=int(args.use_precomputed_mask),
    )

    # Enforce: only one imbalance strategy
    if cfg.balanced_sampler and str(cfg.pos_weight).strip().lower() not in {"none", "0", "0.0"}:
        if str(cfg.pos_weight).strip().lower() == "auto":
            print("[warn] --balanced_sampler + --pos_weight=auto both set. Using balanced_sampler only.")
            cfg.pos_weight = "none"

    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    _write_json(os.path.join(cfg.save_dir, "train_config.json"), asdict(cfg))

    device = pick_device()
    print(f"[info] device: {device.type}")

    # ------------------------------------------------------------
    # Resume: load bundle and lock configs for compatibility
    # ------------------------------------------------------------
    bundle = None
    if cfg.resume:
        bundle = load_ckpt(cfg.resume, map_location="cpu")
        arch0, model_cfg0, feat_cfg0, data_cfg0 = get_cfg(bundle)

        if arch0 and arch0 != "gcn":
            raise SystemExit(f"[err] resume arch mismatch: expected 'gcn', got {arch0}")

        # Override feature flags from checkpoint so inputs match training
        cfg.center = str(feat_cfg0.get("center", cfg.center))
        cfg.use_motion = int(feat_cfg0.get("use_motion", cfg.use_motion))
        cfg.use_conf_channel = int(feat_cfg0.get("use_conf_channel", cfg.use_conf_channel))
        cfg.motion_scale_by_fps = int(feat_cfg0.get("motion_scale_by_fps", cfg.motion_scale_by_fps))
        cfg.conf_gate = float(feat_cfg0.get("conf_gate", cfg.conf_gate))
        cfg.use_precomputed_mask = int(feat_cfg0.get("use_precomputed_mask", cfg.use_precomputed_mask))

        # Override architecture from checkpoint so weights load strictly
        cfg.two_stream = bool(model_cfg0.get("two_stream", cfg.two_stream))
        cfg.fuse = str(model_cfg0.get("fuse", cfg.fuse))
        cfg.gcn_hidden = int(model_cfg0.get("gcn_hidden", cfg.gcn_hidden))
        cfg.tcn_hidden = int(model_cfg0.get("tcn_hidden", cfg.tcn_hidden))
        cfg.dropout = float(model_cfg0.get("dropout", cfg.dropout))
        cfg.use_se = bool(model_cfg0.get("use_se", cfg.use_se))

        cfg.fps_default = float(data_cfg0.get("fps_default", cfg.fps_default))

        print(f"[info] resume: loaded cfg from {cfg.resume}")

    feat_cfg = FeatCfg(
        center=cfg.center,
        use_motion=bool(cfg.use_motion),
        use_conf_channel=bool(cfg.use_conf_channel),
        motion_scale_by_fps=bool(cfg.motion_scale_by_fps),
        conf_gate=float(cfg.conf_gate),
        use_precomputed_mask=bool(cfg.use_precomputed_mask),
    )

    # Augmentation config (train split only)
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

    # Load hard negative list (optional)
    hard_neg_files: Optional[List[str]] = None
    if cfg.hard_neg_list:
        try:
            with open(cfg.hard_neg_list, "r", encoding="utf-8") as f:
                hard_neg_files = [ln.strip() for ln in f.readlines() if ln.strip()]
        except Exception:
            hard_neg_files = None

    # Datasets
    train_ds = WindowDatasetGCN(
        cfg.train_dir,
        split="train",
        feat_cfg=feat_cfg,
        fps_default=cfg.fps_default,
        two_stream=cfg.two_stream,
        skip_unlabeled=True,
        hard_neg_files=hard_neg_files,
        hard_neg_mult=int(cfg.hard_neg_mult),
        aug_cfg=aug_cfg,
        seed=cfg.seed,
    )
    val_ds = WindowDatasetGCN(
        cfg.val_dir,
        split="val",
        feat_cfg=feat_cfg,
        fps_default=cfg.fps_default,
        two_stream=cfg.two_stream,
        skip_unlabeled=True,
        aug_cfg=None,
        seed=cfg.seed,
    )

    # Infer input dims from one sample
    if cfg.two_stream:
        xj0, xm0, _ = train_ds[0]
        T, V, Fj = xj0.shape
        _, _, Fm = xm0.shape
        print(f"[info] window shape: T={T}, V={V}, Fj={Fj}, Fm={Fm} (two-stream)")
    else:
        x0, _ = train_ds[0]
        T, V, F = x0.shape
        print(f"[info] window shape: T={T}, V={V}, F={F} (single-stream)")

    # Build model
    if cfg.two_stream:
        model_cfg = GCNConfig(
            num_joints=int(V),
            gcn_hidden=int(cfg.gcn_hidden),
            tcn_hidden=int(cfg.tcn_hidden),
            dropout=float(cfg.dropout),
            use_se=bool(cfg.use_se),
            two_stream=True,
            fuse=str(cfg.fuse),
        )
        model = TwoStreamGCN(
            num_joints=int(V),
            in_feats_j=int(Fj),
            in_feats_m=int(Fm),
            gcn_hidden=model_cfg.gcn_hidden,
            tcn_hidden=model_cfg.tcn_hidden,
            dropout=model_cfg.dropout,
            use_se=model_cfg.use_se,
            fuse=model_cfg.fuse,
        ).to(device)
    else:
        model_cfg = GCNConfig(
            num_joints=int(V),
            gcn_hidden=int(cfg.gcn_hidden),
            tcn_hidden=int(cfg.tcn_hidden),
            dropout=float(cfg.dropout),
            use_se=bool(cfg.use_se),
            two_stream=False,
            fuse="concat",
        )
        model = GCN(
            num_joints=int(V),
            in_feats=int(F),
            gcn_hidden=model_cfg.gcn_hidden,
            tcn_hidden=model_cfg.tcn_hidden,
            dropout=model_cfg.dropout,
            use_se=model_cfg.use_se,
        ).to(device)

    if bundle is not None:
        model.load_state_dict(bundle["state_dict"], strict=True)
        print(f"[info] resumed weights from: {cfg.resume}")

    # EMA
    ema = None
    if float(cfg.ema_decay) > 0:
        ema = EMA(model, decay=float(cfg.ema_decay))
        ema.update(model)

    # Loss setup
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
        # Focal loss is usually a good default for imbalance + hard examples.
        criterion = BinaryFocalLossWithLogits(
            alpha=float(cfg.focal_alpha),
            gamma=float(cfg.focal_gamma),
            pos_weight=(pos_w.to(device) if pos_w is not None else None),
            reduction="mean",
        )
    else:
        # BCEWithLogitsLoss expects logits.
        if pos_w is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
        else:
            criterion = nn.BCEWithLogitsLoss()

    # Optimizer + scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=float(cfg.lr_plateau_factor),
        patience=int(cfg.lr_plateau_patience),
        min_lr=float(cfg.lr_plateau_min_lr),
    )

    # DataLoaders
    pin = torch.cuda.is_available()

    if cfg.balanced_sampler:
        sampler = make_balanced_sampler(train_ds.labels)
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg.batch),
            sampler=sampler,
            shuffle=False,
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

    # Bookkeeping
    best_score = -1.0
    best_path = os.path.join(cfg.save_dir, "best.pt")
    history_path = os.path.join(cfg.save_dir, "history.jsonl")

    if os.path.exists(history_path):
        os.remove(history_path)

    def log_row(row: Dict[str, Any]) -> None:
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    no_improve = 0

    # ------------------------------------------------------------
    # Training epochs
    # ------------------------------------------------------------
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        pbar = tqdm(train_loader, desc=f"train GCN ep{ep}", leave=False)
        for batch in pbar:
            if cfg.two_stream:
                xj, xm, yb = batch
                xj = xj.to(device)
                xm = xm.to(device)
            else:
                xb, yb = batch
                xb = xb.to(device)

            yb = yb.to(device).view(-1)
            yb = _apply_label_smoothing(yb, float(cfg.label_smoothing))

            if cfg.two_stream:
                logits = logits_1d(model(xj, xm))
            else:
                logits = logits_1d(model(xb))

            loss = criterion(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))

            opt.step()

            if ema is not None:
                ema.update(model)

            bs = int(yb.shape[0])
            running_loss += float(loss.detach().cpu()) * bs
            n_seen += bs
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        train_loss = running_loss / max(1, n_seen)

        # -------------------------
        # Validation
        # -------------------------
        if ema is not None:
            with ema.average_parameters(model):
                probs, y_true = collect_probs(model, val_loader, device, two_stream=cfg.two_stream)
        else:
            probs, y_true = collect_probs(model, val_loader, device, two_stream=cfg.two_stream)

        if probs.size == 0:
            print("[val] empty validation set?")
            continue

        sweep_best = best_threshold_by_f1(probs, y_true, thr_min=0.05, thr_max=0.95, thr_step=0.01)
        extras = ap_auc(probs, y_true)

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

        # ReduceLROnPlateau step only if the metric is finite
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

            # Store explicit dims in model_cfg so rebuilding is always correct
            model_cfg_dict = model_cfg.to_dict()
            if cfg.two_stream:
                model_cfg_dict["in_feats_j"] = int(Fj)
                model_cfg_dict["in_feats_m"] = int(Fm)
            else:
                model_cfg_dict["in_feats"] = int(F)

            save_ckpt(
                best_path,
                arch="gcn",
                state_dict=model.state_dict(),
                ema_state_dict=(ema.state_dict() if ema is not None else None),
                ema_decay=(float(cfg.ema_decay) if ema is not None else 0.0),
                model_cfg=model_cfg_dict,
                feat_cfg=feat_cfg.to_dict(),
                data_cfg={"fps_default": cfg.fps_default},
                best={"val_best": row, "best_thr": float(row["val_thr"])},
                meta={"monitor": cfg.monitor},
            )
            print(f"[save] {best_path} (best {cfg.monitor}={best_score:.4f})")
        else:
            if ep >= int(cfg.min_epochs):
                no_improve += 1
                if cfg.patience > 0 and no_improve >= cfg.patience:
                    print(f"[early stop] patience={cfg.patience} reached at ep={ep}")
                    break

    report = {
        "arch": "gcn",
        "best_ckpt": best_path,
        "monitor": cfg.monitor,
        "best_score": float(best_score),
        "dataset": args.report_dataset_name,
        "two_stream": bool(cfg.two_stream),
    }
    _write_json(os.path.join(cfg.save_dir, "train_report.json"), report)
    _write_json(args.report_json, report)

    print(f"[done] ckpt={best_path}")
    print(f"[ok] wrote history: {history_path}")


if __name__ == "__main__":
    main()