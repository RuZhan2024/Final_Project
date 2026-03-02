#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""models/train_gcn.py

GCN trainer aligned with the current pipeline (core/*).

Fixes / upgrades:
- Uses core.features.read_window_npz + build_canonical_input (matches window NPZ schema).
- Actually applies CLI knobs that your Makefile expects:
  * --weight_decay
  * --label_smoothing
  * --num_workers
  * --mask_joint_p / --mask_frame_p (mask augmentation)
  * --lr_plateau_* (optional; defaults match older behavior)
  * --min_epochs (don’t early-stop too early)
- Keeps hard-negative mining support (--resume + --hard_neg_list + --hard_neg_mult).
- Keeps imbalance strategies: choose ONE of --balanced_sampler OR --pos_weight (unless loss=focal).
"""

from __future__ import annotations

# ---- bootstrap: allow running as a script from repo root ----
def _bootstrap_project_root():
    import sys
    from pathlib import Path
    here = Path(__file__).resolve()
    root = here.parents[1]  # repo root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

_bootstrap_project_root()
# -------------------------------------------------------------

import argparse
import json
import os
import random
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from fall_detection.core.ckpt import get_cfg, load_ckpt, save_ckpt
from fall_detection.core.features import FeatCfg, build_canonical_input, read_window_npz, split_gcn_two_stream
from fall_detection.core.ema import EMA
from fall_detection.core.losses import FocalLossWithLogits
from fall_detection.core.metrics import ap_auc, best_threshold_by_f1
from fall_detection.core.models import GCNConfig, build_model, pick_device


# -------------------------
# Utilities
# -------------------------
def _to_f32(x, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def logits_1d(out: torch.Tensor) -> torch.Tensor:
    if out.ndim == 2 and out.shape[1] == 1:
        return out[:, 0]
    if out.ndim == 1:
        return out
    return out.view(out.shape[0], -1)[:, 0]


def compute_pos_weight(labels01: np.ndarray) -> torch.Tensor:
    y = np.asarray(labels01).astype(int).reshape(-1)
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    return torch.tensor([neg / pos], dtype=torch.float32)


def make_balanced_sampler(y01: np.ndarray) -> WeightedRandomSampler:
    y = np.asarray(y01).reshape(-1)
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    w_pos = 1.0 / pos
    w_neg = 1.0 / neg
    w = np.where(y == 1, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(w), replacement=True)


def augment_mask(mask: np.ndarray, rng: np.random.Generator, mask_joint_p: float, mask_frame_p: float) -> np.ndarray:
    m = np.asarray(mask).copy().astype(bool)
    T, V = m.shape
    if mask_joint_p > 0:
        drop_j = rng.random(V) < float(mask_joint_p)
        if drop_j.any():
            m[:, drop_j] = False
    if mask_frame_p > 0:
        drop_t = rng.random(T) < float(mask_frame_p)
        if drop_t.any():
            m[drop_t, :] = False
    if not m.any():
        m[int(rng.integers(0, T)), int(rng.integers(0, V))] = True
    return m


def list_npz_files(root: str) -> List[str]:
    import glob
    pat = os.path.join(root, "**", "*.npz")
    files = glob.glob(pat, recursive=True)
    files.sort()
    return files


def _validate_hard_neg_paths(
    paths: List[str],
    *,
    train_dir: str,
    allow_nontrain: bool,
) -> None:
    """Guard against val/test leakage through hard-negative lists."""
    if allow_nontrain or not paths:
        return

    train_root = Path(train_dir).expanduser().resolve()
    bad_valtest: List[str] = []
    bad_unknown: List[str] = []

    for raw in paths:
        rp = Path(raw).expanduser().resolve()
        parts_l = {p.lower() for p in rp.parts}
        if "val" in parts_l or "test" in parts_l:
            bad_valtest.append(str(rp))
            continue
        in_train_root = False
        try:
            rp.relative_to(train_root)
            in_train_root = True
        except ValueError:
            in_train_root = False
        has_train_component = "train" in parts_l
        if not (in_train_root or has_train_component):
            bad_unknown.append(str(rp))

    if bad_valtest or bad_unknown:
        lines = [
            "hard_neg_list safety guard rejected candidate paths.",
            "By default, hard negatives must come from train split paths.",
            "Use --allow_hard_neg_nontrain 1 only if you explicitly accept leakage risk.",
        ]
        if bad_valtest:
            lines.append(f"val/test-like paths (showing up to 5): {bad_valtest[:5]}")
        if bad_unknown:
            lines.append(f"non-train paths (showing up to 5): {bad_unknown[:5]}")
        raise ValueError(" ".join(lines))


class WindowDatasetGCN(Dataset):
    def __init__(
        self,
        root: str,
        *,
        split: str,
        feat_cfg: FeatCfg,
        fps_default: float,
        skip_unlabeled: bool,
        two_stream: bool,
        mask_joint_p: float,
        mask_frame_p: float,
        seed: int,
        extra_neg_files: Optional[List[str]] = None,
        extra_neg_mult: int = 1,
    ) -> None:
        self.root = str(root)
        self.split = str(split)
        self.feat_cfg = feat_cfg
        self.fps_default = float(fps_default)
        self.skip_unlabeled = bool(skip_unlabeled)
        self.two_stream = bool(two_stream)
        self.mask_joint_p = float(mask_joint_p)
        self.mask_frame_p = float(mask_frame_p)

        files = list_npz_files(self.root)
        self.files: List[str] = []
        self.labels01: List[int] = []

        fail = 0
        examples: List[str] = []

        # main dataset
        for fp in files:
            try:
                _, _, _, _, _, meta = read_window_npz(fp, fps_default=self.fps_default)
            except Exception as e:
                fail += 1
                if len(examples) < 5:
                    examples.append(f"{fp}: {type(e).__name__}: {e}")
                continue
            y = int(meta.y)
            if self.skip_unlabeled and y < 0:
                continue
            self.files.append(fp)
            self.labels01.append(1 if y == 1 else 0)

        # extra hard negatives
        if extra_neg_files:
            mult = max(1, int(extra_neg_mult))
            extra = list(extra_neg_files) * mult
            for fp in extra:
                fp = fp.strip()
                if not fp:
                    continue
                try:
                    _, _, _, _, _, meta = read_window_npz(fp, fps_default=self.fps_default)
                except Exception as e:
                    fail += 1
                    if len(examples) < 5:
                        examples.append(f"{fp}: {type(e).__name__}: {e}")
                    continue
                y = int(meta.y)
                if y == 1:
                    continue
                self.files.append(fp)
                self.labels01.append(0)


        if fail:
            print(f"[warn] skipped {fail} unreadable windows under: {self.root}")
            for ex in examples:
                print(f"[warn]   example: {ex}")

        if len(self.files) == 0:
            raise RuntimeError(
                f"[err] no readable windows under: {self.root}. "
                f"found={len(files)} failed_reads={fail}. "
                "Check make_windows.py output + key/dtype consistency."
            )

        self.labels01 = np.asarray(self.labels01, dtype=np.int64)
        base = int(seed) + (11 if split == "train" else 22)
        self.rng = np.random.default_rng(base)
        self._missing_warned: set[str] = set()

    def __len__(self) -> int:
        return len(self.files)

    def _read_window_with_fallback(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Any]:
        n = len(self.files)
        if n <= 0:
            raise RuntimeError("[err] empty dataset")
        max_probe = min(n, 32)
        for off in range(max_probe):
            j = (int(i) + off) % n
            fp = self.files[j]
            try:
                return read_window_npz(fp, fps_default=self.fps_default)
            except FileNotFoundError:
                if fp not in self._missing_warned:
                    self._missing_warned.add(fp)
                    print(f"[warn] missing window file during training; skipping: {fp}")
                continue
        raise FileNotFoundError(
            f"[err] unable to read any nearby window files around index={i} "
            f"(probed={max_probe}). Check for concurrent cleanup under: {self.root}"
        )

    def __getitem__(self, i: int):
        joints, motion, conf, mask, fps, meta = self._read_window_with_fallback(i)

        X, mask_used = build_canonical_input(
            joints_xy=joints,
            motion_xy=motion,
            conf=conf,
            mask=mask,
            fps=fps,
            feat_cfg=self.feat_cfg,
        )

        # Random mask augmentation (train only)
        if self.split == "train" and (self.mask_joint_p > 0 or self.mask_frame_p > 0):
            m_aug = augment_mask(mask_used, self.rng, self.mask_joint_p, self.mask_frame_p)
            X = X * m_aug[..., None]

        y = 1 if int(meta.y) == 1 else 0

        if not self.two_stream:
            # NumPy -> Torch
            return (
                torch.as_tensor(X, dtype=torch.float32),
                torch.as_tensor([y], dtype=torch.float32),
            )

        if not self.feat_cfg.use_motion:
            raise RuntimeError("two_stream requires feat_cfg.use_motion=1")

        xj, xm = split_gcn_two_stream(X, self.feat_cfg)

        # NumPy -> Torch
        return (
            torch.as_tensor(xj, dtype=torch.float32),
            torch.as_tensor(xm, dtype=torch.float32),
            torch.as_tensor([y], dtype=torch.float32),
        )

def collect_probs(
    model: nn.Module, loader: DataLoader, device: torch.device, two_stream: bool
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ps: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if two_stream:
                xj, xm, yb = batch
                xj = _to_f32(xj, device)
                xm = _to_f32(xm, device)
                logits = logits_1d(model(xj, xm))
                bsz = int(xj.shape[0])
            else:
                xb, yb = batch
                xb = _to_f32(xb, device)
                logits = logits_1d(model(xb))
                bsz = int(xb.shape[0])

            yb = _to_f32(yb, device).view(-1)

            p = torch.sigmoid(logits).detach().cpu().numpy()
            y = yb.detach().cpu().numpy().reshape(-1)
            ps.append(p)
            ys.append(y)
    return np.concatenate(ps, axis=0), np.concatenate(ys, axis=0)

def compute_loss_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    two_stream: bool = False,
    ema: Optional[EMA] = None,
) -> float:
    """Compute mean loss on a loader with model in eval mode (no grad)."""
    model.eval()
    losses: List[float] = []
    counts: List[int] = []

    ctx = ema.use(model) if ema is not None else nullcontext()
    with ctx, torch.no_grad():
        for batch in loader:
            if two_stream:
                xj, xm, yb = batch
                xj = _to_f32(xj, device)  # numpy OR torch -> torch.float32 on device
                xm = _to_f32(xm, device)
                logits = logits_1d(model(xj, xm))
                bsz = int(xj.shape[0])
            else:
                xb, yb = batch
                xb = _to_f32(xb, device)
                logits = logits_1d(model(xb))
                bsz = int(xb.shape[0])

            yb = _to_f32(yb, device).view(-1)
            loss = criterion(logits, yb).detach()
            losses.append(float(loss.cpu()) * bsz)
            counts.append(bsz)

    return float(sum(losses) / max(1, sum(counts)))


@dataclass
class TrainCfg:
    train_dir: str
    val_dir: str
    save_dir: str
    test_dir: Optional[str] = None

    resume: Optional[str] = None
    hard_neg_list: Optional[str] = None
    hard_neg_mult: int = 1
    allow_hard_neg_nontrain: int = 0

    epochs: int = 200
    min_epochs: int = 0
    batch: int = 128
    lr: float = 1e-3
    seed: int = 33724876
    grad_clip: float = 1.0

    lr_plateau_patience: int = 3
    lr_plateau_factor: float = 0.5
    lr_plateau_min_lr: float = 1e-6

    weight_decay: float = 1e-4
    label_smoothing: float = 0.0

    patience: int = 30
    fps_default: float = 30.0

    loss: str = "bce"           # "bce" | "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    monitor: str = "f1"         # "f1" | "ap"
    pos_weight: str = "auto"    # "auto" | "none" | float-string
    balanced_sampler: bool = False

    mask_joint_p: float = 0.0
    mask_frame_p: float = 0.0

    # --- model (CTR-GCN-ish config used below) ---
    hidden: int = 128
    dropout: float = 0.30
    num_blocks: int = 6
    temporal_kernel: int = 9
    base_channels: int = 48
    two_stream: int = 1
    fuse: str = "concat"        # "concat" | "sum" | "joint_only" | "motion_only"
    use_adaptive_adj: int = 0
    adaptive_adj_embed: int = 16
    use_ctr_gcn_lite: int = 0
    ctr_rank: int = 4
    stream_drop_joint_p: float = 0.0
    stream_drop_motion_p: float = 0.0

    # --- feature knobs (mapped into core.features.FeatCfg) ---
    center: str = "pelvis"      # keep for parity with core.features.FeatCfg
    use_motion: int = 1
    use_conf: int = 1
    use_bone: int = 0
    use_bonelen: int = 0
    motion_scale_by_fps: int = 1
    conf_gate: float = 0.20
    use_precomputed_mask: int = 1

    # These are currently not used by core.features.FeatCfg (kept only so CLI doesn't break)
    use_angles: int = 0
    normalize: str = "torso"
    include_centered: int = 1
    include_abs: int = 0
    include_vel: int = 1

    fixed_thr: float = 0.5
    thr_min: float = 0.05
    thr_max: float = 0.95
    thr_step: float = 0.01

    use_ema: int = 0
    ema_decay: float = 0.995
    save_tag: str = ""

    num_workers: int = 0

def build_feat_cfg(cfg: TrainCfg) -> FeatCfg:
    """
    Map train_gcn CLI -> core.features.FeatCfg.

    NOTE: core/features.py currently supports:
      center, use_motion, use_bone, use_bone_length, use_conf_channel,
      motion_scale_by_fps, conf_gate, use_precomputed_mask

    Flags like --normalize / --include_abs / --include_vel are currently
    not consumed by core.features (safe but no-op).
    """
    return FeatCfg(
        center=str(getattr(cfg, "center", "pelvis")),
        use_motion=bool(int(getattr(cfg, "use_motion", 1))),
        use_bone=bool(int(getattr(cfg, "use_bone", 0))),
        use_bone_length=bool(int(getattr(cfg, "use_bonelen", 0))),
        use_conf_channel=bool(int(getattr(cfg, "use_conf", 1))),
        motion_scale_by_fps=bool(int(getattr(cfg, "motion_scale_by_fps", 1))),
        conf_gate=float(getattr(cfg, "conf_gate", 0.20)),
        use_precomputed_mask=bool(int(getattr(cfg, "use_precomputed_mask", 1))),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train GCN on window NPZs.")

    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--test_dir", default=None)
    ap.add_argument("--save_dir", required=True)

    ap.add_argument("--resume", default=None)
    ap.add_argument("--hard_neg_list", default=None)
    ap.add_argument("--hard_neg_mult", type=int, default=1)
    ap.add_argument(
        "--allow_hard_neg_nontrain",
        type=int,
        default=0,
        help="Set to 1 to disable hard-negative split safety guard (risk: leakage from val/test).",
    )

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--min_epochs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=33724876)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--lr_plateau_patience", type=int, default=3)
    ap.add_argument("--lr_plateau_factor", type=float, default=0.5)
    ap.add_argument("--lr_plateau_min_lr", type=float, default=1e-6)

    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.0)

    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--fps_default", type=float, default=30.0)

    ap.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
    ap.add_argument("--focal_alpha", type=float, default=0.25)
    ap.add_argument("--focal_gamma", type=float, default=2.0)

    ap.add_argument("--monitor", type=str, default="f1", choices=["f1", "ap"])
    ap.add_argument("--pos_weight", type=str, default="auto")
    ap.add_argument("--balanced_sampler", action="store_true")

    ap.add_argument("--mask_joint_p", type=float, default=0.05)
    ap.add_argument("--mask_frame_p", type=float, default=0.05)

    # NOTE: core/models.py GCNConfig expects gcn_hidden + tcn_hidden.
    # We keep --hidden as a convenient alias for gcn_hidden and derive tcn_hidden=2*hidden.
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.30)

    # These exist in your CLI but are not used by the current core GCN implementation.
    ap.add_argument("--num_blocks", type=int, default=6)
    ap.add_argument("--temporal_kernel", type=int, default=9)
    ap.add_argument("--base_channels", type=int, default=48)

    ap.add_argument("--two_stream", type=int, default=1)
    ap.add_argument("--fuse", type=str, default="concat", choices=["concat", "sum", "joint_only", "motion_only"])
    ap.add_argument("--use_adaptive_adj", type=int, default=0)
    ap.add_argument("--adaptive_adj_embed", type=int, default=16)
    ap.add_argument("--use_ctr_gcn_lite", type=int, default=0)
    ap.add_argument("--ctr_rank", type=int, default=4)
    ap.add_argument("--stream_drop_joint_p", type=float, default=0.0)
    ap.add_argument("--stream_drop_motion_p", type=float, default=0.0)

    ap.add_argument("--use_conf", type=int, default=1)
    ap.add_argument("--use_motion", type=int, default=1)
    ap.add_argument("--use_bone", type=int, default=0)
    ap.add_argument("--use_bonelen", type=int, default=0)
    ap.add_argument("--use_angles", type=int, default=0)

    ap.add_argument("--normalize", type=str, default="torso", choices=["none", "torso"])
    ap.add_argument("--include_centered", type=int, default=1)
    ap.add_argument("--include_abs", type=int, default=0)
    ap.add_argument("--include_vel", type=int, default=1)

    ap.add_argument("--fixed_thr", type=float, default=0.5)
    ap.add_argument("--thr_min", type=float, default=0.05)
    ap.add_argument("--thr_max", type=float, default=0.95)
    ap.add_argument("--thr_step", type=float, default=0.01)

    ap.add_argument("--use_ema", type=int, default=0)
    ap.add_argument("--ema_decay", type=float, default=0.995)
    ap.add_argument("--save_tag", type=str, default="")

    ap.add_argument("--num_workers", type=int, default=0)

    args = ap.parse_args()
    cfg = TrainCfg(**vars(args))

    os.makedirs(cfg.save_dir, exist_ok=True)

    set_seed(cfg.seed)
    device = pick_device()
    feat_cfg = build_feat_cfg(cfg)

    # On macOS MPS, keep workers at 0 to avoid spawn instability
    nw = int(cfg.num_workers)
    if device.type == "mps":
        nw = 0

    # Optional extra hard negatives (list of window npz paths)
    extra_neg_files: Optional[List[str]] = None
    if cfg.hard_neg_list:
        try:
            with open(cfg.hard_neg_list, "r", encoding="utf-8", errors="ignore") as f:
                extra_neg_files = [ln.strip() for ln in f if ln.strip()]
            _validate_hard_neg_paths(
                extra_neg_files,
                train_dir=cfg.train_dir,
                allow_nontrain=bool(int(cfg.allow_hard_neg_nontrain)),
            )
            print(f"[hard_neg] loaded {len(extra_neg_files)} paths from: {cfg.hard_neg_list}")
        except Exception as e:
            print(f"[warn] cannot read hard_neg_list={cfg.hard_neg_list}: {type(e).__name__}: {e}")
            extra_neg_files = None

    # ✅ FIX: use the real WindowDatasetGCN signature (no skip_on_error)
    train_ds = WindowDatasetGCN(
        cfg.train_dir,
        split="train",
        feat_cfg=feat_cfg,
        fps_default=cfg.fps_default,
        skip_unlabeled=True,
        two_stream=bool(cfg.two_stream),
        mask_joint_p=cfg.mask_joint_p,
        mask_frame_p=cfg.mask_frame_p,
        seed=cfg.seed,
        extra_neg_files=extra_neg_files,
        extra_neg_mult=int(cfg.hard_neg_mult),
    )
    val_ds = WindowDatasetGCN(
        cfg.val_dir,
        split="val",
        feat_cfg=feat_cfg,
        fps_default=cfg.fps_default,
        skip_unlabeled=True,
        two_stream=bool(cfg.two_stream),
        mask_joint_p=0.0,
        mask_frame_p=0.0,
        seed=cfg.seed,
    )

    if len(train_ds) == 0:
        raise RuntimeError(f"[err] empty train dataset: {cfg.train_dir}")
    if len(val_ds) == 0:
        raise RuntimeError(f"[err] empty val dataset: {cfg.val_dir}")

    # Infer dims from first item
    if bool(cfg.two_stream):
        xj0, xm0, _ = train_ds[0]  # [T,V,C]
        V = int(xj0.shape[1])
        in_feats_j = int(xj0.shape[2])
        in_feats_m = int(xm0.shape[2])
        in_feats = None
    else:
        xb0, _ = train_ds[0]       # [T,V,C]
        V = int(xb0.shape[1])
        in_feats = int(xb0.shape[2])
        in_feats_j = None
        in_feats_m = None

    # ✅ FIX: core/models.py expects gcn_hidden + tcn_hidden (+ use_se)
    gcn_hidden = int(getattr(cfg, "hidden", 96))
    tcn_hidden = int(getattr(cfg, "tcn_hidden", 2 * gcn_hidden))
    use_se = bool(int(getattr(cfg, "use_se", 1)))

    model_cfg_save: Dict[str, Any] = {
        "num_joints": int(V),
        "gcn_hidden": int(gcn_hidden),
        "tcn_hidden": int(tcn_hidden),
        "dropout": float(cfg.dropout),
        "use_se": bool(use_se),
        "two_stream": bool(cfg.two_stream),
        "fuse": str(cfg.fuse),
        "use_adaptive_adj": bool(int(cfg.use_adaptive_adj)),
        "adaptive_adj_embed": int(cfg.adaptive_adj_embed),
        "use_ctr_gcn_lite": bool(int(cfg.use_ctr_gcn_lite)),
        "ctr_rank": int(cfg.ctr_rank),
        "stream_drop_joint_p": float(cfg.stream_drop_joint_p),
        "stream_drop_motion_p": float(cfg.stream_drop_motion_p),
    }
    if bool(cfg.two_stream):
        model_cfg_save["in_feats_j"] = int(in_feats_j)  # type: ignore[arg-type]
        model_cfg_save["in_feats_m"] = int(in_feats_m)  # type: ignore[arg-type]
    else:
        model_cfg_save["in_feats"] = int(in_feats)      # type: ignore[arg-type]

    # Build model
    if bool(cfg.two_stream):
        model = build_model(
            "gcn",
            model_cfg_save,
            feat_cfg=feat_cfg.to_dict(),
            fps_default=float(cfg.fps_default),
            num_joints=int(V),
            in_feats_j=int(in_feats_j),  # type: ignore[arg-type]
            in_feats_m=int(in_feats_m),  # type: ignore[arg-type]
        ).to(device)
    else:
        model = build_model(
            "gcn",
            model_cfg_save,
            feat_cfg=feat_cfg.to_dict(),
            fps_default=float(cfg.fps_default),
            num_joints=int(V),
            in_feats=int(in_feats),      # type: ignore[arg-type]
        ).to(device)

    if cfg.resume:
        bundle = load_ckpt(cfg.resume, map_location="cpu")
        if isinstance(bundle, dict) and "state_dict" in bundle:
            model.load_state_dict(bundle["state_dict"], strict=True)
        elif isinstance(bundle, dict):
            model.load_state_dict(bundle, strict=True)
        print(f"[resume] loaded: {cfg.resume}")

    ema: Optional[EMA] = None
    if int(cfg.use_ema) == 1:
        ema = EMA(model, decay=float(cfg.ema_decay))
        print(f"[ema] enabled decay={cfg.ema_decay}")

    sampler = None
    pos_w = None

    if cfg.balanced_sampler and cfg.loss != "focal":
        sampler = make_balanced_sampler(np.asarray(train_ds.labels01))
        y = np.asarray(train_ds.labels01).astype(int)
        n_pos = max(1, int((y == 1).sum()))
        n_neg = max(1, int((y == 0).sum()))
        print(f"[sampler] balanced_sampler=1 pos={n_pos} neg={n_neg}")
    else:
        if cfg.loss != "focal":
            if str(cfg.pos_weight).lower() == "auto":
                pos_w = compute_pos_weight(np.asarray(train_ds.labels01))
                print(f"[pos_weight] auto={float(pos_w.item()):.3f}")
            elif str(cfg.pos_weight).lower() not in ("none", "0", "0.0"):
                pos_w = torch.tensor([float(cfg.pos_weight)], dtype=torch.float32)

    if cfg.loss == "focal":
        criterion: nn.Module = FocalLossWithLogits(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    else:
        if pos_w is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
        else:
            criterion = nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        patience=int(cfg.lr_plateau_patience),
        factor=float(cfg.lr_plateau_factor),
        min_lr=float(cfg.lr_plateau_min_lr),
        verbose=False,
    )

    pin = (device.type == "cuda")
    if sampler is not None:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch, sampler=sampler, shuffle=False, num_workers=nw, pin_memory=pin
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch, shuffle=True, num_workers=nw, pin_memory=pin
        )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, num_workers=nw, pin_memory=pin)

    best_score = -1.0
    tag = (cfg.save_tag.strip() + "_") if cfg.save_tag.strip() else ""
    best_path = os.path.join(cfg.save_dir, f"{tag}best.pt")
    history_path = os.path.join(cfg.save_dir, f"{tag}history.jsonl")

    if os.path.exists(history_path):
        os.remove(history_path)

    def log_row(row: Dict[str, Any]) -> None:
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    no_improve = 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        nonfinite_skips_ep = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"train ep{ep}", leave=False), start=1):
            if bool(cfg.two_stream):
                xj, xm, yb = batch
                xj = _to_f32(xj, device)
                xm = _to_f32(xm, device)
                yb = _to_f32(yb, device).view(-1)
                logits = logits_1d(model(xj, xm))
                bsz = int(xj.shape[0])
            else:
                xb, yb = batch
                xb = _to_f32(xb, device)
                yb = _to_f32(yb, device).view(-1)
                logits = logits_1d(model(xb))
                bsz = int(xb.shape[0])

            yb_loss = yb
            if cfg.label_smoothing > 0:
                eps = float(cfg.label_smoothing)
                yb_loss = yb * (1.0 - eps) + 0.5 * eps

            opt.zero_grad(set_to_none=True)
            loss = criterion(logits, yb_loss)
            if not torch.isfinite(loss):
                lr_now = float(opt.param_groups[0]["lr"])
                print(
                    f"[warn] non-finite loss; skipping step "
                    f"ep={ep} step={step} lr={lr_now:.5g} loss={float(loss.detach().cpu()):.6g}"
                )
                nonfinite_skips_ep += 1
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            if ema is not None:
                ema.update(model)

            running += float(loss.detach().cpu()) * bsz
            seen += bsz

        train_loss = running / max(1, seen)

        val_loss = compute_loss_on_loader(
            model,
            val_loader,
            device,
            criterion,
            two_stream=bool(cfg.two_stream),
            ema=ema,
        )

        ctx_eval = ema.use(model) if ema is not None else nullcontext()
        with ctx_eval:
            probs, y_true = collect_probs(model, val_loader, device, two_stream=bool(cfg.two_stream))

        best = best_threshold_by_f1(
            probs, y_true, thr_min=cfg.thr_min, thr_max=cfg.thr_max, thr_step=cfg.thr_step
        )
        prec, rec, f1, fpr, thr = (
            best["precision"],
            best["recall"],
            best["f1"],
            best["fpr"],
            best["thr"],
        )

        extras = ap_auc(probs, y_true)
        apv = float(extras.get("ap", float("nan")))
        auc = float(extras.get("auc", float("nan")))

        score = float(f1) if cfg.monitor == "f1" else float(apv)
        lr_now = float(opt.param_groups[0]["lr"])

        print(
            f"[val] ep={ep:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"F1={f1:.3f} P={prec:.3f} R={rec:.3f} FPR={fpr:.3f} thr={thr:.2f} "
            f"AP={apv:.3f} AUC={auc:.3f} lr={lr_now:.5g}"
        )

        row = {
            "epoch": ep,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_f1": float(f1),
            "val_precision": float(prec),
            "val_recall": float(rec),
            "val_fpr": float(fpr),
            "val_thr": float(thr),
            "ap": float(apv),
            "auc": float(auc),
            "lr": float(lr_now),
            "nonfinite_skips": int(nonfinite_skips_ep),
        }
        log_row(row)

        scheduler.step(score)

        improved = score > best_score + 1e-12
        if improved:
            best_score = score
            no_improve = 0
            best_bundle = {
                "arch": "gcn",
                "state_dict": model.state_dict(),
                "model_cfg": dict(model_cfg_save),
                "feat_cfg": feat_cfg.to_dict(),
                "data_cfg": {"fps_default": float(cfg.fps_default)},
                "best": {"val_best": row, "best_thr": float(thr)},
                "meta": {"monitor": cfg.monitor, "seed": int(cfg.seed), "V": int(V)},
                "train_cfg": asdict(cfg),
            }
            save_ckpt(best_path, best_bundle)
            print(f"[save] {best_path} (best {cfg.monitor}={best_score:.4f})")
        else:
            no_improve += 1

        if ep >= int(cfg.min_epochs) and no_improve >= int(cfg.patience):
            print(f"[early_stop] no improvement for {cfg.patience} epochs")
            break

    print(f"[done] best_{cfg.monitor}={best_score:.4f}")

if __name__ == "__main__":
    main()
