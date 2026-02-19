#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""models/train_tcn.py

Upgraded TCN trainer (single source of truth via core/*).

Key fixes vs older script:
- Uses core.features.read_window_npz + core.features.build_canonical_input (matches window NPZ schema).
- Consistent feature flags (FeatCfg) saved into checkpoint bundle.
- Works with hard-negative mining (--resume + --hard_neg_list + --hard_neg_mult).
- Balanced sampler and/or pos_weight supported (choose ONE).
- Applies random mask augmentation (--mask_joint_p/--mask_frame_p) *after* feature-building so it works
  regardless of whether precomputed/derived masks are used.
- Validation collect_probs flattens y correctly.
"""

from __future__ import annotations

def _to_f32(x, device: torch.device) -> torch.Tensor:
    # Works for numpy arrays and torch tensors
    return torch.as_tensor(x, dtype=torch.float32, device=device)

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
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from core.ckpt import get_cfg, load_ckpt, save_ckpt
from core.features import FeatCfg, build_tcn_input, build_canonical_input,read_window_npz
from core.losses import FocalLossWithLogits
from core.metrics import ap_auc, best_threshold_by_f1
from core.models import TCNConfig, build_model, pick_device
from core.ema import EMA


# -------------------------
# Utilities
# -------------------------
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


def prf_fpr_at_threshold(y_true: np.ndarray, p: np.ndarray, thr: float) -> Tuple[float, float, float, float]:
    yb = (np.asarray(y_true).reshape(-1).astype(np.int64) > 0).astype(np.int64)
    pb = (np.asarray(p).reshape(-1) >= float(thr)).astype(np.int64)

    tp = int(((pb == 1) & (yb == 1)).sum())
    fp = int(((pb == 1) & (yb == 0)).sum())
    fn = int(((pb == 0) & (yb == 1)).sum())
    tn = int(((pb == 0) & (yb == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(prec), float(rec), float(f1), float(fpr)


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
    # ensure at least one valid joint/frame remains
    if not m.any():
        m[int(rng.integers(0, T)), int(rng.integers(0, V))] = True
    return m


def flatten_tcn_from_gcn(X: np.ndarray, feat_cfg: FeatCfg) -> np.ndarray:
    """Convert canonical X[T,V,F] into TCN input x[T, V*F].

    This delegates to core.features.build_tcn_input so the layout is single-source-of-truth.
    """
    return build_tcn_input(X, feat_cfg)




def collect_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ema: Optional[EMA] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ps: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ctx = ema.use(model) if ema is not None else nullcontext()
    with ctx, torch.no_grad():
        for xb, yb in loader:
            xb = _to_f32(xb, device)  # [B,T,C]
            logits = logits_1d(model(xb))
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
    ema: Optional[EMA] = None,
) -> float:
    """Compute mean loss on a loader with model in eval mode (no grad)."""
    model.eval()
    losses: List[float] = []
    counts: List[int] = []
    ctx = ema.use(model) if ema is not None else nullcontext()
    with ctx, torch.no_grad():
        for xb, yb in loader:
            xb = _to_f32(xb, device)
            yb = _to_f32(yb, device).view(-1)
            logits = logits_1d(model(xb))
            loss = criterion(logits, yb).detach()
            losses.append(float(loss.cpu()) * xb.shape[0])
            counts.append(int(xb.shape[0]))
    return float(sum(losses) / max(1, sum(counts)))



def list_npz_files(root: str) -> List[str]:
    import glob
    pat = os.path.join(root, "**", "*.npz")
    files = glob.glob(pat, recursive=True)
    files.sort()
    return files


class WindowDatasetTCN(Dataset):
    def __init__(
        self,
        root: str,
        *,
        split: str,
        feat_cfg: FeatCfg,
        fps_default: float,
        skip_unlabeled: bool,
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
        self.mask_joint_p = float(mask_joint_p)
        self.mask_frame_p = float(mask_frame_p)

        files = list_npz_files(self.root)
        self.files: List[str] = []
        self.labels01: List[int] = []

        fail = 0
        examples: List[str] = []

        # main dataset files
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

        # extra hard negatives (treated as y=0 unless y==1, which is skipped)
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

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        fp = self.files[i]
        joints, motion, conf, mask, fps, meta = read_window_npz(fp, fps_default=self.fps_default)

        # Build [T,V,F] first; get mask_used after conf gating.
        Xg, mask_used = build_canonical_input(
            joints_xy=joints,
            motion_xy=motion,
            conf=conf,
            mask=mask,
            fps=fps,
            feat_cfg=self.feat_cfg,
        )

        # Random mask augmentation (train only) applied post-feature-build.
        if self.split == "train" and (self.mask_joint_p > 0 or self.mask_frame_p > 0):
            m_aug = augment_mask(mask_used, self.rng, self.mask_joint_p, self.mask_frame_p)
            Xg = Xg * m_aug[..., None]

        Xt = flatten_tcn_from_gcn(Xg, self.feat_cfg)  # [T,C]
        y = 1 if int(meta.y) == 1 else 0

        # NumPy -> Torch (explicit, no copy when possible)
        return (
            torch.as_tensor(Xt, dtype=torch.float32),
            torch.as_tensor([y], dtype=torch.float32),
        )

@dataclass
class TrainCfg:
    train_dir: str
    val_dir: str
    save_dir: str
    test_dir: Optional[str] = None

    # resume / hard negatives
    resume: Optional[str] = None
    hard_neg_list: Optional[str] = None
    hard_neg_mult: int = 1

    epochs: int = 200
    batch: int = 128
    lr: float = 1e-3
    seed: int = 33724876
    grad_clip: float = 1.0

    lr_plateau_patience: int = 5
    lr_plateau_factor: float = 0.5
    lr_plateau_min_lr: float = 1e-6

    weight_decay: float = 1e-4
    label_smoothing: float = 0.0

    fixed_thr: float = 0.5
    thr_min: float = 0.05
    thr_max: float = 0.95
    thr_step: float = 0.01

    use_ema: int = 0
    ema_decay: float = 0.995

    patience: int = 30
    fps_default: float = 30.0

    loss: str = "bce"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    monitor: str = "f1"
    pos_weight: str = "auto"
    balanced_sampler: bool = False

    mask_joint_p: float = 0.05
    mask_frame_p: float = 0.05

    hidden: int = 128
    dropout: float = 0.30
    num_blocks: int = 4
    kernel: int = 3

    # feature flags (must match fit/eval)
    center: str = "pelvis"
    use_motion: int = 1
    use_conf_channel: int = 1
    use_bone: int = 0
    use_bone_length: int = 0
    motion_scale_by_fps: int = 1
    conf_gate: float = 0.20
    use_precomputed_mask: int = 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Train TCN on window NPZs.")

    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--test_dir", default=None)
    ap.add_argument("--save_dir", required=True)

    ap.add_argument("--resume", default=None, help="Optional checkpoint to init weights from.")
    ap.add_argument("--hard_neg_list", default=None, help="Optional text file listing extra negative window NPZ paths.")
    ap.add_argument("--hard_neg_mult", type=int, default=1, help="Repeat hard negatives N times.")

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=33724876)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--lr_plateau_patience", type=int, default=5)
    ap.add_argument("--lr_plateau_factor", type=float, default=0.5)
    ap.add_argument("--lr_plateau_min_lr", type=float, default=1e-6)

    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.0)

    ap.add_argument("--fixed_thr", type=float, default=0.5)
    ap.add_argument("--thr_min", type=float, default=0.05)
    ap.add_argument("--thr_max", type=float, default=0.95)
    ap.add_argument("--thr_step", type=float, default=0.01)

    ap.add_argument("--use_ema", type=int, default=0)
    ap.add_argument("--ema_decay", type=float, default=0.995)

    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--fps_default", type=float, default=30.0)

    ap.add_argument("--monitor", choices=["f1", "ap"], default="f1")
    ap.add_argument("--pos_weight", default="auto")
    ap.add_argument("--balanced_sampler", action="store_true")

    ap.add_argument("--loss", choices=["bce", "focal"], default="bce")
    ap.add_argument("--focal_alpha", type=float, default=0.25)
    ap.add_argument("--focal_gamma", type=float, default=2.0)

    ap.add_argument("--mask_joint_p", type=float, default=0.05)
    ap.add_argument("--mask_frame_p", type=float, default=0.05)

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--num_blocks", type=int, default=4)
    ap.add_argument("--kernel", type=int, default=3)

    ap.add_argument("--center", choices=["pelvis", "none"], default="pelvis")
    ap.add_argument("--use_motion", type=int, default=1)
    ap.add_argument("--use_conf_channel", type=int, default=1)
    ap.add_argument("--use_bone", type=int, default=0)
    ap.add_argument("--use_bone_length", type=int, default=0)
    ap.add_argument("--motion_scale_by_fps", type=int, default=1)
    ap.add_argument("--conf_gate", type=float, default=0.20)
    ap.add_argument("--use_precomputed_mask", type=int, default=1)

    args = ap.parse_args()
    cfg = TrainCfg(**vars(args))

    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # If resuming, load cfg from checkpoint (single source of truth).
    if cfg.resume:
        bundle = load_ckpt(cfg.resume, map_location="cpu")
        arch0, model_cfg_d0, feat_cfg_d0, data_cfg0 = get_cfg(bundle)
        if arch0 and arch0 != "tcn":
            raise SystemExit(f"[err] resume arch mismatch: expected tcn, got {arch0}")

        # Override feature flags (so train/fit/eval stay aligned).
        if isinstance(feat_cfg_d0, dict) and feat_cfg_d0:
            cfg.center = str(feat_cfg_d0.get("center", cfg.center))
            cfg.use_motion = 1 if bool(feat_cfg_d0.get("use_motion", cfg.use_motion)) else 0
            cfg.use_conf_channel = 1 if bool(feat_cfg_d0.get("use_conf_channel", cfg.use_conf_channel)) else 0
            cfg.use_bone = 1 if bool(feat_cfg_d0.get("use_bone", cfg.use_bone)) else 0
            cfg.use_bone_length = 1 if bool(feat_cfg_d0.get("use_bone_length", cfg.use_bone_length)) else 0
            cfg.motion_scale_by_fps = 1 if bool(feat_cfg_d0.get("motion_scale_by_fps", cfg.motion_scale_by_fps)) else 0
            cfg.conf_gate = float(feat_cfg_d0.get("conf_gate", cfg.conf_gate))
            cfg.use_precomputed_mask = 1 if bool(feat_cfg_d0.get("use_precomputed_mask", cfg.use_precomputed_mask)) else 0

        # Override model cfg (avoid accidental mismatch when mining hard negatives).
        if isinstance(model_cfg_d0, dict) and model_cfg_d0:
            cfg.hidden = int(model_cfg_d0.get("hidden", cfg.hidden))
            cfg.dropout = float(model_cfg_d0.get("dropout", cfg.dropout))
            cfg.num_blocks = int(model_cfg_d0.get("num_blocks", model_cfg_d0.get("blocks", cfg.num_blocks)))
            cfg.kernel = int(model_cfg_d0.get("kernel", cfg.kernel))

        if isinstance(data_cfg0, dict) and "fps_default" in data_cfg0:
            cfg.fps_default = float(data_cfg0["fps_default"])

    with open(os.path.join(cfg.save_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = pick_device()
    print(f"[info] device: {device.type}")

    feat_cfg = FeatCfg(
        center=cfg.center,
        use_motion=bool(cfg.use_motion),
        use_conf_channel=bool(cfg.use_conf_channel),
        use_bone=bool(cfg.use_bone),
        use_bone_length=bool(cfg.use_bone_length),
        motion_scale_by_fps=bool(cfg.motion_scale_by_fps),
        conf_gate=float(cfg.conf_gate),
        use_precomputed_mask=bool(cfg.use_precomputed_mask),
    )

    extra_neg_files: Optional[List[str]] = None
    if cfg.hard_neg_list:
        with open(cfg.hard_neg_list, "r", encoding="utf-8") as f:
            extra_neg_files = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        print(f"[info] hard_neg_list: {cfg.hard_neg_list} (n={len(extra_neg_files)}) mult={cfg.hard_neg_mult}")

    train_ds = WindowDatasetTCN(
        cfg.train_dir,
        split="train",
        feat_cfg=feat_cfg,
        fps_default=cfg.fps_default,
        skip_unlabeled=True,
        mask_joint_p=cfg.mask_joint_p,
        mask_frame_p=cfg.mask_frame_p,
        seed=cfg.seed,
        extra_neg_files=extra_neg_files,
        extra_neg_mult=cfg.hard_neg_mult,
    )
    val_ds = WindowDatasetTCN(
        cfg.val_dir,
        split="val",
        feat_cfg=feat_cfg,
        fps_default=cfg.fps_default,
        skip_unlabeled=True,
        mask_joint_p=0.0,
        mask_frame_p=0.0,
        seed=cfg.seed,
    )

    x0, _ = train_ds[0]
    T, C = x0.shape
    print(f"[info] window shape: T={T}, C={C}")

    from core.features import channel_layout

    layout = channel_layout(feat_cfg)
    F = max(end for (_, end) in layout.values())  # feature channels per joint
    if C % F != 0:
        raise ValueError(f"Flattened C={C} not divisible by F={F}. Check feature layout / flatten.")
    V = C // F

    model_cfg = TCNConfig(hidden=cfg.hidden, dropout=cfg.dropout, num_blocks=cfg.num_blocks, kernel=cfg.kernel)
    model = build_model("tcn", model_cfg.to_dict(), in_ch=C).to(device)

    model_cfg_save = model_cfg.to_dict()
    model_cfg_save["in_ch"] = int(C)
    model_cfg_save["num_joints"] = int(V)

    resume_bundle = None
    if cfg.resume:
        bundle = load_ckpt(cfg.resume, map_location="cpu")
        sd = bundle.get("state_dict", {})
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[warn] resume: missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"[warn] resume: unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        print(f"[info] resumed weights from: {cfg.resume}")
        resume_bundle = bundle

    ema = EMA(model, decay=cfg.ema_decay) if int(cfg.use_ema) == 1 else None

    if ema is not None and resume_bundle is not None and isinstance(resume_bundle, dict) and "ema_state" in resume_bundle:
        try:
            ema.load_state_dict(resume_bundle["ema_state"])
            print("[info] resumed EMA state")
        except Exception as e:
            print(f"[warn] failed to load EMA state: {type(e).__name__}: {e}")

    
    # dataset summary (helps catch silent skips / tiny splits)
    n_tr = len(train_ds)
    n_va = len(val_ds)
    tr_pos = int((train_ds.labels01 == 1).sum()) if hasattr(train_ds, "labels01") else -1
    tr_neg = int((train_ds.labels01 == 0).sum()) if hasattr(train_ds, "labels01") else -1
    va_pos = int((val_ds.labels01 == 1).sum()) if hasattr(val_ds, "labels01") else -1
    va_neg = int((val_ds.labels01 == 0).sum()) if hasattr(val_ds, "labels01") else -1
    print(f"[info] train windows: n={n_tr} pos={tr_pos} neg={tr_neg}")
    print(f"[info] val   windows: n={n_va} pos={va_pos} neg={va_neg}")
# imbalance strategy guard
    # IMPORTANT: avoid "double correction" which often hurts calibration and AP:
    # - For BCE: choose ONE of (--balanced_sampler) OR (--pos_weight auto/number).
    # - For focal: prefer NOT using --balanced_sampler (focal already emphasizes hard samples).
    if cfg.loss != "focal":
        if cfg.balanced_sampler and str(cfg.pos_weight).lower() not in ("none", "0", "0.0"):
            raise SystemExit("[err] choose ONE imbalance strategy for BCE: --balanced_sampler OR --pos_weight (auto/number).")
    else:
        if cfg.balanced_sampler:
            print("[warn] loss=focal with --balanced_sampler can hurt probability calibration / AP. Consider dropping --balanced_sampler.")

    # loss
    if cfg.loss == "focal":
        criterion = FocalLossWithLogits(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
        print("[warn] loss=focal: ignoring --pos_weight (use focal_alpha/gamma instead).")
    else:
        pos_w = None
        if str(cfg.pos_weight).lower() == "auto":
            pos_w = compute_pos_weight(train_ds.labels01)
        elif str(cfg.pos_weight).lower() not in ("none", "0", "0.0"):
            pos_w = torch.tensor([float(cfg.pos_weight)], dtype=torch.float32)

        if pos_w is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
        else:
            criterion = nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=cfg.lr_plateau_factor, patience=cfg.lr_plateau_patience, min_lr=cfg.lr_plateau_min_lr
    )

    pin = torch.cuda.is_available()
    if cfg.balanced_sampler:
        sampler = make_balanced_sampler(train_ds.labels01)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch, sampler=sampler, shuffle=False, num_workers=0, pin_memory=pin)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=0, pin_memory=pin)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, num_workers=0, pin_memory=pin)

    best_score = -1.0
    best_path = os.path.join(cfg.save_dir, "best.pt")
    history_path = os.path.join(cfg.save_dir, "history.jsonl")
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

        for xb, yb in tqdm(train_loader, desc=f"train ep{ep}", leave=False):
            xb = _to_f32(xb, device)         # [B,T,C]
            yb = _to_f32(yb, device).view(-1)  # [B]

            opt.zero_grad(set_to_none=True)
            logits = logits_1d(model(xb))

            # label smoothing
            yb_loss = yb
            if cfg.label_smoothing > 0:
                eps = float(cfg.label_smoothing)
                yb_loss = yb * (1.0 - eps) + 0.5 * eps

            loss = criterion(logits, yb_loss)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            if ema is not None:
                ema.update(model)

            running += float(loss.detach().cpu()) * xb.shape[0]
            seen += xb.shape[0]

        train_loss = running / max(1, seen)

        probs, y_true = collect_probs(model, val_loader, device, ema=ema)

        # compute val loss (for debugging divergence vs overfit)
        val_loss = compute_loss_on_loader(model, val_loader, device, criterion, ema=ema)


        best = best_threshold_by_f1(probs, y_true, thr_min=cfg.thr_min, thr_max=cfg.thr_max, thr_step=cfg.thr_step)
        prec, rec, f1, fpr, thr = best["precision"], best["recall"], best["f1"], best["fpr"], best["thr"]
        p_fixed, r_fixed, f1_fixed, fpr_fixed = prf_fpr_at_threshold(y_true, probs, cfg.fixed_thr)
        extras = ap_auc(probs, y_true)
        apv = float(extras.get("ap", float("nan")))
        auc = float(extras.get("auc", float("nan")))

        score = float(f1) if cfg.monitor == "f1" else float(apv)
        lr_now = float(opt.param_groups[0]["lr"])

        print(
            f"[val] ep={ep:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"F1={f1:.3f} (F1@{cfg.fixed_thr:.2f}={f1_fixed:.3f}) "
            f"P={prec:.3f} R={rec:.3f} FPR={fpr:.3f} thr={thr:.2f} "
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

            "val_f1_fixed": float(f1_fixed),
            "val_p_fixed": float(p_fixed),
            "val_r_fixed": float(r_fixed),
            "val_fpr_fixed": float(fpr_fixed),
            "val_thr_fixed": float(cfg.fixed_thr),

            "val_ap": float(apv),
            "val_auc": float(auc),

            "monitor": cfg.monitor,
            "monitor_score": float(score),
            "lr": lr_now,
        }
        log_row(row)

        sched.step(score if np.isfinite(score) else float(f1))

        if score > best_score + 1e-6:
            best_score = float(score)
            no_improve = 0

            ctx_save = ema.use(model) if ema is not None else nullcontext()
            with ctx_save:
                save_ckpt(
                    best_path,
                    arch="tcn",
                    state_dict=model.state_dict(),
                    model_cfg=model_cfg_save,
                    feat_cfg=feat_cfg.to_dict(),
                    data_cfg={"fps_default": cfg.fps_default},
                    best={"val_best": row, "best_thr": float(thr)},
                    meta={"monitor": cfg.monitor},
                    **({"ema_state": ema.state_dict()} if ema is not None else {}),
                )
            print(f"[save] {best_path} (best {cfg.monitor}={best_score:.4f})")
        else:
            no_improve += 1
            if cfg.patience > 0 and no_improve >= cfg.patience:
                print(f"[early stop] patience={cfg.patience} reached at ep={ep}")
                break

    print(f"[done] ckpt={best_path}")
    print(f"[ok] wrote history: {history_path}")


if __name__ == "__main__":
    main()
