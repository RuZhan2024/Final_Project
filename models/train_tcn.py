#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""models/train_tcn.py

Redesigned trainer that uses core/* as the single source of truth.

Key properties
- Saves a checkpoint BUNDLE (core/ckpt.py) with arch/model_cfg/feat_cfg/state_dict.
- Uses consistent feature flags for train/eval (core/features.py).
- Threshold sweep uses thr_min=0.05 and thr_step=0.01 by default.
- Early stopping can monitor F1 or AP (recommended AP for MUVIM/LE2i).

Typical use:
  python models/train_tcn.py --train_dir .../train --val_dir .../val --save_dir outputs/le2i_tcn_W48S12
"""


from __future__ import annotations

# -------------------------
# Path bootstrap (so `from core.*` works when running as a script)
# -------------------------
import os as _os
import sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)


import argparse
def _write_train_report(path, payload):
    if not path:
        return
    try:
        import os, json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"[warn] failed to write report_json={path}: {e}")

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

from core.ckpt import save_ckpt, load_ckpt, get_cfg
from core.features import FeatCfg, read_window_npz, build_tcn_input
from core.metrics import ap_auc, best_threshold_by_f1
from core.models import TCN, TCNConfig, pick_device, logits_1d


# -------------------------
# Repro
# -------------------------
def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# -------------------------
# DropGraph augmentation (mask only)
# -------------------------
def augment_mask(mask: np.ndarray, rng: np.random.Generator, mask_joint_p: float, mask_frame_p: float) -> np.ndarray:
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
    if not m.any():
        m[int(rng.integers(0, T)), int(rng.integers(0, V))] = True
    return m


class WindowDatasetTCN(Dataset):
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
        mask_joint_p: float = 0.0,
        mask_frame_p: float = 0.0,
        seed: int = 33724876,
    ):
        import glob

        self.fps_default = float(fps_default)

        self.files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No .npz under: {root}")

        kept: List[str] = []
        labels: List[int] = []
        skipped = 0
        for p in self.files:
            try:
                _, _, _, _, _, meta = read_window_npz(p, fps_default=fps_default)
                if skip_unlabeled and meta.y < 0:
                    skipped += 1
                    continue
                kept.append(p)
                labels.append(int(meta.y if meta.y >= 0 else 0))
            except Exception:
                skipped += 1
        if not kept:
            raise FileNotFoundError(f"All windows under {root} were unlabeled/unreadable.")
        if skipped:
            print(f"[dataset] skipped {skipped} unlabeled/unreadable windows in {root}")

        self.files = kept
        self.labels = np.asarray(labels, dtype=np.int64)

        # Add hard negatives (can be unlabeled NPZ) and oversample them by duplicating paths.
        if hard_neg_files:
            extra: List[str] = []
            for p in hard_neg_files:
                p = str(p).strip()
                if not p:
                    continue
                if not os.path.exists(p):
                    continue
                # avoid accidentally treating labeled positives as negatives
                _, _, _, _, _, meta2 = read_window_npz(p, fps_default=self.fps_default)
                if meta2.y == 1:
                    continue
                extra.append(p)

            if extra:
                mult = int(hard_neg_mult) if int(hard_neg_mult) > 0 else 1
                extra_rep = extra * mult
                self.files.extend(extra_rep)
                self.labels = np.concatenate([self.labels, np.zeros(len(extra_rep), dtype=np.int64)], axis=0)

        self.split = str(split)
        self.feat_cfg = feat_cfg
        self.fps_default = float(fps_default)

        self.mask_joint_p = float(mask_joint_p) if self.split == "train" else 0.0
        self.mask_frame_p = float(mask_frame_p) if self.split == "train" else 0.0
        self.rng = np.random.default_rng(int(seed) + (0 if self.split == "train" else 991))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        joints, motion, conf, mask, fps, meta = read_window_npz(p, fps_default=self.fps_default)

        # mask selection happens in build_tcn_input; but we may augment the precomputed mask
        if mask is None:
            # provide None; build_tcn_input will derive
            pass
        else:
            if self.mask_joint_p > 0 or self.mask_frame_p > 0:
                mask = augment_mask(mask, self.rng, self.mask_joint_p, self.mask_frame_p)

        X, _ = build_tcn_input(joints, motion, conf, mask, fps, self.feat_cfg)  # [T,C]
        y = float(meta.y if meta.y >= 0 else 0.0)
        return torch.from_numpy(X).float(), torch.tensor([y], dtype=torch.float32)


def compute_pos_weight(y: np.ndarray) -> Optional[torch.Tensor]:
    y = np.asarray(y).reshape(-1)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos <= 0 or neg <= 0:
        return None
    w = neg / pos
    print(f"[info] class balance: pos={int(pos)} neg={int(neg)} pos_weight={w:.2f}")
    return torch.tensor([w], dtype=torch.float32)


def make_balanced_sampler(y: np.ndarray) -> WeightedRandomSampler:
    y = np.asarray(y).reshape(-1)
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    w_pos = 1.0 / pos
    w_neg = 1.0 / neg
    w = np.where(y == 1, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(w), replacement=True)


@torch.no_grad()
def collect_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ps: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = logits_1d(model(xb))
        p = torch.sigmoid(logits).detach().cpu().numpy()
        y = yb.detach().cpu().numpy().reshape(-1).astype(int)
        ps.append(p)
        ys.append(y)
    return (np.concatenate(ps) if ps else np.array([])), (np.concatenate(ys) if ys else np.array([]))


@dataclass
class TrainCfg:
    train_dir: str
    val_dir: str
    save_dir: str
    test_dir: Optional[str] = None

    resume: Optional[str] = None
    hard_neg_list: Optional[str] = None
    hard_neg_mult: int = 0

    epochs: int = 50
    batch: int = 128
    lr: float = 1e-3
    seed: int = 33724876
    grad_clip: float = 1.0
    patience: int = 12
    fps_default: float = 30.0

    # monitoring
    monitor: str = "f1"  # f1|ap

    # imbalance (use only ONE)
    pos_weight: str = "auto"   # auto|none|<float>
    balanced_sampler: bool = False

    # augmentation
    mask_joint_p: float = 0.15
    mask_frame_p: float = 0.10

    # tcn
    hidden: int = 128
    dropout: float = 0.30
    num_blocks: int = 4
    kernel: int = 3

    # features
    center: str = "pelvis"
    use_motion: int = 1
    use_conf_channel: int = 1
    motion_scale_by_fps: int = 1
    conf_gate: float = 0.20
    use_precomputed_mask: int = 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--test_dir", default=None)
    ap.add_argument("--save_dir", required=True)

    # Optional training report (for Makefile compatibility / logging)
    ap.add_argument("--report_json", default=None, help="Write a training summary JSON to this path.")
    ap.add_argument("--report_dataset_name", default=None, help="Dataset name to store in the report JSON.")

    ap.add_argument("--resume", default=None, help="Path to a checkpoint bundle to fine-tune from")
    ap.add_argument("--hard_neg_list", default=None, help="TXT of NPZ paths to oversample as hard negatives")
    ap.add_argument("--hard_neg_mult", type=int, default=0, help="Oversampling multiplier for hard negatives (e.g., 5)")

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=33724876)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--fps_default", type=float, default=30.0)

    ap.add_argument("--monitor", choices=["f1", "ap"], default="f1")

    ap.add_argument("--pos_weight", default="auto", help="auto|none|<float>")
    ap.add_argument("--balanced_sampler", action="store_true")

    ap.add_argument("--mask_joint_p", type=float, default=0.15)
    ap.add_argument("--mask_frame_p", type=float, default=0.10)

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--num_blocks", type=int, default=4)
    ap.add_argument("--kernel", type=int, default=3)

    # feature flags (Makefile-friendly ints)
    ap.add_argument("--center", choices=["pelvis", "none"], default="pelvis")
    # Deprecated: older Makefiles used --scale (e.g., 'torso').
    # Scaling is handled inside core/features.py; we accept the flag to avoid breaking runs.
    ap.add_argument("--scale", default=None, help="(deprecated) kept for Makefile compatibility; ignored")
    ap.add_argument("--use_motion", type=int, default=1)
    ap.add_argument("--use_conf_channel", type=int, default=1)
    ap.add_argument("--motion_scale_by_fps", type=int, default=1)
    ap.add_argument("--conf_gate", type=float, default=0.20)
    ap.add_argument("--use_precomputed_mask", type=int, default=1)

    args = ap.parse_args()

    if getattr(args, "scale", None) not in (None, "", "none"):
        print(f"[warn] --scale={args.scale!r} is deprecated and ignored (scaling is handled in core/features.py)")

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
        seed=int(args.seed),
        grad_clip=float(args.grad_clip),
        patience=int(args.patience),
        fps_default=float(args.fps_default),
        monitor=str(args.monitor),
        pos_weight=str(args.pos_weight),
        balanced_sampler=bool(args.balanced_sampler),
        mask_joint_p=float(args.mask_joint_p),
        mask_frame_p=float(args.mask_frame_p),
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

    # enforce imbalance rule
    if cfg.balanced_sampler and str(cfg.pos_weight).strip().lower() not in {"none", "0", "0.0"}:
        if str(cfg.pos_weight).strip().lower() == "auto":
            print("[warn] You enabled --balanced_sampler and --pos_weight=auto. Using balanced_sampler only.")
            cfg.pos_weight = "none"

    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = pick_device()
    print(f"[info] device: {device.type}")


    bundle = None
    if cfg.resume:
        bundle = load_ckpt(cfg.resume, map_location="cpu")
        arch0, model_cfg_d0, feat_cfg_d0, data_cfg0 = get_cfg(bundle)
        if arch0 and arch0 != "tcn":
            raise SystemExit(f"[err] resume arch mismatch: expected tcn, got {arch0}")
        # Override feature flags from checkpoint
        try:
            cfg.center = str(feat_cfg_d0.get("center", cfg.center))
            cfg.use_motion = int(feat_cfg_d0.get("use_motion", cfg.use_motion))
            cfg.use_conf_channel = int(feat_cfg_d0.get("use_conf_channel", cfg.use_conf_channel))
            cfg.motion_scale_by_fps = int(feat_cfg_d0.get("motion_scale_by_fps", cfg.motion_scale_by_fps))
            cfg.conf_gate = float(feat_cfg_d0.get("conf_gate", cfg.conf_gate))
            cfg.use_precomputed_mask = int(feat_cfg_d0.get("use_precomputed_mask", cfg.use_precomputed_mask))
        except Exception:
            pass
        # Override model config (keeps architecture identical)
        try:
            cfg.hidden = int(model_cfg_d0.get("hidden", cfg.hidden))
            cfg.dropout = float(model_cfg_d0.get("dropout", cfg.dropout))
            cfg.num_blocks = int(model_cfg_d0.get("num_blocks", cfg.num_blocks))
            cfg.kernel = int(model_cfg_d0.get("kernel", cfg.kernel))
        except Exception:
            pass
        # Prefer fps_default from checkpoint (for consistent motion scaling)
        try:
            cfg.fps_default = float(data_cfg0.get("fps_default", cfg.fps_default))
        except Exception:
            pass
    feat_cfg = FeatCfg(
        center=cfg.center,
        use_motion=bool(cfg.use_motion),
        use_conf_channel=bool(cfg.use_conf_channel),
        motion_scale_by_fps=bool(cfg.motion_scale_by_fps),
        conf_gate=float(cfg.conf_gate),
        use_precomputed_mask=bool(cfg.use_precomputed_mask),
    )


    hard_neg_files: Optional[List[str]] = None
    if cfg.hard_neg_list:
        try:
            with open(cfg.hard_neg_list, "r", encoding="utf-8") as f:
                hard_neg_files = [ln.strip() for ln in f.readlines() if ln.strip()]
        except Exception:
            hard_neg_files = None
    train_ds = WindowDatasetTCN(
        cfg.train_dir,
        split="train",
        feat_cfg=feat_cfg,
        fps_default=cfg.fps_default,
        skip_unlabeled=True,
        hard_neg_files=hard_neg_files,
        hard_neg_mult=int(cfg.hard_neg_mult),
        mask_joint_p=cfg.mask_joint_p,
        mask_frame_p=cfg.mask_frame_p,
        seed=cfg.seed,
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

    model_cfg = TCNConfig(hidden=cfg.hidden, dropout=cfg.dropout, num_blocks=cfg.num_blocks, kernel=cfg.kernel)
    model = TCN(in_ch=C, hidden=model_cfg.hidden, dropout=model_cfg.dropout, num_blocks=model_cfg.num_blocks, kernel=model_cfg.kernel).to(device)
    if bundle is not None:
        model.load_state_dict(bundle["state_dict"], strict=True)
        print(f"[info] resumed weights from: {cfg.resume}")

    # Safety: use only ONE imbalance strategy at a time
    if cfg.balanced_sampler and str(cfg.pos_weight).strip().lower() not in ["none", "0", "0.0"]:
        cfg.pos_weight = "none"
    # loss
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

    if pos_w is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3, min_lr=1e-6)

    pin = torch.cuda.is_available()
    if cfg.balanced_sampler:
        sampler = make_balanced_sampler(train_ds.labels)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch, sampler=sampler, shuffle=False, num_workers=0, pin_memory=pin)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, num_workers=0, pin_memory=pin)

    best_score = -1.0
    best_path = os.path.join(cfg.save_dir, "best.pt")
    no_improve = 0
    history_path = os.path.join(cfg.save_dir, "history.jsonl")
    if os.path.exists(history_path):
        os.remove(history_path)

    def log_row(row: Dict[str, Any]) -> None:
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"train TCN ep{ep}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device).view(-1)
            opt.zero_grad(set_to_none=True)
            logits = logits_1d(model(xb))
            loss = criterion(logits, yb)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += float(loss.detach().cpu()) * int(xb.shape[0])
            n_seen += int(xb.shape[0])
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        train_loss = running / max(1, n_seen)

        probs, y_true = collect_probs(model, val_loader, device)
        if probs.size == 0:
            print("[val] empty val set?")
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

        sched.step(score if np.isfinite(score) else row["val_f1"])

        if score > best_score + 1e-6:
            best_score = float(score)
            no_improve = 0
            save_ckpt(
                best_path,
                arch="tcn",
                state_dict=model.state_dict(),
                model_cfg=model_cfg.to_dict(),
                feat_cfg=feat_cfg.to_dict(),
                data_cfg={"fps_default": cfg.fps_default},
                best={"val_best": row, "best_thr": float(row["val_thr"])},
                meta={"monitor": cfg.monitor},
            )
            print(f"[save] {best_path} (best {cfg.monitor}={best_score:.4f})")
        else:
            no_improve += 1
            if cfg.patience > 0 and no_improve >= cfg.patience:
                print(f"[early stop] patience={cfg.patience} reached at ep={ep}")
                break

    report = {"arch": "tcn", "best_ckpt": best_path, "monitor": cfg.monitor, "best_score": float(best_score)}
    with open(os.path.join(cfg.save_dir, "train_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[done] ckpt={best_path}")
    print(f"[ok] wrote history: {history_path}")


if __name__ == "__main__":
    main()
