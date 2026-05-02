#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a standalone CTR-GCN line without modifying the current GCN trainer."""

from __future__ import annotations


def _bootstrap_project_root():
    import sys
    from pathlib import Path
    here = Path(__file__).resolve()
    root = here.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_bootstrap_project_root()

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fall_detection.core.ckpt import save_ckpt
from fall_detection.core.ctr_gcn import CTRGCNConfig
from fall_detection.core.features import FeatCfg
from fall_detection.core.losses import FocalLossWithLogits
from fall_detection.core.metrics import ap_auc, best_threshold_by_f1, prf_fpr_at_threshold
from fall_detection.core.models import build_model, logits_1d, pick_device
from fall_detection.training.train_gcn import (
    WindowDatasetGCN,
    _to_f32,
    build_data_cfg_dict,
    collect_probs,
    compute_pos_weight,
    make_balanced_sampler,
    set_seed,
)


@dataclass
class TrainCTRGCNCfg:
    train_dir: str
    val_dir: str
    save_dir: str
    epochs: int = 180
    min_epochs: int = 0
    batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 33724876
    patience: int = 25
    grad_clip: float = 1.0
    fps_default: float = 30.0
    center: str = "pelvis"
    loss: str = "bce"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    monitor: str = "ap"
    pos_weight: str = "auto"
    balanced_sampler: bool = False
    mask_joint_p: float = 0.05
    mask_frame_p: float = 0.05
    x_noise_std: float = 0.0
    x_quant_step: float = 0.0
    temporal_dropout_p: float = 0.0
    channel_schedule: str = "64,64,64,128"
    rel_channels: int = 8
    ctr_rank: int = 8
    temporal_kernel: int = 9
    dropout: float = 0.30
    use_conf_channel: int = 1
    use_motion: int = 1
    use_bone: int = 0
    use_bone_length: int = 0
    motion_scale_by_fps: int = 1
    conf_gate: float = 0.20
    use_precomputed_mask: int = 1
    thr_min: float = 0.05
    thr_max: float = 0.95
    thr_step: float = 0.01
    num_workers: int = 0
    deterministic: int = 1
    amp: int = 0


def _make_feat_cfg(cfg: TrainCTRGCNCfg) -> FeatCfg:
    return FeatCfg(
        center=str(cfg.center),
        use_motion=bool(int(cfg.use_motion)),
        use_bone=bool(int(cfg.use_bone)),
        use_bone_length=bool(int(cfg.use_bone_length)),
        use_conf_channel=bool(int(cfg.use_conf_channel)),
        motion_scale_by_fps=bool(int(cfg.motion_scale_by_fps)),
        conf_gate=float(cfg.conf_gate),
        use_precomputed_mask=bool(int(cfg.use_precomputed_mask)),
    )


def _parse_channel_schedule(raw: str) -> tuple[int, ...]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("channel_schedule must contain at least one integer")
    return tuple(vals)


def _make_loader(ds, cfg: TrainCTRGCNCfg, *, train: bool):
    sampler = None
    shuffle = bool(train)
    if train and cfg.balanced_sampler:
        labels = [int(v) for v in np.asarray(ds.labels01).tolist()]
        sampler = make_balanced_sampler(labels)
        shuffle = False
    return DataLoader(
        ds,
        batch_size=int(cfg.batch),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=max(0, int(cfg.num_workers)),
        pin_memory=torch.cuda.is_available(),
    )


def _make_criterion(cfg: TrainCTRGCNCfg, train_ds, device: torch.device):
    if str(cfg.loss).lower() == "focal":
        return FocalLossWithLogits(alpha=float(cfg.focal_alpha), gamma=float(cfg.focal_gamma))
    if str(cfg.pos_weight).lower() == "auto":
        labels = [int(v) for v in np.asarray(train_ds.labels01).tolist()]
        pos_weight = compute_pos_weight(labels).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CTR-GCN on window NPZs.")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--min_epochs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=33724876)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--fps_default", type=float, default=30.0)
    ap.add_argument("--center", type=str, default="pelvis")
    ap.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
    ap.add_argument("--focal_alpha", type=float, default=0.25)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--monitor", type=str, default="ap", choices=["f1", "ap"])
    ap.add_argument("--pos_weight", type=str, default="auto")
    ap.add_argument("--balanced_sampler", action="store_true")
    ap.add_argument("--mask_joint_p", type=float, default=0.05)
    ap.add_argument("--mask_frame_p", type=float, default=0.05)
    ap.add_argument("--x_noise_std", type=float, default=0.0)
    ap.add_argument("--x_quant_step", type=float, default=0.0)
    ap.add_argument("--temporal_dropout_p", type=float, default=0.0)
    ap.add_argument("--channel_schedule", type=str, default="64,64,64,128")
    ap.add_argument("--rel_channels", type=int, default=8)
    ap.add_argument("--ctr_rank", type=int, default=8)
    ap.add_argument("--temporal_kernel", type=int, default=9)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--use_conf_channel", type=int, default=1)
    ap.add_argument("--use_motion", type=int, default=1)
    ap.add_argument("--use_bone", type=int, default=0)
    ap.add_argument("--use_bone_length", type=int, default=0)
    ap.add_argument("--motion_scale_by_fps", type=int, default=1)
    ap.add_argument("--conf_gate", type=float, default=0.20)
    ap.add_argument("--use_precomputed_mask", type=int, default=1)
    ap.add_argument("--thr_min", type=float, default=0.05)
    ap.add_argument("--thr_max", type=float, default=0.95)
    ap.add_argument("--thr_step", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--amp", type=int, default=0)
    args = ap.parse_args()
    cfg = TrainCTRGCNCfg(**vars(args))

    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = pick_device()
    feat_cfg = _make_feat_cfg(cfg)

    train_ds = WindowDatasetGCN(
        cfg.train_dir,
        split="train",
        feat_cfg=feat_cfg,
        fps_default=cfg.fps_default,
        skip_unlabeled=True,
        two_stream=False,
        mask_joint_p=cfg.mask_joint_p,
        mask_frame_p=cfg.mask_frame_p,
        x_noise_std=cfg.x_noise_std,
        x_quant_step=cfg.x_quant_step,
        temporal_dropout_p=cfg.temporal_dropout_p,
        seed=cfg.seed,
    )
    val_ds = WindowDatasetGCN(
        cfg.val_dir,
        split="val",
        feat_cfg=feat_cfg,
        fps_default=cfg.fps_default,
        skip_unlabeled=True,
        two_stream=False,
        mask_joint_p=0.0,
        mask_frame_p=0.0,
        x_noise_std=0.0,
        x_quant_step=0.0,
        temporal_dropout_p=0.0,
        seed=cfg.seed,
    )
    train_loader = _make_loader(train_ds, cfg, train=True)
    val_loader = _make_loader(val_ds, cfg, train=False)

    sample_x, _ = train_ds[0]
    model_cfg = CTRGCNConfig(
        num_joints=int(sample_x.shape[1]),
        channels=_parse_channel_schedule(cfg.channel_schedule),
        rel_channels=cfg.rel_channels,
        ctr_rank=cfg.ctr_rank,
        temporal_kernel=cfg.temporal_kernel,
        dropout=cfg.dropout,
    )
    model_cfg_d: Dict[str, Any] = {**model_cfg.to_dict(), "in_feats": int(sample_x.shape[-1])}
    model = build_model("ctr_gcn", model_cfg_d, feat_cfg.to_dict()).to(device)
    criterion = _make_criterion(cfg, train_ds, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(int(cfg.amp)) and device.type == "cuda")

    best_metric = float("-inf")
    best_bundle: Dict[str, Any] | None = None
    no_improve = 0

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = _to_f32(xb, device)
            yb = _to_f32(yb, device).view(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=bool(int(cfg.amp)) and device.type in {"cuda", "cpu"}):
                logits = logits_1d(model(xb))
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        p_val, y_val = collect_probs(model, val_loader, device, two_stream=False)
        aucs = ap_auc(p_val, y_val)
        ap_val = float(aucs["ap"])
        auc_val = float(aucs["auc"])
        best = best_threshold_by_f1(p_val, y_val, cfg.thr_min, cfg.thr_max, cfg.thr_step)
        thr = float(best["thr"])
        f1 = float(best["f1"])
        prf = prf_fpr_at_threshold(p_val, y_val, thr)
        prec = float(prf["precision"])
        rec = float(prf["recall"])
        fpr = float(prf["fpr"])
        metric = float(ap_val if cfg.monitor == "ap" else f1)
        print(
            f"[ep {epoch:03d}] train_loss={np.mean(losses):.4f} "
            f"AP={ap_val:.4f} AUC={auc_val:.4f} F1={f1:.4f} "
            f"P={prec:.3f} R={rec:.3f} FPR={fpr:.3f} thr={thr:.3f}"
        )

        if metric > best_metric:
            best_metric = metric
            no_improve = 0
            best_bundle = {
                "arch": "ctr_gcn",
                "state_dict": model.state_dict(),
                "model_cfg": model_cfg_d,
                "feat_cfg": feat_cfg.to_dict(),
                "data_cfg": build_data_cfg_dict(cfg.fps_default),
                "meta": {
                    "epoch": epoch,
                    "best_metric": metric,
                    "monitor": cfg.monitor,
                    "ap": ap_val,
                    "auc": auc_val,
                    "f1": f1,
                    "precision": prec,
                    "recall": rec,
                    "fpr": fpr,
                    "thr": thr,
                },
            }
            save_ckpt(os.path.join(cfg.save_dir, "best.pt"), best_bundle)
        else:
            no_improve += 1
            if epoch >= int(cfg.min_epochs) and no_improve >= int(cfg.patience):
                print(f"[early-stop] no improvement for {no_improve} epochs")
                break

    if best_bundle is None:
        raise RuntimeError("CTR-GCN training finished without producing a checkpoint")

    save_ckpt(
        os.path.join(cfg.save_dir, "last.pt"),
        {
            "arch": "ctr_gcn",
            "state_dict": model.state_dict(),
            "model_cfg": model_cfg_d,
            "feat_cfg": feat_cfg.to_dict(),
            "data_cfg": build_data_cfg_dict(cfg.fps_default),
            "meta": {"best_metric": best_metric},
        },
    )


if __name__ == "__main__":
    main()
