#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the CTR-GCN experiment model from window NPZs."""

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
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fall_detection.core.ckpt import load_ckpt, save_ckpt
from fall_detection.core.ctr_gcn import CTRGCNConfig
from fall_detection.core.ema import EMA
from fall_detection.core.features import FeatCfg
from fall_detection.core.losses import FocalLossWithLogits
from fall_detection.core.metrics import ap_auc, best_threshold_by_f1, prf_fpr_at_threshold
from fall_detection.core.models import build_model, logits_1d, pick_device
from fall_detection.training.graph_training_utils import (
    GraphWindowDataset,
    _validate_hard_neg_paths,
    _to_f32,
    build_data_cfg_dict,
    compute_loss_on_loader,
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
    resume: Optional[str] = None
    hard_neg_list: Optional[str] = None
    hard_neg_mult: int = 1
    allow_hard_neg_nontrain: int = 0
    epochs: int = 180
    min_epochs: int = 0
    batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    seed: int = 33724876
    patience: int = 25
    grad_clip: float = 1.0
    scheduler: str = "plateau"
    scheduler_metric: Optional[str] = None
    scheduler_ema_beta: float = 0.0
    lr_plateau_patience: int = 10
    lr_plateau_factor: float = 0.5
    lr_plateau_min_lr: float = 1e-6
    max_lr: Optional[float] = None
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
    two_stream: int = 0
    stream_mode: str = "joint_bone"
    fuse: str = "concat"
    stream_drop_joint_p: float = 0.0
    stream_drop_bone_p: float = 0.0
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
    use_ema: int = 0
    ema_decay: float = 0.995
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


def _select_monitor_metric(
    monitor: str,
    ap_val: float,
    f1: float,
    train_loss: float,
) -> tuple[float, str]:
    candidates = []
    if monitor == "ap":
        candidates.extend((("ap", ap_val), ("f1", f1), ("neg_train_loss", -train_loss)))
    else:
        candidates.extend((("f1", f1), ("ap", ap_val), ("neg_train_loss", -train_loss)))

    for name, value in candidates:
        metric = float(value)
        if np.isfinite(metric):
            return metric, name
    return 0.0, "constant"


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
    pos_weight_raw = str(cfg.pos_weight).lower()
    if pos_weight_raw == "auto":
        labels = [int(v) for v in np.asarray(train_ds.labels01).tolist()]
        pos_weight = compute_pos_weight(labels).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if pos_weight_raw not in ("none", "0", "0.0", "false"):
        try:
            pos_weight = torch.tensor([float(cfg.pos_weight)], dtype=torch.float32, device=device)
            if float(pos_weight.item()) > 0:
                return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        except ValueError:
            pass
    return nn.BCEWithLogitsLoss()


def _scheduler_metric_name(cfg: TrainCTRGCNCfg) -> str:
    if cfg.scheduler_metric:
        return str(cfg.scheduler_metric)
    return "val_ap" if str(cfg.monitor) == "ap" else "val_f1"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CTR-GCN on window NPZs.")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--resume", default=None, help="Optional CTR-GCN checkpoint to initialize from.")
    ap.add_argument("--hard_neg_list", default=None, help="Optional train-only hard-negative window path list.")
    ap.add_argument("--hard_neg_mult", type=int, default=1)
    ap.add_argument("--allow_hard_neg_nontrain", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--min_epochs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=33724876)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--scheduler", choices=["plateau", "cosine", "onecycle"], default="plateau")
    ap.add_argument("--scheduler_metric", choices=["val_loss", "val_ap", "val_f1"], default=None)
    ap.add_argument("--scheduler_ema_beta", type=float, default=0.0)
    ap.add_argument("--lr_plateau_patience", type=int, default=10)
    ap.add_argument("--lr_plateau_factor", type=float, default=0.5)
    ap.add_argument("--lr_plateau_min_lr", type=float, default=1e-6)
    ap.add_argument("--max_lr", type=float, default=None)
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
    ap.add_argument("--two_stream", type=int, default=0)
    ap.add_argument("--stream_mode", type=str, default="joint_bone", choices=["joint_bone"])
    ap.add_argument("--fuse", type=str, default="concat", choices=["concat", "sum", "joint_only", "bone_only"])
    ap.add_argument("--stream_drop_joint_p", type=float, default=0.0)
    ap.add_argument("--stream_drop_bone_p", type=float, default=0.0)
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
    ap.add_argument("--use_ema", type=int, default=0)
    ap.add_argument("--ema_decay", type=float, default=0.995)
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
    two_stream = bool(int(cfg.two_stream))
    if two_stream and not feat_cfg.use_bone:
        raise SystemExit("[err] CTR-GCN --two_stream 1 with stream_mode=joint_bone requires --use_bone 1.")

    if cfg.loss != "focal" and cfg.balanced_sampler and str(cfg.pos_weight).lower() not in ("none", "0", "0.0", "false"):
        raise SystemExit("[err] choose ONE imbalance strategy for BCE: --balanced_sampler OR --pos_weight.")
    if cfg.loss == "focal" and cfg.balanced_sampler:
        print("[warn] loss=focal with --balanced_sampler can hurt probability calibration / AP.")

    extra_neg_files = None
    if cfg.hard_neg_list:
        with open(cfg.hard_neg_list, "r", encoding="utf-8") as f:
            extra_neg_files = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        _validate_hard_neg_paths(
            extra_neg_files,
            train_dir=cfg.train_dir,
            allow_nontrain=bool(int(cfg.allow_hard_neg_nontrain)),
        )
        print(f"[info] hard_neg_list: {cfg.hard_neg_list} (n={len(extra_neg_files)}) mult={cfg.hard_neg_mult}")

    train_ds = GraphWindowDataset(
        cfg.train_dir,
        split="train",
        feat_cfg=feat_cfg,
        fps_default=cfg.fps_default,
        skip_unlabeled=True,
        two_stream=two_stream,
        mask_joint_p=cfg.mask_joint_p,
        mask_frame_p=cfg.mask_frame_p,
        x_noise_std=cfg.x_noise_std,
        x_quant_step=cfg.x_quant_step,
        temporal_dropout_p=cfg.temporal_dropout_p,
        seed=cfg.seed,
        stream_mode=str(cfg.stream_mode),
        extra_neg_files=extra_neg_files,
        extra_neg_mult=int(cfg.hard_neg_mult),
    )
    val_ds = GraphWindowDataset(
        cfg.val_dir,
        split="val",
        feat_cfg=feat_cfg,
        fps_default=cfg.fps_default,
        skip_unlabeled=True,
        two_stream=two_stream,
        mask_joint_p=0.0,
        mask_frame_p=0.0,
        x_noise_std=0.0,
        x_quant_step=0.0,
        temporal_dropout_p=0.0,
        seed=cfg.seed,
        stream_mode=str(cfg.stream_mode),
    )
    train_loader = _make_loader(train_ds, cfg, train=True)
    val_loader = _make_loader(val_ds, cfg, train=False)

    if two_stream:
        sample_xj, sample_xb, _ = train_ds[0]
        num_joints = int(sample_xj.shape[1])
        model_cfg = CTRGCNConfig(
            num_joints=num_joints,
            channels=_parse_channel_schedule(cfg.channel_schedule),
            rel_channels=cfg.rel_channels,
            ctr_rank=cfg.ctr_rank,
            temporal_kernel=cfg.temporal_kernel,
            dropout=cfg.dropout,
            two_stream=True,
            stream_mode=str(cfg.stream_mode),
            fuse=str(cfg.fuse),
            stream_drop_joint_p=float(cfg.stream_drop_joint_p),
            stream_drop_bone_p=float(cfg.stream_drop_bone_p),
        )
        model_cfg_d: Dict[str, Any] = {
            **model_cfg.to_dict(),
            "in_feats_j": int(sample_xj.shape[-1]),
            "in_feats_b": int(sample_xb.shape[-1]),
        }
        model = build_model(
            "ctr_gcn",
            model_cfg_d,
            feat_cfg.to_dict(),
            num_joints=num_joints,
            in_feats_j=int(sample_xj.shape[-1]),
            in_feats_b=int(sample_xb.shape[-1]),
        ).to(device)
    else:
        sample_x, _ = train_ds[0]
        model_cfg = CTRGCNConfig(
            num_joints=int(sample_x.shape[1]),
            channels=_parse_channel_schedule(cfg.channel_schedule),
            rel_channels=cfg.rel_channels,
            ctr_rank=cfg.ctr_rank,
            temporal_kernel=cfg.temporal_kernel,
            dropout=cfg.dropout,
        )
        model_cfg_d = {**model_cfg.to_dict(), "in_feats": int(sample_x.shape[-1])}
        model = build_model("ctr_gcn", model_cfg_d, feat_cfg.to_dict()).to(device)

    resume_bundle: Dict[str, Any] | None = None
    if cfg.resume:
        resume_bundle = load_ckpt(cfg.resume, map_location="cpu")
        missing, unexpected = model.load_state_dict(resume_bundle["state_dict"], strict=False)
        if missing:
            print(f"[warn] resume: missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[warn] resume: unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        print(f"[info] resumed weights from: {cfg.resume}")

    ema = EMA(model, decay=float(cfg.ema_decay)) if int(cfg.use_ema) == 1 else None
    if ema is not None:
        print(f"[ema] enabled decay={cfg.ema_decay}")
        if resume_bundle is not None and isinstance(resume_bundle, dict) and "ema_state" in resume_bundle:
            try:
                ema.load_state_dict(resume_bundle["ema_state"])
                print("[info] resumed EMA state")
            except Exception as e:
                print(f"[warn] failed to load EMA state: {type(e).__name__}: {e}")

    criterion = _make_criterion(cfg, train_ds, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = bool(int(cfg.amp)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    sched_kind = str(cfg.scheduler).lower()
    sched_metric_name = _scheduler_metric_name(cfg)
    sched_mode = "min" if sched_metric_name == "val_loss" else "max"
    if sched_kind == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_mode,
            factor=float(cfg.lr_plateau_factor),
            patience=int(cfg.lr_plateau_patience),
            min_lr=float(cfg.lr_plateau_min_lr),
        )
    elif sched_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(cfg.epochs)),
            eta_min=float(cfg.lr_plateau_min_lr),
        )
    else:
        onecycle_max_lr = float(cfg.max_lr) if cfg.max_lr is not None else float(cfg.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=onecycle_max_lr,
            epochs=int(cfg.epochs),
            steps_per_epoch=max(1, len(train_loader)),
        )

    best_metric = float("-inf")
    best_bundle: Dict[str, Any] | None = None
    no_improve = 0
    scheduler_metric_ema: float | None = None

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            if two_stream:
                xj, xb, yb = batch
                xj = _to_f32(xj, device)
                xb = _to_f32(xb, device)
            else:
                xb, yb = batch
                xb = _to_f32(xb, device)
            yb = _to_f32(yb, device).view(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = logits_1d(model(xj, xb) if two_stream else model(xb))
                yb_loss = yb
                if cfg.label_smoothing > 0:
                    eps = float(cfg.label_smoothing)
                    yb_loss = yb * (1.0 - eps) + 0.5 * eps
                loss = criterion(logits, yb_loss)
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss; skipping step ep={epoch} loss={float(loss.detach().cpu()):.6g}")
                continue
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            if sched_kind == "onecycle":
                scheduler.step()
            if ema is not None:
                ema.update(model)
            losses.append(float(loss.detach().cpu()))

        eval_ctx = ema.use(model) if ema is not None else nullcontext()
        with eval_ctx:
            p_val, y_val = collect_probs(model, val_loader, device, two_stream=two_stream)
        val_loss = compute_loss_on_loader(model, val_loader, device, criterion, two_stream=two_stream, ema=ema)
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
        train_loss = float(np.mean(losses))
        metric, effective_monitor = _select_monitor_metric(cfg.monitor, ap_val, f1, train_loss)
        sched_metric_raw = float(val_loss if sched_metric_name == "val_loss" else (ap_val if sched_metric_name == "val_ap" else f1))
        if sched_kind == "plateau":
            beta = float(cfg.scheduler_ema_beta)
            if beta > 0.0:
                scheduler_metric_ema = (
                    sched_metric_raw
                    if scheduler_metric_ema is None
                    else (beta * scheduler_metric_ema + (1.0 - beta) * sched_metric_raw)
                )
                sched_metric_step = float(scheduler_metric_ema)
            else:
                sched_metric_step = sched_metric_raw
            if np.isfinite(sched_metric_step):
                scheduler.step(sched_metric_step)
        elif sched_kind == "cosine":
            scheduler.step()
        if effective_monitor != cfg.monitor:
            print(
                f"[warn] monitor={cfg.monitor} unavailable for checkpoint selection; "
                f"using {effective_monitor}"
            )
        print(
            f"[ep {epoch:03d}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} AP={ap_val:.4f} AUC={auc_val:.4f} F1={f1:.4f} "
            f"P={prec:.3f} R={rec:.3f} FPR={fpr:.3f} thr={thr:.3f} "
            f"lr={float(optimizer.param_groups[0]['lr']):.5g}"
        )

        if metric > best_metric:
            best_metric = metric
            no_improve = 0
            save_ctx = ema.use(model) if ema is not None else nullcontext()
            with save_ctx:
                best_bundle = {
                    "arch": "ctr_gcn",
                    "state_dict": model.state_dict(),
                    "model_cfg": model_cfg_d,
                    "feat_cfg": feat_cfg.to_dict(),
                    "data_cfg": build_data_cfg_dict(cfg.fps_default),
                    "train_cfg": asdict(cfg),
                    "meta": {
                        "epoch": epoch,
                        "best_metric": metric,
                        "monitor": cfg.monitor,
                        "effective_monitor": effective_monitor,
                        "scheduler_metric": sched_metric_name,
                        "ap": ap_val,
                        "auc": auc_val,
                        "f1": f1,
                        "precision": prec,
                        "recall": rec,
                        "fpr": fpr,
                        "thr": thr,
                    },
                    **({"ema_state": ema.state_dict()} if ema is not None else {}),
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
            "train_cfg": asdict(cfg),
            "meta": {"best_metric": best_metric, "seed": int(cfg.seed), "is_last": True},
            **({"ema_state": ema.state_dict()} if ema is not None else {}),
        },
    )


if __name__ == "__main__":
    main()
