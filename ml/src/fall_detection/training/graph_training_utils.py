#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared graph-model training utilities.

This module contains the window dataset, augmentation, hard-negative guard, and
evaluation helpers used by CTR-GCN training. It deliberately has no standalone
trainer entry point, keeping experiment launch code separate from reusable data
and optimization utilities.
"""

from __future__ import annotations

import glob
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from fall_detection.core.features import (
    FeatCfg,
    build_canonical_input,
    read_window_npz,
    split_ctr_gcn_two_stream,
)
from fall_detection.core.models import logits_1d


def _to_f32(x: Any, device: torch.device) -> torch.Tensor:
    """Convert NumPy or torch inputs into float32 tensors on the target device."""
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def set_seed(seed: int, *, deterministic: int = 1) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if bool(int(deterministic)):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def compute_pos_weight(labels01: np.ndarray) -> torch.Tensor:
    """Compute BCE positive-class weighting from binary training labels."""
    y = np.asarray(labels01).astype(int).reshape(-1)
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    return torch.tensor([neg / pos], dtype=torch.float32)


def make_balanced_sampler(y01: np.ndarray) -> WeightedRandomSampler:
    """Build a replacement sampler that equalizes positive/negative sampling mass."""
    y = np.asarray(y01).reshape(-1)
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    w_pos = 1.0 / pos
    w_neg = 1.0 / neg
    w = np.where(y == 1, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(w), replacement=True)


def augment_mask(mask: np.ndarray, rng: np.random.Generator, mask_joint_p: float, mask_frame_p: float) -> np.ndarray:
    """Drop random joints/frames while preserving at least one valid pose entry."""
    m = np.asarray(mask).copy().astype(bool)
    t, v = m.shape
    if mask_joint_p > 0:
        drop_j = rng.random(v) < float(mask_joint_p)
        if drop_j.any():
            m[:, drop_j] = False
    if mask_frame_p > 0:
        drop_t = rng.random(t) < float(mask_frame_p)
        if drop_t.any():
            m[drop_t, :] = False
    if not m.any():
        m[int(rng.integers(0, t)), int(rng.integers(0, v))] = True
    return m


def augment_feature_tensor(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    x_noise_std: float,
    x_quant_step: float,
    temporal_dropout_p: float,
) -> np.ndarray:
    """Apply lightweight robustness augmentation on canonical feature tensor [T,V,C]."""
    out = np.asarray(x, dtype=np.float32).copy()
    t, _, _ = out.shape
    if temporal_dropout_p > 0:
        drop_t = rng.random(t) < float(temporal_dropout_p)
        if drop_t.any():
            out[drop_t, :, :] = 0.0
    if x_noise_std > 0:
        out += rng.normal(0.0, float(x_noise_std), size=out.shape).astype(np.float32)
    if x_quant_step > 0:
        step = float(x_quant_step)
        out = np.round(out / step) * step
    return out


def list_npz_files(root: str) -> List[str]:
    files = glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)
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


class GraphWindowDataset(Dataset):
    """Window dataset that rebuilds canonical or CTR-GCN two-stream features."""

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
        x_noise_std: float,
        x_quant_step: float,
        temporal_dropout_p: float,
        seed: int,
        stream_mode: str = "joint_bone",
        extra_neg_files: Optional[List[str]] = None,
        extra_neg_mult: int = 1,
    ) -> None:
        self.root = str(root)
        self.split = str(split)
        self.feat_cfg = feat_cfg
        self.fps_default = float(fps_default)
        self.skip_unlabeled = bool(skip_unlabeled)
        self.two_stream = bool(two_stream)
        self.stream_mode = str(stream_mode)
        self.mask_joint_p = float(mask_joint_p)
        self.mask_frame_p = float(mask_frame_p)
        self.x_noise_std = float(x_noise_std)
        self.x_quant_step = float(x_quant_step)
        self.temporal_dropout_p = float(temporal_dropout_p)

        files = list_npz_files(self.root)
        self.files: List[str] = []
        self.labels01: List[int] = []

        fail = 0
        examples: List[str] = []
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

        if extra_neg_files:
            mult = max(1, int(extra_neg_mult))
            for fp in list(extra_neg_files) * mult:
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
                if int(meta.y) == 1:
                    continue
                self.files.append(fp)
                self.labels01.append(0)

        if fail:
            print(f"[warn] skipped {fail} unreadable windows under: {self.root}")
            for ex in examples:
                print(f"[warn]   example: {ex}")

        if not self.files:
            raise RuntimeError(
                f"[err] no readable windows under: {self.root}. "
                f"found={len(files)} failed_reads={fail}. "
                "Check window output and key/dtype consistency."
            )

        self.labels01 = np.asarray(self.labels01, dtype=np.int64)
        base = int(seed) + (11 if split == "train" else 22)
        self.rng = np.random.default_rng(base)
        self._missing_warned: set[str] = set()

    def __len__(self) -> int:
        return len(self.files)

    def _read_window_with_fallback(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Any]:
        """Read one window, probing nearby indices if files disappeared concurrently."""
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
            f"[err] unable to read nearby window files around index={i} "
            f"(probed={max_probe}). Check for concurrent cleanup under: {self.root}"
        )

    def __getitem__(self, i: int):
        joints, motion, conf, mask, fps, meta = self._read_window_with_fallback(i)
        x, mask_used = build_canonical_input(
            joints_xy=joints,
            motion_xy=motion,
            conf=conf,
            mask=mask,
            fps=fps,
            feat_cfg=self.feat_cfg,
        )

        if self.split == "train" and (self.mask_joint_p > 0 or self.mask_frame_p > 0):
            m_aug = augment_mask(mask_used, self.rng, self.mask_joint_p, self.mask_frame_p)
            x = x * m_aug[..., None]
        if self.split == "train" and (
            self.x_noise_std > 0 or self.x_quant_step > 0 or self.temporal_dropout_p > 0
        ):
            x = augment_feature_tensor(
                x,
                self.rng,
                x_noise_std=self.x_noise_std,
                x_quant_step=self.x_quant_step,
                temporal_dropout_p=self.temporal_dropout_p,
            )

        y = 1 if int(meta.y) == 1 else 0
        if not self.two_stream:
            return torch.as_tensor(x, dtype=torch.float32), torch.as_tensor([y], dtype=torch.float32)

        if not self.feat_cfg.use_bone:
            raise RuntimeError("CTR-GCN two_stream requires feat_cfg.use_bone=1")
        xj, xb = split_ctr_gcn_two_stream(x, self.feat_cfg, stream_mode=self.stream_mode)
        return (
            torch.as_tensor(xj, dtype=torch.float32),
            torch.as_tensor(xb, dtype=torch.float32),
            torch.as_tensor([y], dtype=torch.float32),
        )


def collect_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    two_stream: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect validation probabilities in the same tensor layout used for training."""
    model.eval()
    ps: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if two_stream:
                xj, xb, yb = batch
                logits = logits_1d(model(_to_f32(xj, device), _to_f32(xb, device)))
            else:
                xb, yb = batch
                logits = logits_1d(model(_to_f32(xb, device)))
            yb_t = _to_f32(yb, device).view(-1)
            ps.append(torch.sigmoid(logits).detach().cpu().numpy())
            ys.append(yb_t.detach().cpu().numpy().reshape(-1))
    return np.concatenate(ps, axis=0), np.concatenate(ys, axis=0)


def compute_loss_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    two_stream: bool = False,
    ema: Optional[Any] = None,
) -> float:
    """Compute mean loss on a loader with model in eval mode."""
    model.eval()
    losses: List[float] = []
    counts: List[int] = []
    ctx = ema.use(model) if ema is not None else nullcontext()
    with ctx, torch.no_grad():
        for batch in loader:
            if two_stream:
                xj, xb, yb = batch
                xj_t = _to_f32(xj, device)
                xb_t = _to_f32(xb, device)
                logits = logits_1d(model(xj_t, xb_t))
                bsz = int(xj_t.shape[0])
            else:
                xb, yb = batch
                xb_t = _to_f32(xb, device)
                logits = logits_1d(model(xb_t))
                bsz = int(xb_t.shape[0])
            yb_t = _to_f32(yb, device).view(-1)
            loss = criterion(logits, yb_t).detach()
            losses.append(float(loss.cpu()) * bsz)
            counts.append(bsz)
    return float(sum(losses) / max(1, sum(counts)))


def build_data_cfg_dict(fps_default: float) -> Dict[str, Any]:
    """Persist the minimal data facts later eval code can trust from training."""
    return {"fps_default": float(fps_default)}
