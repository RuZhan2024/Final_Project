#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deploy/common.py

Shared helpers for deploy-time runners.

Safety + parity rules:
- Build canonical features with core.features.build_canonical_input (same as eval/training).
- For TCN, flatten canonical X via core.features.build_tcn_input.
- Confirm scores (lying + motion) must NOT "confirm" on empty/missing skeletons.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .confirm import WindowRaw, compute_confirm_scores
from fall_detection.core.ckpt import load_ckpt, get_cfg
from fall_detection.core.models import build_model, logits_1d
from fall_detection.core.uncertainty import mc_predict_mu_sigma
from fall_detection.core.features import (
    FeatCfg,
    read_window_npz,
    build_canonical_input,
    split_gcn_two_stream,
    build_tcn_input,
)


def load_model_bundle(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, str, dict, FeatCfg, bool, float]:
    """Load ckpt and rebuild (model + FeatCfg).

    Returns: (model, arch, model_cfg, feat_cfg, two_stream, fps_default)

    NOTE: strict=True prevents silently running the wrong weights.
    """
    b = load_ckpt(ckpt_path, map_location="cpu")
    arch = str(b.get("arch", "unknown")).lower()
    model_cfg = get_cfg(b, "model_cfg", {}) or {}
    feat_cfg_d = get_cfg(b, "feat_cfg", {}) or {}
    data_cfg = get_cfg(b, "data_cfg", {}) or {}

    fps_default = float(data_cfg.get("fps_default", 30.0))
    two_stream = bool(model_cfg.get("two_stream", False))

    model = build_model(arch=arch, model_cfg=model_cfg, feat_cfg=feat_cfg_d, fps_default=fps_default)

    sd = b.get("state_dict", b.get("model", None))
    if not isinstance(sd, dict):
        raise ValueError(f"Checkpoint missing state_dict: {ckpt_path}")

    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    return model, arch, model_cfg, FeatCfg.from_dict(feat_cfg_d), two_stream, fps_default


def load_window_raw(path: str, fps_default: float = 30.0) -> WindowRaw:
    joints_xy, motion_xy, conf, mask, fps, meta = read_window_npz(path, fps_default=fps_default)
    return WindowRaw(
        joints_xy=joints_xy,
        motion_xy=motion_xy,
        conf=conf,
        mask=mask,
        fps=float(fps),
        meta=meta,
    )


def build_input_from_raw(raw: WindowRaw, feat_cfg: FeatCfg, arch: str, *, two_stream: bool = False):
    """Build model input from WindowRaw with training/eval parity.

    - Always build canonical X[T,V,F] first.
    - TCN gets flattened x[T, V*F].
    - GCN gets X[T,V,F] or (xj,xm) for two-stream.
    """
    arch = str(arch).lower()

    X_can, m = build_canonical_input(
        joints_xy=raw.joints_xy,
        motion_xy=raw.motion_xy,
        conf=raw.conf,
        mask=raw.mask,
        fps=float(raw.fps),
        feat_cfg=feat_cfg,
    )

    if arch == "tcn":
        x = build_tcn_input(X_can, feat_cfg)
        return x, m

    if arch == "gcn":
        X = X_can
        if two_stream:
            X = split_gcn_two_stream(X_can, feat_cfg)
        return X, m

    raise ValueError(f"Unknown arch: {arch}")


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32))[None, ...].to(device)


@torch.inference_mode()
def predict_prob(model: torch.nn.Module, arch: str, X, device: torch.device, two_stream: bool) -> float:
    arch = str(arch).lower()
    if arch == "tcn":
        xb = _to_tensor(X, device)
        logits = logits_1d(model(xb))
    elif arch == "gcn":
        if two_stream:
            xj = _to_tensor(X[0], device)
            xm = _to_tensor(X[1], device)
            logits = logits_1d(model(xj, xm))
        else:
            xb = _to_tensor(X, device)
            logits = logits_1d(model(xb))
    else:
        raise ValueError(f"Unknown arch: {arch}")

    p = torch.sigmoid(logits).detach().float().cpu().numpy().reshape(-1)[0]
    return float(p)


def predict_mu_sigma(
    model: torch.nn.Module,
    arch: str,
    X,
    device: torch.device,
    two_stream: bool,
    M: int,
) -> Tuple[float, float]:
    """MC Dropout mean/std with BatchNorm kept in eval mode."""
    arch = str(arch).lower()

    def forward_one() -> torch.Tensor:
        if arch == "tcn":
            xb = _to_tensor(X, device)
            logits = logits_1d(model(xb))
        elif arch == "gcn":
            if two_stream:
                xj = _to_tensor(X[0], device)
                xm = _to_tensor(X[1], device)
                logits = logits_1d(model(xj, xm))
            else:
                xb = _to_tensor(X, device)
                logits = logits_1d(model(xb))
        else:
            raise ValueError(f"Unknown arch: {arch}")
        return torch.sigmoid(logits)

    # Reuse the shared uncertainty helper so deploy-time sampling matches the
    # same dropout-only contract used elsewhere in the project.
    mu, sigma = mc_predict_mu_sigma(model, forward_one, M=int(M))
    mu_f = float(mu.detach().float().cpu().numpy().reshape(-1)[0])
    sigma_f = float(sigma.detach().float().cpu().numpy().reshape(-1)[0])
    return mu_f, sigma_f
