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

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch

from core.ckpt import load_ckpt, get_cfg
from core.models import build_model, logits_1d, fuse_conv_bn_eval_inplace
from core.features import (
    FeatCfg,
    read_window_npz,
    build_canonical_input,
    split_gcn_two_stream,
    build_tcn_input,
)


@dataclass(frozen=True)
class WindowRaw:
    joints_xy: np.ndarray
    motion_xy: Optional[np.ndarray]
    conf: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    fps: float
    meta: Any  # WindowMeta


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
    # Optional CPU-only dynamic quantization for on-device latency/memory.
    # Disabled by default; enable with FD_DYNAMIC_QUANT_LINEAR=1.
    if str(os.getenv("FD_DYNAMIC_QUANT_LINEAR", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        try:
            if getattr(device, "type", "") == "cpu":
                model = torch.ao.quantization.quantize_dynamic(  # type: ignore[attr-defined]
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                model.eval()
        except Exception:
            pass
    # Runtime-only optimization: fuse Conv1d+BN pairs for lower inference latency.
    try:
        fuse_conv_bn_eval_inplace(model)
    except Exception:
        pass

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


def build_dual_inputs_from_raw(
    raw: WindowRaw,
    feat_cfg_tcn: FeatCfg,
    feat_cfg_gcn: FeatCfg,
    *,
    two_stream_gcn: bool = False,
):
    """Build (Xt, Xg, mask) for dual-mode inference.

    Fast path:
    - If TCN and GCN use the same feature config, canonicalize once and fan-out
      to both model inputs.
    """
    if feat_cfg_tcn == feat_cfg_gcn:
        X_can, m = build_canonical_input(
            joints_xy=raw.joints_xy,
            motion_xy=raw.motion_xy,
            conf=raw.conf,
            mask=raw.mask,
            fps=float(raw.fps),
            feat_cfg=feat_cfg_tcn,
        )
        Xt = build_tcn_input(X_can, feat_cfg_tcn)
        Xg = split_gcn_two_stream(X_can, feat_cfg_gcn) if two_stream_gcn else X_can
        return Xt, Xg, m

    Xt, m = build_input_from_raw(raw, feat_cfg_tcn, "tcn", two_stream=False)
    Xg, _ = build_input_from_raw(raw, feat_cfg_gcn, "gcn", two_stream=two_stream_gcn)
    return Xt, Xg, m


def compute_confirm_scores(
    raw: WindowRaw,
    *,
    conf_thr: float = 0.20,
    last_s: float = 0.7,
    min_valid_ratio: float = 0.25,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute (lying_score, motion_score) for a window.

    Safety guards:
    - If too few valid joints exist in the last segment, returns (None, None).
    - If hips/shoulders are missing, returns (None, None).

    lying_score in [0,1] (0 vertical → 1 horizontal)
    motion_score in [0,1] (lower = more still)
    """
    joints = np.asarray(raw.joints_xy, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[0] <= 1:
        return None, None

    T, V, _ = joints.shape

    # Visibility mask [T,V]
    if raw.mask is not None:
        m = np.asarray(raw.mask, dtype=bool)
    elif raw.conf is not None:
        m = np.asarray(raw.conf, dtype=np.float32) >= float(conf_thr)
    else:
        m = np.ones((T, V), dtype=bool)

    valid_ratio = m.sum(axis=1).astype(np.float32) / float(max(1, V))

    fps = float(raw.fps) if float(raw.fps) > 0 else 30.0
    last_n = int(max(1, round(float(last_s) * fps)))
    last_n = min(last_n, T)

    # If the tail of the window has too few valid joints, don't compute confirm scores.
    if float(np.mean(valid_ratio[-last_n:])) < float(min_valid_ratio):
        return None, None

    def mean_xy_and_ok(idxs):
        idxs = [int(i) for i in idxs if 0 <= int(i) < V]
        if not idxs:
            return None, None
        vm = m[:, idxs].astype(np.float32)          # [T,K]
        cnt = vm.sum(axis=1)                        # [T]
        sub = joints[:, idxs, :]                    # [T,K,2]
        s = (sub * vm[..., None]).sum(axis=1)       # [T,2]
        out = np.zeros((T, 2), dtype=np.float32)
        ok = cnt > 0
        out[ok] = s[ok] / cnt[ok, None]
        return out, ok

    # Hard-coded MediaPipe indices (assumption!):
    hip_mid, hip_ok = mean_xy_and_ok([23, 24])
    sh_mid, sh_ok = mean_xy_and_ok([11, 12])
    if hip_mid is None or sh_mid is None:
        return None, None

    torso_ok = hip_ok & sh_ok
    if int(torso_ok.sum()) < max(1, int(0.10 * T)):
        return None, None

    v = sh_mid - hip_mid
    n = np.linalg.norm(v, axis=1) + 1e-8
    vy = np.abs(v[:, 1]) / n
    lying_frame = np.clip(1.0 - vy, 0.0, 1.0).astype(np.float32)
    lying_frame[~torso_ok] = np.nan

    # Motion (prefer stored motion; else finite-diff)
    motion = None
    if raw.motion_xy is not None:
        mm = np.asarray(raw.motion_xy, dtype=np.float32)
        if mm.shape == joints.shape:
            motion = mm
    if motion is None:
        motion = np.zeros_like(joints, dtype=np.float32)
        motion[1:] = joints[1:] - joints[:-1]
        motion[0] = 0.0

    speed = np.linalg.norm(motion, axis=-1)         # [T,V]
    speed = speed * m.astype(np.float32)
    cnt_all = m.sum(axis=1).astype(np.float32) + 1e-6
    speed_mean = speed.sum(axis=1) / cnt_all        # [T]

    lying_score = float(np.nanmean(lying_frame[-last_n:]))
    if not np.isfinite(lying_score):
        return None, None

    peak = float(np.percentile(speed_mean, 90))
    tail = float(np.mean(speed_mean[-last_n:]))
    motion_score = float(np.clip(tail / (peak + 1e-6), 0.0, 1.0))
    return lying_score, motion_score


def _to_tensor(x: Any, device: torch.device) -> torch.Tensor:
    use_non_blocking = isinstance(device, torch.device) and device.type in {"cuda", "mps"}
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32, non_blocking=use_non_blocking).unsqueeze(0)
    if isinstance(x, np.ndarray) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]:
        arr = x
    else:
        arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    return torch.from_numpy(arr).unsqueeze(0).to(device=device, non_blocking=use_non_blocking)


def _make_forward_fn(
    model: torch.nn.Module,
    arch: str,
    X,
    device: torch.device,
    two_stream: bool,
):
    """Build a single callable returning probability tensor [B]."""
    arch = str(arch).lower()
    if arch == "tcn":
        xb = _to_tensor(X, device)

        def forward_fn() -> torch.Tensor:
            logits = logits_1d(model(xb))
            return torch.sigmoid(logits).view(-1)

        return forward_fn

    if arch == "gcn":
        if two_stream:
            xj = _to_tensor(X[0], device)
            xm = _to_tensor(X[1], device)

            def forward_fn() -> torch.Tensor:
                logits = logits_1d(model(xj, xm))
                return torch.sigmoid(logits).view(-1)

            return forward_fn

        xb = _to_tensor(X, device)

        def forward_fn() -> torch.Tensor:
            logits = logits_1d(model(xb))
            return torch.sigmoid(logits).view(-1)

        return forward_fn

    raise ValueError(f"Unknown arch: {arch}")


@torch.inference_mode()
def predict_prob(model: torch.nn.Module, arch: str, X, device: torch.device, two_stream: bool) -> float:
    arch = str(arch).lower()
    if arch == "tcn":
        xb = _to_tensor(X, device)
        logits = logits_1d(model(xb))
        return float(torch.sigmoid(logits).view(-1)[0].item())

    if arch == "gcn":
        if two_stream:
            xj = _to_tensor(X[0], device)
            xm = _to_tensor(X[1], device)
            logits = logits_1d(model(xj, xm))
            return float(torch.sigmoid(logits).view(-1)[0].item())

        xb = _to_tensor(X, device)
        logits = logits_1d(model(xb))
        return float(torch.sigmoid(logits).view(-1)[0].item())

    raise ValueError(f"Unknown arch: {arch}")


def predict_mu_sigma(
    model: torch.nn.Module,
    arch: str,
    X,
    device: torch.device,
    two_stream: bool,
    M: int,
) -> Tuple[float, float]:
    """MC Dropout mean/std (dropout-only sampling, BN kept in eval mode)."""
    from core.uncertainty import mc_predict_mu_sigma

    forward_fn = _make_forward_fn(model=model, arch=arch, X=X, device=device, two_stream=two_stream)
    mu_t, sig_t = mc_predict_mu_sigma(model, forward_fn=forward_fn, M=int(M))
    return float(mu_t.detach().cpu().view(-1)[0].item()), float(sig_t.detach().cpu().view(-1)[0].item())
