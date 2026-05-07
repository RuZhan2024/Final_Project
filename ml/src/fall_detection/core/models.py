#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/models.py

Model builders (TCN + CTR-GCN) used by both training and evaluation.

Key goals for this project:
- Training / fit_ops / metrics must rebuild *exactly* the same model from a checkpoint bundle.
- Avoid brittle signature mismatches across versions.

This module therefore provides a `build_model()` that is:
- Backward compatible with older call sites that pass explicit input dims (in_ch / in_feats...).
- Forward compatible with newer call sites that pass (arch, model_cfg, feat_cfg, fps_default=...).

All models return logits of shape (B,) (recommended) or (B,1) which we normalise in helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .ctr_gcn import CTRGCN, CTRGCNConfig, TwoStreamCTRGCN

# ---------------------------
# Config normalisation
# ---------------------------

def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    """Return a plain dict for cfg values that may be dicts, dataclasses, or simple objects.

    This project sometimes stores feat_cfg as a dataclass-like object in checkpoints.
    Evaluation code may pass that object back into build_model(); we normalise it here.
    """
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    # dataclass instance
    try:
        if is_dataclass(cfg):
            return asdict(cfg)  # type: ignore[arg-type]
    except Exception:
        pass
    # generic object with attributes
    try:
        return dict(vars(cfg))
    except Exception:
        return {}



# ---------------------------
# Convenience helpers
# ---------------------------

def pick_device() -> torch.device:
    """Choose the best available torch device with a stable repo preference order."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def logits_1d(logits: torch.Tensor) -> torch.Tensor:
    """Normalize logits to shape (B,) for downstream code."""
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    if logits.ndim == 1:
        return logits
    # tolerate accidental extra dims
    return logits.view(logits.shape[0], -1)[:, 0]


def p_fall_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert model logits to fall probabilities using the repo's 1D convention."""
    return torch.sigmoid(logits_1d(logits))


# ---------------------------
# TCN
# ---------------------------

class TemporalShift1D(nn.Module):
    """Channel-wise temporal shift for [B,C,T] tensors with no extra parameters."""

    def __init__(self, fold_div: int = 8):
        super().__init__()
        self.fold_div = max(2, int(fold_div))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T]
        if x.ndim != 3:
            return x
        b, c, t = x.shape
        fold = c // self.fold_div
        if fold <= 0 or t <= 1:
            return x
        out = x.clone()
        # shift first fold left (use previous time), second fold right (use next time)
        out[:, :fold, :-1] = x[:, :fold, 1:]
        out[:, :fold, -1] = 0.0
        out[:, fold : 2 * fold, 1:] = x[:, fold : 2 * fold, :-1]
        out[:, fold : 2 * fold, 0] = 0.0
        return out


class ResTCNBlock(nn.Module):
    def __init__(
        self,
        ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.30,
        use_tsm: bool = False,
        tsm_fold_div: int = 8,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.tsm = TemporalShift1D(fold_div=tsm_fold_div) if bool(use_tsm) else nn.Identity()
        self.conv = nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.tsm(x)
        y = self.drop(self.act(self.bn(self.conv(y))))
        return x + y


class TCN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden: int = 128,
        dropout: float = 0.30,
        num_blocks: int = 4,
        kernel: int = 3,
        use_tsm: bool = False,
        tsm_fold_div: int = 8,
    ):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        dilations = [2 ** i for i in range(max(1, int(num_blocks)))]
        self.blocks = nn.ModuleList(
            [
                ResTCNBlock(
                    hidden,
                    kernel_size=kernel,
                    dilation=d,
                    dropout=dropout,
                    use_tsm=bool(use_tsm),
                    tsm_fold_div=int(tsm_fold_div),
                )
                for d in dilations
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TCN expects flattened-per-frame features; the transpose converts from
        # the repo-standard [B,T,C] feature layout to Conv1d's [B,C,T].
        x = x.transpose(1, 2)
        x = self.conv_in(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


@dataclass
class TCNConfig:
    """Training/runtime knobs for the flattened-joint TCN classifier.

    This config owns architecture shape only. Input dimensionality stays outside
    the dataclass because it must match the saved feature contract exactly and
    may be injected explicitly by older checkpoints/evaluation entry points.
    """
    hidden: int = 128
    dropout: float = 0.30
    num_blocks: int = 4
    kernel: int = 3
    use_tsm: bool = False
    tsm_fold_div: int = 8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TCNConfig":
        if not isinstance(d, dict):
            return TCNConfig()
        return TCNConfig(
            hidden=int(d.get("hidden", d.get("hid", 128))),
            dropout=float(d.get("dropout", d.get("p", 0.30))),
            num_blocks=int(d.get("num_blocks", 4)),
            kernel=int(d.get("kernel", 3)),
            use_tsm=bool(d.get("use_tsm", False)),
            tsm_fold_div=int(d.get("tsm_fold_div", 8)),
        )


# ---------------------------
# Input-dimension inference
# ---------------------------

def _bool(d: Dict[str, Any], key: str, default: bool = False) -> bool:
    """Parse permissive bool-like config values from YAML/checkpoint dicts."""
    v = d.get(key, default)
    # tolerate "0"/"1" and ints
    if isinstance(v, str):
        v = v.strip().lower()
        if v in ("1", "true", "yes", "y", "t"):
            return True
        if v in ("0", "false", "no", "n", "f"):
            return False
    return bool(v)


def infer_input_dims(
    arch: str,
    model_cfg: Dict[str, Any],
    feat_cfg: Optional[Dict[str, Any]] = None,
    *,
    num_joints_default: int = 33,
) -> Dict[str, int]:
    """Infer the input dimensions required to build the model.

    Preference order:
    1) Explicit dims stored in model_cfg (best, because it matches training exactly).
    2) Infer from feat_cfg (reasonable default for this project).

    The returned dimensions must be treated as part of the checkpoint contract:
    once a model is trained, later evaluation code should prefer serialized
    dimensions over recomputing them from today's feature defaults.
    """
    arch = str(arch).lower()
    model_cfg = _cfg_to_dict(model_cfg)
    feat_cfg = _cfg_to_dict(feat_cfg)

    num_joints = int(model_cfg.get("num_joints", feat_cfg.get("num_joints", num_joints_default)))

    # If training stored explicit dims, always prefer them.
    out: Dict[str, int] = {"num_joints": num_joints}
    if "in_ch" in model_cfg:
        out["in_ch"] = int(model_cfg["in_ch"])
    if "in_feats" in model_cfg:
        out["in_feats"] = int(model_cfg["in_feats"])
    if "in_feats_j" in model_cfg:
        out["in_feats_j"] = int(model_cfg["in_feats_j"])
    if "in_feats_b" in model_cfg:
        out["in_feats_b"] = int(model_cfg["in_feats_b"])

    if arch == "tcn" and "in_ch" in out:
        return out
    if arch == "ctr_gcn":
        cfg = CTRGCNConfig.from_dict(model_cfg)
        if cfg.two_stream and "in_feats_j" in out and "in_feats_b" in out:
            return out
        if (not cfg.two_stream) and "in_feats" in out:
            return out

    # Otherwise infer from feature flags (project defaults).
    use_motion = _bool(feat_cfg, "use_motion", default=False)
    use_conf = _bool(feat_cfg, "use_conf_channel", default=False)
    use_bone = _bool(feat_cfg, "use_bone", default=False)
    use_bone_len = _bool(feat_cfg, "use_bone_length", default=False)
    # Note: use_precomputed_mask affects masking, not feature dimensionality.

    # Per-joint features for a single-stream representation
    per_joint = 2  # (x,y)
    if use_motion:
        per_joint += 2  # (dx,dy)
    if use_bone:
        per_joint += 2  # (bx,by)
    if use_bone_len:
        per_joint += 1  # (bone_len)
    if use_conf:
        per_joint += 1  # (conf)

    if arch == "tcn":
        # TCN takes flattened joints: [B,T, J * per_joint]
        out["in_ch"] = int(num_joints * per_joint)
        return out

    if arch == "ctr_gcn":
        cfg = CTRGCNConfig.from_dict(model_cfg)
        if cfg.two_stream:
            in_feats_j = 2
            if use_motion:
                in_feats_j += 2
            if use_conf:
                in_feats_j += 1
            in_feats_b = 2
            if use_bone_len:
                in_feats_b += 1
            if use_conf:
                in_feats_b += 1
            out["in_feats_j"] = int(in_feats_j)
            out["in_feats_b"] = int(in_feats_b)
        else:
            out["in_feats"] = int(per_joint)
        return out

    raise ValueError(f"Unknown arch: {arch}")


# ---------------------------
# Model builder (compatibility-focused)
# ---------------------------

def build_model(
    arch: str,
    model_cfg: Dict[str, Any],
    feat_cfg: Optional[Dict[str, Any]] = None,
    *,
    fps_default: Optional[float] = None,
    in_ch: int = 0,
    num_joints: int = 33,
    in_feats: int = 0,
    in_feats_j: int = 0,
    in_feats_b: int = 0,
    **kwargs: Any,
) -> nn.Module:
    """Build a model for training/evaluation.

    Supported call styles:

    New (preferred; used by eval/fit_ops.py):
        build_model(arch, model_cfg, feat_cfg, fps_default=...)

    Old (explicit dims):
        build_model(arch, model_cfg, in_ch=..., in_feats=..., ...)

    Notes:
    - ``fps_default`` is accepted for compatibility; architectures here do not
      consume it directly.
    - Dimension resolution prefers explicit caller args, then serialized
      checkpoint fields, then feature-config inference. That order preserves old
      checkpoints whose runtime feature defaults may no longer match current code.
    """
    arch = str(arch).lower()
    model_cfg = _cfg_to_dict(model_cfg)
    feat_cfg = _cfg_to_dict(feat_cfg)

    # Only infer dimensions when the caller did not supply them explicitly.
    # This preserves historical checkpoints whose stored feature contract should
    # override today's inferred defaults.
    if arch in ("tcn", "ctr_gcn"):
        inferred = infer_input_dims(arch, model_cfg, feat_cfg, num_joints_default=num_joints)
        num_joints = int(inferred.get("num_joints", num_joints))
        if arch == "tcn":
            if not in_ch:
                in_ch = int(inferred.get("in_ch", 0))
        else:
            if not in_feats:
                in_feats = int(inferred.get("in_feats", 0))
            if not in_feats_j:
                in_feats_j = int(inferred.get("in_feats_j", 0))
            if not in_feats_b:
                in_feats_b = int(inferred.get("in_feats_b", 0))

    if arch == "tcn":
        if not in_ch or in_ch <= 0:
            raise ValueError(
                "TCN requires in_ch > 0. "
                "Store in_ch in model_cfg when saving checkpoints, or provide a valid feat_cfg."
            )
        cfg = TCNConfig.from_dict(model_cfg)
        return TCN(
            in_ch=int(in_ch),
            hidden=cfg.hidden,
            dropout=cfg.dropout,
            num_blocks=cfg.num_blocks,
            kernel=cfg.kernel,
            use_tsm=cfg.use_tsm,
            tsm_fold_div=cfg.tsm_fold_div,
        )

    if arch == "ctr_gcn":
        cfg = CTRGCNConfig.from_dict(model_cfg)
        if cfg.two_stream:
            if not in_feats_j or not in_feats_b:
                raise ValueError(
                    "Two-stream CTR-GCN requires in_feats_j and in_feats_b. "
                    "Store them in model_cfg when saving checkpoints, or provide a valid feat_cfg."
                )
            return TwoStreamCTRGCN(
                num_joints=int(num_joints),
                in_feats_j=int(in_feats_j),
                in_feats_b=int(in_feats_b),
                channels=tuple(cfg.channels),
                rel_channels=cfg.rel_channels,
                ctr_rank=cfg.ctr_rank,
                temporal_kernel=cfg.temporal_kernel,
                dropout=cfg.dropout,
                fuse=cfg.fuse,
                stream_drop_joint_p=cfg.stream_drop_joint_p,
                stream_drop_bone_p=cfg.stream_drop_bone_p,
            )
        if not in_feats or in_feats <= 0:
            raise ValueError(
                "CTR-GCN requires in_feats > 0. "
                "Store in_feats in model_cfg when saving checkpoints, or provide a valid feat_cfg."
            )
        return CTRGCN(
            num_joints=int(num_joints),
            in_feats=int(in_feats),
            channels=tuple(cfg.channels),
            rel_channels=cfg.rel_channels,
            ctr_rank=cfg.ctr_rank,
            temporal_kernel=cfg.temporal_kernel,
            dropout=cfg.dropout,
        )

    raise ValueError(f"Unknown arch: {arch}")
