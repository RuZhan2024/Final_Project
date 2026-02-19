#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/models.py

Model builders (TCN + GCN) used by both training and evaluation.

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
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

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
    return torch.sigmoid(logits_1d(logits))


# ---------------------------
# TCN
# ---------------------------

class ResTCNBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.30):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act(self.bn(self.conv(x))))
        return x + y


class TCN(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 128, dropout: float = 0.30, num_blocks: int = 4, kernel: int = 3):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        dilations = [2 ** i for i in range(max(1, int(num_blocks)))]
        self.blocks = nn.ModuleList([ResTCNBlock(hidden, kernel_size=kernel, dilation=d, dropout=dropout) for d in dilations])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C] -> [B,C,T]
        x = x.transpose(1, 2)
        x = self.conv_in(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


@dataclass
class TCNConfig:
    hidden: int = 128
    dropout: float = 0.30
    num_blocks: int = 4
    kernel: int = 3

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
        )


# ---------------------------
# GCN (+ optional two-stream)
# ---------------------------

def build_mediapipe_adjacency(num_joints: int = 33) -> np.ndarray:
    # Undirected edges (reasonable subset) for MediaPipe Pose 33.
    edges = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
        (15, 17), (15, 19), (15, 21),
        (16, 18), (16, 20), (16, 22),
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (0, 9), (0, 10),
    ]
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        if 0 <= i < num_joints and 0 <= j < num_joints:
            A[i, j] = 1.0
            A[j, i] = 1.0
    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32)
    I = np.eye(A.shape[0], dtype=np.float32)
    A_hat = A + I
    D = np.sum(A_hat, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-8)))
    return (D_inv_sqrt @ A_hat @ D_inv_sqrt).astype(np.float32)


class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        mid = max(4, ch // r)
        self.fc1 = nn.Linear(ch, mid)
        self.fc2 = nn.Linear(mid, ch)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,V,C]
        b, t, v, c = x.shape
        z = x.mean(dim=(1, 2))          # [B,C]
        z = self.relu(self.fc1(z))
        s = self.sig(self.fc2(z)).view(b, 1, 1, c)
        return x * s


class GCNLayer(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)
        self.ln = nn.LayerNorm(out_feats)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # x: [B,T,V,C]
        # x: [B,T,V,C], A_hat: [V,V]
        # Use matmul instead of einsum for better backend stability (e.g., MPS).
        x = x.permute(0, 1, 3, 2)              # [B,T,C,V]
        x = torch.matmul(x, A_hat.t())         # [B,T,C,V]
        x = x.permute(0, 1, 3, 2)              # [B,T,V,C]
        x = self.lin(x)
        x = self.drop(self.act(self.ln(x)))
        return x


class TemporalConv(nn.Module):
    def __init__(self, ch: int, hidden: int, dropout: float = 0.30, num_blocks: int = 3):
        super().__init__()
        self.proj = nn.Linear(ch, hidden)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.blocks = nn.ModuleList()
        for i in range(max(1, int(num_blocks))):
            d = 2 ** i
            pad = d
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=pad, dilation=d),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C]
        x = self.drop(self.act(self.proj(x)))  # [B,T,H]
        x = x.transpose(1, 2)                  # [B,H,T]
        for blk in self.blocks:
            x = x + blk(x)
        x = x.mean(dim=-1)                     # [B,H]
        return x


class GCNEncoder(nn.Module):
    def __init__(self, num_joints: int, in_feats: int, gcn_hidden: int, tcn_hidden: int, dropout: float, use_se: bool):
        super().__init__()
        A_hat = normalize_adjacency(build_mediapipe_adjacency(num_joints))
        self.register_buffer("A_hat", torch.from_numpy(A_hat))

        self.g1 = GCNLayer(in_feats, gcn_hidden, dropout=dropout)
        self.g2 = GCNLayer(gcn_hidden, gcn_hidden, dropout=dropout)
        self.se = SEBlock(gcn_hidden) if use_se else nn.Identity()
        self.temporal = TemporalConv(gcn_hidden, tcn_hidden, dropout=dropout, num_blocks=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,V,F]
        A_hat = self.A_hat
        x = self.g1(x, A_hat)
        x = self.g2(x, A_hat)
        x = self.se(x)
        x = x.mean(dim=2)  # mean over joints -> [B,T,C]
        return self.temporal(x)  # [B,E]


class GCN(nn.Module):
    def __init__(
        self,
        num_joints: int,
        in_feats: int,
        gcn_hidden: int = 96,
        tcn_hidden: int = 192,
        dropout: float = 0.35,
        use_se: bool = True,
    ):
        super().__init__()
        self.encoder = GCNEncoder(num_joints, in_feats, gcn_hidden, tcn_hidden, dropout, use_se)
        self.head = nn.Linear(tcn_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z).squeeze(-1)


class TwoStreamGCN(nn.Module):
    def __init__(
        self,
        num_joints: int,
        in_feats_j: int,
        in_feats_m: int,
        gcn_hidden: int = 96,
        tcn_hidden: int = 192,
        dropout: float = 0.35,
        use_se: bool = True,
        fuse: str = "concat",   # "concat" | "sum"
    ):
        super().__init__()
        self.j_enc = GCNEncoder(num_joints, in_feats_j, gcn_hidden, tcn_hidden, dropout, use_se)
        self.m_enc = GCNEncoder(num_joints, in_feats_m, gcn_hidden, tcn_hidden, dropout, use_se)
        self.fuse = str(fuse).lower()
        out_dim = (2 * tcn_hidden) if self.fuse == "concat" else tcn_hidden
        self.head = nn.Linear(out_dim, 1)

    def forward(self, xj: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        zj = self.j_enc(xj)
        zm = self.m_enc(xm)
        if self.fuse == "sum":
            z = zj + zm
        else:
            z = torch.cat([zj, zm], dim=-1)
        return self.head(z).squeeze(-1)


@dataclass
class GCNConfig:
    num_joints: int = 33
    gcn_hidden: int = 96
    tcn_hidden: int = 192
    dropout: float = 0.35
    use_se: bool = True
    two_stream: bool = False
    fuse: str = "concat"  # concat|sum

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GCNConfig":
        if not isinstance(d, dict):
            return GCNConfig()
        return GCNConfig(
            num_joints=int(d.get("num_joints", 33)),
            gcn_hidden=int(d.get("gcn_hidden", 96)),
            tcn_hidden=int(d.get("tcn_hidden", 192)),
            dropout=float(d.get("dropout", 0.35)),
            use_se=bool(d.get("use_se", True)),
            two_stream=bool(d.get("two_stream", False)),
            fuse=str(d.get("fuse", "concat")),
        )


# ---------------------------
# Input-dimension inference
# ---------------------------

def _bool(d: Dict[str, Any], key: str, default: bool = False) -> bool:
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
    if "in_feats_m" in model_cfg:
        out["in_feats_m"] = int(model_cfg["in_feats_m"])

    if arch == "tcn" and "in_ch" in out:
        return out
    if arch == "gcn" and (("in_feats" in out) or ("in_feats_j" in out and "in_feats_m" in out)):
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

    if arch == "gcn":
        cfg = GCNConfig.from_dict(model_cfg)
        if cfg.two_stream:
            # Two-stream default in *this project*:
            # - joints stream uses (x,y[,bone][,bone_len][,conf])
            # - motion stream uses (dx,dy)  (NO conf channel)
            # This must match models/train_gcn.py and all eval scripts.
            in_feats_j = 2
            if use_bone:
                in_feats_j += 2
            if use_bone_len:
                in_feats_j += 1
            if use_conf:
                in_feats_j += 1
            in_feats_m = 2  # keep 2 even if motion disabled (we feed zeros)
            out["in_feats_j"] = int(in_feats_j)
            out["in_feats_m"] = int(in_feats_m)
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
    in_feats_m: int = 0,
    **kwargs: Any,
) -> nn.Module:
    """Build a model for training/evaluation.

    Supported call styles:

    New (preferred; used by eval/fit_ops.py):
        build_model(arch, model_cfg, feat_cfg, fps_default=...)

    Old (explicit dims):
        build_model(arch, model_cfg, in_ch=..., in_feats=..., ...)

    Notes:
    - fps_default is accepted for compatibility; architectures here don't require it directly.
      (fps_default belongs to feature scaling and time conversion logic elsewhere.)
    """
    arch = str(arch).lower()
    model_cfg = _cfg_to_dict(model_cfg)
    feat_cfg = _cfg_to_dict(feat_cfg)

    # If caller didn't provide explicit dims, infer them from cfgs.
    if arch in ("tcn", "gcn"):
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
            if not in_feats_m:
                in_feats_m = int(inferred.get("in_feats_m", 0))

    if arch == "tcn":
        if not in_ch or in_ch <= 0:
            raise ValueError(
                "TCN requires in_ch > 0. "
                "Store in_ch in model_cfg when saving checkpoints, or provide a valid feat_cfg."
            )
        cfg = TCNConfig.from_dict(model_cfg)
        return TCN(in_ch=int(in_ch), hidden=cfg.hidden, dropout=cfg.dropout, num_blocks=cfg.num_blocks, kernel=cfg.kernel)

    if arch == "gcn":
        cfg = GCNConfig.from_dict(model_cfg)
        if cfg.two_stream:
            if not in_feats_j or not in_feats_m:
                raise ValueError(
                    "Two-stream GCN requires in_feats_j and in_feats_m. "
                    "Store them in model_cfg when saving checkpoints, or provide a valid feat_cfg."
                )
            return TwoStreamGCN(
                num_joints=int(num_joints),
                in_feats_j=int(in_feats_j),
                in_feats_m=int(in_feats_m),
                gcn_hidden=cfg.gcn_hidden,
                tcn_hidden=cfg.tcn_hidden,
                dropout=cfg.dropout,
                use_se=cfg.use_se,
                fuse=cfg.fuse,
            )

        if not in_feats or in_feats <= 0:
            raise ValueError(
                "GCN requires in_feats > 0. "
                "Store in_feats in model_cfg when saving checkpoints, or provide a valid feat_cfg."
            )
        return GCN(
            num_joints=int(num_joints),
            in_feats=int(in_feats),
            gcn_hidden=cfg.gcn_hidden,
            tcn_hidden=cfg.tcn_hidden,
            dropout=cfg.dropout,
            use_se=cfg.use_se,
        )

    raise ValueError(f"Unknown arch: {arch}")
