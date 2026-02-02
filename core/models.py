#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/models.py

Model builders for this repo (TCN + GCN).

Why this file exists
-------------------
Your project has multiple stages that must rebuild *exactly the same model*:
- training (train_tcn.py / train_gcn.py)
- evaluation (fit_ops.py / replay_eval.py / mining)
- checkpoint loading (ckpt bundles)

If any stage uses a slightly different model definition, you get:
- "shape mismatch" errors when loading checkpoints
- inconsistent offline vs deployment metrics

So this module provides a compatibility-focused API:
- build_model(arch, model_cfg, feat_cfg, ...)
- infer_input_dims(arch, model_cfg, feat_cfg, ...)
- and small helpers for consistent logits handling.

Shape conventions used in this file
-----------------------------------
B = batch size
T = time steps (frames) in a window
V = number of joints (MediaPipe Pose default: 33)
C/F = feature channels

TCN input:  x  shape [B, T, C]
GCN input:  x  shape [B, T, V, F]
Two-stream GCN input: (xj, xm)
  xj shape [B, T, V, Fj]   joints stream (xy [+conf])
  xm shape [B, T, V, Fm]   motion stream (dxdy)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 1) Config normalization helpers
# ============================================================
def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    """
    Normalize cfg into a plain dict.

    Why:
    - Sometimes cfg comes from JSON/YAML -> dict
    - Sometimes cfg is a dataclass stored in a checkpoint bundle
    - Sometimes cfg is a simple object with attributes

    This function makes build_model() tolerant to all these forms.
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


def _bool(d: Dict[str, Any], key: str, default: bool = False) -> bool:
    """
    Robust bool parsing from config dict:
    - accepts True/False
    - accepts 1/0
    - accepts strings: "true"/"false"/"1"/"0"/"yes"/"no"
    """
    v = d.get(key, default)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "t"):
            return True
        if s in ("0", "false", "no", "n", "f"):
            return False
    return bool(v)


# ============================================================
# 2) Convenience helpers used across the repo
# ============================================================
def pick_device() -> torch.device:
    """
    Pick the best available device.

    Priority for your typical setup (Mac):
      1) MPS (Apple Silicon GPU)
      2) CUDA
      3) CPU

    Note:
    - If you run on an NVIDIA machine, CUDA will be available.
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def logits_1d(logits: torch.Tensor) -> torch.Tensor:
    """
    Normalize logits to shape [B].

    Many PyTorch models output either:
    - [B]   (ideal)
    - [B,1] (common)

    This helper makes downstream code consistent.
    """
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    if logits.ndim == 1:
        return logits
    # tolerate accidental extra dims (e.g., [B,1,1])
    return logits.view(logits.shape[0], -1)[:, 0]


def p_fall_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits -> probability of fall (sigmoid).

    Output shape is [B].
    """
    return torch.sigmoid(logits_1d(logits))


# ============================================================
# 3) TCN (Temporal Convolution Network)
# ============================================================
class ResTCNBlock(nn.Module):
    """
    A residual temporal block using Conv1D over time.

    Input/Output:
      x: [B, C, T]
      returns: [B, C, T]

    Why residual:
    - Helps gradient flow
    - Allows deeper temporal stacks without collapsing
    """

    def __init__(self, ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.30):
        super().__init__()
        # Padding chosen so output length stays the same as input length.
        # For kernel=3, padding=dilation keeps "same length".
        padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act(self.bn(self.conv(x))))
        return x + y


class TCN(nn.Module):
    """
    Temporal model for flattened per-frame features.

    Expected input:
      x: [B, T, C]
    Internally we use Conv1D:
      [B, C, T]

    Output:
      logits: [B]
    """

    def __init__(self, in_ch: int, hidden: int = 128, dropout: float = 0.30, num_blocks: int = 4, kernel: int = 3):
        super().__init__()

        # First projection from input channels -> hidden channels
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )

        # Dilations expand receptive field: 1,2,4,8,... across blocks
        dilations = [2**i for i in range(max(1, int(num_blocks)))]
        self.blocks = nn.ModuleList(
            [ResTCNBlock(hidden, kernel_size=kernel, dilation=d, dropout=dropout) for d in dilations]
        )

        # Pool over time (AdaptiveAvgPool1d(1) keeps model robust to different T)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Binary classification head
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> Conv1D expects [B, C, T]
        x = x.transpose(1, 2)

        x = self.conv_in(x)
        for blk in self.blocks:
            x = blk(x)

        # pool output: [B, hidden, 1] -> squeeze -> [B, hidden]
        x = self.pool(x).squeeze(-1)

        # head output: [B, 1] -> squeeze -> [B]
        return self.head(x).squeeze(-1)


@dataclass
class TCNConfig:
    """
    Configuration for building a TCN.

    These values are stored in checkpoints, so they must be stable and serializable.
    """
    hidden: int = 128
    dropout: float = 0.30
    num_blocks: int = 4
    kernel: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TCNConfig":
        """
        Backward compatible parsing:
        - accepts older key aliases like "hid" or "p"
        """
        if not isinstance(d, dict):
            return TCNConfig()
        return TCNConfig(
            hidden=int(d.get("hidden", d.get("hid", 128))),
            dropout=float(d.get("dropout", d.get("p", 0.30))),
            num_blocks=int(d.get("num_blocks", 4)),
            kernel=int(d.get("kernel", 3)),
        )


# ============================================================
# 4) GCN (Graph Convolution + Temporal Conv)
# ============================================================
def build_mediapipe_adjacency(num_joints: int = 33) -> np.ndarray:
    """
    Build a basic undirected adjacency matrix for MediaPipe Pose.

    Why adjacency:
    - GCN uses graph edges to mix information between related joints.
    - This adjacency is a reasonable subset for pose structure.

    Output:
      A: float32 [V, V] with 1 on edges, 0 otherwise.
    """
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
    """
    Symmetric normalization:
      A_hat = A + I
      D = sum(A_hat rows)
      A_norm = D^{-1/2} * A_hat * D^{-1/2}

    This is a common stable normalization for GCNs.
    """
    A = A.astype(np.float32)
    I = np.eye(A.shape[0], dtype=np.float32)
    A_hat = A + I
    D = np.sum(A_hat, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-8)))
    return (D_inv_sqrt @ A_hat @ D_inv_sqrt).astype(np.float32)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block over channels.

    Input:
      x: [B, T, V, C]
    Output:
      x scaled by learned per-channel gates (same shape).

    Why:
    - Helps the model learn "which channels matter more"
    - Often improves performance with minimal cost
    """

    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        mid = max(4, ch // r)
        self.fc1 = nn.Linear(ch, mid)
        self.fc2 = nn.Linear(mid, ch)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, v, c = x.shape
        z = x.mean(dim=(1, 2))              # [B, C]
        z = self.relu(self.fc1(z))          # [B, mid]
        s = self.sig(self.fc2(z)).view(b, 1, 1, c)  # [B,1,1,C]
        return x * s


class GCNLayer(nn.Module):
    """
    One graph convolution layer.

    Input:
      x: [B, T, V, Cin]
      A_hat: [V, V] normalized adjacency

    Operation:
      1) Mix joints using adjacency:
           x' = A_hat @ x   (for each time step)
      2) Linear projection Cin -> Cout
      3) LayerNorm + ReLU + Dropout
    """

    def __init__(self, in_feats: int, out_feats: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)
        self.ln = nn.LayerNorm(out_feats)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # x: [B,T,V,Cin]
        x = torch.einsum("vw,btwc->btvc", A_hat, x)  # mix neighbors
        x = self.lin(x)                               # [B,T,V,Cout]
        x = self.drop(self.act(self.ln(x)))
        return x


class TemporalConv(nn.Module):
    """
    Temporal convolution encoder for sequences after graph mixing.

    Input:
      x: [B, T, C]
    Output:
      embedding: [B, H]

    Idea:
    - Project per-frame features to a hidden space
    - Apply several dilated Conv1D residual blocks over time
    - Mean pool over time
    """

    def __init__(self, ch: int, hidden: int, dropout: float = 0.30, num_blocks: int = 3):
        super().__init__()
        self.proj = nn.Linear(ch, hidden)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.blocks = nn.ModuleList()
        for i in range(max(1, int(num_blocks))):
            d = 2**i
            pad = d  # kernel=3 -> padding=d keeps same length
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=pad, dilation=d),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C] -> [B,T,H]
        x = self.drop(self.act(self.proj(x)))

        # Conv1D expects [B,H,T]
        x = x.transpose(1, 2)

        # Residual blocks
        for blk in self.blocks:
            x = x + blk(x)

        # Mean pool over time -> [B,H]
        x = x.mean(dim=-1)
        return x


class GCNEncoder(nn.Module):
    """
    Graph+Temporal encoder.

    Steps:
    1) Apply 2 GCN layers over joints (for each time step)
    2) Optional SEBlock for channel reweighting
    3) Average over joints -> [B,T,C]
    4) TemporalConv -> [B,E]

    Input:
      x: [B,T,V,F]
    Output:
      z: [B,E]
    """

    def __init__(self, num_joints: int, in_feats: int, gcn_hidden: int, tcn_hidden: int, dropout: float, use_se: bool):
        super().__init__()
        A_hat = normalize_adjacency(build_mediapipe_adjacency(num_joints))
        self.register_buffer("A_hat", torch.from_numpy(A_hat))

        self.g1 = GCNLayer(in_feats, gcn_hidden, dropout=dropout)
        self.g2 = GCNLayer(gcn_hidden, gcn_hidden, dropout=dropout)
        self.se = SEBlock(gcn_hidden) if use_se else nn.Identity()
        self.temporal = TemporalConv(gcn_hidden, tcn_hidden, dropout=dropout, num_blocks=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A_hat = self.A_hat
        x = self.g1(x, A_hat)   # [B,T,V,C]
        x = self.g2(x, A_hat)   # [B,T,V,C]
        x = self.se(x)          # [B,T,V,C]
        x = x.mean(dim=2)       # average over joints -> [B,T,C]
        return self.temporal(x) # [B,E]


class GCN(nn.Module):
    """
    Single-stream GCN classifier.

    Input:
      x: [B,T,V,F]
    Output:
      logits: [B]
    """

    def __init__(self, num_joints: int, in_feats: int, gcn_hidden: int = 96, tcn_hidden: int = 192,
                 dropout: float = 0.35, use_se: bool = True):
        super().__init__()
        self.encoder = GCNEncoder(num_joints, in_feats, gcn_hidden, tcn_hidden, dropout, use_se)
        self.head = nn.Linear(tcn_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)              # [B,E]
        return self.head(z).squeeze(-1)  # [B]


class TwoStreamGCN(nn.Module):
    """
    Two-stream GCN classifier.

    - Stream J (joints): xy (+ conf optional)
    - Stream M (motion): dxdy

    Input:
      xj: [B,T,V,Fj]
      xm: [B,T,V,Fm]
    Output:
      logits: [B]

    fuse:
      - "concat": concatenate embeddings -> bigger head
      - "sum": add embeddings (requires same dimension)
    """

    def __init__(self, num_joints: int, in_feats_j: int, in_feats_m: int, gcn_hidden: int = 96, tcn_hidden: int = 192,
                 dropout: float = 0.35, use_se: bool = True, fuse: str = "concat"):
        super().__init__()
        self.j_enc = GCNEncoder(num_joints, in_feats_j, gcn_hidden, tcn_hidden, dropout, use_se)
        self.m_enc = GCNEncoder(num_joints, in_feats_m, gcn_hidden, tcn_hidden, dropout, use_se)

        fuse = str(fuse).lower().strip()
        if fuse not in ("concat", "sum"):
            fuse = "concat"
        self.fuse = fuse

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
    """
    Config for building a GCN model.
    """
    num_joints: int = 33
    gcn_hidden: int = 96
    tcn_hidden: int = 192
    dropout: float = 0.35
    use_se: bool = True
    two_stream: bool = False
    fuse: str = "concat"  # "concat" | "sum"

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


# ============================================================
# 5) Input-dimension inference
# ============================================================
def infer_input_dims(
    arch: str,
    model_cfg: Dict[str, Any],
    feat_cfg: Optional[Dict[str, Any]] = None,
    *,
    num_joints_default: int = 33,
) -> Dict[str, int]:
    """
    Infer the input dimensions required to build the model.

    Preference order:
    1) Explicit dims stored in model_cfg (best: matches training exactly)
       e.g., model_cfg["in_ch"], model_cfg["in_feats"], ...
    2) Infer from feat_cfg flags (reasonable defaults for this repo)

    Returns a dict that can contain:
      - num_joints
      - in_ch (for TCN)
      - in_feats (single-stream GCN)
      - in_feats_j, in_feats_m (two-stream GCN)
    """
    arch = str(arch).lower()
    model_cfg = _cfg_to_dict(model_cfg)
    feat_cfg = _cfg_to_dict(feat_cfg)

    # num_joints usually comes from model_cfg (stored at training time)
    # feat_cfg typically does not contain it, but we allow it.
    num_joints = int(model_cfg.get("num_joints", feat_cfg.get("num_joints", num_joints_default)))

    out: Dict[str, int] = {"num_joints": num_joints}

    # If training stored explicit dims, prefer them always.
    for k in ("in_ch", "in_feats", "in_feats_j", "in_feats_m"):
        if k in model_cfg:
            out[k] = int(model_cfg[k])

    # If we already have enough info, return early.
    if arch == "tcn" and "in_ch" in out:
        return out
    if arch == "gcn" and ("in_feats" in out or ("in_feats_j" in out and "in_feats_m" in out)):
        return out

    # Otherwise infer from feature flags.
    #
    # IMPORTANT: defaults here should match your repo defaults.
    # If feat_cfg is missing, we assume "typical pipeline":
    #   use_motion=True, use_conf_channel=True
    use_motion = _bool(feat_cfg, "use_motion", default=True)
    use_conf = _bool(feat_cfg, "use_conf_channel", default=True)

    # Single-stream per-joint features:
    per_joint = 2  # xy
    if use_motion:
        per_joint += 2  # dxdy
    if use_conf:
        per_joint += 1  # conf

    if arch == "tcn":
        out["in_ch"] = int(num_joints * per_joint)
        return out

    if arch == "gcn":
        cfg = GCNConfig.from_dict(model_cfg)
        if cfg.two_stream:
            # Two-stream conventions in this project:
            # - joints stream: xy (+conf optional)
            # - motion stream: dxdy (2) (we feed zeros if motion disabled)
            out["in_feats_j"] = int(2 + (1 if use_conf else 0))
            out["in_feats_m"] = 2
        else:
            out["in_feats"] = int(per_joint)
        return out

    raise ValueError(f"Unknown arch: {arch}")


# ============================================================
# 6) Model builder (compatibility-focused)
# ============================================================
def build_model(
    arch: str,
    model_cfg: Dict[str, Any],
    feat_cfg: Optional[Dict[str, Any]] = None,
    *,
    fps_default: Optional[float] = None,  # accepted for compatibility (not used here)
    in_ch: int = 0,
    num_joints: int = 33,
    in_feats: int = 0,
    in_feats_j: int = 0,
    in_feats_m: int = 0,
    **kwargs: Any,  # keep this so older call sites don't crash
) -> nn.Module:
    """
    Build a model for training/evaluation.

    Supported call styles:

    (A) New style (recommended):
        build_model(arch, model_cfg, feat_cfg, fps_default=...)

    (B) Old style (explicit dims):
        build_model(arch, model_cfg, in_ch=..., in_feats=..., ...)

    Why accept both:
    - Some scripts already call build_model with explicit dims
    - Others rebuild from checkpoint bundles where dims may be inferred

    Note:
    - fps_default is not used by the model itself.
      FPS affects *features* (motion scaling) not network architecture.
    """
    arch = str(arch).lower()
    model_cfg = _cfg_to_dict(model_cfg)
    feat_cfg = _cfg_to_dict(feat_cfg)

    # If caller didn't provide explicit dims, infer them.
    inferred = infer_input_dims(arch, model_cfg, feat_cfg, num_joints_default=num_joints)
    num_joints = int(inferred.get("num_joints", num_joints))

    if arch == "tcn":
        if not in_ch:
            in_ch = int(inferred.get("in_ch", 0))
        if in_ch <= 0:
            raise ValueError(
                "TCN requires in_ch > 0. "
                "Store in_ch in model_cfg when saving checkpoints, or provide a valid feat_cfg."
            )
        cfg = TCNConfig.from_dict(model_cfg)
        return TCN(in_ch=int(in_ch), hidden=cfg.hidden, dropout=cfg.dropout, num_blocks=cfg.num_blocks, kernel=cfg.kernel)

    if arch == "gcn":
        cfg = GCNConfig.from_dict(model_cfg)

        if cfg.two_stream:
            if not in_feats_j:
                in_feats_j = int(inferred.get("in_feats_j", 0))
            if not in_feats_m:
                in_feats_m = int(inferred.get("in_feats_m", 0))
            if in_feats_j <= 0 or in_feats_m <= 0:
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

        if not in_feats:
            in_feats = int(inferred.get("in_feats", 0))
        if in_feats <= 0:
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
